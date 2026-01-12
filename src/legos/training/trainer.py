"""
Trainer class for RL training with streaming gradient accumulation.

The Trainer supports two modes:
1. Streaming: accumulate() processes micro-batches one at a time, steps when batch_size reached
2. Batch: train_step() processes all records at once (for synchronous batch processing)

Both modes share the same underlying gradient accumulation logic.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten
from pathlib import Path

from collections import defaultdict

from ..core.types import TrainingBatch, TrainingRecord
from ..core.clients import InferenceClient
from .config import TrainerConfig
from .batching import split_by_token_budget, collate
from .loss import make_loss_fn


def compute_per_actor_reward_stats(records: List[TrainingRecord]) -> Dict[str, float]:
    """Compute per-actor reward statistics (avg/min/max)."""
    if not records:
        return {}

    actor_rewards: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        actor_rewards[r.actor_id].append(r.reward)

    stats: Dict[str, float] = {}
    for actor_id, rewards in actor_rewards.items():
        stats[f"reward_{actor_id}_avg"] = sum(rewards) / len(rewards)
        stats[f"reward_{actor_id}_min"] = min(rewards)
        stats[f"reward_{actor_id}_max"] = max(rewards)
        stats[f"reward_{actor_id}_count"] = float(len(rewards))

    return stats


class Trainer:
    """
    RL trainer with streaming gradient accumulation and weight publishing.

    Supports both streaming (accumulate()) and batch (train_step()) modes.

    Example (streaming):
        trainer = Trainer(model, optimizer, config, client)
        for micro_batch in micro_batches:
            result = await trainer.accumulate(micro_batch)
            if result['stepped']:
                # Handle step completion

    Example (batch):
        trainer = Trainer(model, optimizer, config, client)
        metrics = await trainer.train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        config: TrainerConfig,
        client: Optional[InferenceClient] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: MLX model with LoRA attached (trainable parameters)
            optimizer: MLX optimizer (e.g., mx.optimizers.Adam)
            config: TrainerConfig with hyperparameters
            client: Optional InferenceClient for hot-swap weights to inference server
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.client = client

        self.train_step_idx = 0

        # Create loss function for value_and_grad
        self._loss_fn = make_loss_fn(
            clip_low=config.ppo_clip_min,
            clip_high=config.ppo_clip_max,
            kl_coef=config.kl_coef,
            use_kl_penalty=config.use_kl_penalty,
            loss_type=config.loss_type,
            importance_sampling=config.importance_sampling,
            gspo_clip_epsilon=config.gspo_clip_epsilon,
            kl_clip_max=config.kl_max,
            clip_skip_threshold=config.clip_skip_threshold,
        )

        # GSPO requires sample-level weighting (not token-level) for gradient accumulation
        if config.importance_sampling == "sequence":
            self._effective_loss_type = "sample"
        else:
            self._effective_loss_type = config.loss_type

        # Streaming gradient accumulation state
        self._reset_pending()

    def _reset_pending(self) -> None:
        """Reset streaming accumulation state after optimizer step."""
        self._weighted_grad_sum = None  # Weighted sum of gradients
        self._pending_samples = 0  # Samples accumulated (excluding skipped)
        self._pending_tokens = 0  # For token-weighted mode and metrics
        self._pending_records: List[TrainingRecord] = []  # For reward stats
        self._pending_metrics: List[Tuple[Dict, int]] = []  # (metrics_dict, n_samples)
        self._pending_skipped = 0  # Count of skipped micro-batches
        self._step_start_time: Optional[float] = None

    @property
    def pending_samples(self) -> int:
        """Number of samples accumulated but not yet stepped."""
        return self._pending_samples

    def filter_stale_records(
        self,
        records: List[TrainingRecord],
    ) -> List[TrainingRecord]:
        """
        Filter out records that are too old based on policy version.

        Args:
            records: List of training records

        Returns:
            List of fresh records (within staleness_limit steps of current)
        """
        fresh = []
        for r in records:
            policy_version = r.meta.get("policy_version", 0)
            lag = self.train_step_idx - policy_version
            if lag <= self.config.staleness_limit:
                fresh.append(r)
        return fresh

    def _accumulate_micro_batch_sync(
        self,
        records: List[TrainingRecord],
    ) -> Tuple[int, int, bool]:
        """
        Process one micro-batch synchronously, accumulating gradients.

        Args:
            records: Training records for this micro-batch

        Returns:
            Tuple of (samples_added, tokens_added, was_skipped)
        """
        if not records:
            return 0, 0, False

        # Start timing on first micro-batch of this step
        if self._step_start_time is None:
            self._step_start_time = time.perf_counter()

        # Set memory limit - helps with memory pressure
        try:
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
        except Exception:
            pass  # May not be available on all systems

        # Collate into tensors
        input_ids, loss_mask, inference_logprobs, advantages = collate(
            records,
            pad_token_id=self.config.pad_token_id,
        )

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

        # Compute loss, metrics, and gradients in a SINGLE forward pass
        (loss, metrics), grads = loss_and_grad_fn(
            self.model,
            input_ids,
            inference_logprobs,
            advantages,
            loss_mask,
        )

        n_samples = len(records)
        n_tokens = sum(len(r.completion_token_ids) for r in records)

        # For memory-constrained systems: eval immediately to free activations
        if self.config.eval_per_micro_batch:
            mx.eval(loss, metrics["clip_fraction"], metrics["kl"], grads)

        # Check if this micro-batch was skipped (high clip fraction)
        was_skipped = metrics.get("skip_batch", mx.array(False)).item()

        if was_skipped:
            self._pending_skipped += 1
            if self.config.eval_per_micro_batch:
                mx.clear_cache()
            return 0, 0, True

        # Accumulate gradients weighted by sample or token count
        # We'll normalize at step time by dividing by total
        if self._effective_loss_type == "token":
            weight = n_tokens
        else:
            weight = n_samples

        if self._weighted_grad_sum is None:
            self._weighted_grad_sum = tree_map(lambda g: g * weight, grads)
        else:
            self._weighted_grad_sum = tree_map(
                lambda s, g: s + g * weight,
                self._weighted_grad_sum,
                grads,
            )

        # Track state for this micro-batch
        self._pending_samples += n_samples
        self._pending_tokens += n_tokens
        self._pending_records.extend(records)
        self._pending_metrics.append((metrics, n_samples))

        if self.config.eval_per_micro_batch:
            mx.clear_cache()

        return n_samples, n_tokens, False

    def _optimizer_step_sync(self) -> Dict[str, float]:
        """
        Apply accumulated gradients and return metrics.

        Returns:
            Dictionary of training metrics
        """
        if self._pending_samples == 0 or self._weighted_grad_sum is None:
            return {"skipped": 1, "records": 0}

        # Normalize gradients by total weight
        if self._effective_loss_type == "token":
            normalizer = self._pending_tokens
        else:
            normalizer = self._pending_samples

        final_grads = tree_map(
            lambda g: g / normalizer,
            self._weighted_grad_sum,
        )

        # Optimizer step
        self.optimizer.update(self.model, final_grads)

        # Materialize model parameters AND optimizer state (momentum buffers)
        # This breaks the lazy computation graph that would otherwise hold
        # references to all previous gradients, causing memory to accumulate
        mx.eval(self.model.parameters())
        mx.eval(self.optimizer.state)
        mx.clear_cache()

        # Evaluate metrics if not done per-micro-batch
        if not self.config.eval_per_micro_batch:
            all_clip_fracs = [m["clip_fraction"] for m, _ in self._pending_metrics]
            all_kls = [m["kl"] for m, _ in self._pending_metrics]
            all_losses = [m.get("loss", mx.array(0.0)) for m, _ in self._pending_metrics]
            mx.eval(*all_losses, *all_clip_fracs, *all_kls)

        # Aggregate metrics across micro-batches
        total_loss = 0.0
        total_clip_fraction = 0.0
        total_kl = 0.0
        total_k1_seq = 0.0
        total_k1_clipped_frac = 0.0
        has_k1_metrics = any("k1_seq_mean" in m for m, _ in self._pending_metrics)

        for m, n in self._pending_metrics:
            total_loss += float(m.get("loss", mx.array(0.0)).item()) * n
            total_clip_fraction += float(m["clip_fraction"].item()) * n
            total_kl += float(m["kl"].item()) * n
            if has_k1_metrics and "k1_seq_mean" in m:
                total_k1_seq += float(m["k1_seq_mean"].item()) * n
                total_k1_clipped_frac += float(m["k1_clipped_frac"].item()) * n

        # Increment step counter
        self.train_step_idx += 1

        step_total = time.perf_counter() - (self._step_start_time or time.perf_counter())

        # Compute averages
        num_samples = self._pending_samples
        avg_loss = total_loss / num_samples
        avg_kl = total_kl / num_samples
        avg_clip = total_clip_fraction / num_samples

        # Compute reward stats from accumulated records
        reward_stats = compute_per_actor_reward_stats(self._pending_records)
        if num_samples > 0:
            reward_stats["avg_reward"] = sum(r.reward for r in self._pending_records) / num_samples
            reward_stats["avg_advantage"] = sum(r.advantage for r in self._pending_records) / num_samples
        else:
            reward_stats["avg_reward"] = 0.0
            reward_stats["avg_advantage"] = 0.0

        result = {
            "loss": avg_loss,
            "tokens": self._pending_tokens,
            "records": num_samples,
            "clip_fraction": avg_clip,
            "kl": avg_kl,
            "train_step": self.train_step_idx,
            "step_time_s": step_total,
            "skipped_micro_batches": self._pending_skipped,
        }

        # Add reward stats with train/ prefix
        result.update({f"train/{k}": v for k, v in reward_stats.items()})

        # Add K1 metrics if present
        if has_k1_metrics:
            result["k1_seq_mean"] = total_k1_seq / num_samples
            result["k1_clipped_frac"] = total_k1_clipped_frac / num_samples

        mx.clear_cache()

        # Reset state for next step
        self._reset_pending()

        return result

    def save_checkpoint(self, step: Optional[int] = None) -> Optional[str]:
        """
        Save LoRA weights to checkpoint directory.

        Args:
            step: Step number for checkpoint name. If None, uses train_step_idx.

        Returns:
            Path to saved checkpoint, or None if checkpointing disabled.
        """
        if not self.config.checkpoint_dir:
            return None

        step = step if step is not None else self.train_step_idx
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"step_{step}.safetensors"

        # Save only trainable parameters (LoRA weights)
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(str(checkpoint_path), weights)

        return str(checkpoint_path)

    async def accumulate(self, records: List[TrainingRecord]) -> Dict:
        """
        Process records as micro-batch, accumulating gradients.
        Triggers optimizer step when pending_samples >= min_samples_per_step.

        Args:
            records: Training records for one micro-batch

        Returns:
            Dict with keys:
                samples_added: Samples that contributed gradients
                samples_skipped: Samples in skipped micro-batches
                stale_filtered: Stale records filtered out
                stepped: Whether optimizer step occurred
                step_metrics: Metrics dict if stepped, else None
        """
        self.model.train()

        # Filter stale records
        fresh = self.filter_stale_records(records)
        stale_count = len(records) - len(fresh)

        result = {
            "samples_added": 0,
            "samples_skipped": 0,
            "stale_filtered": stale_count,
            "stepped": False,
            "step_metrics": None,
        }

        if not fresh:
            return result

        # Process micro-batch in thread (CPU/GPU-bound work)
        samples_added, tokens_added, was_skipped = await asyncio.to_thread(
            self._accumulate_micro_batch_sync, fresh
        )

        result["samples_added"] = samples_added
        if was_skipped:
            result["samples_skipped"] = len(fresh)

        # Check if we should step
        if self._pending_samples >= self.config.min_samples_per_step:
            step_metrics = await asyncio.to_thread(self._optimizer_step_sync)

            # Check for divergence (NaN loss = corrupted gradients = dead model)
            if math.isnan(step_metrics.get("loss", 0.0)):
                raise RuntimeError(
                    f"Training diverged (NaN loss) at step {self.train_step_idx}. "
                    f"Reduce learning rate and restart from last checkpoint."
                )

            # Push weights to inference server
            if self.client is not None:
                try:
                    await self.client.publish_weights(self.model, version=self.train_step_idx)
                except Exception:
                    pass  # Weight publishing failures are non-fatal

            result["stepped"] = True
            result["step_metrics"] = step_metrics

        return result

    async def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        Execute one training step with gradient accumulation.

        Processes all records in batch, splits into micro-batches, accumulates
        gradients, and performs one optimizer step.

        Args:
            batch: TrainingBatch containing records to train on

        Returns:
            Dictionary of metrics (loss, tokens, records, stale_dropped, etc.)
        """
        self.model.train()

        # Filter stale records
        fresh_records = self.filter_stale_records(batch.records)
        stale_count = len(batch.records) - len(fresh_records)

        if not fresh_records:
            return {
                "skipped": 1,
                "stale_dropped": stale_count,
                "records": 0,
            }

        # Split into micro-batches
        micro_batches = split_by_token_budget(
            fresh_records,
            self.config.micro_batch_tokens,
        )

        # Process each micro-batch (accumulating gradients)
        for micro in micro_batches:
            await asyncio.to_thread(self._accumulate_micro_batch_sync, micro)

        # Force optimizer step (we want exactly one step per train_step call)
        if self._pending_samples > 0:
            metrics = await asyncio.to_thread(self._optimizer_step_sync)

            # Check for divergence (NaN loss = corrupted gradients = dead model)
            if math.isnan(metrics.get("loss", 0.0)):
                raise RuntimeError(
                    f"Training diverged (NaN loss) at step {self.train_step_idx}. "
                    f"Reduce learning rate and restart from last checkpoint."
                )

            # Push weights to inference server
            if self.client is not None:
                try:
                    await self.client.publish_weights(self.model, version=self.train_step_idx)
                except Exception:
                    pass  # Weight publishing failures are non-fatal

            metrics["stale_dropped"] = stale_count
            return metrics

        return {
            "skipped": 1,
            "stale_dropped": stale_count,
            "records": 0,
        }
