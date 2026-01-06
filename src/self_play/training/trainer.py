"""
Trainer class for RL training with gradient accumulation.

The Trainer orchestrates:
1. Filtering stale records based on policy version
2. Splitting into micro-batches by token budget
3. Gradient accumulation across micro-batches
4. Optimizer step
5. Weight publishing to inference server
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from ..core.types import TrainingBatch, TrainingRecord
from .config import TrainerConfig
from .batching import split_by_token_budget, collate
from .loss import make_loss_fn
from .weight_publisher import WeightPublisher


class Trainer:
    """
    RL trainer with gradient accumulation and weight publishing.

    This is a clean, minimal implementation that embodies the core patterns
    of modern RL training:
    - Staleness filtering by policy version
    - Token-budget-based micro-batching
    - Gradient accumulation
    - Online weight updates to inference server

    Example:
        model, tokenizer = load_model_with_lora(...)
        optimizer = mx.optimizers.Adam(learning_rate=1e-5)
        config = TrainerConfig()
        publisher = WeightPublisher(base_url="http://localhost:8000")

        trainer = Trainer(model, optimizer, config, publisher)
        metrics = await trainer.train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        config: TrainerConfig,
        publisher: Optional[WeightPublisher] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: MLX model with LoRA attached (trainable parameters)
            optimizer: MLX optimizer (e.g., mx.optimizers.Adam)
            config: TrainerConfig with hyperparameters
            publisher: Optional WeightPublisher for hot-swap to inference server
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.publisher = publisher

        self.train_step_idx = 0

        # Create loss function for value_and_grad
        self._loss_fn = make_loss_fn(
            clip_low=config.clip_low,
            clip_high=config.clip_high,
            kl_coef=config.kl_coef,
            use_kl_penalty=config.use_kl_penalty,
            loss_type=config.loss_type,
            importance_sampling=config.importance_sampling,
            gspo_clip_epsilon=config.gspo_clip_epsilon,
        )

        # GSPO requires sample-level weighting (not token-level) for gradient accumulation
        if config.importance_sampling == "sequence":
            self._effective_loss_type = "sample"
        else:
            self._effective_loss_type = config.loss_type

    def filter_stale_records(
        self,
        records: List[TrainingRecord],
    ) -> List[TrainingRecord]:
        """
        Filter out records that are too old based on policy version.

        Args:
            records: List of training records

        Returns:
            List of fresh records (within max_policy_lag steps of current)
        """
        fresh = []
        for r in records:
            policy_version = r.meta.get("policy_version", 0)
            lag = self.train_step_idx - policy_version
            if lag <= self.config.max_policy_lag:
                fresh.append(r)
        return fresh

    def _train_step_sync(
        self,
        fresh_records: List[TrainingRecord],
    ) -> Tuple[Dict[str, float], int]:
        """
        Synchronous training step - runs in a separate thread.

        This contains all the CPU/GPU-bound MLX work that would otherwise
        block the asyncio event loop.

        Args:
            fresh_records: Pre-filtered list of fresh training records

        Returns:
            Tuple of (metrics dict, new train_step_idx)
        """
        step_start = time.perf_counter()

        # Set memory limit - helps with memory pressure
        try:
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
        except Exception:
            pass  # May not be available on all systems

        # Split into micro-batches by token budget
        micro_batches = split_by_token_budget(
            fresh_records,
            self.config.micro_token_budget,
        )

        # Gradient accumulation loop
        accumulated_grads = None
        total_tokens = 0
        total_records = len(fresh_records)

        # For token-level weighting, compute total tokens upfront
        if self._effective_loss_type == "token":
            total_tokens_for_weight = sum(len(r.completion_token_ids) for r in fresh_records)

        # Track lazy losses/metrics for batched eval at the end
        micro_losses = []  # List of (loss, num_records)
        micro_metrics = []  # List of (metrics_dict, num_records)

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

        for micro in micro_batches:
            # Collate into tensors
            input_ids, loss_mask, inference_logprobs, advantages = collate(
                micro,
                pad_token_id=self.config.pad_token_id,
            )

            # Compute loss, metrics, and gradients in a SINGLE forward pass
            (loss, metrics), grads = loss_and_grad_fn(
                self.model,
                input_ids,
                inference_logprobs,
                advantages,
                loss_mask,
            )

            # For memory-constrained systems: eval immediately to free activations
            if self.config.eval_per_micro_batch:
                mx.eval(loss, metrics["clip_fraction"], metrics["kl"], grads)

            # Track metrics (compute before weighting for token-level mode)
            micro_tokens = sum(len(r.completion_token_ids) for r in micro)

            # Accumulate gradients with appropriate weighting
            if self._effective_loss_type == "token":
                # DAPO: weight by token count
                weight = micro_tokens / total_tokens_for_weight
            else:
                # GRPO/GSPO: weight by sample count
                weight = len(micro) / total_records

            if accumulated_grads is None:
                accumulated_grads = tree_map(lambda g: g * weight, grads)
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g * weight,
                    accumulated_grads,
                    grads,
                )
            micro_losses.append((loss, len(micro)))
            micro_metrics.append((metrics, len(micro)))
            total_tokens += micro_tokens

            # Clear cache after each micro-batch to free memory
            if self.config.eval_per_micro_batch:
                mx.clear_cache()

        # Optimizer step
        self.optimizer.update(self.model, accumulated_grads)
        accumulated_grads = None  # Release reference before eval

        # Materialize model parameters AND optimizer state (momentum buffers)
        # This breaks the lazy computation graph that would otherwise hold
        # references to all previous gradients, causing memory to accumulate
        mx.eval(self.model.parameters())
        mx.eval(self.optimizer.state)
        mx.clear_cache()

        # Evaluate losses/metrics if not done per-micro-batch
        if not self.config.eval_per_micro_batch:
            all_losses = [loss for loss, _ in micro_losses]
            all_clip_fracs = [m["clip_fraction"] for m, _ in micro_metrics]
            all_kls = [m["kl"] for m, _ in micro_metrics]
            mx.eval(*all_losses, *all_clip_fracs, *all_kls)

        # Extract scalar values
        total_loss = sum(float(loss.item()) * n for loss, n in micro_losses)
        total_clip_fraction = sum(float(m["clip_fraction"].item()) * n for m, n in micro_metrics)
        total_kl = sum(float(m["kl"].item()) * n for m, n in micro_metrics)

        mx.clear_cache()

        # Increment step counter
        self.train_step_idx += 1

        step_total = time.perf_counter() - step_start

        # Compute averages
        num_fresh = len(fresh_records)
        avg_loss = total_loss / num_fresh
        avg_kl = total_kl / num_fresh
        avg_clip = total_clip_fraction / num_fresh

        return {
            "loss": avg_loss,
            "tokens": total_tokens,
            "records": num_fresh,
            "clip_fraction": avg_clip,
            "kl": avg_kl,
            "train_step": self.train_step_idx,
            "step_time_s": step_total,
        }, self.train_step_idx

    async def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        Execute one training step with gradient accumulation.

        The CPU/GPU-bound MLX work runs in a separate thread via asyncio.to_thread()
        to avoid blocking the event loop, allowing generation to proceed in parallel.

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

        # Run CPU/GPU-bound training in a separate thread
        metrics, new_step_idx = await asyncio.to_thread(
            self._train_step_sync,
            fresh_records,
        )
        metrics["stale_dropped"] = stale_count

        # Check for divergence (NaN in loss or KL)
        if math.isnan(metrics.get("loss", 0.0)) or math.isnan(metrics.get("kl", 0.0)):
            print(f"\n[trainer] WARNING: NaN detected at step {new_step_idx}!")
            print(f"  metrics: {metrics}")
            print("  This usually means the learning rate is too high or the model has diverged.")

        # Push weights to inference server
        if self.publisher is not None:
            try:
                await self.publisher.publish(self.model, version=new_step_idx)
            except Exception:
                pass  # Weight publishing failures are non-fatal

        return metrics
