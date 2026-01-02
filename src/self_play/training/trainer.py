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

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from ..core.arena import TrainingBatch
from ..core.types import TrainingRecord
from .config import TrainerConfig
from .batching import split_by_token_budget, collate
from .loss import compute_loss, make_loss_fn
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
        config = TrainerConfig(use_importance_sampling=False)
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
            use_importance_sampling=config.use_importance_sampling,
            clip_low=config.clip_low,
            clip_high=config.clip_high,
        )

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

    async def train_step(
        self,
        batch: TrainingBatch,
    ) -> Dict[str, float]:
        """
        Execute one training step with gradient accumulation.

        Args:
            batch: TrainingBatch containing records to train on

        Returns:
            Dictionary of metrics (loss, tokens, records, stale_dropped, etc.)
        """
        self.model.train()

        # 1. Filter stale records
        fresh_records = self.filter_stale_records(batch.records)
        stale_count = len(batch.records) - len(fresh_records)

        if not fresh_records:
            return {
                "skipped": 1,
                "stale_dropped": stale_count,
                "records": 0,
            }

        # 2. Split into micro-batches by token budget
        micro_batches = split_by_token_budget(
            fresh_records,
            self.config.micro_token_budget,
        )

        # 3. Gradient accumulation loop
        # We weight each micro-batch by its record proportion to ensure
        # each record contributes equally to the final gradient, regardless
        # of which micro-batch it landed in (per-record equality).
        accumulated_grads = None
        total_loss = 0.0
        total_tokens = 0
        total_clip_fraction = 0.0
        total_kl = 0.0
        total_records = len(fresh_records)

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

        for micro in micro_batches:
            # Collate into tensors
            input_ids, loss_mask, inference_logprobs, advantages = collate(
                micro,
                pad_token_id=self.config.pad_token_id,
            )

            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(
                self.model,
                input_ids,
                inference_logprobs,
                advantages,
                loss_mask,
            )

            # Compute metrics (separate call since value_and_grad only returns loss)
            _, metrics = compute_loss(
                model=self.model,
                input_ids=input_ids,
                inference_logprobs=inference_logprobs,
                advantages=advantages,
                loss_mask=loss_mask,
                use_importance_sampling=self.config.use_importance_sampling,
                clip_low=self.config.clip_low,
                clip_high=self.config.clip_high,
            )

            # Accumulate gradients weighted by record proportion
            # This ensures each record contributes equally to the final gradient
            # I think this is GSPO? And goes against DAPO which is token-level averaging?
            weight = len(micro) / total_records
            if accumulated_grads is None:
                accumulated_grads = tree_map(lambda g: g * weight, grads)
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g * weight,
                    accumulated_grads,
                    grads,
                )

            # Accumulate metrics
            micro_tokens = sum(len(r.completion_token_ids) for r in micro)
            total_loss += float(loss.item()) * len(micro)
            total_tokens += micro_tokens
            total_clip_fraction += float(metrics["clip_fraction"].item()) * len(micro)
            total_kl += float(metrics["kl"].item()) * len(micro)

            # Clear cache after each micro-batch
            mx.clear_cache()

        # 4. Optimizer step (gradients already properly weighted)
        self.optimizer.update(self.model, accumulated_grads)

        # 5. Evaluate to materialize updated parameters
        mx.eval(self.model.parameters())

        # 6. Increment step counter
        self.train_step_idx += 1

        # 7. Push weights to inference server (if publisher configured)
        if self.publisher is not None:
            try:
                await self.publisher.publish(
                    self.model,
                    version=self.train_step_idx,
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"[trainer] failed to publish weights: {e}")

        # 8. Return metrics
        num_fresh = len(fresh_records)
        return {
            "loss": total_loss / num_fresh,
            "tokens": total_tokens,
            "records": num_fresh,
            "stale_dropped": stale_count,
            "clip_fraction": total_clip_fraction / num_fresh,
            "kl": total_kl / num_fresh,
            "train_step": self.train_step_idx,
        }
