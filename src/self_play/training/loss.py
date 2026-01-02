"""
RL loss computation for policy gradient training.

Supports two modes:
1. Importance sampling (use_importance_sampling=True):
   - Forward pass to recompute logprobs under current policy
   - PPO-style clipped objective with importance ratio
   - Use when inference and training engines differ

2. Simple REINFORCE (use_importance_sampling=False):
   - No forward pass, use inference logprobs directly
   - Loss = -log_prob * advantage
   - Much faster, use for MLX where inference = training
"""
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def get_per_token_logps(
    model: nn.Module,
    input_ids: mx.array,
    lengths: mx.array,
) -> mx.array:
    """
    Forward pass to compute per-token log-probabilities.

    Args:
        model: The language model (must return logits)
        input_ids: Token IDs, shape (batch, seq_len)
        lengths: Actual sequence lengths (before padding), shape (batch,)

    Returns:
        Log-probabilities for each token, shape (batch, seq_len - 1)
        logprobs[i] is the log-prob of input_ids[i+1] given input_ids[:i+1]
    """
    # Forward pass to get logits
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # Shift: logits[:-1] predicts targets[1:]
    logits = logits[:, :-1, :]  # (batch, seq_len - 1, vocab_size)
    targets = input_ids[:, 1:]  # (batch, seq_len - 1)

    # Compute log-softmax
    log_probs = nn.log_softmax(logits, axis=-1)  # (batch, seq_len - 1, vocab_size)

    # Gather log-probs for the actual tokens
    # targets shape: (batch, seq_len - 1)
    # We need to index into log_probs using targets
    batch_size, seq_len_minus_1, vocab_size = log_probs.shape

    # Expand targets for gathering
    targets_expanded = mx.expand_dims(targets, axis=-1)  # (batch, seq_len - 1, 1)
    token_log_probs = mx.take_along_axis(log_probs, targets_expanded, axis=-1)
    token_log_probs = mx.squeeze(token_log_probs, axis=-1)  # (batch, seq_len - 1)

    return token_log_probs


def compute_loss(
    model: nn.Module,
    input_ids: mx.array,
    inference_logprobs: mx.array,
    advantages: mx.array,
    loss_mask: mx.array,
    use_importance_sampling: bool = True,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Compute the RL loss for policy gradient training.

    Args:
        model: The language model (only used if use_importance_sampling=True)
        input_ids: Token IDs, shape (batch, seq_len)
        inference_logprobs: Log-probs from inference, shape (batch, seq_len)
            Will be sliced to (batch, seq_len - 1) internally.
        advantages: Per-token advantages, shape (batch, seq_len)
            Will be sliced to (batch, seq_len - 1) internally.
        loss_mask: Binary mask for which tokens to train on, shape (batch, seq_len)
            Will be sliced to (batch, seq_len - 1) internally.
        use_importance_sampling: Whether to do forward pass and compute importance ratio
        clip_low: Lower bound for importance ratio clipping (e.g., 0.8)
        clip_high: Upper bound for importance ratio clipping (e.g., 1.2)

    Returns:
        Tuple of (loss, metrics_dict)
        - loss: Scalar loss value
        - metrics_dict: Dictionary with clip_fraction, kl, etc.
    """
    # Slice to (batch, seq_len - 1) to align with prediction semantics
    # collate() returns (batch, seq_len) but we need to match trainer_logprobs
    # which is (batch, seq_len - 1) from get_per_token_logps
    inference_logprobs = inference_logprobs[:, :-1]
    advantages = advantages[:, :-1]
    loss_mask = loss_mask[:, :-1]

    metrics = {}

    if use_importance_sampling:
        # Forward pass to get current policy log-probs
        lengths = loss_mask.sum(axis=1) + 1  # Approximate lengths
        trainer_logprobs = get_per_token_logps(model, input_ids, lengths)

        # Compute log importance ratio
        log_ratio = trainer_logprobs - inference_logprobs

        # Importance ratio: π_θ(a|s) / π_old(a|s)
        importance_ratio = mx.exp(log_ratio)

        # Clipped ratio
        clipped_ratio = mx.clip(importance_ratio, clip_low, clip_high)

        # PPO-style clipped surrogate objective
        # We want to maximize: min(ratio * A, clip(ratio) * A)
        # So we minimize: -min(ratio * A, clip(ratio) * A)
        unclipped_obj = importance_ratio * advantages
        clipped_obj = clipped_ratio * advantages

        # Take the minimum (pessimistic bound)
        pg_loss = -mx.minimum(unclipped_obj, clipped_obj)

        # Compute clipping metrics
        is_clipped = (importance_ratio < clip_low) | (importance_ratio > clip_high)
        clip_fraction = (is_clipped * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1.0)
        metrics["clip_fraction"] = clip_fraction

        # Approximate KL divergence: (r - 1) - log(r) ≈ 0.5 * (log_r)^2
        # where r = importance_ratio
        approx_kl = 0.5 * (log_ratio ** 2)
        mean_kl = (approx_kl * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1.0)
        metrics["kl"] = mean_kl

    else:
        # Simple REINFORCE: no forward pass
        # Loss = -log_prob * advantage
        pg_loss = -inference_logprobs * advantages

        # No clipping metrics when not using importance sampling
        metrics["clip_fraction"] = mx.array(0.0)
        metrics["kl"] = mx.array(0.0)

    # Apply mask and compute mean loss
    masked_loss = pg_loss * loss_mask
    loss = masked_loss.sum() / mx.maximum(loss_mask.sum(), 1.0)

    metrics["loss"] = loss

    return loss, metrics


def make_loss_fn(
    use_importance_sampling: bool = True,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
):
    """
    Create a loss function suitable for use with nn.value_and_grad.

    The returned function takes (model, batch_data) and returns (loss, metrics).

    Args:
        use_importance_sampling: Whether to do forward pass and compute importance ratio
        clip_low: Lower bound for importance ratio clipping
        clip_high: Upper bound for importance ratio clipping

    Returns:
        A loss function compatible with nn.value_and_grad
    """
    def loss_fn(
        model: nn.Module,
        input_ids: mx.array,
        inference_logprobs: mx.array,
        advantages: mx.array,
        loss_mask: mx.array,
    ) -> mx.array:
        loss, _ = compute_loss(
            model=model,
            input_ids=input_ids,
            inference_logprobs=inference_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            use_importance_sampling=use_importance_sampling,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        return loss

    return loss_fn
