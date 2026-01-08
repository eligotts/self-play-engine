"""
RL loss computation for policy gradient training.

Uses PPO-style clipped importance sampling:
- Forward pass to recompute logprobs under current policy
- PPO-style clipped objective with importance ratio
- KL penalty to prevent policy collapse (K1 advantage shaping)
"""
from contextlib import contextmanager
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn


def get_per_token_logps(
    model: nn.Module,
    input_ids: mx.array,
) -> mx.array:
    """
    Forward pass to compute per-token log-probabilities.

    Args:
        model: The language model (must return logits)
        input_ids: Token IDs, shape (batch, seq_len)

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


@contextmanager
def disable_lora(model: nn.Module):
    """
    Context manager to temporarily disable LoRA contributions.

    Sets all LoRA layer scales to 0, computes in eval mode, then restores.
    This allows computing reference policy logprobs (base model without LoRA).
    """
    try:
        from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear, LoRAEmbedding

        lora_classes = (LoRALinear, LoRASwitchLinear, LoRAEmbedding)
    except ImportError:
        # mlx_lm not installed or no LoRA classes available
        lora_classes = ()

    original_scales = {}
    for name, module in model.named_modules():
        if lora_classes and isinstance(module, lora_classes):
            original_scales[name] = module.scale
            module.scale = 0.0

    was_training = model.training
    model.eval()

    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in original_scales:
                module.scale = original_scales[name]
        if was_training:
            model.train()


def get_ref_logprobs(model: nn.Module, input_ids: mx.array) -> mx.array:
    """
    Get log-probabilities from the base model with LoRA disabled.

    Temporarily sets all LoRA scales to 0, computes logprobs, then restores.
    Uses stop_gradient since we don't need gradients through reference policy.

    Args:
        model: The language model with LoRA attached
        input_ids: Token IDs, shape (batch, seq_len)

    Returns:
        Reference log-probabilities, shape (batch, seq_len - 1)
    """
    with disable_lora(model):
        ref_logprobs = mx.stop_gradient(get_per_token_logps(model, input_ids))
        mx.eval(ref_logprobs)  # Materialize to free computation graph
    return ref_logprobs


def compute_loss(
    model: nn.Module,
    input_ids: mx.array,
    inference_logprobs: mx.array,
    advantages: mx.array,
    loss_mask: mx.array,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    kl_coef: float = 0.1,
    use_kl_penalty: bool = True,
    loss_type: str = "token",
    importance_sampling: str = "token",
    gspo_clip_epsilon: float = 3e-4,
    kl_clip_max: float = 0.5,
    clip_skip_threshold: float = 0.5,
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Compute the RL loss for policy gradient training.

    Uses PPO-style clipped importance sampling with K1 KL advantage shaping.

    Args:
        model: The language model
        input_ids: Token IDs, shape (batch, seq_len)
        inference_logprobs: Log-probs from inference, shape (batch, seq_len)
            Will be sliced to (batch, seq_len - 1) internally.
        advantages: Per-token advantages, shape (batch, seq_len)
            Will be sliced to (batch, seq_len - 1) internally.
        loss_mask: Binary mask for which tokens to train on, shape (batch, seq_len)
            Will be sliced to (batch, seq_len - 1) internally.
        clip_low: Lower bound for importance ratio clipping (e.g., 0.8)
        clip_high: Upper bound for importance ratio clipping (e.g., 1.2)
        kl_coef: K1 KL shaping coefficient (subtracts from advantages)
        use_kl_penalty: Whether to enable K1 KL advantage shaping
        loss_type: "token" (DAPO) or "sample" (GRPO) normalization
        importance_sampling: "token" (per-token ratios) or "sequence" (GSPO)
        gspo_clip_epsilon: Clip epsilon for GSPO (sequence mode uses 1 ± epsilon)
        kl_clip_max: Maximum K1 value before clipping (prevents extreme shaping)
        clip_skip_threshold: Skip batch if clip fraction exceeds this threshold

    Returns:
        Tuple of (loss, metrics_dict)
        - loss: Scalar loss value (0 if batch skipped)
        - metrics_dict: Dictionary with clip_fraction, kl, skip_batch, etc.
    """
    # Slice to (batch, seq_len - 1) to align with prediction semantics
    # collate() returns (batch, seq_len) but we need to match trainer_logprobs
    # which is (batch, seq_len - 1) from get_per_token_logps
    inference_logprobs = inference_logprobs[:, :-1]
    advantages = advantages[:, :-1]
    loss_mask = loss_mask[:, :-1]

    metrics = {}

    # Forward pass to get current policy log-probs
    trainer_logprobs = get_per_token_logps(model, input_ids)

    # K1 KL shaping: compute KL divergence from reference policy (base model)
    # and subtract from advantages to discourage policy drift
    if use_kl_penalty:
        # Get reference policy logprobs (base model, LoRA disabled)
        ref_logprobs = get_ref_logprobs(model, input_ids)

        # Per-token K1: log π_new(a) - log π_ref(a)
        k1_t = trainer_logprobs - ref_logprobs  # (batch, seq_len-1)

        # Length-normalized K1 (masked mean, not sum)
        k1_seq = (k1_t * loss_mask).sum(axis=1) / mx.maximum(
            loss_mask.sum(axis=1), 1.0
        )  # (batch,)

        # Clip K1 for stability (clip raw K1 first, then scale)
        k1_seq_clipped = mx.clip(k1_seq, -kl_clip_max, kl_clip_max)

        # Subtract from advantages: A' = A - kl_coef * stop_gradient(K1)
        # Multiply by kl_coef AFTER clipping so kl_clip_max is interpretable as raw K1 bounds
        k1_penalty = (
            mx.expand_dims(mx.stop_gradient(k1_seq_clipped), axis=1) * kl_coef
        )
        adjusted_advantages = advantages - k1_penalty  # (batch, seq_len-1)

        # Log K1 metrics
        metrics["k1_seq_mean"] = k1_seq.mean()
        metrics["k1_clipped_frac"] = (
            (k1_seq < -kl_clip_max) | (k1_seq > kl_clip_max)
        ).mean()
    else:
        adjusted_advantages = advantages

    # Compute log importance ratio (per-token)
    log_ratio = trainer_logprobs - inference_logprobs

    # Determine effective clip bounds
    if importance_sampling == "sequence":
        # GSPO uses much tighter clips (paper: ~3e-4)
        effective_clip_low = 1.0 - gspo_clip_epsilon
        effective_clip_high = 1.0 + gspo_clip_epsilon
    else:
        effective_clip_low = clip_low
        effective_clip_high = clip_high

    # Compute importance weights based on sampling level
    if importance_sampling == "sequence":
        # =====================================================================
        # Pure GSPO (Eq. 5): sequence-level importance ratio and advantage
        # =====================================================================
        # s_i = exp(mean(log_ratio over masked tokens))
        seq_log_ratio = (log_ratio * loss_mask).sum(axis=1) / mx.maximum(
            loss_mask.sum(axis=1), 1.0
        )
        si = mx.exp(seq_log_ratio)  # (batch,)

        # Clip at sequence level
        si_clipped = mx.clip(si, effective_clip_low, effective_clip_high)  # (batch,)

        # Collapse per-token adjusted advantages to sequence mean (Eq. 5 expects scalar per sequence)
        seq_adv = (adjusted_advantages * loss_mask).sum(axis=1) / mx.maximum(
            loss_mask.sum(axis=1), 1.0
        )

        # Track advantage variance as metric (don't eval inside loss - would cause sync)
        adv_mean = mx.expand_dims(seq_adv, axis=1)
        adv_dev = (mx.abs((adjusted_advantages - adv_mean) * loss_mask)).max()
        metrics["adv_max_dev"] = adv_dev  # Logged outside grad path

        # PPO-style clipped objective at sequence level (scalar per sequence)
        unclipped_obj = si * seq_adv
        clipped_obj = si_clipped * seq_adv
        seq_loss = -mx.minimum(unclipped_obj, clipped_obj)  # (batch,)

        # Average across sequences (GSPO Eq. 5 reduction)
        pg_loss_scalar = seq_loss.mean()

        # Clipping metrics at sequence level
        is_clipped = (si < effective_clip_low) | (si > effective_clip_high)
        clip_fraction = is_clipped.mean()
        metrics["clip_fraction"] = clip_fraction

        # ---------------------------------------------------------------------
        # GSPO-token (Eq. 13-14) - commented out, for future token-varying advantages
        # ---------------------------------------------------------------------
        # si_expanded = mx.expand_dims(si, axis=1)  # (batch, 1)
        # si_detached = mx.stop_gradient(si_expanded)
        # token_factor = mx.exp(trainer_logprobs - mx.stop_gradient(trainer_logprobs))  # π/sg[π] = 1
        # importance_ratio = si_detached * token_factor  # (batch, seq_len-1)
        # clipped_ratio = mx.clip(si_detached, effective_clip_low, effective_clip_high) * token_factor
        # ... then use token-level PPO objective with per-token advantages
        # ---------------------------------------------------------------------

    else:
        # Token-level (default): each token has its own importance ratio
        importance_ratio = mx.exp(log_ratio)  # (batch, seq_len-1)
        clipped_ratio = mx.clip(importance_ratio, effective_clip_low, effective_clip_high)

        # PPO-style clipped surrogate objective
        # We want to maximize: min(ratio * A, clip(ratio) * A)
        # So we minimize: -min(ratio * A, clip(ratio) * A)
        unclipped_obj = importance_ratio * adjusted_advantages
        clipped_obj = clipped_ratio * adjusted_advantages

        # Take the minimum (pessimistic bound)
        pg_loss = -mx.minimum(unclipped_obj, clipped_obj)

        # Compute clipping metrics (use effective bounds)
        is_clipped = (importance_ratio < effective_clip_low) | (
            importance_ratio > effective_clip_high
        )
        clip_fraction = (is_clipped * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1.0)
        metrics["clip_fraction"] = clip_fraction

    # Approximate KL divergence (K2): (r - 1) - log(r) ≈ 0.5 * (log_r)^2
    # Kept for monitoring, not added to loss (K1 shaping is used instead when enabled)
    approx_kl = 0.5 * (log_ratio**2)
    mean_kl = (approx_kl * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1.0)
    metrics["kl"] = mean_kl

    # Clip fraction gating: skip batch if too many tokens are clipped
    # This indicates the policy has diverged too much from the reference
    skip_batch = clip_fraction > clip_skip_threshold
    metrics["skip_batch"] = skip_batch

    if skip_batch:
        # Zero loss means zero gradients (d/dx(0) = 0)
        # Metrics still computed for monitoring
        metrics["loss"] = mx.array(0.0)
        metrics["pg_loss"] = mx.array(0.0)
        return mx.array(0.0), metrics

    # Compute final loss scalar
    # Note: GSPO (sequence branch) already computed pg_loss_scalar directly above
    if importance_sampling != "sequence":
        # Token-level: apply mask and reduce
        masked_loss = pg_loss * loss_mask

        if loss_type == "token":
            # DAPO: all tokens equal weight
            pg_loss_scalar = masked_loss.sum() / mx.maximum(loss_mask.sum(), 1.0)
        else:
            # GRPO: per-sample mean, then batch mean
            per_sample_loss = masked_loss.sum(axis=1) / mx.maximum(
                loss_mask.sum(axis=1), 1.0
            )
            pg_loss_scalar = per_sample_loss.mean()

    # Final loss is just the PG loss
    # K1 KL shaping is applied to advantages (above), not added to loss
    loss = pg_loss_scalar

    metrics["loss"] = loss
    metrics["pg_loss"] = pg_loss_scalar

    return loss, metrics


def make_loss_fn(
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    kl_coef: float = 0.1,
    use_kl_penalty: bool = True,
    loss_type: str = "token",
    importance_sampling: str = "token",
    gspo_clip_epsilon: float = 3e-4,
    kl_clip_max: float = 0.5,
    clip_skip_threshold: float = 0.5,
):
    """
    Create a loss function suitable for use with nn.value_and_grad.

    The returned function takes (model, batch_data) and returns (loss, metrics).

    IMPORTANT: nn.value_and_grad differentiates with respect to the FIRST returned
    value, but passes through all return values. So if this returns (loss, metrics),
    value_and_grad returns ((loss, metrics), grads). This allows us to get both
    loss and metrics in a single forward pass.

    Args:
        clip_low: Lower bound for importance ratio clipping
        clip_high: Upper bound for importance ratio clipping
        kl_coef: K1 KL shaping coefficient (subtracts from advantages)
        use_kl_penalty: Whether to enable K1 KL advantage shaping
        loss_type: "token" (DAPO) or "sample" (GRPO) normalization
        importance_sampling: "token" (per-token ratios) or "sequence" (GSPO)
        gspo_clip_epsilon: Clip epsilon for GSPO (sequence mode uses 1 ± epsilon)
        kl_clip_max: Maximum K1 value before clipping (prevents extreme shaping)
        clip_skip_threshold: Skip batch if clip fraction exceeds this threshold

    Returns:
        A loss function compatible with nn.value_and_grad that returns (loss, metrics)
    """

    def loss_fn(
        model: nn.Module,
        input_ids: mx.array,
        inference_logprobs: mx.array,
        advantages: mx.array,
        loss_mask: mx.array,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        loss, metrics = compute_loss(
            model=model,
            input_ids=input_ids,
            inference_logprobs=inference_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            clip_low=clip_low,
            clip_high=clip_high,
            kl_coef=kl_coef,
            use_kl_penalty=use_kl_penalty,
            loss_type=loss_type,
            importance_sampling=importance_sampling,
            gspo_clip_epsilon=gspo_clip_epsilon,
            kl_clip_max=kl_clip_max,
            clip_skip_threshold=clip_skip_threshold,
        )
        # Return (loss, metrics) - value_and_grad diffs w.r.t. loss, passes through metrics
        return loss, metrics

    return loss_fn
