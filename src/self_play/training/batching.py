"""
Micro-batch utilities for token-budget-based splitting and collation.
"""
from typing import List, Tuple

import mlx.core as mx

from ..core.types import TrainingRecord


def split_by_token_budget(
    records: List[TrainingRecord],
    budget: int,
) -> List[List[TrainingRecord]]:
    """
    Split records into chunks where each chunk has <= budget total tokens.

    This ensures consistent memory usage across micro-batches regardless of
    individual sequence lengths.

    Args:
        records: List of training records to split
        budget: Maximum total tokens per chunk

    Returns:
        List of record chunks, each with total tokens <= budget
    """
    if not records:
        return []

    chunks: List[List[TrainingRecord]] = []
    current_chunk: List[TrainingRecord] = []
    current_tokens = 0

    for record in records:
        record_tokens = len(record.input_ids)

        # If this record alone exceeds budget, put it in its own chunk
        if record_tokens > budget:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            chunks.append([record])
            continue

        # If adding this record would exceed budget, start new chunk
        if current_tokens + record_tokens > budget and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(record)
        current_tokens += record_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def collate(
    records: List[TrainingRecord],
    pad_token_id: int = 0,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Collate training records into padded tensors for training.

    Handles the offset between input_ids and targets (logprobs are for predicting
    the next token, so we shift everything by 1).

    Args:
        records: List of training records
        pad_token_id: Token ID to use for padding

    Returns:
        Tuple of (input_ids, loss_mask, inference_logprobs, advantages)
        All tensors have shape (batch_size, max_seq_len)
    """
    if not records:
        raise ValueError("Cannot collate empty records list")

    max_len = max(len(r.input_ids) for r in records)

    input_ids_list = []
    loss_mask_list = []
    inference_logprobs_list = []
    advantages_list = []

    for r in records:
        seq_len = len(r.input_ids)
        pad_len = max_len - seq_len

        # Pad input_ids
        padded_ids = r.input_ids + [pad_token_id] * pad_len
        input_ids_list.append(padded_ids)

        # Loss mask: action_mask shifted by 1 (since logprobs[i] predicts token[i+1])
        # Then padded with zeros
        # Original action_mask: [0]*prompt_len + [1]*completion_len
        # After shift: [0]*(prompt_len-1) + [1]*completion_len + [0] for the last position
        # We want to mask positions where we compute loss (completion tokens only)
        shifted_mask = r.action_mask[1:] + [0]
        padded_mask = shifted_mask + [0] * pad_len
        loss_mask_list.append(padded_mask)

        # Inference logprobs: pad with zeros for prompt, then include completion logprobs
        # Logprobs are already aligned to completion tokens
        # We need: [0]*prompt_len + logprobs, then shift by 1 and pad
        prompt_len = len(r.prompt_token_ids)
        full_logprobs = [0.0] * prompt_len + r.logprobs
        # Pad to seq_len, then take shifted view
        if len(full_logprobs) < seq_len:
            full_logprobs = full_logprobs + [0.0] * (seq_len - len(full_logprobs))
        # Shift by 1 (logprobs[i] is for predicting token[i])
        shifted_lp = full_logprobs[1:] + [0.0]
        padded_lp = shifted_lp + [0.0] * pad_len
        inference_logprobs_list.append(padded_lp)

        # Advantages: broadcast step-level advantage to all tokens, shift, pad
        # In RL, advantage is per-action, and here each token is an action
        full_adv = [r.advantage] * seq_len
        shifted_adv = full_adv[1:] + [0.0]
        padded_adv = shifted_adv + [0.0] * pad_len
        advantages_list.append(padded_adv)

    return (
        mx.array(input_ids_list),
        mx.array(loss_mask_list, dtype=mx.float32),
        mx.array(inference_logprobs_list),
        mx.array(advantages_list),
    )
