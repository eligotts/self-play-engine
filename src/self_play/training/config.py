"""
Training configuration for the RL trainer.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrainerConfig:
    """
    Configuration for the RL trainer.

    This trainer is designed as a clean, minimal implementation of core RL
    training patterns for MLX.
    """

    # Learning rate
    lr: float = 1e-5

    # Micro-batching: max tokens per micro-batch (for gradient accumulation)
    micro_token_budget: int = 4096

    # Staleness: discard records from policies more than N steps behind
    max_policy_lag: int = 3

    # Minimum records needed before training (trainer waits for full batch)
    batch_size: int = 32

    # PPO-style importance ratio clipping bounds
    clip_low: float = 0.8
    clip_high: float = 1.2

    # Loss normalization: "token" (DAPO) or "sample" (GRPO)
    # - "token": all tokens weighted equally across batch (recommended)
    # - "sample": each sample weighted equally regardless of length
    loss_type: Literal["token", "sample"] = "token"

    # Importance sampling level: "token" or "sequence" (GSPO)
    # - "token": per-token importance ratios (GRPO/DAPO style)
    # - "sequence": sequence-level importance ratios (GSPO, recommended for stability)
    importance_sampling: Literal["token", "sequence"] = "token"

    # GSPO clip epsilon (only used when importance_sampling="sequence")
    # GSPO uses much tighter clips than token-level PPO (paper uses ~3e-4)
    # clip bounds become [1 - epsilon, 1 + epsilon]
    gspo_clip_epsilon: float = 3e-4

    # KL divergence penalty coefficient (prevents policy collapse)
    # Higher values = stronger regularization toward reference policy
    # Recommended: 0.05-0.2 for stable training
    kl_coef: float = 0.1

    # Whether to add KL penalty to the loss (set True to enable regularization)
    use_kl_penalty: bool = False

    # Weight publishing: URL of mlx-vllm inference server
    weight_push_url: str = "http://localhost:8000"

    # Pad token ID for collation (typically 0 or tokenizer.pad_token_id)
    pad_token_id: int = 0

    # Memory management: eval after each micro-batch to free memory
    # Set to True for memory-constrained systems (like MacBook)
    # Set to False for systems with lots of VRAM (batched eval is faster)
    eval_per_micro_batch: bool = True

    # Wandb logging (set wandb_project to enable)
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Evaluation settings
    eval_every: int = 10  # Run eval every N steps (0 = disabled)
    eval_concurrency: int = 8  # Concurrency for eval rollouts
