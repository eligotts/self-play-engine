"""
Training configuration for the RL trainer.
"""
from dataclasses import dataclass, field
from typing import Optional


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

    # Importance sampling toggle:
    # - True: Forward pass to recompute logprobs, PPO-style clipped objective
    # - False: Use inference logprobs directly (faster, for MLX inference=training)
    use_importance_sampling: bool = True

    # PPO-style importance ratio clipping bounds
    clip_low: float = 0.8
    clip_high: float = 1.2

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
