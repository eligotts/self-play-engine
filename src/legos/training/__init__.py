"""
Training module for self-play RL.

This module provides a clean, minimal trainer for MLX that implements
the core patterns of modern RL training:

- Async generation and training with queue-based coordination
- Staleness filtering by policy version
- Token-budget-based micro-batching
- Gradient accumulation
- Online weight updates to inference server (LoRA hot-swap)

Example usage:
    from legos.core import OpenAIClient
    from legos.tasks import GSM8KArena
    from legos.training import Trainer, TrainerConfig, training_loop

    # Setup arena
    client = OpenAIClient(base_url="http://localhost:8000/v1")
    arena = GSM8KArena(client=client, episodes_per_step=8)

    # Setup trainer
    model, tokenizer = load_model_with_lora(...)
    optimizer = mx.optimizers.Adam(learning_rate=1e-5)
    config = TrainerConfig(
        micro_batch_tokens=4096,
        staleness_limit=3,
    )
    trainer = Trainer(model, optimizer, config, client)

    # Run training
    batch_queue = asyncio.Queue(maxsize=4)
    await training_loop(arena, trainer, batch_queue, num_steps=1000)
"""

from .config import TrainerConfig
from .trainer import Trainer
from .loop import training_loop, synchronous_training_loop
from .batching import split_by_token_budget, collate, estimate_tokens, form_micro_batch
from .loss import compute_loss, get_per_token_logps, make_loss_fn

__all__ = [
    # Core classes
    "TrainerConfig",
    "Trainer",
    # Training loops
    "training_loop",
    "synchronous_training_loop",
    # Utilities
    "split_by_token_budget",
    "collate",
    "estimate_tokens",
    "form_micro_batch",
    # Loss functions
    "compute_loss",
    "get_per_token_logps",
    "make_loss_fn",
]
