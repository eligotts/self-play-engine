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
    from self_play.core import Arena, OpenAIClient
    from self_play.training import Trainer, TrainerConfig, WeightPublisher, training_loop

    # Setup arena
    client = OpenAIClient.for_local(port=8000)
    arena = DebateArena(client=client, ...)

    # Setup trainer
    model, tokenizer = load_model_with_lora(...)
    optimizer = mx.optimizers.Adam(learning_rate=1e-5)
    config = TrainerConfig(
        micro_token_budget=4096,
        max_policy_lag=3,
    )
    publisher = WeightPublisher(base_url="http://localhost:8000")
    trainer = Trainer(model, optimizer, config, publisher)

    # Run training
    batch_queue = asyncio.Queue(maxsize=4)
    await training_loop(arena, trainer, batch_queue, num_steps=1000)
"""

from .config import TrainerConfig
from .trainer import Trainer
from .weight_publisher import WeightPublisher
from .loop import training_loop, simple_training_loop
from .batching import split_by_token_budget, collate
from .loss import compute_loss, get_per_token_logps, make_loss_fn

__all__ = [
    # Core classes
    "TrainerConfig",
    "Trainer",
    "WeightPublisher",
    # Training loops
    "training_loop",
    "simple_training_loop",
    # Utilities
    "split_by_token_budget",
    "collate",
    # Loss functions
    "compute_loss",
    "get_per_token_logps",
    "make_loss_fn",
]
