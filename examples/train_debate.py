"""
Example: Training a debate agent with RL.

This script demonstrates how to use the self-play training module to train
a debate agent. It combines:
- The DebateArena from the examples module (for generation)
- The Trainer from the training module (for training)
- An mlx-vllm server for inference (with LoRA hot-swap)

To run this example:
1. Start mlx-vllm server with LoRA enabled:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \\
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Run this script:
   python examples/train_debate.py
"""
import asyncio
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient
from self_play.examples.debate import create_debate_arena
from self_play.training import (
    Trainer,
    TrainerConfig,
    WeightPublisher,
    training_loop,
    simple_training_loop,
)


def load_model_with_lora(
    model_path: str,
    lora_rank: int = 8,
    lora_layers: int = 16,
    lora_scale: float = 20.0,
):
    """
    Load model and attach LoRA adapters.

    Args:
        model_path: Path to the base model
        lora_rank: LoRA rank (must match server config)
        lora_layers: Number of layers to apply LoRA to
        lora_scale: LoRA scaling factor

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    print(f"Attaching LoRA (rank={lora_rank}, layers={lora_layers})...")
    lora_config = {
        "rank": lora_rank,
        "scale": lora_scale,
        "dropout": 0.0,
    }
    linear_to_lora_layers(model, lora_layers, lora_config)

    # Count trainable parameters
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


async def main(args):
    # Topics for debate training
    topics = [
        "Social media does more harm than good to society",
        "Artificial intelligence will create more jobs than it destroys",
        "Climate change is the most pressing issue of our time",
        "Universal basic income should be implemented globally",
        "Space exploration is a worthwhile investment for humanity",
        "Remote work is better than office work",
        "Nuclear energy is the solution to climate change",
        "Cryptocurrencies will replace traditional currencies",
        "Genetic engineering of humans should be allowed",
        "Autonomous vehicles should replace human drivers",
        "Free speech should have no legal limits",
        "Democracy is the best form of government",
        "Zoos do more harm than good to animal conservation",
        "College education is no longer worth the cost",
        "The death penalty should be abolished worldwide",
        "Billionaires should not exist in a just society",
        "Animal testing for medical research is ethically justified",
        "Voting should be mandatory for all citizens",
        "Privacy is more important than national security",
        "Art created by AI should not be considered real art",
        "Professional athletes are overpaid relative to their social value",
        "Childhood social media use should be heavily restricted by law",
    ]

    # Setup inference client
    print(f"\nConnecting to inference server at localhost:{args.port}...")
    client = OpenAIClient.for_local(port=args.port, timeout=120.0)

    # Setup arena
    print("Setting up debate arena...")
    arena = create_debate_arena(
        client=client,
        topics=topics,
        num_rounds=3,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Load model with LoRA
    model, tokenizer = load_model_with_lora(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
    )

    # Setup optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # Setup trainer config
    config = TrainerConfig(
        lr=args.lr,
        micro_token_budget=args.micro_token_budget,
        max_policy_lag=args.max_policy_lag,
        batch_size=args.train_batch_size,
        use_importance_sampling=args.use_importance_sampling,
        clip_low=0.8,
        clip_high=1.2,
        weight_push_url=f"http://localhost:{args.port}",
        pad_token_id=tokenizer.pad_token_id or 0,
        verbose=args.verbose,
    )

    # Setup weight publisher
    publisher = WeightPublisher(
        base_url=f"http://localhost:{args.port}",
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        publisher=publisher,
    )

    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"  - Batch size: {args.batch_size} debates per arena step")
    print(f"  - Train batch size: {args.train_batch_size} records per train step")
    print(f"  - Micro token budget: {args.micro_token_budget}")
    print(f"  - Max policy lag: {args.max_policy_lag}")
    print(f"  - Importance sampling: {args.use_importance_sampling}")
    print()

    if args.simple_loop:
        # Use simple sequential loop (easier to debug)
        await simple_training_loop(
            arena=arena,
            trainer=trainer,
            num_steps=args.num_steps,
            concurrency=args.concurrency,
            verbose=args.verbose,
        )
    else:
        # Use async loop with queue (better throughput)
        batch_queue = asyncio.Queue(maxsize=4)
        await training_loop(
            arena=arena,
            trainer=trainer,
            batch_queue=batch_queue,
            num_steps=args.num_steps,
            concurrency=args.concurrency,
            verbose=args.verbose,
        )

    # Cleanup
    await publisher.close()
    await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a debate agent with RL")

    # Model args
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit",
        # default="/Users/eligottlieb/.lmstudio/models/mlx-community/Trinity-Nano-Preview-8bit",
        help="Path to the base model",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=16, help="LoRA layers")

    # Server args
    parser.add_argument("--port", type=int, default=8000, help="Inference server port")

    # Training args
    parser.add_argument("--num-steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Debates per arena step")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Records per train step")
    parser.add_argument("--micro-token-budget", type=int, default=4096, help="Tokens per micro-batch")
    parser.add_argument("--max-policy-lag", type=int, default=3, help="Max staleness (steps)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent episodes")

    # Mode args
    parser.add_argument(
        "--use-importance-sampling",
        action="store_true",
        help="Use importance sampling (slower but needed for off-policy)",
    )
    parser.add_argument(
        "--simple-loop",
        action="store_true",
        help="Use simple sequential loop instead of async",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug info")

    args = parser.parse_args()
    asyncio.run(main(args))
