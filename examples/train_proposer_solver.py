"""
Example: Training a proposer/solver agent with RL.

This script demonstrates how to use the self-play training module to train
proposer and solver agents. It combines:
- The ProposerSolverArena from the examples module (for generation)
- The Trainer from the training module (for training)
- An mlx-vllm server for inference (with LoRA hot-swap)

The proposer learns to generate questions at an optimal difficulty level,
while the solver learns to answer them correctly.

To run this example:
1. Start mlx-vllm server with LoRA enabled:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Run this script:
   python examples/train_proposer_solver.py
"""
import asyncio
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient, Role, Artifact
from self_play.examples.proposer_solver import ProposerSolverArena, ProposerEpisode, SolveEpisode
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
    # Seed questions for the proposer to learn from
    initial_questions = [
        {"question": "What is 15 + 27? Answer as a number.", "ground_truth": "42"},
        {"question": "If x + 5 = 12, what is x? Answer as just the number.", "ground_truth": "7"},
        {"question": "What is 8 * 5? Answer as a number.", "ground_truth": "40"},
        {"question": "What is 144 / 12? Answer as a number.", "ground_truth": "12"},
        {"question": "What is 2^8? Answer as a number.", "ground_truth": "256"},
        {"question": "If 3x = 21, what is x? Answer as just the number.", "ground_truth": "7"},
        {"question": "What is the sum of 13, 17, and 22? Answer as a number.", "ground_truth": "52"},
        {"question": "What is 7! / 6!? Answer as a number.", "ground_truth": "7"},
        {"question": "If a rectangle has length 8 and width 5, what is its area? Answer as a number.", "ground_truth": "40"},
        {"question": "What is the square root of 169? Answer as a number.", "ground_truth": "13"},
    ]

    # Setup inference server URL
    base_url = args.url if args.url else f"http://{args.host}:{args.port}"
    print(f"\nConnecting to inference server at {base_url}...")

    # Setup inference client
    if args.url:
        client = OpenAIClient(
            api_key="not-needed",
            model="local",
            base_url=f"{base_url.rstrip('/')}/v1",
            timeout=120.0,
        )
    else:
        client = OpenAIClient.for_local(host=args.host, port=args.port, timeout=120.0)

    # Setup arena
    print("Setting up proposer/solver arena...")
    arena = ProposerSolverArena(client=client, batch_size=args.batch_size, verbose=args.verbose)

    # Add roles
    arena.add_role(Role(
        id="Proposer",
        system_prompt="You are a creative math problem creator. "
                      "Generate interesting, well-formed problems with clear answers."
                      "Ensure that in your question you explicitly state the form the answer should be provided in.",
        temperature=0.9,
        max_tokens=1024,
    ))

    arena.add_role(Role(
        id="Solver",
        system_prompt="You are a skilled math problem solver. "
                      "Think step by step and provide clear, correct answers."
                      "Ensure that your answer is provided in the form specified in the question.",
        temperature=0.7,
        max_tokens=1024,
    ))

    # Add episodes
    arena.add_episode("propose", ProposerEpisode(n_solver_rollouts=args.n_solver_rollouts))
    arena.add_episode("solve", SolveEpisode())

    # Add initial questions to store
    question_store = arena.add_store("questions")
    for i, q in enumerate(initial_questions):
        question_store.add(Artifact(id=f"seed_{i}", data=q))

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
        weight_push_url=base_url,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Setup weight publisher
    publisher = WeightPublisher(
        base_url=base_url,
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        publisher=publisher,
    )

    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"  - Batch size: {args.batch_size} episodes per arena step")
    print(f"  - Train batch size: {args.train_batch_size} records per train step")
    print(f"  - Solver rollouts per proposal: {args.n_solver_rollouts}")
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
    parser = argparse.ArgumentParser(description="Train proposer/solver agents with RL")

    # Model args
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit",
        help="Path to the base model",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=16, help="LoRA layers")

    # Server args
    parser.add_argument("--url", type=str, default=None, help="Full base URL of inference server (overrides host/port)")
    parser.add_argument("--host", type=str, default="localhost", help="Inference server host")
    parser.add_argument("--port", type=int, default=8000, help="Inference server port")

    # Training args
    parser.add_argument("--num-steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Episodes per arena step")
    parser.add_argument("--train-batch-size", type=int, default=24, help="Records per train step")
    parser.add_argument("--n-solver-rollouts", type=int, default=4, help="Solver rollouts per proposal")
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

    # Wandb args
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name (enables logging)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()
    asyncio.run(main(args))
