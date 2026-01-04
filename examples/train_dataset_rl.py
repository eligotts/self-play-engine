"""
Example: Training with fixed dataset RL using GSM8K.

This script demonstrates how to use the self-play training module for
standard RL with a fixed dataset. It combines:
- The DatasetArena from the examples module (for GRPO-style generation)
- The Trainer from the training module (for training)
- An mlx-vllm server for inference (with LoRA hot-swap)
- GSM8K dataset from Hugging Face

To run this example:
1. Start mlx-vllm server with LoRA enabled:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Run this script:
   python examples/train_dataset_rl.py

Requirements:
   pip install datasets
"""
import asyncio
import argparse
import re
from typing import Dict, List, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient, Rollout, Arena, Role, Artifact, GRPOCredit
from self_play.examples.dataset_rl import (
    DatasetArena,
    DatasetEpisode,
)
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


def load_gsm8k(split: str = "train", max_samples: int | None = None) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from Hugging Face.

    Args:
        split: Dataset split ("train" or "test")
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of dicts with "question" and "answer" keys
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading GSM8K {split} split from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        # GSM8K format: question field and answer field
        # Answer contains reasoning followed by "#### <number>"
        question = item["question"]
        full_answer = item["answer"]

        # Extract the final numerical answer after ####
        answer = extract_gsm8k_answer(full_answer)

        samples.append({
            "question": question,
            "answer": answer,
            "full_answer": full_answer,  # Keep full reasoning for reference
        })

    print(f"Loaded {len(samples)} samples from GSM8K")
    return samples


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K format."""
    # GSM8K answers end with "#### <number>"
    match = re.search(r"####\s*(.+?)$", answer_text.strip())
    if match:
        # Clean up: remove commas from numbers, strip whitespace
        answer = match.group(1).strip()
        answer = answer.replace(",", "")
        return answer
    # Fallback: return last line
    return answer_text.strip().split("\n")[-1]


def gsm8k_reward(
    rollout: Rollout,
    arena: Arena,
    max_chars: int = 500,
) -> Dict[str, float]:
    """
    GSM8K reward: correctness + brevity (only if correct).

    - If wrong: reward = 0.0
    - If correct: reward = 0.5 + 0.5 * brevity (range 0.5 to 1.0)

    Brevity uses linear decay tuned for 50-100 token responses.
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    completion = rollout.steps[-1].completion_text.strip()
    ground_truth = str(rollout.artifact.get("answer", "")).strip()

    # --- Extract answer from completion ---
    extracted = None

    # Try "The answer is: X" format
    match = re.search(r"[Tt]he answer is:?\s*(.+?)(?:\.|$)", completion)
    if match:
        extracted = match.group(1).strip()

    # Try "#### X" format (GSM8K style)
    if extracted is None:
        match = re.search(r"####\s*(.+?)$", completion, re.MULTILINE)
        if match:
            extracted = match.group(1).strip()

    # Try to find last number in response
    if extracted is None:
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", completion)
        if numbers:
            extracted = numbers[-1]

    # Fallback to last line
    if extracted is None:
        extracted = completion.split("\n")[-1].strip()

    # --- Normalize for comparison ---
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = s.replace(",", "")
        s = re.sub(r"[^\d.-]", "", s)
        return s

    extracted_norm = normalize(extracted)
    truth_norm = normalize(ground_truth)
    is_correct = extracted_norm == truth_norm

    # --- Compute reward ---
    if is_correct:
        # Brevity bonus: linear decay from 1.0 at 0 chars to 0.0 at max_chars
        brevity = max(0.0, 1.0 - (len(completion) / max_chars))
        reward = 0.5 + 0.5 * brevity  # Range: [0.5, 1.0]
    else:
        reward = 0.0

    if arena.verbose:
        if is_correct:
            print(f"    [gsm8k] CORRECT: '{extracted_norm}' | len={len(completion)} | reward={reward:.2f}")
        else:
            print(f"    [gsm8k] WRONG: got '{extracted_norm}' vs '{truth_norm}' | reward=0.0")

    return {actor: reward}


async def main(args):
    # Load GSM8K dataset
    dataset = load_gsm8k(
        split=args.split,
        max_samples=args.max_samples,
    )

    # Setup inference server URL
    base_url = args.url if args.url else f"http://{args.host}:{args.port}"
    print(f"\nConnecting to inference server at {base_url}...")

    # Setup inference client
    if args.url:
        client = OpenAIClient(
            api_key="not-needed",
            model="local",
            base_url=f"{base_url.rstrip('/')}/v1",
            timeout=240.0,
        )
    else:
        client = OpenAIClient.for_local(host=args.host, port=args.port, timeout=240.0)

    # Setup arena with GSM8K
    print("Setting up dataset RL arena with GSM8K...")
    arena = DatasetArena(
        client=client,
        batch_size=args.batch_size,
        credit_assigner=GRPOCredit(),
        verbose=args.verbose,
    )

    # Add trainable role
    arena.add_role(Role(
        id="model",
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    # Add episode
    episode = DatasetEpisode(
        role_id="model",
        question_key="question",
        answer_key="answer",
        reward_fn=gsm8k_reward,
        prompt_template=args.prompt_template,
    )
    arena.add_episode("dataset_rl", episode)

    # Load dataset
    store = arena.add_store("dataset")
    for i, item in enumerate(dataset):
        store.add(Artifact(id=f"row_{i}", data=item))

    print(f"Loaded {store.count()} questions from GSM8K")

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
    print(f"  - Dataset: GSM8K ({args.split} split, {len(dataset)} samples)")
    print(f"  - Batch size: {args.batch_size} rollouts per question (GRPO)")
    print(f"  - Train batch size: {args.train_batch_size} records per train step")
    print(f"  - Micro token budget: {args.micro_token_budget}")
    print(f"  - Max policy lag: {args.max_policy_lag}")
    print(f"  - Importance sampling: {args.use_importance_sampling}")
    print()

    try:
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
                step_concurrency=args.step_concurrency,
                verbose=args.verbose,
            )
    finally:
        # Always cleanup, even on error
        await publisher.close()
        await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with fixed dataset RL using GSM8K")

    # Model args
    parser.add_argument(
        "--model-path",
        type=str,
        # default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit",
        default="/Users/eligottlieb/.lmstudio/models/mlx-community/Qwen3-0.6B-8bit",
        help="Path to the base model",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=16, help="LoRA layers")

    # Server args
    parser.add_argument("--url", type=str, default=None, help="Full base URL of inference server (overrides host/port)")
    parser.add_argument("--host", type=str, default="localhost", help="Inference server host")
    parser.add_argument("--port", type=int, default=8000, help="Inference server port")

    # Dataset args
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="GSM8K split")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to load (None for all)")

    # Training args
    parser.add_argument("--num-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Rollouts per question (GRPO)")
    parser.add_argument("--train-batch-size", type=int, default=12, help="Records per train step")
    parser.add_argument("--micro-token-budget", type=int, default=2048, help="Tokens per micro-batch")
    parser.add_argument("--max-policy-lag", type=int, default=3, help="Max staleness (steps)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent episodes")
    parser.add_argument("--step-concurrency", type=int, default=1, help="Max concurrent arena.step() calls")

    # Generation args
    parser.add_argument(
        "--system-prompt",
        default="You are an expert at math. Solve the question given to you efficiently. You must end your response with \"The answer is: \" followed by the answer.",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--prompt-template",
        default="{question}\n\nSolve this step by step. End with \"The answer is: \" followed by the answer.",
        help="Template for formatting questions",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per completion")

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
