"""
Train on GSM8K math dataset using GRPO-style sampling.

To run:
1. Start mlx-vllm server with LoRA enabled:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Run training:
   python examples/train_gsm8k.py

Requirements:
   pip install datasets
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient, Role, Artifact, GRPOCredit
from self_play.tasks.gsm8k import (
    GSM8KArena,
    GSM8KEpisode,
    SYSTEM_PROMPT,
    load_gsm8k,
    gsm8k_reward,
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
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    print(f"Attaching LoRA (rank={lora_rank}, layers={lora_layers})...")
    lora_config = {
        "rank": lora_rank,
        "scale": lora_scale,
        "dropout": 0.0,
    }
    linear_to_lora_layers(model, lora_layers, lora_config)

    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML <answer> tags."""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


async def preview_gsm8k(arena, concurrency: int = 4):
    """Preview GSM8K arena output with math-specific metrics."""
    print("\n=== GSM8K Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    # Group by question (same artifact = same question in GRPO)
    # Use artifact content as key since rollout_id is unique per response
    questions = {}
    for r in batch.records:
        artifact = r.meta.get("artifact", {}) or {}
        q_key = artifact.get("question", r.rollout_id)[:50]  # Use first 50 chars as key
        if q_key not in questions:
            questions[q_key] = {"artifact": artifact, "records": []}
        questions[q_key]["records"].append(r)

    print(f"Questions: {len(questions)} | Responses: {len(batch.records)}\n")

    # Count correct/incorrect
    correct_count = 0
    total_count = len(batch.records)

    for i, (q_key, data) in enumerate(questions.items(), 1):
        artifact = data["artifact"]
        records = data["records"]

        question = artifact.get("question", "N/A")
        expected = str(artifact.get("answer", "N/A")).strip()

        print(f"[{i}] Q: {truncate(question, 80)}")
        print(f"    Expected: {expected}")

        # Show each response for this question
        for j, record in enumerate(records):
            response = record.completion_text or ""
            extracted = extract_xml_answer(response)

            # Normalize for comparison
            extracted_norm = extracted.replace(",", "").replace("$", "").strip()
            expected_norm = expected.replace(",", "").replace("$", "").strip()
            is_correct = extracted_norm == expected_norm

            if is_correct:
                correct_count += 1

            status = "OK" if is_correct else "X"
            print(f"    [{status}] Got: {extracted_norm or '(none)'} | reward={record.reward:.2f} | {truncate(response, 60)}")

        print()

    # Summary
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")

    # Reward stats
    all_rewards = [r.reward for r in batch.records]
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"Avg reward: {avg_reward:.3f} | Range: {min(all_rewards):.3f} to {max(all_rewards):.3f}")


async def main(args):
    # Load dataset
    dataset = load_gsm8k(split=args.split, max_samples=args.max_samples)

    # Setup inference client
    base_url = args.url if args.url else f"http://{args.host}:{args.port}"
    print(f"\nConnecting to inference server at {base_url}...")

    if args.url:
        client = OpenAIClient(
            api_key="not-needed",
            model="local",
            base_url=f"{base_url.rstrip('/')}/v1",
            timeout=240.0,
        )
    else:
        client = OpenAIClient.for_local(host=args.host, port=args.port, timeout=240.0)

    # Setup arena
    arena = GSM8KArena(
        client=client,
        batch_size=args.batch_size,
        credit_assigner=GRPOCredit(normalize=True),
        verbose=args.verbose,
    )

    # Add trainable role
    system_prompt = args.system_prompt if args.system_prompt else SYSTEM_PROMPT
    arena.add_role(Role(
        id="model",
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    # Add episode
    arena.add_episode("gsm8k", GSM8KEpisode(
        role_id="model",
        reward_fn=gsm8k_reward,
        prompt_template=args.prompt_template,
    ))

    # Load dataset into store
    store = arena.add_store("gsm8k")
    for i, item in enumerate(dataset):
        store.add(Artifact(id=f"row_{i}", data=item))
    print(f"Loaded {store.count()} questions")

    # Dry-run mode: preview arena performance without training
    if args.dry_run:
        await preview_gsm8k(arena, concurrency=args.concurrency)
        await client.close()
        return

    # Load model with LoRA
    model, tokenizer = load_model_with_lora(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
    )

    # Setup trainer
    optimizer = optim.Adam(learning_rate=args.lr)
    config = TrainerConfig(
        lr=args.lr,
        micro_token_budget=args.micro_token_budget,
        max_policy_lag=args.max_policy_lag,
        batch_size=args.train_batch_size,
        clip_low=0.8,
        clip_high=1.2,
        kl_coef=args.kl_coef,
        use_kl_penalty=args.use_kl_penalty,
        weight_push_url=base_url,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    publisher = WeightPublisher(base_url=base_url)
    trainer = Trainer(model=model, optimizer=optimizer, config=config, publisher=publisher)

    # Run training
    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"  - Dataset: GSM8K ({args.split} split, {len(dataset)} samples)")
    print(f"  - Batch size: {args.batch_size} rollouts per question (GRPO)")
    print(f"  - Train batch size: {args.train_batch_size} records per step")
    print()

    try:
        if args.simple_loop:
            await simple_training_loop(
                arena=arena,
                trainer=trainer,
                num_steps=args.num_steps,
                concurrency=args.concurrency,
                step_concurrency=args.step_concurrency,
                verbose=args.verbose,
            )
        else:
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
        await publisher.close()
        await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on GSM8K with GRPO")

    # Model
    parser.add_argument("--model-path", type=str,
        # default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit")
        default="/Users/eligottlieb/.lmstudio/models/mlx-community/Qwen3-0.6B-8bit")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=int, default=16)

    # Server
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    # Dataset
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)

    # Training
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=12)
    parser.add_argument("--micro-token-budget", type=int, default=2048)
    parser.add_argument("--max-policy-lag", type=int, default=3)
    parser.add_argument("--kl-coef", type=float, default=0.2)
    parser.add_argument("--use-kl-penalty", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--step-concurrency", type=int, default=1)

    # Generation
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--prompt-template", default="{question} \n\n/no_think")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=786)

    # Mode
    parser.add_argument("--simple-loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    asyncio.run(main(args))
