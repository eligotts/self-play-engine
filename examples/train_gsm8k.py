"""
GSM8K: Train on math word problems using GRPO-style sampling.
This is an example of how the self-play framework can be used in a more standard dataset-based training loop.

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Start inference server:
   legos serve --model /path/to/model

3. Run training:
   uv run examples/train_gsm8k.py

Requirements:
   pip install datasets
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, Artifact, GRPOCredit
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.gsm8k import (
    GSM8KArena,
    GSM8KEpisode,
    load_gsm8k,
    gsm8k_reward,
)
from legos.training import (
    Trainer,
    TrainerConfig,
    training_loop,
    synchronous_training_loop,
)


# =============================================================================
# CONFIGURATION - Edit these values directly
# =============================================================================

# Model
MODEL_PATH = "mlx_model"

# Inference server
INFERENCE_URL = "http://localhost:8000"

# Dataset
DATASET_SPLIT = "train"
MAX_SAMPLES = None  # None for all

# Generation (per arena.step())
SAMPLES_PER_QUESTION = 8  # GRPO: multiple rollouts per question
EPISODE_CONCURRENCY = 8  # Max concurrent episodes within an arena.step() call
STEP_CONCURRENCY = 4      # Max concurrent arena.step() calls

# Training
NUM_STEPS = 200
LR = 1e-5
MIN_SAMPLES_PER_STEP = 32  # Records needed before optimizer step
MICRO_BATCH_TOKENS = 1600  # Max tokens per micro-batch
STALENESS_LIMIT = 2        # Discard records older than N steps

# KL regularization
KL_COEF = 0.1
USE_KL_PENALTY = False

# Generation parameters
PROMPT_TEMPLATE = "{question}"
MAX_TOKENS = 512

# Training mode
USE_SIMPLE_LOOP = True  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


SYSTEM_PROMPT = """Solve the problem CONCISELY. Don't think too hard. Provide your final numerical answer wrapped in <answer></answer> tags.

Example format:
I'll calculate...
Therefore...
<answer>42</answer>"""


def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

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

    # Group by question
    questions = {}
    for r in batch.records:
        artifact = r.meta.get("artifact", {}) or {}
        q_key = artifact.get("question", r.rollout_id)[:50]
        if q_key not in questions:
            questions[q_key] = {"artifact": artifact, "records": []}
        questions[q_key]["records"].append(r)

    print(f"Questions: {len(questions)} | Responses: {len(batch.records)}\n")

    correct_count = 0
    total_count = len(batch.records)

    for i, (q_key, data) in enumerate(questions.items(), 1):
        artifact = data["artifact"]
        records = data["records"]

        question = artifact.get("question", "N/A")
        expected = str(artifact.get("answer", "N/A")).strip()

        print(f"[{i}] Q: {truncate(question, 80)}")
        print(f"    Expected: {expected}")

        for j, record in enumerate(records):
            response = record.completion_text or ""
            extracted = extract_xml_answer(response)

            extracted_norm = extracted.replace(",", "").replace("$", "").strip()
            expected_norm = expected.replace(",", "").replace("$", "").strip()
            is_correct = extracted_norm == expected_norm

            if is_correct:
                correct_count += 1

            status = "OK" if is_correct else "X"
            print(f"    [{status}] Got: {extracted_norm or '(none)'} | reward={record.reward:.2f} | {truncate(response, 60)}")

        print()

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")

    all_rewards = [r.reward for r in batch.records]
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"Avg reward: {avg_reward:.3f} | Range: {min(all_rewards):.3f} to {max(all_rewards):.3f}")


async def main(args):
    # Load dataset
    dataset = load_gsm8k(split=DATASET_SPLIT, max_samples=MAX_SAMPLES)

    # Setup inference client
    print(f"\nConnecting to inference server at {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=240.0,
    )

    # Setup arena
    arena = GSM8KArena(
        client=client,
        episodes_per_step=SAMPLES_PER_QUESTION,
        credit_assigner=GRPOCredit(normalize=True),
        verbose=args.verbose,
    )

    # Add trainable actor
    arena.add_actor(Actor(
        id="model",
        system_prompt=SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    # Add episode
    arena.add_episode("gsm8k", GSM8KEpisode(
        actor_id="model",
        reward_fn=gsm8k_reward,
        prompt_template=PROMPT_TEMPLATE,
    ))

    # Load dataset into store
    store = arena.add_store("gsm8k")
    for i, item in enumerate(dataset):
        store.add(Artifact(id=f"row_{i}", data=item))
    print(f"Loaded {store.count()} questions")

    # Dry-run mode
    if args.dry_run:
        await preview_gsm8k(arena, concurrency=EPISODE_CONCURRENCY)
        await client.close()
        return

    # Load model with LoRA
    model, tokenizer = load_model_with_lora(MODEL_PATH)

    # Setup trainer
    optimizer = optim.Adam(learning_rate=LR)
    config = TrainerConfig(
        lr=LR,
        micro_batch_tokens=MICRO_BATCH_TOKENS,
        staleness_limit=STALENESS_LIMIT,
        min_samples_per_step=MIN_SAMPLES_PER_STEP,
        ppo_clip_min=0.8,
        ppo_clip_max=1.2,
        kl_coef=KL_COEF,
        use_kl_penalty=USE_KL_PENALTY,
        inference_url=INFERENCE_URL,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
    )

    trainer = Trainer(model=model, optimizer=optimizer, config=config, client=client)

    # Run training
    print(f"\nStarting training for {NUM_STEPS} steps...")
    print(f"  - Dataset: GSM8K ({DATASET_SPLIT} split, {len(dataset)} samples)")
    print(f"  - Samples per question: {SAMPLES_PER_QUESTION} (GRPO)")
    print(f"  - Min samples per step: {MIN_SAMPLES_PER_STEP}")
    print()

    try:
        if USE_SIMPLE_LOOP:
            await synchronous_training_loop(
                arena=arena,
                trainer=trainer,
                num_steps=NUM_STEPS,
                episode_concurrency=EPISODE_CONCURRENCY,
                step_concurrency=STEP_CONCURRENCY,
                verbose=args.verbose,
            )
        else:
            batch_queue = asyncio.Queue(maxsize=4)
            await training_loop(
                arena=arena,
                trainer=trainer,
                batch_queue=batch_queue,
                num_steps=NUM_STEPS,
                episode_concurrency=EPISODE_CONCURRENCY,
                step_concurrency=STEP_CONCURRENCY,
                verbose=args.verbose,
            )
    finally:
        await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on GSM8K with GRPO")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    args = parser.parse_args()
    asyncio.run(main(args))
