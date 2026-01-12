"""
RefineLoop: Generator-Critic iterative refinement.

Trains a Generator and Critic to collaboratively improve content.
The Generator creates drafts, the Critic provides feedback, and this iterates.
Both actors share the same reward.

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Start inference server:
   legos serve --model /path/to/model

3. Run training:
   uv run examples/train_refine_loop.py
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, Artifact, GRPOCredit
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.refine_loop import RefineLoopArena, RefineLoopEpisode, TaskProposerEpisode
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

# Task settings
NUM_ITERATIONS = 2  # Refinement iterations per episode

# Generation (per arena.step())
EPISODES_PER_STEP = 4           # Tasks per arena.step()
PROPOSER_EPISODES_PER_STEP = 4  # New tasks generated per arena.step()
EPISODE_CONCURRENCY = 8         # Max concurrent episodes
STEP_CONCURRENCY = 1            # Max concurrent arena.step() calls

# Training
NUM_STEPS = 200
LR = 1e-5
MIN_SAMPLES_PER_STEP = 32  # Records needed before optimizer step
MICRO_BATCH_TOKENS = 2048  # Max tokens per micro-batch
STALENESS_LIMIT = 2        # Discard records older than N steps

# KL regularization
KL_COEF = 0.1
USE_KL_PENALTY = False

# Loss configuration
LOSS_TYPE = "token"  # "token" (DAPO) or "sample" (GRPO)
IMPORTANCE_SAMPLING = "token"  # "token" or "sequence" (GSPO)

# Generation parameters
MAX_TOKENS = 512

# Training mode
USE_SIMPLE_LOOP = False  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


# Creative writing tasks for refinement
TASKS = [
    {
        "task": "Write a poem about programming",
        "requirements": "Should capture the essence of coding.",
    },
    {
        "task": "Write a product description for a smart water bottle",
        "requirements": "150 words max. Highlight 3 key features. Use persuasive language.",
    },
    {
        "task": "Write an opening paragraph for a mystery novel",
        "requirements": "Set the scene, introduce tension, hook the reader. 100 words.",
    },
    {
        "task": "Write a professional email declining a meeting",
        "requirements": "Polite but firm. Suggest an alternative. Keep it brief.",
    },
    {
        "task": "Write a tweet announcing a new feature",
        "requirements": "Max 280 characters. Include call to action. Be engaging.",
    },
    {
        "task": "Write a short bio for a software engineer",
        "requirements": "Third person. 50 words. Professional but personable.",
    },
    {
        "task": "Write an apology for a service outage",
        "requirements": "Acknowledge the issue, explain briefly, commit to improvement.",
    },
]


GENERATOR_SYSTEM_PROMPT = """You are a skilled writer. Create high-quality content based on the given task.
When receiving feedback, thoughtfully revise your work to address the suggestions.
Focus on clarity, creativity, and meeting all requirements.
After you receive feedback, do not directly respond to it. Instead, output your revised draft based on the feedback.
It is CRUCIAL that you ONLY output your revised draft based on the feedback. Do not include any other commentary."""

CRITIC_SYSTEM_PROMPT = """You are a constructive critic. Provide specific, concise feedback.
Focus on: clarity, meeting requirements, style, and impact.
ONLY PROVIDE CONCISE FEEDBACK. This feedback MUST NOT include a suggestion for what a revised draft should look like.

For example:
If the first draft is overly verbose, you might say:
"Your writing is too verbose. Please shorten your response."
But DO NOT then suggest a revised draft.
"""

TASK_PROPOSER_SYSTEM_PROMPT = """You are a creative writing task designer.
Generate novel, well-structured writing tasks with clear requirements.
Tasks should be challenging but achievable with iterative refinement.
Don't ask for something incredibly complex like a limerick or a haiku, or something similar that requires a very specific format."""


def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_refine_loop(arena, concurrency: int = 4):
    """Preview arena output with truncated drafts/critiques."""
    print("\n=== RefineLoop Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    rollouts = {}
    for r in batch.records:
        if r.rollout_id not in rollouts:
            rollouts[r.rollout_id] = {"meta": r.meta, "reward": r.reward}

    print(f"Episodes: {len(rollouts)} | Records: {len(batch.records)}\n")

    for i, (_, data) in enumerate(rollouts.items(), 1):
        meta = data["meta"]
        extras = meta.get("extras", {})
        artifact = meta.get("artifact", {}) or {}

        print(f"[{i}] {meta.get('episode_type', '?')} | reward={data['reward']:.2f}")
        print(f"    task: {truncate(artifact.get('task', 'N/A'), 60)}")

        drafts = extras.get("drafts", [])
        feedback = extras.get("feedback", [])
        for j, draft in enumerate(drafts):
            print(f"    draft{j+1}: {truncate(draft, 10000)}")
        for j, fb in enumerate(feedback):
            print(f"    critique{j+1}: {truncate(fb, 10000)}")
        print()

    qualities = [d["reward"] for d in rollouts.values()]
    if qualities:
        print(f"Avg quality: {sum(qualities)/len(qualities):.2f} | Range: {min(qualities):.2f}-{max(qualities):.2f}")


async def main(args):
    print(f"\nConnecting to {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=120.0,
    )

    # Setup arena
    arena = RefineLoopArena(
        client=client,
        episodes_per_step=EPISODES_PER_STEP,
        proposer_episodes_per_step=PROPOSER_EPISODES_PER_STEP,
        credit_assigner=GRPOCredit(),
        verbose=args.verbose,
    )

    arena.add_actor(Actor(
        id="Generator",
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_actor(Actor(
        id="Critic",
        system_prompt=CRITIC_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_episode("refine_loop", RefineLoopEpisode(
        generator_actor="Generator",
        critic_actor="Critic",
        num_iterations=NUM_ITERATIONS,
    ))

    # Task proposer (non-trainable) generates new tasks
    arena.add_actor(Actor(
        id="TaskProposer",
        system_prompt=TASK_PROPOSER_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_episode("task_propose", TaskProposerEpisode(
        proposer_actor="TaskProposer",
    ))

    # Load tasks
    store = arena.add_store("tasks")
    for i, task in enumerate(TASKS):
        store.add(Artifact(id=f"task_{i}", data=task))
    print(f"Loaded {store.count()} tasks")

    # Dry-run mode
    if args.dry_run:
        await preview_refine_loop(arena, concurrency=EPISODE_CONCURRENCY)
        await client.close()
        return

    # Load model with LoRA
    model, tokenizer = load_model_with_lora(MODEL_PATH)

    # Setup training
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
        loss_type=LOSS_TYPE,
        importance_sampling=IMPORTANCE_SAMPLING,
        inference_url=INFERENCE_URL,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
    )

    trainer = Trainer(model=model, optimizer=optimizer, config=config, client=client)

    print(f"\nStarting RefineLoop training for {NUM_STEPS} steps...")
    print(f"  - {NUM_ITERATIONS} refinement iterations per episode")
    print(f"  - Episodes per step: {EPISODES_PER_STEP}")
    print(f"  - Min samples per step: {MIN_SAMPLES_PER_STEP}")

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

    print(f"\nTraining complete! Step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefineLoop: Generator-Critic iterative refinement")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    asyncio.run(main(parser.parse_args()))
