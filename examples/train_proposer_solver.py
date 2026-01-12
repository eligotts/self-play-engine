"""
ProposerSolver: Train proposer/solver agents with RL.

The proposer learns to generate questions at an optimal difficulty level,
while the solver learns to answer them correctly.
More of a toy example just to show how this proposer/solver paradigm works, we
obviously don't want the proposer model to just generate the question AND the answer,
would rather an external validator like deterministic execution of code to get the 'answer'

Inspired by general structure from Absolute Zero paper (arXiv:2505.03335).

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Start inference server:
   legos serve --model /path/to/model

3. Run training:
   uv run examples/train_proposer_solver.py
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, Artifact
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.proposer_solver import ProposerSolverArena, ProposerEpisode, SolveEpisode
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
N_SOLVER_ROLLOUTS = 4  # Monte Carlo solver attempts per proposed question 

# Generation (per arena.step())
EPISODES_PER_STEP = 4     # Episodes per arena.step()
EPISODE_CONCURRENCY = 4   # Max concurrent episodes
STEP_CONCURRENCY = 1      # Max concurrent arena.step() calls

# Training
NUM_STEPS = 100
LR = 1e-5
MIN_SAMPLES_PER_STEP = 32  # Records needed before optimizer step
MICRO_BATCH_TOKENS = 2048  # Max tokens per micro-batch
STALENESS_LIMIT = 2        # Discard records older than N steps

# KL regularization
KL_COEF = 0.1
USE_KL_PENALTY = False

# Generation parameters
MAX_TOKENS = 512

# Training mode
USE_SIMPLE_LOOP = False  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


PROPOSER_SYSTEM_PROMPT = """You are a creative math problem creator.
Generate interesting, well-formed problems with clear answers.
Ensure that in your question you explicitly state the form the answer should be provided in."""

SOLVER_SYSTEM_PROMPT = """You are a skilled math problem solver.
Think step by step and provide clear, correct answers.
Ensure that your answer is provided in the form specified in the question."""


# Seed questions for the proposer to learn from
INITIAL_QUESTIONS = [
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


def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_proposer_solver(arena, concurrency: int = 4):
    """Preview ProposerSolver arena output with task-specific metrics."""
    print("\n=== ProposerSolver Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    episodes = {}
    for r in batch.records:
        if r.rollout_id not in episodes:
            episodes[r.rollout_id] = {"meta": r.meta, "records": []}
        episodes[r.rollout_id]["records"].append(r)

    print(f"Episodes: {len(episodes)} | Records: {len(batch.records)}\n")

    propose_stats = {"rewards": [], "solve_rates": []}
    solve_stats = {"correct": 0, "total": 0, "rewards": []}

    for i, (_, data) in enumerate(episodes.items(), 1):
        meta = data["meta"]
        episode_type = meta.get("episode_type", "?")
        extras = meta.get("extras", {})
        artifact = meta.get("artifact", {}) or {}

        if episode_type == "propose":
            question = extras.get("question", artifact.get("question", "N/A"))
            ground_truth = extras.get("ground_truth", artifact.get("ground_truth", "N/A"))
            solve_rate = extras.get("solve_rate", 0)
            reward = data["records"][0].reward if data["records"] else 0

            propose_stats["rewards"].append(reward)
            propose_stats["solve_rates"].append(solve_rate)

            print(f"[{i}] Propose | reward={reward:.2f} | solve_rate={solve_rate:.1%}")
            print(f"    Q: {truncate(question, 80)}")
            print(f"    Answer: {ground_truth}")

            solver_results = extras.get("solver_results", [])
            if solver_results:
                correct = sum(1 for r in solver_results if r.get("correct"))
                print(f"    Solver attempts: {correct}/{len(solver_results)} correct")

        elif episode_type == "solve":
            question = artifact.get("question", "N/A")
            ground_truth = artifact.get("ground_truth", "N/A")
            response = extras.get("response", "N/A")
            extracted = extras.get("extracted_answer", "N/A")
            reward = data["records"][0].reward if data["records"] else 0
            # Derive correctness from reward (exact match rubric gives 1.0 for correct)
            is_correct = reward > 0.5

            solve_stats["total"] += 1
            if is_correct:
                solve_stats["correct"] += 1
            solve_stats["rewards"].append(reward)

            status = "CORRECT" if is_correct else "WRONG"
            print(f"[{i}] Solve | {status} | reward={reward:.2f}")
            print(f"    Q: {truncate(question, 80)}")
            print(f"    Expected: {ground_truth} | Got: {extracted}")
            print(f"    Response: {truncate(response, 100)}")

        print()

    print("Summary:")
    if propose_stats["rewards"]:
        avg_reward = sum(propose_stats["rewards"]) / len(propose_stats["rewards"])
        avg_solve_rate = sum(propose_stats["solve_rates"]) / len(propose_stats["solve_rates"])
        print(f"  Proposer: {len(propose_stats['rewards'])} episodes, avg_reward={avg_reward:.3f}, avg_solve_rate={avg_solve_rate:.1%}")

    if solve_stats["total"] > 0:
        accuracy = solve_stats["correct"] / solve_stats["total"]
        avg_reward = sum(solve_stats["rewards"]) / len(solve_stats["rewards"])
        print(f"  Solver: {solve_stats['correct']}/{solve_stats['total']} correct ({accuracy:.1%}), avg_reward={avg_reward:.3f}")


async def main(args):
    print(f"\nConnecting to inference server at {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=120.0,
    )

    # Setup arena
    arena = ProposerSolverArena(
        client=client,
        episodes_per_step=EPISODES_PER_STEP,
        verbose=args.verbose,
    )

    # Add actors
    arena.add_actor(Actor(
        id="Proposer",
        system_prompt=PROPOSER_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_actor(Actor(
        id="Solver",
        system_prompt=SOLVER_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    # Add episodes
    arena.add_episode("propose", ProposerEpisode(n_solver_rollouts=N_SOLVER_ROLLOUTS))
    arena.add_episode("solve", SolveEpisode())

    # Add initial questions to store
    question_store = arena.add_store("questions")
    for i, q in enumerate(INITIAL_QUESTIONS):
        question_store.add(Artifact(id=f"seed_{i}", data=q))

    # Dry-run mode
    if args.dry_run:
        await preview_proposer_solver(arena, concurrency=EPISODE_CONCURRENCY)
        await client.close()
        return

    # Warmup: ensure we have enough questions before training
    print("Running arena warmup...")
    await arena.on_train_start()

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

    print(f"\nStarting training for {NUM_STEPS} steps...")
    print(f"  - Episodes per step: {EPISODES_PER_STEP}")
    print(f"  - Min samples per step: {MIN_SAMPLES_PER_STEP}")
    print(f"  - Solver rollouts per proposal: {N_SOLVER_ROLLOUTS}")
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
        await arena.on_train_end()
        await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train proposer/solver agents with RL")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    args = parser.parse_args()
    asyncio.run(main(args))
