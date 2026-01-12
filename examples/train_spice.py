"""
SPICE: Self-Play In Corpus Environment.

Trains proposer and solver agents using corpus-grounded question generation.
Inspired by the SPICE paper (arXiv:2510.24684).

Key concepts:
- Proposer reads documents and generates questions
- Solver answers questions without access to source documents
- Proposer is rewarded for questions at the frontier of solver capability (~50% pass rate)
- Solver is rewarded via LLM-as-judge for correctness
- Example of how to use llm as a judge for a solver episode
- Example of how solver episodes can be used as training examples (compare to proposer_solver.py which does NOT train on solver episodes)

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Set your OpenRouter API key (for the LLM judge):
   export OPENROUTER_API_KEY=your-key-here

3. Start inference server:
   legos serve --model /path/to/model

4. Run training:
   uv run examples/train_spice.py
"""
import asyncio
import argparse
import json
from pathlib import Path

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, Artifact
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.spice import SpiceArena, SpiceProposerEpisode, SpiceSolverEpisode
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

# Corpus
CORPUS_PATH = Path(__file__).parent.parent / "sample_data" / "spice_corpus.json"

# Task settings
N_SOLVER_ROLLOUTS = 4     # Solver attempts per proposed question
TARGET_PASS_RATE = 0.5    # Target solver pass rate for proposer reward

# Generation (per arena.step())
EPISODES_PER_STEP = 4     # Proposer episodes per arena.step()
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

# Training mode
USE_SIMPLE_LOOP = False  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

PROPOSER_SYSTEM = """You are a question generator. Given a document, create challenging but answerable questions.
Your questions should:
- Test comprehension and reasoning
- Have clear, unambiguous answers
- Be diverse in style (factual, inferential, analytical)

Always respond with valid JSON only."""

SOLVER_SYSTEM = """You are a knowledgeable question answerer. Answer questions accurately and concisely.
Think step by step when needed, then provide a clear final answer.
Always end your response with: "The answer is: " followed by your answer."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def load_corpus_from_file(filepath: str) -> list:
    """Load corpus from a JSON or JSONL file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    with open(path, "r") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

async def preview_spice(arena, concurrency: int = 4):
    """Preview SPICE arena output with task-specific metrics."""
    print("\n=== SPICE Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    # Group by rollout
    episodes = {}
    for r in batch.records:
        if r.rollout_id not in episodes:
            episodes[r.rollout_id] = {"meta": r.meta, "records": []}
        episodes[r.rollout_id]["records"].append(r)

    print(f"Episodes: {len(episodes)} | Records: {len(batch.records)}\n")

    # Track stats by episode type
    propose_stats = {"rewards": [], "pass_rates": []}
    solve_stats = {"correct": 0, "total": 0, "rewards": []}

    for i, (_, data) in enumerate(episodes.items(), 1):
        meta = data["meta"]
        episode_type = meta.get("episode_type", "?")
        extras = meta.get("extras", {})
        artifact = meta.get("artifact", {}) or {}

        if episode_type == "spice_propose":
            # Proposer episode
            proposed = extras.get("proposed_question", {})
            question = proposed.get("question", "N/A") if proposed else "N/A"
            ground_truth = proposed.get("ground_truth", "N/A") if proposed else "N/A"
            pass_rate = extras.get("pass_rate", 0)
            solver_rewards = extras.get("solver_rewards", [])
            reward = data["records"][0].reward if data["records"] else 0

            propose_stats["rewards"].append(reward)
            propose_stats["pass_rates"].append(pass_rate)

            # Get source document info
            source_doc = extras.get("source_document", {})
            doc_title = source_doc.get("title", "Untitled") if isinstance(source_doc, dict) else "N/A"

            print(f"[{i}] PROPOSE | reward={reward:.2f} | pass_rate={pass_rate:.1%}")
            print(f"    Source: {doc_title}")
            print(f"    Q: {truncate(question, 80)}")
            print(f"    A: {ground_truth}")
            if solver_rewards:
                correct = sum(1 for r in solver_rewards if r > 0.5)
                print(f"    Solver results: {correct}/{len(solver_rewards)} correct")

        elif episode_type == "spice_solve":
            # Solver episode
            question = artifact.get("question", "N/A")
            ground_truth = artifact.get("ground_truth", "N/A")
            reward = data["records"][0].reward if data["records"] else 0
            is_correct = reward > 0.5

            solve_stats["total"] += 1
            if is_correct:
                solve_stats["correct"] += 1
            solve_stats["rewards"].append(reward)

            status = "CORRECT" if is_correct else "WRONG"
            print(f"[{i}] SOLVE | {status} | reward={reward:.2f}")
            print(f"    Q: {truncate(question, 80)}")
            print(f"    Expected: {ground_truth}")

        print()

    # Summary stats
    print("=" * 50)
    print("Summary:")
    if propose_stats["rewards"]:
        avg_reward = sum(propose_stats["rewards"]) / len(propose_stats["rewards"])
        avg_pass_rate = sum(propose_stats["pass_rates"]) / len(propose_stats["pass_rates"])
        print(f"  Proposer: {len(propose_stats['rewards'])} episodes, "
              f"avg_reward={avg_reward:.3f}, avg_pass_rate={avg_pass_rate:.1%}")

    if solve_stats["total"] > 0:
        accuracy = solve_stats["correct"] / solve_stats["total"]
        avg_reward = sum(solve_stats["rewards"]) / len(solve_stats["rewards"])
        print(f"  Solver: {solve_stats['correct']}/{solve_stats['total']} correct ({accuracy:.1%}), "
              f"avg_reward={avg_reward:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    # Load corpus
    print(f"Loading corpus from {CORPUS_PATH}...")
    corpus = load_corpus_from_file(str(CORPUS_PATH))
    print(f"Corpus size: {len(corpus)} documents\n")

    # Setup client
    print(f"Connecting to inference server at {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=120.0,
    )

    # Setup arena
    arena = SpiceArena(client=client, batch_size=EPISODES_PER_STEP, verbose=args.verbose)

    # Add actors
    arena.add_actor(Actor(
        id="Proposer",
        system_prompt=PROPOSER_SYSTEM,
        max_tokens=512,
    ))

    arena.add_actor(Actor(
        id="Solver",
        system_prompt=SOLVER_SYSTEM,
        max_tokens=1024,
    ))

    # Add episodes
    arena.add_episode("spice_propose", SpiceProposerEpisode(
        n_solver_rollouts=N_SOLVER_ROLLOUTS,
        target_pass_rate=TARGET_PASS_RATE,
    ))
    arena.add_episode("spice_solve", SpiceSolverEpisode())

    # Add stores
    corpus_store = arena.add_store("corpus")
    questions_store = arena.add_store("questions")

    # Populate corpus
    for doc in corpus:
        doc_id = doc.get("id", f"doc_{corpus_store.count()}")
        corpus_store.add(Artifact(id=doc_id, data=doc))

    print(f"Loaded {corpus_store.count()} documents into corpus store")

    # Dry-run mode
    if args.dry_run:
        await preview_spice(arena, concurrency=EPISODE_CONCURRENCY)
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
        inference_url=INFERENCE_URL,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
    )

    trainer = Trainer(model=model, optimizer=optimizer, config=config, client=client)

    print(f"\nStarting SPICE training for {NUM_STEPS} steps...")
    print(f"  - Corpus size: {corpus_store.count()} documents")
    print(f"  - Episodes per step: {EPISODES_PER_STEP}")
    print(f"  - Solver rollouts per proposal: {N_SOLVER_ROLLOUTS}")
    print(f"  - Target pass rate: {TARGET_PASS_RATE:.0%}")
    print(f"  - Min samples per step: {MIN_SAMPLES_PER_STEP}")

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

    # Cleanup
    await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")
    print(f"Questions generated: {questions_store.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPICE: Self-Play In Corpus Environment")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    asyncio.run(main(parser.parse_args()))
