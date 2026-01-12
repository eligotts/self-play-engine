"""
HeadToHead: Tournament competition with LLM-judged matches.

Trains a model through head-to-head competition on creative challenges.
The same model plays both sides, with an LLM judge picking winners.

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Start inference server:
   legos serve --model /path/to/model

3. Run training:
   uv run examples/train_head_to_head.py
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, Artifact, GRPOCredit
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.head_to_head import HeadToHeadArena, MatchEpisode, ChallengeProposerEpisode
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

# Generation (per arena.step())
EPISODES_PER_STEP = 4           # Matches per arena.step()
PROPOSER_EPISODES_PER_STEP = 4  # New challenges generated per arena.step()
EPISODE_CONCURRENCY = 4         # Max concurrent episodes
STEP_CONCURRENCY = 1            # Max concurrent arena.step() calls

# Training
NUM_STEPS = 50
LR = 1e-5
MIN_SAMPLES_PER_STEP = 32   # Records needed before optimizer step
MICRO_BATCH_TOKENS = 2048  # Max tokens per micro-batch
STALENESS_LIMIT = 2        # Discard records older than N steps

# KL regularization
KL_COEF = 0.1
USE_KL_PENALTY = False

# Generation parameters
MAX_TOKENS = 512

# Training mode
USE_SIMPLE_LOOP = True  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


# Creative challenges for head-to-head competition
CHALLENGES = [
    {"challenge": "Write a creative opening line for a mystery novel."},
    {"challenge": "Explain quantum entanglement using a cooking analogy."},
    {"challenge": "Write a haiku about the feeling of debugging code at 3am."},
    {"challenge": "Describe a color to someone who has never seen it."},
    {"challenge": "Write a one-paragraph horror story set in an office."},
    {"challenge": "Create a memorable slogan for a time travel agency."},
    {"challenge": "Explain why the sky is blue to a curious 5-year-old."},
    {"challenge": "Write a fortune cookie message that's both funny and profound."},
    {"challenge": "Describe the taste of coffee to an alien."},
    {"challenge": "Write an uplifting message for someone having a bad day."},
    {"challenge": "Create a metaphor for procrastination."},
    {"challenge": "Write a persuasive argument for why Mondays are actually great."},
]


COMPETITOR0_SYSTEM_PROMPT = """You are a competitor in a writing challenge.
Give your best response to each challenge.
You are logical and analytical at heart, so your output should always prioritize logical and analytical writing.
Quality matters more than length."""

COMPETITOR1_SYSTEM_PROMPT = """You are a competitor in a writing challenge.
Give your best response to each challenge.
You are a creative at heart, so your output should always prioritize creative writing.
Quality matters more than length."""

CHALLENGE_PROPOSER_SYSTEM_PROMPT = """You are a creative challenge designer.
Generate novel, engaging writing challenges that test creativity and skill.
Make challenges specific, clear, and answerable in 1-3 sentences."""


def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_head_to_head(arena, concurrency: int = 4):
    """Preview HeadToHead output."""
    print("\n=== HeadToHead Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    matches = {}
    for r in batch.records:
        if r.rollout_id not in matches:
            matches[r.rollout_id] = {"meta": r.meta, "records": []}
        matches[r.rollout_id]["records"].append(r)

    print(f"Matches: {len(matches)} | Records: {len(batch.records)}\n")

    for i, (_, data) in enumerate(matches.items(), 1):
        meta = data["meta"]
        episode_type = meta.get("episode_type", "?")

        if episode_type == "match":
            challenge = meta.get("artifact", {}).get("challenge", "N/A") if meta.get("artifact") else "N/A"
            actor_rewards = {r.actor_id: r.reward for r in data["records"]}
            actor_advantages = {r.actor_id: r.advantage for r in data["records"]}

            if actor_rewards:
                max_reward = max(actor_rewards.values())
                if max_reward > 0:
                    winner = [k for k, v in actor_rewards.items() if v == max_reward][0]
                elif max_reward == 0:
                    winner = "Tie"
                else:
                    winner = "?"
            else:
                winner = "?"

            print(f"[{i}] Match | Winner: {winner}")
            print(f"    Challenge: {truncate(challenge, 60)}")

            for actor_id, reward in sorted(actor_rewards.items()):
                adv = actor_advantages.get(actor_id, 0)
                print(f"    {actor_id}: reward={reward:+.1f}, advantage={adv:+.3f}")

        elif episode_type == "challenge_propose":
            new_challenge = meta.get("extras", {}).get("proposed_challenge", {})
            challenge_text = new_challenge.get("challenge", "N/A") if new_challenge else "N/A"
            reward = data["records"][0].reward if data["records"] else 0
            print(f"[{i}] ChallengePropose | reward={reward:.2f}")
            print(f"    New challenge: {truncate(challenge_text, 100)}")

        print()


async def main(args):
    print(f"\nConnecting to {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=120.0,
    )

    # Setup arena
    arena = HeadToHeadArena(
        client=client,
        episodes_per_step=EPISODES_PER_STEP,
        proposer_episodes_per_step=PROPOSER_EPISODES_PER_STEP,
        credit_assigner=GRPOCredit(),
        verbose=args.verbose,
    )

    # Both players use same system prompt (same model, different "virtual" players)
    arena.add_actor(Actor(
        id="Player0",
        system_prompt=COMPETITOR0_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_actor(Actor(
        id="Player1",
        system_prompt=COMPETITOR1_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_episode("match", MatchEpisode(
        player0_actor="Player0",
        player1_actor="Player1",
    ))

    # Challenge proposer (non-trainable) generates new challenges
    arena.add_actor(Actor(
        id="ChallengeProposer",
        system_prompt=CHALLENGE_PROPOSER_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_episode("challenge_propose", ChallengeProposerEpisode(
        proposer_actor="ChallengeProposer",
    ))

    # Load challenges
    store = arena.add_store("challenges")
    for i, challenge in enumerate(CHALLENGES):
        store.add(Artifact(id=f"challenge_{i}", data=challenge))
    print(f"Loaded {store.count()} challenges")

    # Dry-run mode
    if args.dry_run:
        await preview_head_to_head(arena, concurrency=EPISODE_CONCURRENCY)
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

    print(f"\nStarting HeadToHead training for {NUM_STEPS} steps...")
    print(f"  - Head-to-head creative competition")
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
    parser = argparse.ArgumentParser(description="HeadToHead: Tournament competition with LLM judge")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    asyncio.run(main(parser.parse_args()))
