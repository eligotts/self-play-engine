"""
Example: EloArena - Tournament competition with persistent Elo ratings.

This script trains a model through head-to-head competition on creative challenges.
The same model plays both sides, with an LLM judge picking winners.
Elo ratings persist and update throughout training.

To run:
1. Start mlx-vllm server:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Run this script:
   python examples/train_elo_arena.py --verbose
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient, Role, Artifact, RAECredit
from self_play.tasks.elo_arena import EloArena, EloMatchEpisode, ChallengeProposerEpisode
from self_play.training import (
    Trainer,
    TrainerConfig,
    WeightPublisher,
    simple_training_loop,
)


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


COMPETITOR_SYSTEM_PROMPT = """You are a creative competitor in a writing challenge.
Give your best, most creative response to each challenge.
Be original, engaging, and thoughtful.
Quality matters more than length."""

CHALLENGE_PROPOSER_SYSTEM_PROMPT = """You are a creative challenge designer.
Generate novel, engaging writing challenges that test creativity and skill.
Make challenges specific, clear, and answerable in 1-3 sentences."""


def load_model_with_lora(model_path: str, lora_rank: int = 8, lora_layers: int = 16):
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    print(f"Attaching LoRA (rank={lora_rank}, layers={lora_layers})...")
    linear_to_lora_layers(model, lora_layers, {"rank": lora_rank, "scale": 20.0, "dropout": 0.0})

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_elo_arena(arena, concurrency: int = 4):
    """Preview EloArena output with Elo-specific metrics."""
    print("\n=== EloArena Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    # Group by rollout for match-level view
    matches = {}
    for r in batch.records:
        if r.rollout_id not in matches:
            matches[r.rollout_id] = {"meta": r.meta, "records": []}
        matches[r.rollout_id]["records"].append(r)

    print(f"Matches: {len(matches)} | Records: {len(batch.records)}\n")

    # Display each match
    for i, (rid, data) in enumerate(matches.items(), 1):
        meta = data["meta"]
        episode_type = meta.get("episode_type", "?")

        if episode_type == "elo_match":
            winner = meta.get("winner", "?")
            challenge = meta.get("artifact", {}).get("challenge", "N/A") if meta.get("artifact") else "N/A"
            elo_changes = meta.get("elo_changes", {})

            print(f"[{i}] EloMatch | Winner: {winner}")
            print(f"    Challenge: {truncate(challenge, 60)}")

            # Show Elo changes
            if elo_changes:
                changes_str = ", ".join(f"{p}: {c:+.0f}" for p, c in elo_changes.items())
                print(f"    Elo changes: {changes_str}")

            # Show responses
            extras = meta.get("extras", {})
            responses = extras.get("responses", {})
            for player, response in responses.items():
                print(f"    {player}: {truncate(response, 80)}")

        elif episode_type == "challenge_propose":
            new_challenge = meta.get("extras", {}).get("challenge", "N/A")
            reward = data["records"][0].reward if data["records"] else 0
            print(f"[{i}] ChallengePropose | reward={reward:.2f}")
            print(f"    New challenge: {truncate(new_challenge, 100)}")

        print()

    # Current Elo ratings
    print("Current Elo Ratings:")
    for player, elo in sorted(arena.elo_ratings.items()):
        print(f"  {player}: {elo:.0f}")

    # Match history summary
    if arena.match_history:
        wins = {}
        for match in arena.match_history:
            winner = match.get("winner")
            if winner:
                wins[winner] = wins.get(winner, 0) + 1
        print(f"\nMatch history ({len(arena.match_history)} matches):")
        for player, win_count in sorted(wins.items()):
            print(f"  {player}: {win_count} wins")


async def main(args):
    base_url = args.url if args.url else f"http://{args.host}:{args.port}"
    print(f"\nConnecting to {base_url}...")

    if args.url:
        client = OpenAIClient(api_key="not-needed", model="local",
                              base_url=f"{base_url.rstrip('/')}/v1", timeout=120.0)
    else:
        client = OpenAIClient.for_local(host=args.host, port=args.port, timeout=120.0)

    # Setup arena with Elo tracking
    arena = EloArena(
        client=client,
        batch_size=args.batch_size,
        proposer_batch_size=args.proposer_batch_size,
        initial_elo=1500.0,
        k_factor=32.0,
        credit_assigner=RAECredit(decay=args.rae_decay),
        verbose=args.verbose,
    )

    # Both players use same system prompt (same model, different "virtual" players)
    arena.add_role(Role(
        id="Player0",
        system_prompt=COMPETITOR_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    arena.add_role(Role(
        id="Player1",
        system_prompt=COMPETITOR_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    arena.add_episode("elo_match", EloMatchEpisode(
        player0_role="Player0",
        player1_role="Player1",
    ))

    # Challenge proposer (non-trainable) generates new challenges
    arena.add_role(Role(
        id="ChallengeProposer",
        system_prompt=CHALLENGE_PROPOSER_SYSTEM_PROMPT,
        temperature=0.9,  # Higher temp for more creative challenges
        max_tokens=args.max_tokens,
    ))

    arena.add_episode("challenge_propose", ChallengeProposerEpisode(
        proposer_role="ChallengeProposer",
    ))

    # Load challenges
    store = arena.add_store("challenges")
    for i, challenge in enumerate(CHALLENGES):
        store.add(Artifact(id=f"challenge_{i}", data=challenge))
    print(f"Loaded {store.count()} challenges")

    # Dry-run mode: preview arena performance without training
    if args.dry_run:
        await preview_elo_arena(arena, concurrency=args.concurrency)
        await client.close()
        return

    # Load model
    model, tokenizer = load_model_with_lora(args.model_path, args.lora_rank, args.lora_layers)

    # Setup training
    optimizer = optim.Adam(learning_rate=args.lr)
    config = TrainerConfig(
        lr=args.lr,
        micro_token_budget=args.micro_token_budget,
        max_policy_lag=args.max_policy_lag,
        batch_size=args.train_batch_size,
        clip_low=0.8, clip_high=1.2,
        kl_coef=args.kl_coef,
        weight_push_url=base_url,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=args.wandb_project,
    )

    publisher = WeightPublisher(base_url=base_url)
    trainer = Trainer(model=model, optimizer=optimizer, config=config, publisher=publisher)

    print(f"\nStarting EloArena training for {args.num_steps} steps...")
    print(f"  - Head-to-head creative competition")
    print(f"  - LLM judge picks winners")
    print(f"  - Elo ratings tracked across training")
    print(f"  - Dynamic challenge generation ({args.proposer_batch_size} per batch)")

    try:
        await simple_training_loop(
            arena=arena,
            trainer=trainer,
            num_steps=args.num_steps,
            concurrency=args.concurrency,
            verbose=args.verbose,
        )
    finally:
        await publisher.close()
        await client.close()

    # Print final Elo ratings
    print(f"\nTraining complete! Step: {trainer.train_step_idx}")
    print(f"\nFinal Elo ratings:")
    for player, elo in sorted(arena.elo_ratings.items()):
        print(f"  {player}: {elo:.0f}")
    print(f"\nTotal matches: {len(arena.match_history)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EloArena: Tournament competition with Elo ratings")
    parser.add_argument("--model-path", default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--proposer-batch-size", type=int, default=1, help="Number of new challenges to generate per batch")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--micro-token-budget", type=int, default=2048)
    parser.add_argument("--max-policy-lag", type=int, default=3)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--rae-decay", type=float, default=0.9)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    asyncio.run(main(parser.parse_args()))
