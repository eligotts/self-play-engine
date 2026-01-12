"""
Negotiation: Train negotiation agents with RL (SPIRAL SimpleNegotiation arXiv:2506.24119).

The game:
- Two players with opposite resource preferences
- Player 0 values Gold more (Wood=5, Gold=15)
- Player 1 values Wood more (Wood=15, Gold=5)
- Both start with 10 Wood + 10 Gold
- Players negotiate and trade to maximize their inventory value
- Winner = player with larger inventory value change

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Start inference server:
   legos serve --model /path/to/model

3. Run training:
   uv run examples/train_negotiation.py
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, RAECredit
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.negotiation import NegotiationArena, NegotiationEpisode
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

# Game settings
MAX_GAME_TURNS = 6

# Generation (per arena.step())
EPISODES_PER_STEP = 4     # Games per arena.step()
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

# RAE credit assignment
RAE_DECAY = 0.95

# Generation parameters
MAX_TOKENS = 512

# Training mode
USE_SIMPLE_LOOP = False  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


# System prompts for each player
PLAYER_0_SYSTEM_PROMPT = """You are Player 0 in a negotiation game. Your goal is to maximize your inventory value through trades.

YOUR VALUES:
- Wood = 5 points each
- Gold = 15 points each
You prefer Gold! Try to trade away Wood to get more Gold.

STARTING RESOURCES: 10 Wood, 10 Gold

ACTIONS (you MUST use one of these exact formats):
- [Offer: I give X Wood, Y Gold for Z Wood, W Gold]
- [Accept]
- [Deny]

EXAMPLES:
- To offer 3 wood for 2 gold: [Offer: I give 3 Wood, 0 Gold for 0 Wood, 2 Gold]
- To accept an offer: [Accept]
- To reject an offer: [Deny]

WARNING: Don't ask for more than opponent has or accept trades you can't fulfill. Invalid trades are penalized.

IMPORTANT:
- Every response MUST end with an action in brackets.
- Only output YOUR reasoning and YOUR action. Do NOT mimic transcript format or generate responses for other players.
- Your output must end with exactly one action: [Offer: ...], [Accept], or [Deny]."""

PLAYER_1_SYSTEM_PROMPT = """You are Player 1 in a negotiation game. Your goal is to maximize your inventory value through trades.

YOUR VALUES:
- Wood = 15 points each
- Gold = 5 points each
You prefer Wood! Try to trade away Gold to get more Wood.

STARTING RESOURCES: 10 Wood, 10 Gold

ACTIONS (you MUST use one of these exact formats):
- [Offer: I give X Wood, Y Gold for Z Wood, W Gold]
- [Accept]
- [Deny]

EXAMPLES:
- To offer 2 gold for 3 wood: [Offer: I give 0 Wood, 2 Gold for 3 Wood, 0 Gold]
- To accept an offer: [Accept]
- To reject an offer: [Deny]

WARNING: Don't ask for more than opponent has or accept trades you can't fulfill. Invalid trades are penalized.

IMPORTANT:
- Every response MUST end with an action in brackets.
- Only output YOUR reasoning and YOUR action. Do NOT mimic transcript format or generate responses for other players.
- Your output must end with exactly one action: [Offer: ...], [Accept], or [Deny]."""


def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_negotiation(arena, concurrency: int = 4):
    """Preview Negotiation arena output with game-specific metrics."""
    print("\n=== Negotiation Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    # Group by rollout for game-level view
    games = {}
    for r in batch.records:
        if r.rollout_id not in games:
            games[r.rollout_id] = {"meta": r.meta, "records": []}
        games[r.rollout_id]["records"].append(r)

    print(f"Games: {len(games)} | Records: {len(batch.records)}\n")

    # Track stats
    total_value_changes = {"Player0": [], "Player1": []}
    winners = {"Player0": 0, "Player1": 0, "Draw": 0}

    for i, (_, data) in enumerate(games.items(), 1):
        meta = data["meta"]
        extras = meta.get("extras", {})

        winner = extras.get("winner", "?")
        if winner in winners:
            winners[winner] += 1
        else:
            winners["Draw"] += 1

        value_changes = extras.get("value_changes", {})
        for player, change in value_changes.items():
            if player in total_value_changes:
                total_value_changes[player].append(change)

        trades = extras.get("trades", [])
        num_trades = len(trades) if trades else 0

        invalid_player = extras.get("invalid_action")
        if invalid_player:
            game_rewards = {
                invalid_player: -1.5,
                ("Player1" if invalid_player == "Player0" else "Player0"): 0.5
            }
        elif winner == "Player0":
            game_rewards = {"Player0": 1.0, "Player1": -1.0}
        elif winner == "Player1":
            game_rewards = {"Player0": -1.0, "Player1": 1.0}
        else:
            game_rewards = {"Player0": 0.0, "Player1": 0.0}

        turn_counts = {"Player0": 0, "Player1": 0}
        for rec in data["records"]:
            if rec.actor_id in turn_counts:
                turn_counts[rec.actor_id] += 1

        rewards_str = ", ".join(f"{p}: {r:+.1f}" for p, r in sorted(game_rewards.items()))
        turns_str = ", ".join(f"{p}: {c}" for p, c in sorted(turn_counts.items()))
        print(f"[{i}] Winner: {winner} | Trades: {num_trades} | Rewards: {rewards_str} | Turns: {turns_str}")

        if value_changes:
            changes_str = ", ".join(f"{p}: {c:+.0f}" for p, c in value_changes.items())
            print(f"    Value changes: {changes_str}")

        final_inventories = extras.get("player_resources", {})
        if final_inventories:
            for player, inv in final_inventories.items():
                inv_str = ", ".join(f"{k}={v}" for k, v in inv.items())
                print(f"    {player} inventory: {inv_str}")

        print()

    print("Game Summary:")
    print(f"  Winners: Player0={winners['Player0']}, Player1={winners['Player1']}, Draw={winners['Draw']}")

    for player, changes in total_value_changes.items():
        if changes:
            avg_change = sum(changes) / len(changes)
            print(f"  {player} avg value change: {avg_change:+.1f}")

    all_rewards = [r.reward for r in batch.records]
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"\nAvg reward: {avg_reward:.3f} | Range: {min(all_rewards):.3f} to {max(all_rewards):.3f}")


async def main(args):
    # Setup inference client
    print(f"\nConnecting to inference server at {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=120.0,
    )

    # Setup arena with RAECredit for actor-conditioned baselines
    arena = NegotiationArena(
        client=client,
        episodes_per_step=EPISODES_PER_STEP,
        verbose=args.verbose,
        credit_assigner=RAECredit(decay=RAE_DECAY),
    )

    # Add actors with negotiation-specific system prompts
    arena.add_actor(Actor(
        id="Player0",
        system_prompt=PLAYER_0_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_actor(Actor(
        id="Player1",
        system_prompt=PLAYER_1_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    # Add episode
    arena.add_episode("negotiation", NegotiationEpisode(
        player_0_actor_id="Player0",
        player_1_actor_id="Player1",
        max_turns=MAX_GAME_TURNS,
    ))

    # Dry-run mode
    if args.dry_run:
        await preview_negotiation(arena, concurrency=EPISODE_CONCURRENCY)
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
    print(f"  - Game: SimpleNegotiation ({MAX_GAME_TURNS} max turns)")
    print(f"  - Episodes per step: {EPISODES_PER_STEP}")
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
    parser = argparse.ArgumentParser(description="Train negotiation agents with RL")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    args = parser.parse_args()
    asyncio.run(main(args))
