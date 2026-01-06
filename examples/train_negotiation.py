"""
Example: Training negotiation agents with RL.

This script demonstrates how to use the self-play training module to train
negotiation agents using SPIRAL's SimpleNegotiation environment. It combines:
- The NegotiationArena from the tasks module (for generation)
- The Trainer from the training module (for training)
- An mlx-vllm server for inference (with LoRA hot-swap)
- RAECredit for role-conditioned advantage estimation

The game:
- Two players with opposite resource preferences
- Player 0 values Gold more (Wood=5, Gold=15)
- Player 1 values Wood more (Wood=15, Gold=5)
- Both start with 10 Wood + 10 Gold
- Players negotiate and trade to maximize their inventory value
- Winner = player with larger inventory value change

To run this example:
1. Start mlx-vllm server with LoRA enabled:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \\
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Run this script:
   python examples/train_negotiation.py
"""
import asyncio
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient, Role, RAECredit
from self_play.tasks.negotiation import NegotiationArena, NegotiationEpisode
from self_play.training import (
    Trainer,
    TrainerConfig,
    WeightPublisher,
    training_loop,
    simple_training_loop,
)


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

    for i, (rid, data) in enumerate(games.items(), 1):
        meta = data["meta"]
        extras = meta.get("extras", {})

        # Game outcome
        winner = extras.get("winner", "?")
        if winner in winners:
            winners[winner] += 1
        else:
            winners["Draw"] += 1

        # Value changes per player
        value_changes = extras.get("value_changes", {})
        for player, change in value_changes.items():
            if player in total_value_changes:
                total_value_changes[player].append(change)

        # Trades made
        trades = extras.get("trades", [])
        num_trades = len(trades) if trades else 0

        print(f"[{i}] Winner: {winner} | Trades: {num_trades}")

        # Show value changes
        if value_changes:
            changes_str = ", ".join(f"{p}: {c:+.0f}" for p, c in value_changes.items())
            print(f"    Value changes: {changes_str}")

        # Show final inventories
        final_inventories = extras.get("player_resources", {})
        if final_inventories:
            for player, inv in final_inventories.items():
                inv_str = ", ".join(f"{k}={v}" for k, v in inv.items())
                print(f"    {player} inventory: {inv_str}")

        # Show trade history (truncated)
        if trades:
            for j, trade in enumerate(trades[:2]):  # Show first 2 trades
                print(f"    Trade {j+1}: {truncate(str(trade), 80)}")
            if len(trades) > 2:
                print(f"    ... and {len(trades) - 2} more trades")

        # Show sample dialogue
        dialogue = extras.get("dialogue", [])
        if dialogue:
            print("    Sample dialogue:")
            for msg in dialogue[:3]:  # Show first 3 messages
                role = msg.get("role", "?")
                content = msg.get("content", "")
                print(f"      {role}: {truncate(content, 60)}")
            if len(dialogue) > 3:
                print(f"      ... and {len(dialogue) - 3} more messages")

        print()

    # Summary stats
    print("Game Summary:")
    print(f"  Winners: Player0={winners['Player0']}, Player1={winners['Player1']}, Draw={winners['Draw']}")

    for player, changes in total_value_changes.items():
        if changes:
            avg_change = sum(changes) / len(changes)
            print(f"  {player} avg value change: {avg_change:+.1f}")

    # Reward stats
    all_rewards = [r.reward for r in batch.records]
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"\nAvg reward: {avg_reward:.3f} | Range: {min(all_rewards):.3f} to {max(all_rewards):.3f}")


async def main(args):
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

    # Setup arena with RAECredit for role-conditioned baselines
    print("Setting up negotiation arena...")
    arena = NegotiationArena(
        client=client,
        batch_size=args.batch_size,
        verbose=args.verbose,
        credit_assigner=RAECredit(decay=args.rae_decay),
    )

    # Add roles with negotiation-specific system prompts
    arena.add_role(Role(
        id="Player0",
        system_prompt=PLAYER_0_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    arena.add_role(Role(
        id="Player1",
        system_prompt=PLAYER_1_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    # Add episode
    arena.add_episode("negotiation", NegotiationEpisode(
        player_0_role_id="Player0",
        player_1_role_id="Player1",
        max_turns=args.max_game_turns,
    ))

    # Dry-run mode: preview arena performance without training
    if args.dry_run:
        await preview_negotiation(arena, concurrency=args.concurrency)
        await client.close()
        return

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
        clip_low=0.8,
        clip_high=1.2,
        kl_coef=args.kl_coef,
        use_kl_penalty=args.use_kl_penalty,
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
    print(f"  - Game: SimpleNegotiation ({args.max_game_turns} max turns)")
    print(f"  - Batch size: {args.batch_size} games per arena step")
    print(f"  - Train batch size: {args.train_batch_size} records per train step")
    print(f"  - Micro token budget: {args.micro_token_budget}")
    print(f"  - Max policy lag: {args.max_policy_lag}")
    print(f"  - RAE decay: {args.rae_decay}")
    print()

    try:
        if args.simple_loop:
            # Use simple sequential loop (easier to debug)
            await simple_training_loop(
                arena=arena,
                trainer=trainer,
                num_steps=args.num_steps,
                concurrency=args.concurrency,
                step_concurrency=args.step_concurrency,
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
        # Cleanup
        await publisher.close()
        await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train negotiation agents with RL (SPIRAL SimpleNegotiation)")

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

    # Game args
    parser.add_argument("--max-game-turns", type=int, default=10, help="Max turns per negotiation game")

    # Training args
    parser.add_argument("--num-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Games per arena step")
    parser.add_argument("--train-batch-size", type=int, default=24, help="Records per train step")
    parser.add_argument("--micro-token-budget", type=int, default=4096, help="Tokens per micro-batch")
    parser.add_argument("--max-policy-lag", type=int, default=3, help="Max staleness (steps)")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--use-kl-penalty", action="store_true", help="Add KL penalty to loss")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent episodes")
    parser.add_argument("--step-concurrency", type=int, default=1, help="Max concurrent arena.step() calls")

    # RAE args
    parser.add_argument("--rae-decay", type=float, default=0.95, help="RAE baseline EMA decay factor")

    # Generation args
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per completion")

    # Mode args
    parser.add_argument(
        "--simple-loop",
        action="store_true",
        help="Use simple sequential loop instead of async",
    )
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")

    # Wandb args
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name (enables logging)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()
    asyncio.run(main(args))
