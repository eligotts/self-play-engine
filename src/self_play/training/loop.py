"""
Training loop that coordinates async generation and training.

The training loop runs two concurrent async tasks:
1. Generator: Calls arena.step() and pushes TrainingBatches to a queue
2. Trainer: Pulls from queue, accumulates until full batch, trains

This pattern allows generation and training to proceed concurrently,
maximizing hardware utilization.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Dict, List, Optional

from collections import defaultdict

from ..core.arena import Arena
from ..core.types import TrainingBatch, TrainingRecord
from .trainer import Trainer


def compute_per_role_reward_stats(records: List[TrainingRecord]) -> Dict[str, float]:
    """Compute per-role reward statistics (avg/min/max)."""
    if not records:
        return {}

    role_rewards: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        role_rewards[r.role_id].append(r.reward)

    stats: Dict[str, float] = {}
    for role_id, rewards in role_rewards.items():
        stats[f"reward_{role_id}_avg"] = sum(rewards) / len(rewards)
        stats[f"reward_{role_id}_min"] = min(rewards)
        stats[f"reward_{role_id}_max"] = max(rewards)
        stats[f"reward_{role_id}_count"] = float(len(rewards))

    return stats


def compute_eval_metrics(batch: TrainingBatch) -> Dict[str, float]:
    """Compute aggregate metrics from evaluation batch."""
    if not batch.records:
        return {}

    rewards = [r.reward for r in batch.records]
    advantages = [r.advantage for r in batch.records]

    metrics = {
        "num_records": len(batch.records),
        "avg_reward": sum(rewards) / len(rewards),
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "avg_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
    }

    # Add per-role reward breakdown
    metrics.update(compute_per_role_reward_stats(batch.records))

    return metrics


async def training_loop(
    arena: Arena,
    trainer: Trainer,
    batch_queue: asyncio.Queue,
    num_steps: int,
    concurrency: int = 8,
    step_concurrency: int = 1,
    verbose: bool = False,
) -> None:
    """
    Run the main training loop with async generation and training.

    Two async tasks run concurrently:
    1. Generator loop: Calls arena.step() and pushes TrainingBatches to queue
    2. Trainer loop: Pulls from queue, waits for full batch, trains

    Args:
        arena: Arena instance for generating rollouts
        trainer: Trainer instance for training
        batch_queue: asyncio.Queue for passing batches between generator and trainer
        num_steps: Number of training steps to run
        concurrency: Max concurrent episodes during generation
        step_concurrency: Max concurrent arena.step() calls
        verbose: Print debug info
    """
    # Initialize wandb if configured
    if trainer.config.wandb_project:
        import wandb
        wandb.init(
            project=trainer.config.wandb_project,
            name=trainer.config.wandb_run_name,
            config=asdict(trainer.config),
        )

    shutdown = asyncio.Event()

    # Evaluation coordination
    eval_event = asyncio.Event()
    eval_step_holder = [0]  # Mutable container to share current step

    async def generator_loop():
        """Generate rollouts and push to queue."""
        batches_generated = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        pending: set[asyncio.Task] = set()

        while not shutdown.is_set():
            # Launch new step tasks up to step_concurrency
            while len(pending) < step_concurrency and not shutdown.is_set():
                task = asyncio.create_task(
                    arena.step(concurrency=concurrency)
                )
                pending.add(task)

            if not pending:
                break

            # Wait for at least one to complete
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0,  # Allow checking shutdown periodically
            )

            for task in done:
                try:
                    batch = task.result()
                    consecutive_errors = 0  # Reset on success

                    if batch.records:
                        await batch_queue.put(batch)  # Blocks if queue full (backpressure)
                        batches_generated += 1

                        if verbose:
                            print(f"[generator] batch {batches_generated}: {len(batch.records)} records")

                except Exception as e:
                    consecutive_errors += 1
                    print(f"[generator] error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[generator] too many consecutive errors, shutting down")
                        shutdown.set()
                        break

        # Cancel any remaining tasks on shutdown
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        if verbose:
            print(f"[generator] shutdown after {batches_generated} batches")

    async def trainer_loop():
        """Pull batches and train when we have enough records."""
        records_buffer: List[TrainingRecord] = []
        train_step = 0

        while train_step < num_steps:
            try:
                # Try to get a batch with timeout
                batch = await asyncio.wait_for(
                    batch_queue.get(),
                    timeout=1.0,
                )

                # Filter stale records immediately
                current_version = trainer.train_step_idx
                max_lag = trainer.config.max_policy_lag

                fresh = []
                for r in batch.records:
                    policy_version = r.meta.get("policy_version", 0)
                    if current_version - policy_version <= max_lag:
                        fresh.append(r)

                records_buffer.extend(fresh)

                stale_count = len(batch.records) - len(fresh)
                if verbose and stale_count > 0:
                    print(f"[trainer] dropped {stale_count} stale records")

                # Train when we have a full batch
                while len(records_buffer) >= trainer.config.batch_size:
                    # Take exactly batch_size records
                    train_records = records_buffer[:trainer.config.batch_size]
                    records_buffer = records_buffer[trainer.config.batch_size:]

                    # Create batch and train
                    train_batch = TrainingBatch(
                        records=train_records,
                        meta={},
                    )

                    metrics = await trainer.train_step(train_batch)
                    train_step += 1

                    # Add training batch reward stats
                    reward_stats = compute_per_role_reward_stats(train_records)
                    reward_stats["avg_reward"] = sum(r.reward for r in train_records) / len(train_records)
                    reward_stats["avg_advantage"] = sum(r.advantage for r in train_records) / len(train_records)
                    metrics.update({f"train/{k}": v for k, v in reward_stats.items()})

                    # Log to wandb
                    if trainer.config.wandb_project:
                        import wandb
                        wandb.log(metrics, step=train_step)

                    # Signal evaluation loop
                    eval_step_holder[0] = train_step
                    eval_event.set()

                    if verbose:
                        print(f"[trainer] step {train_step}: loss={metrics.get('loss', 0):.4f}, "
                              f"records={metrics.get('records', 0)}, "
                              f"tokens={metrics.get('tokens', 0)}")

                    if train_step >= num_steps:
                        break

            except asyncio.TimeoutError:
                # No batch available, check if we should continue
                if shutdown.is_set() and batch_queue.empty():
                    break
                continue

            except Exception as e:
                if verbose:
                    print(f"[trainer] error: {e}")
                raise

        # Signal shutdown
        shutdown.set()

        if verbose:
            print(f"[trainer] finished after {train_step} steps")
            if records_buffer:
                print(f"[trainer] {len(records_buffer)} records left in buffer")

    async def evaluation_loop():
        """Run evaluation periodically (non-blocking)."""
        # Skip if evaluation disabled
        if trainer.config.eval_every <= 0:
            return

        while not shutdown.is_set():
            # Wait for trainer to signal a step completed
            try:
                await asyncio.wait_for(eval_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            eval_event.clear()

            # Capture step at time of signal
            current_step = eval_step_holder[0]

            # Only evaluate on eval_every steps
            if current_step % trainer.config.eval_every != 0:
                continue

            if verbose:
                print(f"[eval] Starting evaluation at step {current_step}")

            try:
                # Run evaluation rollouts
                batch = await arena.step(
                    concurrency=trainer.config.eval_concurrency,
                )

                if not batch.records:
                    if verbose:
                        print(f"[eval] No records from arena")
                    continue

                # Compute evaluation metrics
                eval_metrics = compute_eval_metrics(batch)

                # Log with eval/ prefix
                if trainer.config.wandb_project:
                    import wandb
                    prefixed = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    wandb.log(prefixed, step=current_step)

                if verbose:
                    print(f"[eval] step {current_step}: "
                          f"avg_reward={eval_metrics.get('avg_reward', 0):.4f}, "
                          f"num_records={eval_metrics.get('num_records', 0)}")

            except Exception as e:
                if verbose:
                    print(f"[eval] error: {e}")
                # Continue evaluating despite errors

        if verbose:
            print(f"[eval] shutdown")

    # Run all three loops concurrently
    await asyncio.gather(
        generator_loop(),
        trainer_loop(),
        evaluation_loop(),
    )

    # Finish wandb run
    if trainer.config.wandb_project:
        import wandb
        wandb.finish()


async def simple_training_loop(
    arena: Arena,
    trainer: Trainer,
    num_steps: int,
    concurrency: int = 8,
    step_concurrency: int = 1,
    verbose: bool = False,
) -> None:
    """
    Simple sequential training loop (no async overlap).

    For simpler debugging or when generation is fast enough that
    async overlap isn't needed.

    Args:
        arena: Arena instance for generating rollouts
        trainer: Trainer instance for training
        num_steps: Number of training steps to run
        concurrency: Max concurrent episodes during generation
        step_concurrency: Number of arena.step() calls to run in parallel
        verbose: Print debug info
    """
    # Initialize wandb if configured
    if trainer.config.wandb_project:
        import wandb
        wandb.init(
            project=trainer.config.wandb_project,
            name=trainer.config.wandb_run_name,
            config=asdict(trainer.config),
        )

    records_buffer: List[TrainingRecord] = []
    consecutive_errors = 0
    max_consecutive_errors = 5

    for step in range(num_steps):
        # Generate until we have enough records
        while len(records_buffer) < trainer.config.batch_size:
            # Launch step_concurrency arena.step() calls in parallel
            tasks = [
                asyncio.create_task(arena.step(concurrency=concurrency))
                for _ in range(step_concurrency)
            ]

            # Wait for ALL to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            round_records = 0
            round_stale = 0
            for result in results:
                if isinstance(result, Exception):
                    consecutive_errors += 1
                    print(f"[step {step}] error ({consecutive_errors}/{max_consecutive_errors}): {result}")
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[step {step}] too many consecutive errors, aborting")
                        return
                    continue

                consecutive_errors = 0  # Reset on success
                batch = result

                if not batch.records:
                    continue

                # Filter stale records
                current_version = trainer.train_step_idx
                max_lag = trainer.config.max_policy_lag

                fresh = [
                    r for r in batch.records
                    if current_version - r.meta.get("policy_version", 0) <= max_lag
                ]
                records_buffer.extend(fresh)
                round_records += len(fresh)
                round_stale += len(batch.records) - len(fresh)

            if verbose and round_records > 0:
                print(f"[step {step}] generated {round_records + round_stale} records "
                      f"({round_records} fresh, {round_stale} stale), "
                      f"buffer={len(records_buffer)}")

        # Take batch_size records and train
        train_records = records_buffer[:trainer.config.batch_size]
        records_buffer = records_buffer[trainer.config.batch_size:]

        train_batch = TrainingBatch(
            records=train_records,
            meta={},
        )

        metrics = await trainer.train_step(train_batch)

        # Add training batch reward stats
        reward_stats = compute_per_role_reward_stats(train_records)
        reward_stats["avg_reward"] = sum(r.reward for r in train_records) / len(train_records)
        reward_stats["avg_advantage"] = sum(r.advantage for r in train_records) / len(train_records)
        metrics.update({f"train/{k}": v for k, v in reward_stats.items()})

        # Log to wandb
        if trainer.config.wandb_project:
            import wandb
            wandb.log(metrics, step=step + 1)

        if verbose:
            print(f"[step {step}] trained: loss={metrics.get('loss', 0):.4f}, "
                  f"records={metrics.get('records', 0)}, "
                  f"tokens={metrics.get('tokens', 0)}")

        # Run evaluation periodically
        if trainer.config.eval_every > 0 and (step + 1) % trainer.config.eval_every == 0:
            if verbose:
                print(f"[step {step}] running evaluation...")

            eval_batch = await arena.step(
                concurrency=trainer.config.eval_concurrency,
            )
            eval_metrics = compute_eval_metrics(eval_batch)

            if trainer.config.wandb_project:
                import wandb
                prefixed = {f"eval/{k}": v for k, v in eval_metrics.items()}
                wandb.log(prefixed, step=step + 1)

            if verbose:
                print(f"[step {step}] eval: avg_reward={eval_metrics.get('avg_reward', 0):.4f}, "
                      f"num_records={eval_metrics.get('num_records', 0)}")

    # Finish wandb run
    if trainer.config.wandb_project:
        import wandb
        wandb.finish()
