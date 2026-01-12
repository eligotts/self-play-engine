"""
Training loop that coordinates async generation and training.

Two concurrent async tasks:
1. Generator: Calls arena.step() and pushes TrainingBatches to a queue
2. Trainer: Pulls from queue, streams micro-batches, runs eval inline

Eval runs synchronously within the trainer loop to ensure correct wandb step ordering.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Dict, List

from ..core.arena import Arena
from ..core.types import TrainingBatch, TrainingRecord
from .trainer import Trainer, compute_per_actor_reward_stats
from .batching import estimate_tokens, form_micro_batch


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

    # Add per-actor reward breakdown
    metrics.update(compute_per_actor_reward_stats(batch.records))

    return metrics


async def training_loop(
    arena: Arena,
    trainer: Trainer,
    batch_queue: asyncio.Queue,
    num_steps: int,
    episode_concurrency: int = 8,
    step_concurrency: int = 1,
    verbose: bool = False,
) -> None:
    """
    Run the main training loop with async generation and streaming training.

    Two async tasks run concurrently:
    1. Generator loop: Calls arena.step() and pushes TrainingBatches to queue
    2. Trainer loop: Pulls from queue, streams micro-batches, runs eval inline

    Args:
        arena: Arena instance for generating rollouts
        trainer: Trainer instance for training
        batch_queue: asyncio.Queue for passing batches between generator and trainer
        num_steps: Number of training steps to run
        episode_concurrency: Max concurrent episodes during generation
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
                    arena.step(concurrency=episode_concurrency)
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
        """Pull batches and stream micro-batches to trainer as they form."""
        records_buffer: List[TrainingRecord] = []
        train_step = 0

        while train_step < num_steps:
            try:
                # Try to get a batch with timeout
                batch = await asyncio.wait_for(
                    batch_queue.get(),
                    timeout=1.0,
                )

                # Filter stale records immediately (trainer will also filter, but
                # filtering here prevents buffering records that will be dropped)
                current_version = trainer.train_step_idx
                staleness_limit = trainer.config.staleness_limit

                fresh = []
                for r in batch.records:
                    policy_version = r.meta.get("policy_version", 0)
                    if current_version - policy_version <= staleness_limit:
                        fresh.append(r)

                records_buffer.extend(fresh)

                stale_count = len(batch.records) - len(fresh)
                if verbose and stale_count > 0:
                    print(f"[trainer] dropped {stale_count} stale records")

                # Form and process micro-batches as soon as we have enough tokens
                micro_batch_tokens = trainer.config.micro_batch_tokens
                while estimate_tokens(records_buffer) >= micro_batch_tokens:
                    micro_batch, records_buffer = form_micro_batch(
                        records_buffer, micro_batch_tokens
                    )

                    if verbose:
                        print(f"[trainer] processing micro-batch: {len(micro_batch)} records, "
                              f"{estimate_tokens(micro_batch)} tokens, "
                              f"pending={trainer.pending_samples}")

                    # Send micro-batch to trainer (accumulates gradients, may step)
                    result = await trainer.accumulate(micro_batch)

                    if result["stepped"]:
                        train_step += 1
                        metrics = result["step_metrics"]

                        # Save checkpoint if it's time
                        if trainer.config.checkpoint_every > 0 and train_step % trainer.config.checkpoint_every == 0:
                            checkpoint_path = trainer.save_checkpoint(train_step)
                            if verbose and checkpoint_path:
                                print(f"[trainer] saved checkpoint: {checkpoint_path}")

                        # Run eval inline if it's time (ensures correct step ordering)
                        if trainer.config.eval_every > 0 and train_step % trainer.config.eval_every == 0:
                            if verbose:
                                print(f"[trainer] running evaluation at step {train_step}...")
                            eval_batch = await arena.step(
                                concurrency=trainer.config.eval_concurrency,
                            )
                            if eval_batch.records:
                                eval_metrics = compute_eval_metrics(eval_batch)
                                for k, v in eval_metrics.items():
                                    metrics[f"eval/{k}"] = v

                        # Log all metrics together
                        if trainer.config.wandb_project:
                            import wandb
                            wandb.log(metrics, step=train_step)

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
            if trainer.pending_samples > 0:
                print(f"[trainer] {trainer.pending_samples} samples pending (not enough for full batch)")

    # Run generator and trainer concurrently
    await asyncio.gather(
        generator_loop(),
        trainer_loop(),
    )

    # Finish wandb run
    if trainer.config.wandb_project:
        import wandb
        wandb.finish()


async def synchronous_training_loop(
    arena: Arena,
    trainer: Trainer,
    num_steps: int,
    episode_concurrency: int = 8,
    step_concurrency: int = 1,
    verbose: bool = False,
) -> None:
    """
    Simple sequential training loop with micro-batch streaming.

    Generation and training don't overlap, but training uses the same
    micro-batch streaming approach as training_loop - accumulating
    gradients until min_samples_per_step is reached.

    Args:
        arena: Arena instance for generating rollouts
        trainer: Trainer instance for training
        num_steps: Number of training steps to run
        episode_concurrency: Max concurrent episodes during generation
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
    train_step = 0

    async def generate_batch() -> List[TrainingRecord]:
        """Generate one round of rollouts concurrently."""
        nonlocal consecutive_errors

        tasks = [
            asyncio.create_task(arena.step(concurrency=episode_concurrency))
            for _ in range(step_concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        fresh = []
        for result in results:
            if isinstance(result, Exception):
                consecutive_errors += 1
                print(f"[generate] error ({consecutive_errors}/{max_consecutive_errors}): {result}")
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError("Too many consecutive generation errors")
                continue

            consecutive_errors = 0
            if result.records:
                fresh.extend(result.records)

        return fresh

    while train_step < num_steps:
        # Phase 1: Ensure buffer has at least one micro-batch worth of data
        while estimate_tokens(records_buffer) < trainer.config.micro_batch_tokens:
            new_records = await generate_batch()
            records_buffer.extend(new_records)
            if verbose:
                print(f"[step {train_step}] generated {len(new_records)} records, "
                      f"buffer={len(records_buffer)}")

        # Phase 2: Process micro-batches until we step
        stepped_this_iteration = False
        while not stepped_this_iteration:
            # Form and process micro-batches while we have enough tokens
            while estimate_tokens(records_buffer) >= trainer.config.micro_batch_tokens:
                micro_batch, records_buffer = form_micro_batch(
                    records_buffer, trainer.config.micro_batch_tokens
                )

                if verbose:
                    print(f"[step {train_step}] processing micro-batch: {len(micro_batch)} records, "
                          f"{estimate_tokens(micro_batch)} tokens, "
                          f"pending={trainer.pending_samples}")

                result = await trainer.accumulate(micro_batch)

                if result["stepped"]:
                    stepped_this_iteration = True
                    train_step += 1
                    metrics = result["step_metrics"]

                    # Log to wandb
                    if trainer.config.wandb_project:
                        import wandb
                        wandb.log(metrics, step=train_step)

                    if verbose:
                        print(f"[step {train_step}] trained: loss={metrics.get('loss', 0):.4f}, "
                              f"records={metrics.get('records', 0)}, "
                              f"tokens={metrics.get('tokens', 0)}")

                    # Save checkpoint if it's time
                    if trainer.config.checkpoint_every > 0 and train_step % trainer.config.checkpoint_every == 0:
                        checkpoint_path = trainer.save_checkpoint(train_step)
                        if verbose and checkpoint_path:
                            print(f"[step {train_step}] saved checkpoint: {checkpoint_path}")

                    # Run evaluation periodically
                    if trainer.config.eval_every > 0 and train_step % trainer.config.eval_every == 0:
                        if verbose:
                            print(f"[step {train_step}] running evaluation...")
                        eval_batch = await arena.step(
                            concurrency=trainer.config.eval_concurrency,
                        )
                        eval_metrics = compute_eval_metrics(eval_batch)

                        if trainer.config.wandb_project:
                            import wandb
                            prefixed = {f"eval/{k}": v for k, v in eval_metrics.items()}
                            wandb.log(prefixed, step=train_step)

                        if verbose:
                            print(f"[step {train_step}] eval: avg_reward={eval_metrics.get('avg_reward', 0):.4f}")

                    break  # Exit micro-batch loop after step

            if stepped_this_iteration:
                break  # Exit to outer loop to check num_steps

            # Buffer exhausted but didn't step - need more data
            if verbose:
                print(f"[step {train_step}] buffer exhausted, pending={trainer.pending_samples}, "
                      f"need {trainer.config.min_samples_per_step}, generating more...")

            new_records = await generate_batch()
            records_buffer.extend(new_records)

            if verbose:
                print(f"[step {train_step}] generated {len(new_records)} records, "
                      f"buffer={len(records_buffer)}")

    # Finish wandb run
    if trainer.config.wandb_project:
        import wandb
        wandb.finish()

    if verbose:
        print(f"[training] finished after {train_step} steps")
        if records_buffer:
            print(f"[training] {len(records_buffer)} records left in buffer")
        if trainer.pending_samples > 0:
            print(f"[training] {trainer.pending_samples} samples pending (not enough for full batch)")
