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
from typing import List, Optional

from ..core.arena import Arena, TrainingBatch
from ..core.types import TrainingRecord
from .trainer import Trainer


async def training_loop(
    arena: Arena,
    trainer: Trainer,
    batch_queue: asyncio.Queue,
    num_steps: int,
    concurrency: int = 8,
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
        verbose: Print debug info
    """
    generation_step = 0
    shutdown = asyncio.Event()

    async def generator_loop():
        """Generate rollouts and push to queue."""
        nonlocal generation_step

        while not shutdown.is_set() and generation_step < num_steps:
            try:
                batch = await arena.step(concurrency=concurrency, verbose=verbose)

                if batch.records:
                    await batch_queue.put(batch)
                    generation_step += 1

                    if verbose:
                        print(f"[generator] step {generation_step}: {len(batch.records)} records")

            except Exception as e:
                if verbose:
                    print(f"[generator] error: {e}")
                # Continue generating despite errors

        if verbose:
            print(f"[generator] finished after {generation_step} steps")

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

    # Run both loops concurrently
    await asyncio.gather(
        generator_loop(),
        trainer_loop(),
    )


async def simple_training_loop(
    arena: Arena,
    trainer: Trainer,
    num_steps: int,
    concurrency: int = 8,
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
        verbose: Print debug info
    """
    records_buffer: List[TrainingRecord] = []

    for step in range(num_steps):
        # Generate until we have enough records
        while len(records_buffer) < trainer.config.batch_size:
            batch = await arena.step(concurrency=concurrency, verbose=verbose)

            if not batch.records:
                if verbose:
                    print(f"[step {step}] no records from arena.step()")
                continue

            # Filter stale records
            current_version = trainer.train_step_idx
            max_lag = trainer.config.max_policy_lag

            fresh = [
                r for r in batch.records
                if current_version - r.meta.get("policy_version", 0) <= max_lag
            ]
            records_buffer.extend(fresh)

            if verbose:
                stale = len(batch.records) - len(fresh)
                print(f"[step {step}] generated {len(batch.records)} records "
                      f"({len(fresh)} fresh, {stale} stale), "
                      f"buffer={len(records_buffer)}")

        # Take batch_size records and train
        train_records = records_buffer[:trainer.config.batch_size]
        records_buffer = records_buffer[trainer.config.batch_size:]

        train_batch = TrainingBatch(
            records=train_records,
            meta={},
        )

        metrics = await trainer.train_step(train_batch)

        if verbose:
            print(f"[step {step}] trained: loss={metrics.get('loss', 0):.4f}, "
                  f"records={metrics.get('records', 0)}, "
                  f"tokens={metrics.get('tokens', 0)}")
