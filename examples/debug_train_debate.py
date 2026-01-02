"""
Debug version of train_debate.py with additional logging and memory monitoring.
"""
import asyncio
import argparse
import traceback
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient
from self_play.examples.debate import create_debate_arena
from self_play.training import (
    Trainer,
    TrainerConfig,
    WeightPublisher,
)
from self_play.core.arena import TrainingBatch
from self_play.core.types import TrainingRecord


def get_memory_info():
    """Get current Metal memory usage."""
    try:
        active = mx.metal.get_active_memory() / 1e9
        peak = mx.metal.get_peak_memory() / 1e9
        cache = mx.metal.get_cache_memory() / 1e9
        return f"active={active:.2f}GB, peak={peak:.2f}GB, cache={cache:.2f}GB"
    except Exception as e:
        return f"(memory info unavailable: {e})"


def load_model_with_lora(
    model_path: str,
    lora_rank: int = 8,
    lora_layers: int = 16,
    lora_scale: float = 20.0,
):
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    print(f"Attaching LoRA (rank={lora_rank}, layers={lora_layers})...")
    lora_config = {
        "rank": lora_rank,
        "scale": lora_scale,
        "dropout": 0.0,
    }
    linear_to_lora_layers(model, lora_layers, lora_config)

    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    print(f"Memory after model load: {get_memory_info()}")

    return model, tokenizer


async def debug_train_step(trainer, batch, step_num):
    """Train step with detailed debugging."""
    print(f"\n{'='*60}")
    print(f"DEBUG TRAIN STEP {step_num}")
    print(f"{'='*60}")
    print(f"  Input records: {len(batch.records)}")
    print(f"  Memory before: {get_memory_info()}")

    try:
        # Step through trainer manually for debugging
        trainer.model.train()
        fresh_records = trainer.filter_stale_records(batch.records)
        print(f"  Fresh records: {len(fresh_records)}")

        if not fresh_records:
            print("  -> All records stale, skipping")
            return {"skipped": 1}

        from self_play.training.batching import split_by_token_budget, collate

        micro_batches = split_by_token_budget(
            fresh_records,
            trainer.config.micro_token_budget,
        )
        print(f"  Micro-batches: {len(micro_batches)}")

        for i, micro in enumerate(micro_batches):
            print(f"\n  -- Micro-batch {i+1}/{len(micro_batches)} --")
            print(f"     Records: {len(micro)}")
            print(f"     Tokens: {sum(len(r.input_ids) for r in micro)}")
            print(f"     Memory before collate: {get_memory_info()}")

            input_ids, loss_mask, inference_logprobs, advantages = collate(
                micro,
                pad_token_id=trainer.config.pad_token_id,
            )
            print(f"     Collated shapes: input_ids={input_ids.shape}, loss_mask={loss_mask.shape}")
            print(f"     Memory after collate: {get_memory_info()}")

            print(f"     Running forward pass...")
            sys.stdout.flush()

            # This is where the crash likely happens
            from self_play.training.loss import compute_loss

            try:
                loss, metrics = compute_loss(
                    model=trainer.model,
                    input_ids=input_ids,
                    inference_logprobs=inference_logprobs,
                    advantages=advantages,
                    loss_mask=loss_mask,
                    use_importance_sampling=trainer.config.use_importance_sampling,
                    clip_low=trainer.config.clip_low,
                    clip_high=trainer.config.clip_high,
                )
                # Force evaluation
                mx.eval(loss)
                print(f"     Loss: {float(loss.item()):.4f}")
                print(f"     Memory after forward: {get_memory_info()}")
            except Exception as e:
                print(f"     FORWARD PASS FAILED: {e}")
                traceback.print_exc()
                raise

            mx.clear_cache()
            print(f"     Memory after clear_cache: {get_memory_info()}")

        print(f"\n  Full train_step...")
        sys.stdout.flush()

        # Now try the actual train step
        metrics = await trainer.train_step(batch)
        print(f"  Train step complete: {metrics}")
        print(f"  Memory after train_step: {get_memory_info()}")
        return metrics

    except Exception as e:
        print(f"  TRAIN STEP FAILED: {e}")
        traceback.print_exc()
        raise


async def main(args):
    topics = [
        "Social media does more harm than good to society",
        "Artificial intelligence will create more jobs than it destroys",
        "Climate change is the most pressing issue of our time",
    ]

    print(f"\n{'='*60}")
    print("DEBUG MODE")
    print(f"{'='*60}")
    print(f"Initial memory: {get_memory_info()}")

    # Setup inference client
    print(f"\nConnecting to inference server at localhost:{args.port}...")
    client = OpenAIClient.for_local(port=args.port, timeout=120.0)

    # Check server is responding
    try:
        version = await client.get_policy_version()
        print(f"Server responding, policy version: {version}")
    except Exception as e:
        print(f"ERROR: Server not responding: {e}")
        return

    # Setup arena
    print("Setting up debate arena...")
    arena = create_debate_arena(
        client=client,
        topics=topics,
        num_rounds=2,  # Fewer rounds for debugging
        batch_size=2,  # Smaller batch for debugging
        verbose=True,
    )

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
        max_policy_lag=100,  # High lag for debugging to not drop records
        batch_size=4,  # Small batch for debugging
        use_importance_sampling=args.use_importance_sampling,
        clip_low=0.8,
        clip_high=1.2,
        weight_push_url=f"http://localhost:{args.port}",
        pad_token_id=tokenizer.pad_token_id or 0,
        verbose=True,
    )

    # Setup weight publisher - but skip publishing for now
    publisher = None  # WeightPublisher(base_url=f"http://localhost:{args.port}")

    # Setup trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        publisher=publisher,
    )

    print(f"\n{'='*60}")
    print("STARTING DEBUG LOOP")
    print(f"{'='*60}")
    print(f"Memory before generation: {get_memory_info()}")

    records_buffer = []

    for step in range(args.num_steps):
        print(f"\n{'='*60}")
        print(f"GENERATION STEP {step + 1}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Generate until we have enough records
        while len(records_buffer) < config.batch_size:
            print(f"  Generating... (buffer has {len(records_buffer)} records)")
            print(f"  Memory before arena.step: {get_memory_info()}")
            sys.stdout.flush()

            try:
                batch = await arena.step(concurrency=2, verbose=True)
                print(f"  arena.step returned {len(batch.records)} records")
                print(f"  Memory after arena.step: {get_memory_info()}")
                records_buffer.extend(batch.records)
            except Exception as e:
                print(f"  ARENA.STEP FAILED: {e}")
                traceback.print_exc()
                raise

        # Take batch_size records and train
        train_records = records_buffer[:config.batch_size]
        records_buffer = records_buffer[config.batch_size:]

        train_batch = TrainingBatch(
            records=train_records,
            meta={},
        )

        await debug_train_step(trainer, train_batch, step + 1)

    await client.close()
    print(f"\nDebug complete! Final memory: {get_memory_info()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug train debate")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/eligottlieb/.lmstudio/models/mlx-community/Trinity-Nano-Preview-8bit",
        help="Path to the base model",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--micro-token-budget", type=int, default=2048)
    parser.add_argument("--use-importance-sampling", action="store_true")

    args = parser.parse_args()
    asyncio.run(main(args))
