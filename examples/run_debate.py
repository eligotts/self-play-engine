#!/usr/bin/env python3
"""
Run a debate self-play training session.

Usage:
    # With local server (mlx-vllm):
    python examples/run_debate.py --local "AI is beneficial"

    # With OpenRouter:
    python examples/run_debate.py "AI is beneficial" "Climate change is urgent"

    # With topics from file (one per line):
    python examples/run_debate.py --topics-file topics.txt

    # With verbose logging:
    python examples/run_debate.py --verbose "AI is beneficial"

    # With custom model (OpenRouter):
    python examples/run_debate.py --model anthropic/claude-3-haiku "AI is beneficial"

Environment:
    OPENROUTER_API_KEY: Your OpenRouter API key (for remote APIs)
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from self_play.core.clients import OpenAIClient
from self_play.core.logging import VerboseLogger
from self_play.examples.debate import create_debate_arena


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run debate self-play training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "topics",
        nargs="*",
        help="Debate topics (or use --topics-file)",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        help="File with topics (one per line)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Number of training steps (default: 3)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=2,
        help="Rounds per debate (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Episodes per step (default: 2)",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-3-flash-preview",
        help="Model to use (default: google/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--api-key",
        help="API key (or set OPENROUTER_API_KEY / OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local mlx-vllm server at localhost:8000",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Local server port (default: 8000)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging to file",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("debate_run.log"),
        help="Log file path (default: debate_run.log)",
    )
    return parser.parse_args()


def load_topics(args) -> list[str]:
    """Load topics from args or file."""
    topics = list(args.topics) if args.topics else []

    if args.topics_file:
        if not args.topics_file.exists():
            print(f"Error: Topics file not found: {args.topics_file}")
            sys.exit(1)
        with open(args.topics_file) as f:
            file_topics = [line.strip() for line in f if line.strip()]
            topics.extend(file_topics)

    if not topics:
        # Default topics for testing
        topics = [
            "Artificial intelligence will do more good than harm for humanity",
            "Remote work is better than office work",
            "Social media has a net negative effect on society",
            "AI will replace all jobs",
            "Abortion should be legal",
            "The death penalty should be abolished",
            "The earth is flat"
        ]
        print(f"No topics provided, using defaults: {topics}")

    return topics


async def run_debate(args):
    """Run the debate training loop."""
    topics = load_topics(args)

    # Initialize client
    if args.local:
        client = OpenAIClient.for_local(port=args.port, timeout=240.0)
        model_name = f"local (port {args.port})"
    else:
        client = OpenAIClient.for_openrouter(api_key=args.api_key, model=args.model)
        model_name = args.model

    print(f"Starting debate training")
    print(f"  model: {model_name}")
    print(f"  topics: {len(topics)}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  num_rounds: {args.num_rounds}")
    print(f"  batch_size: {args.batch_size}")
    print()

    # Create arena
    arena = create_debate_arena(
        client=client,
        topics=topics,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Run with or without verbose logging
    if args.verbose:
        await run_with_logging(arena, args, topics)
    else:
        await run_simple(arena, args)

    # Cleanup
    await client.close()
    print(f"\nTotal API calls: {client.call_count}")


async def run_simple(arena, args):
    """Run without verbose logging."""
    step_num = 0
    async for batch in arena.run(num_steps=args.num_steps, verbose=True):
        step_num += 1
        print(f"Step {step_num}: {batch.meta['num_records']} records")

        # Print per-record summary
        for record in batch.records:
            print(f"  {record.role_id}: reward={record.reward:.4f}")


async def run_with_logging(arena, args, topics):
    """Run with verbose logging to file."""
    print(f"Verbose logging to: {args.log_file}")

    with VerboseLogger(args.log_file) as logger:
        # Log config
        logger.log_run_start({
            "model": args.model,
            "num_topics": len(topics),
            "topics": topics[:5],  # First 5 for brevity
            "num_steps": args.num_steps,
            "num_rounds": args.num_rounds,
            "batch_size": args.batch_size,
        })

        step_num = 0
        async for batch in arena.run(num_steps=args.num_steps, verbose=True):
            step_num += 1

            # Console output
            print(f"Step {step_num}: {batch.meta['num_records']} records")

            # Detailed logging
            logger.log_step_start(step_num, batch.meta.get("num_results", 0))
            logger.log_training_records(batch.records)
            logger.log_step_end(step_num, batch.meta)

            # Log individual records
            for record in batch.records:
                logger.log(f"  {record.role_id} [{record.rollout_id[:8]}]: reward={record.reward:.4f}")

        logger.log(f"Training complete. Total steps: {step_num}")


def main():
    args = parse_args()
    try:
        asyncio.run(run_debate(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
