#!/usr/bin/env python3
"""
Run a proposer/solver self-play training session.

The proposer generates math questions, and solvers attempt to answer them.
The proposer is rewarded based on how well the solvers perform (targeting ~50% pass rate).

Usage:
    # With local server (mlx-vllm):
    python examples/run_proposer_solver.py --local

    # With default seed questions (OpenRouter):
    python examples/run_proposer_solver.py

    # With custom seed questions (JSON format):
    python examples/run_proposer_solver.py --questions '[{"question": "What is 5+5?", "ground_truth": "10"}]'

    # With questions from file (JSON array):
    python examples/run_proposer_solver.py --questions-file questions.json

    # With verbose logging:
    python examples/run_proposer_solver.py --verbose

Environment:
    OPENROUTER_API_KEY: Your OpenRouter API key (for remote APIs)
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from self_play.core.clients import OpenAIClient
from self_play.core.logging import VerboseLogger
from self_play.examples.proposer_solver import create_proposer_solver_arena


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run proposer/solver self-play training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--questions",
        type=str,
        help="Seed questions as JSON array",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="File with seed questions (JSON array)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Number of training steps (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Episodes per step (default: 3)",
    )
    parser.add_argument(
        "--solver-rollouts",
        type=int,
        default=4,
        help="Solver attempts per question (default: 4)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model to use (default: openai/gpt-4o-mini)",
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
        default=Path("proposer_solver_run.log"),
        help="Log file path (default: proposer_solver_run.log)",
    )
    return parser.parse_args()


def load_questions(args) -> list[dict] | None:
    """Load seed questions from args or file."""
    questions = None

    if args.questions:
        try:
            questions = json.loads(args.questions)
        except json.JSONDecodeError as e:
            print(f"Error parsing questions JSON: {e}")
            sys.exit(1)

    if args.questions_file:
        if not args.questions_file.exists():
            print(f"Error: Questions file not found: {args.questions_file}")
            sys.exit(1)
        with open(args.questions_file) as f:
            file_questions = json.load(f)
            if questions:
                questions.extend(file_questions)
            else:
                questions = file_questions

    return questions


async def run_proposer_solver(args):
    """Run the proposer/solver training loop."""
    questions = load_questions(args)

    # Initialize client
    if args.local:
        client = OpenAIClient.for_local(port=args.port, timeout=120.0)
        model_name = f"local (port {args.port})"
    else:
        client = OpenAIClient.for_openrouter(api_key=args.api_key, model=args.model)
        model_name = args.model

    print(f"Starting proposer/solver training")
    print(f"  model: {model_name}")
    print(f"  seed_questions: {len(questions) if questions else 'default (3)'}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  solver_rollouts: {args.solver_rollouts}")
    print()

    # Create arena
    arena = create_proposer_solver_arena(
        client=client,
        initial_questions=questions,
        n_solver_rollouts=args.solver_rollouts,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Run with or without verbose logging
    if args.verbose:
        await run_with_logging(arena, args, questions)
    else:
        await run_simple(arena, args)

    # Cleanup
    await client.close()
    print(f"\nTotal API calls: {client.call_count}")

    # Show generated questions
    if "questions" in arena.stores:
        print(f"\nQuestions in store: {arena.stores['questions'].count()}")


async def run_simple(arena, args):
    """Run without verbose logging."""
    step_num = 0
    async for batch in arena.run(num_steps=args.num_steps, verbose=True):
        step_num += 1
        print(f"Step {step_num}: {batch.meta['num_records']} records")

        # Group by role
        proposer_rewards = [r.reward for r in batch.records if r.role_id == "Proposer"]
        solver_rewards = [r.reward for r in batch.records if r.role_id == "Solver"]

        if proposer_rewards:
            avg_p = sum(proposer_rewards) / len(proposer_rewards)
            print(f"  Proposer: {len(proposer_rewards)} records, avg={avg_p:.4f}")
        if solver_rewards:
            avg_s = sum(solver_rewards) / len(solver_rewards)
            print(f"  Solver: {len(solver_rewards)} records, avg={avg_s:.4f}")


async def run_with_logging(arena, args, questions):
    """Run with verbose logging to file."""
    print(f"Verbose logging to: {args.log_file}")

    with VerboseLogger(args.log_file) as logger:
        # Log config
        logger.log_run_start({
            "mode": "proposer_solver",
            "model": args.model,
            "num_seed_questions": len(questions) if questions else 3,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "solver_rollouts": args.solver_rollouts,
        })

        # Log seed questions
        if questions:
            logger.log("Seed questions:")
            for i, q in enumerate(questions[:5]):
                logger.log(f"  [{i}] {q}")

        step_num = 0
        async for batch in arena.run(num_steps=args.num_steps, verbose=True):
            step_num += 1

            # Console output
            print(f"Step {step_num}: {batch.meta['num_records']} records")

            # Detailed logging
            logger.log_step_start(step_num, batch.meta.get("num_results", 0))

            # Group records by episode type
            proposer_records = [r for r in batch.records if r.role_id == "Proposer"]
            solver_records = [r for r in batch.records if r.role_id == "Solver"]

            if proposer_records:
                avg_p = sum(r.reward for r in proposer_records) / len(proposer_records)
                logger.log(f"Proposer: {len(proposer_records)} records, avg_reward={avg_p:.4f}")
                for r in proposer_records:
                    logger.log(f"  [{r.rollout_id[:8]}] reward={r.reward:.4f}")

            if solver_records:
                avg_s = sum(r.reward for r in solver_records) / len(solver_records)
                correct = sum(1 for r in solver_records if r.reward > 0.5)
                logger.log(f"Solver: {len(solver_records)} records, avg_reward={avg_s:.4f}, correct={correct}/{len(solver_records)}")

            logger.log_step_end(step_num, batch.meta)

        # Log final state
        if "questions" in arena.stores:
            count = arena.stores["questions"].count()
            logger.log(f"Final question store size: {count}")

        logger.log(f"Training complete. Total steps: {step_num}")


def main():
    args = parse_args()
    try:
        asyncio.run(run_proposer_solver(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
