"""
Example: RefineLoop - Generator-Critic iterative refinement.

This script trains a Generator and Critic to collaboratively improve content.
The Generator creates drafts, the Critic provides feedback, and this iterates.

To run:
1. Start mlx-vllm server:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Run this script:
   python examples/train_refine_loop.py --verbose
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient, Role, Artifact, GRPOCredit
from self_play.tasks.refine_loop import RefineLoopArena, RefineLoopEpisode, TaskProposerEpisode
from self_play.training import (
    Trainer,
    TrainerConfig,
    WeightPublisher,
    training_loop,
    simple_training_loop,
)


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_refine_loop(arena, concurrency: int = 4):
    """Preview arena output with truncated drafts/critiques."""
    print("\n=== RefineLoop Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    # Group by rollout
    rollouts = {}
    for r in batch.records:
        if r.rollout_id not in rollouts:
            rollouts[r.rollout_id] = {"meta": r.meta, "reward": r.reward}

    print(f"Episodes: {len(rollouts)} | Records: {len(batch.records)}\n")

    for i, (rid, data) in enumerate(rollouts.items(), 1):
        meta = data["meta"]
        extras = meta.get("extras", {})
        artifact = meta.get("artifact", {}) or {}

        print(f"[{i}] {meta.get('episode_type', '?')} | reward={data['reward']:.2f}")
        print(f"    task: {truncate(artifact.get('task', 'N/A'), 60)}")

        drafts = extras.get("drafts", [])
        feedback = extras.get("feedback", [])
        for j, draft in enumerate(drafts):
            print(f"    draft{j+1}: {truncate(draft, 10000)}")
        for j, fb in enumerate(feedback):
            print(f"    critique{j+1}: {truncate(fb, 10000)}")
        print()

    # Summary
    qualities = [d["reward"] for d in rollouts.values()]
    if qualities:
        print(f"Avg quality: {sum(qualities)/len(qualities):.2f} | Range: {min(qualities):.2f}-{max(qualities):.2f}")


# Creative writing tasks for refinement
TASKS = [
    {
        "task": "Write a poem about programming",
        "requirements": "Should capture the essence of coding.",
    },
    {
        "task": "Write a product description for a smart water bottle",
        "requirements": "150 words max. Highlight 3 key features. Use persuasive language.",
    },
    {
        "task": "Write an opening paragraph for a mystery novel",
        "requirements": "Set the scene, introduce tension, hook the reader. 100 words.",
    },
    {
        "task": "Write a professional email declining a meeting",
        "requirements": "Polite but firm. Suggest an alternative. Keep it brief.",
    },
    {
        "task": "Write a tweet announcing a new feature",
        "requirements": "Max 280 characters. Include call to action. Be engaging.",
    },
    {
        "task": "Write a short bio for a software engineer",
        "requirements": "Third person. 50 words. Professional but personable.",
    },
    {
        "task": "Write an apology for a service outage",
        "requirements": "Acknowledge the issue, explain briefly, commit to improvement.",
    },
]


GENERATOR_SYSTEM_PROMPT = """You are a skilled writer. Create high-quality content based on the given task.
When receiving feedback, thoughtfully revise your work to address the suggestions.
Focus on clarity, creativity, and meeting all requirements.
After you receive feedback, do not directly respond to it. Instead, output your revised draft based on the feedback.
It is CRUCIAL that you ONLY output your revised draft based on the feedback. Do not include any other commentary."""

CRITIC_SYSTEM_PROMPT = """You are a constructive critic. Provide specific, concise feedback.
Focus on: clarity, meeting requirements, style, and impact.
ONLY PROVIDE CONCISE FEEDBACK. This feedback MUST NOT include a suggestion for what a revised draft should look like.

For example:
If the first draft is overly verbose, you might say:  
"Your writing is too verbose. Please shorten your response."
But DO NOT then suggest a revised draft.
"""

TASK_PROPOSER_SYSTEM_PROMPT = """You are a creative writing task designer.
Generate novel, well-structured writing tasks with clear requirements.
Tasks should be challenging but achievable with iterative refinement.
Don't ask for something incredibly complex like a limerick or a haiku, or something similar that requires a very specific format."""


def load_model_with_lora(model_path: str, lora_rank: int = 16, lora_layers: int = 16):
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    # Defaults match official LiquidAI/PEFT recommendations
    lora_keys = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"}

    print(f"Attaching LoRA (rank={lora_rank}, layers={lora_layers}, keys={lora_keys})...")
    linear_to_lora_layers(model, lora_layers, {"rank": lora_rank, "scale": 32.0, "dropout": 0.05, "keys": lora_keys})

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


async def main(args):
    base_url = args.url if args.url else f"http://{args.host}:{args.port}"
    print(f"\nConnecting to {base_url}...")

    if args.url:
        client = OpenAIClient(api_key="not-needed", model="local",
                              base_url=f"{base_url.rstrip('/')}/v1", timeout=120.0)
    else:
        client = OpenAIClient.for_local(host=args.host, port=args.port, timeout=120.0)

    # Setup arena
    arena = RefineLoopArena(
        client=client,
        batch_size=args.batch_size,
        proposer_batch_size=args.proposer_batch_size,
        credit_assigner=GRPOCredit(),
        verbose=args.verbose,
    )

    arena.add_role(Role(
        id="Generator",
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    arena.add_role(Role(
        id="Critic",
        system_prompt=CRITIC_SYSTEM_PROMPT,
        temperature=0.5,  # Lower temp for more consistent feedback
        max_tokens=args.max_tokens,
    ))

    arena.add_episode("refine_loop", RefineLoopEpisode(
        generator_role="Generator",
        critic_role="Critic",
        num_iterations=args.num_iterations,
    ))

    # Task proposer (non-trainable) generates new tasks
    arena.add_role(Role(
        id="TaskProposer",
        system_prompt=TASK_PROPOSER_SYSTEM_PROMPT,
        temperature=0.9,  # Higher temp for more creative tasks
        max_tokens=args.max_tokens,
    ))

    arena.add_episode("task_propose", TaskProposerEpisode(
        proposer_role="TaskProposer",
    ))

    # Load tasks
    store = arena.add_store("tasks")
    for i, task in enumerate(TASKS):
        store.add(Artifact(id=f"task_{i}", data=task))
    print(f"Loaded {store.count()} tasks")

    # Dry-run mode: preview arena performance without training
    if args.dry_run:
        await preview_refine_loop(arena, concurrency=args.concurrency)
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
        use_kl_penalty=args.use_kl_penalty,
        loss_type=args.loss_type,
        importance_sampling=args.importance_sampling,
        weight_push_url=base_url,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=args.wandb_project,
    )

    publisher = WeightPublisher(base_url=base_url)
    trainer = Trainer(model=model, optimizer=optimizer, config=config, publisher=publisher)

    print(f"\nStarting RefineLoop training for {args.num_steps} steps...")
    print(f"  - {args.num_iterations} refinement iterations per episode")
    print(f"  - Dynamic task generation ({args.proposer_batch_size} per batch)")

    try:
        if args.simple_loop:
            await simple_training_loop(
                arena=arena,
                trainer=trainer,
                num_steps=args.num_steps,
                concurrency=args.concurrency,
                step_concurrency=args.step_concurrency,
                verbose=args.verbose,
            )
        else:
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
        await publisher.close()
        await client.close()

    print(f"\nTraining complete! Step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefineLoop: Generator-Critic iterative refinement")
    parser.add_argument("--model-path", 
        # default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit",
        # default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen2.5-1.5B-Instruct-MLX-8bit",
        default="/Users/eligottlieb/.lmstudio/models/LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit",
    help="Path to the model to train on")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-iterations", type=int, default=2, help="Refinement iterations per episode")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--proposer-batch-size", type=int, default=4, help="Number of new tasks to generate per batch")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--micro-token-budget", type=int, default=2048)
    parser.add_argument("--max-policy-lag", type=int, default=3)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--use-kl-penalty", action="store_true",
        help="Add KL penalty to loss for regularization toward reference policy")
    parser.add_argument("--loss-type", type=str, default="token", choices=["token", "sample"],
        help="Loss normalization: 'token' (DAPO) or 'sample' (GRPO)")
    parser.add_argument("--importance-sampling", type=str, default="token", choices=["token", "sequence"],
        help="Importance sampling level: 'token' (GRPO/DAPO) or 'sequence' (GSPO)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--step-concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--simple-loop", action="store_true",
        help="Use simple sequential loop instead of async")
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    asyncio.run(main(parser.parse_args()))
