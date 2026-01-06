# Self-Play Engine

**Composable primitives that turn any multi-agent interaction into a self-improving training loop.**

Debate, curriculum generation, adversarial games, verification cascades—all expressible in 50 lines of Python.

```
pip install self-play-engine
```

---

## The Core Insight

Self-play isn't a technique—it's a paradigm. Every training setup that can be expressed as:

1. Agents interacting (even if they're the same model with different prompts)
2. Producing outcomes that can be scored
3. With credit assigned back to individual actions

...can be implemented with five abstractions: **Role**, **Episode**, **Rubric**, **CreditAssigner**, and **Arena**.

This framework makes self-play compositional.

---

## Quick Start

```bash
# Start the inference server with LoRA hot-swap
python -m self_play.inference.server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --lora-rank 8

# Train debate agents (different terminal)
python examples/train_debate.py --num-steps 100 --batch-size 4
```

That's it. Two debaters argue topics, an LLM judge scores transcripts, advantages are computed via GRPO, and LoRA weights update in real-time.

---

## What Makes This Different

### 1. Hierarchical Episodes = Monte Carlo Scoring for Free

Episodes can spawn sub-episodes. Credit flows through the tree.

```python
class ProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        # Generate a question
        response = await self.call_model("Proposer", prompt, arena)
        question = parse_question(response.text)

        # Spawn N solver attempts (Monte Carlo)
        requests = [EpisodeRequest("solve", question) for _ in range(8)]
        results = await arena.generate_rollouts(requests)
        state.child_results = results

        return state

    def get_extras(self, state):
        # Pass rate exposed to rubric
        wins = sum(1 for r in state.child_results if r.rewards.get("Solver", 0) > 0)
        return {"pass_rate": wins / len(state.child_results)}
```

The Proposer's reward is based on Solver pass rate. **Target 50% difficulty**—if problems are too easy or too hard, the Proposer gets negative reward. The model learns to generate problems at the edge of its own capability.

**Novel setups this enables:**
- **Self-Evolving Datasets**: Generate training data, filter by Monte Carlo quality, train on survivors
- **Difficulty-Aware Fine-Tuning**: Generate problems at the edge of capability, train on near-misses
- **Code Generation with Execution Verification**: Generator writes code. N test runs execute it. Pass rate = reward

### 2. Credit Assignment That Respects Hierarchy

GRPO with per-level grouping solves a subtle problem: in hierarchical episodes, you compare apples to apples.

```
Top-level (Proposer): compared against other Proposers in batch
  └── Children (Solvers): compared against sibling Solvers under same parent
```

A Solver's advantage is relative to its siblings—not to Solvers under a different Proposer. The credit signal is clean.

This enables:
- **Nested Games**: A meta-agent that spawns games doesn't get credit noise from inner games
- **Multi-Objective Training**: Different roles, different rewards, all in one forward pass

### 3. LLM-as-Judge Without Reward Hacking

Judges call `arena.call_model()` without a role_id—they're oracles, not participants. Their completions never enter the training batch.

```python
async def llm_judge(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    # Judge is NOT a role - no training data produced
    response = await arena.call_model(judge_prompt, temperature=0.3)
    winner = parse_winner(response.text)
    return {winner: 1.0, loser: -1.0}
```

But they can be the same underlying model. This creates **self-consistency pressure via internal judging**—the model learns to produce outputs it would judge as good. Constitutional AI, but emergent rather than handcrafted.

### 4. Composable Episode Trees = Arbitrary Training Pipelines

The `GenerateResult.children` field is deceptively powerful:

```python
class TournamentEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        # Round 1: generate candidates
        candidates = await arena.generate_rollouts([
            EpisodeRequest("proposal", {}) for _ in range(8)
        ])

        # Round 2: head-to-head debates
        for a, b in itertools.combinations(candidates, 2):
            battle = await arena.run_episode("debate", {"a": a, "b": b})
            state.child_results.append(battle)

        # Elo-like scoring from battle outcomes
        return state
```

**Novel setups this enables:**
- **Tournament Selection**: N proposals compete head-to-head. Winners propagate. Evolution without explicit mutation
- **Multi-Stage Pipelines**: Plan → Execute → Critique → Revise. Each stage is a role. Each produces training data
- **Ensemble Verification**: Multiple solvers attempt. Majority vote = ground truth. Outliers get negative reward

---

## Core Abstractions

| Abstraction | Purpose |
|-------------|---------|
| **Role** | Trainable entity with system prompt, temperature, token limits |
| **Episode** | Rollout protocol—defines how agents interact |
| **Rubric** | Composable reward functions (sync or async) |
| **CreditAssigner** | Computes advantages from rewards (GRPO, RAE, constant) |
| **Arena** | Orchestrates everything—registration, batching, rollouts, training |

```python
# Register roles
arena.add_role(Role(id="Attacker", system_prompt="Find vulnerabilities...", temperature=0.9))
arena.add_role(Role(id="Defender", system_prompt="Patch the code...", temperature=0.7))

# Register episode
arena.add_episode("redblue", RedBlueEpisode())

# Run training
await simple_training_loop(arena, trainer, num_steps=1000)
```

---

## Built-in Task Types

| Task | Pattern | Reward |
|------|---------|--------|
| **Debate** | Two agents argue, judge picks winner | Zero-sum (+1/-1) |
| **Proposer/Solver** | Proposer generates questions, Solvers attempt | Goldilocks (pass_rate → 0.5) |
| **Negotiation** | Two players trade resources with hidden preferences | Inventory value change |
| **RedBlue** | Attacker crafts adversarial input, Defender completes task | Zero-sum (attack success) |
| **RefineLoop** | Generator → Critic → Revise → repeat | LLM quality score |
| **EloArena** | N agents, round-robin matches, Elo ratings | Relative skill |
| **DatasetRL** | Standard RLHF on HuggingFace datasets | Configurable |

Each is ~100 lines. Extend `Episode` to build your own.

---

## Experiments This Enables

| Experiment | Episode Structure | Rubric | What You Learn |
|------------|-------------------|--------|----------------|
| Constitutional AI via Debate | Two agents argue whether response violates principles | Judge picks winner | Alignment without RLHF |
| Self-Distillation | Generator → Verifier → Filter | Pass/fail on verification | High-quality data from noise |
| Proof Verification | Prover proposes proof → Verifier checks steps | Valid/invalid | Mathematical reasoning |
| Multi-Agent Negotiation | N agents with hidden objectives negotiate | Pareto optimality | Emergent cooperation |
| Iterated Amplification | Human-in-the-loop as sub-episode | Human rating | Scalable oversight |
| Automated Red Teaming | Attacker finds vulnerabilities → Defender patches | Attack success rate | Robustness + attack capability |

---

## Architecture Highlights

**Designed for Apple Silicon.** MLX inference with LoRA hot-swap. Weights update on the server without restart.

**Fully async.** Generation and training run concurrently. Semaphore-bounded parallelism. Embarrassingly parallel rollouts are exploited.

**Policy versioning.** Rollouts tagged with version via `get_policy_version()`. Stale records dropped. PPO-style clipping against off-policy data.

**Clean separation:**
- `EpisodeRequest.is_trainable` — spawn evaluation rollouts without polluting training batch
- `get_extras()` hook — episodes expose arbitrary data to rubrics
- `ArtifactStore` with weighted sampling — curriculum learning out of the box

---

## Installation

```bash
# Requires Python 3.12+
pip install self-play-engine

# Or from source
git clone https://github.com/your-org/self-play-engine
cd self-play-engine
pip install -e .
```

**Dependencies:** MLX, mlx-lm, FastAPI, httpx, transformers, wandb

---

## Project Structure

```
src/self_play/
├── core/
│   ├── types.py       # Role, Step, Rollout, TrainingRecord
│   ├── episode.py     # Episode, EpisodeState, ChatEpisode
│   ├── rubric.py      # Rubric, RewardFn
│   ├── credit.py      # GRPOCredit, RAECredit, ConstantCredit
│   ├── arena.py       # Arena, ArtifactStore
│   └── clients.py     # OpenAIClient
├── tasks/
│   ├── debate.py
│   ├── proposer_solver.py
│   ├── negotiation.py
│   ├── redblue.py
│   ├── refine_loop.py
│   └── elo_arena.py
├── training/
│   ├── trainer.py     # MLX training with micro-batching
│   ├── loop.py        # Async training loop
│   └── weight_publisher.py  # LoRA hot-swap
└── inference/
    ├── server.py      # FastAPI + OpenAI-compatible API
    └── engine/        # Async batched generation
```

---

## Running Examples

```bash
# Debate training
python examples/train_debate.py \
    --model-path mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --num-steps 100 \
    --batch-size 4 \
    --wandb-project my-debate-run

# Proposer/Solver with curriculum
python examples/train_proposer_solver.py \
    --n-solver-rollouts 8 \
    --num-steps 200

# Adversarial red/blue
python examples/train_redblue.py \
    --batch-size 8 \
    --verbose

# Multi-agent negotiation
python examples/train_negotiation.py \
    --max-turns 10
```

---

## The Pitch in One Sentence

You've written a 50-line Episode subclass. Now you have a self-improving training loop that generates its own curriculum, scores via Monte Carlo, and trains both sides of an adversarial game—all without human-labeled data.

**This is infrastructure-level work. The experiments it enables are the product.**

---

## License

MIT
