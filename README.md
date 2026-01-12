# Legos

**Composable primitives that turn any multi-agent interaction into a self-improving training loop.**

Debate, curriculum generation, adversarial games, verification cascades—all expressible in the same 50-line Episode subclass.

---

## The Insight

Self-play isn't a technique—it's a paradigm. This framework makes it compositional.

Every training setup that can be expressed as:

1. Agents interacting (even if they're the same model with different prompts)
2. Producing outcomes that can be scored
3. With credit assigned back to individual actions

...can be implemented with five abstractions: **Actor**, **Episode**, **Rubric**, **CreditAssigner**, and **Arena**.

You've written a 50-line Episode subclass. Now you have a self-improving training loop that generates its own curriculum, scores via Monte Carlo, and trains both sides of an adversarial game—all without human-labeled data.

---

## What This Enables

| Experiment | Episode Structure | Rubric | What You Learn |
|------------|-------------------|--------|----------------|
| Constitutional AI via Debate | Two agents argue whether response violates principles | Judge picks winner | Alignment without RLHF |
| Self-Distillation | Generator → Verifier → Filter | Pass/fail on verification | High-quality data from noise |
| Proof Verification | Prover proposes proof → Verifier checks steps | Valid/invalid | Mathematical reasoning |
| Multi-Agent Negotiation | N agents with hidden objectives negotiate | Pareto optimality | Emergent cooperation |
| Automated Red Teaming | Attacker finds vulnerabilities → Defender patches | Attack success rate | Robustness + attack capability |
| Difficulty-Aware Curriculum | Proposer generates problems → Solvers attempt | Target 50% pass rate | Training at capability frontier |

---

## Quick Start

```bash
# Terminal 1: Start the inference server with LoRA hot-swap
uv run legos serve \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit

# Terminal 2: Run training
uv run examples/train_gsm8k.py
```

Weights update in real-time. Watch the model improve on math problems as you train.

---

## Core Abstractions

### Actor

Trainable entities. Same underlying model, different system prompts and instructions.

```python
arena.add_actor(Actor(
    id="Proposer",
    system_prompt="Generate challenging math problems...",
    temperature=0.9
))
arena.add_actor(Actor(
    id="Solver",
    system_prompt="Solve math problems step by step...",
    temperature=0.7
))
```

Two actors, one policy. The Proposer learns to generate problems at the edge of the Solver's capability. The Solver learns to solve them.

### Episode

Self-contained rollouts. Optionally seeded with data, produce steps, end with rewards. Episodes are stateless—all training-relevant information flows through the `Rollout`.

The key insight: **episodes can nest within each other**.

```python
class ProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        # Generate a question
        response = await self.call_model("Proposer", prompt, arena)
        question = parse_question(response.text)

        # Spawn N solver attempts (Monte Carlo)
        requests = [EpisodeRequest("solve", question, is_trainable=False) for _ in range(8)]
        results = await arena.generate_rollouts(requests)
        state.child_results = results

        return state

    def get_extras(self, state):
        # Expose pass rate to rubric
        wins = sum(1 for r in state.child_results if r.rewards.get("Solver", 0) > 0)
        return {"pass_rate": wins / len(state.child_results)}
```

The `is_trainable=False` flag means solver completions inform the proposer's reward but don't enter the training batch themselves.

### Rubric

Composable reward functions. Sync or async. Can use LLM-as-judge.

```python
def goldilocks_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """Reward proposer for 50% difficulty—not too easy, not too hard."""
    pass_rate = rollout.extras["pass_rate"]
    reward = 1.0 - 2 * abs(pass_rate - 0.5)
    return {"Proposer": reward}

rubric = Rubric([goldilocks_reward])
```

Judges call `arena.call_model()` without an actor ID—they're oracles, not participants. Their completions never enter the training batch, preventing reward hacking.

### CreditAssigner

Computes advantages from rewards. Respects hierarchy—compare apples to apples.

```
Top-level (Proposer): compared against other Proposers in batch
  └── Children (Solvers): compared against sibling Solvers under same parent
```

Built-in options:
- **GRPOCredit**: Group-relative advantages, hierarchy-aware
- **RAECredit**: Actor-conditioned EMA baselines (from SPIRAL)
- **ConstantCredit**: All steps get weight 1.0

### Arena

Orchestrates everything. Registers actors, schedules episodes, manages artifact stores.

```python
class MyArena(Arena):
    def get_batch(self) -> List[EpisodeRequest]:
        # Define your training distribution
        samples = self.stores["questions"].sample(k=self.batch_size)
        return [EpisodeRequest("solve", s.data) for s in samples]
```

**ArtifactStore** enables curriculum learning out of the box. Proposer generates questions → stored in artifact store → sampled for future training.

[See docs/concepts.md for deep dive →](docs/concepts.md)

---

## Training Infrastructure

Built on MLX for Apple Silicon. Async by design.

- **Producer-consumer architecture**: Generation and training run concurrently
- **Importance sampling**: Token-level (PPO-style) or sequence-level (GSPO)
- **Staleness filtering**: Records tagged with policy version, stale data dropped
- **LoRA hot-swap**: Weights update on inference server without restart
- **Continuous batching**: Efficient inference with dynamic batch sizes

[See docs/training.md for details →](docs/training.md)

---

## Built-in Examples

| Task | Pattern | What It Demonstrates |
|------|---------|---------------------|
| **GSM8K** | Single actor, GRPO | Basic reward shaping, answer verification |
| **Negotiation** | Two-player game | Multi-agent state, RAE credit assignment |
| **Proposer/Solver** | Nested episodes | Monte Carlo scoring, curriculum generation |
| **RefineLoop** | Cooperative multi-turn | Generator → Critic cycles, external LLM judge |
| **HeadToHead** | Self-play tournament | Same-model competition, zero-sum rewards |
| **SPICE** | Corpus-grounded | External knowledge, trainable nested episodes |

Each is ~100-150 lines. Extend `Episode` to build your own.

[See docs/examples.md for walkthroughs →](docs/examples.md)

---

## Installation

```bash
# Requires Python 3.12+, Apple Silicon recommended
pip install legos

# Or from source
git clone https://github.com/eligottlieb/legos
cd legos
pip install -e .
```

**Dependencies**: MLX, mlx-lm, FastAPI, httpx, transformers, wandb

---

## Project Structure

```
src/legos/
├── core/
│   ├── types.py       # Actor, Step, Rollout, TrainingRecord
│   ├── episode.py     # Episode, EpisodeState, ChatEpisode
│   ├── rubric.py      # Rubric, RewardFn
│   ├── credit.py      # GRPOCredit, RAECredit
│   ├── arena.py       # Arena, ArtifactStore
│   └── clients.py     # OpenAIClient
├── tasks/             # Built-in episode implementations
├── training/          # Trainer, loss functions, async loop
└── inference/         # FastAPI server, continuous batching
```

---
