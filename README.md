# LEGOS

**Learning through Evolutionary Game Optimization Systems**

This project was borne out of my goal to build a library that 1) treats the patterns of LLM self-play as first class entities, and 2) uses the latest best practices in reinforcement learning training optimized for Apple Silicon using MLX.

---

## The Self-Play Thesis

As model capabilities increase, self-play becomes a viable method for improving model performance in certain domains. [Absolute Zero](https://arxiv.org/abs/2505.03335) (Zhao et al., 2025) showed this back in March 2025; [SPIRAL](https://arxiv.org/abs/2506.24119) (Liu et al., 2025) applied this concept to zero sum games; [SPICE](https://arxiv.org/abs/2510.24684) (Liu et al., 2025) added a creative external signal to drive training. See more thoughts of mine [here](https://eligottlieb1.substack.com/p/the-independent-pursuit-of-intelligence).

When you look deeper at these various self-play setups, common patterns emerge. This project serves as a first attempt to capture those patterns, abstracting away the orchestration details and leaving only the experiment specifics to be implemented.

## Why MLX?

After passively riding the RL hype train this past year, which manifested in way too many hours of TPOT doomscrolling plus adding a few environments to [PI's Environments Hub](https://app.primeintellect.ai/dashboard/environments/ergotts/socratic-method), it felt like there was no better way to consolidate everything I'd been learning than just building a training library from scratch (whatever "from scratch" means these days). And why MLX? Feels like a growing community that is lacking an RL library like this, so maybe this could actually make an impact? Also, so I could finally make use of the somewhat rash decision to max out my macbook air memory.

And yes, LoRA only. Easy, simple, and all I really wanted to handle for now.

---

## Quick Start

```bash
# Terminal 1: Start the inference server with LoRA hot-swap
legos serve \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit

# Terminal 2: Run training
uv run examples/train_gsm8k.py
```

Weights update in real-time. Watch the model improve on math problems while answers become more concise.

---

## Self-Play Abstractions

Self-play training doesn't fit into the standard training framework where you have a dataset, take a chunk of that dataset, generate rollouts, send back to trainer. So here's what I came up with:

### Actors

These are our trainable entities. Single policy is currently all that is supported, so you can think of them as entities that receive different system prompts and instructions. For example, if our setup is training a model to play against itself in a two player zero sum game (SPIRAL), we'd have two actors. In this case they both might have the same system prompt but receive different instructions at different stages of the game. Or consider a setup where a model proposes its own training data then trains on it (Absolute Zero). These are two distinct actors with different system prompts that receive different instructions to do different things.

```python
arena.add_actor(Actor(
    id="Proposer",
    system_prompt="Generate challenging math problems...",
))
arena.add_actor(Actor(
    id="Solver",
    system_prompt="Solve math problems step by step...",
))
```

### Episodes

Actors are composed together in episodes. I think of episodes as self-contained "rollouts" - they can optionally be seeded with some piece of data, involve some sort of model call, and finish with a reward for that episode. Episodes are self-contained in the sense that they don't carry state beyond the execution of the episode itself. In the two-player game example, the episode would be a full run from game initialization to completion, at which point a reward would be assigned to each player.

```python
class NegotiationEpisode(MultiTurnEpisode):
    async def rollout(self, arena, artifact, state):
        # Game logic here - trading, offers, accepts/denies
        # Returns state with trajectory of all moves
        ...
```

**Episodes can be nested within each other.** Consider the setup where a model proposes its own training data then trains on it (Absolute Zero). If we want to assign a reward to the proposer actor for how "good" its question is by running Monte Carlo rollouts on that proposed question (maybe maximize its reward when the solver pass rate is 50%), we'd want to nest the solver episode within the proposer episode - in order to reward the proposer we need to know how well the solver did.

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
```

The `is_trainable=False` flag means solver completions inform the proposer's reward but don't enter the training batch themselves.

### Rubrics

You can register reward functions within an episode through Rubrics. These are how episodes are scored. Once an episode run completes, the reward functions will be run against it to assign a reward to each actor in the episode.

```python
def goldilocks_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """Reward proposer for 50% difficulty - not too easy, not too hard."""
    pass_rate = rollout.extras["pass_rate"]
    reward = 1.0 - 2 * abs(pass_rate - 0.5)
    return {"Proposer": reward}

rubric = Rubric([goldilocks_reward])
```

Rubrics can also use LLM-as-judge by calling `arena.call_model()` without an actor ID - judge completions never enter the training batch (or just calling an external API to score the episode).

### Arenas

Arenas are the stateful containers that launch episodes. In an arena you define how episodes are scheduled. Arenas can optionally contain artifact stores, which are persistent across the full training lifecycle and are places where you can store data that can be used as inputs or outputs from episodes.

For example, in proposer/solver, the proposer episode would generate a question and which is then stored in the question artifact store. If we want to launch solver episodes independent of the Monte Carlo rollouts that are run to score a proposed question, we'd just pull from this question artifact store.

```python
class MyArena(Arena):
    def get_batch(self) -> List[EpisodeRequest]:
        samples = self.stores["questions"].sample(k=self.batch_size)
        return [EpisodeRequest("solve", s.data) for s in samples]
```

### Credit Assignment

How do we turn rewards into advantages? Built-in options:

- **GRPOCredit**: Group-relative advantages, hierarchy-aware
- **RAECredit**: Actor-conditioned EMA baselines (from SPIRAL) - useful when actors have structural advantages like first-mover
- **ConstantCredit**: All steps get weight 1.0

[See docs/concepts.md for the deep dive](docs/concepts.md)

---

## MLX Trainer

Complete separation between how data is generated and how it's trained on. Asynchronous streaming setup.

Inspired by [Ludic](https://github.com/hallerite/ludic), we use token count to define micro batches. Once we've processed enough samples across all microbatches (sample = one LLM completion, so a multiturn episode might produce multiple samples), we step our optimizer and update LoRA weights. This happens in a continuous producer-consumer loop: the arena generates rollouts, the trainer pulls them in, and stale samples (generated from an old policy) get dropped.

We implement [PipelineRL](https://arxiv.org/abs/2509.19128) (Piché et al., 2025) style training where updated weights are sent to the inference server mid-generation. The inference server runs continuous batching with LoRA hot-swap - it can accept new adapter weights between generation steps without interrupting in-flight requests. 

### Features

Implemented best practices to consolidate my learnings (shoutout to [Nathan Lambert's deep dive on post-training](https://www.youtube.com/watch?v=uaZ3yRdYg8A&t=1584s)):

- **Zero gradient filtering**: Don't include samples with zero advantage ([`arena.py:239`](src/legos/core/arena.py#L239))
- **Active sampling**: Continue to pull data until we have enough samples to step ([`trainer.py:404`](src/legos/training/trainer.py#L404))
- **Token vs sample level loss**: Default is token (DAPO), option for sample (GRPO)
- **KL penalty**: K1 advantage shaping (subtracted from advantage, per [Shah et al., 2025](https://arxiv.org/abs/2512.21852) - not exact implementation, but similar)
- **Truncated importance sampling**: PPO clipping (default 0.8-1.2)
- **Sequence vs token level importance sampling**: Option for GSPO-style sequence-level
- **Stability safeguards**: Skip micro-batches where clip percentage is too high

[See docs/training.md for details](docs/training.md)

---

## Examples

| Task | Pattern | What It Demonstrates |
|------|---------|---------------------|
| **GSM8K** | Single actor, GRPO | Basic reward shaping, answer verification |
| **Negotiation** | Two-player game | Multi-agent state, RAE credit assignment |
| **Proposer/Solver** | Nested episodes | Monte Carlo scoring, curriculum generation |
| **RefineLoop** | Cooperative multi-turn | Generator/Critic cycles, external LLM judge |
| **HeadToHead** | Self-play tournament | Same-model competition, zero-sum rewards |
| **SPICE** | Corpus-grounded | External knowledge, trainable nested episodes |

[See docs/examples.md for walkthroughs](docs/examples.md)

---

## Installation

```bash
# Requires Python 3.12+, Apple Silicon
git clone https://github.com/eligottlieb/legos
cd legos
uv sync
```

---

## Project Structure

```
src/legos/
├── core/
│   ├── types.py       # Actor, Step, Rollout, TrainingRecord
│   ├── episode.py     # Episode, EpisodeState, MultiTurnEpisode
│   ├── rubric.py      # Rubric, RewardFn
│   ├── credit.py      # GRPOCredit, RAECredit
│   ├── arena.py       # Arena, ArtifactStore
│   └── clients.py     # OpenAIClient
├── tasks/             # Built-in episode implementations
├── training/          # Trainer, loss functions, async loop
└── inference/         # FastAPI server, continuous batching
```

---

## Acknowledgments

This project would very much not be possible without other open source projects:

**[verifiers](https://github.com/PrimeIntellect-ai/verifiers)** - Original inspiration behind this project and the work they did surrounding RL abstractions of environments. Pretty much directly applied the rubric framework where you can register reward functions for scoring, and the notion of an env response in a multiturn setting.

**[ludic](https://github.com/hallerite/ludic)** - From ludic I really took the clear separation between trainer and data generator. All the trainer should do is pull samples to train on, and shouldn't care how they were generated. Also took the idea of a credit assigner, which is a way to assign advantages to rollouts given a set of rewards.

**[mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)** - The core RL training code implemented in MLX.

---

## References

- Zhao et al. (2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. [arXiv:2505.03335](https://arxiv.org/abs/2505.03335)
- Liu et al. (2025). *SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning*. [arXiv:2506.24119](https://arxiv.org/abs/2506.24119)
- Liu et al. (2025). *SPICE: Self-Play In Corpus Environments Improves Reasoning*. [arXiv:2510.24684](https://arxiv.org/abs/2510.24684)
- Kuba et al. (2025). *Language Self-Play For Data-Free Training*. [arXiv:2509.07414](https://arxiv.org/abs/2509.07414)
- Piché et al. (2025). *PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation*. [arXiv:2509.19128](https://arxiv.org/abs/2509.19128)
- Shah et al. (2025). *A Comedy of Estimators: On KL Regularization in RL Training of LLMs*. [arXiv:2512.21852](https://arxiv.org/abs/2512.21852)
