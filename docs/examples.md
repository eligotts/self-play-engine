# Example Walkthroughs

This document walks through each built-in example, showing how the core abstractions compose to create different self-play setups.

---

## Two Main Patterns I've Identified

**Pattern 1: Proposer/Solver** - Same model generates data then trains on it. A proposer generates a question, we get ground truth somehow, then get reward by using the model to solve that question and seeing how well it does.

Examples:
- [Absolute Zero](https://arxiv.org/abs/2505.03335) - Reinforced self-play reasoning with zero data
- [SPICE](https://arxiv.org/abs/2510.24684) - Corpus-grounded self-play
- [Language Self-Play](https://arxiv.org/abs/2509.07414) - Data-free training

**Pattern 2: Two-Player Games** - Same model plays against itself in a more game-like setting. Can be zero-sum games like [SPIRAL](https://arxiv.org/abs/2506.24119), head-to-head competition, or cooperative setups like a generator and critic working together to improve (refine loop).

---

## Negotiation: Two-Player Games

**Pattern**: Two-player zero-sum game, actor-conditioned credit (RAE), complex state management

Based on [SPIRAL](https://arxiv.org/abs/2506.24119)'s SimpleNegotiation. Two players trade resources. Player0 values Gold highly; Player1 values Wood highly. Both try to maximize their inventory value.

### Game Setup

```python
# Player 0: values Gold more (Wood=5, Gold=15)
# Player 1: values Wood more (Wood=15, Gold=5)
# Both start with 10 Wood + 10 Gold
# Winner = player with larger inventory value change
```

State is tracked in `EpisodeState.data`:
- Current inventories
- Trade history
- Whose turn it is
- Pending offers
- Whether an invalid action occurred

### Turn Flow

```python
class NegotiationEpisode(MultiTurnEpisode):
    async def env_response(self, state, arena, artifact) -> str:
        action = parse_action(state.last_completion_text)

        if action.type == "offer":
            state.data["pending_offer"] = action.offer
            return f"Offer pending: {action.offer}"

        elif action.type == "accept":
            execute_trade(state.data)
            return f"Trade completed."

        elif action.type == "deny":
            state.data["pending_offer"] = None
            return "Offer denied."
```

Players can make offers (`[Offer: I give X Wood, Y Gold for Z Wood, W Gold]`), accept (`[Accept]`), or deny (`[Deny]`). Invalid actions (offering more than you have, accepting when no offer pending) result in immediate termination with penalty.

### Private Observations

Each player gets private information about their own values:

```python
def get_observation(self, state, arena, artifact) -> str:
    """Private to current player - NOT in transcript."""
    player = state.current_actor
    inv = state.data["player_resources"][player]
    return f"Your resources: Wood={inv['Wood']}, Gold={inv['Gold']}"
```

This prevents opponents from inferring player-specific values from the conversation history.

### Reward Function

```python
def negotiation_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    extras = rollout.extras

    # Penalty for invalid actions
    if extras.get("invalid_action"):
        invalid_player = extras["invalid_action"]
        return {invalid_player: -1.5, other_player: 0.5}

    # Zero-sum based on value change
    winner = extras.get("winner")
    if winner == "Player0":
        return {"Player0": 1.0, "Player1": -1.0}
    elif winner == "Player1":
        return {"Player0": -1.0, "Player1": 1.0}
    else:
        return {"Player0": 0.0, "Player1": 0.0}
```

### Why RAE Credit?

```python
arena = NegotiationArena(
    client=client,
    credit_assigner=RAECredit(decay=0.95)
)
```

Player0 might have a structural advantage (first-mover). RAE maintains separate baselines per actor, adapting slowly to account for this. Without it, the first mover would consistently get higher advantages, distorting training.

### Running It

```bash
uv run examples/train_negotiation.py
```

---

## Proposer/Solver: Nested Episodes

**Pattern**: Hierarchical episodes, Monte Carlo scoring, curriculum generation

Inspired by [Absolute Zero](https://arxiv.org/abs/2505.03335). This is more of a toy example to demonstrate the proposer/solver pattern - in practice you wouldn't want the model generating both the question AND the answer. You'd want an external validator (like deterministic code execution) to get ground truth. But the structure is the same.

The Proposer generates math problems. Solvers attempt them. The Proposer is rewarded for generating problems at ~50% difficulty - not too easy, not too hard.

### Episode Structure

```python
class ProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        # 1. Generate a question
        prompt = "Generate a challenging math problem with a clear numerical answer..."
        response = await self.call_model("Proposer", prompt, arena)
        question = parse_question(response.text)  # Extract question and answer

        # 2. Store for future curriculum
        if question.get("question"):
            arena.stores["questions"].add(Artifact(data=question))

        # 3. Spawn solver sub-episodes (Monte Carlo)
        requests = [
            EpisodeRequest(
                episode_type="solve",
                artifact=question,
                is_trainable=False  # Don't train on these
            )
            for _ in range(self.n_solver_rollouts)
        ]
        results = await arena.generate_rollouts(requests)
        state.child_results = results

        return state
```

Key insight: `is_trainable=False` means solver completions inform the proposer's reward but don't enter the training batch. Only the proposer steps get trained.

### Pass Rate Extraction

```python
def get_extras(self, state) -> Dict[str, Any]:
    solver_rewards = [
        r.rewards.get("Solver", 0.0)
        for r in state.child_results
    ]
    correct = sum(1 for r in solver_rewards if r > 0)
    pass_rate = correct / len(solver_rewards)

    return {
        "pass_rate": pass_rate,
        "solver_results": solver_rewards,
        "question": state.data.get("question"),
        "ground_truth": state.data.get("ground_truth"),
    }
```

### Goldilocks Reward

```python
def proposer_pass_rate_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    pass_rate = rollout.extras.get("pass_rate", 0.0)

    # Maximum reward at 50% pass rate
    reward = 1.0 - 2 * abs(pass_rate - 0.5)

    return {"Proposer": reward}
```

- Pass rate 0.5 → reward 1.0 (optimal - at capability frontier)
- Pass rate 0.0 or 1.0 → reward 0.0 (too hard or too easy)

This drives the proposer to generate problems that are challenging but solvable.

### Credit Assignment

GRPO with hierarchy:

```
Proposers: compared against other Proposers in the batch
  └── Solvers: compared against sibling Solvers (same parent)
```

A solver's advantage is relative to its siblings - solvers under a different proposer are a different group.

### Warmup Phase

Before training, populate the question store:

```python
async def on_train_start(self):
    # Generate initial questions so solvers have something to work with
    while self.stores["questions"].count() < self.min_questions:
        await self.step(concurrency=1)
```

### Running It

```bash
uv run examples/train_proposer_solver.py
```

---

## HeadToHead: Self-Play Tournaments

**Pattern**: Same-model competition, LLM judging, zero-sum rewards

Two "players" respond to the same creative challenge. An LLM judge picks the winner.

### Symmetric Setup

```python
class MatchEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        challenge = artifact["challenge"]

        # Both players respond to same challenge
        p0_response = await self.call_model("Player0", challenge, arena)
        p1_response = await self.call_model("Player1", challenge, arena)

        state.data["responses"] = {
            "Player0": p0_response.text,
            "Player1": p1_response.text,
        }
        return state
```

Both players are the same model with the same prompt - pure self-play.

### LLM Judge Reward

```python
async def judge_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    responses = rollout.extras["responses"]

    prompt = f"""
    Challenge: {rollout.artifact["challenge"]}

    Response A: {responses["Player0"]}
    Response B: {responses["Player1"]}

    Which response is better? Reply with just "A" or "B" or "tie".
    """

    # Judge calls arena.call_model without actor_id - not training data
    response = await arena.call_model(prompt, temperature=0.3)
    winner = parse_winner(response.text)

    if winner == "A":
        return {"Player0": 1.0, "Player1": -1.0}
    elif winner == "B":
        return {"Player0": -1.0, "Player1": 1.0}
    else:
        return {"Player0": 0.0, "Player1": 0.0}
```

Zero-sum: one player's gain is the other's loss.

### Challenge Generation

Challenges are generated dynamically to prevent overfitting:

```python
class ChallengeProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        prompt = "Generate an interesting creative challenge that tests reasoning..."
        response = await self.call_model("ChallengeProposer", prompt, arena)
        challenge = parse_challenge(response.text)

        arena.stores["challenges"].add(Artifact(data={"challenge": challenge}))
        return state
```

The ChallengeProposer is non-trainable - it just populates the challenge store.

### Running It

```bash
uv run examples/train_head_to_head.py
```

---

## RefineLoop: Cooperative Multi-Turn

**Pattern**: Generator/Critic loop, cooperative rewards, external LLM judge

A Generator writes drafts. A Critic provides feedback. The Generator revises. Repeat N times. Final quality judged by external LLM.

### Turn Structure

```
Turn 0: Generator writes initial draft
Turn 1: Critic provides feedback
Turn 2: Generator revises based on feedback
Turn 3: Critic provides more feedback
...
Turn N: Final draft evaluated by judge
```

### Cooperative Rewards

Both Generator and Critic share the same reward:

```python
async def judge_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    final_draft = rollout.extras["final_draft"]

    # External judge evaluates final quality
    prompt = f"Rate this writing 0-10: {final_draft}"
    response = await judge_client.complete(prompt)
    score = parse_score(response.text) / 10.0

    # Both actors get same reward (cooperative)
    return {"Generator": score, "Critic": score}
```

Why cooperative? The Critic's job is to help the Generator improve. If the Critic were adversarial, it might give unhelpful feedback just to make the Generator fail.

### Running It

```bash
uv run examples/train_refine_loop.py
```

---

## SPICE: Corpus-Grounded Curriculum

**Pattern**: External knowledge source, trainable nested episodes, semantic similarity judging

Like Proposer/Solver, but:
1. Proposer reads from a corpus (external documents)
2. Solver episodes ARE trainable (curriculum learning for both)
3. Judge uses semantic similarity, not exact match

Based on [SPICE](https://arxiv.org/abs/2510.24684).

### Corpus Integration

```python
class SPICEProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        # Sample a document from corpus
        doc = arena.stores["corpus"].sample_one()

        prompt = f"""
        Read this document and create a comprehension question:

        {doc.data["content"]}

        Generate a question that tests understanding.
        """

        response = await self.call_model("Proposer", prompt, arena)
        question = parse_question(response.text)
        question["source_doc"] = doc.data

        # Spawn solver episodes - these ARE trainable
        requests = [
            EpisodeRequest(
                episode_type="solve",
                artifact=question,
                is_trainable=True  # Train on these too!
            )
            for _ in range(self.n_solver_rollouts)
        ]
        results = await arena.generate_rollouts(requests)
        state.child_results = results

        return state
```

Key difference from ProposerSolver: `is_trainable=True` means the solver rollouts are trained on.

### Semantic Similarity Judge

```python
async def semantic_judge(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    predicted = rollout.extras["predicted_answer"]
    ground_truth = rollout.artifact["answer"]

    prompt = f"""
    Are these two answers semantically equivalent?

    Answer 1: {predicted}
    Answer 2: {ground_truth}

    Reply with just "yes" or "no".
    """

    response = await judge_client.complete(prompt, temperature=0.1)
    equivalent = "yes" in response.text.lower()

    return {"Solver": 1.0 if equivalent else 0.0}
```

More flexible than exact match - captures paraphrases and equivalent formulations.

### Running It

```bash
uv run examples/train_spice.py
```

---

## GSM8K: Dataset-Based Training

**Pattern**: Single actor, dataset-based, GRPO credit assignment

GSM8K shows that the self-play framework can also handle more standard dataset-based training loops. A model learns to solve math word problems using reward shaping - no self-play involved, just sampling from a dataset and training on correctness.

### How It Works

```python
class GSM8KArena(Arena):
    def get_batch(self) -> List[EpisodeRequest]:
        # Sample one question, repeat N times (GRPO)
        sample = self.stores["gsm8k"].sample_one()
        return [
            EpisodeRequest("gsm8k", sample.data)
            for _ in range(self.episodes_per_step)
        ]
```

Each batch contains multiple rollouts of the same question. GRPO computes advantages relative to the group - if one rollout gets the right answer and another doesn't, the correct one gets positive advantage.

### Reward Function

```python
def gsm8k_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    text = rollout.steps[0].completion_text
    parsed = extract_answer(text)
    correct = normalize(parsed) == normalize(rollout.artifact["answer"])
    return {"model": 1.0 if correct else 0.0}
```

Simple exact-match against ground truth. The model learns to use the correct format and get the right answer.

### Running It

```bash
uv run examples/train_gsm8k.py
```

---

## Creating Your Own

To create a new self-play setup:

1. **Define your actors**: What personalities exist? What are their system prompts?

2. **Design your episode**: What's the interaction protocol? Single-turn, multi-turn, or hierarchical?

3. **Write your rubric**: How do you score outcomes? Do you need an LLM judge?

4. **Choose credit assignment**: GRPO for group-relative, RAE for actor-conditioned baselines, or custom.

5. **Set up your arena**: Where does data come from? How are batches scheduled?

### Skeleton

```python
class MyEpisode(Episode):
    @property
    def episode_type(self) -> str:
        return "my_task"

    @property
    def rubric(self) -> Rubric:
        return Rubric([my_reward_fn])

    async def rollout(self, arena, artifact, state):
        response = await self.call_model("Actor", prompt, arena)
        # Your logic here
        return state

    def get_extras(self, state) -> Dict[str, Any]:
        return {"result": state.data["result"]}

class MyArena(Arena):
    def get_batch(self) -> List[EpisodeRequest]:
        samples = self.stores["data"].sample(k=self.batch_size)
        return [EpisodeRequest("my_task", s.data) for s in samples]

# Run it
arena = MyArena(client, credit_assigner=GRPOCredit())
arena.add_actor(Actor(id="Actor", system_prompt="..."))
arena.add_episode("my_task", MyEpisode())
arena.add_store("data")

# Load data
for item in my_dataset:
    arena.stores["data"].add(Artifact(data=item))

# Train
await training_loop(arena, trainer, num_steps=1000)
```

The framework handles the rest: rollout generation, scoring, credit assignment, batching, and training.
