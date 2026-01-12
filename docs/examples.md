# Example Walkthroughs

This document walks through each built-in example, showing how the core abstractions compose to create different self-play setups.

---

## GSM8K: Basic GRPO Training

**Pattern**: Single actor, dataset-based, GRPO credit assignment

GSM8K is the simplest example—a model learns to solve math word problems using reward shaping.

### Episode Structure

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

Each batch contains multiple rollouts of the same question. GRPO computes advantages relative to the group—if one rollout gets the right answer and another doesn't, the correct one gets positive advantage.

### Reward Composition

Three reward signals combined:

```python
def gsm8k_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    text = rollout.steps[0].completion_text

    # Format: did the model use <answer> tags?
    format_reward = 0.25 if "<answer>" in text else 0.0

    # Correctness: does the answer match ground truth?
    parsed = extract_answer(text)
    correct = normalize(parsed) == normalize(rollout.artifact["answer"])
    correctness_reward = 1.0 if correct else 0.0

    # Brevity: bonus for concise answers
    tokens = len(text.split())
    brevity_reward = max(0, 0.5 - tokens / 200)

    return {"Solver": format_reward + correctness_reward + brevity_reward}
```

The model learns to:
1. Use the correct format (`<answer>` tags)
2. Get the right answer
3. Be concise

### Running It

```bash
uv run examples/train_gsm8k.py
```

---

## Negotiation: Multi-Agent Games

**Pattern**: Two-player game, actor-conditioned credit (RAE), complex state management

Two players trade resources. Player0 values Gold highly; Player1 values Wood highly. Both try to maximize their inventory value.

### Game State

```python
class NegotiationEpisode(MultiTurnEpisode):
    def __init__(self):
        self.initial_inventory = {"Gold": 3, "Wood": 3, "Stone": 3}
        self.player_values = {
            "Player0": {"Gold": 10, "Wood": 1, "Stone": 5},
            "Player1": {"Gold": 1, "Wood": 10, "Stone": 5},
        }
```

State is tracked in `EpisodeState.data`:
- Current inventories
- Trade history
- Whose turn it is
- Whether an invalid action occurred

### Turn Flow

```python
async def env_response(self, state, arena, artifact) -> str:
    action = parse_action(state.last_completion_text)

    if action.type == "offer":
        state.data["pending_offer"] = action.offer
        return f"Offer pending: {action.offer}"

    elif action.type == "accept":
        execute_trade(state.data, action.offer)
        return f"Trade completed."

    elif action.type == "deny":
        state.data["pending_offer"] = None
        return "Offer denied."
```

Invalid actions (offering more than you have, accepting when no offer pending) result in immediate episode termination with penalty.

### Reward Function

```python
def negotiation_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    extras = rollout.extras

    # Penalty for invalid actions
    if extras.get("invalid_action"):
        invalid_player = extras["invalid_action"]
        return {invalid_player: -1.0, other_player: 0.5}

    # Reward based on inventory value change
    rewards = {}
    for player_id in ["Player0", "Player1"]:
        initial_value = compute_value(
            extras["initial_inventory"],
            extras["player_values"][player_id]
        )
        final_value = compute_value(
            extras["player_resources"][player_id],
            extras["player_values"][player_id]
        )
        rewards[player_id] = (final_value - initial_value) / initial_value

    return rewards
```

### Credit Assignment (RAE)

Uses actor-conditioned EMA baselines:

```python
arena = NegotiationArena(
    client=client,
    credit_assigner=RAECredit(gamma=0.99)
)
```

Why RAE? Player0 might have a structural advantage (first-mover). RAE maintains separate baselines per actor, adapting slowly to account for this.

### Running It

```bash
uv run examples/train_negotiation.py
```

---

## Proposer/Solver: Nested Episodes

**Pattern**: Hierarchical episodes, Monte Carlo scoring, curriculum generation

The Proposer generates math problems. Solvers attempt them. The Proposer is rewarded for generating problems at 50% difficulty—not too easy, not too hard.

### Episode Structure

```python
class ProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        # 1. Generate a question
        prompt = self.build_proposer_prompt(arena)
        response = await self.call_model("Proposer", prompt, arena)
        question = parse_question(response.text)

        # 2. Store for future curriculum
        if question.get("question"):
            arena.stores["questions"].add(Artifact(data=question))

        # 3. Spawn solver sub-episodes
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

Key insight: `is_trainable=False` means solver completions inform the proposer's reward but don't enter the training batch.

### Pass Rate Extraction

```python
def get_extras(self, state) -> Dict[str, Any]:
    solver_rewards = [
        r.rewards.get("Solver", 0.0)
        for r in state.child_results
    ]
    pass_rate = sum(1 for r in solver_rewards if r > 0) / len(solver_rewards)

    return {
        "pass_rate": pass_rate,
        "solver_rewards": solver_rewards,
        "question": state.data.get("question"),
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

- Pass rate 0.5 → reward 1.0 (optimal)
- Pass rate 0.0 or 1.0 → reward 0.0 (too hard or too easy)

### Credit Assignment

GRPO with hierarchy:

```
Proposers: compared against other Proposers
  └── Solvers: compared against sibling Solvers (same parent)
```

A solver's advantage is relative to its siblings—solvers under a different proposer are a different group.

### Warmup Phase

Before training, populate the question store:

```python
async def on_train_start(self):
    # Generate initial questions
    for _ in range(self.min_questions):
        batch = await self.step(concurrency=1)
        # Questions are stored as side effect
```

### Running It

```bash
uv run examples/train_proposer_solver.py
```

---

## RefineLoop: External Judges

**Pattern**: Cooperative multi-turn, generator/critic loop, external LLM judge

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

### Cooperative Actors

Both Generator and Critic share the same reward:

```python
def judge_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    final_draft = rollout.extras["final_draft"]

    # External judge (OpenRouter)
    prompt = f"Rate this writing 0-10: {final_draft}"
    response = await judge_client.complete(prompt)
    score = parse_score(response.text) / 10.0

    # Both actors get same reward (cooperative)
    return {"Generator": score, "Critic": score}
```

Why cooperative? The Critic's job is to help the Generator improve. If the Critic is adversarial, it might give unhelpful feedback.

### Task Generation

Tasks are generated dynamically to prevent overfitting:

```python
class TaskProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        prompt = "Generate an interesting creative writing task..."
        response = await self.call_model("TaskProposer", prompt, arena)
        task = parse_task(response.text)

        arena.stores["tasks"].add(Artifact(data=task))
        return state
```

The TaskProposer is non-trainable—it just populates the task store.

### Running It

```bash
uv run examples/train_refine_loop.py
```

---

## HeadToHead: Self-Play Tournaments

**Pattern**: Same-model competition, LLM judging, zero-sum rewards

Two players respond to the same creative challenge. An LLM judge picks the winner.

### Symmetric Setup

```python
class HeadToHeadEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        challenge = artifact["challenge"]

        # Both players respond
        p0_response = await self.call_model("Player0", challenge, arena)
        p1_response = await self.call_model("Player1", challenge, arena)

        state.data["responses"] = {
            "Player0": p0_response.text,
            "Player1": p1_response.text,
        }
        return state
```

Both players are the same model with the same prompt—pure self-play.

### LLM Judge

```python
async def judge_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    responses = rollout.extras["responses"]

    prompt = f"""
    Challenge: {rollout.artifact["challenge"]}

    Response A: {responses["Player0"]}
    Response B: {responses["Player1"]}

    Which response is better? Reply with just "A" or "B" or "tie".
    """

    response = await judge_client.complete(prompt, temperature=0.3)
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

Similar to RefineLoop, challenges are generated dynamically:

```python
class ChallengeProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state):
        prompt = "Generate a creative challenge that tests reasoning..."
        response = await self.call_model("ChallengeProposer", prompt, arena)
        challenge = parse_challenge(response.text)

        arena.stores["challenges"].add(Artifact(data={"challenge": challenge}))
        return state
```

### Running It

```bash
uv run examples/train_head_to_head.py
```

---

## SPICE: Corpus-Grounded Curriculum

**Pattern**: External knowledge source, trainable nested episodes, semantic similarity judging

Like Proposer/Solver, but:
1. Proposer reads from a corpus (external documents)
2. Solver episodes ARE trainable (curriculum learning for both)
3. Judge uses semantic similarity, not exact match

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

        # Spawn solver episodes (these ARE trainable)
        requests = [
            EpisodeRequest(
                episode_type="solve",
                artifact=question,
                is_trainable=True  # Train on these too
            )
            for _ in range(self.n_solver_rollouts)
        ]
        results = await arena.generate_rollouts(requests)
        state.child_results = results

        return state
```

Key difference: solver episodes are trainable. Both proposer and solver improve together.

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

More flexible than exact match—captures paraphrases and equivalent formulations.

### Corpus Store

```python
# Load corpus at initialization
async def on_train_start(self):
    for doc in load_documents("data/corpus/*.txt"):
        self.stores["corpus"].add(Artifact(data={"content": doc}))
```

### Running It

```bash
uv run examples/train_spice.py
```

---

## Creating Your Own

To create a new self-play setup:

1. **Define your actors**: What actors exist? What are their system prompts?

2. **Design your episode**: What's the interaction protocol? Is it single-turn, multi-turn, or hierarchical?

3. **Write your rubric**: How do you score outcomes? Do you need an LLM judge?

4. **Choose credit assignment**: GRPO for group-relative, RAE for actor-conditioned baselines, or custom.

5. **Set up your arena**: Where does data come from? How are batches scheduled?

Example skeleton:

```python
class MyEpisode(Episode):
    @property
    def episode_type(self) -> str:
        return "my_task"

    @property
    def rubric(self) -> Rubric:
        return Rubric([my_reward_fn])

    async def rollout(self, arena, artifact, state):
        # Your logic here
        response = await self.call_model("Actor", prompt, arena)
        # ...
        return state

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
