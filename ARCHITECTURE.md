# Self-Play LLM RL Engine: Architecture

## Overview

This engine provides clean abstractions for self-play LLM training. The core idea:

> **Role** is a trainable entity. **Episode** defines a rollout protocol.
> Executing it produces a **Rollout**. A **Verifier** scores the rollout.
> **Arena** orchestrates the execution flow, managing persistent state and 
> parallel rollout generation.

## Core Principles

1.  **Role = trainable entity only** - Only entities whose completions get trained on are roles.
2.  **Judge is NOT a role** - It's part of the verifier (scoring mechanism).
3.  **Arena = persistent state + orchestration** - Episode state is transient.
4.  **Hierarchical Episodes** - Episodes can spawn sub-episodes (composable units).
5.  **TrainingRecord = trainer contract** - Trainer is blind to rollout logic.
6.  **Fully async** - Designed for parallel rollout generation and scoring.

## File Structure

```
src/self_play/
├── core/
│   ├── types.py      # Role, Step, Rollout, TrainingRecord
│   ├── episode.py    # Episode, ChatEpisode, GenerateResult
│   ├── verifier.py   # Verifier, LLMJudgeVerifier, ZeroSumJudgeVerifier
│   └── arena.py      # Arena, InferenceClient, ArtifactStore
└── examples/
    ├── debate.py           # Multi-turn zero-sum example
    └── proposer_solver.py  # Composable hierarchical example
```

## Core Abstractions

### 1. Types (`core/types.py`)

**Role**: A configuration for a trainable entity.
```python
@dataclass
class Role:
    id: str
    system_prompt: str = ""
    temperature: float = 1.0
    max_tokens: Optional[int] = None
```

**Step**: One model call in a rollout, containing prompt, completion, and token-level data.
```python
@dataclass
class Step:
    role_id: str
    prompt: Messages
    completion: Messages
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    completion_logprobs: Optional[List[float]] = None
```

**Rollout**: The complete trace and artifacts produced by an episode execution.
```python
@dataclass
class Rollout:
    id: str
    episode_type: str
    seed: Dict[str, Any]
    steps: List[Step]
    artifacts: Dict[str, Any]  # Extracted by Episode.finalize()
```

**TrainingRecord**: The unit of training data.
```python
@dataclass
class TrainingRecord:
    role_id: str
    rollout_id: str
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    logprobs: List[float]
    reward: float  # Assigned by CreditAssigner
```

### 2. Episodes (`core/episode.py`)

**Episode**: The base class for rollout logic. Subclasses implement the `rollout()` method.

**GenerateResult**: Captures the rollout, its rewards, and any nested results from sub-episodes.
```python
@dataclass
class GenerateResult:
    rollout: Rollout
    rewards: Dict[str, float]
    children: List["GenerateResult"]  # For hierarchical episodes
```

**Episode Lifecycle**:
1.  `get_artifact(arena, seed)`: Prepares data for the rollout.
2.  `rollout(arena, artifact)`: Executes the logic (model calls, sub-episodes).
3.  `finalize(state, artifact)`: Extracts summary artifacts for the verifier.
4.  `verifier.score(rollout, arena)`: Produces rewards.

**Common Patterns**:
-   `ChatEpisode`: Base for turn-taking conversations using a standard loop.
-   `SingleTurnEpisode`: One prompt, one completion.
-   `MultiTurnEpisode`: Chat loop with optional turn limits.
-   `AlternatingRolesEpisode`: Two roles taking turns.

### 3. Verifiers (`core/verifier.py`)

**Verifier**: An async interface for scoring rollouts.
```python
class Verifier(ABC):
    @abstractmethod
    async def score(self, rollout: Rollout, arena: Arena) -> Dict[str, float]:
        """Returns {role_id: reward}"""
```

**Key Implementations**:
-   `ExactMatchVerifier`: Simple ground-truth checking.
-   `LLMJudgeVerifier`: Uses an LLM to evaluate the transcript (Judge is NOT a role).
-   `ZeroSumJudgeVerifier`: Competitive scoring (e.g., Debate).
-   `MonteCarloVerifier`: Scores by running multiple sub-episodes (e.g., Proposer/Solver).

### 4. Arena (`core/arena.py`)

The **Arena** orchestrates the end-to-end training step.

**Orchestration Flow (`Arena.step()`):**
1.  `get_batch()`: Returns `List[EpisodeRequest]` (the workload).
2.  `generate_rollouts(requests)`: Runs episodes in parallel.
3.  `assign_credit(results)`: Maps rewards to specific steps (via `CreditAssigner`).
4.  `build_training_batch(results, weights)`: Flattens everything into `TrainingBatch`.

**Components**:
-   `InferenceClient`: Interface for model calls (with `MockInferenceClient` for testing).
-   `ArtifactStore`: Weighted sampling buffer for seeds/examples.
-   `CreditAssigner`: Strategy for distributing rewards to steps.

## Data Flow

```
Arena.step()
     ↓
Arena.get_batch() → [EpisodeRequest]
     ↓
Arena.generate_rollouts()
     │
     └── Episode.generate(seed)
              ↓
         Episode.rollout()
              │
              ├── Arena.call_model(role_id, messages) → ModelResponse
              │        (Step added to trajectory)
              │
              └── Episode.run_sub_episode() → GenerateResult
                       (Recursion for hierarchy)
              ↓
         Verifier.score(rollout) → rewards
     ↓
Arena.assign_credit(results) → weights
     ↓
Arena.build_training_batch() → TrainingBatch
     ↓
  Trainer
```

## Examples

### 1. Debate (Multi-turn, Zero-sum)
Two debaters alternate. An LLM judge scores the transcript, giving the winner `+1.0` and the loser `-1.0`.

```python
# Defined in src/self_play/examples/debate.py
class DebateEpisode(AlternatingRolesEpisode):
    def __init__(self, num_rounds=3):
        super().__init__(max_turns=num_rounds * 2)
        self._verifier = ZeroSumJudgeVerifier(role_a="Aff", role_b="Neg")
    
    # ... implements get_initial_prompt and env_response ...
```

### 2. Proposer/Solver (Hierarchical, MC)
A Proposer generates a question. The Arena then spawns `N` Solver episodes. The Proposer's reward is based on the Solver's pass rate (aiming for a 50% "Goldilocks" difficulty).

```python
# Defined in src/self_play/examples/proposer_solver.py
class ProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state=None):
        # 1. Generate question
        response = await self.call_model("Proposer", prompt, arena)
        
        # 2. Spawn sub-episodes
        requests = [EpisodeRequest("solve", seed=parsed_question) for _ in range(N)]
        results = await arena.generate_rollouts(requests)
        state.child_results.extend(results)
        
        return state
```

## Key Design Decisions

1.  **Role = Trainable Only**: Prevents contamination of training data with judge or system responses.
2.  **Hierarchical Results**: `GenerateResult` trees allow complex dependency chains (e.g., training a proposer based on its output's difficulty for a solver).
3.  **Credit Assignment Decoupling**: Rewards from the verifier are mapped to steps by the `Arena`, allowing for different advantage estimation strategies (REINFORCE, PPO, etc.).
4.  **Ephemeral Episode State**: All state needed for training is captured in the `Rollout` and `GenerateResult`, making the core loop stateless and easier to parallelize.
5.  **ArtifactStore**: Built-in support for experience replay or few-shot example management.
