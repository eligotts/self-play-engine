# Self-Play LLM RL Engine: Architecture

## Overview

This engine provides clean abstractions for self-play LLM training. The core idea:

> **Role** is a trainable entity. **Episode** defines a rollout protocol.
> Executing it produces a **Rollout**. A **Rubric** scores the rollout.
> **CreditAssigner** computes advantages. **Arena** orchestrates everything.

## Core Principles

1. **Role = trainable entity only** - Only entities whose completions get trained on are roles.
2. **Judge is NOT a role** - It's part of the rubric (scoring mechanism).
3. **Arena = persistent state + orchestration** - Episode state is transient.
4. **Hierarchical Episodes** - Episodes can spawn sub-episodes (composable units).
5. **TrainingRecord = trainer contract** - Trainer is blind to rollout logic.
6. **Credit Assignment** - Separates reward computation from advantage estimation.
7. **Fully async** - Designed for parallel rollout generation and scoring.

## File Structure

```
src/self_play/
├── core/
│   ├── types.py      # Role, Step, Rollout, TrainingRecord
│   ├── episode.py    # Episode, EpisodeState, GenerateResult, ChatEpisode
│   ├── rubric.py     # Rubric, RewardFn
│   ├── credit.py     # CreditAssigner, GRPOCredit, ConstantCredit
│   ├── arena.py      # Arena, InferenceClient, ArtifactStore
│   ├── clients.py    # OpenAIClient (OpenRouter, local, OpenAI)
│   └── logging.py    # VerboseLogger
└── examples/
    ├── debate.py           # Multi-turn zero-sum debate
    └── proposer_solver.py  # Hierarchical Monte Carlo scoring

examples/
├── run_debate.py           # CLI runner for debates
└── run_proposer_solver.py  # CLI runner for proposer/solver
```

## Core Abstractions

### 1. Types (`core/types.py`)

**Role**: Configuration for a trainable entity.
```python
@dataclass
class Role:
    id: str
    system_prompt: str = ""
    temperature: float = 1.0
    max_tokens: Optional[int] = None

    def build_messages(self, user_content: str, history: Optional[Messages] = None) -> Messages:
        """Build OpenAI-style messages for inference."""
```

**Step**: One model call in a rollout.
```python
@dataclass
class Step:
    role_id: str
    prompt: Messages
    completion: Messages

    # Token data for training
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    completion_logprobs: Optional[List[float]] = None

    # Set by Rubric and CreditAssigner
    reward: float = 0.0
    advantage: float = 0.0

    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def completion_text(self) -> str:
        """Extract text from completion messages."""
```

**Rollout**: Complete trace of an episode execution.
```python
@dataclass
class Rollout:
    id: str  # Auto-generated UUID
    episode_type: str
    artifact: Any
    meta: Dict[str, Any]  # policy_version, etc.

    steps: List[Step]
    extras: Dict[str, Any]  # Episode-specific data via get_extras()

    # Set by Rubric.score()
    rewards: Dict[str, float]     # role_id -> reward
    advantages: Dict[str, float]  # role_id -> advantage
    metrics: Dict[str, float]     # func_name -> value

    started_at: float
    ended_at: Optional[float]
```

**TrainingRecord**: What gets sent to the trainer.
```python
@dataclass
class TrainingRecord:
    role_id: str
    rollout_id: str

    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    logprobs: List[float]

    # Action mask: 0s for prompt tokens, 1s for completion tokens
    action_mask: List[int]

    reward: float       # From Rubric
    advantage: float    # From CreditAssigner

    meta: Dict[str, Any]

    @property
    def input_ids(self) -> List[int]:
        return self.prompt_token_ids + self.completion_token_ids
```

### 2. Episodes (`core/episode.py`)

**GenerateResult**: Result from `episode.generate()`.
```python
@dataclass
class GenerateResult:
    rollout: Rollout
    children: List["GenerateResult"]  # For hierarchical episodes

    @property
    def rewards(self) -> Dict[str, float]:
        """Rewards dict from the rollout."""

    def all_rollouts(self) -> List[Rollout]:
        """Flatten tree into list of rollouts."""
```

**EpisodeState**: Mutable state during a rollout.
```python
@dataclass
class EpisodeState:
    trajectory: List[Step]
    current_actor: str = ""
    data: Dict[str, Any]       # Custom episode data
    done: bool = False
    child_results: List[GenerateResult]  # Results from sub-episodes

    @property
    def turn(self) -> int
    @property
    def last_step(self) -> Optional[Step]
    @property
    def last_completion_text(self) -> str
```

**Episode**: Abstract base class for rollout logic.
```python
class Episode(ABC):
    @property
    @abstractmethod
    def episode_type(self) -> str:
        """Unique identifier for this episode type."""

    @property
    @abstractmethod
    def rubric(self) -> Rubric:
        """Rubric to score this episode."""

    @abstractmethod
    async def rollout(
        self,
        arena: Arena,
        artifact: Any,
        state: Optional[EpisodeState] = None,
    ) -> EpisodeState:
        """Execute episode logic (model calls, sub-episodes)."""

    async def call_model(
        self,
        role_id: str,
        messages: Messages,
        arena: Arena,
    ) -> ModelResponse:
        """Make a model call for a role."""

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Override to add episode-specific data to rollout.extras."""

    async def generate(
        self,
        arena: Arena,
        artifact: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Top-level entry: rollout → build Rollout → score → return result."""
```

**ChatEpisode**: Standard turn-taking chat loop.
```python
class ChatEpisode(Episode):
    @abstractmethod
    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str: ...

    @abstractmethod
    def get_initial_prompt(self, arena: Arena, artifact: Any, state: EpisodeState) -> str: ...

    @abstractmethod
    async def env_response(self, state: EpisodeState, arena: Arena, artifact: Any) -> str: ...

    @abstractmethod
    def is_done(self, state: EpisodeState, artifact: Any) -> bool: ...

    @abstractmethod
    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str: ...
```

**Convenience Subclasses**:
- `SingleTurnEpisode`: One prompt, one completion
- `MultiTurnEpisode`: Chat loop with optional `max_turns`
- `AlternatingRolesEpisode`: Two roles taking turns

### 3. Rubrics (`core/rubric.py`)

**RewardFn**: Type alias for reward functions.
```python
RewardFn = Callable[
    [Rollout, Arena],
    Union[Dict[str, float], Awaitable[Dict[str, float]]]
]
```

**Rubric**: Composable reward functions.
```python
class Rubric:
    def __init__(
        self,
        funcs: List[RewardFn],
        weights: Optional[List[float]] = None,  # Default: all 1.0
    ): ...

    async def score(self, rollout: Rollout, arena: Arena) -> None:
        """
        Score a rollout in-place:
        1. Call each reward function (sync or async)
        2. Aggregate into rollout.rewards (weighted sum per role)
        3. Set step.reward for each step based on its role_id
        """
```

### 4. Credit Assignment (`core/credit.py`)

**CreditAssigner**: Protocol for computing advantages.
```python
RolloutStepKey = Tuple[str, int]  # (rollout_id, step_index)

class CreditAssigner(ABC):
    @abstractmethod
    def compute(
        self,
        results: List[GenerateResult],
    ) -> Dict[RolloutStepKey, float]:
        """Compute credit weights for all steps in all rollouts."""

def apply_credit(
    results: List[GenerateResult],
    weights: Dict[RolloutStepKey, float],
) -> None:
    """Apply computed weights to step.advantage fields in-place."""
```

**GRPOCredit**: Group Relative Policy Optimization.
```python
@dataclass
class GRPOCredit(CreditAssigner):
    """
    A_i = R_i - mean(R_1, ..., R_N)

    Groups by hierarchy level:
    - Top-level results form one group
    - Each parent's children form independent groups
    """
    normalize: bool = False      # Normalize to zero mean, unit std
    positive_only: bool = False  # Clamp negative advantages to 0
```

**Other Assigners**:
- `ConstantCredit`: All steps receive same weight
- `EpisodicRewardCredit`: Each step gets its role's reward (simple REINFORCE)

### 5. Arena (`core/arena.py`)

**ModelResponse**: Response from model inference.
```python
@dataclass
class ModelResponse:
    text: str
    completion: Messages
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    completion_logprobs: Optional[List[float]] = None
```

**InferenceClient**: Abstract interface for model calls.
```python
class InferenceClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: Messages,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        return_tokens: bool = True,
    ) -> ModelResponse: ...

    async def get_policy_version(self) -> int:
        """Get current policy version (default: 0)."""
```

**MockInferenceClient**: Testing client with configurable responses.

**ArtifactStore**: In-memory storage with weighted sampling.
```python
class ArtifactStore:
    def add(self, artifact: Artifact) -> None
    def sample(self, k: int = 1, weighted: bool = False) -> List[Artifact]
    def sample_one(self, weighted: bool = False) -> Optional[Artifact]
    def count(self) -> int
    def get(self, artifact_id: str) -> Optional[Artifact]
```

**EpisodeRequest**: Request to run an episode.
```python
@dataclass
class EpisodeRequest:
    episode_type: str
    artifact: Any
    meta: Dict[str, Any] = None
```

**TrainingBatch**: Batch ready for trainer.
```python
@dataclass
class TrainingBatch:
    records: List[TrainingRecord]
    meta: Dict[str, Any]
```

**Arena**: Orchestration engine.
```python
class Arena:
    def __init__(
        self,
        client: InferenceClient,
        credit_assigner: Optional[CreditAssigner] = None,
        verbose: bool = False,
    ):
        self.roles: Dict[str, Role] = {}
        self.episodes: Dict[str, Episode] = {}
        self.stores: Dict[str, ArtifactStore] = {}

    # Registration
    def add_role(self, role: Role) -> None
    def add_episode(self, episode_type: str, episode: Episode) -> None
    def add_store(self, name: str, store: Optional[ArtifactStore] = None) -> ArtifactStore

    # Model calls
    async def call_model(
        self,
        messages: Messages,
        role_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_tokens: Optional[bool] = None,
    ) -> ModelResponse

    # Batch definition (override for custom scheduling)
    def get_batch(self) -> List[EpisodeRequest]

    # Rollout generation
    async def run_episode(self, episode_type: str, artifact: Any, meta: Optional[Dict] = None) -> GenerateResult
    async def resolve_artifact(self, request: EpisodeRequest) -> Any
    async def generate_rollouts(self, requests: List[EpisodeRequest], concurrency: int = 8) -> List[GenerateResult]

    # Training batch
    def build_training_batch(self, results: List[GenerateResult], verbose: bool = False) -> TrainingBatch
    def sanity_check_batch(self, results: List[GenerateResult], batch: TrainingBatch, num_examples: int = 2) -> None

    # High-level training step
    async def step(self, concurrency: int = 8, verbose: bool = False) -> TrainingBatch:
        """
        1. get_batch() → requests
        2. Tag with policy_version
        3. generate_rollouts() in parallel
        4. Apply credit assignment
        5. build_training_batch()
        """

    async def run(self, num_steps: Optional[int] = None, concurrency: int = 8, verbose: bool = False):
        """Training loop yielding batches."""

    # Lifecycle
    async def startup(self) -> None
    async def shutdown(self) -> None
```

### 6. Inference Clients (`core/clients.py`)

**OpenAIClient**: OpenAI-compatible inference.
```python
class OpenAIClient(InferenceClient):
    """
    Works with OpenAI, OpenRouter, local servers (mlx-vllm),
    and any OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,  # Tries OPENROUTER_API_KEY, OPENAI_API_KEY
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
    ): ...

    @classmethod
    def for_local(cls, port: int = 8000, timeout: float = 120.0) -> "OpenAIClient":
        """Create client for local OpenAI-compatible server."""

    @classmethod
    def for_openrouter(cls, api_key: Optional[str] = None, model: str = "openai/gpt-4o-mini") -> "OpenAIClient":
        """Create client for OpenRouter."""

    async def get_policy_version(self) -> int:
        """Hit /adapters/version endpoint (for LoRA hot-swap)."""
```

### 7. Logging (`core/logging.py`)

**VerboseLogger**: Structured logging for training runs.
```python
class VerboseLogger:
    def __init__(self, log_path: str | Path): ...

    def __enter__(self) -> "VerboseLogger": ...
    def __exit__(self, *args): ...

    # High-level events
    def log_run_start(self, config: Dict[str, Any])
    def log_step_start(self, step_num: int, num_requests: int)
    def log_step_end(self, step_num: int, batch_meta: Dict[str, Any])

    # Episode events
    def log_episode_start(self, episode_type: str, artifact: Any, rollout_id: str)
    def log_episode_end(self, episode_type: str, rollout_id: str, rewards: Dict[str, float], num_steps: int)

    # Model call events
    def log_model_call(self, role_id: str, messages: Messages, response_text: str, has_tokens: bool)

    # Results
    def log_result(self, result: GenerateResult, depth: int = 0)
    def log_batch_summary(self, results: List[GenerateResult])
    def log_training_records(self, records: List[Any])
```

## Data Flow

### Training Step Flow

```
arena.step()
  │
  ├─> get_batch() → List[EpisodeRequest]
  │
  ├─> get_policy_version() → tag requests with version
  │
  ├─> generate_rollouts(requests, concurrency=8)
  │    │
  │    └─> for each request (parallel with semaphore):
  │         │
  │         └─> run_episode(episode_type, artifact)
  │              │
  │              └─> episode.generate(arena, artifact)
  │                   │
  │                   ├─> rollout(arena, artifact, state)
  │                   │    │
  │                   │    ├─> call_model() → Step added to trajectory
  │                   │    │
  │                   │    └─> arena.generate_rollouts() (sub-episodes)
  │                   │         └─> results → state.child_results
  │                   │
  │                   ├─> rubric.score(rollout, arena)
  │                   │    └─> rollout.rewards, step.reward populated
  │                   │
  │                   └─> return GenerateResult(rollout, children)
  │
  ├─> credit_assigner.compute(results) → weights
  │    └─> apply_credit(results, weights) → step.advantage populated
  │
  └─> build_training_batch(results) → TrainingBatch
       └─> flatten all rollouts (including nested children)
       └─> for each step with token IDs → TrainingRecord
```

### Hierarchical Episode Flow (Proposer/Solver)

```
ProposerEpisode.rollout():
  │
  ├─> call_model("Proposer", prompt) → proposed question
  │
  ├─> arena.generate_rollouts([SolveRequest × N])
  │    │
  │    └─> SolveEpisode.generate() × N (parallel)
  │         │
  │         ├─> rollout() → solver answer
  │         │
  │         └─> rubric.score() → {Solver: 1.0 or 0.0}
  │
  ├─> state.child_results = solver results
  │
  └─> return state

ProposerEpisode.get_extras():
  └─> {pass_rate: sum(solver wins) / N, ...}

Proposer rubric:
  └─> reward = 1.0 - 2*|pass_rate - 0.5|  # Goldilocks scoring

Credit assignment (GRPO):
  └─> Proposer group: all proposer rollouts
  └─> Solver group: each proposer's children form independent groups
```

### Reward Flow

```
1. Rubric.score(rollout, arena)
   ├─> Call each reward function
   ├─> Aggregate: rollout.rewards[role_id] = weighted_sum
   └─> Propagate: step.reward = rollout.rewards[step.role_id]

2. CreditAssigner.compute(results)
   ├─> GRPO: advantage = reward - group_mean
   ├─> Group by hierarchy level
   └─> Return: {(rollout_id, step_idx): advantage}

3. apply_credit(results, weights)
   └─> step.advantage = weights[(rollout.id, idx)]

4. TrainingRecord
   ├─> reward: step.reward (from Rubric)
   └─> advantage: step.advantage (from CreditAssigner)
```

## Examples

### 1. Debate (Multi-turn, Zero-sum)

Two debaters alternate. An LLM judge scores the transcript, giving the winner `+score` and the loser `-score`.

```python
# src/self_play/examples/debate.py

def make_debate_rubric(aff_role: str, neg_role: str) -> Rubric:
    """LLM judge returns zero-sum rewards."""
    async def llm_judge(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        # Extract transcript, call judge (NOT a role), parse winner
        return {winner: score, loser: -score}
    return Rubric([llm_judge])

class DebateEpisode(AlternatingRolesEpisode):
    def __init__(self, aff_role_id="Aff", neg_role_id="Neg", num_rounds=3):
        super().__init__(max_turns=num_rounds * 2)
        self._rubric = make_debate_rubric(aff_role_id, neg_role_id)

    @property
    def episode_type(self) -> str:
        return "debate"

    def get_initial_prompt(self, arena, artifact, state) -> str:
        return f"Topic: {artifact.get('topic', 'Unknown')}"

class DebateArena(Arena):
    def get_batch(self) -> List[EpisodeRequest]:
        topics = self.stores["topics"].sample(k=self.batch_size)
        return [EpisodeRequest("debate", t.data) for t in topics]
```

### 2. Proposer/Solver (Hierarchical, Monte Carlo)

A Proposer generates a question. The Arena spawns N Solver episodes. The Proposer's reward is based on the Solver pass rate (targeting ~50% difficulty).

```python
# src/self_play/examples/proposer_solver.py

def make_proposer_rubric(proposer_role: str, target_pass_rate: float = 0.5) -> Rubric:
    """Reward based on how close pass_rate is to target."""
    def goldilocks(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        pass_rate = rollout.extras.get("pass_rate", 0.0)
        reward = 1.0 - 2 * abs(pass_rate - target_pass_rate)
        return {proposer_role: reward}
    return Rubric([goldilocks])

class ProposerEpisode(Episode):
    async def rollout(self, arena, artifact, state=None):
        # 1. Generate question
        response = await self.call_model("Proposer", prompt, arena)

        # 2. Spawn solver sub-episodes
        requests = [EpisodeRequest("solve", parsed_question) for _ in range(N)]
        results = await arena.generate_rollouts(requests)
        state.child_results.extend(results)

        return state

    def get_extras(self, state) -> Dict[str, Any]:
        # Compute pass_rate from child_results for rubric
        return {"pass_rate": ..., "solver_rewards": ...}
```

## Key Design Decisions

1. **Role = Trainable Only**: Prevents contamination of training data with judge or system responses. Judges call `arena.call_model()` without a role_id.

2. **Hierarchical Results**: `GenerateResult` trees allow complex dependency chains. Credit assignment groups by hierarchy level.

3. **Credit Assignment Decoupling**: Rewards from rubrics are separate from advantages computed by `CreditAssigner`. Supports REINFORCE, GRPO, PPO, etc.

4. **Ephemeral Episode State**: All training-relevant state is captured in `Rollout` and `GenerateResult`. Episodes are stateless and parallelizable.

5. **Token Handling**: Steps carry `prompt_token_ids`, `completion_token_ids`, and `completion_logprobs`. TrainingRecord builds `action_mask` (0 for prompt, 1 for completion).

6. **Flexible Inference**: `OpenAIClient` works with OpenRouter, local mlx-vllm servers, or OpenAI directly. Gracefully handles missing logprobs.

7. **Structured Logging**: `VerboseLogger` provides human-readable logs with timestamps, episode tracking, and batch summaries.

## Running Examples

```bash
# Debate with OpenRouter
python examples/run_debate.py "AI will benefit humanity" "Social media is harmful" \
    --num-steps 5 --batch-size 4 --model openai/gpt-4o-mini

# Debate with local server
python examples/run_debate.py "Topic 1" "Topic 2" --local --port 8000

# Proposer/Solver with seed questions
python examples/run_proposer_solver.py \
    --questions '[{"question": "What is 2+2?", "ground_truth": "4"}]' \
    --num-steps 3 --solver-rollouts 4

# With verbose logging
python examples/run_debate.py "Topic" --verbose  # Writes to debate_run.log
```
