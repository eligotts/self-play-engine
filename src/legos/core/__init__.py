"""
Core abstractions for legos.

Simple, focused:
- Actor: trainable entity
- Episode: single-turn or multi-turn rollout protocol
- Rubric: function that scores rollouts
- Arena: persistent state + orchestration
"""
from .types import (
    # Message types
    Message,
    Messages,
    # Core types
    Actor,
    Step,
    Rollout,
    TrainingRecord,
    # Inference types
    ModelResponse,
    # Artifact types
    Artifact,
    # Episode types
    GenerateResult,
    EpisodeRequest,
    TrainingBatch,
)

from .episode import (
    Episode,
    EpisodeState,
    SingleTurnEpisode,
    MultiTurnEpisode,
)

from .rubric import Rubric, RewardFn

from .credit import (
    RolloutStepKey,
    CreditAssigner,
    GRPOCredit,
    RAECredit,
    ConstantCredit,
    EpisodicRewardCredit,
    apply_credit,
)

from .clients import (
    InferenceClient,
    MockInferenceClient,
    OpenAIClient,
)

from .arena import (
    ArtifactStore,
    Arena,
)

__all__ = [
    # Types
    "Message",
    "Messages",
    "Actor",
    "Step",
    "Rollout",
    "TrainingRecord",
    "ModelResponse",
    "Artifact",
    "GenerateResult",
    "EpisodeRequest",
    "TrainingBatch",
    # Episode
    "Episode",
    "EpisodeState",
    "SingleTurnEpisode",
    "MultiTurnEpisode",
    # Rubric
    "Rubric",
    "RewardFn",
    # Credit Assignment
    "RolloutStepKey",
    "CreditAssigner",
    "GRPOCredit",
    "RAECredit",
    "ConstantCredit",
    "EpisodicRewardCredit",
    "apply_credit",
    # Clients
    "InferenceClient",
    "MockInferenceClient",
    "OpenAIClient",
    # Arena
    "ArtifactStore",
    "Arena",
]
