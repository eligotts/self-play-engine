"""
Core abstractions for the self-play engine.

Simple, focused:
- Role: trainable entity
- Episode: single-turn or multi-turn rollout protocol
- Rubric: function that scores rollouts
- Arena: persistent state + orchestration
"""
from .types import (
    Message,
    Messages,
    Role,
    Step,
    Rollout,
    TrainingRecord,
)

from .episode import (
    Episode,
    EpisodeState,
    GenerateResult,
    ChatEpisode,
    SingleTurnEpisode,
    MultiTurnEpisode,
    AlternatingRolesEpisode,
)

from .rubric import Rubric, RewardFn

from .arena import (
    ModelResponse,
    InferenceClient,
    MockInferenceClient,
    Artifact,
    ArtifactStore,
    EpisodeRequest,
    TrainingBatch,
    Arena,
)

__all__ = [
    # Types
    "Message",
    "Messages",
    "Role",
    "Step",
    "Rollout",
    "TrainingRecord",
    # Episode
    "Episode",
    "EpisodeState",
    "GenerateResult",
    "ChatEpisode",
    "SingleTurnEpisode",
    "MultiTurnEpisode",
    "AlternatingRolesEpisode",
    # Rubric
    "Rubric",
    "RewardFn",
    # Arena
    "ModelResponse",
    "InferenceClient",
    "MockInferenceClient",
    "Artifact",
    "ArtifactStore",
    "EpisodeRequest",
    "TrainingBatch",
    "Arena",
]
