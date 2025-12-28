"""
Self-Play LLM RL Engine

A modular framework for self-play training of language models.

Core Concepts:
- Role: A trainable entity (only trainable entities are roles)
- Episode: Defines how a single rollout unfolds (SingleTurn or MultiTurn)
- Rubric: Function that scores rollouts
- Arena: Persistent state container + orchestration

Example Usage:
    from self_play import Arena, Role, MockInferenceClient
    from self_play.examples import create_debate_arena

    # Create arena with debate setup
    arena = create_debate_arena(
        client=MockInferenceClient(),
        topics=["AI safety is more important than AI capabilities"],
        num_rounds=3,
    )

    # Run a training step
    batch = await arena.step()
    print(f"Records: {len(batch.records)}")
"""
from .core import (
    # Types
    Message,
    Messages,
    Role,
    Step,
    Rollout,
    TrainingRecord,
    # Episode
    Episode,
    EpisodeState,
    GenerateResult,
    ChatEpisode,
    SingleTurnEpisode,
    MultiTurnEpisode,
    AlternatingRolesEpisode,
    # Rubric
    Rubric,
    RewardFn,
    # Arena
    ModelResponse,
    InferenceClient,
    MockInferenceClient,
    Artifact,
    ArtifactStore,
    EpisodeRequest,
    TrainingBatch,
    Arena,
)

__version__ = "0.1.0"

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
