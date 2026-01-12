"""
Self-Play LLM RL Engine

A modular framework for self-play training of language models.

Core Concepts:
- Actor: A trainable entity (only trainable entities are actors)
- Episode: Defines how a single rollout unfolds (SingleTurn or MultiTurn)
- Rubric: Function that scores rollouts
- Arena: Persistent state container + orchestration

Example Usage:
    from legos import Arena, Actor, MockInferenceClient
    from legos.tasks import GSM8KArena, GSM8KEpisode

    # Create arena with GSM8K setup
    arena = GSM8KArena(client=MockInferenceClient(), episodes_per_step=4)
    arena.add_episode("gsm8k", GSM8KEpisode())

    # Run a training step
    batch = await arena.step()
    print(f"Records: {len(batch.records)}")
"""
from .core import (
    # Types
    Message,
    Messages,
    Actor,
    Step,
    Rollout,
    TrainingRecord,
    # Episode
    Episode,
    EpisodeState,
    GenerateResult,
    SingleTurnEpisode,
    MultiTurnEpisode,
    # Rubric
    Rubric,
    RewardFn,
    # Credit Assignment
    RolloutStepKey,
    CreditAssigner,
    GRPOCredit,
    RAECredit,
    ConstantCredit,
    EpisodicRewardCredit,
    apply_credit,
    # Arena
    ModelResponse,
    InferenceClient,
    MockInferenceClient,
    Artifact,
    ArtifactStore,
    EpisodeRequest,
    TrainingBatch,
    Arena,
    # Clients
    OpenAIClient,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Message",
    "Messages",
    "Actor",
    "Step",
    "Rollout",
    "TrainingRecord",
    # Episode
    "Episode",
    "EpisodeState",
    "GenerateResult",
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
    # Arena
    "ModelResponse",
    "InferenceClient",
    "MockInferenceClient",
    "Artifact",
    "ArtifactStore",
    "EpisodeRequest",
    "TrainingBatch",
    "Arena",
    # Clients
    "OpenAIClient",
]
