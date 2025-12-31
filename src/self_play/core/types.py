"""
Core types for the self-play engine.

Minimal set of types:
- Role: a trainable entity (only trainable entities are roles)
- Step: one model call in a rollout
- Rollout: complete trace of an episode (includes artifact + metadata)
- TrainingRecord: what gets sent to the trainer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import time

# Message format (OpenAI-style)
Message = Dict[str, Any]  # {"role": "system"|"user"|"assistant", "content": "..."}
Messages = List[Message]


@dataclass
class Role:
    """
    A trainable role. Only roles produce training data.

    The judge in a debate is NOT a role - it's part of the verifier.
    Only entities whose completions get trained on are roles.
    """
    id: str
    system_prompt: str = ""
    temperature: float = 1.0
    max_tokens: Optional[int] = None

    def build_messages(self, user_content: str, history: Optional[Messages] = None) -> Messages:
        """Build messages for a model call."""
        messages: Messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages


@dataclass
class Step:
    """
    One model call in a rollout.

    Stores the full prompt and completion messages for this turn,
    plus token-level data for training.
    """
    role_id: str
    prompt: Messages  # Full prompt for this turn
    completion: Messages  # Model's response as messages

    # Token data for training (optional for eval-only)
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    completion_logprobs: Optional[List[float]] = None

    # Reward/advantage (set by Rubric)
    reward: float = 0.0
    advantage: float = 0.0

    # Metadata
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def completion_text(self) -> str:
        """Extract text from completion messages."""
        if self.completion and len(self.completion) > 0:
            return self.completion[0].get("content", "")
        return ""


@dataclass
class Rollout:
    """
    Complete trace of an episode execution.

    Episodes populate standard fields; Rubrics consume them for scoring.
    The artifact is the primary input content for the episode.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_type: str = ""
    artifact: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # The conversation trajectory
    steps: List[Step] = field(default_factory=list)

    # === Standard fields for scoring (populated by episode) ===
    extras: Dict[str, Any] = field(default_factory=dict)        # Episode-specific data

    # === Set by Rubric.score() ===
    rewards: Dict[str, float] = field(default_factory=dict)     # role_id -> reward
    advantages: Dict[str, float] = field(default_factory=dict)  # role_id -> advantage
    metrics: Dict[str, float] = field(default_factory=dict)     # func_name -> value

    # Timing
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None


@dataclass
class TrainingRecord:
    """
    What gets sent to the trainer.

    One record per trainable step. The trainer doesn't know
    about episodes, verifiers, or arenas.
    """
    role_id: str
    rollout_id: str

    # Token sequences
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    logprobs: List[float]

    # Action mask: 0s for prompt tokens, 1s for completion tokens
    action_mask: List[int]

    # Reward/advantage for this step
    reward: float  # Raw reward from rubric
    advantage: float = 0.0  # Credit-assigned weight (for training loss)

    # Metadata (for logging, not training)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_ids(self) -> List[int]:
        return self.prompt_token_ids + self.completion_token_ids
