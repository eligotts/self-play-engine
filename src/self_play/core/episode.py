"""
Episode: Core rollout logic for the self-play engine.

Episodes are composable units of work tied together by reward dependency.
The base Episode class is minimal - subclasses implement rollout() however they need.

Provided patterns:
- ChatEpisode: Standard turn-taking conversation loop
- SingleTurnEpisode: One prompt, one completion
- MultiTurnEpisode: Chat with max turns
- AlternatingRolesEpisode: Two roles taking turns
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .types import Messages, Rollout, Step
from .rubric import Rubric

if TYPE_CHECKING:
    from .arena import Arena, ModelResponse


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class GenerateResult:
    """Result from episode.generate(), supports hierarchical episodes."""
    rollout: Rollout
    children: List["GenerateResult"] = field(default_factory=list)

    @property
    def rewards(self) -> Dict[str, float]:
        """Rewards dict from the rollout (role_id -> reward)."""
        return self.rollout.rewards

    def all_rollouts(self) -> List[Rollout]:
        """Flatten tree into list of rollouts."""
        result = [self.rollout]
        for child in self.children:
            result.extend(child.all_rollouts())
        return result


# ---------------------------------------------------------------------------
# Episode State
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    """Mutable state during a rollout."""
    trajectory: List[Step] = field(default_factory=list)
    current_actor: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    done: bool = False

    # Child results from sub-episodes
    child_results: List[GenerateResult] = field(default_factory=list)

    @property
    def turn(self) -> int:
        return len(self.trajectory)

    @property
    def last_step(self) -> Optional[Step]:
        return self.trajectory[-1] if self.trajectory else None

    @property
    def last_completion_text(self) -> str:
        if self.last_step:
            return self.last_step.completion_text
        return ""


# ---------------------------------------------------------------------------
# Base Episode
# ---------------------------------------------------------------------------

class Episode(ABC):
    """
    Base class for episodes.

    Subclasses implement:
    - episode_type: unique identifier
    - rubric: function to score the rollout
    - rollout(): the actual episode logic

    The rollout() method is fully flexible - it can:
    - Make model calls via call_model()
    - Spawn sub-episodes (use arena.generate_rollouts)
    - Do whatever custom logic needed
    
    Override get_extras() to add episode-specific data to the Rollout.
    """

    @property
    @abstractmethod
    def episode_type(self) -> str:
        """Unique identifier for this episode type."""
        ...

    @property
    @abstractmethod
    def rubric(self) -> Rubric:
        """
        Rubric to score this episode.

        The rubric's score() method is called with (rollout, arena).
        It must set rewards for each role_id in rollout.steps.
        """
        ...

    # ---------------------------------------------------------------------------
    # Core abstract method - subclasses implement this
    # ---------------------------------------------------------------------------

    @abstractmethod
    async def rollout(
        self,
        arena: Arena,
        artifact: Any,
        state: Optional[EpisodeState] = None,
    ) -> EpisodeState:
        """
        Execute this episode's logic.

        Override to implement any pattern:
        - Chat loops (use ChatEpisode or run_chat_loop helper)
        - Sub-episode spawning (use arena.generate_rollouts)
        - Custom flows
        """
        ...

    # ---------------------------------------------------------------------------
    # Helpers for common operations
    # ---------------------------------------------------------------------------

    async def call_model(
        self,
        role_id: str,
        messages: Messages,
        arena: Arena,
    ) -> "ModelResponse":
        """Make a model call for a role."""
        return await arena.call_model(messages, role_id=role_id)

    # ---------------------------------------------------------------------------
    # Extraction methods - override to customize
    # ---------------------------------------------------------------------------

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Get episode-specific extras. Override to add custom data."""
        return {}

    # ---------------------------------------------------------------------------
    # Top-level entry point
    # ---------------------------------------------------------------------------

    async def generate(
        self,
        arena: Arena,
        artifact: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """
        Top-level entry point for episode execution.

        1. Run rollout
        2. Build Rollout with standard fields
        3. Score with rubric (sets step.reward and rollout.reward)
        4. Return GenerateResult with children
        """
        import time

        state = await self.rollout(arena, artifact)

        # Build rollout
        rollout = Rollout(
            episode_type=self.episode_type,
            artifact=artifact,
            meta=meta or {},
            steps=state.trajectory,
            extras=self.get_extras(state),
            ended_at=time.time(),
        )

        # Score (sets step.reward and rollout.rewards)
        await self.rubric.score(rollout, arena)

        return GenerateResult(
            rollout=rollout,
            children=state.child_results,
        )


# ---------------------------------------------------------------------------
# Chat Loop Helper
# ---------------------------------------------------------------------------

async def run_chat_loop(
    episode: Episode,
    arena: Arena,
    artifact: Any,
    state: EpisodeState,
    get_initial_actor,
    get_initial_prompt,
    env_response,
    is_done,
    get_next_actor,
) -> EpisodeState:
    """
    Standard chat loop for turn-taking conversations.

    Each model call receives: system prompt (from current actor's role) + single user message with full history.
    System prompts are loaded dynamically based on the current actor.
    History is built as a string, with env_response returning strings to append.
    """
    state.current_actor = get_initial_actor(artifact)

    # Get initial history from get_initial_prompt (system prompt is now loaded dynamically per actor)
    history: str = get_initial_prompt(arena, artifact, state)

    while not state.done:
        # Get system prompt dynamically from the current actor's role
        current_role = arena.roles.get(state.current_actor)
        system_prompt = current_role.system_prompt if current_role else None

        # Build prompt: system + user with full history
        prompt: Messages = []
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        user_string = ""
        if state.turn > 0:
            user_string = "Conversation history: \n\n" + history + "\n\nEnd of conversation history."
        else:
            user_string = history
        prompt.append({"role": "user", "content": user_string})

        # Call model
        response = await episode.call_model(state.current_actor, prompt, arena)

        # Record step
        step = Step(
            role_id=state.current_actor,
            prompt=prompt,
            completion=response.completion,
            prompt_token_ids=response.prompt_token_ids,
            completion_token_ids=response.completion_token_ids,
            completion_logprobs=response.completion_logprobs,
        )
        state.trajectory.append(step)

        # Add completion to history
        history += f"\n\n{state.current_actor}: {step.completion_text}"

        # Check done
        if is_done(state, artifact):
            state.done = True
            break

        # Get env response (a string) and append to history
        env_str = await env_response(state, arena, artifact)
        if env_str:
            history += f"\n\n{env_str}"

        state.current_actor = get_next_actor(state, artifact)

    return state


# ---------------------------------------------------------------------------
# Chat Episode Base Class
# ---------------------------------------------------------------------------

class ChatEpisode(Episode):
    """
    Episode with standard turn-taking chat loop.

    Subclasses implement the chat-specific methods:
    - get_initial_actor, get_initial_prompt, env_response, is_done, get_next_actor
    """

    @abstractmethod
    def get_initial_actor(self, artifact: Any) -> str:
        ...

    @abstractmethod
    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        """Return the initial history string. System prompts are loaded dynamically per actor."""
        ...

    @abstractmethod
    async def env_response(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> str:
        """Return a string to append to the history for the next turn."""
        ...

    @abstractmethod
    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        ...

    @abstractmethod
    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        ...

    async def rollout(
        self,
        arena: Arena,
        artifact: Any,
        state: Optional[EpisodeState] = None,
    ) -> EpisodeState:
        """Run the standard chat loop."""
        if state is None:
            state = EpisodeState()

        return await run_chat_loop(
            episode=self,
            arena=arena,
            artifact=artifact,
            state=state,
            get_initial_actor=self.get_initial_actor,
            get_initial_prompt=self.get_initial_prompt,
            env_response=self.env_response,
            is_done=self.is_done,
            get_next_actor=self.get_next_actor,
        )


# ---------------------------------------------------------------------------
# Convenience Subclasses
# ---------------------------------------------------------------------------

class SingleTurnEpisode(ChatEpisode):
    """Single model call episode."""

    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        return state.turn >= 1

    async def env_response(self, state: EpisodeState, arena: Arena, artifact: Any) -> str:
        return ""

    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        return state.current_actor


class MultiTurnEpisode(ChatEpisode):
    """Multi-turn chat with optional max turns."""

    def __init__(self, max_turns: int = -1):
        self.max_turns = max_turns

    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        if self.max_turns > 0 and state.turn >= self.max_turns:
            return True
        return False


class AlternatingRolesEpisode(MultiTurnEpisode):
    """Two roles taking turns."""

    @property
    @abstractmethod
    def roles(self) -> tuple[str, str]:
        ...

    def get_initial_actor(self, artifact: Any) -> str:
        return self.roles[1]

    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        return self.roles[(state.turn + 1) % 2]
