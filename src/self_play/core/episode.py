"""
Episode: Core rollout logic for the self-play engine.

Episodes are composable units of work tied together by reward dependency.
The base Episode class is minimal - subclasses implement rollout() however they need.

Provided patterns:
- MultiTurnEpisode: Standard turn-taking conversation loop with optional max turns
- SingleTurnEpisode: One prompt, one completion
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .types import GenerateResult, Messages, ModelResponse, Rollout, Step
from .rubric import Rubric

if TYPE_CHECKING:
    from .arena import Arena


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
    ) -> ModelResponse:
        """Make a model call for a role."""
        return await arena.call_model(messages, role_id=role_id)

    # ---------------------------------------------------------------------------
    # Lifecycle methods - override to customize
    # ---------------------------------------------------------------------------

    def startup(self, state: EpisodeState, artifact: Any) -> None:
        """
        Called at the beginning of generate() before rollout().

        Override to initialize episode-specific state in state.data.
        Default implementation does nothing.
        """
        pass

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
        is_trainable: bool = True,
    ) -> GenerateResult:
        """
        Top-level entry point for episode execution.

        1. Create state and call startup()
        2. Run rollout
        3. Build Rollout with standard fields
        4. Score with rubric (sets step.reward and rollout.reward)
        5. Return GenerateResult with children

        Args:
            is_trainable: If False, this episode's steps won't be used for training.
                          Credit assignment will skip non-trainable results and their children.
        """
        import time

        # Create state and call startup for initialization
        state = EpisodeState()
        self.startup(state, artifact)

        state = await self.rollout(arena, artifact, state)

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
            is_trainable=is_trainable,
        )


# ---------------------------------------------------------------------------
# Chat Loop Helper (Private)
# ---------------------------------------------------------------------------

async def _run_chat_loop(
    episode: Episode,
    arena: Arena,
    artifact: Any,
    state: EpisodeState,
    get_initial_actor,
    get_initial_prompt,
    env_response,
    is_done,
    get_next_actor,
    get_observation=None,  # New: player-specific observation each turn
) -> EpisodeState:
    """
    Standard chat loop for turn-taking conversations.

    Each model call receives: system prompt + user message with observation and history.

    Key concepts:
    - System prompt: Static per-role instructions (from arena.roles)
    - Initial prompt: Shared context at game start (from get_initial_prompt)
    - Observation: Player-specific state, regenerated each turn (from get_observation)
    - Transcript: Shared conversation log (player responses + action results)

    The observation is NOT added to transcript - it's private to each player's turn.
    This prevents leaking player-specific info (like personal values) to opponents.
    """
    state.current_actor = get_initial_actor(artifact, state)

    # Initial context (shared, shown once at start)
    initial_context: str = get_initial_prompt(arena, artifact, state)

    # Transcript: conversation log (shared between players)
    # Only contains: player responses + action results from env_response
    transcript: str = ""

    while not state.done:
        # Get system prompt dynamically from the current actor's role
        current_role = arena.roles.get(state.current_actor)
        system_prompt = current_role.system_prompt if current_role else None

        # Build user message with three parts:
        # 1. Observation (player-specific, fresh each turn, NOT in transcript)
        # 2. Initial context (shared game setup)
        # 3. Transcript (conversation so far)

        user_parts = []

        # 1. Player-specific observation (if provided)
        obs = get_observation(state, arena, artifact) if get_observation else None
        if obs:
            user_parts.append(obs)

        # 2. Initial context (on first turn, or always if no observation returned)
        if state.turn == 0 or not obs:
            if initial_context:
                user_parts.append(initial_context)

        # 3. Transcript (conversation history)
        if transcript:
            user_parts.append(f"Conversation so far:\n{transcript}")

        user_string = "\n\n".join(user_parts)
        # Build prompt
        prompt: Messages = []
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
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

        # Add player's response to transcript (this IS shared)
        transcript += f"{state.current_actor}: {step.completion_text}\n\n"

        # Check done
        if is_done(state, artifact):
            state.done = True
            break

        # Process action and get result string (shared info like "trade executed")
        env_str = await env_response(state, arena, artifact)
        if env_str:
            transcript += f"{env_str}\n\n"

        state.current_actor = get_next_actor(state, artifact)

    return state


# ---------------------------------------------------------------------------
# Multi-Turn Episode Base Class
# ---------------------------------------------------------------------------

class MultiTurnEpisode(Episode):
    """
    Episode with standard turn-taking chat loop.

    Subclasses must implement:
    - get_initial_actor: which role starts
    - get_initial_prompt: shared initial context string

    Subclasses may override:
    - get_observation: player-specific state, regenerated each turn (default: None)
    - env_response: action result to append to transcript (default: "")
    - get_next_actor: which role goes next (default: same actor)
    - is_done: when to stop (default: check max_turns)

    Prompt structure each turn:
    - System: Role's system_prompt (static)
    - User: [observation] + [initial_context on turn 0] + [transcript]

    The observation is NOT added to transcript - it's private to each player.
    """

    def __init__(self, max_turns: int = -1):
        self.max_turns = max_turns

    @abstractmethod
    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        ...

    @abstractmethod
    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        """Return the shared initial context string (shown on turn 0)."""
        ...

    def get_observation(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> Optional[str]:
        """
        Return player-specific observation for the current actor.

        Override this for games with mutable state (like negotiation).
        The observation is shown each turn but NOT added to the transcript,
        preventing player-specific info from leaking to opponents.

        Returns None by default (no per-turn observation).
        """
        return None

    async def env_response(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> str:
        """
        Process action and return result string for transcript.

        Called after each turn. The returned string IS added to the shared
        transcript (both players see it). Use for action results like
        "Trade executed" or "Offer denied".

        Returns empty string by default.
        """
        return ""

    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        if self.max_turns > 0 and state.turn >= self.max_turns:
            return True
        return False

    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        return state.current_actor

    async def rollout(
        self,
        arena: Arena,
        artifact: Any,
        state: Optional[EpisodeState] = None,
    ) -> EpisodeState:
        """Run the standard chat loop."""
        if state is None:
            state = EpisodeState()

        return await _run_chat_loop(
            episode=self,
            arena=arena,
            artifact=artifact,
            state=state,
            get_initial_actor=self.get_initial_actor,
            get_initial_prompt=self.get_initial_prompt,
            env_response=self.env_response,
            is_done=self.is_done,
            get_next_actor=self.get_next_actor,
            get_observation=self.get_observation,
        )


# ---------------------------------------------------------------------------
# Single-Turn Episode
# ---------------------------------------------------------------------------

class SingleTurnEpisode(MultiTurnEpisode):
    """Single model call episode (one prompt, one completion)."""

    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        return state.turn >= 1
