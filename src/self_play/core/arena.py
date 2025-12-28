"""
Arena: Orchestration engine for self-play training.

The Arena orchestrates the training loop:
1. get_batch() - Define what episodes to run
2. generate_rollouts() - Execute episodes in parallel
3. build_training_batch() - Convert to trainer-consumable format

Episode state is transient (within one episode).
Arena state persists across the entire training run.
Rewards are computed by each episode's Rubric during generation.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .types import Messages, Role, TrainingRecord
from .episode import Episode, GenerateResult


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    """Response from a model call."""
    text: str
    completion: Messages

    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    completion_logprobs: Optional[List[float]] = None


class InferenceClient(ABC):
    """Abstract interface for model inference."""

    @abstractmethod
    async def complete(
        self,
        messages: Messages,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        return_tokens: bool = True,
    ) -> ModelResponse:
        ...


class MockInferenceClient(InferenceClient):
    """Mock client for testing."""

    def __init__(self, response_fn: Optional[Callable[[Messages], str]] = None):
        self.response_fn = response_fn or (lambda _: "Mock response")
        self._call_count = 0

    async def complete(
        self,
        messages: Messages,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        return_tokens: bool = True,
    ) -> ModelResponse:
        self._call_count += 1
        text = self.response_fn(messages)

        fake_prompt_ids = list(range(10))
        fake_completion_ids = list(range(len(text.split())))
        fake_logprobs = [-0.5] * len(fake_completion_ids)

        return ModelResponse(
            text=text,
            completion=[{"role": "assistant", "content": text}],
            prompt_token_ids=fake_prompt_ids if return_tokens else None,
            completion_token_ids=fake_completion_ids if return_tokens else None,
            completion_logprobs=fake_logprobs if return_tokens else None,
        )


# ---------------------------------------------------------------------------
# Artifact Store
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    """An item in an artifact store."""
    id: str
    data: Dict[str, Any]
    weight: float = 1.0


class ArtifactStore:
    """Simple in-memory artifact store with weighted sampling."""

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self._items: Dict[str, Artifact] = {}
        self._order: List[str] = []

    def add(self, artifact: Artifact) -> None:
        if self.max_size and len(self._items) >= self.max_size:
            oldest_id = self._order.pop(0)
            del self._items[oldest_id]
        self._items[artifact.id] = artifact
        self._order.append(artifact.id)

    def sample(self, k: int = 1, weighted: bool = False) -> List[Artifact]:
        import random
        items = list(self._items.values())
        if not items:
            return []
        k = min(k, len(items))
        if weighted:
            weights = [a.weight for a in items]
            return random.choices(items, weights=weights, k=k)
        return random.sample(items, k)

    def sample_one(self, weighted: bool = False) -> Optional[Artifact]:
        results = self.sample(1, weighted)
        return results[0] if results else None

    def count(self) -> int:
        return len(self._items)

    def get(self, artifact_id: str) -> Optional[Artifact]:
        return self._items.get(artifact_id)


# ---------------------------------------------------------------------------
# Batch Request
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRequest:
    """Request to run an episode."""
    episode_type: str
    seed: Dict[str, Any]
    meta: Dict[str, Any] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


# ---------------------------------------------------------------------------
# Training Batch
# ---------------------------------------------------------------------------

@dataclass
class TrainingBatch:
    """Batch of training records ready for the trainer."""
    records: List[TrainingRecord]
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class Arena:
    """
    Orchestration engine for self-play training.

    Flow:
        requests = arena.get_batch()
        results = await arena.generate_rollouts(requests)
        batch = arena.build_training_batch(results)

    Rewards are computed by each episode's Rubric during generation.
    Each step gets the reward for its role.
    """

    def __init__(self, client: InferenceClient):
        self.client = client

        # Registries
        self.roles: Dict[str, Role] = {}
        self.episodes: Dict[str, Episode] = {}
        self.stores: Dict[str, ArtifactStore] = {}

    # ---------------------------------------------------------------------------
    # Registration
    # ---------------------------------------------------------------------------

    def add_role(self, role: Role) -> None:
        self.roles[role.id] = role

    def add_episode(self, episode_type: str, episode: Episode) -> None:
        self.episodes[episode_type] = episode

    def add_store(self, name: str, store: Optional[ArtifactStore] = None) -> ArtifactStore:
        if store is None:
            store = ArtifactStore()
        self.stores[name] = store
        return store

    # ---------------------------------------------------------------------------
    # Model Calls
    # ---------------------------------------------------------------------------

    async def call_model(
        self,
        messages: Messages,
        role_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_tokens: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Make a model call.

        If role_id is provided, uses the role's config (and returns tokens for training).
        Otherwise, uses explicit parameters.
        """
        if role_id is not None:
            role = self.roles[role_id]
            return await self.client.complete(
                messages=messages,
                temperature=temperature if temperature is not None else role.temperature,
                max_tokens=max_tokens if max_tokens is not None else role.max_tokens,
                return_tokens=True,
            )

        return await self.client.complete(
            messages=messages,
            temperature=temperature if temperature is not None else 1.0,
            max_tokens=max_tokens,
            return_tokens=return_tokens if return_tokens is not None else False,
        )

    # ---------------------------------------------------------------------------
    # Batch Definition (override for custom scheduling)
    # ---------------------------------------------------------------------------

    def get_batch(self) -> List[EpisodeRequest]:
        """
        Define the next batch of episodes to run.

        Override this to implement custom scheduling logic:
        - For debate: pull next N topics
        - For propose/solve: schedule proposer episodes (solvers nested inside)
        - For ordered execution: return requests with ordering metadata

        Default: returns empty list (override in subclass).
        """
        return []

    # ---------------------------------------------------------------------------
    # Rollout Generation
    # ---------------------------------------------------------------------------

    async def run_episode(
        self,
        episode_type: str,
        seed: Dict[str, Any],
    ) -> GenerateResult:
        """Run a single episode."""
        episode = self.episodes[episode_type]
        return await episode.generate(self, seed)

    async def generate_rollouts(
        self,
        requests: List[EpisodeRequest],
        concurrency: int = 8,
    ) -> List[GenerateResult]:
        """
        Execute episodes in parallel with bounded concurrency.

        Returns list of GenerateResults (may contain nested children).
        """
        if not requests:
            return []

        sem = asyncio.Semaphore(concurrency)

        async def run_one(request: EpisodeRequest) -> GenerateResult:
            async with sem:
                return await self.run_episode(request.episode_type, request.seed)

        tasks = [asyncio.create_task(run_one(req)) for req in requests]
        return await asyncio.gather(*tasks)

    # ---------------------------------------------------------------------------
    # Training Batch Construction
    # ---------------------------------------------------------------------------

    def build_training_batch(
        self,
        results: List[GenerateResult],
    ) -> TrainingBatch:
        """
        Convert rollouts into trainer-consumable batch.

        Flattens all results (including children) into TrainingRecords.
        Each step's reward comes from step.reward (set by Rubric).
        """
        records: List[TrainingRecord] = []
        total_reward = 0.0
        total_steps = 0

        def process(result: GenerateResult) -> None:
            nonlocal total_reward, total_steps
            rollout = result.rollout

            for step in rollout.steps:
                if step.prompt_token_ids is None or step.completion_token_ids is None:
                    continue

                records.append(TrainingRecord(
                    role_id=step.role_id,
                    rollout_id=rollout.id,
                    prompt_token_ids=step.prompt_token_ids,
                    completion_token_ids=step.completion_token_ids,
                    logprobs=step.completion_logprobs or [],
                    reward=step.reward,
                    meta={
                        "episode_type": rollout.episode_type,
                        "seed": rollout.seed,
                    },
                ))

                total_reward += step.reward
                total_steps += 1

            for child in result.children:
                process(child)

        for result in results:
            process(result)

        meta = {
            "num_results": len(results),
            "num_records": len(records),
            "avg_reward": total_reward / total_steps if total_steps > 0 else 0.0,
        }

        return TrainingBatch(records=records, meta=meta)

    # ---------------------------------------------------------------------------
    # High-Level Training Step
    # ---------------------------------------------------------------------------

    async def step(self, concurrency: int = 8) -> TrainingBatch:
        """
        Execute one training step:
        1. Get batch of episode requests
        2. Generate rollouts in parallel (each episode's Rubric scores it)
        3. Build training batch
        """
        requests = self.get_batch()
        if not requests:
            return TrainingBatch(records=[], meta={"num_results": 0, "num_records": 0})

        results = await self.generate_rollouts(requests, concurrency=concurrency)
        ## TODO: assign credit to steps
        # List of generate results, where each generate result has a rollout and children, where children is a list of generate results
        # Goal should be to flatten (maybe?) and assign credit to each step
        # Rollout has a field steps which is a list of steps
        return self.build_training_batch(results)

    # ---------------------------------------------------------------------------
    # Convenience: Run Loop
    # ---------------------------------------------------------------------------

    async def run(
        self,
        num_steps: Optional[int] = None,
        concurrency: int = 8,
    ):
        """
        Run training loop, yielding batches.

        Args:
            num_steps: Number of steps to run (None = until get_batch returns empty)
            concurrency: Max parallel episodes
        """
        step_count = 0
        while num_steps is None or step_count < num_steps:
            batch = await self.step(concurrency=concurrency)
            if not batch.records:
                break
            yield batch
            step_count += 1

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    async def startup(self) -> None:
        """Called before training starts. Override for warmup."""
        pass

    async def shutdown(self) -> None:
        """Called after training ends. Override for cleanup."""
        pass
