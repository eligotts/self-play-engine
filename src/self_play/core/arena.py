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
from .credit import CreditAssigner, apply_credit


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

    async def get_policy_version(self) -> int:
        """
        Get the current policy version from the inference server.

        Returns 0 if the server doesn't support versioning (e.g., OpenRouter, OpenAI).
        Override in subclasses that support LoRA hot-swap or similar.
        """
        return 0


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
    """Request to run an episode with a resolved artifact payload."""
    episode_type: str
    artifact: Any
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

    def __init__(
        self,
        client: InferenceClient,
        credit_assigner: Optional[CreditAssigner] = None,
        verbose: bool = False,
    ):
        self.client = client
        self.credit_assigner = credit_assigner
        self.verbose = verbose
        self._call_counter = 0

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
        self._call_counter += 1
        call_id = self._call_counter
        caller = role_id if role_id else "judge"

        try:
            if role_id is not None:
                role = self.roles[role_id]
                response = await self.client.complete(
                    messages=messages,
                    temperature=temperature if temperature is not None else role.temperature,
                    max_tokens=max_tokens if max_tokens is not None else role.max_tokens,
                    return_tokens=True,
                )
            else:
                response = await self.client.complete(
                    messages=messages,
                    temperature=temperature if temperature is not None else 1.0,
                    max_tokens=max_tokens,
                    return_tokens=return_tokens if return_tokens is not None else False,
                )

            # Print all output atomically after response to avoid race conditions
            if self.verbose:
                lines = [f"\n    [call #{call_id}] {caller}"]
                for msg in messages:
                    msg_role = msg.get("role", "?")
                    content = msg.get("content", "")
                    lines.append(f"      [{msg_role}] {content}")
                lines.append(f"      [response] {response.text if response.text else '(empty)'}")
                print("\n".join(lines))

            return response

        except Exception as e:
            if self.verbose:
                print(f"\n    [call #{call_id}] {caller}\n      [ERROR] {type(e).__name__}: {e}")
            raise

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
        artifact: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Run a single episode."""
        episode = self.episodes[episode_type]
        return await episode.generate(self, artifact, meta=meta)

    async def resolve_artifact(self, request: EpisodeRequest) -> Any:
        """
        Resolve the artifact for a request.

        Override to load artifacts from external stores or services.
        """
        return request.artifact

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
                artifact = await self.resolve_artifact(request)
                return await self.run_episode(
                    request.episode_type,
                    artifact,
                    meta=request.meta,
                )

        tasks = [asyncio.create_task(run_one(req)) for req in requests]
        return await asyncio.gather(*tasks)

    # ---------------------------------------------------------------------------
    # Training Batch Construction
    # ---------------------------------------------------------------------------

    def build_training_batch(
        self,
        results: List[GenerateResult],
        verbose: bool = False,
    ) -> TrainingBatch:
        """
        Convert rollouts into trainer-consumable batch.

        Flattens all results (including children) into TrainingRecords.
        Each step's reward comes from step.reward (set by Rubric).
        """
        records: List[TrainingRecord] = []
        skipped_steps = 0

        def process(result: GenerateResult) -> None:
            nonlocal skipped_steps
            rollout = result.rollout

            for step in rollout.steps:
                if step.prompt_token_ids is None or step.completion_token_ids is None:
                    skipped_steps += 1
                    if verbose:
                        print(f"    [build] skipping step {step.role_id}: no token IDs")
                    continue

                action_mask = [0] * len(step.prompt_token_ids) + [1] * len(step.completion_token_ids)

                # maybe here just not add the record if advantage is 0?

                records.append(TrainingRecord(
                    role_id=step.role_id,
                    rollout_id=rollout.id,
                    prompt_token_ids=step.prompt_token_ids,
                    completion_token_ids=step.completion_token_ids,
                    logprobs=step.completion_logprobs or [],
                    action_mask=action_mask,
                    reward=step.reward,
                    advantage=step.advantage,
                    meta={
                        "episode_type": rollout.episode_type,
                        **rollout.meta,
                    },
                ))

            for child in result.children:
                process(child)

        for result in results:
            process(result)

        meta = {
            "num_results": len(results),
            "num_records": len(records),
        }

        return TrainingBatch(records=records, meta=meta)

    def sanity_check_batch(
        self,
        results: List[GenerateResult],
        batch: TrainingBatch,
        num_examples: int = 2,
    ) -> None:
        """
        Validate that build_training_batch produced correct output.

        Checks:
        1. logprobs length == completion_token_ids length for each record
        2. action_mask length == prompt_token_ids + completion_token_ids
        3. action_mask has correct structure (0s then 1s)
        4. Total records == total steps across all rollouts (minus skipped)

        Prints a few example records for inspection.
        """
        print("\n" + "=" * 60)
        print("SANITY CHECK: build_training_batch")
        print("=" * 60)

        # Count expected steps from results
        def count_steps(result: GenerateResult) -> tuple[int, int]:
            """Returns (total_steps, trainable_steps)"""
            total = 0
            trainable = 0
            for step in result.rollout.steps:
                total += 1
                if step.prompt_token_ids is not None and step.completion_token_ids is not None:
                    trainable += 1
            for child in result.children:
                child_total, child_trainable = count_steps(child)
                total += child_total
                trainable += child_trainable
            return total, trainable

        total_steps = 0
        trainable_steps = 0
        for result in results:
            t, tr = count_steps(result)
            total_steps += t
            trainable_steps += tr

        print(f"\nRollout Summary:")
        print(f"  Total results: {len(results)}")
        print(f"  Total steps across all rollouts: {total_steps}")
        print(f"  Trainable steps (have token IDs): {trainable_steps}")
        print(f"  Skipped steps (no token IDs): {total_steps - trainable_steps}")
        print(f"  TrainingRecords created: {len(batch.records)}")

        # Check: records count matches trainable steps
        if len(batch.records) == trainable_steps:
            print(f"  ✓ Record count matches trainable step count")
        else:
            print(f"  ✗ MISMATCH: {len(batch.records)} records vs {trainable_steps} trainable steps")

        # Show reward/advantage distribution by hierarchy level
        def collect_rewards(result: GenerateResult, level: int, data: list) -> None:
            for step in result.rollout.steps:
                data.append({
                    "level": level,
                    "episode_type": result.rollout.episode_type,
                    "role_id": step.role_id,
                    "reward": step.reward,
                    "advantage": step.advantage,
                })
            for child in result.children:
                collect_rewards(child, level + 1, data)

        all_steps_data: list = []
        for result in results:
            collect_rewards(result, 0, all_steps_data)

        print(f"\nReward/Advantage Distribution:")
        # Group by (level, episode_type, role_id)
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for d in all_steps_data:
            key = (d["level"], d["episode_type"], d["role_id"])
            groups[key].append((d["reward"], d["advantage"]))

        for (level, ep_type, role_id), values in sorted(groups.items()):
            rewards = [v[0] for v in values]
            advantages = [v[1] for v in values]
            print(f"  Level {level} | {ep_type} | {role_id}: {len(values)} steps")
            print(f"    rewards: {rewards}")
            print(f"    advantages: {advantages}")

        # Validate each record
        errors = []
        for i, record in enumerate(batch.records):
            # Check logprobs length
            if len(record.logprobs) != len(record.completion_token_ids):
                errors.append(
                    f"Record {i} ({record.role_id}): logprobs length {len(record.logprobs)} "
                    f"!= completion_token_ids length {len(record.completion_token_ids)}"
                )

            # Check action_mask length
            expected_mask_len = len(record.prompt_token_ids) + len(record.completion_token_ids)
            if len(record.action_mask) != expected_mask_len:
                errors.append(
                    f"Record {i} ({record.role_id}): action_mask length {len(record.action_mask)} "
                    f"!= expected {expected_mask_len}"
                )

            # Check action_mask structure (0s for prompt, 1s for completion)
            prompt_len = len(record.prompt_token_ids)
            completion_len = len(record.completion_token_ids)
            expected_mask = [0] * prompt_len + [1] * completion_len
            if record.action_mask != expected_mask:
                errors.append(
                    f"Record {i} ({record.role_id}): action_mask has incorrect structure"
                )

        if errors:
            print(f"\n✗ ERRORS FOUND ({len(errors)}):")
            for err in errors[:10]:  # Show first 10 errors
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        else:
            print(f"\n✓ All {len(batch.records)} records passed validation:")
            print(f"  - logprobs length == completion_token_ids length")
            print(f"  - action_mask length == total tokens")
            print(f"  - action_mask structure correct (0s for prompt, 1s for completion)")

        # Print example records
        print(f"\n--- Example TrainingRecords (first {min(num_examples, len(batch.records))}) ---")
        for i, record in enumerate(batch.records[:num_examples]):
            print(f"\nRecord {i}:")
            print(f"  role_id: {record.role_id}")
            print(f"  rollout_id: {record.rollout_id[:8]}...")
            print(f"  prompt_token_ids: {len(record.prompt_token_ids)} tokens")
            print(f"  completion_token_ids: {len(record.completion_token_ids)} tokens")
            print(f"  logprobs: {len(record.logprobs)} values (sum={sum(record.logprobs):.4f})")
            print(f"  action_mask: {len(record.action_mask)} values ({sum(record.action_mask)} ones, {len(record.action_mask) - sum(record.action_mask)} zeros)")
            print(f"  reward: {record.reward:.4f}")
            print(f"  advantage: {record.advantage:.4f}")
            print(f"  meta: {record.meta}")

        print("\n" + "=" * 60)

    # ---------------------------------------------------------------------------
    # High-Level Training Step
    # ---------------------------------------------------------------------------

    async def step(self, concurrency: int = 8, verbose: bool = False) -> TrainingBatch:
        """
        Execute one training step:
        1. Get batch of episode requests
        2. Tag each request with current policy version
        3. Generate rollouts in parallel (each episode's Rubric scores it)
        4. Build training batch
        """
        requests = self.get_batch()
        if verbose:
            print(f"  [step] get_batch() returned {len(requests)} requests")
        if not requests:
            return TrainingBatch(records=[], meta={"num_results": 0, "num_records": 0})

        # Fetch current policy version and tag all requests
        policy_version = await self.client.get_policy_version()
        for req in requests:
            req.meta["policy_version"] = policy_version
        if verbose:
            print(f"  [step] tagged requests with policy_version={policy_version}")

        if verbose:
            for i, req in enumerate(requests):
                print(f"    [{i}] {req.episode_type}: {req.artifact}")

        results = await self.generate_rollouts(requests, concurrency=concurrency)
        if verbose:
            print(f"  [step] generate_rollouts() returned {len(results)} results")
            for i, res in enumerate(results):
                print(f"    [{i}] {res.rollout.episode_type}: {len(res.rollout.steps)} steps, rewards={res.rewards}")

        # Assign credit to steps
        if self.credit_assigner is not None:
            weights = self.credit_assigner.compute(results)
            apply_credit(results, weights)

        batch = self.build_training_batch(results, verbose=verbose)
        if verbose:
            print(f"  [step] build_training_batch() returned {len(batch.records)} records")
            self.sanity_check_batch(results, batch)
        return batch

    # ---------------------------------------------------------------------------
    # Convenience: Run Loop
    # ---------------------------------------------------------------------------

    async def run(
        self,
        num_steps: Optional[int] = None,
        concurrency: int = 8,
        verbose: bool = False,
    ):
        """
        Run training loop, yielding batches.

        Args:
            num_steps: Number of steps to run (None = until get_batch returns empty)
            concurrency: Max parallel episodes
            verbose: Print debug info to terminal
        """
        step_count = 0
        while num_steps is None or step_count < num_steps:
            if verbose:
                print(f"\n[run] Starting step {step_count + 1}")
            batch = await self.step(concurrency=concurrency, verbose=verbose)
            if not batch.records:
                if verbose:
                    print(f"[run] No records in batch, stopping")
                break
            yield batch
            step_count += 1
        if verbose:
            print(f"[run] Completed {step_count} steps")

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    async def startup(self) -> None:
        """Called before training starts. Override for warmup."""
        pass

    async def shutdown(self) -> None:
        """Called after training ends. Override for cleanup."""
        pass
