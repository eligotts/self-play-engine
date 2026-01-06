"""Continuous batching generation engine."""

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx

from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler


def _patch_arrays_cache():
    """Patch mlx_lm's ArraysCache to support batch extraction.

    ArraysCache is used by state-space and conv-based models (LFM2, RWKV7, etc.)
    but lacks the extract() method needed by BatchGenerator for managing
    completed sequences in a batch.
    """
    from mlx_lm.models.cache import ArraysCache

    if hasattr(ArraysCache, "extract"):
        return  # Already patched or upstream added support

    def extract(self, idx):
        """Extract a single cache from the batch at the given index."""
        new_cache = ArraysCache(size=len(self.cache))
        new_cache.cache = [
            c[idx : idx + 1] if c is not None else None for c in self.cache
        ]
        return new_cache

    ArraysCache.extract = extract


# Apply patch on module load
_patch_arrays_cache()


@dataclass
class GenerationOutput:
    """Result of a completed generation request."""

    request_id: int
    tokens: list[int]
    logprobs: list[float]
    finish_reason: Literal["stop", "length"]


@dataclass
class _ActiveRequest:
    """Internal tracking for an active generation."""

    request_id: int
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


class ContinuousBatchingEngine:
    """
    Continuous batching generation engine.

    Wraps mlx_lm's BatchGenerator with dynamic request management.
    Requests can be added at any time and will be scheduled into
    the batch as space becomes available.

    Usage:
        engine = ContinuousBatchingEngine(model, tokenizer)

        # Add requests (returns request_id for tracking)
        for prompt in prompts:
            engine.add(prompt, max_tokens=256)

        # Run until all complete
        outputs = []
        while engine.has_work():
            completed = engine.step()
            outputs.extend(completed)
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        default_max_tokens: int = 1024,
        extra_stop_tokens: set[int] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.default_max_tokens = default_max_tokens

        # pending: (request_id, prompt_tokens, max_tokens)
        self._pending: deque[tuple[int, list[int], int]] = deque()
        # active: batch_uid -> _ActiveRequest
        self._active: dict[int, _ActiveRequest] = {}
        self._request_id_counter = 0

        # Build stop tokens set
        stop_tokens = set(tokenizer.eos_token_ids)
        if extra_stop_tokens:
            stop_tokens |= extra_stop_tokens
        self.stop_tokens = stop_tokens

        self._generator = BatchGenerator(
            model,
            max_tokens=default_max_tokens,
            stop_tokens=stop_tokens,
            completion_batch_size=max_batch_size,
            prefill_batch_size=1,
            sampler=make_sampler(temp=0.7),
        )

    def add(self, prompt: list[int], max_tokens: int | None = None) -> int:
        """
        Add a generation request to the queue.

        Args:
            prompt: Tokenized prompt (list of token ids)
            max_tokens: Maximum tokens to generate

        Returns:
            request_id for tracking this request in outputs
        """
        request_id = self._request_id_counter
        self._request_id_counter += 1
        self._pending.append((
            request_id,
            prompt,
            max_tokens or self.default_max_tokens,
        ))
        return request_id

    def step(self) -> list[GenerationOutput]:
        """
        Run one generation step.

        - Fills batch with pending requests if space available
        - Runs one step of token generation
        - Returns list of any completed generations

        Returns:
            List of GenerationOutput for requests that finished this step
        """
        self._fill_batch()

        if not self._active:
            return []

        responses = self._generator.next()
        return self._process_responses(responses)

    def _fill_batch(self) -> None:
        """Add pending requests to batch if there's room."""
        available = self.max_batch_size - len(self._active)
        while self._pending and available > 0:
            request_id, prompt, max_tokens = self._pending.popleft()
            batch_uids = self._generator.insert([prompt], max_tokens=[max_tokens])
            self._active[batch_uids[0]] = _ActiveRequest(request_id=request_id)
            available -= 1

    def _process_responses(self, responses: list) -> list[GenerationOutput]:
        """Process responses from BatchGenerator, return completed ones."""
        completed = []
        for r in responses:
            active = self._active[r.uid]

            # Don't include stop tokens in output (they signal end, not content)
            if r.finish_reason != "stop":
                active.tokens.append(r.token)
                active.logprobs.append(r.logprobs[r.token].item())

            if r.finish_reason is not None:
                del self._active[r.uid]
                completed.append(GenerationOutput(
                    request_id=active.request_id,
                    tokens=active.tokens,
                    logprobs=active.logprobs,
                    finish_reason=r.finish_reason,
                ))
        return completed

    @property
    def num_pending(self) -> int:
        """Number of requests waiting to enter the batch."""
        return len(self._pending)

    @property
    def num_active(self) -> int:
        """Number of requests currently generating."""
        return len(self._active)

    def has_work(self) -> bool:
        """Returns True if there are pending or active requests."""
        return bool(self._pending or self._active)

    def close(self) -> None:
        """Clean up resources."""
        self._generator.close()

    def load_lora_weights(self, weights: dict[str, mx.array]) -> None:
        """
        Load LoRA adapter weights directly into the model.

        Safe to call between step() calls. At most 1 in-flight token
        uses old weights; subsequent tokens use new weights.

        Args:
            weights: Dict mapping param names to mx.array values
        """
        weight_items = list(weights.items())
        self.model.load_weights(weight_items, strict=False)
        self.model.eval()
