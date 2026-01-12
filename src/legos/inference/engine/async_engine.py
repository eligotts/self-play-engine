"""Async wrapper for the continuous batching engine."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import mlx.core as mx

from legos.inference.engine.generation import ContinuousBatchingEngine


@dataclass
class AsyncGenerationResult:
    """Result from async generation."""

    request_id: int
    tokens: list[int]
    logprobs: list[float]
    finish_reason: str
    prompt_tokens: int


@dataclass
class _WeightUpdate:
    """Pending weight update."""

    weights: dict[str, mx.array]
    version: int | None
    future: asyncio.Future


class AsyncEngine:
    """
    Async wrapper around ContinuousBatchingEngine.

    Runs the engine loop in a background task and provides async interface
    for submitting generation requests.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        default_max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id: str = getattr(tokenizer, "name_or_path", "unknown")

        # Build stop tokens
        extra_stop = self._get_extra_stop_tokens(tokenizer)

        self._engine = ContinuousBatchingEngine(
            model=model,
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            default_max_tokens=default_max_tokens,
            extra_stop_tokens=extra_stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # request_id -> (Future, prompt_token_count)
        self._pending_futures: dict[int, tuple[asyncio.Future[AsyncGenerationResult], int]] = {}
        self._loop_task: asyncio.Task | None = None
        self._running = False
        self._lora_version = 0

        # Thread pool for running blocking step() calls
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")

        # Queue for pending weight updates (applied between steps for thread safety)
        self._weight_update_queue: asyncio.Queue[_WeightUpdate] = asyncio.Queue()

    def _get_extra_stop_tokens(self, tokenizer) -> set[int]:
        """Get additional stop tokens beyond eos_token_ids."""
        extra = set()
        # Common stop tokens across different model families:
        # - <|im_end|>: Qwen, ChatML format
        # - <|eot_id|>: Llama 3
        # - <|end|>: Phi-3
        # - <|endoftext|>: GPT-2 style
        for token in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>"]:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                extra.add(ids[0])
        return extra - set(tokenizer.eos_token_ids)

    async def start(self) -> None:
        """Start the background engine loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._engine_loop())

    async def stop(self) -> None:
        """Stop the background engine loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._engine.close()
        self._executor.shutdown(wait=False)

    def _apply_pending_weight_updates(self) -> None:
        """Apply any pending weight updates. Called from engine loop between steps."""
        while True:
            try:
                update = self._weight_update_queue.get_nowait()
                self._engine.load_lora_weights(update.weights)
                if update.version is not None:
                    self._lora_version = update.version
                else:
                    self._lora_version += 1
                if not update.future.done():
                    update.future.set_result(self._lora_version)
            except asyncio.QueueEmpty:
                break

    async def _engine_loop(self) -> None:
        """Background loop that runs engine steps."""
        loop = asyncio.get_running_loop()
        while self._running:
            # Apply pending weight updates at safe point (between steps)
            self._apply_pending_weight_updates()

            if not self._engine.has_work():
                # No work to do, yield control and wait a bit
                await asyncio.sleep(0.001)
                continue

            # Run step in thread pool to avoid blocking the event loop.
            # This allows HTTP requests to be received while generation runs.
            completed = await loop.run_in_executor(self._executor, self._engine.step)

            # Resolve futures for completed requests
            for output in completed:
                if output.request_id in self._pending_futures:
                    future, prompt_tokens = self._pending_futures.pop(output.request_id)
                    result = AsyncGenerationResult(
                        request_id=output.request_id,
                        tokens=output.tokens,
                        logprobs=output.logprobs,
                        finish_reason=output.finish_reason,
                        prompt_tokens=prompt_tokens,
                    )
                    if not future.done():
                        future.set_result(result)

    async def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int | None = None,
    ) -> AsyncGenerationResult:
        """
        Submit a generation request and wait for completion.

        Args:
            prompt_tokens: Tokenized prompt
            max_tokens: Maximum tokens to generate

        Returns:
            AsyncGenerationResult with tokens, logprobs, and finish reason
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AsyncGenerationResult] = loop.create_future()

        request_id = self._engine.add(prompt_tokens, max_tokens)
        self._pending_futures[request_id] = (future, len(prompt_tokens))

        return await future

    @property
    def num_pending(self) -> int:
        """Number of requests waiting to enter the batch."""
        return self._engine.num_pending

    @property
    def num_active(self) -> int:
        """Number of requests currently generating."""
        return self._engine.num_active

    @property
    def is_ready(self) -> bool:
        """Check if engine is ready to accept requests."""
        return self._running

    @property
    def lora_version(self) -> int:
        """Current LoRA adapter version."""
        return self._lora_version

    async def load_lora_weights(
        self, weights: dict[str, mx.array], version: int | None = None
    ) -> int:
        """
        Load LoRA adapter weights into the model.

        Queues the update to be applied between generation steps, ensuring
        thread safety. The method returns once the weights are actually loaded.

        Args:
            weights: Dict mapping param names to mx.array values
            version: Optional explicit version number. If None, auto-increments.

        Returns:
            The new version number.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[int] = loop.create_future()
        await self._weight_update_queue.put(_WeightUpdate(weights, version, future))
        return await future
