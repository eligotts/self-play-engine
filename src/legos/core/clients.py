"""
Inference clients for model API endpoints.

InferenceClient: Abstract base class for all inference clients.
MockInferenceClient: Mock client for testing.
OpenAIClient: OpenAI-compatible client for mlx-vllm
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import httpx

from .types import Messages, ModelResponse


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

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

    async def publish_weights(self, model, version: int) -> Optional[dict]:
        """
        Push updated LoRA weights to the inference server.

        Only supported by local servers with LoRA hot-swap (e.g., mlx-vllm).
        Returns None if not supported. Override in subclasses that support it.

        Args:
            model: MLX model with trainable LoRA parameters
            version: Version number (typically training step)

        Returns:
            Response dict from server, or None if not supported
        """
        return None


# ---------------------------------------------------------------------------
# Mock Client (for testing)
# ---------------------------------------------------------------------------

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
# OpenAI-Compatible Client
# ---------------------------------------------------------------------------

class OpenAIClient(InferenceClient):
    """
    OpenAI-compatible inference client.

    Works with OpenAI, OpenRouter, local servers (mlx-vllm), and any OpenAI-compatible API.
    Gracefully handles missing logprobs (returns None for token fields).

    Examples:
        # OpenRouter (hosted API)
        client = OpenAIClient(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
        )

        # Local mlx-vllm server (no API key needed)
        client = OpenAIClient(
            api_key="not-needed",
            model="local",  # ignored by server
            base_url="http://localhost:8000/v1",
        )

        # OpenAI directly
        client = OpenAIClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
    ):
        # Try multiple env vars for API key
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._warned_missing_tokens = False

        # For local servers, any non-empty key works
        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not self.api_key:
            if is_local:
                self.api_key = "not-needed"  # Local servers don't require auth
            else:
                raise ValueError(
                    "API key required for remote APIs. Set OPENROUTER_API_KEY or OPENAI_API_KEY env var, or pass api_key."
                )

        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._call_count = 0

    async def complete(
        self,
        messages: Messages,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        return_tokens: bool = True,
    ) -> ModelResponse:
        self._call_count += 1

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Request logprobs if available (OpenAI-style)
        if return_tokens:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1
            # Request real token IDs (supported by local mlx-vllm server)
            payload["return_token_ids"] = True

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter-specific headers (only for openrouter.ai)
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/eligottlieb/legos"
            headers["X-Title"] = "Legos"

        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]
        text = message.get("content", "")

        # Extract token IDs and logprobs
        prompt_token_ids = None
        completion_token_ids = None
        completion_logprobs = None

        if return_tokens:
            # Try to get real token IDs from server (mlx-vllm style)
            # prompt_token_ids is at top level, token_ids is inside choice
            real_prompt_ids = data.get("prompt_token_ids")
            real_completion_ids = choice.get("token_ids")

            # Extract logprobs
            logprobs_data = choice.get("logprobs")
            if (
                not self._warned_missing_tokens
                and (real_prompt_ids is None or real_completion_ids is None or not logprobs_data)
            ):
                print(
                    "[legos] WARNING: Response missing token IDs/logprobs; "
                    "using hashed tokens. Remote APIs are not supported for training.",
                    file=sys.stderr,
                )
                self._warned_missing_tokens = True
            if logprobs_data and logprobs_data.get("content"):
                content_logprobs = logprobs_data["content"]
                completion_logprobs = [token_info.get("logprob", 0.0) for token_info in content_logprobs]

                # If we have real completion token IDs, use them
                if real_completion_ids is not None:
                    completion_token_ids = real_completion_ids
                else:
                    # Fallback: use hash of token string as fake token ID
                    completion_token_ids = [
                        hash(token_info.get("token", "")) % (2**31)
                        for token_info in content_logprobs
                    ]
            else:
                # API doesn't support logprobs - generate fake token IDs for debugging
                # This allows the orchestration to work even without real token data
                if real_completion_ids is not None:
                    completion_token_ids = real_completion_ids
                    completion_logprobs = [-1.0] * len(completion_token_ids)
                else:
                    words = text.split()
                    completion_token_ids = [hash(w) % (2**31) for w in words] if words else [0]
                    completion_logprobs = [-1.0] * len(completion_token_ids)

            # Use real prompt token IDs if available, otherwise fake them
            if real_prompt_ids is not None:
                prompt_token_ids = real_prompt_ids
            else:
                # Generate fake prompt token IDs (OpenRouter/OpenAI don't return these)
                prompt_text = " ".join(m.get("content", "") for m in messages)
                prompt_words = prompt_text.split()[:50]  # Cap at 50 for efficiency
                prompt_token_ids = [hash(w) % (2**31) for w in prompt_words] if prompt_words else [0]

        return ModelResponse(
            text=text,
            completion=[{"role": "assistant", "content": text}],
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            completion_logprobs=completion_logprobs,
        )

    async def get_policy_version(self) -> int:
        """
        Get the current policy version from the inference server.

        Hits the /adapters/version endpoint (used by mlx-vllm for LoRA hot-swap).
        Returns 0 if the endpoint doesn't exist or the request fails.
        """
        # Compute base URL without /v1 suffix
        base = self.base_url
        if base.endswith("/v1"):
            base = base[:-3]

        try:
            response = await self._client.get(
                f"{base}/adapters/version",
                timeout=5.0,  # Short timeout for version check
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("version", 0)
            return 0
        except Exception:
            # Connection refused, timeout, 404, etc. - all return 0
            return 0

    async def publish_weights(self, model, version: int) -> Optional[dict]:
        """
        Push updated LoRA weights to the inference server.

        Works with any server that supports the /adapters/load endpoint
        (e.g., mlx-vllm with LoRA hot-swap).

        Args:
            model: MLX model with trainable LoRA parameters
            version: Version number (typically training step)

        Returns:
            Response dict from server with {"status": "ok", "version": N}
        """
        # Lazy imports to avoid forcing MLX deps on inference-only usage
        import base64
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        from safetensors.numpy import save as save_safetensors

        # Extract LoRA weights (trainable parameters only)
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))

        if not adapter_weights:
            raise ValueError("Model has no trainable parameters. Did you attach LoRA?")

        # Evaluate to materialize lazy arrays
        mx.eval(adapter_weights)

        # Convert to numpy (cast bfloat16 to float32 since numpy doesn't support bf16)
        def to_numpy(arr: mx.array) -> np.ndarray:
            if arr.dtype == mx.bfloat16:
                return np.array(arr.astype(mx.float32))
            return np.array(arr)

        weights_np = {k: to_numpy(v) for k, v in adapter_weights.items()}

        # Serialize to safetensors and base64 encode
        weight_bytes = save_safetensors(weights_np)
        weights_b64 = base64.b64encode(weight_bytes).decode("utf-8")

        # Compute base URL without /v1 suffix
        base = self.base_url
        if base.endswith("/v1"):
            base = base[:-3]

        # POST to /adapters/load
        response = await self._client.post(
            f"{base}/adapters/load",
            json={"weights": weights_b64, "version": version},
            timeout=30.0,
        )
        response.raise_for_status()
        result = response.json()

        # Clean up intermediate data to free memory
        del adapter_weights
        del weights_np
        del weight_bytes
        del weights_b64
        mx.clear_cache()

        return result

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    @property
    def call_count(self) -> int:
        return self._call_count
