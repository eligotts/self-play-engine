"""
Inference clients for real API endpoints.

OpenAIClient: OpenAI-compatible client for OpenAI, OpenRouter, and local servers.
Handles missing logprobs gracefully (common with hosted APIs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .arena import InferenceClient, ModelResponse
from .types import Messages


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI-compatible API."""
    api_key: str
    model: str = "openai/gpt-4o-mini"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 60.0


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

    @classmethod
    def for_local(cls, port: int = 8000, timeout: float = 120.0) -> "OpenAIClient":
        """Create a client for a local OpenAI-compatible server (e.g., mlx-vllm).

        Args:
            port: The port the server is running on (default: 8000)
            timeout: Request timeout in seconds (default: 120s for local inference)

        Returns:
            OpenAIClient configured for localhost
        """
        return cls(
            api_key="not-needed",
            model="local",
            base_url=f"http://localhost:{port}/v1",
            timeout=timeout,
        )

    @classmethod
    def for_openrouter(
        cls,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        timeout: float = 60.0,
    ) -> "OpenAIClient":
        """Create a client for OpenRouter.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model ID (default: openai/gpt-4o-mini)
            timeout: Request timeout in seconds

        Returns:
            OpenAIClient configured for OpenRouter
        """
        return cls(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
        )

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
            headers["HTTP-Referer"] = "https://github.com/self-play-engine"
            headers["X-Title"] = "Self-Play Engine"

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

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    @property
    def call_count(self) -> int:
        return self._call_count
