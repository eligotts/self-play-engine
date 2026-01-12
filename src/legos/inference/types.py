"""OpenAI API-compatible types for /v1/chat/completions and /v1/models."""

from typing import Literal

from pydantic import BaseModel

# Chat completions request


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    logprobs: bool = False
    return_token_ids: bool = False  # Return prompt_token_ids and token_ids


# Chat completions response - logprobs


class TokenLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class ChoiceLogprobs(BaseModel):
    content: list[TokenLogprob] | None = None


# Chat completions response


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] | None = None
    logprobs: ChoiceLogprobs | None = None
    token_ids: list[int] | None = None  # Completion token IDs (unpadded)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    prompt_token_ids: list[int] | None = None  # Prompt token IDs (unpadded)


# Models endpoint


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "legos"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# LoRA adapter loading


class LoadAdapterRequest(BaseModel):
    weights: str  # Base64-encoded safetensors bytes
    version: int | None = None  # Optional explicit version number
