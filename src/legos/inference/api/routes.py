"""OpenAI-compatible API routes."""

import base64
import math
import time
import uuid

import mlx.core as mx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from safetensors.numpy import load as load_safetensors

from legos.inference.engine.async_engine import AsyncEngine
from legos.inference.types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChoiceLogprobs,
    LoadAdapterRequest,
    ModelInfo,
    ModelListResponse,
    TokenLogprob,
    Usage,
)

router = APIRouter(prefix="/v1")

# Adapter management endpoints (outside /v1 namespace)
adapter_router = APIRouter(prefix="/adapters", tags=["adapters"])


def get_engine(request: Request) -> AsyncEngine:
    """Get the engine from app state."""
    return request.app.state.engine


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse | JSONResponse:
    """Generate a chat completion."""
    engine = get_engine(request)

    if body.stream:
        # Streaming not yet implemented
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Streaming not yet implemented",
                    "type": "not_implemented",
                    "code": None,
                }
            },
        )

    # Apply chat template and tokenize
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    prompt = engine.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = engine.tokenizer.encode(prompt)

    # Submit to engine and wait for completion
    max_tokens = body.max_tokens or request.app.state.config.max_tokens
    result = await engine.generate(prompt_tokens, max_tokens=max_tokens)

    # Decode the response
    response_text = engine.tokenizer.decode(result.tokens)

    # Build logprobs if requested
    logprobs = None
    if body.logprobs:
        token_logprobs = []
        for token_id, logprob in zip(result.tokens, result.logprobs):
            # Safe JSON serialization: replace NaN with a very low logprob
            if math.isnan(logprob):
                print(f"[inference] WARNING: NaN logprob for token {token_id}, using -100.0")
                safe_logprob = -100.0
            else:
                safe_logprob = logprob
            token_str = engine.tokenizer.decode([token_id])
            token_logprobs.append(
                TokenLogprob(
                    token=token_str,
                    logprob=safe_logprob,
                    bytes=list(token_str.encode("utf-8")),
                )
            )
        logprobs = ChoiceLogprobs(content=token_logprobs)

    # Build response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=engine.model_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason=result.finish_reason,
                logprobs=logprobs,
                token_ids=result.tokens if body.return_token_ids else None,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=len(result.tokens),
            total_tokens=result.prompt_tokens + len(result.tokens),
        ),
        prompt_token_ids=prompt_tokens if body.return_token_ids else None,
    )


@router.get("/models")
async def list_models(request: Request) -> ModelListResponse:
    """List available models."""
    engine = get_engine(request)
    return ModelListResponse(
        data=[
            ModelInfo(
                id=engine.model_id,
                created=request.app.state.model_loaded_at,
            )
        ]
    )


@router.get("/health")
async def health(request: Request) -> dict:
    """Health check endpoint."""
    engine = get_engine(request)
    return {
        "status": "healthy" if engine.is_ready else "not_ready",
        "model": engine.model_id,
        "pending_requests": engine.num_pending,
        "active_requests": engine.num_active,
    }


# ---- Adapter management endpoints ----


@adapter_router.post("/load")
async def load_adapter(request: Request, body: LoadAdapterRequest) -> dict:
    """
    Load LoRA adapter weights into the running model.

    Weights are base64-encoded safetensors bytes. The weights are applied
    immediately. At most 1 in-flight token uses old weights; subsequent
    tokens use new weights.
    """
    engine = get_engine(request)

    # Decode base64
    weight_bytes = base64.b64decode(body.weights)

    # Parse safetensors
    weights_np = load_safetensors(weight_bytes)

    # Convert numpy arrays to mx arrays
    weights_mx = {k: mx.array(v) for k, v in weights_np.items()}

    # Load into model (async - queued and applied between generation steps)
    new_version = await engine.load_lora_weights(weights_mx, body.version)

    return {
        "status": "ok",
        "version": new_version,
    }


@adapter_router.get("/version")
async def get_adapter_version(request: Request) -> dict:
    """Get current LoRA adapter version."""
    engine = get_engine(request)
    return {"version": engine.lora_version}
