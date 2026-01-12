"""FastAPI server application."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlx_lm import load

from legos.inference.api import adapter_router, router
from legos.inference.config import ServerConfig
from legos.inference.engine.async_engine import AsyncEngine

# Global config - can be overridden before create_app() is called
_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config


def set_config(config: ServerConfig) -> None:
    global _config
    _config = config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - load model and start engine."""
    config = get_config()

    print(f"Loading model: {config.model_path}")
    start = time.time()
    model, tokenizer = load(config.model_path)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Initialize LoRA layers (for dynamic weight updates)
    if config.enable_lora:
        from legos.lora import apply_lora
        apply_lora(model, inference_mode=True)

    engine = AsyncEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=config.max_batch_size,
        default_max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
    )
    await engine.start()
    print("Engine started")

    # Store in app state for routes to access
    app.state.engine = engine
    app.state.config = config
    app.state.model_loaded_at = int(time.time())

    yield

    # Cleanup
    print("Shutting down engine...")
    await engine.stop()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Self-Play Inference Server",
        description="OpenAI-compatible inference server using MLX",
        version="0.1.0",
        lifespan=lifespan,
    )

    # OpenAI-style error handler
    @app.exception_handler(Exception)
    async def openai_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "code": None,
                }
            },
        )

    # Root-level health check (for load balancers)
    @app.get("/health")
    async def root_health() -> dict:
        engine = app.state.engine
        return {
            "status": "healthy" if engine.is_ready else "not_ready",
            "model": engine.model_id,
        }

    app.include_router(router)
    app.include_router(adapter_router)
    return app


app = create_app()
