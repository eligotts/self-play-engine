"""
Inference Server: OpenAI-compatible inference using MLX.

This module provides a high-performance inference server with:
- Continuous batching for efficient request processing
- Dynamic LoRA adapter hot-swapping for online learning
- OpenAI-compatible API endpoints

Core Components:
- ContinuousBatchingEngine: Manages efficient batch processing
- AsyncEngine: Async wrapper for HTTP server integration
- GenerationOutput: Result from completed generation

Usage:
    from legos.inference import ContinuousBatchingEngine, GenerationOutput

    engine = ContinuousBatchingEngine(model, tokenizer)
    request_id = engine.add(prompt_tokens, max_tokens=256)
    while engine.has_work():
        completed = engine.step()
"""

from legos.inference.engine import ContinuousBatchingEngine, GenerationOutput, AsyncEngine

__all__ = ["ContinuousBatchingEngine", "GenerationOutput", "AsyncEngine"]
