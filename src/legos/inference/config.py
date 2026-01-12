"""Server configuration."""

from pydantic import BaseModel


class ServerConfig(BaseModel):
    """Configuration for inference server."""

    model_path: str = "mlx_model"
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_tokens: int = 4096

    # Sampler defaults
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0

    # LoRA - uses settings from lora.py
    enable_lora: bool = True
