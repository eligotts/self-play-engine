"""
LoRA configuration for training and inference.

Edit LORA_CONFIG below to set your LoRA parameters.
Both training scripts and the inference server read from this file.
"""

# =============================================================================
# LORA CONFIGURATION - Edit these values
# =============================================================================

LORA_CONFIG = {
    "rank": 8,           # LoRA rank (bottleneck dimension). Higher = more capacity.
    "layers": 16,         # Number of layers to apply LoRA to. -1 = all layers.
    "scale": 16.0,        # Scaling factor (lora_alpha). Controls update magnitude.
    "dropout": 0.05,      # Dropout rate. Set to 0.0 for inference.
    "keys": {             # Which linear layers to apply LoRA to.
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    },
}

# =============================================================================


def apply_lora(model, inference_mode: bool = False, verbose: bool = True):
    """
    Apply LoRA adapters to a model using LORA_CONFIG.

    Args:
        model: MLX model to modify in-place.
        inference_mode: If True, sets dropout=0 and calls model.eval().
        verbose: If True, prints configuration info.

    Returns:
        The modified model (same object, modified in-place).
    """
    from mlx_lm.tuner.utils import linear_to_lora_layers

    config = LORA_CONFIG
    mlx_config = {
        "rank": config["rank"],
        "scale": config["scale"],
        "dropout": 0.0 if inference_mode else config["dropout"],
        "keys": config["keys"],
    }
    linear_to_lora_layers(model, config["layers"], mlx_config)

    if inference_mode:
        model.eval()

    if verbose:
        mode = "inference" if inference_mode else "training"
        print(f"LoRA applied ({mode}): rank={config['rank']}, layers={config['layers']}, "
              f"scale={config['scale']}, keys={sorted(config['keys'])}")

    return model


def print_trainable_params(model):
    """Print trainable parameter summary."""
    from mlx.utils import tree_flatten

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
