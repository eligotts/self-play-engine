# Training Infrastructure

This document covers the RL training infrastructure built on MLX for Apple Silicon. For self-play abstractions, see [concepts.md](concepts.md).

---

## Architecture Overview

The training system uses a **producer-consumer architecture**:

```
┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│   Arena     │────▶│  Async Queue  │────▶│   Trainer   │
│ (Generator) │     │               │     │             │
└─────────────┘     └───────────────┘     └─────────────┘
       │                                         │
       │         ┌───────────────────┐           │
       └────────▶│ Inference Server  │◀──────────┘
                 │ (LoRA hot-swap)   │
                 └───────────────────┘
```

- **Arena** generates rollouts, scores them, and pushes `TrainingBatch` objects to a queue
- **Trainer** pulls batches, computes loss, updates weights
- **Inference Server** serves the model with LoRA adapters that update in real-time

Generation and training run concurrently—the arena doesn't wait for the trainer, and the trainer doesn't wait for the arena.

---

## Training Loops

### Asynchronous Loop

The default `training_loop()` runs generation and training concurrently:

```python
async def training_loop(
    arena: Arena,
    trainer: Trainer,
    num_steps: int,
    episode_concurrency: int = 8,
    step_concurrency: int = 2,
):
    queue = asyncio.Queue(maxsize=step_concurrency)

    async def generator():
        for _ in range(num_steps):
            batch = await arena.step(concurrency=episode_concurrency)
            await queue.put(batch)

    async def consumer():
        for _ in range(num_steps):
            batch = await queue.get()
            trainer.train_step(batch)

    await asyncio.gather(generator(), consumer())
```

Benefits:
- Higher throughput: generation doesn't block training
- Natural backpressure: queue limits prevent memory explosion
- Efficient GPU utilization

### Synchronous Loop

For debugging or when you need strict ordering:

```python
async def synchronous_training_loop(arena, trainer, num_steps):
    for step in range(num_steps):
        batch = await arena.step()
        trainer.train_step(batch)
```

Simpler, easier to debug, but lower throughput.

---

## Micro-Batch Streaming

Training records have variable lengths. Instead of padding everything to max length, the trainer uses **token-budget-based micro-batching**:

```python
def form_micro_batch(records, token_budget):
    """Greedily add records until token budget is exhausted."""
    batch = []
    tokens = 0
    for record in records:
        record_tokens = len(record.input_ids)
        if tokens + record_tokens > token_budget:
            break
        batch.append(record)
        tokens += record_tokens
    return batch
```

Benefits:
- Consistent memory usage regardless of sequence length distribution
- No wasted compute on padding tokens
- Works with mixed-length batches (e.g., different episode types)

### Gradient Accumulation

Gradients accumulate across micro-batches until `min_samples_per_step` is reached:

```python
class Trainer:
    def train_step(self, batch):
        for micro_batch in form_micro_batches(batch.records):
            loss, grads = value_and_grad(self.loss_fn)(micro_batch)
            self.accumulated_grads += grads * weight

        if self.accumulated_samples >= self.min_samples_per_step:
            self.optimizer.step(self.accumulated_grads)
            self.accumulated_grads = None
            self.accumulated_samples = 0
```

---

## Importance Sampling

Off-policy training requires importance sampling to correct for distribution shift.

### Token-Level (PPO-Style)

The default mode computes per-token importance ratios:

```python
log_ratio = trainer_logprobs - inference_logprobs  # Per-token
importance_ratio = exp(log_ratio)

# PPO clipping
clipped_ratio = clip(importance_ratio, 1 - epsilon, 1 + epsilon)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

Typical clipping range: `[0.8, 1.2]`

### Sequence-Level (GSPO)

For tighter control, sequence-level importance sampling:

```python
# Average log ratio across completion tokens
seq_log_ratio = mean(log_ratio[mask])
si = exp(seq_log_ratio)

# Much tighter clipping
si_clipped = clip(si, 1 - epsilon, 1 + epsilon)  # epsilon ≈ 3e-4
```

Based on [arXiv:2512.21852](https://arxiv.org/abs/2512.21852). Prevents large policy updates from single sequences.

### Clip Skip Threshold

If too many tokens are clipped, the batch is likely off-policy garbage:

```python
clip_fraction = (importance_ratio < 0.8 | importance_ratio > 1.2).mean()
if clip_fraction > clip_skip_threshold:  # Default: 0.3
    return zero_loss  # Skip this batch entirely
```

This prevents training on severely divergent samples.

---

## Staleness Filtering

Rollouts are tagged with their policy version at generation time:

```python
# In arena.step()
version = await self.client.get_policy_version()
request.meta["policy_version"] = version
```

The trainer filters out stale records:

```python
def filter_fresh(records, current_version, staleness_limit=3):
    return [
        r for r in records
        if current_version - r.meta["policy_version"] <= staleness_limit
    ]
```

This prevents training on data generated by an old policy, which could cause distribution shift.

---

## Loss Functions

### GRPO vs DAPO Normalization

Two normalization modes for the policy gradient loss:

**Token-level (DAPO)**:
```python
loss = masked_loss.sum() / mask.sum()
```
Equal weight per token. Longer completions contribute more.

**Sample-level (GRPO)**:
```python
loss = (masked_loss.sum(dim=-1) / mask.sum(dim=-1)).mean()
```
Equal weight per sample. Length-invariant.

### KL Penalty

Regularize against a reference policy to prevent collapse:

```python
kl = trainer_logprobs - reference_logprobs
kl_penalty = kl_coef * max(kl, 0)  # One-sided (K1 shaping)
```

The reference policy is the base model (without LoRA). We compute it efficiently by temporarily setting LoRA scales to zero:

```python
@contextmanager
def disable_lora(model):
    scales = save_lora_scales(model)
    set_lora_scales(model, 0.0)
    yield
    restore_lora_scales(model, scales)

with disable_lora(model):
    ref_logprobs = model(input_ids)
```

No need for a separate reference model in memory.

---

## LoRA Hot-Swap

The inference server supports hot-swapping LoRA weights without restart.

### Weight Publishing

After each training step, the trainer publishes weights:

```python
class Trainer:
    async def publish_weights(self, server_url):
        weights = get_lora_weights(self.model)
        encoded = base64_encode(safetensors_serialize(weights))

        await httpx.post(
            f"{server_url}/adapters/load",
            json={"weights": encoded, "version": self.train_step_idx}
        )
```

### Async Application

The inference server applies weights between generation steps:

```python
class AsyncEngine:
    async def _engine_loop(self):
        while True:
            # Check for weight updates
            if not self._weight_update_queue.empty():
                update = await self._weight_update_queue.get()
                self._apply_weights(update.weights)
                self._policy_version = update.version

            # Process inference requests
            await self._generation_step()
```

This ensures thread safety—weights only update between requests, never during.

---

## Inference Server

The inference server provides an OpenAI-compatible API with continuous batching.

### Endpoints

```
POST /v1/chat/completions  # Standard chat completions
GET  /v1/models            # List available models
POST /adapters/load        # Hot-swap LoRA weights
GET  /adapters/version     # Current policy version
```

### Continuous Batching

Instead of waiting for fixed batch sizes, the engine processes requests as they arrive:

```python
class ContinuousBatchingEngine:
    async def step(self):
        # Gather pending requests
        requests = self._pending_requests[:max_batch_size]

        # Generate one token for each
        logits = self.model(input_ids)
        next_tokens = sample(logits)

        # Some requests may complete
        for request in requests:
            if request.is_complete():
                request.future.set_result(request.output)
```

Benefits:
- Low latency for short requests
- High throughput for long requests
- Efficient GPU utilization

### Configuration

```python
uv run legos serve \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --port 8000
```

---

## Memory Management

MLX on Apple Silicon requires careful memory management to avoid lazy evaluation accumulating computation graphs.

### Explicit Materialization

After each micro-batch:

```python
mx.eval(self.model.parameters())
mx.eval(self.optimizer.state)
mx.clear_cache()
```

This forces computation and releases intermediate tensors.

### Gradient Checkpointing

For large models, use gradient checkpointing to trade compute for memory:

```python
# Recompute activations during backward pass
with mx.checkpoint():
    loss = self.loss_fn(batch)
```

---

## Configuration

Training configuration via `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    # Batching
    micro_batch_tokens: int = 4096
    min_samples_per_step: int = 16

    # Importance sampling
    importance_sampling: str = "token"  # or "sequence"
    clip_range: float = 0.2
    clip_skip_threshold: float = 0.3

    # Staleness
    staleness_limit: int = 3

    # Regularization
    kl_coef: float = 0.1

    # LoRA
    lora_rank: int = 8
    lora_layers: int = 16
```

---

## Logging and Monitoring

Training metrics are logged to Weights & Biases:

```python
wandb.log({
    "loss": loss,
    "clip_fraction": clip_fraction,
    "kl_divergence": kl.mean(),
    "advantage_mean": advantages.mean(),
    "advantage_std": advantages.std(),
    "policy_version": version,
    "samples_per_step": num_samples,
})
```

The async training loop ensures correct W&B step ordering even with concurrent generation.

---

## Putting It Together

A complete training run:

```python
# 1. Initialize components
model = load_model("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
apply_lora(model, rank=8, layers=16)

trainer = Trainer(
    model=model,
    config=TrainingConfig(
        micro_batch_tokens=4096,
        staleness_limit=3,
        kl_coef=0.1,
    ),
)

client = OpenAIClient(base_url="http://localhost:8000/v1")
arena = MyArena(client, credit_assigner=GRPOCredit())

# 2. Run training
await training_loop(
    arena=arena,
    trainer=trainer,
    num_steps=1000,
    episode_concurrency=8,
)
```

The arena generates rollouts, the trainer updates weights, and weights flow to the inference server in real-time.
