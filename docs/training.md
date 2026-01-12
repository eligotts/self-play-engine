# Training Infrastructure

This document covers the RL training infrastructure built on MLX for Apple Silicon. For self-play abstractions, see [concepts.md](concepts.md).

---

## The Core Idea

Complete separation between how data is generated and how it's trained on. The trainer doesn't care where the data comes from - it just pulls samples and trains. The arena doesn't care how training works - it just generates rollouts.

---

## Architecture Overview

- **Arena** generates rollouts, scores them, and pushes `TrainingBatch` objects to a queue
- **Trainer** pulls batches, computes loss, updates weights
- **Inference Server** serves the model with LoRA adapters that update in real-time

Generation and training run concurrently - the arena doesn't wait for the trainer, and the trainer doesn't wait for the arena.

---

## Consumer-Producer Pattern

### The Async Loop

```python
async def training_loop(arena, trainer, num_steps, batch_queue):
    async def generator():
        """Keep generating batches and pushing to queue."""
        for _ in range(num_steps):
            batch = await arena.step(concurrency=episode_concurrency)
            await batch_queue.put(batch)

    async def consumer():
        """Pull batches from queue and train."""
        for _ in range(num_steps):
            batch = await batch_queue.get()
            # Filter stale records, form micro-batches, train
            trainer.accumulate(batch.records)

    await asyncio.gather(generator(), consumer())
```

The queue provides natural backpressure - if training falls behind, the queue fills up and generation waits.

### Synchronous Loop

For when you dont want to both generate and train at the same time:

```python
async def synchronous_training_loop(arena, trainer, num_steps):
    for step in range(num_steps):
        batch = await arena.step()
        trainer.train_step(batch)
```

Simpler, easier to debug, but lower throughput.

---

## Micro-Batch Streaming

We use **token-budget-based micro-batching** to control memory usage:

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

### Gradient Accumulation

Gradients accumulate across micro-batches until `min_samples_per_step` is reached:

```python
class Trainer:
    def accumulate(self, records):
        while records:
            micro_batch = form_micro_batch(records, self.micro_batch_tokens)
            records = records[len(micro_batch):]

            loss, grads = value_and_grad(self.loss_fn)(micro_batch)
            self.accumulated_grads += grads * weight

        if self.accumulated_samples >= self.min_samples_per_step:
            self.optimizer.step(self.accumulated_grads)
            # Publish new weights to inference server
            await self.publish_weights()
```

---

## Loss Functions

The loss function is where all the RL magic happens. Here's what's implemented:

### Token vs Sample Level Normalization

Two modes for the policy gradient loss:

**Token-level (DAPO)** - Default:
```python
loss = masked_loss.sum() / mask.sum()
```
Equal weight per token. Longer completions contribute more to the gradient.

**Sample-level (GRPO)**:
```python
loss = (masked_loss.sum(dim=-1) / mask.sum(dim=-1)).mean()
```
Equal weight per sample. Length-invariant.

### Importance Sampling

Off-policy training requires importance sampling to correct for distribution shift.

**Token-Level (PPO-Style)** - Default:
```python
log_ratio = trainer_logprobs - inference_logprobs  # Per-token
importance_ratio = exp(log_ratio)

# PPO clipping
clipped_ratio = clip(importance_ratio, 1 - epsilon, 1 + epsilon)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```
**Sequence-Level (GSPO)**:
```python
# Average log ratio across completion tokens
seq_log_ratio = mean(log_ratio[mask])
si = exp(seq_log_ratio)

# Much tighter clipping
si_clipped = clip(si, 1 - epsilon, 1 + epsilon)  # epsilon ≈ 3e-4
```

### KL Penalty (K1 Advantage Shaping)

Based on [Shah et al., 2025](https://arxiv.org/abs/2512.21852). Regularize against a reference policy to prevent collapse. In the paper they subtract from rewards, but here we don't have access to raw rewards, so we subtract from advantages:

```python
# K1 = log π_new - log π_ref (per-token KL contribution)
k1_per_token = trainer_logprobs - reference_logprobs
k1_seq = mean(k1_per_token[mask])  # Sequence-level

# Clip for stability
k1_clipped = clip(k1_seq, -kl_max, kl_max)

# Subtract from advantage (shaping, not penalty)
shaped_advantage = advantage - kl_coef * k1_clipped
```

The reference policy is the base model (without LoRA). We compute it efficiently by temporarily disabling LoRA:

```python
def get_ref_logprobs(model, input_ids):
    # Temporarily disable LoRA
    original_scales = save_lora_scales(model)
    set_lora_scales(model, 0.0)

    ref_logprobs = forward(model, input_ids)

    # Restore LoRA
    restore_lora_scales(model, original_scales)
    return ref_logprobs
```

No need for a separate reference model in memory.

### Zero Gradient Filtering (Clip Skip)

If too many tokens are clipped, the batch is likely off-policy garbage:

```python
clip_fraction = (importance_ratio < 0.8 | importance_ratio > 1.2).mean()
if clip_fraction > clip_skip_threshold:  # Default: 0.3
    return zero_loss, {"skip_batch": True}
```

This prevents training on severely divergent samples. The trainer tracks skipped batches in metrics.

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

---

## LoRA Hot-Swap

The inference server supports hot-swapping LoRA weights without restart.

### Weight Publishing

After each training step, the trainer publishes weights:

```python
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

```bash
uv run legos serve \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --port 8000
```

---

## Configuration

Training configuration via `TrainerConfig`:

```python
@dataclass
class TrainerConfig:
    # Optimizer
    lr: float = 1e-5

    # Batching
    micro_batch_tokens: int = 4096
    min_samples_per_step: int = 16

    # Importance sampling
    ppo_clip_min: float = 0.8
    ppo_clip_max: float = 1.2
    clip_skip_threshold: float = 0.3
    importance_sampling: str = "token"  # or "sequence"
    gspo_clip_epsilon: float = 3e-4

    # Loss normalization
    loss_type: str = "token"  # or "sample"

    # KL regularization
    kl_coef: float = 0.1
    use_kl_penalty: bool = False
    kl_max: float = 10.0

    # Off-policy
    staleness_limit: int = 3

    # Infrastructure
    inference_url: str = "http://localhost:8000"
    pad_token_id: int = 0

    # Evaluation
    eval_every: int = 10
    eval_concurrency: int = 8

    # Checkpointing
    checkpoint_every: int = 100
    checkpoint_dir: str = "checkpoints"

    # Logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
```

---

## Putting It Together

A complete training run:

```python
# 1. Initialize components
model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
apply_lora(model, inference_mode=False)

optimizer = optim.Adam(learning_rate=1e-5)
config = TrainerConfig(
    micro_batch_tokens=4096,
    staleness_limit=3,
    kl_coef=0.1,
    use_kl_penalty=False,
)

client = OpenAIClient(base_url="http://localhost:8000/v1")
trainer = Trainer(model, optimizer, config, client)

arena = MyArena(client, credit_assigner=GRPOCredit())
# ... add actors, episodes, stores ...

# 2. Run training
await training_loop(
    arena=arena,
    trainer=trainer,
    num_steps=1000,
    episode_concurrency=8,
)
```

The arena generates rollouts, the trainer updates weights, and weights flow to the inference server in real-time.
