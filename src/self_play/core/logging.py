"""
Verbose logging for self-play training runs.

Designed for easy debugging and orchestration tracking.
Writes structured, human-readable logs to a file.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .types import Messages, Rollout, Step
from .episode import GenerateResult


class VerboseLogger:
    """
    Structured logger for self-play training runs.

    Logs are written in a human-readable format with clear sections:
    - Episode starts/completions
    - Model calls (prompts and responses)
    - Rewards and credit assignment
    - Batch summaries

    Each entry has a timestamp and is easy to grep/parse.
    """

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self._file: Optional[TextIO] = None
        self._indent = 0

    def __enter__(self) -> "VerboseLogger":
        self._file = open(self.log_path, "w")
        self._write_header()
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _write(self, text: str):
        if self._file:
            indent = "  " * self._indent
            for line in text.split("\n"):
                self._file.write(f"{indent}{line}\n")
            self._file.flush()

    def _write_header(self):
        self._write("=" * 80)
        self._write(f"SELF-PLAY TRAINING LOG - {datetime.now().isoformat()}")
        self._write("=" * 80)
        self._write("")

    # -------------------------------------------------------------------------
    # High-level events
    # -------------------------------------------------------------------------

    def log_run_start(self, config: Dict[str, Any]):
        """Log the start of a training run."""
        self._write(f"[{self._ts()}] RUN START")
        self._write("-" * 40)
        for key, value in config.items():
            self._write(f"  {key}: {value}")
        self._write("")

    def log_step_start(self, step_num: int, num_requests: int):
        """Log the start of a training step."""
        self._write(f"[{self._ts()}] STEP {step_num} - {num_requests} episodes")
        self._write("-" * 40)
        self._indent += 1

    def log_step_end(self, step_num: int, batch_meta: Dict[str, Any]):
        """Log the end of a training step."""
        self._indent -= 1
        self._write(f"[{self._ts()}] STEP {step_num} COMPLETE")
        self._write(f"  records: {batch_meta.get('num_records', 0)}")
        self._write("")

    # -------------------------------------------------------------------------
    # Episode events
    # -------------------------------------------------------------------------

    def log_episode_start(self, episode_type: str, artifact: Any, rollout_id: str):
        """Log the start of an episode."""
        self._write(f"[{self._ts()}] EPISODE START: {episode_type}")
        self._write(f"  rollout_id: {rollout_id}")
        artifact_str = json.dumps(artifact, default=str)[:200]
        self._write(f"  artifact: {artifact_str}")
        self._indent += 1

    def log_episode_end(
        self,
        episode_type: str,
        rollout_id: str,
        rewards: Dict[str, float],
        num_steps: int,
    ):
        """Log the completion of an episode."""
        self._indent -= 1
        self._write(f"[{self._ts()}] EPISODE COMPLETE: {episode_type}")
        self._write(f"  rollout_id: {rollout_id}")
        self._write(f"  steps: {num_steps}")
        self._write(f"  rewards: {rewards}")
        self._write("")

    # -------------------------------------------------------------------------
    # Model call events
    # -------------------------------------------------------------------------

    def log_model_call(
        self,
        role_id: str,
        messages: Messages,
        response_text: str,
        has_tokens: bool,
    ):
        """Log a model call with prompt and response."""
        self._write(f"[{self._ts()}] MODEL CALL: {role_id}")

        # Log prompt (truncated if long)
        self._write("  PROMPT:")
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            self._write(f"    [{role}] {content[:100]}...")

        # Log response (truncated if long)
        response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
        self._write(f"  RESPONSE: {response_preview}")
        self._write(f"  has_tokens: {has_tokens}")

    # -------------------------------------------------------------------------
    # Result logging
    # -------------------------------------------------------------------------

    def log_result(self, result: GenerateResult, depth: int = 0):
        """Recursively log a GenerateResult and its children."""
        rollout = result.rollout
        prefix = "  " * depth

        self._write(f"{prefix}RESULT: {rollout.episode_type} [{rollout.id}]")
        self._write(f"{prefix}  steps: {len(rollout.steps)}")
        self._write(f"{prefix}  rewards: {result.rewards}")

        # Log step summaries
        for i, step in enumerate(rollout.steps):
            completion_preview = step.completion_text[:100] + "..." if len(step.completion_text) > 100 else step.completion_text
            self._write(f"{prefix}  step[{i}] {step.role_id}: {completion_preview}")

        # Log extras
        if rollout.extras:
            extras_str = json.dumps(rollout.extras, default=str)[:200]
            self._write(f"{prefix}  extras: {extras_str}")

        # Recurse into children
        for child in result.children:
            self.log_result(child, depth + 1)

    def log_batch_summary(self, results: List[GenerateResult]):
        """Log a summary of all results in a batch."""
        self._write(f"[{self._ts()}] BATCH SUMMARY")
        self._write("-" * 40)

        total_steps = 0
        total_reward = 0.0
        role_rewards: Dict[str, List[float]] = {}

        def collect_stats(result: GenerateResult):
            nonlocal total_steps, total_reward
            total_steps += len(result.rollout.steps)
            for role_id, reward in result.rewards.items():
                total_reward += reward
                if role_id not in role_rewards:
                    role_rewards[role_id] = []
                role_rewards[role_id].append(reward)
            for child in result.children:
                collect_stats(child)

        for result in results:
            collect_stats(result)
            self.log_result(result)

        self._write("")
        self._write(f"  total_results: {len(results)}")
        self._write(f"  total_steps: {total_steps}")
        for role_id, rewards in role_rewards.items():
            avg = sum(rewards) / len(rewards) if rewards else 0
            self._write(f"  {role_id}: avg={avg:.4f}, n={len(rewards)}")
        self._write("")

    # -------------------------------------------------------------------------
    # Training record logging
    # -------------------------------------------------------------------------

    def log_training_records(self, records: List[Any]):
        """Log training records summary."""
        self._write(f"[{self._ts()}] TRAINING RECORDS: {len(records)}")

        role_counts: Dict[str, int] = {}
        role_rewards: Dict[str, List[float]] = {}

        for record in records:
            role_id = record.role_id
            role_counts[role_id] = role_counts.get(role_id, 0) + 1
            if role_id not in role_rewards:
                role_rewards[role_id] = []
            role_rewards[role_id].append(record.reward)

        for role_id in role_counts:
            rewards = role_rewards[role_id]
            avg = sum(rewards) / len(rewards) if rewards else 0
            self._write(f"  {role_id}: {role_counts[role_id]} records, avg_reward={avg:.4f}")
        self._write("")

    # -------------------------------------------------------------------------
    # Custom messages
    # -------------------------------------------------------------------------

    def log(self, message: str):
        """Log a custom message."""
        self._write(f"[{self._ts()}] {message}")

    def log_error(self, message: str, error: Exception):
        """Log an error."""
        self._write(f"[{self._ts()}] ERROR: {message}")
        self._write(f"  {type(error).__name__}: {error}")
        self._write("")
