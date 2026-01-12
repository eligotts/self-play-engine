"""
GSM8K Task: Math reasoning with GRPO-style training.

This task implements training on the GSM8K math dataset using:
- GRPO-style sampling (one question repeated N times per batch)
- Format + correctness + brevity rewards
- XML answer tags for structured output
}
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..core import (
    Rollout,
    EpisodeState,
    SingleTurnEpisode,
    Rubric,
    RewardFn,
    Arena,
    InferenceClient,
    EpisodeRequest,
    CreditAssigner,
    GRPOCredit,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_hash_answer(text: str) -> str | None:
    """Extract answer from GSM8K format (after ####)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML <answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def load_gsm8k(split: str = "train", max_samples: int | None = None) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from Hugging Face.

    Args:
        split: Dataset split ("train" or "test")
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of dicts with "question" and "answer" keys
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading GSM8K {split} split from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        question = item["question"]
        full_answer = item["answer"]
        answer = extract_hash_answer(full_answer)

        samples.append({
            "question": question,
            "answer": answer,
            "full_answer": full_answer,
        })

    print(f"Loaded {len(samples)} samples from GSM8K")
    return samples


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

def gsm8k_reward(
    rollout: Rollout,
    arena: Arena,
) -> Dict[str, float]:
    """
    Reward function for GSM8K combining:
    - Format: 0.25 if proper <answer>INT</answer> tags with integer inside
    - Correctness: 1.0 if the extracted answer matches ground truth
    - Brevity: Up to 0.5 bonus for responses under 700 characters
      (only awarded if format is correct AND answer is correct)

    Total possible reward: 1.75
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    completion = rollout.steps[-1].completion_text.strip()
    ground_truth = str(rollout.artifact.get("answer", "")).strip()

    total_reward = 0.0

    # Extract answer from <answer> tags
    extracted = extract_xml_answer(completion)

    # Normalize for comparison (remove commas, dollar signs, whitespace)
    extracted_norm = extracted.replace(",", "").replace("$", "").strip()
    truth_norm = ground_truth.replace(",", "").replace("$", "").strip()

    # 1. Format reward: 0.25 if has <answer>...</answer> with integer inside
    has_answer_tags = "<answer>" in completion and "</answer>" in completion
    is_integer = extracted_norm.lstrip("-").isdigit()  # handles negative integers
    format_correct = has_answer_tags and is_integer
    format_reward = 0.25 if format_correct else 0.0
    total_reward += format_reward

    # 2. Correctness reward (1.0 if correct)
    is_correct = extracted_norm == truth_norm
    correctness_reward = 1.0 if is_correct else 0.0
    total_reward += correctness_reward

    # 3. Brevity reward: only if format correct AND answer correct
    char_count = len(completion)
    brevity_threshold = 700
    max_brevity_reward = 0.5
    if format_correct and is_correct and char_count < brevity_threshold:
        brevity_reward = max_brevity_reward * (brevity_threshold - char_count) / brevity_threshold
    else:
        brevity_reward = 0.0
    total_reward += brevity_reward

    if arena.verbose:
        print("-" * 20)
        print(f"Chars: {char_count} | Rewards: format={format_reward:.2f} correct={correctness_reward:.1f} brevity={brevity_reward:.3f} total={total_reward:.3f}")
        print(f"End of completion: {completion[-100:]}")

    return {actor: total_reward}


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

class GSM8KEpisode(SingleTurnEpisode):
    """
    Single-turn episode for GSM8K math reasoning.

    Formats the question and sends to model. The reward function
    scores the completion based on format and correctness.
    """

    def __init__(
        self,
        actor_id: str = "model",
        reward_fn: RewardFn | None = None,
        prompt_template: str = "{question}",
        episode_type: str = "gsm8k",
    ):
        self.actor_id = actor_id
        self.prompt_template = prompt_template
        self._episode_type = episode_type
        self._reward_fn = reward_fn or gsm8k_reward
        self._rubric = Rubric(funcs=[self._reward_fn])

    @property
    def episode_type(self) -> str:
        return self._episode_type

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        return self.actor_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        if isinstance(artifact, dict):
            return self.prompt_template.format(**artifact)
        return self.prompt_template.format(question=artifact)


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class GSM8KArena(Arena):
    """
    Arena for GSM8K training with GRPO-style sampling.

    Samples ONE question, repeats it episodes_per_step times.
    This produces multiple rollouts per question for group-relative advantage.
    """

    def __init__(
        self,
        client: InferenceClient,
        episodes_per_step: int = 8,
        store_name: str = "gsm8k",
        episode_type: str = "gsm8k",
        credit_assigner: CreditAssigner | None = None,
        verbose: bool = False,
    ):
        super().__init__(client, credit_assigner=credit_assigner or GRPOCredit(), verbose=verbose)
        self.episodes_per_step = episodes_per_step
        self.store_name = store_name
        self._episode_type = episode_type

    def get_batch(self) -> List[EpisodeRequest]:
        """Sample ONE question, repeat episodes_per_step times for GRPO."""
        if self.store_name not in self.stores:
            return []

        sample = self.stores[self.store_name].sample_one()
        if sample is None:
            return []

        return [
            EpisodeRequest(
                episode_type=self._episode_type,
                artifact=sample.data,
                meta={"artifact_id": sample.id},
            )
            for _ in range(self.episodes_per_step)
        ]
