"""
Dataset RL Example: Standard RL with fixed datasets.

This example demonstrates using the self-play abstractions for traditional RL:
- Fixed dataset with question/answer pairs
- Single-turn episodes (one question -> one completion)
- GRPO-style sampling (one question repeated N times per batch)
- Pluggable reward functions (exact match, LLM judge, etc.)

Structure:
1. DatasetEpisode: SingleTurnEpisode that prompts with question from artifact
2. DatasetArena: Samples ONE question, repeats batch_size times for GRPO
3. Reward functions: exact_match, llm_judge, make_llm_judge factory
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

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
# Reward Functions
# ---------------------------------------------------------------------------

def exact_match(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Exact match reward: 1.0 if completion matches answer, 0.0 otherwise.

    Case-insensitive, strips whitespace.
    """
    if not rollout.steps:
        return {}

    completion = rollout.steps[-1].completion_text.strip().lower()
    if "the answer is:" in completion:
        completion = completion.split("the answer is:")[-1].strip()
    ground_truth = str(rollout.artifact.get("answer", "")).strip().lower()

    reward = 1.0 if completion == ground_truth else 0.0

    if arena.verbose:
        match_str = "MATCH" if reward == 1.0 else "MISMATCH"
        print(f"    [exact_match] {match_str}: '{completion[:50]}' vs '{ground_truth}'")

    return {rollout.steps[-1].role_id: reward}


DEFAULT_JUDGE_PROMPT = """You are an impartial judge evaluating a model's response to a question.

Score the response on correctness from 0.0 to 1.0:
- 1.0: Completely correct, matches expected answer
- 0.5-0.9: Partially correct or correct but with extra content
- 0.0-0.4: Incorrect or irrelevant

Output ONLY a JSON object with your score:
{"score": 0.8, "reasoning": "Brief explanation"}

Start your response with { and end with }."""


async def llm_judge(
    rollout: Rollout,
    arena: Arena,
    *,
    judge_prompt: str = DEFAULT_JUDGE_PROMPT,
    answer_key: str = "answer",
    question_key: str = "question",
) -> Dict[str, float]:
    """
    LLM-as-judge reward function.

    Calls arena.call_model() WITHOUT role_id (not trainable).
    Parses JSON response for score.
    """
    if not rollout.steps:
        return {}

    completion = rollout.steps[-1].completion_text
    question = rollout.artifact.get(question_key, "")
    answer = rollout.artifact.get(answer_key, "")

    response = await arena.call_model(
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": f"Question: {question}\n\nExpected answer: {answer}\n\nModel response: {completion}"},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    # Parse score from JSON
    score = _parse_judge_response(response.text)

    if arena.verbose:
        print(f"    [llm_judge] score={score:.2f} for completion: '{completion[:50]}...'")

    return {rollout.steps[-1].role_id: score}


def _parse_judge_response(text: str) -> float:
    """Parse score from judge response, defaulting to 0.0 on failure."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            return float(data.get("score", 0.0))
    except (json.JSONDecodeError, ValueError):
        pass

    return 0.0


def make_llm_judge(
    prompt: str = DEFAULT_JUDGE_PROMPT,
    answer_key: str = "answer",
    question_key: str = "question",
) -> RewardFn:
    """Factory for custom LLM judge prompts."""
    async def judge(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        return await llm_judge(
            rollout, arena,
            judge_prompt=prompt,
            answer_key=answer_key,
            question_key=question_key,
        )
    return judge


# ---------------------------------------------------------------------------
# Dataset Episode
# ---------------------------------------------------------------------------

class DatasetEpisode(SingleTurnEpisode):
    """
    Single-turn episode for fixed dataset RL.

    Takes a question from the artifact, sends to model, scores against answer.
    """

    def __init__(
        self,
        role_id: str = "model",
        question_key: str = "question",
        answer_key: str = "answer",
        reward_fn: RewardFn | None = None,
        prompt_template: str = "{question}\n\nYou must end your response with \"The answer is: \" followed by the answer.",
    ):
        self.role_id = role_id
        self.question_key = question_key
        self.answer_key = answer_key
        self.prompt_template = prompt_template
        self._reward_fn = reward_fn or exact_match
        self._rubric = Rubric(funcs=[self._reward_fn])

    @property
    def episode_type(self) -> str:
        return "dataset_rl"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        return self.role_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        question = artifact.get(self.question_key, "")
        # Store ground truth for get_extras
        state.data["ground_truth"] = artifact.get(self.answer_key, "")
        return self.prompt_template.format(question=question)

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        return {
            "ground_truth": state.data.get("ground_truth", ""),
        }


# ---------------------------------------------------------------------------
# Dataset Arena
# ---------------------------------------------------------------------------

class DatasetArena(Arena):
    """
    Arena that samples from a fixed dataset.

    GRPO-style sampling: samples ONE question, repeats it batch_size times.
    This produces multiple rollouts per question for group-relative advantage.
    """

    def __init__(
        self,
        client: InferenceClient,
        batch_size: int = 8,
        store_name: str = "dataset",
        episode_type: str = "dataset_rl",
        weighted_sampling: bool = False,
        credit_assigner: CreditAssigner | None = None,
        verbose: bool = False,
    ):
        super().__init__(client, credit_assigner=credit_assigner or GRPOCredit(), verbose=verbose)
        self.batch_size = batch_size
        self.store_name = store_name
        self._episode_type = episode_type
        self.weighted_sampling = weighted_sampling

    def get_batch(self) -> List[EpisodeRequest]:
        """Sample ONE question, repeat batch_size times for GRPO."""
        if self.store_name not in self.stores:
            return []

        sample = self.stores[self.store_name].sample_one(weighted=self.weighted_sampling)
        if sample is None:
            return []

        # Repeat same question batch_size times
        return [
            EpisodeRequest(
                episode_type=self._episode_type,
                artifact=sample.data,
                meta={"artifact_id": sample.id},
            )
            for _ in range(self.batch_size)
        ]


# ---------------------------------------------------------------------------
# Dataset Loading Utilities
# ---------------------------------------------------------------------------

def load_dataset(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load dataset from JSON or JSONL file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)
