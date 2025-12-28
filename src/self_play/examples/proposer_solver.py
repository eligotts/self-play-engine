"""
Proposer/Solver Example: Composable episodes with sub-episode spawning.

This example demonstrates:
- ProposerEpisode spawns multiple SolverEpisodes
- Each SolverEpisode produces its own training records
- Proposer's reward derives from solver performance

Structure:
1. ProposerEpisode.rollout():
   - One model call to generate a question
   - Spawns N SolverEpisodes on that question
   - Collects results for scoring
2. SolverEpisode: Simple single-turn solve attempt
3. Rewards flow up: solver results → proposer reward
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..core import (
    Role,
    Messages,
    Rollout,
    Episode,
    EpisodeState,
    SingleTurnEpisode,
    Rubric,
    Arena,
    InferenceClient,
    Artifact,
    Step,
    EpisodeRequest,
)


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

def make_solver_rubric(solver_role: str) -> Rubric:
    """
    Rubric for solver: exact match against ground truth.
    """
    def exact_match(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        # Extract answer from last step
        if not rollout.steps:
            return {solver_role: 0.0}

        completion = rollout.steps[-1].completion_text
        if "The answer is:" in completion:
            answer = completion.split("The answer is:")[-1].strip().rstrip(".")
        else:
            answer = completion.strip().split("\n")[-1]

        # Compare to ground truth
        ground_truth = str(rollout.seed.get("ground_truth", "")).strip().lower()
        predicted = answer.strip().lower()

        reward = 1.0 if predicted == ground_truth else 0.0
        return {solver_role: reward}

    return Rubric(funcs=[exact_match])


def make_proposer_rubric(proposer_role: str, target_pass_rate: float = 0.5) -> Rubric:
    """
    Rubric for proposer: reward based on solver pass rate.

    Target ~50% pass rate (not too easy, not too hard).
    """
    def pass_rate_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        proposed = rollout.extras.get("proposed_question")

        # Invalid question = negative reward
        if not proposed or not proposed.get("question") or not proposed.get("answer"):
            return {proposer_role: -1.0}

        # Get pass rate from extras (computed in get_extras)
        pass_rate = rollout.extras.get("pass_rate", 0.0)

        # Reward peaks at target pass rate
        distance = abs(pass_rate - target_pass_rate)
        reward = 1.0 - (distance * 2)  # Max 1.0 at target, 0.0 at extremes

        return {proposer_role: max(-0.5, reward)}

    return Rubric(funcs=[pass_rate_reward])


# ---------------------------------------------------------------------------
# Solve Episode
# ---------------------------------------------------------------------------

class SolveEpisode(SingleTurnEpisode):
    """Single-turn episode where the Solver answers a question."""

    def __init__(self, solver_role_id: str = "Solver"):
        self.solver_role_id = solver_role_id
        self._rubric = make_solver_rubric(solver_role_id)

    @property
    def episode_type(self) -> str:
        return "solve"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def get_initial_actor(self, artifact: Any) -> str:
        return self.solver_role_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> Messages:
        role = arena.roles[self.solver_role_id]
        question = artifact.get("question", "What is 2 + 2?")

        messages: Messages = []
        if role.system_prompt:
            messages.append({"role": "system", "content": role.system_prompt})

        messages.append({
            "role": "user",
            "content": f"""Solve the following question:

{question}

Think step by step, then provide your final answer.
End your response with: "The answer is: <your answer>\""""
        })

        return messages


# ---------------------------------------------------------------------------
# Proposer Episode
# ---------------------------------------------------------------------------

class ProposerEpisode(Episode):
    """
    Composable episode that spawns solver sub-episodes.

    1. rollout() generates a question (one model call)
    2. rollout() spawns N solver episodes
    3. solver results stored in state.child_results
    4. rubric uses extras to compute proposer reward
    """

    def __init__(
        self,
        proposer_role_id: str = "Proposer",
        n_solver_rollouts: int = 4,
        target_pass_rate: float = 0.5,
    ):
        self.proposer_role_id = proposer_role_id
        self.n_solver_rollouts = n_solver_rollouts
        self.target_pass_rate = target_pass_rate
        self._rubric = make_proposer_rubric(proposer_role_id, target_pass_rate)

    @property
    def episode_type(self) -> str:
        return "propose"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    async def rollout(
        self,
        arena: Arena,
        artifact: Any,
        state: Optional[EpisodeState] = None,
    ) -> EpisodeState:
        if state is None:
            state = EpisodeState()

        # Generate question
        prompt = self._build_prompt(arena, artifact)
        response = await self.call_model(self.proposer_role_id, prompt, arena)

        step = Step(
            role_id=self.proposer_role_id,
            prompt=prompt,
            completion=response.completion,
            prompt_token_ids=response.prompt_token_ids,
            completion_token_ids=response.completion_token_ids,
            completion_logprobs=response.completion_logprobs,
        )
        state.trajectory.append(step)

        # Parse proposed question
        proposed = self._parse_question(response.text)
        state.data["proposed_question"] = proposed

        # Spawn solver episodes if valid question
        if proposed and proposed.get("question") and proposed.get("answer"):
            solver_seed = {
                "question": proposed["question"],
                "ground_truth": proposed["answer"],
            }

            requests = [
                EpisodeRequest(episode_type="solve", seed=solver_seed)
                for _ in range(self.n_solver_rollouts)
            ]
            results = await arena.generate_rollouts(requests)
            state.child_results.extend(results)

        state.done = True
        return state

    def _build_prompt(self, arena: Arena, artifact: Any) -> Messages:
        role = arena.roles[self.proposer_role_id]

        few_shot_text = ""
        if "questions" in arena.stores:
            examples = arena.stores["questions"].sample(k=3)
            if examples:
                few_shot_text = "Here are some example questions:\n\n"
                for i, ex in enumerate(examples, 1):
                    few_shot_text += f"Example {i}:\n{json.dumps(ex.data, indent=2)}\n\n"

        user_content = f"""{few_shot_text}Generate a new math question. The question should:
1. Be challenging but solvable
2. Have a clear, unambiguous answer
3. Be different from the examples

Respond with a JSON object containing:
{{"question": "<the question text>", "answer": "<the correct answer>", "difficulty": "easy|medium|hard"}}"""

        messages: Messages = []
        if role.system_prompt:
            messages.append({"role": "system", "content": role.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_question(self, text: str) -> Optional[Dict[str, str]]:
        try:
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Include child results summary for rubric."""
        child_rewards = [
            child.rewards.get("Solver", 0.0)
            for child in state.child_results
        ]
        return {
            "proposed_question": state.data.get("proposed_question"),
            "proposed_raw": state.last_completion_text,
            "solver_rewards": child_rewards,
            "pass_rate": sum(1 for r in child_rewards if r > 0.5) / len(child_rewards) if child_rewards else 0.0,
        }


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class ProposerSolverArena(Arena):
    """Arena that schedules proposer episodes (solvers are nested inside)."""

    def __init__(self, client: InferenceClient, batch_size: int = 4):
        super().__init__(client)
        self.batch_size = batch_size

    def get_batch(self) -> List[EpisodeRequest]:
        return [
            EpisodeRequest(episode_type="propose", seed={})
            for _ in range(self.batch_size)
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_proposer_solver_arena(
    client: InferenceClient,
    initial_questions: Optional[List[Dict[str, str]]] = None,
    n_solver_rollouts: int = 4,
    batch_size: int = 4,
) -> ProposerSolverArena:
    """Create a complete proposer/solver arena."""
    arena = ProposerSolverArena(client, batch_size=batch_size)

    arena.add_role(Role(
        id="Proposer",
        system_prompt="You are a creative math problem creator. "
                      "Generate interesting, well-formed problems with clear answers.",
        temperature=0.9,
        max_tokens=300,
    ))

    arena.add_role(Role(
        id="Solver",
        system_prompt="You are a skilled math problem solver. "
                      "Think step by step and provide clear, correct answers.",
        temperature=0.7,
        max_tokens=500,
    ))

    question_store = arena.add_store("questions")
    if initial_questions is None:
        initial_questions = [
            {"question": "What is 15 + 27?", "answer": "42", "difficulty": "easy"},
            {"question": "If x + 5 = 12, what is x?", "answer": "7", "difficulty": "easy"},
            {"question": "What is 8 * 5?", "answer": "40", "difficulty": "medium"},
        ]
    for i, q in enumerate(initial_questions):
        question_store.add(Artifact(id=f"seed_{i}", data=q, weight=1.0))

    arena.add_episode("propose", ProposerEpisode(n_solver_rollouts=n_solver_rollouts))
    arena.add_episode("solve", SolveEpisode())

    return arena
