"""
Proposer/Solver Example: Composable episodes with sub-episode spawning.

This example demonstrates:
- ProposerEpisode spawns multiple SolverEpisodes
- Each SolverEpisode produces its own training records
- Proposer's reward derives from solver performance
- Solver completions that are used to get a proposer reward are NOT trained on

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
    Messages,
    Rollout,
    Episode,
    EpisodeState,
    GenerateResult,
    SingleTurnEpisode,
    Rubric,
    Arena,
    InferenceClient,
    Artifact,
    Step,
    EpisodeRequest,
    GRPOCredit,
)


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

def solver_exact_match(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Rubric for solver: exact match against ground truth.
    Discovers actor dynamically from rollout.
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    completion = rollout.steps[-1].completion_text

    if "The answer is:" in completion:
        answer = completion.split("The answer is:")[-1].strip().rstrip(".")
    else:
        answer = completion.strip().split("\n")[-1]

    # Compare to ground truth
    ground_truth = str(rollout.artifact.get("ground_truth", "")).strip().lower()
    predicted = answer.strip().lower()

    reward = 1.0 if predicted == ground_truth else 0.0

    if arena.verbose:
        if reward == 0.0:
            print(f"    [solver_rubric] MISMATCH: predicted='{predicted[:40]}' vs ground_truth='{ground_truth}' → reward=0.0")
        else:
            print(f"    [solver_rubric] MATCH: predicted='{predicted[:40]}' vs ground_truth='{ground_truth}' → reward=1.0")

    return {actor: reward}


def proposer_pass_rate_reward(
    rollout: Rollout,
    arena: Arena,
    target_pass_rate: float = 0.5,
) -> Dict[str, float]:
    """
    Rubric for proposer: reward based on solver pass rate.
    Discovers actor dynamically from rollout.

    Target ~50% pass rate (not too easy, not too hard).
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    proposed = rollout.extras.get("proposed_question")

    # Invalid question = negative reward
    if not proposed or not proposed.get("question") or not proposed.get("ground_truth"):
        if arena.verbose:
            print(f"    [proposer_rubric] invalid question → reward=-1.0")
        return {actor: -1.0}

    # Get pass rate from extras (computed in get_extras)
    pass_rate = rollout.extras.get("pass_rate", 0.0)

    # Reward peaks at target pass rate
    distance = abs(pass_rate - target_pass_rate)
    reward = 1.0 - (distance * 2)  # Max 1.0 at target, 0.0 at extremes
    final_reward = max(-0.5, reward)

    if arena.verbose:
        print(f"    [proposer_rubric] pass_rate={pass_rate:.2f}, target={target_pass_rate}, reward={final_reward:.2f}")

    return {actor: final_reward}


# ---------------------------------------------------------------------------
# Solve Episode
# ---------------------------------------------------------------------------

class SolveEpisode(SingleTurnEpisode):
    """Single-turn episode where the Solver answers a question."""

    def __init__(self, solver_actor_id: str = "Solver"):
        self.solver_actor_id = solver_actor_id
        self._rubric = Rubric(funcs=[solver_exact_match])

    @property
    def episode_type(self) -> str:
        return "solve"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        return self.solver_actor_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        question = artifact.get("question", "What is 2 + 2?")

        return f"""Solve the following question:

{question}

Think step by step, then provide your final answer.
You MUST end your response with: "The answer is: " followed by the answer."""

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Include response and extracted answer for preview/debugging."""
        response = state.last_completion_text or ""

        # Extract answer using same logic as rubric
        if "The answer is:" in response:
            extracted = response.split("The answer is:")[-1].strip().rstrip(".")
        else:
            extracted = response.strip().split("\n")[-1] if response else ""

        return {
            "response": response,
            "extracted_answer": extracted,
        }


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
        proposer_actor_id: str = "Proposer",
        n_solver_rollouts: int = 4,
        target_pass_rate: float = 0.5,
    ):
        self.proposer_actor_id = proposer_actor_id
        self.n_solver_rollouts = n_solver_rollouts
        self.target_pass_rate = target_pass_rate
        self._rubric = Rubric(funcs=[proposer_pass_rate_reward])

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
        response = await self.call_model(self.proposer_actor_id, prompt, arena)

        step = Step(
            actor_id=self.proposer_actor_id,
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
        if proposed and proposed.get("question") and proposed.get("ground_truth"):
            solver_artifact = {
                "question": proposed["question"],
                "ground_truth": proposed["ground_truth"],
            }

            requests = [
                EpisodeRequest(episode_type="solve", artifact=solver_artifact, is_trainable=False)
                for _ in range(self.n_solver_rollouts)
            ]
            results = await arena.generate_rollouts(requests)
            state.child_results.extend(results)

        state.done = True
        return state

    async def generate(
        self,
        arena: Arena,
        artifact: Any,
        meta: Optional[Dict[str, Any]] = None,
        is_trainable: bool = True,
    ) -> GenerateResult:
        result = await super().generate(arena, artifact, meta=meta, is_trainable=is_trainable)

        proposed = result.rollout.extras.get("proposed_question")
        if proposed and proposed.get("question") and proposed.get("ground_truth"):
            if "questions" in arena.stores:
                arena.stores["questions"].add(Artifact(
                    id=f"gen_{result.rollout.id}",
                    data=proposed
                ))

        return result

    def _build_prompt(self, arena: Arena, artifact: Any) -> Messages:
        actor = arena.actors[self.proposer_actor_id]

        few_shot_text = ""
        examples = artifact.get("examples", [])
        if examples:
            few_shot_text = "Here are some example questions:\n\n"
            for i, ex in enumerate(examples, 1):
                few_shot_text += f"Example {i}:\n{json.dumps(ex, indent=2)}\n\n"

        user_content = f"""{few_shot_text}Generate a new math question. The question should:
        1. Be challenging but solvable
        2. Have a clear, unambiguous answer
        3. Be different from the examples

        Respond with a JSON object containing:
        {{"question": "<the question text>", "ground_truth": "<the correct answer>"}}

        CRITICAL INSTRUCTIONS:
        - Do NOT write any text before or after the JSON
        - Do NOT use markdown code blocks (backticks)
        - Output ONLY a single JSON object
        - Start your response with {{ and end with }}
        - Respond with ONLY JSON:"""

        messages: Messages = []
        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})
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
        # Get reward for each child using the child's actor
        child_rewards = []
        for child in state.child_results:
            actor = next(iter(child.rollout.actors)) if child.rollout.actors else None
            reward = child.rewards.get(actor, 0.0) if actor else 0.0
            child_rewards.append(reward)

        pass_rate = sum(1 for r in child_rewards if r > 0.5) / len(child_rewards) if child_rewards else 0.0

        return {
            "proposed_question": state.data.get("proposed_question"),
            "proposed_raw": state.last_completion_text,
            "solver_rewards": child_rewards,
            "pass_rate": pass_rate,
        }


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class ProposerSolverArena(Arena):
    """Arena that schedules proposer episodes (solvers are nested inside)."""

    def __init__(
        self,
        client: InferenceClient,
        episodes_per_step: int = 4,
        min_questions: int = 10,
        verbose: bool = False,
    ):
        super().__init__(client, credit_assigner=GRPOCredit(), verbose=verbose)
        self.episodes_per_step = episodes_per_step
        self.min_questions = min_questions

    async def on_train_start(self) -> None:
        """Ensure we have at least min_questions in the store before training."""
        if "questions" not in self.stores:
            return

        store = self.stores["questions"]
        current_count = store.count()

        if current_count >= self.min_questions:
            if self.verbose:
                print(f"[on_train_start] Already have {current_count} questions, skipping warmup")
            return

        needed = self.min_questions - current_count
        if self.verbose:
            print(f"[on_train_start] Need {needed} more questions (have {current_count}, want {self.min_questions})")

        # Generate questions by running proposer episodes (non-trainable)
        while store.count() < self.min_questions:
            # Sample examples for the proposer
            samples = store.sample(k=min(3, store.count()))
            examples = [s.data for s in samples]

            # Run a single proposer episode (non-trainable since we're just warming up)
            request = EpisodeRequest(
                episode_type="propose",
                artifact={"examples": examples},
                is_trainable=False,
            )
            await self.generate_rollouts([request])

            # The ProposerEpisode.generate() already adds valid questions to the store
            if self.verbose:
                print(f"[on_train_start] Questions in store: {store.count()}")

        if self.verbose:
            print(f"[on_train_start] Warmup complete, {store.count()} questions ready")

    def get_batch(self) -> List[EpisodeRequest]:
        propose_requests: List[EpisodeRequest] = []
        solve_requests: List[EpisodeRequest] = []
        if "questions" in self.stores:
            for _ in range(self.episodes_per_step):
                question = self.stores["questions"].sample_one()
                if question is None:
                    break
                solve_requests.append(EpisodeRequest(
                    episode_type="solve",
                    artifact=question.data,
                ))
                samples = self.stores["questions"].sample(k=3)
                examples = [s.data for s in samples]
                propose_requests.append(EpisodeRequest(
                    episode_type="propose",
                    artifact={"examples": examples},
                ))
        return propose_requests + solve_requests
