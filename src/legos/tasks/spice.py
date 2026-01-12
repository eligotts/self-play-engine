"""
SPICE: Self-Play In Corpus Environment.

Inspired by the SPICE paper (arXiv:2510.24684).

This arena implements a variant of proposer-solver where:
- Proposer reads documents from a corpus and generates questions
- Solver answers questions without access to the source document
- Solver episodes ARE trainable (key difference from proposer_solver.py)

Structure:
1. SpiceProposerEpisode.rollout():
   - Samples a document from corpus
   - Generates question + ground_truth from the document
   - Spawns N trainable SolverEpisodes
   - Reward based on solver pass rate (peaks at 50%)
2. SpiceSolverEpisode: Single-turn answer attempt
   - Reward via LLM-as-judge (OpenRouter)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# OpenRouter client for judge calls (following refine_loop.py pattern)
_openrouter_client: Optional[AsyncOpenAI] = None


def _get_openrouter_client() -> AsyncOpenAI:
    """Lazily initialize the OpenRouter client."""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30.0,  # 30 second timeout to prevent hanging
        )
    return _openrouter_client


from ..core import (
    Messages,
    Rollout,
    Episode,
    EpisodeState,
    SingleTurnEpisode,
    GenerateResult,
    Rubric,
    Arena,
    InferenceClient,
    Artifact,
    Step,
    EpisodeRequest,
    GRPOCredit,
    CreditAssigner,
)


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

async def solver_llm_judge_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    LLM-as-judge for solver correctness. Returns 0.0 or 1.0.

    Uses OpenRouter to determine if the model's answer is semantically
    equivalent to the ground truth.
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    question = rollout.artifact.get("question", "")
    ground_truth = rollout.artifact.get("ground_truth", "")
    model_answer = rollout.steps[-1].completion_text if rollout.steps else ""

    if not model_answer:
        if arena.verbose:
            print(f"    [solver_judge] empty answer -> reward=0.0")
        return {actor: 0.0}

    judge_messages = [
        {
            "role": "system",
            "content": "You are a correctness judge. Compare answers semantically."
        },
        {
            "role": "user",
            "content": f"""Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

Is the model's answer semantically correct? Consider if it conveys the same meaning/information as the ground truth, even if worded differently.

Respond with ONLY "CORRECT" or "INCORRECT"."""
        }
    ]

    try:
        client = _get_openrouter_client()
        response = await client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=judge_messages,
            temperature=0.0,
            max_tokens=20,
        )

        judge_text = response.choices[0].message.content.strip().upper()
        is_correct = "CORRECT" in judge_text and "INCORRECT" not in judge_text

        reward = 1.0 if is_correct else 0.0
        if arena.verbose:
            print(f"    [solver_judge] {judge_text} -> reward={reward}")

        return {actor: reward}

    except Exception as e:
        if arena.verbose:
            print(f"    [solver_judge] error: {e} -> reward=0.0")
        return {actor: 0.0}


def proposer_pass_rate_reward(
    rollout: Rollout,
    arena: Arena,
    target_pass_rate: float = 0.5,
) -> Dict[str, float]:
    """
    Rubric for proposer: reward based on solver pass rate.

    Target ~50% pass rate (not too easy, not too hard).
    This encourages the proposer to generate questions at the
    frontier of the solver's capability.
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    proposed = rollout.extras.get("proposed_question")

    # Invalid question = negative reward
    if not proposed or not proposed.get("question") or not proposed.get("ground_truth"):
        if arena.verbose:
            print(f"    [proposer_rubric] invalid question -> reward=-1.0")
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
# Solver Episode
# ---------------------------------------------------------------------------

class SpiceSolverEpisode(SingleTurnEpisode):
    """
    Single-turn episode where the Solver answers a question.

    The solver does NOT have access to the source document -
    it must answer based solely on its own knowledge.

    Artifact format:
    {
        "question": "What is the capital of France?",
        "ground_truth": "Paris"
    }
    """

    def __init__(self, solver_actor_id: str = "Solver"):
        self.solver_actor_id = solver_actor_id
        self._rubric = Rubric(funcs=[solver_llm_judge_reward])

    @property
    def episode_type(self) -> str:
        return "spice_solve"

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
        question = artifact.get("question", "")

        return f"""Answer the following question:

{question}

Think step by step, then provide your final answer.
You MUST end your response with: "The answer is: " followed by your answer."""


# ---------------------------------------------------------------------------
# Proposer Episode
# ---------------------------------------------------------------------------

class SpiceProposerEpisode(Episode):
    """
    Proposer reads a document and generates a question about it.

    Spawns trainable solver episodes to test the question difficulty.
    Reward based on solver pass rate (peaks at target_pass_rate).

    Artifact format (input):
    {
        "document": {
            "text": "The document content...",
            "title": "Optional title",
            "id": "doc_123"
        }
    }
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
        return "spice_propose"

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

        # 1. Extract document text
        document = artifact.get("document", {})
        if isinstance(document, dict):
            doc_text = document.get("text", "")
            doc_title = document.get("title", "")
        else:
            doc_text = str(document)
            doc_title = ""

        # 2. Build prompt and generate Q&A
        prompt = self._build_prompt(arena, doc_text, doc_title)
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

        # 3. Parse proposed question
        proposed = self._parse_question(response.text)
        state.data["proposed_question"] = proposed
        state.data["source_document"] = document

        # 4. Spawn solver episodes (TRAINABLE!) if valid question
        if proposed and proposed.get("question") and proposed.get("ground_truth"):
            solver_artifact = {
                "question": proposed["question"],
                "ground_truth": proposed["ground_truth"],
            }

            requests = [
                EpisodeRequest(
                    episode_type="spice_solve",
                    artifact=solver_artifact,
                    is_trainable=True,  # KEY DIFFERENCE from proposer_solver!
                )
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

        # Store generated question in questions store
        proposed = result.rollout.extras.get("proposed_question")
        if proposed and proposed.get("question") and proposed.get("ground_truth"):
            if "questions" in arena.stores:
                arena.stores["questions"].add(Artifact(
                    id=f"gen_{result.rollout.id}",
                    data=proposed,
                ))

        return result

    def _build_prompt(self, arena: Arena, doc_text: str, doc_title: str = "") -> Messages:
        actor = arena.actors.get(self.proposer_actor_id)

        title_line = f"Title: {doc_title}\n\n" if doc_title else ""

        user_content = f"""Read the following document and generate a question about it.

{title_line}Document:
{doc_text}

---

Generate a question that:
1. Can be answered using information from the document
2. Has a clear, unambiguous answer
3. Tests comprehension or reasoning about the content
4. Is challenging but answerable

Respond with a JSON object containing:
{{"question": "<the question text>", "ground_truth": "<the correct answer>"}}

CRITICAL INSTRUCTIONS:
- Do NOT write any text before or after the JSON
- Do NOT use markdown code blocks (backticks)
- Output ONLY a single JSON object
- Start your response with {{ and end with }}
- The ground_truth should be concise (a word, phrase, or number)"""

        messages: Messages = []
        if actor and actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_question(self, text: str) -> Optional[Dict[str, str]]:
        """Parse JSON {"question": "...", "ground_truth": "..."} from response."""
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
            "source_document": state.data.get("source_document"),
            "solver_rewards": child_rewards,
            "pass_rate": pass_rate,
        }


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class SpiceArena(Arena):
    """
    Arena for SPICE: Self-Play In Corpus Environment.

    Schedules proposer episodes that spawn nested solver episodes.
    Only proposer episodes are scheduled directly; solvers are spawned within.

    Requires:
    - "corpus" store: Pre-populated with documents
    - "questions" store: Grows as proposer generates questions
    """

    def __init__(
        self,
        client: InferenceClient,
        batch_size: int = 4,
        verbose: bool = False,
        credit_assigner: Optional[CreditAssigner] = None,
    ):
        super().__init__(
            client,
            credit_assigner=credit_assigner or GRPOCredit(),
            verbose=verbose,
        )
        self.batch_size = batch_size

    def get_batch(self) -> List[EpisodeRequest]:
        """
        Only launch proposer episodes.
        Solver episodes are spawned within each proposer's rollout.
        """
        corpus_store = self.stores.get("corpus")
        if not corpus_store or corpus_store.count() == 0:
            return []

        requests = []
        for _ in range(self.batch_size):
            doc = corpus_store.sample_one()
            if doc:
                requests.append(EpisodeRequest(
                    episode_type="spice_propose",
                    artifact={"document": doc.data},
                ))

        return requests
