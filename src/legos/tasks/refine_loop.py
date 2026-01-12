"""
RefineLoop: Generator-Critic iterative refinement.

Pattern: Generator produces draft → Critic provides feedback → Generator revises → repeat N times.

This demonstrates:
- Cooperative multi-actor training (both actors share same reward)
- Iterative refinement loop with state tracking
- LLM judge for final quality assessment
- Dynamic task generation via TaskProposerEpisode
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# OpenRouter client for judge calls
_openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

from ..core import (
    Rollout,
    EpisodeState,
    Episode,
    MultiTurnEpisode,
    Step,
    Rubric,
    Arena,
    InferenceClient,
    EpisodeRequest,
    Artifact,
    GRPOCredit,
    CreditAssigner,
    Messages,
    GenerateResult,
)


async def refine_loop_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Score based on final output quality using LLM judge.
    Both Generator and Critic receive the same reward (cooperative).
    """
    actors = list(rollout.actors)
    if not actors:
        return {}

    extras = rollout.extras
    final_draft = extras.get("final_draft", "")
    task = rollout.artifact.get("task", "")
    requirements = rollout.artifact.get("requirements", "")

    if not final_draft:
        return {actor: 0.0 for actor in actors}

    # Use LLM judge (non-trainable) via OpenRouter to score final quality
    judge_messages = [
        {"role": "system", "content": "You are a quality assessor. Rate the given output on a scale of 0-10."},
        {"role": "user", "content": f"""Task: {task}

Requirements:
{requirements}

Final Output:
{final_draft}

Rate the quality of this output from 0-10 based on how well it meets the requirements.
Consider: clarity, completeness, accuracy, and style.

Respond with ONLY a JSON object: {{"score": <0-10>, "reason": "<very very brief reason>"}}"""},
    ]

    try:
        response = await _openrouter_client.chat.completions.create(
            model="google/gemini-3-flash-preview",
            messages=judge_messages,
            temperature=0.3,
            max_tokens=200,
        )
    except Exception as exc:
        raise RuntimeError(
            "LLM judge call failed. Check OPENROUTER_API_KEY and network access."
        ) from exc

    # Parse score
    try:
        text = response.choices[0].message.content.strip()
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            score = float(data.get("score", 5)) / 10.0  # Normalize to 0-1
        else:
            score = 0.5
    except:
        score = 0.5

    score = max(0.0, min(1.0, score))

    if arena.verbose:
        print(f"    [refine_loop] final_quality={score:.2f} iterations={extras.get('num_iterations', 0)}")

    # Both actors get same reward (cooperative)
    return {actor: score for actor in actors}


class RefineLoopEpisode(MultiTurnEpisode):
    """
    Iterative refinement with Generator and Critic alternating.

    Artifact format:
    {
        "task": "Write a haiku about coding",
        "requirements": "Must follow 5-7-5 syllable structure, be about programming"
    }
    """

    def __init__(
        self,
        generator_actor: str = "Generator",
        critic_actor: str = "Critic",
        num_iterations: int = 2,
    ):
        super().__init__(max_turns=num_iterations * 2)
        self.generator_actor = generator_actor
        self.critic_actor = critic_actor
        self.num_iterations = num_iterations
        self._rubric = Rubric(funcs=[refine_loop_reward])

    @property
    def episode_type(self) -> str:
        return "refine_loop"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def init_state(self, state: EpisodeState, artifact: Any) -> None:
        state.data["drafts"] = []
        state.data["feedback"] = []

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        return self.generator_actor

    def get_initial_prompt(self, arena: Arena, artifact: Any, state: EpisodeState) -> str:
        return f"""Task: {artifact.get('task', '')}

Requirements:
{artifact.get('requirements', '')}

Write your first draft."""

    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        if state.current_actor == self.generator_actor:
            return self.critic_actor
        return self.generator_actor

    def get_observation(self, state: EpisodeState, arena: Arena, artifact: Any) -> str:
        """Provide actor-specific context."""
        if state.current_actor == self.critic_actor:
            # Critic sees the latest draft
            draft = state.data["drafts"][-1] if state.data["drafts"] else ""
            iteration = len(state.data["drafts"])
            return f"""=== REVIEW DRAFT {iteration} ===

{draft}

Provide specific feedback to improve this draft.
Focus on: clarity, completeness, and meeting the requirements.
Be constructive but critical.
ONLY PROVIDE CONCISE FEEDBACK. This feedback MUST NOT include a suggestion for what a revised draft should look like.
"""
        else:
            # Generator sees latest feedback
            if state.data["feedback"]:
                feedback = state.data["feedback"][-1]
                return f"""=== FEEDBACK RECEIVED ===

{feedback}

Revise your draft based on this feedback.
DO NOT DIRECTLY RESPOND TO THE FEEDBACK. Instead, output your revised draft based on the feedback.
IT IS CRUCIAL THAT YOU ONLY OUTPUT YOUR REVISED DRAFT BASED ON THE FEEDBACK. DO NOT INCLUDE ANY OTHER COMMENTARY."""
            return ""

    async def env_response(self, state: EpisodeState, arena: Arena, artifact: Any) -> str:
        """Track drafts and feedback, control termination."""
        last_step = state.last_step
        if not last_step:
            return ""

        completion = last_step.completion_text

        if last_step.actor_id == self.generator_actor:
            state.data["drafts"].append(completion)
            draft_num = len(state.data["drafts"])

            if draft_num > self.num_iterations:
                return f"[Draft {draft_num} - FINAL]"
            return f"[Draft {draft_num} submitted for review]"

        else:  # Critic
            state.data["feedback"].append(completion)
            return "[Feedback provided, awaiting revision]"

    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        # Done after final draft
        return len(state.data.get("drafts", [])) > self.num_iterations

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        drafts = state.data.get("drafts", [])
        return {
            "drafts": drafts,
            "feedback": state.data.get("feedback", []),
            "num_iterations": len(drafts),
            "final_draft": drafts[-1] if drafts else "",
        }


class TaskProposerEpisode(Episode):
    """
    Non-trainable episode that generates new tasks for the arena.

    Pulls example tasks from the store, asks the model to generate
    a new writing task, and adds it to the store.

    Artifact format (input):
    {
        "examples": [{"task": "...", "requirements": "..."}, ...]
    }
    """

    def __init__(self, proposer_actor: str = "TaskProposer"):
        self.proposer_actor = proposer_actor
        self._rubric = Rubric(funcs=[])  # Non-trainable, defaults to 0 reward

    @property
    def episode_type(self) -> str:
        return "task_propose"

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

        # Build prompt with examples
        examples = artifact.get("examples", [])
        prompt = self._build_prompt(arena, examples)

        # Generate new task via arena.call_model
        response = await self.call_model(self.proposer_actor, prompt, arena)

        step = Step(
            actor_id=self.proposer_actor,
            prompt=prompt,
            completion=response.completion,
            prompt_token_ids=response.prompt_token_ids,
            completion_token_ids=response.completion_token_ids,
            completion_logprobs=response.completion_logprobs,
        )
        state.trajectory.append(step)

        # Parse proposed task
        new_task = self._parse_task(response.text)
        state.data["proposed_task"] = new_task
        state.data["raw_response"] = response.text  # Store raw for debugging

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

        # Store generated task in tasks store
        proposed = result.rollout.extras.get("proposed_task")
        if proposed and proposed.get("task") and proposed.get("requirements"):
            if "tasks" in arena.stores:
                task_id = f"gen_{arena.stores['tasks'].count()}"
                arena.stores["tasks"].add(Artifact(
                    id=task_id,
                    data=proposed,
                ))
                if arena.verbose:
                    print(f"    [task_proposer] added: {proposed['task']}...{proposed['requirements']}")

        return result

    def _build_prompt(self, arena: Arena, examples: List[Dict]) -> Messages:
        actor = arena.actors.get(self.proposer_actor)

        examples_text = ""
        if examples:
            examples_text = "Here are some example tasks:\n\n"
            for i, ex in enumerate(examples, 1):
                examples_text += f"{i}. Task: {ex.get('task', '')}\n   Requirements: {ex.get('requirements', '')}\n\n"

        user_content = f"""{examples_text}Generate a new writing task for iterative refinement. The task should:
1. Be a specific writing assignment (e.g., poem, email, description, etc.)
2. Have clear, measurable requirements
3. Be completable in a few sentences (no more than 200 words)
4. Be different from the examples above
5. Not be too hard (dont ask for a limerick or something incredibly complex like that)

Respond with ONLY a JSON object:
{{"task": "<the task description>", "requirements": "<specific requirements for the task>"}}

Do NOT use markdown code blocks. Output ONLY the JSON object."""

        messages: Messages = []
        if actor and actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_task(self, text: str) -> Optional[Dict[str, str]]:
        try:
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        return {
            "proposed_task": state.data.get("proposed_task"),
        }


class RefineLoopArena(Arena):
    """Arena for refinement tasks."""

    def __init__(
        self,
        client: InferenceClient,
        episodes_per_step: int = 4,
        proposer_episodes_per_step: int = 1,
        verbose: bool = False,
        credit_assigner: CreditAssigner | None = None,
    ):
        super().__init__(
            client,
            credit_assigner=credit_assigner or GRPOCredit(),
            verbose=verbose,
        )
        self.episodes_per_step = episodes_per_step
        self.proposer_episodes_per_step = proposer_episodes_per_step

    def get_batch(self) -> List[EpisodeRequest]:
        """Sample tasks from store and include proposer requests."""
        store = self.stores.get("tasks")
        if not store or store.count() == 0:
            return []

        # Sample tasks for refinement
        tasks = store.sample(k=self.episodes_per_step)
        refine_requests = [
            EpisodeRequest(episode_type="refine_loop", artifact=task.data)
            for task in tasks
        ]

        # Add proposer requests (non-trainable) to generate new tasks
        proposer_requests = []
        for _ in range(self.proposer_episodes_per_step):
            examples = store.sample(k=min(3, store.count()))
            proposer_requests.append(EpisodeRequest(
                episode_type="task_propose",
                artifact={"examples": [e.data for e in examples]},
                is_trainable=False,
            ))

        return refine_requests + proposer_requests
