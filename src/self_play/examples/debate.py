"""
Debate Example: Multi-turn, zero-sum competitive dialogue.

This example demonstrates:
- Multi-turn episodes with alternating roles
- Zero-sum scoring (one wins, one loses)
- LLM judge as rubric (NOT a trainable role)

Debate Structure:
1. Topic is selected from artifact
2. Aff and Neg alternate for N rounds
3. Rubric calls LLM to judge and scores zero-sum
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()



from ..core import (
    Rollout,
    EpisodeState,
    MultiTurnEpisode,
    Rubric,
    Arena,
    InferenceClient,
    EpisodeRequest,
    CreditAssigner,
)


# ---------------------------------------------------------------------------
# Rubric: LLM Judge (zero-sum)
# ---------------------------------------------------------------------------

async def debate_judge(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    LLM judge for debate. Discovers actors dynamically from rollout.

    Returns zero-sum rewards: winner gets +score, loser gets -score.
    """
    actors = rollout.actors

    if not actors or len(actors) < 2:
        return {actor: 0.0 for actor in actors}

    # Build actor list for judge prompt (consistent ordering)
    actor_list = sorted(actors)

    judge_prompt = f"""You are a JUDGE, not a debater. Your sole task is to evaluate the debate transcript below and output a JSON verdict.

CRITICAL INSTRUCTIONS:
- You are NOT a participant in this debate
- Do NOT write any arguments or continue the debate
- Do NOT write any text before or after the JSON
- Output ONLY a single JSON object

Evaluation criteria: YOUR SOLE EVALUATION CRITTERIA IS WHICH SIDE WAS KINDER IN THEIR RESPONSES TO THE OTHER SIDE'S ARGUMENTS.
You should be looking for evidence of kindness in the responses, not the quality of the arguments. Responses that attack the other side's arguments without being kind lose.

Required JSON format (output ONLY this, nothing else):
{{"winner": "<one of the actors>", "score": 0.7, "reasoning": "One sentence explaining why"}}

Rules:
- "winner" must be exactly one of: {', '.join(actor_list)}
- "score" is winner's margin of victory from 0.5 (very close) to 1.0 (decisive)
- Start your response with {{ and end with }}

Example valid response:
{{"winner": "{actor_list[1]}", "score": 0.65, "reasoning": "{actor_list[1]} provided stronger evidence and effectively rebutted {actor_list[0]}'s main points."}}

Now evaluate the following debate transcript and respond with ONLY JSON: /no_think"""

    # The final step's prompt contains the full history (built by run_chat_loop)
    # Extract it from the "user" message, then append the final completion
    final_step = rollout.steps[-1]

    # Get history from final step's prompt (the user message content)
    history = ""
    for msg in final_step.prompt:
        if msg.get("role") == "user":
            history = msg.get("content", "")
            break

    # Build full transcript: history + final completion
    # History already has format "Topic: ...\n\n{role}: {text}\n\n{role}: {text}..."
    transcript = history + f"\n\n{final_step.role_id}: {final_step.completion_text}"

    # Call Gemini judge via OpenRouter (not tracked for training)
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    response = await openrouter_client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    # Parse judgment - handle markdown code fences
    text = (response.choices[0].message.content or "").strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try to extract JSON from the response
    try:
        # Find JSON object in text
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            text = text[start:end]

        data = json.loads(text)
        winner = data.get("winner", "")
        score = float(data.get("score", 0.5))

        if winner in actors:
            # Zero-sum: winner gets +score, all others get -score
            return {actor: score if actor == winner else -score for actor in actors}
        else:
            if arena.verbose:
                print(f"      [judge] invalid winner '{winner}', expected one of {actors}")
    except (json.JSONDecodeError, ValueError) as e:
        if arena.verbose:
            raw_text = response.choices[0].message.content or ""
            preview = raw_text[:100].replace('\n', ' ')
            print(f"      [judge] JSON parse failed: {e}")
            print(f"      [judge] response was: {preview}...")

    # Default: tie
    return {actor: 0.0 for actor in actors}


# ---------------------------------------------------------------------------
# Debate Episode
# ---------------------------------------------------------------------------

class DebateEpisode(MultiTurnEpisode):
    """
    Multi-turn debate episode.

    Two debaters (Aff and Neg) alternate arguing for/against a topic.
    Each turn, the current speaker sees the topic and transcript so far.
    """

    def __init__(
        self,
        aff_role_id: str = "Aff",
        neg_role_id: str = "Neg",
        num_rounds: int = 3,
    ):
        super().__init__(max_turns=num_rounds * 2)
        self.aff_role_id = aff_role_id
        self.neg_role_id = neg_role_id
        self.num_rounds = num_rounds
        self._rubric = Rubric(funcs=[debate_judge])

    @property
    def episode_type(self) -> str:
        return "debate"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        start_idx = random.randint(0, 1)
        state.data["start_idx"] = start_idx
        return self.aff_role_id if start_idx == 0 else self.neg_role_id

    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        start_idx = state.data.get("start_idx", 0)
        idx = (start_idx + state.turn) % 2
        return self.aff_role_id if idx == 0 else self.neg_role_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        topic = artifact.get("topic", "Unknown topic")
        initial_actor = state.current_actor
        side = "affirmative" if initial_actor == self.aff_role_id else "negative"
        state.data["topic"] = topic

        return f"Topic: {topic}"

    async def env_response(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> str:
        """Return transition text for next speaker."""
        next_role_id = self.get_next_actor(state, artifact)
        side = "affirmative" if next_role_id == self.aff_role_id else "negative"
        # return f"You are arguing for the {side} side. Respond to your opponent's points. Be concise."
        return ""

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        return {
            "topic": state.data.get("topic", "Unknown topic"),
            "num_turns": state.turn,
        }


# ---------------------------------------------------------------------------
# Debate Arena
# ---------------------------------------------------------------------------

class DebateArena(Arena):
    """Arena that schedules debate episodes from a topic store."""

    def __init__(
        self,
        client: InferenceClient,
        batch_size: int = 4,
        verbose: bool = False,
        credit_assigner: CreditAssigner | None = None,
    ):
        super().__init__(client, credit_assigner=credit_assigner, verbose=verbose)
        self.batch_size = batch_size

    def get_batch(self) -> List[EpisodeRequest]:
        if "topics" not in self.stores:
            return []

        topics = self.stores["topics"].sample(k=self.batch_size)
        return [
            EpisodeRequest(
                episode_type="debate",
                artifact=topic.data,
                meta={"artifact_id": topic.id},
            )
            for topic in topics
        ]
