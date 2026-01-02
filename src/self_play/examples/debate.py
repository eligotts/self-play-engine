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
import random
from typing import Any, Dict, List, Tuple

from ..core import (
    Role,
    Rollout,
    EpisodeState,
    AlternatingRolesEpisode,
    Rubric,
    Arena,
    InferenceClient,
    Artifact,
    EpisodeRequest,
    CreditAssigner,
    GRPOCredit,
)


# ---------------------------------------------------------------------------
# Rubric: LLM Judge (zero-sum)
# ---------------------------------------------------------------------------

def make_debate_rubric(aff_role: str, neg_role: str) -> Rubric:
    """
    Create a rubric that uses an LLM to judge the debate.

    Returns zero-sum rewards: winner gets +score, loser gets -score.
    """
    judge_prompt = f"""You are a JUDGE, not a debater. Your sole task is to evaluate the debate transcript below and output a JSON verdict.

CRITICAL INSTRUCTIONS:
- You are NOT a participant in this debate
- Do NOT write any arguments or continue the debate
- Do NOT write any text before or after the JSON
- Output ONLY a single JSON object

Evaluation criteria: quality of reasoning and how well each role directly responds to the details of the other role's argument. If they
do not directly respond to the other role's argument, they lose.

Required JSON format (output ONLY this, nothing else):
{{"winner": "{aff_role}", "score": 0.7, "reasoning": "One sentence explaining why"}}

Rules:
- "winner" must be exactly "{aff_role}" or "{neg_role}"
- "score" is winner's margin of victory from 0.5 (very close) to 1.0 (decisive)
- Start your response with {{ and end with }}

Example valid response:
{{"winner": "{neg_role}", "score": 0.65, "reasoning": "{neg_role} provided stronger evidence and effectively rebutted {aff_role}'s main points."}}

Now evaluate the following debate transcript and respond with ONLY JSON: /no_think"""

    async def llm_judge(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        if not rollout.steps:
            return {aff_role: 0.0, neg_role: 0.0}

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

        # Call LLM judge (not tracked for training - no role_id)
        response = await arena.call_model(
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        # Parse judgment - handle markdown code fences
        text = response.text.strip()

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

            if winner == aff_role:
                return {aff_role: score, neg_role: -score}
            elif winner == neg_role:
                return {aff_role: -score, neg_role: score}
            else:
                if arena.verbose:
                    print(f"      [judge] invalid winner '{winner}', expected {aff_role} or {neg_role}")
        except (json.JSONDecodeError, ValueError) as e:
            if arena.verbose:
                preview = response.text[:100].replace('\n', ' ')
                print(f"      [judge] JSON parse failed: {e}")
                print(f"      [judge] response was: {preview}...")

        # Default: tie
        return {aff_role: 0.0, neg_role: 0.0}

    return Rubric(funcs=[llm_judge])


# ---------------------------------------------------------------------------
# Debate Episode
# ---------------------------------------------------------------------------

class DebateEpisode(AlternatingRolesEpisode):
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
        self._rubric = make_debate_rubric(aff_role_id, neg_role_id)

    @property
    def episode_type(self) -> str:
        return "debate"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    @property
    def roles(self) -> Tuple[str, str]:
        return (self.aff_role_id, self.neg_role_id)

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_debate_arena(
    client: InferenceClient,
    topics: List[str] | None = None,
    num_rounds: int = 3,
    batch_size: int = 4,
    verbose: bool = False,
) -> DebateArena:
    """Create a complete debate arena."""
    arena = DebateArena(
        client,
        batch_size=batch_size,
        verbose=verbose,
        credit_assigner=GRPOCredit(),
    )

    arena.add_role(Role(
        id="Aff",
        system_prompt="You are a skilled debater arguing for the AFFIRMATIVE side (supporting the proposition). "
                      "Make clear, persuasive arguments supported by logic and evidence. Be concise. Each of your turns should be no more than 3 sentences.",
        temperature=0.8,
        max_tokens=1024,
    ))

    arena.add_role(Role(
        id="Neg",
        system_prompt="You are a skilled debater arguing for the NEGATIVE side (opposing the proposition). "
                      "Make clear, persuasive arguments supported by logic and evidence. Be concise. Each of your turns should be no more than 3 sentences.",
        temperature=0.8,
        max_tokens=1024,
    ))

    arena.add_episode("debate", DebateEpisode(num_rounds=num_rounds))

    topic_store = arena.add_store("topics")
    if topics:
        for topic in topics:
            topic_store.add(Artifact(
                id=f"topic_{hash(topic) % 10000}",
                data={"topic": topic},
            ))

    return arena
