"""
Debate Example: Multi-turn, zero-sum competitive dialogue.

This example demonstrates:
- Multi-turn episodes with alternating roles
- Zero-sum scoring (one wins, one loses)
- LLM judge as rubric (NOT a trainable role)

Debate Structure:
1. Topic is selected from seed
2. Aff and Neg alternate for N rounds
3. Rubric calls LLM to judge and scores zero-sum
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..core import (
    Role,
    Messages,
    Rollout,
    EpisodeState,
    AlternatingRolesEpisode,
    Rubric,
    Arena,
    InferenceClient,
    Artifact,
    EpisodeRequest,
)


# ---------------------------------------------------------------------------
# Rubric: LLM Judge (zero-sum)
# ---------------------------------------------------------------------------

def make_debate_rubric(aff_role: str, neg_role: str) -> Rubric:
    """
    Create a rubric that uses an LLM to judge the debate.

    Returns zero-sum rewards: winner gets +score, loser gets -score.
    """
    judge_prompt = f"""You are an impartial judge evaluating a debate between {aff_role} and {neg_role}.

Evaluate based on:
1. Strength of arguments
2. Quality of reasoning
3. Persuasiveness

Respond with JSON: {{"winner": "{aff_role}" or "{neg_role}", "score": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    async def llm_judge(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        # Build transcript from steps
        lines = [f"[{step.role_id}]: {step.completion_text}" for step in rollout.steps]
        transcript = "\n\n".join(lines)

        # Call LLM judge (not tracked for training - no role_id)
        response = await arena.call_model(
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        # Parse judgment
        try:
            data = json.loads(response.text)
            winner = data.get("winner", "")
            score = float(data.get("score", 0.5))

            if winner == aff_role:
                return {aff_role: score, neg_role: -score}
            elif winner == neg_role:
                return {aff_role: -score, neg_role: score}
        except (json.JSONDecodeError, ValueError):
            pass

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
    ) -> Messages:
        topic = artifact.get("topic", "Unknown topic")
        role = arena.roles[self.aff_role_id]

        messages: Messages = []
        if role.system_prompt:
            messages.append({"role": "system", "content": role.system_prompt})

        messages.append({
            "role": "user",
            "content": f"""Topic: {topic}

You are arguing for the affirmative side.
This is your opening statement. Present your main argument."""
        })

        state.data["topic"] = topic
        return messages

    async def env_response(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> str:
        """Return transition text for next speaker."""
        next_role_id = self.roles[state.turn % 2]
        side = "affirmative" if next_role_id == self.aff_role_id else "negative"
        return f"You are arguing for the {side} side. Respond to your opponent's points."

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

    def __init__(self, client: InferenceClient, batch_size: int = 4):
        super().__init__(client)
        self.batch_size = batch_size

    def get_batch(self) -> List[EpisodeRequest]:
        if "topics" not in self.stores:
            return []

        topics = self.stores["topics"].sample(k=self.batch_size)
        return [
            EpisodeRequest(episode_type="debate", seed=topic.data)
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
) -> DebateArena:
    """Create a complete debate arena."""
    arena = DebateArena(client, batch_size=batch_size)

    arena.add_role(Role(
        id="Aff",
        system_prompt="You are a skilled debater arguing for the affirmative side. "
                      "Make clear, persuasive arguments supported by logic and evidence.",
        temperature=0.8,
        max_tokens=300,
    ))

    arena.add_role(Role(
        id="Neg",
        system_prompt="You are a skilled debater arguing for the negative side. "
                      "Make clear, persuasive arguments supported by logic and evidence.",
        temperature=0.8,
        max_tokens=300,
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
