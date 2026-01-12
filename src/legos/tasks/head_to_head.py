"""
HeadToHead: Tournament competition with LLM-judged matches.

Pattern: Head-to-head matches where both players respond to same challenge → LLM judge picks winner → zero-sum rewards.

This demonstrates:
- Same model playing both sides (self-play)
- Dynamic challenge generation via ChallengeProposerEpisode
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..core import (
    Rollout,
    EpisodeState,
    Episode,
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


async def match_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Head-to-head comparison using LLM judge.
    Returns zero-sum rewards.
    """
    extras = rollout.extras
    player0 = extras.get("player0_actor", "Player0")
    player1 = extras.get("player1_actor", "Player1")
    challenge = rollout.artifact.get("challenge", "")

    response0 = extras.get("response0", "")
    response1 = extras.get("response1", "")

    if not response0 or not response1:
        return {player0: 0.0, player1: 0.0}

    # Use LLM judge (non-trainable) to compare responses
    judge_prompt = [
        {"role": "system", "content": "You are a fair judge comparing two responses. Pick the better one."},
        {"role": "user", "content": f"""Challenge: {challenge}

Response A:
{response0}

Response B:
{response1}

Which response is better? Consider: accuracy, clarity, completeness, and creativity.
Respond with ONLY a JSON object: {{"winner": "A" or "B" or "tie", "reason": "<brief reason>"}}"""},
    ]

    response = await arena.call_model(judge_prompt, max_tokens=200)

    # Parse winner
    try:
        text = response.text.strip()
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            winner = data.get("winner", "tie").upper()
        else:
            winner = "TIE"
    except:
        winner = "TIE"

    # Assign zero-sum rewards
    if winner == "A":
        rewards = {player0: 1.0, player1: -1.0}
    elif winner == "B":
        rewards = {player0: -1.0, player1: 1.0}
    else:
        rewards = {player0: 0.0, player1: 0.0}

    if arena.verbose:
        print(f"    [match] winner={winner} | {player0} vs {player1}")

    return rewards


class MatchEpisode(Episode):
    """
    Single head-to-head match in the tournament.

    Artifact format:
    {
        "challenge": "Write a creative opening line for a mystery novel"
    }
    """

    def __init__(
        self,
        player0_actor: str = "Player0",
        player1_actor: str = "Player1",
    ):
        self.player0_actor = player0_actor
        self.player1_actor = player1_actor
        self._rubric = Rubric(funcs=[match_reward])

    @property
    def episode_type(self) -> str:
        return "match"

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

        challenge = artifact.get("challenge", "")

        # Store actor info
        state.data["player0_actor"] = self.player0_actor
        state.data["player1_actor"] = self.player1_actor

        # Both players respond to the same challenge
        for i, actor_id in enumerate([self.player0_actor, self.player1_actor]):
            actor_config = arena.actors.get(actor_id)
            prompt = [
                {"role": "system", "content": actor_config.system_prompt if actor_config else "You are a creative competitor."},
                {"role": "user", "content": f"Challenge: {challenge}\n\nProvide your best response."},
            ]

            response = await self.call_model(actor_id, prompt, arena)
            state.data[f"response{i}"] = response.text

            # Create Step and append to trajectory for training
            step = Step(
                actor_id=actor_id,
                prompt=prompt,
                completion=response.completion,
                prompt_token_ids=response.prompt_token_ids,
                completion_token_ids=response.completion_token_ids,
                completion_logprobs=response.completion_logprobs,
            )
            state.trajectory.append(step)

        state.done = True
        return state

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        return {
            "player0_actor": state.data.get("player0_actor", self.player0_actor),
            "player1_actor": state.data.get("player1_actor", self.player1_actor),
            "response0": state.data.get("response0", ""),
            "response1": state.data.get("response1", ""),
        }


class ChallengeProposerEpisode(Episode):
    """
    Non-trainable episode that generates new challenges for the arena.

    Pulls example challenges from the store, asks the model to generate
    a new creative challenge, and adds it to the store via generate().

    Artifact format (input):
    {
        "examples": [{"challenge": "..."}, {"challenge": "..."}, ...]
    }
    """

    def __init__(self, proposer_actor: str = "ChallengeProposer"):
        self.proposer_actor = proposer_actor
        self._rubric = Rubric(funcs=[])  # Non-trainable, defaults to 0 reward

    @property
    def episode_type(self) -> str:
        return "challenge_propose"

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

        # Generate new challenge
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

        # Store proposed challenge in state
        state.data["proposed_challenge"] = {"challenge": response.text.strip()}

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

        # Add valid challenge to store
        proposed = result.rollout.extras.get("proposed_challenge")
        if proposed and proposed.get("challenge"):  # Check challenge isn't empty
            if "challenges" in arena.stores:
                challenge_id = f"gen_{arena.stores['challenges'].count()}"
                arena.stores["challenges"].add(Artifact(
                    id=challenge_id,
                    data=proposed,
                ))
                if arena.verbose:
                    print(f"    [challenge_proposer] added: {proposed['challenge'][:50]}...")

        return result

    def _build_prompt(self, arena: Arena, examples: List[Dict]) -> Messages:
        actor = arena.actors.get(self.proposer_actor)

        examples_text = ""
        if examples:
            examples_text = "Here are some example challenges:\n\n"
            for i, ex in enumerate(examples, 1):
                examples_text += f"{i}. {ex.get('challenge', '')}\n"
            examples_text += "\n"

        user_content = f"""{examples_text}Generate a new creative challenge for a writing competition. The challenge should:
1. Be specific and clear
2. Allow for creative responses
3. Be different from the examples above
4. Be answerable in 1-3 sentences

Respond with ONLY the challenge text. No other commentary or formatting."""

        messages: Messages = []
        if actor and actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        return {
            "proposed_challenge": state.data.get("proposed_challenge"),
        }


class HeadToHeadArena(Arena):
    """
    Tournament arena for head-to-head competition.

    Runs matches where both players respond to the same challenge,
    with an LLM judge picking the winner.
    """

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
        """Sample challenges from store and include proposer requests."""
        store = self.stores.get("challenges")
        if not store or store.count() == 0:
            return []

        # Sample challenges for matches
        challenges = store.sample(k=self.episodes_per_step)
        match_requests = [
            EpisodeRequest(episode_type="match", artifact=c.data)
            for c in challenges
        ]

        # Add proposer requests (non-trainable) to generate new challenges
        proposer_requests = []
        for _ in range(self.proposer_episodes_per_step):
            examples = store.sample(k=min(3, store.count()))
            proposer_requests.append(EpisodeRequest(
                episode_type="challenge_propose",
                artifact={"examples": [e.data for e in examples]},
                is_trainable=False,
            ))

        return match_requests + proposer_requests
