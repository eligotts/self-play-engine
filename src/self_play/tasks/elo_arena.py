"""
EloArena: Tournament competition with persistent Elo ratings.

Pattern: Head-to-head matches where both players respond to same challenge → LLM judge picks winner → Elo updates.

This demonstrates:
- Persistent arena state (Elo ratings across batches)
- Same model playing both sides (self-play)
- RAECredit for role-conditioned advantages
- Dynamic challenge generation via ChallengeProposerEpisode
"""
from __future__ import annotations

import json
import math
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
    RAECredit,
    CreditAssigner,
    Messages,
)


async def elo_match_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Head-to-head comparison using LLM judge.
    Returns zero-sum rewards and updates arena Elo ratings.
    """
    extras = rollout.extras
    player0 = extras.get("player0_role", "Player0")
    player1 = extras.get("player1_role", "Player1")
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

    response = await arena.call_model(judge_prompt, temperature=0.3, max_tokens=200)

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

    # Assign rewards
    if winner == "A":
        rewards = {player0: 1.0, player1: -1.0}
        winner_role, loser_role = player0, player1
    elif winner == "B":
        rewards = {player0: -1.0, player1: 1.0}
        winner_role, loser_role = player1, player0
    else:
        rewards = {player0: 0.0, player1: 0.0}
        winner_role, loser_role = None, None

    # Update Elo ratings if arena has them
    if hasattr(arena, "elo_ratings") and winner_role:
        arena.update_elo(winner_role, loser_role)

    if arena.verbose:
        print(f"    [elo_match] winner={winner} | {player0} vs {player1}")

    return rewards


class EloMatchEpisode(Episode):
    """
    Single head-to-head match in the tournament.

    Artifact format:
    {
        "challenge": "Write a creative opening line for a mystery novel"
    }
    """

    def __init__(
        self,
        player0_role: str = "Player0",
        player1_role: str = "Player1",
    ):
        self.player0_role = player0_role
        self.player1_role = player1_role
        self._rubric = Rubric(funcs=[elo_match_reward])

    @property
    def episode_type(self) -> str:
        return "elo_match"

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

        # Store role info
        state.data["player0_role"] = self.player0_role
        state.data["player1_role"] = self.player1_role

        # Both players respond to the same challenge
        for i, role_id in enumerate([self.player0_role, self.player1_role]):
            role_config = arena.roles.get(role_id)
            prompt = [
                {"role": "system", "content": role_config.system_prompt if role_config else "You are a creative competitor."},
                {"role": "user", "content": f"Challenge: {challenge}\n\nProvide your best response."},
            ]

            response = await self.call_model(role_id, prompt, arena)
            state.data[f"response{i}"] = response.text

        state.done = True
        return state

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        return {
            "player0_role": state.data.get("player0_role", self.player0_role),
            "player1_role": state.data.get("player1_role", self.player1_role),
            "response0": state.data.get("response0", ""),
            "response1": state.data.get("response1", ""),
        }


async def challenge_proposer_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """Empty reward function - proposer episodes are not trained."""
    # Must return a score for all actors, even though this episode is non-trainable
    return {actor: 0.0 for actor in rollout.actors}


class ChallengeProposerEpisode(Episode):
    """
    Non-trainable episode that generates new challenges for the arena.

    Pulls example challenges from the store, asks the model to generate
    a new creative challenge, and adds it to the store.

    Artifact format (input):
    {
        "examples": [{"challenge": "..."}, {"challenge": "..."}, ...]
    }
    """

    def __init__(self, proposer_role: str = "ChallengeProposer"):
        self.proposer_role = proposer_role
        self._rubric = Rubric(funcs=[challenge_proposer_reward])

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
        response = await self.call_model(self.proposer_role, prompt, arena)

        step = Step(
            role_id=self.proposer_role,
            prompt=prompt,
            completion=response.completion,
            prompt_token_ids=response.prompt_token_ids,
            completion_token_ids=response.completion_token_ids,
            completion_logprobs=response.completion_logprobs,
        )
        state.trajectory.append(step)

        # Parse and store new challenge
        new_challenge = self._parse_challenge(response.text)
        state.data["proposed_challenge"] = new_challenge

        if new_challenge and new_challenge.get("challenge"):
            if "challenges" in arena.stores:
                challenge_id = f"gen_{len(arena.stores['challenges'].artifacts)}"
                arena.stores["challenges"].add(Artifact(
                    id=challenge_id,
                    data=new_challenge,
                ))
                if arena.verbose:
                    print(f"    [challenge_proposer] added: {new_challenge['challenge'][:50]}...")

        state.done = True
        return state

    def _build_prompt(self, arena: Arena, examples: List[Dict]) -> Messages:
        role = arena.roles.get(self.proposer_role)

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

Respond with ONLY a JSON object:
{{"challenge": "<the challenge text>"}}

Do NOT use markdown code blocks. Output ONLY the JSON object."""

        messages: Messages = []
        if role and role.system_prompt:
            messages.append({"role": "system", "content": role.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_challenge(self, text: str) -> Optional[Dict[str, str]]:
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
            "proposed_challenge": state.data.get("proposed_challenge"),
        }


class EloArena(Arena):
    """
    Tournament arena with persistent Elo ratings.

    Tracks ratings across batches and logs progression.
    """

    def __init__(
        self,
        client: InferenceClient,
        batch_size: int = 4,
        proposer_batch_size: int = 1,
        initial_elo: float = 1500.0,
        k_factor: float = 32.0,
        verbose: bool = False,
        credit_assigner: CreditAssigner | None = None,
    ):
        super().__init__(
            client,
            credit_assigner=credit_assigner or RAECredit(decay=0.9),
            verbose=verbose,
        )
        self.batch_size = batch_size
        self.proposer_batch_size = proposer_batch_size
        self.initial_elo = initial_elo
        self.k_factor = k_factor

        # Persistent Elo ratings
        self.elo_ratings: Dict[str, float] = {}
        self.match_history: List[Dict] = []

    def get_elo(self, player: str) -> float:
        """Get Elo rating for a player, initializing if needed."""
        if player not in self.elo_ratings:
            self.elo_ratings[player] = self.initial_elo
        return self.elo_ratings[player]

    def update_elo(self, winner: str, loser: str) -> None:
        """Update Elo ratings after a match."""
        winner_elo = self.get_elo(winner)
        loser_elo = self.get_elo(loser)

        # Expected scores
        exp_winner = 1.0 / (1.0 + math.pow(10, (loser_elo - winner_elo) / 400))
        exp_loser = 1.0 - exp_winner

        # Update ratings
        self.elo_ratings[winner] = winner_elo + self.k_factor * (1.0 - exp_winner)
        self.elo_ratings[loser] = loser_elo + self.k_factor * (0.0 - exp_loser)

        # Log
        self.match_history.append({
            "winner": winner,
            "loser": loser,
            "winner_elo_before": winner_elo,
            "loser_elo_before": loser_elo,
            "winner_elo_after": self.elo_ratings[winner],
            "loser_elo_after": self.elo_ratings[loser],
        })

        if self.verbose:
            print(f"    [elo] {winner}: {winner_elo:.0f} -> {self.elo_ratings[winner]:.0f} | "
                  f"{loser}: {loser_elo:.0f} -> {self.elo_ratings[loser]:.0f}")

    def get_batch(self) -> List[EpisodeRequest]:
        """Sample challenges from store and include proposer requests."""
        store = self.stores.get("challenges")
        if not store or store.count() == 0:
            return []

        # Sample challenges for matches
        challenges = store.sample(k=self.batch_size)
        match_requests = [
            EpisodeRequest(episode_type="elo_match", artifact=c.data)
            for c in challenges
        ]

        # Add proposer requests (non-trainable) to generate new challenges
        proposer_requests = []
        for _ in range(self.proposer_batch_size):
            examples = store.sample(k=min(3, store.count()))
            proposer_requests.append(EpisodeRequest(
                episode_type="challenge_propose",
                artifact={"examples": [e.data for e in examples]},
                is_trainable=False,
            ))

        return match_requests + proposer_requests
