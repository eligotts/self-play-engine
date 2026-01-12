"""
SimpleNegotiation: Two-player resource trading game.

This example demonstrates a faithful replication of SPIRAL's SimpleNegotiation environment:
- Two players with opposite resource preferences
- Structured actions: [Offer], [Accept], [Deny]
- Winner determined by inventory value change
- Uses RAECredit for actor-conditioned advantage estimation

Game Structure:
1. Both players start with 10 Wood + 10 Gold
2. Player 0 values: Wood=5, Gold=15 (prefers Gold)
3. Player 1 values: Wood=15, Gold=5 (prefers Wood)
4. Players alternate making offers, accepting, denying, or negotiating
5. Game ends at max_turns (default 10)
6. Winner = player with larger inventory value change
"""
from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core import (
    Rollout,
    EpisodeState,
    MultiTurnEpisode,
    Rubric,
    Arena,
    InferenceClient,
    EpisodeRequest,
    CreditAssigner,
    RAECredit,
)


# ---------------------------------------------------------------------------
# Constants (matching SPIRAL's SimpleNegotiation)
# ---------------------------------------------------------------------------

RESOURCE_NAMES = ["Wood", "Gold"]
STARTING_RESOURCES = {"Wood": 10, "Gold": 10}

# Opposite preferences - makes mutually beneficial trades possible
PLAYER_VALUES = {
    "Player0": {"Wood": 5, "Gold": 15},   # Prefers Gold
    "Player1": {"Wood": 15, "Gold": 5},   # Prefers Wood
}

# Rewards (matching SPIRAL)
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = 0.0
REWARD_INVALID = -1.5
REWARD_OPPONENT_INVALID = 0.5


# ---------------------------------------------------------------------------
# Action Parsing (from SPIRAL)
# ---------------------------------------------------------------------------

ACCEPT_PATTERN = re.compile(r"\[Accept\]", re.IGNORECASE)
DENY_PATTERN = re.compile(r"\[Deny\]", re.IGNORECASE)
# Pattern: [Offer: I give X Wood, Y Gold for Z Wood, W Gold]
# Or simpler: [Offer: 3 Wood, 2 Gold -> 1 Wood, 5 Gold]
OFFER_PATTERN = re.compile(
    r"\[Offer:\s*(?:I\s+(?:give|offer)\s+)?(.+?)\s*(?:->|for)\s*(.+?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)


def parse_resources(text: str) -> Optional[Dict[str, int]]:
    """
    Parse resource string like "3 Wood, 2 Gold" into dict.

    Handles formats:
    - "3 Wood, 2 Gold"
    - "3 Wood and 2 Gold"
    - "3 Woods, 2 Golds" (plurals)
    """
    resources = {}

    # Split on comma or "and"
    parts = re.split(r',|\s+and\s+', text, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Match "N Resource" pattern
        match = re.match(r'(\d+)\s*(\w+)', part)
        if not match:
            return None

        amount = int(match.group(1))
        resource = match.group(2).strip()

        # Handle plurals
        if resource.endswith('s') and resource[:-1] in RESOURCE_NAMES:
            resource = resource[:-1]

        # Capitalize first letter
        resource = resource.capitalize()

        if resource not in RESOURCE_NAMES:
            return None

        resources[resource] = amount

    # Fill in missing resources with 0
    for res in RESOURCE_NAMES:
        if res not in resources:
            resources[res] = 0

    return resources


def parse_action(text: str) -> Tuple[str, Optional[Dict]]:
    """
    Parse an action from the model's completion.

    Returns:
        (action_type, action_data)
        action_type: "accept", "deny", "offer", or "talk"
        action_data: For offers, {"give": {...}, "receive": {...}}
    """
    # Check for Accept
    if ACCEPT_PATTERN.search(text):
        return ("accept", None)

    # Check for Deny
    if DENY_PATTERN.search(text):
        return ("deny", None)

    # Check for Offer
    offer_match = OFFER_PATTERN.search(text)
    if offer_match:
        give_str = offer_match.group(1)
        receive_str = offer_match.group(2)

        give = parse_resources(give_str)
        receive = parse_resources(receive_str)

        if give is not None and receive is not None:
            return ("offer", {"give": give, "receive": receive})
        else:
            # Invalid offer format
            return ("invalid_offer", None)

    # Default: just talking/negotiating
    return ("talk", None)


# ---------------------------------------------------------------------------
# Inventory Value Calculation
# ---------------------------------------------------------------------------

def calculate_inventory_value(resources: Dict[str, int], values: Dict[str, int]) -> int:
    """Calculate total inventory value: sum(quantity * personal_value)."""
    return sum(qty * values.get(res, 0) for res, qty in resources.items())


# ---------------------------------------------------------------------------
# Rubric: Inventory Value Change
# ---------------------------------------------------------------------------

def negotiation_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Compute rewards based on inventory value change.

    Winner = player with larger change in inventory value
    Win: 1.0, Lose: 0.0, Draw: 0.0
    Invalid action: -1.5 (opponent gets 0.5)
    """
    state_data = rollout.extras

    # Use player_values keys to get all players (not just those who took turns).
    # This ensures consistent reward keys across all rollouts, which is required
    # by credit assigners like RAECredit and GRPOCredit.
    player_values = state_data.get("player_values", {})
    if player_values:
        actors = sorted(player_values.keys())
    else:
        # Fallback to actors from rollout steps
        actors = sorted(rollout.actors)

    if len(actors) < 2:
        return {actor: 0.0 for actor in actors}

    # Check for invalid action
    invalid_player = state_data.get("invalid_action")
    if invalid_player:
        other_player = [a for a in actors if a != invalid_player][0]
        rewards = {invalid_player: REWARD_INVALID, other_player: REWARD_OPPONENT_INVALID}
        if arena.verbose:
            print(f"    [negotiation] Invalid action by {invalid_player}")
        return rewards

    # Calculate value changes
    changes = {}
    for player in actors:
        initial = state_data["initial_values"][player]
        final = calculate_inventory_value(
            state_data["player_resources"][player],
            state_data["player_values"][player]
        )
        changes[player] = final - initial

    # Determine winner based on value change
    sorted_actors = sorted(actors)
    p0, p1 = sorted_actors[0], sorted_actors[1]

    if changes[p0] > changes[p1]:
        rewards = {p0: REWARD_WIN, p1: REWARD_LOSE}
    elif changes[p1] > changes[p0]:
        rewards = {p0: REWARD_LOSE, p1: REWARD_WIN}
    else:
        rewards = {p0: REWARD_DRAW, p1: REWARD_DRAW}

    if arena.verbose:
        print(f"    [negotiation] Value changes: {changes}")
        print(f"    [negotiation] Rewards: {rewards}")

    return rewards


# ---------------------------------------------------------------------------
# Negotiation Episode
# ---------------------------------------------------------------------------

class NegotiationEpisode(MultiTurnEpisode):
    """
    Two-player resource trading game.

    Players have opposite preferences and try to maximize their
    inventory value change through trades.
    """

    def __init__(
        self,
        player_0_actor_id: str = "Player0",
        player_1_actor_id: str = "Player1",
        max_turns: int = 10,
    ):
        super().__init__(max_turns=max_turns)
        self.player_0_actor_id = player_0_actor_id
        self.player_1_actor_id = player_1_actor_id
        self._rubric = Rubric(funcs=[negotiation_reward])

    @property
    def episode_type(self) -> str:
        return "negotiation"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def _get_opponent(self, player: str) -> str:
        """Get the opponent of the given player."""
        if player == self.player_0_actor_id:
            return self.player_1_actor_id
        return self.player_0_actor_id

    def init_state(self, state: EpisodeState, artifact: Any) -> None:
        """Initialize the game state in state.data."""
        state.data["player_resources"] = {
            self.player_0_actor_id: copy.deepcopy(STARTING_RESOURCES),
            self.player_1_actor_id: copy.deepcopy(STARTING_RESOURCES),
        }
        state.data["player_values"] = {
            self.player_0_actor_id: PLAYER_VALUES["Player0"].copy(),
            self.player_1_actor_id: PLAYER_VALUES["Player1"].copy(),
        }

        # Calculate initial inventory values
        state.data["initial_values"] = {}
        for player in [self.player_0_actor_id, self.player_1_actor_id]:
            state.data["initial_values"][player] = calculate_inventory_value(
                state.data["player_resources"][player],
                state.data["player_values"][player]
            )

        state.data["pending_offer"] = None  # {"from": ..., "to": ..., "give": {...}, "receive": {...}}
        state.data["trade_history"] = []
        state.data["invalid_action"] = None

    def _format_resources(self, resources: Dict[str, int]) -> str:
        """Format resources dict as readable string."""
        parts = [f"{qty} {res}" for res, qty in resources.items() if qty > 0]
        return ", ".join(parts) if parts else "nothing"

    def _format_offer(self, offer: Dict) -> str:
        """Format an offer for display."""
        give = self._format_resources(offer["give"])
        receive = self._format_resources(offer["receive"])
        return f"{offer['from']} offers to give {give} for {receive}"

    def _build_initial_state(self, state: EpisodeState) -> str:
        """Build the initial game state announcement (shared, no player-specific values)."""
        p0 = self.player_0_actor_id
        p1 = self.player_1_actor_id
        resources = state.data["player_resources"]

        return f"""=== NEGOTIATION GAME ===
Starting resources - {p0}: {self._format_resources(resources[p0])} | {p1}: {self._format_resources(resources[p1])}

Rules:
- Each player values resources differently (your values are in your system prompt)
- Trade to maximize YOUR inventory value change
- Actions: [Offer: I give X Wood, Y Gold for Z Wood, W Gold], [Accept], [Deny], or just talk
- Game ends after {self.max_turns} turns. Winner = largest inventory value gain."""

    def get_observation(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> str:
        """
        Build player-specific observation for current actor.

        This includes their resources, values, and current inventory value.
        NOT added to transcript - private to this player's turn.
        """
        player = state.current_actor
        opponent = self._get_opponent(player)
        resources = state.data["player_resources"]
        values = state.data["player_values"]
        pending = state.data["pending_offer"]

        my_resources = resources[player]
        my_values = values[player]
        my_value = calculate_inventory_value(my_resources, my_values)
        initial_value = state.data["initial_values"][player]
        value_change = my_value - initial_value

        opp_resources = resources[opponent]

        obs = f"""=== YOUR STATUS (Turn {state.turn + 1}/{self.max_turns}) ===
Your resources: {self._format_resources(my_resources)}
Your values: Wood={my_values['Wood']}, Gold={my_values['Gold']}
Your inventory value: {my_value} (change: {value_change:+d})

Opponent's resources: {self._format_resources(opp_resources)}"""

        # Pending offer info
        if pending:
            if pending["to"] == player:
                give = self._format_resources(pending["give"])
                receive = self._format_resources(pending["receive"])
                obs += f"\n\nPENDING OFFER TO YOU: {pending['from']} offers {give} for {receive}"
                obs += "\nYou can [Accept], [Deny], or make a counter-offer."
            else:
                give = self._format_resources(pending["give"])
                receive = self._format_resources(pending["receive"])
                obs += f"\n\nYour pending offer: Give {give} for {receive}"

        return obs

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        """Randomly select starting player."""
        start_idx = random.randint(0, 1)
        state.data["start_idx"] = start_idx

        return self.player_0_actor_id if start_idx == 0 else self.player_1_actor_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        """Build the initial game state (shared, no player-specific values)."""
        return self._build_initial_state(state)

    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        """Alternate between players."""
        return self._get_opponent(state.current_actor)

    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        """Check if game is over (max turns or invalid action)."""
        if state.data.get("invalid_action"):
            return True
        return super().is_done(state, artifact)

    async def env_response(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> str:
        """
        Process the last action and return the next player's observation.

        This is called after each turn to:
        1. Parse the action from the last completion
        2. Update game state (execute trades, set pending offers, etc.)
        3. Return the observation for the next player
        """
        last_step = state.last_step
        if not last_step:
            return ""

        current_player = last_step.actor_id
        opponent = self._get_opponent(current_player)
        action_text = last_step.completion_text

        action_type, action_data = parse_action(action_text)

        if arena.verbose:
            print(f"    [{current_player}] Action: {action_type}")

        resources = state.data["player_resources"]
        pending = state.data["pending_offer"]

        # Process action
        if action_type == "accept":
            # Can only accept if there's a pending offer TO this player
            if pending and pending["to"] == current_player:
                # Validate resources
                giver = pending["from"]
                receiver = current_player

                # Check if giver has enough to give
                can_trade = True
                for res, qty in pending["give"].items():
                    if resources[giver].get(res, 0) < qty:
                        can_trade = False
                        break

                # Check if receiver has enough to give back
                if can_trade:
                    for res, qty in pending["receive"].items():
                        if resources[receiver].get(res, 0) < qty:
                            can_trade = False
                            break

                if can_trade:
                    # Execute trade
                    for res, qty in pending["give"].items():
                        resources[giver][res] -= qty
                        resources[receiver][res] += qty
                    for res, qty in pending["receive"].items():
                        resources[receiver][res] -= qty
                        resources[giver][res] += qty

                    state.data["trade_history"].append({
                        "type": "trade",
                        "from": giver,
                        "to": receiver,
                        "give": pending["give"],
                        "receive": pending["receive"],
                    })
                    state.data["pending_offer"] = None

                    if arena.verbose:
                        print(f"    [trade] {giver} -> {receiver}: {pending['give']} for {pending['receive']}")
                else:
                    # Invalid: insufficient resources
                    state.data["invalid_action"] = current_player
                    state.done = True
                    if arena.verbose:
                        print(f"    [invalid] Insufficient resources to complete trade")
            else:
                # Invalid: no pending offer to accept
                state.data["invalid_action"] = current_player
                state.done = True
                if arena.verbose:
                    print(f"    [invalid] No pending offer to accept")

        elif action_type == "deny":
            if pending and pending["to"] == current_player:
                state.data["trade_history"].append({
                    "type": "deny",
                    "from": current_player,
                    "offer": pending,
                })
                state.data["pending_offer"] = None
            # Denying when no offer is just a no-op (not invalid)

        elif action_type == "offer":
            # Making an offer cancels any existing pending offer (implicit deny)
            state.data["pending_offer"] = {
                "from": current_player,
                "to": opponent,
                "give": action_data["give"],
                "receive": action_data["receive"],
            }
            state.data["trade_history"].append({
                "type": "offer",
                "from": current_player,
                "to": opponent,
                "give": action_data["give"],
                "receive": action_data["receive"],
            })

        elif action_type == "invalid_offer":
            state.data["invalid_action"] = current_player
            state.done = True
            if arena.verbose:
                print(f"    [invalid] Could not parse offer format")

        # "talk" is always valid - just negotiating

        # Return action result for transcript (shared, no player-specific values)
        return self._format_action_result(current_player, action_type, action_data, state)

    def _format_action_result(
        self,
        actor: str,
        action_type: str,
        action_data: Optional[Dict],
        state: EpisodeState,
    ) -> str:
        """Format action result for the shared transcript."""
        resources = state.data["player_resources"]
        p0 = self.player_0_actor_id
        p1 = self.player_1_actor_id

        # Action-specific result
        if action_type == "accept":
            if state.data.get("invalid_action"):
                result = f"[INVALID: {actor} tried to accept but failed]"
            else:
                result = f"[TRADE EXECUTED]"
                # Show updated resources after trade
                result += f"\nResources now - {p0}: {self._format_resources(resources[p0])} | {p1}: {self._format_resources(resources[p1])}"
                result += "\nNo pending offer. You can make a new [Offer: ...]. Accept or Deny are invalid actions."
            return result
        elif action_type == "deny":
            return f"[{actor} denied the offer]\nNo pending offer. You can make a new [Offer: ...]. Accept or Deny are invalid actions."
        elif action_type == "offer":
            give = self._format_resources(action_data["give"])
            receive = self._format_resources(action_data["receive"])
            return f"[OFFER: {actor} proposes giving {give} for {receive}]\nYou can [Accept], [Deny], or make a counter-offer."
        elif action_type == "invalid_offer":
            return f"[INVALID: {actor} made a malformed offer]"
        else:
            # "talk" - no action result needed
            return ""

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Return game state for rubric and preview."""
        player_resources = state.data.get("player_resources", {})
        player_values = state.data.get("player_values", {})
        initial_values = state.data.get("initial_values", {})
        invalid_action = state.data.get("invalid_action")
        trades = state.data.get("trade_history", [])

        # Compute value changes for each player
        value_changes = {}
        for player in [self.player_0_actor_id, self.player_1_actor_id]:
            if player in player_resources and player in player_values and player in initial_values:
                final_value = calculate_inventory_value(player_resources[player], player_values[player])
                value_changes[player] = final_value - initial_values[player]

        # Determine winner
        if invalid_action:
            winner = self.player_1_actor_id if invalid_action == self.player_0_actor_id else self.player_0_actor_id
        elif value_changes:
            p0_change = value_changes.get(self.player_0_actor_id, 0)
            p1_change = value_changes.get(self.player_1_actor_id, 0)
            if p0_change > p1_change:
                winner = self.player_0_actor_id
            elif p1_change > p0_change:
                winner = self.player_1_actor_id
            else:
                winner = "Draw"
        else:
            winner = "Draw"

        return {
            # For rubric
            "player_resources": player_resources,
            "player_values": player_values,
            "initial_values": initial_values,
            "invalid_action": invalid_action,
            # For preview
            "trades": trades,
            "value_changes": value_changes,
            "winner": winner,
        }


# ---------------------------------------------------------------------------
# Negotiation Arena
# ---------------------------------------------------------------------------

class NegotiationArena(Arena):
    """Arena that schedules negotiation episodes."""

    def __init__(
        self,
        client: InferenceClient,
        episodes_per_step: int = 4,
        verbose: bool = False,
        credit_assigner: CreditAssigner | None = None,
    ):
        # Use RAECredit by default for actor-conditioned baselines
        super().__init__(
            client,
            credit_assigner=credit_assigner or RAECredit(decay=0.95),
            verbose=verbose
        )
        self.episodes_per_step = episodes_per_step

    def get_batch(self) -> List[EpisodeRequest]:
        """Generate episodes_per_step negotiation episodes."""
        return [
            EpisodeRequest(
                episode_type="negotiation",
                artifact={},  # No artifact needed - game is self-contained
            )
            for _ in range(self.episodes_per_step)
        ]
