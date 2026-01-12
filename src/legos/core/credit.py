"""
Credit Assignment for self-play training.

Credit assignment transforms raw rollouts with rewards into per-step weights/advantages.
The key abstraction is CreditAssigner - a protocol that takes hierarchical GenerateResults
and produces a mapping of (rollout_id, step_index) -> weight.

Grouping by Hierarchy:
----------------------
In hierarchical episodes (e.g., Proposer spawns Solvers), rollouts at the same level
of the hierarchy form a group for advantage computation:

- Top-level: All GenerateResults in the batch form one group
- Children: Within each parent, its children form independent groups
- This recurses down the tree

Grouping by Episode Type:
-------------------------
Within each hierarchy level, rollouts are further grouped by their episode_type.
This ensures that different task types (e.g., "gsm8k" vs "negotiation") are not
compared against each other when computing relative advantages.

Per-Actor Advantage Computation:
-------------------------------
Each rollout can have multiple actors (stored in rollout.rewards as a dict mapping
actor_id -> reward). Advantages are computed independently for each actor across the
group. Each step in a rollout is assigned the advantage corresponding to its
step.actor_id, enabling multi-agent scenarios where different actors within the same
episode receive actor-appropriate credit.

For GRPO, advantage = reward - mean(group_rewards), so rollouts are compared
against peers at their hierarchy level.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .types import GenerateResult, Rollout


# Type for addressing a specific step in a rollout
RolloutStepKey = Tuple[str, int]  # (rollout_id, step_index)


class CreditAssigner(ABC):
    """
    Protocol for credit assignment.

    Takes hierarchical GenerateResults and produces per-step weights.
    All steps in all rollouts MUST be covered by the returned dict.
    """

    @abstractmethod
    def compute(
        self,
        results: List[GenerateResult],
    ) -> Dict[RolloutStepKey, float]:
        """
        Compute credit weights for all steps in all rollouts.

        Args:
            results: List of GenerateResults (tree structure with children)

        Returns:
            Mapping of (rollout_id, step_index) -> weight/advantage
        """
        ...


def apply_credit(
    results: List[GenerateResult],
    weights: Dict[RolloutStepKey, float],
) -> None:
    """
    Apply computed weights to step.advantage fields in-place.

    Args:
        results: List of GenerateResults to modify
        weights: Mapping from (rollout_id, step_index) -> advantage
    """
    def apply_to_result(result: GenerateResult) -> None:
        # Skip non-trainable results and their entire subtrees
        if not result.is_trainable:
            return

        rollout = result.rollout
        for i, step in enumerate(rollout.steps):
            key = (rollout.id, i)
            if key in weights:
                step.advantage = weights[key]

        for child in result.children:
            apply_to_result(child)

    for result in results:
        apply_to_result(result)


# ---------------------------------------------------------------------------
# GRPO Credit Assignment
# ---------------------------------------------------------------------------

def _get_actor_keys(rollouts: List[Rollout]) -> set:
    """
    Get the set of actor keys from rollouts, validating consistency.

    All rollouts must have the same set of actor keys in their rewards dict.
    Raises ValueError if rollouts have inconsistent actor keys.
    """
    if not rollouts:
        return set()

    # Get actor keys from first rollout
    first_keys = set(rollouts[0].rewards.keys())

    # Validate all rollouts have same keys
    for i, rollout in enumerate(rollouts[1:], 1):
        keys = set(rollout.rewards.keys())
        if keys != first_keys:
            raise ValueError(
                f"Inconsistent actor keys at hierarchy level. "
                f"Rollout 0 has {first_keys}, rollout {i} has {keys}"
            )

    return first_keys


def _compute_grpo_advantages(
    rewards: List[float],
    normalize: bool,
    positive_only: bool,
) -> List[float]:
    """Compute GRPO advantages for a list of rewards."""
    if not rewards:
        return []

    # Compute advantages: reward - mean
    mean_reward = sum(rewards) / len(rewards)
    advantages = [r - mean_reward for r in rewards]

    # Optional: normalize to unit std
    if normalize and len(advantages) > 1:
        std = (sum(a * a for a in advantages) / len(advantages)) ** 0.5
        if std > 1e-8:
            advantages = [a / std for a in advantages]

    # Optional: clamp negatives
    if positive_only:
        advantages = [max(0.0, a) for a in advantages]

    return advantages


@dataclass
class GRPOCredit(CreditAssigner):
    """
    Group Relative Policy Optimization credit assignment.

    Computes advantages as: A_i = R_i - mean(R_1, ..., R_N)
    where the group consists of rollouts at the same hierarchy level.

    Advantages are computed per-actor: if rollouts have multiple actors in their
    rewards dict, each actor's advantage is computed independently across the group.
    Each step receives the advantage for its actor_id.

    Args:
        normalize: If True, normalize advantages to zero mean, unit std within group
        positive_only: If True, clamp negative advantages to 0 (REINFORCE-style)
    """
    normalize: bool = False
    positive_only: bool = False

    def compute(
        self,
        results: List[GenerateResult],
    ) -> Dict[RolloutStepKey, float]:
        """
        Compute GRPO advantages for all rollouts in the hierarchy.

        Groups are formed by hierarchy level:
        - Top-level results form one group
        - Each parent's children form independent groups
        """
        weights: Dict[RolloutStepKey, float] = {}
        self._compute_level(results, weights)
        return weights

    def _compute_level(
        self,
        results: List[GenerateResult],
        weights: Dict[RolloutStepKey, float],
    ) -> None:
        """Process one level of the hierarchy."""
        if not results:
            return

        # Filter to only trainable results - non-trainable results and their
        # entire subtrees are excluded from credit assignment
        trainable_results = [r for r in results if r.is_trainable]
        if not trainable_results:
            return

        # Group results by episode type for per-type advantage computation.
        groups: Dict[str, List[GenerateResult]] = {}
        for result in trainable_results:
            groups.setdefault(result.rollout.episode_type, []).append(result)

        for group_results in groups.values():
            rollouts = [r.rollout for r in group_results]

            # Get actor keys (validates consistency across rollouts)
            actor_keys = _get_actor_keys(rollouts)

            if not actor_keys:
                # No rewards set - assign 0 to all steps
                for result in group_results:
                    rollout = result.rollout
                    for i, _ in enumerate(rollout.steps):
                        weights[(rollout.id, i)] = 0.0
            else:
                # Compute advantages per actor
                actor_advantages: Dict[str, List[float]] = {}
                for actor_id in actor_keys:
                    rewards = [r.rewards.get(actor_id, 0.0) for r in rollouts]
                    actor_advantages[actor_id] = _compute_grpo_advantages(
                        rewards, self.normalize, self.positive_only
                    )

                # Assign advantage to each step based on step's actor_id
                for idx, result in enumerate(group_results):
                    rollout = result.rollout
                    for i, step in enumerate(rollout.steps):
                        # Get advantage for this step's actor
                        if step.actor_id in actor_advantages:
                            adv = actor_advantages[step.actor_id][idx]
                        else:
                            # Step's actor not in rewards dict - use 0
                            adv = 0.0
                        weights[(rollout.id, i)] = adv

        # Recurse into children (each parent's children form independent groups)
        # Only recurse into trainable results - non-trainable subtrees are skipped
        for result in trainable_results:
            if result.children:
                self._compute_level(result.children, weights)


# ---------------------------------------------------------------------------
# Additional Credit Assigners
# ---------------------------------------------------------------------------

@dataclass
class ConstantCredit(CreditAssigner):
    """
    Constant credit for all steps (useful for SFT/behavioral cloning).

    All steps receive the same weight regardless of reward.
    """
    value: float = 1.0

    def compute(
        self,
        results: List[GenerateResult],
    ) -> Dict[RolloutStepKey, float]:
        weights: Dict[RolloutStepKey, float] = {}
        self._assign_all(results, weights)
        return weights

    def _assign_all(
        self,
        results: List[GenerateResult],
        weights: Dict[RolloutStepKey, float],
    ) -> None:
        for result in results:
            # Skip non-trainable results and their entire subtrees
            if not result.is_trainable:
                continue

            rollout = result.rollout
            for i, _ in enumerate(rollout.steps):
                weights[(rollout.id, i)] = self.value

            self._assign_all(result.children, weights)


@dataclass
class EpisodicRewardCredit(CreditAssigner):
    """
    Episode-level reward as credit (no relative advantage computation).

    Each step gets the reward for its actor from the rollout's rewards dict.
    Useful for simpler REINFORCE without group normalization.
    """

    def compute(
        self,
        results: List[GenerateResult],
    ) -> Dict[RolloutStepKey, float]:
        weights: Dict[RolloutStepKey, float] = {}
        self._assign_all(results, weights)
        return weights

    def _assign_all(
        self,
        results: List[GenerateResult],
        weights: Dict[RolloutStepKey, float],
    ) -> None:
        for result in results:
            # Skip non-trainable results and their entire subtrees
            if not result.is_trainable:
                continue

            rollout = result.rollout

            for i, step in enumerate(rollout.steps):
                # Get reward for this step's actor
                reward = rollout.rewards.get(step.actor_id, 0.0)
                weights[(rollout.id, i)] = reward

            self._assign_all(result.children, weights)


# ---------------------------------------------------------------------------
# Actor-conditioned Advantage Estimation (RAE) from SPIRAL
# ---------------------------------------------------------------------------

@dataclass
class EMA:
    """Exponential Moving Average tracker."""
    decay: float = 0.95
    value: float = 0.0

    def update(self, x: float) -> float:
        """Update EMA with new value and return updated average."""
        self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value

    def get(self) -> float:
        """Get current EMA value."""
        return self.value


@dataclass
class RAECredit(CreditAssigner):
    """
    Actor-conditioned Advantage Estimation from the SPIRAL paper.

    Maintains per-(episode_type, actor_id) EMA baselines across the training run.
    Advantage = reward - baseline, where baseline is computed BEFORE updating
    with the new batch's rewards (unbiased estimation).

    This addresses variance in self-play with shared policies where different
    actors may have structural advantages (e.g., first-move advantage).

    Args:
        decay: EMA decay factor. Higher values = slower baseline adaptation.
               0.95 means ~95% weight on history, ~5% on new batch.
    """
    decay: float = 0.95

    # State: baselines[episode_type][actor_id] = EMA
    # Initialized lazily on first observation of each (episode_type, actor_id)
    _baselines: Dict[str, Dict[str, EMA]] = field(default_factory=dict)

    def compute(
        self,
        results: List[GenerateResult],
    ) -> Dict[RolloutStepKey, float]:
        """
        Compute RAE advantages for all rollouts in the hierarchy.

        For each (episode_type, actor_id):
        1. Get baseline from previous batches
        2. Compute advantages = reward - baseline for each rollout
        3. Update baseline with batch-mean reward
        """
        weights: Dict[RolloutStepKey, float] = {}
        self._compute_level(results, weights)
        return weights

    def _compute_level(
        self,
        results: List[GenerateResult],
        weights: Dict[RolloutStepKey, float],
    ) -> None:
        """Process one level of the hierarchy."""
        if not results:
            return

        # Filter to only trainable results
        trainable_results = [r for r in results if r.is_trainable]
        if not trainable_results:
            return

        # Group results by episode type
        groups: Dict[str, List[GenerateResult]] = {}
        for result in trainable_results:
            groups.setdefault(result.rollout.episode_type, []).append(result)

        for episode_type, group_results in groups.items():
            # Ensure we have a baseline dict for this episode_type
            if episode_type not in self._baselines:
                self._baselines[episode_type] = {}

            rollouts = [r.rollout for r in group_results]

            # Get actor keys (validates consistency across rollouts)
            actor_keys = _get_actor_keys(rollouts)

            if not actor_keys:
                # No rewards set - assign 0 to all steps
                for result in group_results:
                    rollout = result.rollout
                    for i, _ in enumerate(rollout.steps):
                        weights[(rollout.id, i)] = 0.0
                continue

            # Compute mean reward per actor across this batch (for batch-wise EMA update)
            actor_mean_rewards: Dict[str, float] = {}
            for actor_id in actor_keys:
                rewards = [r.rewards.get(actor_id, 0.0) for r in rollouts]
                actor_mean_rewards[actor_id] = sum(rewards) / len(rewards)

            # Compute advantages per actor
            actor_advantages: Dict[str, List[float]] = {}
            for actor_id in actor_keys:
                # Ensure baseline exists for this actor
                if actor_id not in self._baselines[episode_type]:
                    self._baselines[episode_type][actor_id] = EMA(decay=self.decay)

                # Get baseline BEFORE updating (unbiased advantage estimation)
                baseline = self._baselines[episode_type][actor_id].get()

                # Compute advantage for each rollout
                advantages = []
                for rollout in rollouts:
                    reward = rollout.rewards.get(actor_id, 0.0)
                    advantages.append(reward - baseline)
                actor_advantages[actor_id] = advantages

                # Update baseline with batch mean
                self._baselines[episode_type][actor_id].update(actor_mean_rewards[actor_id])

            # Assign advantage to each step based on step's actor_id
            for idx, result in enumerate(group_results):
                rollout = result.rollout
                for i, step in enumerate(rollout.steps):
                    if step.actor_id in actor_advantages:
                        adv = actor_advantages[step.actor_id][idx]
                    else:
                        # Step's actor not in rewards dict - use 0
                        adv = 0.0
                    weights[(rollout.id, i)] = adv

        # Recurse into children (each parent's children form independent groups)
        for result in trainable_results:
            if result.children:
                self._compute_level(result.children, weights)
