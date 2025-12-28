"""
Rubric: Composable reward functions for scoring rollouts.

A Rubric combines multiple reward functions. Each function:
- Takes (rollout: Rollout, arena: Arena)
- Returns Dict[str, float] mapping role_id -> reward
- Can be sync or async

Example:
    def my_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
        return {"Player": 1.0 if won else 0.0}

    rubric = Rubric(funcs=[my_reward, other_reward])
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional, Union

from .types import Rollout

if TYPE_CHECKING:
    from .arena import Arena

# A reward function: (Rollout, Arena) -> Dict[str, float]
RewardFn = Callable[
    [Rollout, "Arena"],
    Union[Dict[str, float], Awaitable[Dict[str, float]]]
]


class Rubric:
    """
    Composes multiple reward functions into a single scoring unit.

    Each function is called with (rollout, arena) and returns
    Dict[str, float] mapping role_id -> reward.

    Results are aggregated with optional weights.
    """

    def __init__(
        self,
        funcs: List[RewardFn],
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            funcs: List of reward functions to compose
            weights: Optional weights for each function (default: all 1.0)

        Raises:
            ValueError: If weights length doesn't match funcs length
        """
        self.funcs = funcs
        self.weights = weights or [1.0] * len(funcs)

        if len(self.weights) != len(self.funcs):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"funcs length ({len(self.funcs)})"
            )

    async def score(self, rollout: Rollout, arena: "Arena") -> None:
        """
        Score a rollout, mutating it in place.

        1. Calls each reward function
        2. Aggregates results (weighted sum per role) into rollout.rewards
        3. Sets step.reward for each step based on its role_id

        Raises:
            ValueError: If any role in rollout.steps has no reward
        """
        # Reset
        for step in rollout.steps:
            step.reward = 0.0
            step.advantage = 0.0
        rollout.rewards = {}
        rollout.advantages = {}
        rollout.metrics = {}

        # Call each function and aggregate
        for func, weight in zip(self.funcs, self.weights):
            result = func(rollout, arena)
            if inspect.iscoroutine(result):
                result = await result

            for role_id, reward in result.items():
                rollout.rewards[role_id] = rollout.rewards.get(role_id, 0.0) + reward * weight

            # Store metric
            func_name = getattr(func, "__name__", str(func))
            if result:
                rollout.metrics[func_name] = next(iter(result.values()))

        # Validate all roles are scored
        step_roles = {step.role_id for step in rollout.steps}
        for role_id in step_roles:
            if role_id not in rollout.rewards:
                raise ValueError(
                    f"No reward function scored role '{role_id}'. "
                    f"Roles in steps: {step_roles}, "
                    f"roles scored: {set(rollout.rewards.keys())}"
                )

        # Propagate to steps
        for step in rollout.steps:
            step.reward = rollout.rewards[step.role_id]
