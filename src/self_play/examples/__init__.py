"""
Example implementations demonstrating the self-play abstractions.

These examples show how to implement:
1. Debate: Multi-turn, zero-sum competitive dialogue
2. Proposer/Solver: AZR-style problem generation and solving
"""
from .debate import (
    DebateEpisode,
    DebateArena,
    create_debate_arena,
    make_debate_rubric,
)

from .proposer_solver import (
    ProposerEpisode,
    SolveEpisode,
    ProposerSolverArena,
    create_proposer_solver_arena,
    make_solver_rubric,
    make_proposer_rubric,
)

__all__ = [
    # Debate
    "DebateEpisode",
    "DebateArena",
    "create_debate_arena",
    "make_debate_rubric",
    # Proposer/Solver
    "ProposerEpisode",
    "SolveEpisode",
    "ProposerSolverArena",
    "create_proposer_solver_arena",
    "make_solver_rubric",
    "make_proposer_rubric",
]
