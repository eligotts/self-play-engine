"""
Task implementations for the self-play engine.

These tasks demonstrate how to implement various self-play scenarios:
1. Proposer/Solver: AZR-style problem generation and solving
2. GSM8K: Math reasoning with GRPO-style training
3. Negotiation: SPIRAL-style SimpleNegotiation with RAE credit assignment
4. RefineLoop: Generator-Critic iterative refinement (cooperative)
5. EloArena: Tournament competition with persistent Elo ratings

See examples/train_*.py scripts for complete usage.
"""
from .proposer_solver import (
    ProposerEpisode,
    SolveEpisode,
    ProposerSolverArena,
)

from .gsm8k import (
    GSM8KEpisode,
    GSM8KArena,
    gsm8k_reward,
    load_gsm8k,
    extract_xml_answer,
    extract_hash_answer,
    SYSTEM_PROMPT as GSM8K_SYSTEM_PROMPT,
)

from .negotiation import (
    NegotiationEpisode,
    NegotiationArena,
    negotiation_reward,
)

from .refine_loop import (
    RefineLoopEpisode,
    RefineLoopArena,
    refine_loop_reward,
)

from .elo_arena import (
    EloMatchEpisode,
    EloArena,
    elo_match_reward,
)

__all__ = [
    # Proposer/Solver
    "ProposerEpisode",
    "SolveEpisode",
    "ProposerSolverArena",
    # GSM8K
    "GSM8KEpisode",
    "GSM8KArena",
    "gsm8k_reward",
    "load_gsm8k",
    "extract_xml_answer",
    "extract_hash_answer",
    "GSM8K_SYSTEM_PROMPT",
    # Negotiation (SPIRAL SimpleNegotiation)
    "NegotiationEpisode",
    "NegotiationArena",
    "negotiation_reward",
    # RefineLoop (generator-critic refinement)
    "RefineLoopEpisode",
    "RefineLoopArena",
    "refine_loop_reward",
    # EloArena (tournament competition)
    "EloMatchEpisode",
    "EloArena",
    "elo_match_reward",
]
