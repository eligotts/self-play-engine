"""
Task implementations for legos.

These tasks demonstrate how to implement various self-play scenarios:
1. Proposer/Solver: AZR-style problem generation and solving
2. GSM8K: Math reasoning with GRPO-style training
3. Negotiation: SPIRAL-style SimpleNegotiation with RAE credit assignment
4. RefineLoop: Generator-Critic iterative refinement (cooperative)
5. HeadToHead: Tournament competition with LLM-judged matches
6. SPICE: Self-Play In Corpus Environment (corpus-grounded Q&A generation)

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

from .head_to_head import (
    MatchEpisode,
    ChallengeProposerEpisode,
    HeadToHeadArena,
    match_reward,
)

from .spice import (
    SpiceProposerEpisode,
    SpiceSolverEpisode,
    SpiceArena,
    solver_llm_judge_reward,
    proposer_pass_rate_reward,
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
    # HeadToHead (tournament competition)
    "MatchEpisode",
    "ChallengeProposerEpisode",
    "HeadToHeadArena",
    "match_reward",
    # SPICE (corpus-grounded Q&A)
    "SpiceProposerEpisode",
    "SpiceSolverEpisode",
    "SpiceArena",
    "solver_llm_judge_reward",
    "proposer_pass_rate_reward",
]
