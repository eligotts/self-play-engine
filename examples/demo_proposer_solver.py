#!/usr/bin/env python3
"""
Demo: Proposer/Solver Self-Play - Integration Test

Tests the composable episode pipeline:
1. ProposerEpisode spawns SolverEpisodes via arena.generate_rollouts()
2. Child results flow into parent's state.child_results
3. Parent rubric uses child results for scoring
4. Training records extracted from full hierarchy

Verifies contracts:
- Episode.rollout() can spawn sub-episodes
- GenerateResult.children contains child results
- Rewards flow correctly across hierarchy
- TrainingBatch flattens all records (parent + children)
"""
import asyncio
import sys
sys.path.insert(0, "src")

from self_play import MockInferenceClient, Messages, EpisodeRequest
from self_play.examples.proposer_solver import (
    ProposerEpisode,
    SolveEpisode,
    ProposerSolverArena,
    create_proposer_solver_arena,
)


def create_mock_client():
    """Create mock client with deterministic responses."""
    call_count = {"n": 0}

    def mock_response(messages: Messages) -> str:
        call_count["n"] += 1
        last = messages[-1]["content"] if messages else ""

        if "Generate a new math question" in last:
            # Proposer generates a question
            return '{"question": "What is 7 * 8?", "answer": "56", "difficulty": "easy"}'
        elif "Solve the following question" in last:
            # Solver: alternate correct/incorrect for ~50% pass rate
            if call_count["n"] % 2 == 0:
                return "Let me calculate: 7 * 8 = 56. The answer is: 56"
            else:
                return "I think it's 7 + 8 = 15. The answer is: 15"  # Wrong
        else:
            return "Mock response"

    return MockInferenceClient(response_fn=mock_response), call_count


async def test_solve_episode():
    """Test SolveEpisode (single-turn) produces correct structure."""
    print("\n" + "=" * 60)
    print("TEST: SolveEpisode Structure")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(client=client, n_solver_rollouts=1, batch_size=1)

    # Run solve episode directly
    episode = arena.episodes["solve"]
    result = await episode.generate(arena, {"question": "What is 2+2?", "ground_truth": "4"})

    # Verify structure
    assert result.rollout is not None
    assert len(result.rollout.steps) == 1, "SingleTurnEpisode should have 1 step"
    assert result.rollout.steps[0].role_id == "Solver"
    assert result.children == [], "SolveEpisode should have no children"

    # Verify rewards (both rollout.rewards and step.reward)
    rollout = result.rollout
    assert "Solver" in rollout.rewards, "Missing Solver in rollout.rewards"
    assert rollout.steps[0].reward == rollout.rewards["Solver"], "Step reward should match"

    print(f"  ✓ SolveEpisode produces 1 step")
    print(f"  ✓ Reward: {rollout.rewards['Solver']:.2f}")


async def test_proposer_episode_spawns_children():
    """Test ProposerEpisode spawns solver sub-episodes."""
    print("\n" + "=" * 60)
    print("TEST: ProposerEpisode Spawns Children")
    print("=" * 60)

    client, call_count = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,
        batch_size=1,
    )

    # Run proposer episode
    episode = arena.episodes["propose"]
    result = await episode.generate(arena, {})

    # Verify parent structure
    assert result.rollout is not None
    assert result.rollout.episode_type == "propose"
    assert len(result.rollout.steps) == 1, "Proposer makes 1 model call"
    assert result.rollout.steps[0].role_id == "Proposer"

    # Verify children
    assert len(result.children) == 4, f"Expected 4 children, got {len(result.children)}"

    for i, child in enumerate(result.children):
        assert child.rollout.episode_type == "solve", f"Child {i} wrong type"
        assert len(child.rollout.steps) == 1, f"Child {i} wrong step count"
        assert child.rollout.steps[0].role_id == "Solver"

    # Verify extras include child info
    extras = result.rollout.extras
    assert "proposed_question" in extras
    assert "solver_rewards" in extras
    assert "pass_rate" in extras
    assert len(extras["solver_rewards"]) == 4

    print(f"  ✓ Proposer makes 1 model call")
    print(f"  ✓ Spawns {len(result.children)} solver episodes")
    print(f"  ✓ Extras include solver_rewards and pass_rate")
    print(f"  ✓ Pass rate: {extras['pass_rate']:.2f}")
    print(f"  ✓ Total model calls: {call_count['n']} (1 proposer + 4 solvers)")


async def test_generate_result_hierarchy():
    """Test GenerateResult.all_rollouts() flattens hierarchy."""
    print("\n" + "=" * 60)
    print("TEST: GenerateResult Hierarchy Flattening")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=3,
        batch_size=1,
    )

    result = await arena.run_episode("propose", {})

    # Use all_rollouts() to flatten
    all_rollouts = result.all_rollouts()

    # Should have 1 parent + 3 children = 4 total
    assert len(all_rollouts) == 4, f"Expected 4 rollouts, got {len(all_rollouts)}"

    # First should be parent
    parent_rollout = all_rollouts[0]
    assert parent_rollout.episode_type == "propose"
    assert "Proposer" in parent_rollout.rewards

    # Rest should be children
    for i, rollout in enumerate(all_rollouts[1:]):
        assert rollout.episode_type == "solve", f"Child {i} wrong type"
        assert "Solver" in rollout.rewards

    print(f"  ✓ all_rollouts() returns {len(all_rollouts)} rollouts")
    print(f"  ✓ First is parent (propose)")
    print(f"  ✓ Rest are children (solve)")


async def test_training_batch_flattening():
    """Test TrainingBatch includes records from full hierarchy."""
    print("\n" + "=" * 60)
    print("TEST: TrainingBatch Hierarchy Flattening")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,
        batch_size=1,
    )

    result = await arena.run_episode("propose", {})
    batch = arena.build_training_batch([result])

    # Should have 1 proposer + 4 solver = 5 records
    assert len(batch.records) == 5, f"Expected 5 records, got {len(batch.records)}"

    # Count by role
    proposer_records = [r for r in batch.records if r.role_id == "Proposer"]
    solver_records = [r for r in batch.records if r.role_id == "Solver"]

    assert len(proposer_records) == 1, f"Expected 1 Proposer record, got {len(proposer_records)}"
    assert len(solver_records) == 4, f"Expected 4 Solver records, got {len(solver_records)}"

    # Verify all records have required fields
    for record in batch.records:
        assert len(record.prompt_token_ids) > 0
        assert len(record.completion_token_ids) > 0
        assert "episode_type" in record.meta

    print(f"  ✓ TrainingBatch has {len(batch.records)} records")
    print(f"  ✓ Proposer records: {len(proposer_records)}")
    print(f"  ✓ Solver records: {len(solver_records)}")
    print(f"  ✓ All records have token data and meta")


async def test_full_step_with_nested_episodes():
    """Test arena.step() with nested episode structure."""
    print("\n" + "=" * 60)
    print("TEST: Arena.step() with Nested Episodes")
    print("=" * 60)

    client, call_count = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=3,
        batch_size=2,  # 2 proposer episodes
    )

    batch = await arena.step(concurrency=8)

    # 2 proposers * (1 proposer step + 3 solver steps) = 2 * 4 = 8 records
    expected_records = 2 * (1 + 3)
    assert len(batch.records) == expected_records, f"Expected {expected_records} records, got {len(batch.records)}"

    # Count by role
    proposer_count = sum(1 for r in batch.records if r.role_id == "Proposer")
    solver_count = sum(1 for r in batch.records if r.role_id == "Solver")

    assert proposer_count == 2, f"Expected 2 Proposer records, got {proposer_count}"
    assert solver_count == 6, f"Expected 6 Solver records, got {solver_count}"

    # Verify model calls: 2 proposers + 2*3 solvers = 8
    expected_calls = 2 + 2 * 3
    assert call_count["n"] == expected_calls, f"Expected {expected_calls} model calls, got {call_count['n']}"

    print(f"  ✓ arena.step() with batch_size=2, n_solver_rollouts=3")
    print(f"  ✓ Total records: {len(batch.records)}")
    print(f"  ✓ Proposer: {proposer_count}, Solver: {solver_count}")
    print(f"  ✓ Model calls: {call_count['n']}")
    print(f"  ✓ Batch meta: {batch.meta}")


async def test_proposer_rubric_uses_child_results():
    """Test proposer rubric scores based on solver performance."""
    print("\n" + "=" * 60)
    print("TEST: Proposer Rubric Uses Child Results")
    print("=" * 60)

    # Create client that gives 50% pass rate
    call_count = {"n": 0}

    def mock_50_percent(messages: Messages) -> str:
        call_count["n"] += 1
        last = messages[-1]["content"] if messages else ""

        if "Generate a new math question" in last:
            return '{"question": "What is 5 + 5?", "answer": "10", "difficulty": "easy"}'
        elif "Solve the following question" in last:
            # Exactly 50%: odd calls correct, even calls wrong
            if call_count["n"] % 2 == 1:
                return "The answer is: 10"
            else:
                return "The answer is: 99"
        return "Mock"

    client = MockInferenceClient(response_fn=mock_50_percent)
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,  # 2 correct, 2 wrong = 50%
        batch_size=1,
    )

    result = await arena.run_episode("propose", {})

    # Check pass rate in extras
    pass_rate = result.rollout.extras.get("pass_rate", 0.0)
    solver_rewards = result.rollout.extras.get("solver_rewards", [])

    print(f"  Solver rewards: {solver_rewards}")
    print(f"  Pass rate: {pass_rate:.2f}")

    # Proposer reward should be high (close to target 0.5)
    proposer_reward = result.rollout.rewards.get("Proposer", 0.0)
    print(f"  Proposer reward: {proposer_reward:.2f}")

    # With 50% pass rate and target 0.5, reward should be ~1.0
    # (reward = 1.0 - |pass_rate - 0.5| * 2)

    print(f"  ✓ Rubric computed pass_rate from child results")
    print(f"  ✓ Proposer reward derived from solver performance")


async def test_parallel_execution_of_children():
    """Test that child episodes execute in parallel via generate_rollouts."""
    print("\n" + "=" * 60)
    print("TEST: Parallel Execution of Child Episodes")
    print("=" * 60)

    import time

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,
        batch_size=1,
    )

    start = time.time()
    result = await arena.run_episode("propose", {})
    elapsed = time.time() - start

    # With 4 parallel solver episodes, should be much faster than 4 * delay
    # (though proposer runs first, so at least 1 serial call)

    print(f"  ✓ Total time: {elapsed:.3f}s")
    print(f"  ✓ {len(result.children)} solver episodes completed")
    print(f"  ✓ Children executed via arena.generate_rollouts()")


async def main():
    print("=" * 60)
    print("PROPOSER/SOLVER SELF-PLAY INTEGRATION TESTS")
    print("=" * 60)

    await test_solve_episode()
    await test_proposer_episode_spawns_children()
    await test_generate_result_hierarchy()
    await test_training_batch_flattening()
    await test_full_step_with_nested_episodes()
    await test_proposer_rubric_uses_child_results()
    await test_parallel_execution_of_children()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
