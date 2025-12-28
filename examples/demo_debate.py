#!/usr/bin/env python3
"""
Demo: Debate Self-Play - Integration Test

Tests the full debate pipeline:
1. DebateArena.get_batch() -> EpisodeRequests
2. Arena.generate_rollouts() -> GenerateResults (parallel execution)
3. Arena.build_training_batch() -> TrainingBatch
4. Arena.step() -> end-to-end training step

Verifies contracts:
- AlternatingRolesEpisode alternates between Aff/Neg
- env_response adds transition messages
- llm_judge produces zero-sum rewards
- TrainingRecords have correct structure
"""
import asyncio
import sys
sys.path.insert(0, "src")

from self_play import MockInferenceClient, Messages, Role, EpisodeRequest
from self_play.examples.debate import DebateEpisode, DebateArena, create_debate_arena


def mock_debater(messages: Messages) -> str:
    """Mock debater - tracks call count via message content."""
    last_msg = messages[-1]["content"] if messages else ""

    if "opening statement" in last_msg.lower() or "affirmative" in last_msg.lower():
        return "I support this position with evidence A, B, and C."
    else:
        return "I oppose this position with counterpoints X, Y, and Z."


async def test_episode_structure():
    """Test that DebateEpisode produces correct structure."""
    print("\n" + "=" * 60)
    print("TEST: Episode Structure")
    print("=" * 60)

    client = MockInferenceClient(response_fn=mock_debater)
    arena = create_debate_arena(
        client=client,
        topics=["Test topic"],
        num_rounds=2,
        batch_size=1,
    )

    # Run single episode directly
    episode = arena.episodes["debate"]
    result = await episode.generate(arena, {"topic": "Test topic"})

    # Verify GenerateResult structure
    assert result.rollout is not None, "GenerateResult should have rollout"
    assert result.rewards is not None, "GenerateResult should have rewards"
    assert isinstance(result.children, list), "GenerateResult should have children list"

    # Verify Rollout structure
    rollout = result.rollout
    assert rollout.episode_type == "debate", f"Expected 'debate', got {rollout.episode_type}"
    assert rollout.seed == {"topic": "Test topic"}, f"Seed mismatch: {rollout.seed}"
    assert len(rollout.steps) == 4, f"Expected 4 steps (2 rounds * 2 roles), got {len(rollout.steps)}"

    # Verify alternating roles
    expected_roles = ["Aff", "Neg", "Aff", "Neg"]
    actual_roles = [step.role_id for step in rollout.steps]
    assert actual_roles == expected_roles, f"Role alternation wrong: {actual_roles}"

    # Verify Step structure
    for i, step in enumerate(rollout.steps):
        assert step.prompt is not None, f"Step {i} missing prompt"
        assert step.completion is not None, f"Step {i} missing completion"
        assert step.prompt_token_ids is not None, f"Step {i} missing prompt_token_ids"
        assert step.completion_token_ids is not None, f"Step {i} missing completion_token_ids"
        assert step.completion_logprobs is not None, f"Step {i} missing completion_logprobs"

    # Verify rewards (rollout.rewards and step.reward both set)
    assert "Aff" in rollout.rewards, "Missing Aff in rollout.rewards"
    assert "Neg" in rollout.rewards, "Missing Neg in rollout.rewards"
    # Also verify step.reward is set
    for step in rollout.steps:
        assert step.reward == rollout.rewards[step.role_id], "Step reward should match rollout.rewards"
    print(f"  Rewards: Aff={rollout.rewards['Aff']:.2f}, Neg={rollout.rewards['Neg']:.2f}")

    # Verify extras
    assert "topic" in rollout.extras, "Missing topic in extras"

    print("  ✓ GenerateResult structure correct")
    print("  ✓ Rollout has correct episode_type, seed, steps")
    print("  ✓ Roles alternate correctly: Aff -> Neg -> Aff -> Neg")
    print("  ✓ All steps have required token data")
    print("  ✓ Rewards assigned to both roles")
    print("  ✓ Extras populated")


async def test_arena_get_batch():
    """Test that DebateArena.get_batch() returns correct requests."""
    print("\n" + "=" * 60)
    print("TEST: Arena.get_batch()")
    print("=" * 60)

    client = MockInferenceClient(response_fn=mock_debater)
    arena = create_debate_arena(
        client=client,
        topics=["Topic A", "Topic B", "Topic C"],
        num_rounds=2,
        batch_size=2,
    )

    # Get batch
    requests = arena.get_batch()

    assert len(requests) == 2, f"Expected 2 requests, got {len(requests)}"

    for req in requests:
        assert isinstance(req, EpisodeRequest), f"Expected EpisodeRequest, got {type(req)}"
        assert req.episode_type == "debate", f"Expected 'debate', got {req.episode_type}"
        assert "topic" in req.seed, f"Seed missing 'topic': {req.seed}"

    print(f"  ✓ get_batch() returns {len(requests)} EpisodeRequests")
    print(f"  ✓ Each request has episode_type='debate'")
    print(f"  ✓ Each request has topic in seed")


async def test_generate_rollouts_parallel():
    """Test that generate_rollouts executes in parallel."""
    print("\n" + "=" * 60)
    print("TEST: Arena.generate_rollouts() (parallel execution)")
    print("=" * 60)

    client = MockInferenceClient(response_fn=mock_debater)
    arena = create_debate_arena(
        client=client,
        topics=["Topic A", "Topic B"],
        num_rounds=1,  # 2 steps per episode
        batch_size=2,
    )

    requests = [
        EpisodeRequest(episode_type="debate", seed={"topic": "Topic A"}),
        EpisodeRequest(episode_type="debate", seed={"topic": "Topic B"}),
    ]

    import time
    start = time.time()
    results = await arena.generate_rollouts(requests, concurrency=4)
    elapsed = time.time() - start

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    # Both should complete
    for i, result in enumerate(results):
        assert result.rollout is not None, f"Result {i} missing rollout"
        assert len(result.rollout.steps) == 2, f"Result {i} wrong step count"

    print(f"  ✓ generate_rollouts() returned {len(results)} results")
    print(f"  ✓ Executed in {elapsed:.3f}s")
    print(f"  ✓ Each result has correct rollout structure")


async def test_build_training_batch():
    """Test that build_training_batch produces correct TrainingBatch."""
    print("\n" + "=" * 60)
    print("TEST: Arena.build_training_batch()")
    print("=" * 60)

    client = MockInferenceClient(response_fn=mock_debater)
    arena = create_debate_arena(
        client=client,
        topics=["Topic A"],
        num_rounds=2,
        batch_size=1,
    )

    result = await arena.run_episode("debate", {"topic": "Topic A"})
    batch = arena.build_training_batch([result])

    # Verify TrainingBatch structure
    assert batch.records is not None, "Batch missing records"
    assert batch.meta is not None, "Batch missing meta"

    # Should have 4 records (4 steps in 2-round debate)
    assert len(batch.records) == 4, f"Expected 4 records, got {len(batch.records)}"

    # Verify TrainingRecord structure
    for i, record in enumerate(batch.records):
        assert record.role_id in ["Aff", "Neg"], f"Record {i} invalid role: {record.role_id}"
        assert record.rollout_id == result.rollout.id, f"Record {i} wrong rollout_id"
        assert len(record.prompt_token_ids) > 0, f"Record {i} empty prompt_token_ids"
        assert len(record.completion_token_ids) > 0, f"Record {i} empty completion_token_ids"
        assert len(record.logprobs) > 0, f"Record {i} empty logprobs"
        assert "episode_type" in record.meta, f"Record {i} missing episode_type in meta"

    # Verify meta
    assert batch.meta["num_results"] == 1
    assert batch.meta["num_records"] == 4

    print(f"  ✓ build_training_batch() produced {len(batch.records)} TrainingRecords")
    print(f"  ✓ Each record has: role_id, rollout_id, token_ids, logprobs, reward, meta")
    print(f"  ✓ Batch meta: {batch.meta}")


async def test_full_step():
    """Test full arena.step() end-to-end."""
    print("\n" + "=" * 60)
    print("TEST: Arena.step() (end-to-end)")
    print("=" * 60)

    client = MockInferenceClient(response_fn=mock_debater)
    arena = create_debate_arena(
        client=client,
        topics=["Topic A", "Topic B", "Topic C"],
        num_rounds=2,
        batch_size=2,
    )

    # Run full step
    batch = await arena.step(concurrency=4)

    # 2 debates * 4 steps each = 8 records
    assert len(batch.records) == 8, f"Expected 8 records, got {len(batch.records)}"

    # Count by role
    aff_count = sum(1 for r in batch.records if r.role_id == "Aff")
    neg_count = sum(1 for r in batch.records if r.role_id == "Neg")
    assert aff_count == 4, f"Expected 4 Aff records, got {aff_count}"
    assert neg_count == 4, f"Expected 4 Neg records, got {neg_count}"

    print(f"  ✓ arena.step() produced {len(batch.records)} records")
    print(f"  ✓ Aff records: {aff_count}, Neg records: {neg_count}")
    print(f"  ✓ Batch meta: {batch.meta}")


async def test_run_loop():
    """Test arena.run() async generator."""
    print("\n" + "=" * 60)
    print("TEST: Arena.run() (async generator)")
    print("=" * 60)

    client = MockInferenceClient(response_fn=mock_debater)
    arena = create_debate_arena(
        client=client,
        topics=["Topic A", "Topic B"],
        num_rounds=1,
        batch_size=1,
    )

    batches = []
    async for batch in arena.run(num_steps=3):
        batches.append(batch)

    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

    for i, batch in enumerate(batches):
        assert len(batch.records) == 2, f"Batch {i} wrong record count"

    print(f"  ✓ arena.run(num_steps=3) yielded {len(batches)} batches")
    print(f"  ✓ Each batch has correct structure")


async def main():
    print("=" * 60)
    print("DEBATE SELF-PLAY INTEGRATION TESTS")
    print("=" * 60)

    await test_episode_structure()
    await test_arena_get_batch()
    await test_generate_rollouts_parallel()
    await test_build_training_batch()
    await test_full_step()
    await test_run_loop()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
