"""Tests for retrieval engine scoring functions."""

from datetime import datetime, timedelta, timezone

import pytest

from neuragram.client import AgentMemory
from neuragram.core.models import Memory, MemoryType, ScoredMemory
from neuragram.retrieval.scoring import (
    apply_recency_boost,
    deduplicate,
    reciprocal_rank_fusion,
)
from tests.conftest import skip_no_fts5


@pytest.mark.asyncio
async def test_rrf_fuses_two_lists():
    """Two ranked lists are fused correctly; memories appearing in both lists get higher scores."""
    now = datetime.now(timezone.utc)
    memory_a = Memory(
        content="apple fruit",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )
    memory_b = Memory(
        content="banana fruit",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )
    memory_c = Memory(
        content="cherry fruit",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )

    list1 = [
        ScoredMemory(memory=memory_a, score=0.9),
        ScoredMemory(memory=memory_b, score=0.8),
    ]
    list2 = [
        ScoredMemory(memory=memory_a, score=0.85),
        ScoredMemory(memory=memory_c, score=0.7),
    ]

    result = reciprocal_rank_fusion([list1, list2])

    # Memory A appears in both lists, should have highest score
    assert result[0].memory.id == memory_a.id
    assert result[0].score > result[1].score
    assert len(result) == 3


@pytest.mark.asyncio
async def test_rrf_weights_affect_order():
    """Different weights affect the final ranking order."""
    now = datetime.now(timezone.utc)
    memory_a = Memory(
        content="apple",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )
    memory_b = Memory(
        content="banana",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )

    list1 = [
        ScoredMemory(memory=memory_a, score=0.9),
        ScoredMemory(memory=memory_b, score=0.8),
    ]
    list2 = [
        ScoredMemory(memory=memory_b, score=0.9),
        ScoredMemory(memory=memory_a, score=0.7),
    ]

    # Give higher weight to list2 where B is ranked first
    result = reciprocal_rank_fusion([list1, list2], weights=[1.0, 2.0])

    assert result[0].memory.id == memory_b.id
    assert result[1].memory.id == memory_a.id


@pytest.mark.asyncio
async def test_rrf_empty_input():
    """Empty input returns empty list."""
    result = reciprocal_rank_fusion([])
    assert result == []

    result = reciprocal_rank_fusion([[], []])
    assert result == []


@pytest.mark.asyncio
async def test_recency_boost_recent_higher():
    """Recent memories get higher scores after recency boost."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=10)

    recent_memory = Memory(
        content="recent fact",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )
    old_memory = Memory(
        content="old fact",
        memory_type=MemoryType.FACT,
        created_at=old_time,
        last_accessed_at=old_time,
    )

    scored = [
        ScoredMemory(memory=old_memory, score=0.9),
        ScoredMemory(memory=recent_memory, score=0.9),
    ]

    result = apply_recency_boost(scored, half_life_days=7.0, weight=0.3)

    # Recent memory should now have higher score
    assert result[0].memory.id == recent_memory.id
    assert result[0].score > result[1].score


@pytest.mark.asyncio
async def test_recency_boost_half_life():
    """Half-life parameter affects the decay rate."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=7)

    memory = Memory(
        content="fact",
        memory_type=MemoryType.FACT,
        created_at=old_time,
        last_accessed_at=old_time,
    )
    scored = [ScoredMemory(memory=memory, score=0.9)]

    # Shorter half-life = more decay
    result_short = apply_recency_boost(scored, half_life_days=3.5, weight=0.5)
    result_long = apply_recency_boost(scored, half_life_days=14.0, weight=0.5)

    assert result_long[0].score > result_short[0].score


@pytest.mark.asyncio
async def test_recency_boost_zero_weight():
    """When weight=0, scores remain unchanged."""
    now = datetime.now(timezone.utc)
    memory = Memory(
        content="fact",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )
    original_score = 0.85
    scored = [ScoredMemory(memory=memory, score=original_score)]

    result = apply_recency_boost(scored, weight=0.0)

    assert result[0].score == original_score


@pytest.mark.asyncio
async def test_deduplicate_removes_similar():
    """High similarity results (same content) are removed."""
    now = datetime.now(timezone.utc)
    memory1 = Memory(
        content="the sky is blue",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )
    memory2 = Memory(
        content="the sky is blue",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )

    scored = [
        ScoredMemory(memory=memory1, score=0.9),
        ScoredMemory(memory=memory2, score=0.8),
    ]

    result = deduplicate(scored, threshold=0.95)

    # Only one should remain (the higher scored one)
    assert len(result) == 1
    assert result[0].memory.id == memory1.id


@pytest.mark.asyncio
async def test_deduplicate_keeps_different():
    """Different content memories are kept."""
    now = datetime.now(timezone.utc)
    memory1 = Memory(
        content="the sky is blue",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )
    memory2 = Memory(
        content="the grass is green",
        memory_type=MemoryType.FACT,
        created_at=now,
        last_accessed_at=now,
    )

    scored = [
        ScoredMemory(memory=memory1, score=0.9),
        ScoredMemory(memory=memory2, score=0.8),
    ]

    result = deduplicate(scored, threshold=0.95)

    # Both should be kept
    assert len(result) == 2


@skip_no_fts5
@pytest.mark.asyncio
async def test_retrieval_engine_end_to_end(agent_memory: AgentMemory):
    """End-to-end test: remember memories and recall them."""
    await agent_memory._ensure_initialized()

    # Store 3 different memories
    id1 = await agent_memory.aremember(
        "user prefers concise answers",
        user_id="user1",
        type="preference",
    )
    id2 = await agent_memory.aremember(
        "user likes dark mode",
        user_id="user1",
        type="preference",
    )
    id3 = await agent_memory.aremember(
        "user is a developer",
        user_id="user1",
        type="fact",
    )

    # Verify IDs are returned
    assert id1
    assert id2
    assert id3

    # Recall memories
    results = await agent_memory.arecall("what does the user prefer?", user_id="user1")

    # Should return non-empty results
    assert len(results) > 0
    assert any("concise" in sm.memory.content.lower() for sm in results)
