"""Tests for core data models."""

import pytest

from neuragram.core.models import (
    Memory,
    MemoryType,
    MemoryStatus,
    ScoredMemory,
    MemoryUpdate,
    StoreStats,
)
from neuragram.core.filters import MemoryFilter


def test_memory_default_values():
    """Test that Memory has correct default values."""
    memory = Memory(content="test content")
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.version == 1
    assert memory.confidence == 1.0
    assert memory.importance == 0.5
    assert memory.access_count == 0
    assert memory.memory_type == MemoryType.FACT
    assert memory.namespace == "default"
    assert memory.user_id == ""
    assert memory.agent_id == ""


def test_memory_unique_ids():
    """Test that two Memory instances have unique IDs."""
    memory1 = Memory(content="first")
    memory2 = Memory(content="second")
    assert memory1.id != memory2.id


def test_memory_type_from_string():
    """Test that MemoryType can be created from string."""
    assert MemoryType("fact") == MemoryType.FACT
    assert MemoryType("episode") == MemoryType.EPISODE
    assert MemoryType("preference") == MemoryType.PREFERENCE
    assert MemoryType("procedure") == MemoryType.PROCEDURE
    assert MemoryType("plan_state") == MemoryType.PLAN_STATE


def test_memory_status_values():
    """Test that MemoryStatus has correct enum values."""
    assert MemoryStatus.ACTIVE.value == "active"
    assert MemoryStatus.ARCHIVED.value == "archived"
    assert MemoryStatus.EXPIRED.value == "expired"
    assert MemoryStatus.DELETED.value == "deleted"


def test_memory_update_partial():
    """Test that MemoryUpdate can have partial fields."""
    update = MemoryUpdate(content="new content")
    assert update.content == "new content"
    assert update.memory_type is None
    assert update.metadata is None
    assert update.tags is None
    assert update.confidence is None
    assert update.importance is None
    assert update.expires_at is None
    assert update.source is None


def test_memory_filter_default_status():
    """Test that MemoryFilter defaults to ACTIVE status."""
    filter_obj = MemoryFilter()
    assert filter_obj.statuses == [MemoryStatus.ACTIVE]


def test_scored_memory_sorting():
    """Test that ScoredMemory sorts by score."""
    memory1 = Memory(content="low score")
    memory2 = Memory(content="high score")
    memory3 = Memory(content="mid score")
    
    scored1 = ScoredMemory(memory=memory1, score=0.3)
    scored2 = ScoredMemory(memory=memory2, score=0.9)
    scored3 = ScoredMemory(memory=memory3, score=0.6)
    
    # Test comparison operators
    assert scored1 < scored2
    assert scored2 > scored3
    assert scored1 <= scored3
    assert scored2 >= scored3
    
    # Test sorting
    sorted_memories = sorted([scored2, scored1, scored3])
    assert sorted_memories[0] == scored1
    assert sorted_memories[1] == scored3
    assert sorted_memories[2] == scored2


def test_store_stats_defaults():
    """Test that StoreStats defaults to zero."""
    stats = StoreStats()
    assert stats.total_memories == 0
    assert stats.active_memories == 0
    assert stats.archived_memories == 0
    assert stats.expired_memories == 0
    assert stats.deleted_memories == 0
    assert stats.total_users == 0
    assert stats.total_namespaces == 0
    assert stats.embedding_dimensions == 0
    assert stats.store_backend == ""
