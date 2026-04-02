"""Tests for SQLiteMemoryStore backend."""

import pytest
from datetime import datetime, timezone, timedelta

from engram.core.models import (
    Memory,
    MemoryType,
    MemoryStatus,
)
from engram.core.filters import MemoryFilter
from engram.core.exceptions import StoreError, MemoryNotFoundError


@pytest.mark.asyncio
async def test_initialize_idempotent(mem_store):
    """Test that calling initialize() twice doesn't raise an error."""
    await mem_store.initialize()
    await mem_store.initialize()
    assert await mem_store.ping() is True


@pytest.mark.asyncio
async def test_insert_and_get(mem_store):
    """Test that insert returns correct ID and get returns correct content."""
    memory = Memory(
        content="test content",
        user_id="u1",
        memory_type=MemoryType.FACT,
    )
    memory_id = await mem_store.insert(memory)
    assert memory_id == memory.id

    retrieved = await mem_store.get(memory_id)
    assert retrieved is not None
    assert retrieved.content == "test content"
    assert retrieved.user_id == "u1"
    assert retrieved.memory_type == MemoryType.FACT
    assert retrieved.status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_insert_duplicate_raises(mem_store):
    """Test that inserting duplicate ID raises StoreError."""
    memory = Memory(content="test")
    await mem_store.insert(memory)

    duplicate = Memory(content="duplicate", id=memory.id)
    with pytest.raises(StoreError, match="already exists"):
        await mem_store.insert(duplicate)


@pytest.mark.asyncio
async def test_batch_insert(mem_store):
    """Test that batch_insert inserts multiple memories."""
    memories = [
        Memory(content=f"memory {i}", user_id=f"u{i}") for i in range(5)
    ]
    ids = await mem_store.batch_insert(memories)
    assert len(ids) == 5

    for memory_id in ids:
        retrieved = await mem_store.get(memory_id)
        assert retrieved is not None


@pytest.mark.asyncio
async def test_get_nonexistent(mem_store):
    """Test that get returns None for non-existent ID."""
    result = await mem_store.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_memories_basic(mem_store):
    """Test that list_memories returns inserted memories."""
    memory1 = Memory(content="first", user_id="u1")
    memory2 = Memory(content="second", user_id="u2")
    await mem_store.insert(memory1)
    await mem_store.insert(memory2)

    memories = await mem_store.list_memories(MemoryFilter())
    assert len(memories) == 2
    contents = [m.content for m in memories]
    assert "first" in contents
    assert "second" in contents


@pytest.mark.asyncio
async def test_list_memories_by_user(mem_store):
    """Test filtering list_memories by user_id."""
    await mem_store.insert(Memory(content="u1 memory", user_id="u1"))
    await mem_store.insert(Memory(content="u2 memory", user_id="u2"))
    await mem_store.insert(Memory(content="another u1", user_id="u1"))

    filter_obj = MemoryFilter(user_id="u1")
    memories = await mem_store.list_memories(filter_obj)
    assert len(memories) == 2
    assert all(m.user_id == "u1" for m in memories)


@pytest.mark.asyncio
async def test_list_memories_by_type(mem_store):
    """Test filtering list_memories by memory type."""
    await mem_store.insert(Memory(content="fact", memory_type=MemoryType.FACT))
    await mem_store.insert(Memory(content="episode", memory_type=MemoryType.EPISODE))
    await mem_store.insert(Memory(content="preference", memory_type=MemoryType.PREFERENCE))

    filter_obj = MemoryFilter(types=[MemoryType.FACT, MemoryType.EPISODE])
    memories = await mem_store.list_memories(filter_obj)
    assert len(memories) == 2
    types = [m.memory_type for m in memories]
    assert MemoryType.PREFERENCE not in types


@pytest.mark.asyncio
async def test_list_memories_by_namespace(mem_store):
    """Test filtering list_memories by namespace."""
    await mem_store.insert(Memory(content="default ns", namespace="default"))
    await mem_store.insert(Memory(content="work ns", namespace="work"))
    await mem_store.insert(Memory(content="another work", namespace="work"))

    filter_obj = MemoryFilter(namespace="work")
    memories = await mem_store.list_memories(filter_obj)
    assert len(memories) == 2
    assert all(m.namespace == "work" for m in memories)


@pytest.mark.asyncio
async def test_list_memories_pagination(mem_store):
    """Test pagination with limit and offset."""
    for i in range(5):
        await mem_store.insert(Memory(content=f"memory {i}"))

    page1 = await mem_store.list_memories(MemoryFilter(), limit=2, offset=0)
    assert len(page1) == 2

    page2 = await mem_store.list_memories(MemoryFilter(), limit=2, offset=2)
    assert len(page2) == 2

    page3 = await mem_store.list_memories(MemoryFilter(), limit=2, offset=4)
    assert len(page3) == 1


@pytest.mark.asyncio
async def test_list_memories_by_confidence(mem_store):
    """Test filtering list_memories by min_confidence."""
    await mem_store.insert(Memory(content="high confidence", confidence=0.9))
    await mem_store.insert(Memory(content="medium confidence", confidence=0.5))
    await mem_store.insert(Memory(content="low confidence", confidence=0.2))

    filter_obj = MemoryFilter(min_confidence=0.6)
    memories = await mem_store.list_memories(filter_obj)
    assert len(memories) == 1
    assert memories[0].confidence == 0.9


@pytest.mark.asyncio
async def test_keyword_search_match(mem_store):
    """Test FTS5 keyword search matching."""
    await mem_store.insert(Memory(content="Python is a programming language"))
    await mem_store.insert(Memory(content="Redis is a key-value store"))
    await mem_store.insert(Memory(content="JavaScript for web development"))

    results = await mem_store.keyword_search("Python", MemoryFilter())
    assert len(results) > 0
    assert "Python" in results[0].memory.content


@pytest.mark.asyncio
async def test_keyword_search_no_match(mem_store):
    """Test keyword search with no matches."""
    await mem_store.insert(Memory(content="Python programming"))

    results = await mem_store.keyword_search("nonexistent keyword", MemoryFilter())
    assert len(results) == 0


@pytest.mark.asyncio
async def test_update_version_increment(mem_store):
    """Test that update increments version number."""
    memory = Memory(content="original", version=1)
    memory_id = await mem_store.insert(memory)

    updated = await mem_store.update(memory_id, content="updated")
    assert updated.version == 2
    assert updated.content == "updated"


@pytest.mark.asyncio
async def test_update_saves_history(mem_store):
    """Test that update saves old version to memory_versions."""
    memory = Memory(content="original", metadata={"key": "value1"})
    memory_id = await mem_store.insert(memory)

    await mem_store.update(memory_id, content="updated", metadata={"key": "value2"})

    versions = await mem_store.get_versions(memory_id)
    assert len(versions) == 1
    assert versions[0].version == 1
    assert versions[0].content == "original"
    assert versions[0].metadata == {"key": "value1"}


@pytest.mark.asyncio
async def test_update_nonexistent_raises(mem_store):
    """Test that updating non-existent ID raises MemoryNotFoundError."""
    with pytest.raises(MemoryNotFoundError):
        await mem_store.update("nonexistent-id", content="updated")


@pytest.mark.asyncio
async def test_touch_updates_access(mem_store):
    """Test that touch updates last_accessed_at and access_count."""
    memory = Memory(content="test")
    memory_id = await mem_store.insert(memory)
    
    original = await mem_store.get(memory_id)
    original_access_time = original.last_accessed_at
    original_count = original.access_count

    # Wait a bit to ensure time difference
    import asyncio
    await asyncio.sleep(0.01)

    await mem_store.touch(memory_id)
    
    updated = await mem_store.get(memory_id)
    assert updated.access_count == original_count + 1
    assert updated.last_accessed_at > original_access_time


@pytest.mark.asyncio
async def test_delete_soft(mem_store):
    """Test that soft delete changes status to DELETED."""
    memory = Memory(content="test")
    memory_id = await mem_store.insert(memory)

    await mem_store.delete(memory_id, hard=False)
    
    retrieved = await mem_store.get(memory_id)
    assert retrieved is not None
    assert retrieved.status == MemoryStatus.DELETED


@pytest.mark.asyncio
async def test_delete_hard(mem_store):
    """Test that hard delete removes memory completely."""
    memory = Memory(content="test")
    memory_id = await mem_store.insert(memory)

    await mem_store.delete(memory_id, hard=True)
    
    retrieved = await mem_store.get(memory_id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_delete_by_filter(mem_store):
    """Test batch delete by filter."""
    await mem_store.insert(Memory(content="u1 memory 1", user_id="u1"))
    await mem_store.insert(Memory(content="u1 memory 2", user_id="u1"))
    await mem_store.insert(Memory(content="u2 memory", user_id="u2"))

    filter_obj = MemoryFilter(user_id="u1")
    deleted_count = await mem_store.delete_by_filter(filter_obj)
    assert deleted_count == 2

    remaining = await mem_store.list_memories(MemoryFilter())
    assert len(remaining) == 1
    assert remaining[0].user_id == "u2"


@pytest.mark.asyncio
async def test_expire_stale(mem_store):
    """Test that expired memories are marked as EXPIRED."""
    past_time = datetime.now(timezone.utc) - timedelta(days=1)
    memory = Memory(
        content="expires soon",
        expires_at=past_time,
        status=MemoryStatus.ACTIVE,
    )
    memory_id = await mem_store.insert(memory)

    expired_count = await mem_store.expire_stale()
    assert expired_count >= 1

    retrieved = await mem_store.get(memory_id)
    assert retrieved.status == MemoryStatus.EXPIRED


@pytest.mark.asyncio
async def test_archive_inactive(mem_store):
    """Test that inactive memories are marked as ARCHIVED."""
    memory = Memory(content="test memory")
    memory_id = await mem_store.insert(memory)

    # Directly update last_accessed_at to be very old using SQL
    db = mem_store._ensure_open()
    old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    await db.execute(
        "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
        (old_time, memory_id)
    )
    await db.commit()

    archived_count = await mem_store.archive_inactive(max_age_days=30)
    assert archived_count >= 1

    retrieved = await mem_store.get(memory_id)
    assert retrieved.status == MemoryStatus.ARCHIVED


@pytest.mark.asyncio
async def test_get_versions(mem_store):
    """Test that get_versions returns version history."""
    memory = Memory(content="v1")
    memory_id = await mem_store.insert(memory)

    await mem_store.update(memory_id, content="v2")
    await mem_store.update(memory_id, content="v3")

    versions = await mem_store.get_versions(memory_id)
    assert len(versions) == 2
    assert versions[0].content == "v1"
    assert versions[1].content == "v2"


@pytest.mark.asyncio
async def test_ping(mem_store):
    """Test that ping returns True for active store."""
    assert await mem_store.ping() is True


@pytest.mark.asyncio
async def test_stats(mem_store):
    """Test that stats returns correct statistics."""
    await mem_store.insert(Memory(content="active 1", status=MemoryStatus.ACTIVE))
    await mem_store.insert(Memory(content="active 2", status=MemoryStatus.ACTIVE))
    await mem_store.insert(Memory(content="archived", status=MemoryStatus.ARCHIVED))
    await mem_store.insert(Memory(content="expired", status=MemoryStatus.EXPIRED))
    await mem_store.insert(Memory(content="deleted", status=MemoryStatus.DELETED))

    stats = await mem_store.stats()
    assert stats.total_memories == 5
    assert stats.active_memories == 2
    assert stats.archived_memories == 1
    assert stats.expired_memories == 1
    assert stats.deleted_memories == 1
    assert stats.embedding_dimensions == 384
    assert stats.store_backend == "sqlite"


@pytest.mark.asyncio
async def test_close_then_raises(mem_store):
    """Test that operations raise error after close."""
    await mem_store.close()

    with pytest.raises(StoreError, match="Store is closed"):
        await mem_store.get("any-id")
