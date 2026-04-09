"""Tests for lifecycle management (decay and forgetting)."""

import pytest
from neuragram.lifecycle.decay import DecayManager
from neuragram.lifecycle.forgetting import ForgettingManager
from neuragram.store.sqlite import SQLiteMemoryStore
from neuragram.core.models import Memory, MemoryType, MemoryStatus
from datetime import datetime, timezone, timedelta


@pytest.mark.asyncio
async def test_decay_expires_ttl():
    """DecayManager.run_decay() expires TTL memories."""
    store = SQLiteMemoryStore(":memory:", dimension=384)
    await store.initialize()

    # Create memory with expired TTL
    now = datetime.now(timezone.utc)
    expired_time = now - timedelta(days=1)
    
    memory = Memory(
        content="will expire",
        memory_type=MemoryType.FACT,
        user_id="user1",
        expires_at=expired_time,
    )
    await store.insert(memory)

    # Run decay
    decay_manager = DecayManager(store=store, ttl_enabled=True)
    result = await decay_manager.run_decay()

    assert result.expired == 1

    # Verify status
    retrieved = await store.get(memory.id)
    assert retrieved is not None
    assert retrieved.status == MemoryStatus.EXPIRED

    await store.close()


@pytest.mark.asyncio
async def test_decay_archives_inactive():
    """DecayManager.run_decay() archives inactive memories."""
    store = SQLiteMemoryStore(":memory:", dimension=384)
    await store.initialize()

    # Create memory then manually set last_accessed_at to long ago via SQL
    memory = Memory(
        content="inactive memory",
        memory_type=MemoryType.FACT,
        user_id="user1",
    )
    await store.insert(memory)

    # Directly update last_accessed_at in DB to simulate old access
    old_time = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    await store._db.execute(
        "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
        (old_time, memory.id),
    )
    await store._db.commit()

    # Run decay with max_age_days=30
    decay_manager = DecayManager(store=store, max_age_days=30, ttl_enabled=False)
    result = await decay_manager.run_decay()

    assert result.archived == 1

    # Verify status
    retrieved = await store.get(memory.id)
    assert retrieved is not None
    assert retrieved.status == MemoryStatus.ARCHIVED

    await store.close()


@pytest.mark.asyncio
async def test_decay_no_op_when_nothing_to_do():
    """Decay returns 0 when there are no memories to process."""
    store = SQLiteMemoryStore(":memory:", dimension=384)
    await store.initialize()

    # Create recent memory that shouldn't be affected
    now = datetime.now(timezone.utc)
    memory = Memory(
        content="recent memory",
        memory_type=MemoryType.FACT,
        user_id="user1",
        created_at=now,
        last_accessed_at=now,
    )
    await store.insert(memory)

    # Run decay
    decay_manager = DecayManager(store=store, max_age_days=30, ttl_enabled=True)
    result = await decay_manager.run_decay()

    assert result.expired == 0
    assert result.archived == 0

    # Verify memory is still active
    retrieved = await store.get(memory.id)
    assert retrieved is not None
    assert retrieved.status == MemoryStatus.ACTIVE

    await store.close()


@pytest.mark.asyncio
async def test_forget_user_soft():
    """ForgettingManager.forget_user() performs soft delete."""
    store = SQLiteMemoryStore(":memory:", dimension=384)
    await store.initialize()

    # Create memories for user1
    now = datetime.now(timezone.utc)
    memory1 = Memory(
        content="user1 memory 1",
        memory_type=MemoryType.FACT,
        user_id="user1",
        created_at=now,
        last_accessed_at=now,
    )
    memory2 = Memory(
        content="user1 memory 2",
        memory_type=MemoryType.FACT,
        user_id="user1",
        created_at=now,
        last_accessed_at=now,
    )
    await store.insert(memory1)
    await store.insert(memory2)

    # Soft delete
    forgetting_manager = ForgettingManager(store=store)
    result = await forgetting_manager.forget_user("user1", hard=False)

    assert result.deleted_count == 2
    assert result.hard is False

    # Verify memories still exist but are marked DELETED
    retrieved1 = await store.get(memory1.id)
    retrieved2 = await store.get(memory2.id)
    assert retrieved1 is not None
    assert retrieved2 is not None
    assert retrieved1.status == MemoryStatus.DELETED
    assert retrieved2.status == MemoryStatus.DELETED

    await store.close()


@pytest.mark.asyncio
async def test_forget_user_hard():
    """ForgettingManager.forget_user(hard=True) performs physical deletion."""
    store = SQLiteMemoryStore(":memory:", dimension=384)
    await store.initialize()

    # Create memories for user1
    now = datetime.now(timezone.utc)
    memory1 = Memory(
        content="user1 memory 1",
        memory_type=MemoryType.FACT,
        user_id="user1",
        created_at=now,
        last_accessed_at=now,
    )
    memory2 = Memory(
        content="user1 memory 2",
        memory_type=MemoryType.FACT,
        user_id="user1",
        created_at=now,
        last_accessed_at=now,
    )
    await store.insert(memory1)
    await store.insert(memory2)

    # Hard delete
    forgetting_manager = ForgettingManager(store=store)
    result = await forgetting_manager.forget_user("user1", hard=True)

    assert result.deleted_count == 2
    assert result.hard is True

    # Verify memories are physically deleted
    retrieved1 = await store.get(memory1.id)
    retrieved2 = await store.get(memory2.id)
    assert retrieved1 is None
    assert retrieved2 is None

    await store.close()


@pytest.mark.asyncio
async def test_forget_single_memory():
    """ForgettingManager.forget_memory() deletes a single memory."""
    store = SQLiteMemoryStore(":memory:", dimension=384)
    await store.initialize()

    # Create memory
    now = datetime.now(timezone.utc)
    memory = Memory(
        content="to be deleted",
        memory_type=MemoryType.FACT,
        user_id="user1",
        created_at=now,
        last_accessed_at=now,
    )
    await store.insert(memory)

    # Delete single memory
    forgetting_manager = ForgettingManager(store=store)
    success = await forgetting_manager.forget_memory(memory.id, hard=False)

    assert success is True

    # Verify memory is marked DELETED
    retrieved = await store.get(memory.id)
    assert retrieved is not None
    assert retrieved.status == MemoryStatus.DELETED

    await store.close()
