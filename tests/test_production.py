"""Production-readiness tests for Engram.

Covers edge cases and production scenarios beyond the happy-path tests:
- Concurrent write safety (asyncio.Lock)
- batch_insert atomicity (rollback on partial failure)
- AccessPolicy enforcement via AgentMemory
- Schema migration mechanism
- Bulk operations performance
"""

from __future__ import annotations

import asyncio

import pytest

from neuragram.client import AgentMemory
from neuragram.core.access import AccessDeniedError, AccessLevel, AccessPolicy
from neuragram.core.models import Memory, MemoryStatus
from neuragram.store.sqlite import _CURRENT_SCHEMA_VERSION, SQLiteMemoryStore
from tests.conftest import skip_no_fts5

# ── Concurrent Write Safety ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_inserts_no_lock_error():
    """Multiple concurrent inserts should not raise 'database is locked'."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    async def insert_one(i: int) -> str:
        mem = Memory(content=f"concurrent memory {i}", user_id=f"u{i}")
        return await store.insert(mem)

    # Launch 20 concurrent inserts
    ids = await asyncio.gather(*[insert_one(i) for i in range(20)])

    assert len(ids) == 20
    assert len(set(ids)) == 20  # all unique

    # Verify all are retrievable
    for mid in ids:
        m = await store.get(mid)
        assert m is not None

    await store.close()


@pytest.mark.asyncio
async def test_concurrent_inserts_and_reads():
    """Concurrent reads and writes should not interfere."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    # Pre-populate
    seed = Memory(content="seed memory", user_id="u0")
    seed_id = await store.insert(seed)

    async def do_insert(i: int) -> str:
        mem = Memory(content=f"write {i}", user_id="u1")
        return await store.insert(mem)

    async def do_read(_: int) -> Memory | None:
        return await store.get(seed_id)

    # Mix reads and writes
    tasks = []
    for i in range(10):
        tasks.append(do_insert(i))
        tasks.append(do_read(i))

    results = await asyncio.gather(*tasks)

    # Even indices are insert results (str), odd are read results (Memory|None)
    for i, result in enumerate(results):
        if i % 2 == 0:
            assert isinstance(result, str)
        else:
            assert result is not None
            assert result.content == "seed memory"

    await store.close()


# ── batch_insert Atomicity ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_insert_atomicity_on_duplicate():
    """If one memory in a batch has a duplicate ID, the entire batch rolls back."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    # Insert one memory first
    existing = Memory(content="existing", user_id="u1")
    await store.insert(existing)

    # Build a batch where the third item has the same ID as the existing one
    batch = [
        Memory(content="new 1", user_id="u1"),
        Memory(content="new 2", user_id="u1"),
        Memory(content="duplicate", user_id="u1", id=existing.id),  # will fail
    ]

    with pytest.raises(Exception):
        await store.batch_insert(batch)

    # The first two memories in the batch should NOT exist (rolled back)
    from neuragram.core.filters import MemoryFilter

    all_memories = await store.list_memories(MemoryFilter())
    assert len(all_memories) == 1  # only the original
    assert all_memories[0].content == "existing"

    await store.close()


@pytest.mark.asyncio
async def test_batch_insert_empty():
    """batch_insert with empty list returns empty list."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    ids = await store.batch_insert([])
    assert ids == []

    await store.close()


@pytest.mark.asyncio
async def test_batch_insert_all_succeed():
    """batch_insert commits all memories atomically on success."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    batch = [Memory(content=f"batch item {i}", user_id="u1") for i in range(10)]
    ids = await store.batch_insert(batch)

    assert len(ids) == 10
    for mid in ids:
        m = await store.get(mid)
        assert m is not None

    await store.close()


# ── AccessPolicy Enforcement ────────────────────────────────────────


@pytest.mark.asyncio
async def test_access_policy_blocks_write_for_reader():
    """A READ-only actor cannot remember (WRITE operation)."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("reader", AccessLevel.READ)

    mem = AgentMemory(
        db_path=":memory:",
        embedding="none",
        access_policy=policy,
        actor_id="reader",
    )
    await mem._ensure_initialized()

    with pytest.raises(AccessDeniedError):
        await mem.aremember("should be blocked", user_id="u1")

    await mem.aclose()


@pytest.mark.asyncio
async def test_access_policy_allows_read_for_reader():
    """A READ actor can recall memories."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("reader", AccessLevel.READ)

    mem = AgentMemory(
        db_path=":memory:",
        embedding="none",
        access_policy=policy,
        actor_id="reader",
    )
    await mem._ensure_initialized()

    # recall should not raise
    results = await mem.arecall("anything")
    assert isinstance(results, list)

    await mem.aclose()


@pytest.mark.asyncio
async def test_access_policy_blocks_admin_ops_for_writer():
    """A WRITE actor cannot run decay (ADMIN operation)."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("writer", AccessLevel.WRITE)

    mem = AgentMemory(
        db_path=":memory:",
        embedding="none",
        access_policy=policy,
        actor_id="writer",
    )
    await mem._ensure_initialized()

    with pytest.raises(AccessDeniedError):
        await mem.adecay()

    await mem.aclose()


@pytest.mark.asyncio
async def test_access_policy_admin_can_do_everything():
    """An ADMIN actor can perform all operations."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("admin", AccessLevel.ADMIN)

    mem = AgentMemory(
        db_path=":memory:",
        embedding="none",
        access_policy=policy,
        actor_id="admin",
    )
    await mem._ensure_initialized()

    # All operations should succeed
    mid = await mem.aremember("admin memory", user_id="u1")
    assert mid

    results = await mem.arecall("admin")
    assert isinstance(results, list)

    await mem.adecay()

    await mem.aforget(memory_id=mid)

    await mem.aclose()


@pytest.mark.asyncio
async def test_access_policy_disabled_allows_all():
    """When AccessPolicy is disabled (default), all operations are allowed."""
    mem = AgentMemory(
        db_path=":memory:",
        embedding="none",
        actor_id="anyone",
    )
    await mem._ensure_initialized()

    mid = await mem.aremember("test", user_id="u1")
    assert mid
    await mem.arecall("test")
    await mem.adecay()

    await mem.aclose()


@pytest.mark.asyncio
async def test_access_policy_namespace_scoping():
    """Permissions scoped to a namespace only apply to that namespace."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("agent", AccessLevel.WRITE, namespace="allowed")

    mem = AgentMemory(
        db_path=":memory:",
        embedding="none",
        access_policy=policy,
        actor_id="agent",
    )
    await mem._ensure_initialized()

    # Should succeed in allowed namespace
    mid = await mem.aremember("ok", namespace="allowed")
    assert mid

    # Should fail in a different namespace
    with pytest.raises(AccessDeniedError):
        await mem.aremember("blocked", namespace="forbidden")

    await mem.aclose()


# ── Schema Migration ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_schema_version_stamped():
    """initialize() stamps the schema version on a fresh database."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    version = await store.schema_version()
    assert version == _CURRENT_SCHEMA_VERSION
    assert version >= 1

    await store.close()


@pytest.mark.asyncio
async def test_schema_version_idempotent():
    """Calling initialize() twice does not change the schema version."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    v1 = await store.schema_version()
    await store.initialize()
    v2 = await store.schema_version()

    assert v1 == v2

    await store.close()


@pytest.mark.asyncio
async def test_schema_meta_table_exists():
    """The _schema_meta table is created during initialization."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    db = store._ensure_open()
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_meta'"
    )
    row = await cursor.fetchone()
    assert row is not None

    await store.close()


# ── Bulk Operations ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bulk_insert_and_list_1000():
    """Insert 1000 memories and verify all are retrievable."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    memories = [
        Memory(content=f"bulk memory {i}", user_id="bulk_user")
        for i in range(1000)
    ]
    ids = await store.batch_insert(memories)
    assert len(ids) == 1000

    from neuragram.core.filters import MemoryFilter

    # List with high limit
    results = await store.list_memories(
        MemoryFilter(user_id="bulk_user"), limit=2000
    )
    assert len(results) == 1000

    await store.close()


@skip_no_fts5
@pytest.mark.asyncio
async def test_bulk_keyword_search():
    """Keyword search works correctly after bulk insertion."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    # Insert memories with distinct keywords
    for i in range(100):
        mem = Memory(
            content=f"memory about topic_alpha number {i}",
            user_id="u1",
        )
        await store.insert(mem)
    for i in range(50):
        mem = Memory(
            content=f"memory about topic_beta number {i}",
            user_id="u1",
        )
        await store.insert(mem)

    from neuragram.core.filters import MemoryFilter

    alpha_results = await store.keyword_search("topic_alpha", MemoryFilter())
    assert len(alpha_results) > 0

    beta_results = await store.keyword_search("topic_beta", MemoryFilter())
    assert len(beta_results) > 0

    # No overlap
    alpha_ids = {r.memory.id for r in alpha_results}
    beta_ids = {r.memory.id for r in beta_results}
    assert alpha_ids.isdisjoint(beta_ids)

    await store.close()


# ── Stats Optimization ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_counts_all_statuses():
    """stats() correctly counts memories across all statuses."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    await store.insert(Memory(content="a1", status=MemoryStatus.ACTIVE))
    await store.insert(Memory(content="a2", status=MemoryStatus.ACTIVE))
    await store.insert(Memory(content="a3", status=MemoryStatus.ACTIVE))
    await store.insert(Memory(content="ar1", status=MemoryStatus.ARCHIVED))
    await store.insert(Memory(content="e1", status=MemoryStatus.EXPIRED))
    await store.insert(Memory(content="d1", status=MemoryStatus.DELETED))
    await store.insert(Memory(content="d2", status=MemoryStatus.DELETED))

    stats = await store.stats()
    assert stats.total_memories == 7
    assert stats.active_memories == 3
    assert stats.archived_memories == 1
    assert stats.expired_memories == 1
    assert stats.deleted_memories == 2

    await store.close()


# ── Write Lock Under Contention ─────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_updates_no_corruption():
    """Concurrent updates to the same memory don't corrupt data."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()

    mem = Memory(content="original", user_id="u1")
    mid = await store.insert(mem)

    async def update_content(suffix: str) -> Memory:
        return await store.update(mid, content=f"updated_{suffix}")

    # 10 concurrent updates
    results = await asyncio.gather(
        *[update_content(str(i)) for i in range(10)]
    )

    # All should succeed (serialized by lock)
    assert len(results) == 10

    # Final state should be one of the updates
    final = await store.get(mid)
    assert final is not None
    assert final.content.startswith("updated_")
    # Version should have incremented 10 times
    assert final.version == 11  # 1 (initial) + 10 updates

    await store.close()
