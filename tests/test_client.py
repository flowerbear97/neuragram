"""Tests for the main AgentMemory client."""

import pytest

from neuragram.client import AgentMemory
from neuragram.core.models import MemoryStatus
from tests.conftest import skip_no_fts5


@pytest.mark.asyncio
async def test_zero_config_init():
    """AgentMemory(db_path=":memory:") initializes successfully."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()
    assert mem is not None
    await mem.aclose()


@pytest.mark.asyncio
async def test_remember_returns_id():
    """remember() returns a non-empty string ID."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    memory_id = await mem.aremember("test memory")

    assert memory_id
    assert isinstance(memory_id, str)
    assert len(memory_id) > 0

    await mem.aclose()


@pytest.mark.asyncio
async def test_remember_with_all_params():
    """remember() with all parameters specified."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    memory_id = await mem.aremember(
        content="comprehensive test memory",
        user_id="test_user",
        type="preference",
        tags=["important", "test"],
        metadata={"key": "value"},
        importance=0.9,
        confidence=0.95,
        source="unit_test",
    )

    memory = await mem.aget(memory_id)
    assert memory is not None
    assert memory.content == "comprehensive test memory"
    assert memory.user_id == "test_user"
    assert memory.memory_type.value == "preference"
    assert "important" in memory.tags
    assert "test" in memory.tags
    assert memory.metadata == {"key": "value"}
    assert memory.importance == 0.9
    assert memory.confidence == 0.95
    assert memory.source == "unit_test"

    await mem.aclose()


@skip_no_fts5
@pytest.mark.asyncio
async def test_recall_finds_remembered():
    """recall() can retrieve recently stored memories (embedding=none uses keyword)."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    await mem.aremember("the user likes python programming", user_id="user1")
    await mem.aremember("the user prefers dark mode", user_id="user1")

    results = await mem.arecall("python", user_id="user1")

    assert len(results) > 0
    assert any("python" in sm.memory.content.lower() for sm in results)

    await mem.aclose()


@pytest.mark.asyncio
async def test_forget_by_memory_id():
    """forget(memory_id=...) marks memory as DELETED."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    memory_id = await mem.aremember("to be forgotten", user_id="user1")

    # Forget by memory ID
    deleted_count = await mem.aforget(memory_id=memory_id)
    assert deleted_count == 1

    # Check status
    memory = await mem.aget(memory_id)
    assert memory is not None
    assert memory.status == MemoryStatus.DELETED

    await mem.aclose()


@pytest.mark.asyncio
async def test_forget_by_user_id():
    """forget(user_id=...) deletes all memories for that user."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    await mem.aremember("memory 1", user_id="user1")
    await mem.aremember("memory 2", user_id="user1")
    await mem.aremember("memory 3", user_id="user2")

    # Forget user1's memories
    deleted_count = await mem.aforget(user_id="user1")
    assert deleted_count == 2

    # Verify user1 memories are deleted
    results = await mem.alist(user_id="user1")
    assert len(results) == 0

    # Verify user2 memories still exist
    results = await mem.alist(user_id="user2")
    assert len(results) == 1

    await mem.aclose()


@pytest.mark.asyncio
async def test_update_content():
    """update() changes content and increments version."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    memory_id = await mem.aremember("original content", user_id="user1")
    original_memory = await mem.aget(memory_id)
    original_version = original_memory.version

    # Update content
    updated = await mem.aupdate(memory_id, content="updated content")

    assert updated.content == "updated content"
    assert updated.version == original_version + 1

    await mem.aclose()


@pytest.mark.asyncio
async def test_history_returns_versions():
    """history() returns list of old versions after update."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    memory_id = await mem.aremember("version 1", user_id="user1", metadata={"v": 1})

    await mem.aupdate(memory_id, content="version 2", metadata={"v": 2})
    await mem.aupdate(memory_id, content="version 3", metadata={"v": 3})

    history = await mem.ahistory(memory_id)

    assert len(history) == 2  # Two old versions
    assert history[0].content == "version 1"
    assert history[0].metadata == {"v": 1}
    assert history[1].content == "version 2"
    assert history[1].metadata == {"v": 2}

    await mem.aclose()


@pytest.mark.asyncio
async def test_decay_runs():
    """decay() executes without error and returns dict with expired and archived."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    # Add some memories
    await mem.aremember("memory 1", user_id="user1")
    await mem.aremember("memory 2", user_id="user1")

    result = await mem.adecay()

    assert isinstance(result, dict)
    assert "expired" in result
    assert "archived" in result
    assert isinstance(result["expired"], int)
    assert isinstance(result["archived"], int)

    await mem.aclose()


@pytest.mark.asyncio
async def test_stats_returns_stats():
    """stats() returns StoreStats object."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    await mem.aremember("test memory", user_id="user1")

    stats = await mem.astats()

    assert stats.total_memories >= 1
    assert stats.active_memories >= 1
    assert stats.store_backend == "sqlite"

    await mem.aclose()


@pytest.mark.asyncio
async def test_list_memories():
    """list() returns list of memories."""
    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    await mem.aremember("memory 1", user_id="user1")
    await mem.aremember("memory 2", user_id="user1")
    await mem.aremember("memory 3", user_id="user2")

    results = await mem.alist(user_id="user1")

    assert len(results) == 2
    assert all(m.user_id == "user1" for m in results)

    await mem.aclose()


def test_context_manager():
    """with AgentMemory(...) as mem: works correctly."""
    with AgentMemory(db_path=":memory:", embedding="none") as mem:
        memory_id = mem.remember("test memory", user_id="user1")
        assert memory_id

        # recall returns list (may be empty if FTS5 is unavailable)
        results = mem.recall("test", user_id="user1")
        assert isinstance(results, list)
