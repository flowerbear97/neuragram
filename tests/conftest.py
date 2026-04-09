"""Shared pytest fixtures for Engram tests."""

import sqlite3

import pytest

from neuragram.client import AgentMemory
from neuragram.core.models import Memory, MemoryType
from neuragram.store.sqlite import SQLiteMemoryStore


def _fts5_available() -> bool:
    """Check whether the current SQLite build includes FTS5."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts5_test USING fts5(x)")
        conn.close()
        return True
    except Exception:
        return False


HAS_FTS5 = _fts5_available()
skip_no_fts5 = pytest.mark.skipif(not HAS_FTS5, reason="FTS5 not available")


@pytest.fixture
async def mem_store():
    """Create an in-memory SQLiteMemoryStore for testing."""
    store = SQLiteMemoryStore(db_path=":memory:", dimension=384)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def agent_memory():
    """Create an in-memory AgentMemory for testing."""
    memory = AgentMemory(store="sqlite", db_path=":memory:", embedding="none")
    await memory._ensure_initialized()
    yield memory
    await memory.aclose()


@pytest.fixture
def sample_memories():
    """Return 5 sample Memory objects of different types."""
    return [
        Memory(
            user_id="u1",
            content="User is a Python developer",
            memory_type=MemoryType.FACT,
        ),
        Memory(
            user_id="u1",
            content="In March 2024 user resolved a Redis connection timeout issue",
            memory_type=MemoryType.EPISODE,
        ),
        Memory(
            user_id="u1",
            content="User prefers concise code comments",
            memory_type=MemoryType.PREFERENCE,
        ),
        Memory(
            user_id="u2",
            content="User uses macOS",
            memory_type=MemoryType.FACT,
        ),
        Memory(
            user_id="u1",
            namespace="work",
            content="Deployment process: first run tests then merge to main branch",
            memory_type=MemoryType.PROCEDURE,
        ),
    ]
