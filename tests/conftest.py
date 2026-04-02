"""Shared pytest fixtures for Engram tests."""

import pytest

from engram.store.sqlite import SQLiteMemoryStore
from engram.client import AgentMemory
from engram.core.models import Memory, MemoryType


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
