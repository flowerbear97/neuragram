"""Tests for Phase 3: MCP Server, REST API, integrations, explain(), multi-agent."""

from __future__ import annotations

import json
import pytest

from engram import AgentMemory, ScoreExplanation


# ── Explain Tests ───────────────────────────────────────────────────


@pytest.fixture
async def explain_memory(tmp_path):
    """Create an AgentMemory with some memories for explain testing."""
    mem = AgentMemory(db_path=str(tmp_path / "explain_test.db"))
    await mem._ensure_initialized()

    await mem.aremember("User is a Python developer", user_id="u1", type="fact")
    await mem.aremember("User prefers dark mode", user_id="u1", type="preference")
    await mem.aremember("User works at Alibaba", user_id="u1", type="fact")

    yield mem
    await mem.aclose()


@pytest.mark.asyncio
async def test_explain_returns_explanations(explain_memory):
    """explain() returns score breakdowns for search results."""
    explanations = await explain_memory.aexplain("Python developer", user_id="u1")
    assert isinstance(explanations, list)
    for exp in explanations:
        assert "memory_id" in exp
        assert "final_score" in exp
        assert "rrf_score" in exp
        assert "recency_factor" in exp
        assert "summary" in exp


@pytest.mark.asyncio
async def test_explain_has_keyword_rank(explain_memory):
    """explain() includes keyword rank when keyword search matches."""
    explanations = await explain_memory.aexplain("Python", user_id="u1")
    # At least one result should have a keyword rank
    has_keyword = any(exp.get("keyword_rank") is not None for exp in explanations)
    assert has_keyword


@pytest.mark.asyncio
async def test_explain_sync(explain_memory):
    """Sync explain() works correctly."""
    explanations = explain_memory.explain("Python developer", user_id="u1")
    assert isinstance(explanations, list)


# ── Multi-Agent Isolation Tests ─────────────────────────────────────


@pytest.fixture
async def multi_agent_memory(tmp_path):
    """Create an AgentMemory with memories from different agents."""
    mem = AgentMemory(db_path=str(tmp_path / "multi_agent.db"))
    await mem._ensure_initialized()

    await mem.aremember(
        "User prefers Python", user_id="u1", agent_id="agent_coder", type="preference"
    )
    await mem.aremember(
        "User likes sushi", user_id="u1", agent_id="agent_lifestyle", type="preference"
    )
    await mem.aremember(
        "User is a senior engineer", user_id="u1", agent_id="agent_coder", type="fact"
    )

    yield mem
    await mem.aclose()


@pytest.mark.asyncio
async def test_recall_with_agent_id_filter(multi_agent_memory):
    """recall() with agent_id filters to that agent's memories only."""
    results = await multi_agent_memory.arecall(
        "preferences", user_id="u1", agent_id="agent_coder"
    )
    for scored in results:
        assert scored.memory.agent_id == "agent_coder"


@pytest.mark.asyncio
async def test_list_with_agent_id_filter(multi_agent_memory):
    """list() with agent_id filters correctly."""
    coder_memories = await multi_agent_memory.alist(
        user_id="u1", agent_id="agent_coder"
    )
    assert len(coder_memories) == 2
    for mem in coder_memories:
        assert mem.agent_id == "agent_coder"

    lifestyle_memories = await multi_agent_memory.alist(
        user_id="u1", agent_id="agent_lifestyle"
    )
    assert len(lifestyle_memories) == 1
    assert lifestyle_memories[0].agent_id == "agent_lifestyle"


@pytest.mark.asyncio
async def test_list_without_agent_id_returns_all(multi_agent_memory):
    """list() without agent_id returns memories from all agents."""
    all_memories = await multi_agent_memory.alist(user_id="u1")
    assert len(all_memories) == 3


# ── MCP Server Tests ───────────────────────────────────────────────


def test_mcp_server_creation_without_mcp_sdk():
    """create_mcp_server raises ImportError when mcp is not installed."""
    # This test verifies the import guard works
    # If mcp IS installed, this test will pass by creating the server
    from engram.server.mcp import create_mcp_server

    try:
        server = create_mcp_server(db_path=":memory:")
        # If we get here, mcp is installed — that's fine
        assert server is not None
    except ImportError as exc:
        assert "mcp" in str(exc).lower()


# ── REST API Tests ──────────────────────────────────────────────────


def test_rest_api_creation_without_fastapi():
    """create_app raises ImportError when fastapi is not installed."""
    from engram.server.api import create_app

    try:
        app = create_app(db_path=":memory:")
        assert app is not None
    except ImportError as exc:
        assert "fastapi" in str(exc).lower()


# ── LangChain Adapter Tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_langchain_adapter_save_and_load(tmp_path):
    """EngramMemory saves context and loads relevant memories."""
    from engram.integrations.langchain import EngramMemory

    memory = EngramMemory(
        db_path=str(tmp_path / "langchain_test.db"),
        user_id="u1",
        memory_key="history",
    )

    # Save context
    memory.save_context(
        inputs={"input": "I prefer concise answers"},
        outputs={"output": "Got it, I'll keep things brief!"},
    )

    # Load memory variables
    result = memory.load_memory_variables({"input": "answer style"})
    assert "history" in result
    # Should find something related to the saved context
    assert isinstance(result["history"], str)

    memory.close()


@pytest.mark.asyncio
async def test_langchain_adapter_memory_variables(tmp_path):
    """EngramMemory exposes correct memory_variables."""
    from engram.integrations.langchain import EngramMemory

    memory = EngramMemory(
        db_path=str(tmp_path / "langchain_test2.db"),
        memory_key="context",
    )
    assert memory.memory_variables == ["context"]
    memory.close()


@pytest.mark.asyncio
async def test_langchain_adapter_clear(tmp_path):
    """EngramMemory.clear() removes user memories."""
    from engram.integrations.langchain import EngramMemory

    memory = EngramMemory(
        db_path=str(tmp_path / "langchain_test3.db"),
        user_id="u1",
    )

    memory.save_context(
        inputs={"input": "test input"},
        outputs={"output": "test output"},
    )
    memory.clear()

    result = memory.load_memory_variables({"input": "test"})
    assert result["history"] == ""

    memory.close()


# ── LlamaIndex Adapter Tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_llamaindex_adapter_put_and_get(tmp_path):
    """EngramChatMemory stores and retrieves memories."""
    from engram.integrations.llamaindex import EngramChatMemory

    memory = EngramChatMemory(
        db_path=str(tmp_path / "llamaindex_test.db"),
        user_id="u1",
    )

    memory_id = memory.put("User is a Python developer", memory_type="fact")
    assert memory_id

    results = memory.get("Python developer")
    assert len(results) >= 1
    assert any("Python" in r["content"] for r in results)

    memory.close()


@pytest.mark.asyncio
async def test_llamaindex_adapter_get_all(tmp_path):
    """EngramChatMemory.get_all() returns all memories."""
    from engram.integrations.llamaindex import EngramChatMemory

    memory = EngramChatMemory(
        db_path=str(tmp_path / "llamaindex_test2.db"),
        user_id="u1",
    )

    memory.put("Fact one", memory_type="fact")
    memory.put("Fact two", memory_type="fact")

    all_memories = memory.get_all()
    assert len(all_memories) == 2

    memory.close()


@pytest.mark.asyncio
async def test_llamaindex_adapter_smart_put(tmp_path):
    """EngramChatMemory.smart_put() auto-classifies."""
    from engram.integrations.llamaindex import EngramChatMemory

    memory = EngramChatMemory(
        db_path=str(tmp_path / "llamaindex_test3.db"),
        user_id="u1",
    )

    ids = memory.smart_put("I prefer dark mode for coding")
    assert len(ids) >= 1

    memory.close()


@pytest.mark.asyncio
async def test_llamaindex_adapter_reset(tmp_path):
    """EngramChatMemory.reset() clears user memories."""
    from engram.integrations.llamaindex import EngramChatMemory

    memory = EngramChatMemory(
        db_path=str(tmp_path / "llamaindex_test4.db"),
        user_id="u1",
    )

    memory.put("Some fact")
    memory.reset()

    all_memories = memory.get_all()
    assert len(all_memories) == 0

    memory.close()


# ── ScoreExplanation Model Tests ────────────────────────────────────


def test_score_explanation_defaults():
    """ScoreExplanation has sensible defaults."""
    exp = ScoreExplanation()
    assert exp.memory_id == ""
    assert exp.final_score == 0.0
    assert exp.vector_rank is None
    assert exp.keyword_rank is None
    assert exp.rrf_score == 0.0
    assert exp.recency_factor == 0.0
    assert exp.summary == ""


def test_score_explanation_with_values():
    """ScoreExplanation stores values correctly."""
    exp = ScoreExplanation(
        memory_id="abc123",
        final_score=0.85,
        vector_rank=2,
        vector_rrf_contribution=0.008,
        keyword_rank=0,
        keyword_rrf_contribution=0.005,
        rrf_score=0.013,
        recency_factor=0.95,
        recency_contribution=0.19,
        age_days=0.5,
        summary="vector rank #3 | keyword rank #1 | recency 0.95",
    )
    assert exp.memory_id == "abc123"
    assert exp.final_score == 0.85
    assert exp.vector_rank == 2
    assert exp.keyword_rank == 0
