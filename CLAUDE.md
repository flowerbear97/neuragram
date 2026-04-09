# CLAUDE.md

## Project Overview

Neuragram (package name: `neuragram`, import name: `neuragram`) is a lightweight, framework-agnostic memory layer for AI agents. Built on SQLite + sqlite-vec + FTS5 with no external services required. Think "the SQLite of agent memory."

**Design philosophy**: Zero external service deps, LLM optional (rule-based fallback for all intelligent features), progressive enhancement via optional extras.

## Quick Reference

```bash
# Install
pip install -e ".[dev]"          # development install
pip install -e ".[all]"          # all optional deps

# Test
pytest                           # run all tests
pytest tests/test_client.py      # run specific test file
pytest -x                        # stop on first failure

# Lint & Type Check
ruff check src/ tests/           # lint
ruff format src/ tests/          # format
mypy src/neuragram/              # type check

# Coverage
coverage run -m pytest && coverage report

# Run servers
neuragram-mcp --db-path ./memory.db                  # MCP server (stdio)
neuragram-api --db-path ./memory.db --port 8100       # REST API server
```

## Architecture

```
src/neuragram/
├── client.py              # AgentMemory — main entry point (Facade pattern)
├── core/
│   ├── models.py          # Memory, ScoredMemory, MemoryType, MemoryStatus, ScoreExplanation
│   ├── config.py          # EngramConfig dataclass with validation
│   ├── filters.py         # MemoryFilter for query scoping
│   ├── exceptions.py      # Exception hierarchy (EngramError base)
│   ├── access.py          # RBAC: AccessPolicy, AccessLevel (NONE/READ/WRITE/ADMIN)
│   └── telemetry.py       # OpenTelemetry with No-Op fallback pattern
├── store/
│   ├── base.py            # BaseMemoryStore ABC (vector_search, keyword_search, etc.)
│   ├── sqlite.py          # SQLiteMemoryStore — 3-index sync (relational + FTS5 + sqlite-vec)
│   └── registry.py        # Factory: create_store()
├── retrieval/
│   ├── engine.py          # RetrievalEngine — hybrid search pipeline with explain()
│   └── scoring.py         # RRF fusion, recency boost (exponential decay), deduplication
├── processing/
│   ├── embeddings.py      # BaseEmbeddingProvider → Null/Local/OpenAI
│   ├── llm.py             # BaseLLMProvider → OpenAI/Ollama/Callable + complete_json()
│   ├── classifier.py      # MemoryClassifier — LLM or regex-based type/importance classification
│   ├── extraction.py      # MemoryExtractor — extract structured memories from conversations (LLM required)
│   ├── conflict.py        # ConflictDetector — detect contradictions + 4 resolution strategies
│   └── merger.py          # MemoryMerger — consolidate similar memories via clustering + summarization
├── lifecycle/
│   ├── decay.py           # DecayManager — TTL expiration + inactivity archival
│   ├── forgetting.py      # ForgettingManager — GDPR-compliant deletion (soft/hard)
│   └── worker.py          # LifecycleWorker — background asyncio.Task for periodic maintenance
├── integrations/
│   ├── langchain.py       # EngramMemory(BaseMemory) adapter
│   └── llamaindex.py      # EngramChatMemory adapter
└── server/
    ├── mcp.py             # FastMCP server — 6 tools (remember, recall, smart_remember, forget, list, stats)
    └── api.py             # FastAPI server — 11 RESTful endpoints
```

## Code Conventions

- Python 3.9+ compatibility required (no 3.10+ syntax like `match` or `X | Y` unions)
- Ruff for linting/formatting: line length 100, rules `E, F, I, N, W, UP`
- Async-first: store operations are async, client provides sync wrappers via `_run_async()`
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- Test fixtures use in-memory SQLite (`:memory:`) — no test DB files
- Build system: Hatchling
- All dataclasses use `from __future__ import annotations` for 3.9 compat

## Key Patterns

- **Facade**: `AgentMemory` is the single public API — all user-facing operations go through it
- **Dual-track intelligence**: Every LLM-powered feature (classify, conflict detect, merge) has a rule-based fallback. Pattern: `if llm: try llm; except: pass; return rules()`
- **Pluggable providers**: Embedding (`BaseEmbeddingProvider`) and LLM (`BaseLLMProvider`) are ABC-based, with factory functions (`create_embedding_provider`, `create_llm_provider`)
- **Graceful degradation**: sqlite-vec load failure → vector-less retrieval; FTS5 unavailable → keyword-less retrieval; no LLM → rule-based classification. Both `_vec_available` and `_fts_available` flags gate index operations
- **No-Op pattern**: Telemetry uses no-op classes when OTel is not installed (zero overhead)
- **Sync/async bridge**: `_run_async()` in `client.py` handles nested event loops (Jupyter) via ThreadPoolExecutor
- **Multi-tenancy**: `namespace` + `user_id` + `agent_id` scoping on all operations
- **Three-index consistency**: insert/update/delete operations keep relational table, FTS5, and sqlite-vec in sync within transactions
- **Write serialization**: All store write operations are serialized via `asyncio.Lock` (`_write_lock`) to prevent "database is locked" errors with the single aiosqlite connection. The lock is lazily created in `initialize()` (not `__init__`) for Python 3.9 compatibility
- **Access enforcement**: `AgentMemory._enforce()` checks `AccessPolicy` before every operation. READ for queries, WRITE for mutations, ADMIN for lifecycle/delete. Disabled by default (`AccessPolicy.enabled=False`)
- **Schema migration**: `_schema_meta` table tracks schema version. `initialize()` applies pending migrations from `_MIGRATIONS` dict in order. Current schema version: 1
- **Atomic batch insert**: `batch_insert()` runs all inserts in a single transaction; any failure rolls back the entire batch

## Important Implementation Details

- Store uses WAL mode, 64MB page cache, 256MB mmap for performance (`store/sqlite.py`)
- LRU cache (OrderedDict, size 256) for `get()` hot path; invalidated on update/delete
- RRF fusion with default weights: vector 0.5 / keyword 0.3 / recency 0.2 (configurable via `EngramConfig`)
- Recency boost uses exponential decay with 7-day half-life
- `search()` auto-touches accessed memories, creating a "use it or lose it" reinforcement effect
- `complete_json()` auto-strips markdown code fences from LLM responses
- Conflict detection threshold: similarity ≥ 0.85 (rule-based), ≥ 0.75 (vector search candidates)
- Embedding serialization uses `struct.pack` for sqlite-vec compatibility
- `stats()` uses a single SQL query with CASE WHEN aggregation (not N separate COUNT queries)
- `_insert_one()` is the shared workhorse for both `insert()` and `batch_insert()` — no lock, no commit; callers handle those
- FTS5-dependent tests use `skip_no_fts5` marker from `tests/conftest.py`
- `AgentMemory` accepts `actor_id` and `access_policy` params for RBAC
