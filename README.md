## Engram

**English** | [中文](README_CN.md)

Lightweight, framework-agnostic memory layer for AI agents. Built on SQLite + sqlite-vec + FTS5 — no external services required.

```bash
pip install engram
```

```python
from engram import AgentMemory

mem = AgentMemory(db_path="./memory.db")
mem.remember("User prefers concise code explanations", user_id="u1", type="preference")
results = mem.recall("What style does the user prefer?", user_id="u1")
print(results[0].memory.content)
mem.close()
```

## Comparison

| | Engram | Mem0 | Letta | Graphiti |
|---|---|---|---|---|
| Install | `pip install` | `pip install` | Docker + Server | pip + Neo4j |
| External Deps | None | Vector DB + LLM | PG + Server + LLM | Graph DB + LLM |
| Framework Lock-in | None | None | Letta runtime | None |
| Memory Lifecycle | Built-in | None | Agent self-managed | Partial |
| LLM Required | No | Yes | Yes | Yes |

## Features

### Storage & Retrieval
- **Hybrid search** — vector similarity + FTS5 keyword + recency scoring, fused via Reciprocal Rank Fusion
- **Explainable ranking** — `explain()` returns full score breakdown per result (vector rank, keyword rank, RRF contribution, recency factor)
- **LRU cache** — hot-path memory lookups served from in-memory cache
- **Performance tuned** — WAL mode, 64 MB page cache, mmap, batch operations in single transactions

### Memory Intelligence
- **Smart remember** — auto-classifies memory type, importance, and confidence (rule-based or LLM-enhanced)
- **Conversation extraction** — extracts structured memories from chat messages (requires LLM)
- **Conflict detection & resolution** — detects contradicting memories and resolves automatically
- **Consolidation** — merges similar memories into summaries to reduce noise

### Lifecycle Management
- **TTL expiration** — memories with `expires_at` are automatically expired
- **Inactivity archival** — memories not accessed for N days are archived
- **GDPR forgetting** — `forget(user_id="u1")` removes all user data
- **Background worker** — `create_worker()` runs maintenance tasks on a schedule
- **Version history** — every update is versioned; full history via `history()`

### Multi-tenancy & Access Control
- **Isolation** — `namespace` + `user_id` + `agent_id` scoping on all operations
- **Role-based access** — `AccessLevel.READ / WRITE / ADMIN` with namespace and user scoping
- **Namespace statistics** — per-namespace memory distribution via `namespace_stats()`

### Integrations
- **MCP Server** — `engram-mcp` CLI exposes Engram as an MCP tool server (Claude Desktop, Cursor, etc.)
- **REST API** — `engram-api` CLI starts a FastAPI HTTP service with 12 endpoints
- **LangChain** — `EngramMemory` implements `BaseMemory` (save_context / load_memory_variables)
- **LlamaIndex** — `EngramChatMemory` provides put / get / get_all / reset

### Observability
- **OpenTelemetry** — automatic traces, metrics, and spans for all operations; zero-overhead no-op when OTel is not installed

## Usage

### Embeddings

```python
# OpenAI
mem = AgentMemory(db_path="./memory.db", embedding="openai", embedding_model="text-embedding-3-small")

# Local (sentence-transformers)
mem = AgentMemory(db_path="./memory.db", embedding="local")

# No embeddings (keyword-only retrieval)
mem = AgentMemory(db_path="./memory.db")
```

### Smart Remember

```python
from engram import AgentMemory, CallableLLMProvider

async def my_llm(prompt):
    return await call_my_llm(prompt)

mem = AgentMemory(db_path="./memory.db", llm=CallableLLMProvider(my_llm))
ids = mem.smart_remember("User prefers Python over JavaScript")
```

### MCP Server

```bash
engram-mcp --db-path ./memory.db
engram-mcp --db-path ./memory.db --embedding openai
```

### REST API

```bash
engram-api --db-path ./memory.db --port 8080
```

### LangChain

```python
from engram.integrations.langchain import EngramMemory

memory = EngramMemory(db_path="./memory.db", user_id="u1")
memory.save_context({"input": "I prefer concise answers"}, {"output": "Got it!"})
result = memory.load_memory_variables({"input": "answer style"})
```

### LlamaIndex

```python
from engram.integrations.llamaindex import EngramChatMemory

memory = EngramChatMemory(db_path="./memory.db", user_id="u1")
memory.put("User is a Python developer", memory_type="fact")
results = memory.get("programming language")
```

### Explainable Retrieval

```python
for exp in mem.explain("user preferences", user_id="u1"):
    print(exp["summary"])
    # "vector rank #1 (+0.0082) | keyword rank #3 (+0.0048) | recency 0.95 (age 0.5d)"
```

### Access Control

```python
from engram import AccessPolicy, AccessLevel

policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
policy.grant("reader_agent", AccessLevel.READ)
policy.grant("writer_agent", AccessLevel.WRITE, namespace="project_a")
policy.grant("admin_bot", AccessLevel.ADMIN)

mem = AgentMemory(db_path="./memory.db", access_policy=policy)
```

### Background Worker

```python
worker = mem.create_worker(interval_seconds=3600)
await worker.start()
# ...
await worker.stop()
```

## Installation

```bash
pip install engram              # core
pip install engram[openai]      # + OpenAI embeddings
pip install engram[local]       # + sentence-transformers
pip install engram[mcp]         # + MCP server
pip install engram[api]         # + REST API (FastAPI)
pip install engram[langchain]   # + LangChain adapter
pip install engram[llamaindex]  # + LlamaIndex adapter
pip install engram[telemetry]   # + OpenTelemetry
pip install engram[all]         # everything
```

## Architecture

```
AgentMemory (client.py)
├── Store Layer          SQLite + sqlite-vec + FTS5
├── Retrieval Engine     RRF fusion, recency boost, deduplication
├── Processing           extraction → classification → conflict → merge
├── Lifecycle            decay, forgetting, background worker
├── Access Control       role-based, namespace-scoped
├── Telemetry            OpenTelemetry traces + metrics
└── Integrations         MCP Server, REST API, LangChain, LlamaIndex
```

## License

Apache-2.0
