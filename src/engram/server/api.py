"""Engram REST API — FastAPI-based HTTP service.

Provides a RESTful HTTP API for Engram's memory operations,
enabling language-agnostic access and independent deployment.

Endpoints:
    POST   /memories          — Store a new memory
    GET    /memories           — List memories with filters
    GET    /memories/{id}      — Get a specific memory
    PUT    /memories/{id}      — Update a memory
    DELETE /memories/{id}      — Delete a memory
    POST   /memories/search    — Hybrid search
    POST   /memories/smart     — Smart remember (auto-classify + conflict detect)
    POST   /conversations      — Process conversation and extract memories
    POST   /memories/consolidate — Merge similar memories
    GET    /stats              — Store statistics
    GET    /health             — Health check

Usage::

    # Start the server
    python -m engram.server.api

    # Or programmatically
    from engram.server.api import create_app
    app = create_app(db_path="./memory.db")

Requires: pip install engram[api]
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def create_app(
    db_path: str = "./engram.db",
    embedding: str = "none",
    embedding_model: str = "",
    llm: str | None = None,
    llm_model: str = "",
    **kwargs: Any,
) -> Any:
    """Create a FastAPI application with Engram endpoints.

    Args:
        db_path: SQLite database path.
        embedding: Embedding provider name.
        embedding_model: Model for the embedding provider.
        llm: LLM provider name (optional).
        llm_model: Model for the LLM provider.
        **kwargs: Additional options passed to AgentMemory.

    Returns:
        A configured FastAPI application.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "FastAPI not installed. Run: pip install engram[api]"
        ) from exc

    from engram.client import AgentMemory
    from engram.core.exceptions import EngramError, MemoryNotFoundError

    app = FastAPI(
        title="Engram Memory API",
        description="RESTful API for Engram agent memory system",
        version="0.3.0",
    )

    # Shared AgentMemory instance
    memory_store: dict[str, AgentMemory | None] = {"instance": None}

    def get_memory() -> AgentMemory:
        if memory_store["instance"] is None:
            memory_store["instance"] = AgentMemory(
                db_path=db_path,
                embedding=embedding,
                embedding_model=embedding_model,
                llm=llm,
                llm_model=llm_model,
                **kwargs,
            )
        return memory_store["instance"]

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        mem = memory_store.get("instance")
        if mem is not None:
            await mem.aclose()

    # ── Request/Response Models ─────────────────────────────────────

    class RememberRequest(BaseModel):
        content: str
        user_id: str = ""
        agent_id: str = ""
        namespace: str = "default"
        memory_type: str = Field(default="fact", alias="type")
        importance: float = 0.5
        confidence: float = 1.0
        tags: list[str] = Field(default_factory=list)

        class Config:
            populate_by_name = True

    class SmartRememberRequest(BaseModel):
        content: str
        user_id: str = ""
        agent_id: str = ""
        namespace: str = "default"

    class SearchRequest(BaseModel):
        query: str
        user_id: str | None = None
        namespace: str | None = None
        top_k: int = 10

    class UpdateRequest(BaseModel):
        content: str | None = None
        importance: float | None = None
        confidence: float | None = None
        tags: list[str] | None = None

    class ConversationRequest(BaseModel):
        messages: list[dict[str, str]]
        user_id: str = ""
        agent_id: str = ""
        namespace: str = "default"

    class ConsolidateRequest(BaseModel):
        user_id: str | None = None
        namespace: str | None = None
        similarity_threshold: float = 0.80

    class MemoryResponse(BaseModel):
        id: str
        content: str
        type: str
        user_id: str
        namespace: str
        importance: float
        confidence: float
        tags: list[str]
        created_at: str
        updated_at: str

    class ScoredMemoryResponse(BaseModel):
        id: str
        content: str
        type: str
        score: float
        importance: float
        confidence: float
        tags: list[str]
        created_at: str

    # ── Endpoints ───────────────────────────────────────────────────

    @app.post("/memories", status_code=201)
    async def create_memory(request: RememberRequest) -> dict[str, Any]:
        """Store a new memory."""
        mem = get_memory()
        memory_id = await mem.aremember(
            content=request.content,
            user_id=request.user_id,
            agent_id=request.agent_id,
            namespace=request.namespace,
            type=request.memory_type,
            importance=request.importance,
            confidence=request.confidence,
            tags=request.tags,
        )
        return {"memory_id": memory_id, "status": "created"}

    @app.get("/memories")
    async def list_memories(
        user_id: str | None = None,
        namespace: str | None = None,
        memory_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List memories with optional filters."""
        mem = get_memory()
        types = [memory_type] if memory_type else None
        memories = await mem.alist(
            user_id=user_id,
            namespace=namespace,
            types=types,
            limit=limit,
            offset=offset,
        )
        items = [
            MemoryResponse(
                id=m.id,
                content=m.content,
                type=m.memory_type.value,
                user_id=m.user_id,
                namespace=m.namespace,
                importance=m.importance,
                confidence=m.confidence,
                tags=m.tags,
                created_at=m.created_at.isoformat(),
                updated_at=m.updated_at.isoformat(),
            ).dict()
            for m in memories
        ]
        return {"memories": items, "count": len(items)}

    @app.get("/memories/{memory_id}")
    async def get_memory_by_id(memory_id: str) -> dict[str, Any]:
        """Get a specific memory by ID."""
        mem = get_memory()
        memory = await mem.aget(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return MemoryResponse(
            id=memory.id,
            content=memory.content,
            type=memory.memory_type.value,
            user_id=memory.user_id,
            namespace=memory.namespace,
            importance=memory.importance,
            confidence=memory.confidence,
            tags=memory.tags,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        ).dict()

    @app.put("/memories/{memory_id}")
    async def update_memory(memory_id: str, request: UpdateRequest) -> dict[str, Any]:
        """Update a memory's fields."""
        mem = get_memory()
        update_fields: dict[str, Any] = {}
        if request.content is not None:
            update_fields["content"] = request.content
        if request.importance is not None:
            update_fields["importance"] = request.importance
        if request.confidence is not None:
            update_fields["confidence"] = request.confidence
        if request.tags is not None:
            update_fields["tags"] = request.tags

        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        try:
            updated = await mem.aupdate(memory_id, **update_fields)
            return {"memory_id": updated.id, "version": updated.version, "status": "updated"}
        except MemoryNotFoundError:
            raise HTTPException(status_code=404, detail="Memory not found")

    @app.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str, hard: bool = False) -> dict[str, Any]:
        """Delete a memory."""
        mem = get_memory()
        count = await mem.aforget(memory_id=memory_id, hard=hard)
        if count == 0:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"deleted": count, "hard": hard}

    @app.post("/memories/search")
    async def search_memories(request: SearchRequest) -> dict[str, Any]:
        """Hybrid search for relevant memories."""
        mem = get_memory()
        results = await mem.arecall(
            query=request.query,
            user_id=request.user_id,
            namespace=request.namespace,
            top_k=request.top_k,
        )
        items = [
            ScoredMemoryResponse(
                id=s.memory.id,
                content=s.memory.content,
                type=s.memory.memory_type.value,
                score=round(s.score, 4),
                importance=s.memory.importance,
                confidence=s.memory.confidence,
                tags=s.memory.tags,
                created_at=s.memory.created_at.isoformat(),
            ).dict()
            for s in results
        ]
        return {"results": items, "count": len(items)}

    @app.post("/memories/smart", status_code=201)
    async def smart_remember(request: SmartRememberRequest) -> dict[str, Any]:
        """Smart remember with auto-classification and conflict detection."""
        mem = get_memory()
        ids = await mem.asmart_remember(
            content=request.content,
            user_id=request.user_id,
            agent_id=request.agent_id,
            namespace=request.namespace,
        )
        return {"memory_ids": ids, "count": len(ids), "status": "created"}

    @app.post("/conversations", status_code=201)
    async def process_conversation(request: ConversationRequest) -> dict[str, Any]:
        """Extract and store memories from a conversation."""
        mem = get_memory()
        try:
            ids = await mem.aprocess_conversation(
                messages=request.messages,
                user_id=request.user_id,
                agent_id=request.agent_id,
                namespace=request.namespace,
            )
            return {"memory_ids": ids, "count": len(ids), "status": "processed"}
        except EngramError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/memories/consolidate")
    async def consolidate_memories(request: ConsolidateRequest) -> dict[str, Any]:
        """Merge similar memories into summaries."""
        mem = get_memory()
        result = await mem.aconsolidate(
            user_id=request.user_id,
            namespace=request.namespace,
            similarity_threshold=request.similarity_threshold,
        )
        return result

    @app.get("/stats")
    async def get_stats() -> dict[str, Any]:
        """Get store statistics."""
        mem = get_memory()
        stats = await mem.astats()
        return {
            "total": stats.total_memories,
            "active": stats.active_memories,
            "archived": stats.archived_memories,
            "expired": stats.expired_memories,
        }

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "service": "engram"}

    return app


def main() -> None:
    """Entry point for running Engram REST API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Engram REST API Server")
    parser.add_argument("--db-path", default="./engram.db", help="SQLite database path")
    parser.add_argument("--embedding", default="none", help="Embedding provider")
    parser.add_argument("--embedding-model", default="", help="Embedding model name")
    parser.add_argument("--llm", default=None, help="LLM provider")
    parser.add_argument("--llm-model", default="", help="LLM model name")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: pip install engram[api]")

    app = create_app(
        db_path=args.db_path,
        embedding=args.embedding,
        embedding_model=args.embedding_model,
        llm=args.llm,
        llm_model=args.llm_model,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
