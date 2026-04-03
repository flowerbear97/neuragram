"""Engram MCP Server — expose memory operations as MCP tools.

This module implements a Model Context Protocol (MCP) server that allows
AI assistants (Claude, Cursor, etc.) to directly interact with Engram's
memory system through standardized tool calls.

Exposed tools:
    - engram_remember: Store a new memory
    - engram_recall: Search and retrieve memories
    - engram_smart_remember: Auto-classify and store with conflict detection
    - engram_forget: Delete memories
    - engram_list: List memories with filters
    - engram_stats: Get store statistics

Usage::

    # As a standalone server (stdio transport for Claude Desktop)
    python -m engram.server.mcp

    # Or programmatically
    from engram.server.mcp import create_mcp_server
    server = create_mcp_server(db_path="./memory.db")
    server.run()

Requires: pip install neuragram[mcp]
"""

from __future__ import annotations

import json
from typing import Any


def create_mcp_server(
    db_path: str = "./engram.db",
    embedding: str = "none",
    embedding_model: str = "",
    llm: str | None = None,
    llm_model: str = "",
    server_name: str = "Engram Memory",
    **kwargs: Any,
) -> Any:
    """Create and configure an MCP server with Engram tools.

    Args:
        db_path: SQLite database path.
        embedding: Embedding provider name.
        embedding_model: Model for the embedding provider.
        llm: LLM provider name (optional, for smart features).
        llm_model: Model for the LLM provider.
        server_name: Display name for the MCP server.
        **kwargs: Additional options passed to AgentMemory.

    Returns:
        A configured FastMCP server instance.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "MCP SDK not installed. Run: pip install neuragram[mcp]"
        ) from exc

    from engram.client import AgentMemory

    mcp = FastMCP(server_name)

    # Lazy-initialized shared AgentMemory instance
    _memory_instance: dict[str, AgentMemory | None] = {"instance": None}

    def _get_memory() -> AgentMemory:
        if _memory_instance["instance"] is None:
            mem = AgentMemory(
                db_path=db_path,
                embedding=embedding,
                embedding_model=embedding_model,
                llm=llm,
                llm_model=llm_model,
                **kwargs,
            )
            _memory_instance["instance"] = mem
        return _memory_instance["instance"]

    @mcp.tool()
    async def engram_remember(
        content: str,
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: str = "",
    ) -> str:
        """Store a new memory in Engram.

        Args:
            content: The text content to remember.
            user_id: Owner of this memory.
            agent_id: The agent creating this memory.
            namespace: Logical grouping for the memory.
            memory_type: One of: fact, preference, episode, procedure, plan_state.
            importance: Importance score from 0.0 to 1.0.
            tags: Comma-separated tags (e.g. "python,coding").

        Returns:
            JSON with the stored memory ID.
        """
        mem = _get_memory()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        memory_id = await mem.aremember(
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            type=memory_type,
            importance=importance,
            tags=tag_list,
        )
        return json.dumps({"memory_id": memory_id, "status": "stored"})

    @mcp.tool()
    async def engram_recall(
        query: str,
        user_id: str = "",
        namespace: str = "",
        top_k: int = 5,
    ) -> str:
        """Search and retrieve relevant memories.

        Args:
            query: Natural language search query.
            user_id: Filter by user ID.
            namespace: Filter by namespace.
            top_k: Maximum number of results to return.

        Returns:
            JSON array of matching memories with scores.
        """
        mem = _get_memory()
        results = await mem.arecall(
            query=query,
            user_id=user_id or None,
            namespace=namespace or None,
            top_k=top_k,
        )
        items = []
        for scored in results:
            items.append({
                "id": scored.memory.id,
                "content": scored.memory.content,
                "type": scored.memory.memory_type.value,
                "score": round(scored.score, 4),
                "importance": scored.memory.importance,
                "confidence": scored.memory.confidence,
                "tags": scored.memory.tags,
                "created_at": scored.memory.created_at.isoformat(),
            })
        return json.dumps({"results": items, "count": len(items)})

    @mcp.tool()
    async def engram_smart_remember(
        content: str,
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
    ) -> str:
        """Intelligently store a memory with auto-classification and conflict detection.

        Automatically determines the memory type, importance, and confidence.
        Detects and resolves conflicts with existing memories.

        Args:
            content: The text content to remember.
            user_id: Owner of this memory.
            agent_id: The agent creating this memory.
            namespace: Logical grouping.

        Returns:
            JSON with stored memory IDs and classification info.
        """
        mem = _get_memory()
        ids = await mem.asmart_remember(
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
        )
        return json.dumps({"memory_ids": ids, "count": len(ids), "status": "stored"})

    @mcp.tool()
    async def engram_forget(
        memory_id: str = "",
        user_id: str = "",
        hard: bool = False,
    ) -> str:
        """Delete memories from Engram.

        Provide either memory_id (delete one) or user_id (delete all for user).

        Args:
            memory_id: Specific memory ID to delete.
            user_id: Delete all memories for this user.
            hard: If true, physically remove (GDPR). If false, soft-delete.

        Returns:
            JSON with deletion count.
        """
        mem = _get_memory()
        if memory_id:
            count = await mem.aforget(memory_id=memory_id, hard=hard)
        elif user_id:
            count = await mem.aforget(user_id=user_id, hard=hard)
        else:
            return json.dumps({"error": "Provide either memory_id or user_id"})
        return json.dumps({"deleted": count, "hard": hard})

    @mcp.tool()
    async def engram_list(
        user_id: str = "",
        namespace: str = "",
        memory_type: str = "",
        limit: int = 20,
    ) -> str:
        """List memories with optional filters.

        Args:
            user_id: Filter by user ID.
            namespace: Filter by namespace.
            memory_type: Filter by type (fact, preference, episode, procedure, plan_state).
            limit: Maximum number of results.

        Returns:
            JSON array of memories.
        """
        mem = _get_memory()
        types = [memory_type] if memory_type else None
        memories = await mem.alist(
            user_id=user_id or None,
            namespace=namespace or None,
            types=types,
            limit=limit,
        )
        items = []
        for memory in memories:
            items.append({
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type.value,
                "importance": memory.importance,
                "confidence": memory.confidence,
                "tags": memory.tags,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            })
        return json.dumps({"memories": items, "count": len(items)})

    @mcp.tool()
    async def engram_stats() -> str:
        """Get memory store statistics.

        Returns:
            JSON with total count, active count, and other stats.
        """
        mem = _get_memory()
        stats = await mem.astats()
        return json.dumps({
            "total": stats.total_memories,
            "active": stats.active_memories,
            "archived": stats.archived_memories,
            "expired": stats.expired_memories,
        })

    return mcp


def main() -> None:
    """Entry point for running Engram as a standalone MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Engram MCP Server")
    parser.add_argument("--db-path", default="./engram.db", help="SQLite database path")
    parser.add_argument("--embedding", default="none", help="Embedding provider")
    parser.add_argument("--embedding-model", default="", help="Embedding model name")
    parser.add_argument("--llm", default=None, help="LLM provider (openai, ollama)")
    parser.add_argument("--llm-model", default="", help="LLM model name")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="MCP transport type",
    )
    args = parser.parse_args()

    server = create_mcp_server(
        db_path=args.db_path,
        embedding=args.embedding,
        embedding_model=args.embedding_model,
        llm=args.llm,
        llm_model=args.llm_model,
    )
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
