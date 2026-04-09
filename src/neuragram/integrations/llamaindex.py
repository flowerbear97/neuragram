"""LlamaIndex integration for Engram.

Provides an adapter that works with LlamaIndex's memory abstractions,
allowing Engram to serve as a persistent memory store for LlamaIndex
agents and chat engines.

Usage::

    from neuragram.integrations.llamaindex import EngramChatMemory

    memory = EngramChatMemory(db_path="./memory.db", user_id="u1")

    # Get relevant context for a query
    context = memory.get("What does the user prefer?")

    # Store a new interaction
    memory.put("User prefers concise answers")

Requires: pip install neuragram[llamaindex]
"""

from __future__ import annotations

from typing import Any


class EngramChatMemory:
    """LlamaIndex-compatible chat memory backed by Engram.

    Provides a simple get/put interface that maps to LlamaIndex's
    memory patterns for chat engines and agents.

    Args:
        db_path: SQLite database path.
        user_id: User ID for memory scoping.
        agent_id: Agent ID for memory scoping.
        namespace: Namespace for memory grouping.
        top_k: Number of memories to retrieve per query.
        embedding: Embedding provider name.
        llm: LLM provider name (optional).
        **kwargs: Additional options passed to AgentMemory.
    """

    def __init__(
        self,
        db_path: str = "./neuragram.db",
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
        top_k: int = 5,
        embedding: str = "none",
        llm: str | None = None,
        **kwargs: Any,
    ) -> None:
        from neuragram.client import AgentMemory

        self._memory = AgentMemory(
            db_path=db_path,
            embedding=embedding,
            llm=llm,
            **kwargs,
        )
        self._user_id = user_id
        self._agent_id = agent_id
        self._namespace = namespace
        self._top_k = top_k

    def get(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve relevant memories for a query.

        Args:
            query: Natural language query.
            top_k: Override default top_k.

        Returns:
            List of dicts with "content", "type", "score", "metadata".
        """
        results = self._memory.recall(
            query=query,
            user_id=self._user_id or None,
            namespace=self._namespace or None,
            top_k=top_k or self._top_k,
        )
        return [
            {
                "content": s.memory.content,
                "type": s.memory.memory_type.value,
                "score": s.score,
                "metadata": {
                    "id": s.memory.id,
                    "importance": s.memory.importance,
                    "confidence": s.memory.confidence,
                    "tags": s.memory.tags,
                    "created_at": s.memory.created_at.isoformat(),
                },
            }
            for s in results
        ]

    def get_all(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve all memories for this user/namespace scope.

        Args:
            limit: Maximum number of memories to return.

        Returns:
            List of memory dicts.
        """
        memories = self._memory.list(
            user_id=self._user_id or None,
            namespace=self._namespace or None,
            limit=limit,
        )
        return [
            {
                "content": m.content,
                "type": m.memory_type.value,
                "metadata": {
                    "id": m.id,
                    "importance": m.importance,
                    "confidence": m.confidence,
                    "tags": m.tags,
                    "created_at": m.created_at.isoformat(),
                },
            }
            for m in memories
        ]

    def put(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str:
        """Store a new memory.

        Args:
            content: Text content to remember.
            memory_type: Memory type (fact, preference, episode, procedure).
            importance: Importance score [0, 1].
            tags: Optional tags.

        Returns:
            The stored memory ID.
        """
        return self._memory.remember(
            content=content,
            user_id=self._user_id,
            agent_id=self._agent_id,
            namespace=self._namespace,
            type=memory_type,
            importance=importance,
            tags=tags or [],
        )

    def smart_put(self, content: str) -> list[str]:
        """Store with auto-classification and conflict detection.

        Args:
            content: Text content to remember.

        Returns:
            List of stored memory IDs.
        """
        return self._memory.smart_remember(
            content=content,
            user_id=self._user_id,
            agent_id=self._agent_id,
            namespace=self._namespace,
        )

    def delete(self, memory_id: str) -> None:
        """Delete a specific memory."""
        self._memory.forget(memory_id=memory_id)

    def reset(self) -> None:
        """Clear all memories for this user/namespace scope."""
        if self._user_id:
            self._memory.forget(user_id=self._user_id)

    async def aget(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Async version of get."""
        results = await self._memory.arecall(
            query=query,
            user_id=self._user_id or None,
            namespace=self._namespace or None,
            top_k=top_k or self._top_k,
        )
        return [
            {
                "content": s.memory.content,
                "type": s.memory.memory_type.value,
                "score": s.score,
                "metadata": {
                    "id": s.memory.id,
                    "importance": s.memory.importance,
                    "confidence": s.memory.confidence,
                    "tags": s.memory.tags,
                    "created_at": s.memory.created_at.isoformat(),
                },
            }
            for s in results
        ]

    async def aput(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str:
        """Async version of put."""
        return await self._memory.aremember(
            content=content,
            user_id=self._user_id,
            agent_id=self._agent_id,
            namespace=self._namespace,
            type=memory_type,
            importance=importance,
            tags=tags or [],
        )

    def close(self) -> None:
        """Release resources."""
        self._memory.close()

    async def aclose(self) -> None:
        """Async release resources."""
        await self._memory.aclose()
