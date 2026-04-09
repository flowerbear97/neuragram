"""Abstract base class for memory storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from neuragram.core.filters import MemoryFilter
from neuragram.core.models import Memory, MemoryVersion, ScoredMemory, StoreStats


class BaseMemoryStore(ABC):
    """Interface that all storage backends must implement.

    Every method is async-first. Synchronous wrappers live in the client layer.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables / indexes if they don't exist. Must be idempotent."""

    @abstractmethod
    async def insert(self, memory: Memory) -> str:
        """Persist a new memory. Returns the memory ID.

        Raises StoreError if a memory with the same ID already exists.
        """

    @abstractmethod
    async def batch_insert(self, memories: list[Memory]) -> list[str]:
        """Insert multiple memories in a single transaction."""

    @abstractmethod
    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID, or None if not found."""

    @abstractmethod
    async def list_memories(
        self,
        filters: MemoryFilter,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories matching the given filters with pagination."""

    @abstractmethod
    async def vector_search(
        self,
        embedding: list[float],
        filters: MemoryFilter,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """Find the top-K most similar memories by vector distance."""

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        filters: MemoryFilter,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """Full-text keyword search using FTS index."""

    @abstractmethod
    async def update(self, memory_id: str, **fields: object) -> Memory:
        """Update fields on an existing memory.

        - Increments version.
        - Saves the previous version to history.
        - Raises MemoryNotFoundError if the ID doesn't exist.
        """

    @abstractmethod
    async def touch(self, memory_id: str) -> None:
        """Update last_accessed_at and increment access_count."""

    @abstractmethod
    async def delete(self, memory_id: str, hard: bool = False) -> bool:
        """Delete a memory.

        soft (default): sets status to DELETED.
        hard: physically removes the row and associated vectors / FTS entries.

        Returns True if a memory was found and deleted.
        """

    @abstractmethod
    async def delete_by_filter(
        self, filters: MemoryFilter, hard: bool = False
    ) -> int:
        """Bulk-delete memories matching filters. Returns count deleted."""

    @abstractmethod
    async def expire_stale(self) -> int:
        """Mark memories past their expires_at as EXPIRED. Returns count."""

    @abstractmethod
    async def archive_inactive(self, max_age_days: int) -> int:
        """Mark memories not accessed for max_age_days as ARCHIVED. Returns count."""

    @abstractmethod
    async def get_versions(self, memory_id: str) -> list[MemoryVersion]:
        """Return the version history for a memory (oldest first)."""

    @abstractmethod
    async def stats(self) -> StoreStats:
        """Return aggregate statistics about the store."""

    @abstractmethod
    async def ping(self) -> bool:
        """Health check. Returns True if the backend is reachable."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources (connections, file handles, etc.)."""
