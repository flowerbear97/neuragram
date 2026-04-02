"""Memory forgetting manager — GDPR-compliant deletion.

Provides controlled deletion of memories at multiple granularities:

- **Single memory**: Forget a specific memory by ID.
- **User-level**: Forget all memories belonging to a user (GDPR "right to be forgotten").
- **Filter-based**: Forget memories matching arbitrary criteria.

Supports both soft delete (status → DELETED, data retained) and
hard delete (physical removal from all indexes).
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.core.filters import MemoryFilter
from engram.core.models import MemoryStatus
from engram.store.base import BaseMemoryStore


@dataclass
class ForgetResult:
    """Summary of a forget operation."""

    deleted_count: int = 0
    hard: bool = False


class ForgettingManager:
    """Manages controlled memory deletion and GDPR compliance.

    Args:
        store: The memory store backend.
    """

    def __init__(self, store: BaseMemoryStore) -> None:
        self._store = store

    async def forget_memory(
        self, memory_id: str, hard: bool = False
    ) -> bool:
        """Delete a single memory by ID.

        Args:
            memory_id: The memory to delete.
            hard: If True, physically remove from all tables/indexes.

        Returns:
            True if the memory was found and deleted.
        """
        return await self._store.delete(memory_id, hard=hard)

    async def forget_user(
        self, user_id: str, hard: bool = False
    ) -> ForgetResult:
        """Delete all memories belonging to a user.

        This is the GDPR "right to be forgotten" operation.

        Args:
            user_id: The user whose memories should be deleted.
            hard: If True, physically remove all data (recommended for GDPR).

        Returns:
            ForgetResult with the count of deleted memories.
        """
        filters = MemoryFilter(
            user_id=user_id,
            statuses=[
                MemoryStatus.ACTIVE,
                MemoryStatus.ARCHIVED,
                MemoryStatus.EXPIRED,
                MemoryStatus.DELETED,
            ],
        )
        count = await self._store.delete_by_filter(filters, hard=hard)
        return ForgetResult(deleted_count=count, hard=hard)

    async def forget_by_filter(
        self, filters: MemoryFilter, hard: bool = False
    ) -> ForgetResult:
        """Delete memories matching arbitrary filter criteria.

        Args:
            filters: Criteria for selecting memories to delete.
            hard: If True, physically remove from all tables/indexes.

        Returns:
            ForgetResult with the count of deleted memories.
        """
        count = await self._store.delete_by_filter(filters, hard=hard)
        return ForgetResult(deleted_count=count, hard=hard)
