"""Memory decay manager.

Handles two forms of automatic memory lifecycle transitions:

1. **TTL expiration**: Memories with an `expires_at` timestamp that has passed
   are transitioned from ACTIVE → EXPIRED.

2. **Inactivity archival**: Memories that haven't been accessed for a
   configurable number of days are transitioned from ACTIVE → ARCHIVED.

Both operations are idempotent and safe to run repeatedly (e.g. on a schedule).
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.store.base import BaseMemoryStore


@dataclass
class DecayResult:
    """Summary of a decay run."""

    expired: int = 0
    archived: int = 0


class DecayManager:
    """Manages automatic memory lifecycle transitions.

    Args:
        store: The memory store backend.
        max_age_days: Days without access before a memory is archived.
        ttl_enabled: Whether to enforce TTL-based expiration.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        max_age_days: int = 30,
        ttl_enabled: bool = True,
    ) -> None:
        self._store = store
        self._max_age_days = max_age_days
        self._ttl_enabled = ttl_enabled

    async def run_decay(self) -> DecayResult:
        """Execute one decay cycle.

        1. Expire memories past their TTL (if ttl_enabled).
        2. Archive memories not accessed within max_age_days.

        Returns:
            DecayResult with counts of expired and archived memories.
        """
        expired_count = 0
        archived_count = 0

        if self._ttl_enabled:
            expired_count = await self._store.expire_stale()

        archived_count = await self._store.archive_inactive(self._max_age_days)

        return DecayResult(expired=expired_count, archived=archived_count)
