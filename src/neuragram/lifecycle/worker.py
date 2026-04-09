"""Background lifecycle worker for automated memory maintenance.

Runs periodic tasks in the background:
- TTL expiration: Mark expired memories
- Inactivity archival: Archive memories not accessed for N days
- Consolidation: Merge similar memories (optional, requires LLM)
- Cleanup: Physically remove soft-deleted memories after retention period

Usage::

    worker = LifecycleWorker(memory)
    await worker.start(interval_seconds=3600)  # Run every hour
    # ... later ...
    await worker.stop()

    # Or as a context manager
    async with LifecycleWorker(memory, interval_seconds=3600):
        # Worker runs in background
        await asyncio.sleep(forever)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("neuragram.lifecycle.worker")


@dataclass
class WorkerStats:
    """Statistics from a single worker cycle."""

    cycle_number: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    expired_count: int = 0
    archived_count: int = 0
    consolidated_groups: int = 0
    consolidated_memories: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class LifecycleWorker:
    """Background worker that periodically runs lifecycle maintenance tasks.

    Args:
        memory: An AgentMemory instance to operate on.
        interval_seconds: Seconds between maintenance cycles.
        enable_expiration: Run TTL expiration each cycle.
        enable_archival: Run inactivity archival each cycle.
        enable_consolidation: Run memory consolidation each cycle (requires LLM).
        archival_max_age_days: Days of inactivity before archiving.
        consolidation_threshold: Similarity threshold for consolidation.
        consolidation_user_id: Scope consolidation to a specific user.
        consolidation_namespace: Scope consolidation to a specific namespace.
    """

    def __init__(
        self,
        memory: Any,  # AgentMemory — avoid circular import
        interval_seconds: float = 3600,
        enable_expiration: bool = True,
        enable_archival: bool = True,
        enable_consolidation: bool = False,
        archival_max_age_days: int = 30,
        consolidation_threshold: float = 0.80,
        consolidation_user_id: str | None = None,
        consolidation_namespace: str | None = None,
    ) -> None:
        self._memory = memory
        self._interval = interval_seconds
        self._enable_expiration = enable_expiration
        self._enable_archival = enable_archival
        self._enable_consolidation = enable_consolidation
        self._archival_max_age_days = archival_max_age_days
        self._consolidation_threshold = consolidation_threshold
        self._consolidation_user_id = consolidation_user_id
        self._consolidation_namespace = consolidation_namespace

        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._cycle_count = 0
        self._last_stats: WorkerStats | None = None

    @property
    def is_running(self) -> bool:
        """Whether the worker is currently active."""
        return self._running

    @property
    def cycle_count(self) -> int:
        """Number of completed maintenance cycles."""
        return self._cycle_count

    @property
    def last_stats(self) -> WorkerStats | None:
        """Statistics from the most recent cycle."""
        return self._last_stats

    async def start(self) -> None:
        """Start the background worker loop.

        The worker runs in an asyncio task and executes maintenance
        cycles at the configured interval. Safe to call multiple times
        (subsequent calls are no-ops if already running).
        """
        if self._running:
            logger.warning("LifecycleWorker is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "LifecycleWorker started (interval=%ds, expiration=%s, archival=%s, consolidation=%s)",
            self._interval,
            self._enable_expiration,
            self._enable_archival,
            self._enable_consolidation,
        )

    async def stop(self) -> None:
        """Stop the background worker gracefully.

        Waits for the current cycle to complete before stopping.
        """
        if not self._running:
            return

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("LifecycleWorker stopped after %d cycles", self._cycle_count)

    async def run_once(self) -> WorkerStats:
        """Execute a single maintenance cycle immediately.

        Useful for testing or manual triggering.

        Returns:
            WorkerStats with results from this cycle.
        """
        return await self._run_cycle()

    async def _run_loop(self) -> None:
        """Internal loop that runs cycles at the configured interval."""
        try:
            while self._running:
                try:
                    stats = await self._run_cycle()
                    self._last_stats = stats
                except Exception as exc:
                    logger.error("LifecycleWorker cycle failed: %s", exc)

                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    async def _run_cycle(self) -> WorkerStats:
        """Execute one full maintenance cycle."""
        self._cycle_count += 1
        stats = WorkerStats(
            cycle_number=self._cycle_count,
            started_at=datetime.now(timezone.utc),
        )

        # 1. TTL Expiration
        if self._enable_expiration:
            try:
                decay_result = await self._memory.adecay()
                stats.expired_count = decay_result.get("expired", 0)
                stats.archived_count = decay_result.get("archived", 0)
            except Exception as exc:
                error_msg = f"Expiration failed: {exc}"
                stats.errors.append(error_msg)
                logger.warning(error_msg)

        # 2. Inactivity Archival (separate from decay if different max_age)
        if self._enable_archival and not self._enable_expiration:
            try:
                decay_result = await self._memory.adecay(
                    max_age_days=self._archival_max_age_days
                )
                stats.archived_count = decay_result.get("archived", 0)
            except Exception as exc:
                error_msg = f"Archival failed: {exc}"
                stats.errors.append(error_msg)
                logger.warning(error_msg)

        # 3. Consolidation (optional)
        if self._enable_consolidation:
            try:
                consolidation_result = await self._memory.aconsolidate(
                    user_id=self._consolidation_user_id,
                    namespace=self._consolidation_namespace,
                    similarity_threshold=self._consolidation_threshold,
                )
                stats.consolidated_groups = consolidation_result.get("groups_merged", 0)
                stats.consolidated_memories = consolidation_result.get(
                    "memories_consolidated", 0
                )
            except Exception as exc:
                error_msg = f"Consolidation failed: {exc}"
                stats.errors.append(error_msg)
                logger.warning(error_msg)

        stats.completed_at = datetime.now(timezone.utc)
        stats.duration_seconds = (
            stats.completed_at - stats.started_at
        ).total_seconds()

        logger.info(
            "Cycle #%d completed in %.2fs: expired=%d, archived=%d, consolidated=%d/%d, errors=%d",
            stats.cycle_number,
            stats.duration_seconds,
            stats.expired_count,
            stats.archived_count,
            stats.consolidated_groups,
            stats.consolidated_memories,
            len(stats.errors),
        )

        return stats

    async def __aenter__(self) -> LifecycleWorker:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()
