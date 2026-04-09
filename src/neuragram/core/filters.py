"""Memory query filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import MemoryStatus, MemoryType


@dataclass
class MemoryFilter:
    """Filter criteria for listing and searching memories.

    All specified fields are combined with AND logic.
    Only memories matching *all* non-None criteria are returned.

    Attributes:
        user_id: Filter by owner.
        agent_id: Filter by creating agent.
        namespace: Filter by logical namespace.
        types: Include only these memory types (empty = all types).
        statuses: Include only these statuses. Defaults to [ACTIVE].
        tags: Include only memories that have ALL of these tags.
        created_after: Include only memories created after this time.
        created_before: Include only memories created before this time.
        min_confidence: Minimum confidence threshold.
        min_importance: Minimum importance threshold.
    """

    user_id: str | None = None
    agent_id: str | None = None
    namespace: str | None = None
    types: list[MemoryType] = field(default_factory=list)
    statuses: list[MemoryStatus] = field(
        default_factory=lambda: [MemoryStatus.ACTIVE]
    )
    tags: list[str] = field(default_factory=list)
    created_after: datetime | None = None
    created_before: datetime | None = None
    min_confidence: float | None = None
    min_importance: float | None = None
