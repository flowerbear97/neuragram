"""Core data models for Engram memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Classification of memory content."""

    FACT = "fact"
    EPISODE = "episode"
    PREFERENCE = "preference"
    PROCEDURE = "procedure"
    PLAN_STATE = "plan_state"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    DELETED = "deleted"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Memory:
    """A single unit of agent memory.

    Attributes:
        content: The textual content of this memory.
        memory_type: Classification (fact, episode, preference, etc.).
        status: Lifecycle status (active, archived, expired, deleted).
        user_id: Owner / subject of this memory.
        agent_id: The agent that created this memory.
        namespace: Logical grouping for multi-tenant isolation.
        tags: Free-form labels for categorization.
        metadata: Arbitrary key-value pairs.
        embedding: Vector representation (populated by embedding provider).
        confidence: How confident we are in this memory's accuracy [0, 1].
        importance: How important this memory is [0, 1].
        version: Monotonically increasing version number.
        source: Provenance — where this memory came from.
        expires_at: Optional TTL expiration timestamp.
        created_at: When this memory was first created.
        updated_at: When this memory was last modified.
        last_accessed_at: When this memory was last retrieved / touched.
        access_count: How many times this memory has been accessed.
        id: Unique identifier (auto-generated UUID hex).
    """

    content: str
    memory_type: MemoryType = MemoryType.FACT
    status: MemoryStatus = MemoryStatus.ACTIVE
    user_id: str = ""
    agent_id: str = ""
    namespace: str = "default"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    confidence: float = 1.0
    importance: float = 0.5
    version: int = 1
    source: str = ""
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    last_accessed_at: datetime = field(default_factory=_utcnow)
    access_count: int = 0
    id: str = field(default_factory=_new_id)


@dataclass
class ScoredMemory:
    """A memory paired with a relevance score from retrieval.

    Supports comparison for sorting (higher score = more relevant).
    """

    memory: Memory
    score: float

    def __lt__(self, other: ScoredMemory) -> bool:
        return self.score < other.score

    def __le__(self, other: ScoredMemory) -> bool:
        return self.score <= other.score

    def __gt__(self, other: ScoredMemory) -> bool:
        return self.score > other.score

    def __ge__(self, other: ScoredMemory) -> bool:
        return self.score >= other.score


@dataclass
class MemoryUpdate:
    """Partial update payload for an existing memory.

    Only non-None fields will be applied.
    """

    content: str | None = None
    memory_type: MemoryType | None = None
    metadata: dict[str, Any] | None = None
    tags: list[str] | None = None
    confidence: float | None = None
    importance: float | None = None
    expires_at: datetime | None = None
    source: str | None = None


@dataclass
class MemoryVersion:
    """A historical snapshot of a memory before it was updated."""

    memory_id: str
    version: int
    content: str
    metadata: dict[str, Any]
    updated_at: datetime


@dataclass
class ScoreExplanation:
    """Breakdown of how a memory's retrieval score was computed.

    Provides transparency into the scoring pipeline so users can
    understand why a particular memory was ranked where it was.
    """

    memory_id: str = ""
    final_score: float = 0.0
    vector_rank: int | None = None
    vector_rrf_contribution: float = 0.0
    keyword_rank: int | None = None
    keyword_rrf_contribution: float = 0.0
    rrf_score: float = 0.0
    recency_factor: float = 0.0
    recency_contribution: float = 0.0
    age_days: float = 0.0
    summary: str = ""


@dataclass
class StoreStats:
    """Aggregate statistics from a memory store."""

    total_memories: int = 0
    active_memories: int = 0
    archived_memories: int = 0
    expired_memories: int = 0
    deleted_memories: int = 0
    total_users: int = 0
    total_namespaces: int = 0
    embedding_dimensions: int = 0
    store_backend: str = ""
