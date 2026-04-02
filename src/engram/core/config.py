"""Engram configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exceptions import ConfigError


@dataclass
class EngramConfig:
    """Central configuration for an Engram instance.

    Attributes:
        store: Storage backend name ("sqlite", "postgres", etc.).
        db_path: Path to the database file (for SQLite).
        embedding: Embedding provider name ("none", "local", "openai", "ollama").
        embedding_model: Model identifier for the embedding provider.
        embedding_dimension: Dimensionality of embedding vectors.
        embedding_options: Extra kwargs passed to the embedding provider.
        retrieval_top_k: Default number of results for recall().
        retrieval_vector_weight: Weight for vector search in RRF fusion.
        retrieval_keyword_weight: Weight for keyword search in RRF fusion.
        retrieval_recency_weight: Weight for recency boost in scoring.
        recency_half_life_days: Half-life for exponential recency decay (days).
        dedup_threshold: Cosine similarity threshold for deduplication.
        decay_max_age_days: Max days without access before archiving.
        decay_ttl_enabled: Whether to enforce TTL expiration.
    """

    store: str = "sqlite"
    db_path: str = "./engram.db"
    embedding: str = "none"
    embedding_model: str = ""
    embedding_dimension: int = 384
    embedding_options: dict[str, Any] = field(default_factory=dict)

    retrieval_top_k: int = 10
    retrieval_vector_weight: float = 0.5
    retrieval_keyword_weight: float = 0.3
    retrieval_recency_weight: float = 0.2
    recency_half_life_days: float = 7.0
    dedup_threshold: float = 0.95

    decay_max_age_days: int = 30
    decay_ttl_enabled: bool = True

    def validate(self) -> None:
        """Validate configuration values, raising ConfigError on problems."""
        if self.store not in ("sqlite", "postgres"):
            raise ConfigError(f"Unknown store backend: {self.store}")

        if self.embedding not in ("none", "local", "openai", "ollama"):
            raise ConfigError(f"Unknown embedding provider: {self.embedding}")

        if self.embedding_dimension <= 0:
            raise ConfigError(
                f"embedding_dimension must be positive, got {self.embedding_dimension}"
            )

        if not (0.0 <= self.retrieval_vector_weight <= 1.0):
            raise ConfigError("retrieval_vector_weight must be in [0, 1]")

        if not (0.0 <= self.retrieval_keyword_weight <= 1.0):
            raise ConfigError("retrieval_keyword_weight must be in [0, 1]")

        if not (0.0 <= self.retrieval_recency_weight <= 1.0):
            raise ConfigError("retrieval_recency_weight must be in [0, 1]")

        if self.recency_half_life_days <= 0:
            raise ConfigError("recency_half_life_days must be positive")

        if not (0.0 <= self.dedup_threshold <= 1.0):
            raise ConfigError("dedup_threshold must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngramConfig:
        """Create config from a dictionary, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
