"""Store backend registry — maps names to factory functions."""

from __future__ import annotations

from typing import Any

from neuragram.core.exceptions import BackendNotAvailableError
from neuragram.store.base import BaseMemoryStore


def create_store(
    backend: str,
    dimension: int = 384,
    **kwargs: Any,
) -> BaseMemoryStore:
    """Create a memory store by backend name.

    Args:
         backend: One of "sqlite" (more backends planned).        dimension: Embedding vector dimensionality.
        **kwargs: Backend-specific options (e.g. db_path for SQLite).

    Returns:
        An uninitialized store instance. Caller must call ``await store.initialize()``.
    """
    if backend == "sqlite":
        from neuragram.store.sqlite import SQLiteMemoryStore

        db_path = kwargs.get("db_path", "./neuragram.db")
        return SQLiteMemoryStore(db_path=str(db_path), dimension=dimension)

    raise BackendNotAvailableError(
        backend, f"Unknown store backend: {backend}"
    )
