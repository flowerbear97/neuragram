"""Engram exception hierarchy."""


class EngramError(Exception):
    """Base exception for all Engram errors."""


class MemoryNotFoundError(EngramError):
    """Raised when a memory with the given ID does not exist."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class StoreError(EngramError):
    """Raised when a storage backend operation fails."""


class EmbeddingError(EngramError):
    """Raised when an embedding operation fails."""


class ConfigError(EngramError):
    """Raised when configuration is invalid."""


class BackendNotAvailableError(EngramError):
    """Raised when a requested storage backend is not installed or available."""

    def __init__(self, backend: str, reason: str = "") -> None:
        self.backend = backend
        msg = f"Backend not available: {backend}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
