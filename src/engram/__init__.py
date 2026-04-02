"""Engram — The SQLite of agent memory.

A lightweight, standalone, framework-agnostic Python library for agent memory.
Zero external dependencies. pip install and go.

Usage::

    from engram import AgentMemory

    mem = AgentMemory(db_path="./memory.db")
    mem.remember("user prefers concise answers", user_id="u1", type="preference")
    results = mem.recall("what style does the user prefer?", user_id="u1")
    print(results[0].memory.content)
    mem.close()
"""

from engram.client import AgentMemory
from engram.core.config import EngramConfig
from engram.core.exceptions import (
    BackendNotAvailableError,
    ConfigError,
    EmbeddingError,
    EngramError,
    MemoryNotFoundError,
    StoreError,
)
from engram.core.filters import MemoryFilter
from engram.core.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    MemoryVersion,
    ScoredMemory,
    StoreStats,
)
from engram.processing.llm import (
    BaseLLMProvider,
    CallableLLMProvider,
    LLMError,
    LLMResponse,
    create_llm_provider,
)
from engram.processing.extraction import ExtractionResult, MemoryExtractor
from engram.processing.classifier import ClassificationResult, MemoryClassifier
from engram.processing.conflict import (
    Conflict,
    ConflictDetector,
    ConflictResolution,
    ResolutionStrategy,
)
from engram.processing.merger import MemoryMerger, MergeGroup, MergeResult
from engram.core.models import ScoreExplanation
from engram.core.access import AccessDeniedError, AccessLevel, AccessPolicy
from engram.core.telemetry import is_otel_available, traced_operation
from engram.lifecycle.worker import LifecycleWorker, WorkerStats

__version__ = "1.0.0"

__all__ = [
    # Main entry point
    "AgentMemory",
    # Models
    "Memory",
    "MemoryType",
    "MemoryStatus",
    "ScoredMemory",
    "MemoryUpdate",
    "MemoryVersion",
    "MemoryFilter",
    "StoreStats",
    "ScoreExplanation",
    # Config
    "EngramConfig",
    # Exceptions
    "EngramError",
    "MemoryNotFoundError",
    "StoreError",
    "EmbeddingError",
    "ConfigError",
    "BackendNotAvailableError",
    # LLM
    "BaseLLMProvider",
    "CallableLLMProvider",
    "LLMError",
    "LLMResponse",
    "create_llm_provider",
    # Extraction
    "MemoryExtractor",
    "ExtractionResult",
    # Classification
    "MemoryClassifier",
    "ClassificationResult",
    # Conflict Detection
    "ConflictDetector",
    "Conflict",
    "ConflictResolution",
    "ResolutionStrategy",
    # Merging
    "MemoryMerger",
    "MergeGroup",
    "MergeResult",
    # Access Control
    "AccessLevel",
    "AccessPolicy",
    "AccessDeniedError",
    # Lifecycle Worker
    "LifecycleWorker",
    "WorkerStats",
    # Telemetry
    "is_otel_available",
    "traced_operation",
]
