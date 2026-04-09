"""Memory conflict detection and resolution.

When a new memory contradicts an existing one, this module detects the
conflict and applies a resolution strategy:

1. **Detection**: Find existing memories that are semantically similar
   to the new memory, then use LLM (or heuristics) to determine if
   they contradict each other.

2. **Resolution strategies**:
   - KEEP_NEWEST: Replace old memory with new (default)
   - KEEP_OLDEST: Discard the new memory
   - MERGE: Combine both into a single updated memory
   - FLAG: Keep both but mark as conflicting for human review

Usage::

    detector = ConflictDetector(store, embedder, llm_provider)
    conflicts = await detector.detect(new_memory)
    if conflicts:
        resolved = await detector.resolve(new_memory, conflicts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from neuragram.core.filters import MemoryFilter
from neuragram.core.models import Memory, MemoryType, ScoredMemory
from neuragram.processing.embeddings import BaseEmbeddingProvider, NullEmbeddingProvider
from neuragram.processing.llm import BaseLLMProvider, LLMError
from neuragram.store.base import BaseMemoryStore

_CONFLICT_DETECTION_PROMPT = """You are a memory conflict detection engine. Given a NEW memory and an EXISTING memory, determine if they contradict each other.

Two memories conflict when:
- They make opposing claims about the same subject (e.g., "User uses Python" vs "User uses Java")
- One updates/supersedes information in the other (e.g., "Email is old@x.com" vs "Email is new@x.com")
- They describe incompatible states (e.g., "User is a beginner" vs "User is an expert")

Two memories do NOT conflict when:
- They are about different subjects
- They are complementary (both can be true simultaneously)
- One is more specific than the other but not contradictory

Respond with a JSON object:
{
  "conflicts": true/false,
  "confidence": 0.9,
  "reasoning": "brief explanation",
  "resolution_hint": "keep_newest|keep_oldest|merge|flag"
}"""

_MERGE_PROMPT = """You are a memory merging engine. Given two conflicting memories, create a single merged memory that captures the most accurate and up-to-date information.

Rules:
1. Prefer newer information when there's a direct contradiction
2. Preserve all non-contradictory details from both memories
3. The merged memory should be a single, clear statement
4. Maintain the same level of specificity

Respond with a JSON object:
{
  "content": "the merged memory text",
  "confidence": 0.85,
  "reasoning": "brief explanation of what was merged"
}"""


class ResolutionStrategy(str, Enum):
    """How to resolve a memory conflict."""

    KEEP_NEWEST = "keep_newest"
    KEEP_OLDEST = "keep_oldest"
    MERGE = "merge"
    FLAG = "flag"


@dataclass
class Conflict:
    """A detected conflict between two memories."""

    existing_memory: Memory
    similarity_score: float = 0.0
    conflict_confidence: float = 0.0
    reasoning: str = ""
    suggested_resolution: ResolutionStrategy = ResolutionStrategy.KEEP_NEWEST


@dataclass
class ConflictResolution:
    """Result of resolving a conflict."""

    strategy_applied: ResolutionStrategy
    resulting_memory: Memory | None = None
    superseded_ids: list[str] = field(default_factory=list)
    reasoning: str = ""


class ConflictDetector:
    """Detects and resolves conflicts between new and existing memories.

    Args:
        store: Memory store for querying existing memories.
        embedding_provider: For computing similarity between memories.
        llm_provider: Optional LLM for intelligent conflict detection.
            If None, uses similarity-based heuristics only.
        similarity_threshold: Minimum similarity to consider as potential conflict.
        default_strategy: Default resolution strategy when LLM is unavailable.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider | None = None,
        similarity_threshold: float = 0.75,
        default_strategy: ResolutionStrategy = ResolutionStrategy.KEEP_NEWEST,
    ) -> None:
        self._store = store
        self._embedder = embedding_provider
        self._llm = llm_provider
        self._similarity_threshold = similarity_threshold
        self._default_strategy = default_strategy

    async def detect(
        self,
        new_memory: Memory,
        max_candidates: int = 5,
    ) -> list[Conflict]:
        """Detect potential conflicts between a new memory and existing ones.

        Args:
            new_memory: The memory about to be stored.
            max_candidates: Max number of similar memories to check.

        Returns:
            List of detected conflicts (may be empty).
        """
        # Skip conflict detection for NullEmbeddingProvider
        if isinstance(self._embedder, NullEmbeddingProvider):
            return await self._detect_by_keyword(new_memory, max_candidates)

        # Find similar existing memories via vector search
        embedding = new_memory.embedding
        if not embedding:
            embedding = await self._embedder.embed_text(new_memory.content)

        filters = MemoryFilter(
            user_id=new_memory.user_id or None,
            namespace=new_memory.namespace or None,
        )

        similar_memories = await self._store.vector_search(
            embedding=embedding,
            filters=filters,
            top_k=max_candidates,
        )

        # Filter by similarity threshold
        candidates = [
            sm for sm in similar_memories
            if sm.score >= self._similarity_threshold
            and sm.memory.id != new_memory.id
        ]

        if not candidates:
            return []

        # Check each candidate for actual conflict
        conflicts: list[Conflict] = []
        for scored in candidates:
            conflict = await self._check_conflict(new_memory, scored)
            if conflict is not None:
                conflicts.append(conflict)

        return conflicts

    async def _detect_by_keyword(
        self,
        new_memory: Memory,
        max_candidates: int,
    ) -> list[Conflict]:
        """Fallback conflict detection using keyword search."""
        filters = MemoryFilter(
            user_id=new_memory.user_id or None,
            namespace=new_memory.namespace or None,
            types=[new_memory.memory_type],
        )

        similar = await self._store.keyword_search(
            query=new_memory.content,
            filters=filters,
            top_k=max_candidates,
        )

        candidates = [
            sm for sm in similar
            if sm.memory.id != new_memory.id
        ]

        if not candidates:
            return []

        conflicts: list[Conflict] = []
        for scored in candidates:
            conflict = await self._check_conflict(new_memory, scored)
            if conflict is not None:
                conflicts.append(conflict)

        return conflicts

    async def _check_conflict(
        self,
        new_memory: Memory,
        scored_existing: ScoredMemory,
    ) -> Conflict | None:
        """Check if two memories actually conflict.

        Uses LLM if available, otherwise uses heuristic rules.
        """
        existing = scored_existing.memory

        if self._llm is not None:
            try:
                return await self._check_conflict_with_llm(
                    new_memory, existing, scored_existing.score
                )
            except (LLMError, Exception):
                pass

        return self._check_conflict_with_rules(
            new_memory, existing, scored_existing.score
        )

    async def _check_conflict_with_llm(
        self,
        new_memory: Memory,
        existing: Memory,
        similarity: float,
    ) -> Conflict | None:
        """Use LLM to determine if two memories conflict."""
        assert self._llm is not None

        user_message = (
            f"NEW memory: {new_memory.content}\n"
            f"EXISTING memory: {existing.content}\n"
            f"Similarity score: {similarity:.2f}"
        )

        raw = await self._llm.complete_json(
            system_prompt=_CONFLICT_DETECTION_PROMPT,
            user_message=user_message,
        )

        if not raw.get("conflicts", False):
            return None

        resolution_str = str(raw.get("resolution_hint", "keep_newest")).lower()
        try:
            suggested = ResolutionStrategy(resolution_str)
        except ValueError:
            suggested = self._default_strategy

        return Conflict(
            existing_memory=existing,
            similarity_score=similarity,
            conflict_confidence=float(raw.get("confidence", 0.8)),
            reasoning=str(raw.get("reasoning", "")),
            suggested_resolution=suggested,
        )

    @staticmethod
    def _check_conflict_with_rules(
        new_memory: Memory,
        existing: Memory,
        similarity: float,
    ) -> Conflict | None:
        """Heuristic conflict detection based on type and similarity.

        High similarity + same type + same user = likely conflict.
        """
        if similarity < 0.85:
            return None

        # Same type memories with very high similarity are likely updates
        if new_memory.memory_type == existing.memory_type:
            return Conflict(
                existing_memory=existing,
                similarity_score=similarity,
                conflict_confidence=min(similarity, 0.7),
                reasoning="High similarity between same-type memories suggests an update",
                suggested_resolution=ResolutionStrategy.KEEP_NEWEST,
            )

        return None

    async def resolve(
        self,
        new_memory: Memory,
        conflicts: list[Conflict],
        strategy: ResolutionStrategy | None = None,
    ) -> ConflictResolution:
        """Resolve detected conflicts.

        Args:
            new_memory: The new memory being stored.
            conflicts: Detected conflicts from detect().
            strategy: Override resolution strategy. If None, uses each
                conflict's suggested_resolution (or default_strategy).

        Returns:
            ConflictResolution describing what was done.
        """
        if not conflicts:
            return ConflictResolution(
                strategy_applied=ResolutionStrategy.KEEP_NEWEST,
                resulting_memory=new_memory,
            )

        effective_strategy = strategy or conflicts[0].suggested_resolution

        if effective_strategy == ResolutionStrategy.KEEP_NEWEST:
            return await self._resolve_keep_newest(new_memory, conflicts)
        elif effective_strategy == ResolutionStrategy.KEEP_OLDEST:
            return self._resolve_keep_oldest(new_memory, conflicts)
        elif effective_strategy == ResolutionStrategy.MERGE:
            return await self._resolve_merge(new_memory, conflicts)
        elif effective_strategy == ResolutionStrategy.FLAG:
            return self._resolve_flag(new_memory, conflicts)
        else:
            return await self._resolve_keep_newest(new_memory, conflicts)

    async def _resolve_keep_newest(
        self,
        new_memory: Memory,
        conflicts: list[Conflict],
    ) -> ConflictResolution:
        """Keep the new memory, soft-delete conflicting old ones."""
        superseded_ids = []
        for conflict in conflicts:
            await self._store.delete(conflict.existing_memory.id, hard=False)
            superseded_ids.append(conflict.existing_memory.id)

        return ConflictResolution(
            strategy_applied=ResolutionStrategy.KEEP_NEWEST,
            resulting_memory=new_memory,
            superseded_ids=superseded_ids,
            reasoning=f"Kept newest memory, superseded {len(superseded_ids)} conflicting memories",
        )

    @staticmethod
    def _resolve_keep_oldest(
        new_memory: Memory,
        conflicts: list[Conflict],
    ) -> ConflictResolution:
        """Discard the new memory, keep existing ones."""
        return ConflictResolution(
            strategy_applied=ResolutionStrategy.KEEP_OLDEST,
            resulting_memory=None,
            reasoning="Kept existing memories, discarded new conflicting memory",
        )

    async def _resolve_merge(
        self,
        new_memory: Memory,
        conflicts: list[Conflict],
    ) -> ConflictResolution:
        """Merge new and existing memories into one."""
        primary_conflict = conflicts[0]
        existing = primary_conflict.existing_memory

        if self._llm is not None:
            try:
                return await self._merge_with_llm(new_memory, existing, conflicts)
            except (LLMError, Exception):
                pass

        # Fallback: keep newest with metadata noting the merge
        return await self._resolve_keep_newest(new_memory, conflicts)

    async def _merge_with_llm(
        self,
        new_memory: Memory,
        existing: Memory,
        conflicts: list[Conflict],
    ) -> ConflictResolution:
        """Use LLM to merge two conflicting memories."""
        assert self._llm is not None

        user_message = (
            f"Memory A (newer): {new_memory.content}\n"
            f"Memory B (older): {existing.content}"
        )

        raw = await self._llm.complete_json(
            system_prompt=_MERGE_PROMPT,
            user_message=user_message,
        )

        merged_content = str(raw.get("content", new_memory.content))
        merged_confidence = float(raw.get("confidence", 0.8))

        # Create merged memory based on the new one
        merged = Memory(
            content=merged_content,
            memory_type=new_memory.memory_type,
            user_id=new_memory.user_id,
            agent_id=new_memory.agent_id,
            namespace=new_memory.namespace,
            tags=list(set(new_memory.tags + existing.tags)),
            metadata={
                **new_memory.metadata,
                "merged_from": [new_memory.id, existing.id],
            },
            confidence=merged_confidence,
            importance=max(new_memory.importance, existing.importance),
            source="llm_merge",
        )

        # Supersede old memories
        superseded_ids = []
        for conflict in conflicts:
            await self._store.delete(conflict.existing_memory.id, hard=False)
            superseded_ids.append(conflict.existing_memory.id)

        return ConflictResolution(
            strategy_applied=ResolutionStrategy.MERGE,
            resulting_memory=merged,
            superseded_ids=superseded_ids,
            reasoning=str(raw.get("reasoning", "Merged conflicting memories via LLM")),
        )

    @staticmethod
    def _resolve_flag(
        new_memory: Memory,
        conflicts: list[Conflict],
    ) -> ConflictResolution:
        """Keep both but flag the new memory as having conflicts."""
        conflict_ids = [c.existing_memory.id for c in conflicts]
        new_memory.metadata["conflicts_with"] = conflict_ids
        new_memory.metadata["conflict_status"] = "unresolved"

        return ConflictResolution(
            strategy_applied=ResolutionStrategy.FLAG,
            resulting_memory=new_memory,
            reasoning=f"Flagged new memory as conflicting with {len(conflict_ids)} existing memories",
        )
