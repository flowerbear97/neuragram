"""Memory merging and summarization.

When multiple memories about the same topic accumulate, this module
can consolidate them into a single summary memory, reducing redundancy
while preserving key information.

Use cases:
- Periodic consolidation of related memories
- Summarizing a user's conversation history into key takeaways
- Reducing memory count while preserving information density

Usage::

    merger = MemoryMerger(store, embedder, llm_provider)
    result = await merger.merge_similar(
        user_id="u1",
        similarity_threshold=0.8,
        max_group_size=5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuragram.core.filters import MemoryFilter
from neuragram.core.models import Memory, MemoryType
from neuragram.processing.embeddings import BaseEmbeddingProvider, NullEmbeddingProvider
from neuragram.processing.llm import BaseLLMProvider, LLMError
from neuragram.store.base import BaseMemoryStore

_SUMMARIZE_PROMPT = """You are a memory summarization engine. Given a group of related memories, create a single concise summary that captures all important information.

Rules:
1. Preserve all key facts, preferences, and details
2. Remove redundancy and repetition
3. The summary should be self-contained and clear
4. If memories contain temporal information, preserve the most recent
5. Maintain specificity — don't over-generalize

Respond with a JSON object:
{
  "content": "the summarized memory text",
  "type": "fact|preference|episode|procedure",
  "importance": 0.8,
  "confidence": 0.9,
  "tags": ["tag1", "tag2"],
  "reasoning": "what was consolidated"
}"""


@dataclass
class MergeGroup:
    """A group of related memories identified for merging."""

    memories: list[Memory] = field(default_factory=list)
    similarity_scores: list[float] = field(default_factory=list)


@dataclass
class MergeResult:
    """Result of a merge operation."""

    summary_memory: Memory | None = None
    merged_ids: list[str] = field(default_factory=list)
    groups_processed: int = 0
    memories_consolidated: int = 0


class MemoryMerger:
    """Consolidates related memories into summaries.

    Args:
        store: Memory store backend.
        embedding_provider: For computing similarity between memories.
        llm_provider: LLM for generating summaries. Required for merge operations.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedding_provider
        self._llm = llm_provider

    async def find_merge_candidates(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
        memory_type: MemoryType | None = None,
        similarity_threshold: float = 0.80,
        max_group_size: int = 10,
    ) -> list[MergeGroup]:
        """Find groups of similar memories that could be merged.

        Args:
            user_id: Filter by user.
            namespace: Filter by namespace.
            memory_type: Filter by memory type.
            similarity_threshold: Minimum similarity to group together.
            max_group_size: Maximum memories per group.

        Returns:
            List of MergeGroup, each containing related memories.
        """
        filters = MemoryFilter(
            user_id=user_id,
            namespace=namespace,
            types=[memory_type] if memory_type else [],
        )

        all_memories = await self._store.list_memories(filters, limit=200)

        if len(all_memories) < 2:
            return []

        # Skip grouping if no embedding available
        if isinstance(self._embedder, NullEmbeddingProvider):
            return self._group_by_keyword_similarity(
                all_memories, similarity_threshold, max_group_size
            )

        return await self._group_by_embedding_similarity(
            all_memories, similarity_threshold, max_group_size
        )

    async def _group_by_embedding_similarity(
        self,
        memories: list[Memory],
        threshold: float,
        max_group_size: int,
    ) -> list[MergeGroup]:
        """Group memories by embedding cosine similarity."""
        import math

        # Compute embeddings for all memories
        texts = [m.content for m in memories]
        embeddings = await self._embedder.embed_batch(texts)

        # Greedy clustering: assign each memory to the first group it's similar to
        groups: list[MergeGroup] = []
        assigned: set[int] = set()

        for i in range(len(memories)):
            if i in assigned:
                continue

            group = MergeGroup(memories=[memories[i]], similarity_scores=[1.0])
            assigned.add(i)

            for j in range(i + 1, len(memories)):
                if j in assigned:
                    continue
                if len(group.memories) >= max_group_size:
                    break

                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    group.memories.append(memories[j])
                    group.similarity_scores.append(sim)
                    assigned.add(j)

            if len(group.memories) >= 2:
                groups.append(group)

        return groups

    @staticmethod
    def _group_by_keyword_similarity(
        memories: list[Memory],
        threshold: float,
        max_group_size: int,
    ) -> list[MergeGroup]:
        """Fallback grouping using Jaccard similarity on word tokens."""
        groups: list[MergeGroup] = []
        assigned: set[int] = set()

        for i in range(len(memories)):
            if i in assigned:
                continue

            tokens_i = set(memories[i].content.lower().split())
            group = MergeGroup(memories=[memories[i]], similarity_scores=[1.0])
            assigned.add(i)

            for j in range(i + 1, len(memories)):
                if j in assigned:
                    continue
                if len(group.memories) >= max_group_size:
                    break

                tokens_j = set(memories[j].content.lower().split())
                if not tokens_i or not tokens_j:
                    continue

                intersection = tokens_i & tokens_j
                union = tokens_i | tokens_j
                jaccard = len(intersection) / len(union) if union else 0.0

                if jaccard >= threshold:
                    group.memories.append(memories[j])
                    group.similarity_scores.append(jaccard)
                    assigned.add(j)

            if len(group.memories) >= 2:
                groups.append(group)

        return groups

    async def merge_group(
        self,
        group: MergeGroup,
        user_id: str = "",
        namespace: str = "default",
    ) -> MergeResult:
        """Merge a group of related memories into a single summary.

        Requires an LLM provider. Falls back to keeping the highest-importance
        memory if no LLM is available.

        Args:
            group: The group of memories to merge.
            user_id: Owner of the resulting summary memory.
            namespace: Namespace for the summary memory.

        Returns:
            MergeResult with the summary memory and merged IDs.
        """
        if len(group.memories) < 2:
            return MergeResult()

        merged_ids = [m.id for m in group.memories]

        if self._llm is not None:
            try:
                summary = await self._summarize_with_llm(group, user_id, namespace)
                return MergeResult(
                    summary_memory=summary,
                    merged_ids=merged_ids,
                    groups_processed=1,
                    memories_consolidated=len(group.memories),
                )
            except (LLMError, Exception):
                pass

        # Fallback: keep the most important memory
        best = max(group.memories, key=lambda m: m.importance)
        best.metadata["consolidated_from"] = merged_ids
        best.source = "consolidation"

        return MergeResult(
            summary_memory=best,
            merged_ids=merged_ids,
            groups_processed=1,
            memories_consolidated=len(group.memories),
        )

    async def _summarize_with_llm(
        self,
        group: MergeGroup,
        user_id: str,
        namespace: str,
    ) -> Memory:
        """Use LLM to generate a summary of related memories."""
        assert self._llm is not None

        memories_text = "\n".join(
            f"- [{m.memory_type.value}] {m.content}" for m in group.memories
        )

        raw = await self._llm.complete_json(
            system_prompt=_SUMMARIZE_PROMPT,
            user_message=f"Memories to summarize:\n{memories_text}",
        )

        content = str(raw.get("content", group.memories[0].content))
        type_str = str(raw.get("type", "fact")).lower()
        try:
            memory_type = MemoryType(type_str)
        except ValueError:
            memory_type = group.memories[0].memory_type

        importance = max(0.0, min(1.0, float(raw.get("importance", 0.7))))
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.8))))

        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t) for t in tags if t]

        all_tags = list(set(tags + [t for m in group.memories for t in m.tags]))
        merged_ids = [m.id for m in group.memories]

        return Memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            namespace=namespace,
            importance=importance,
            confidence=confidence,
            tags=all_tags,
            metadata={"consolidated_from": merged_ids},
            source="llm_summarization",
        )

    async def merge_similar(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
        memory_type: MemoryType | None = None,
        similarity_threshold: float = 0.80,
        max_group_size: int = 10,
    ) -> list[MergeResult]:
        """Find and merge all similar memory groups.

        This is the high-level convenience method that combines
        find_merge_candidates() and merge_group().

        Args:
            user_id: Filter by user.
            namespace: Filter by namespace.
            memory_type: Filter by type.
            similarity_threshold: Grouping threshold.
            max_group_size: Max memories per group.

        Returns:
            List of MergeResult, one per merged group.
        """
        groups = await self.find_merge_candidates(
            user_id=user_id,
            namespace=namespace,
            memory_type=memory_type,
            similarity_threshold=similarity_threshold,
            max_group_size=max_group_size,
        )

        results: list[MergeResult] = []
        for group in groups:
            result = await self.merge_group(
                group,
                user_id=user_id or "",
                namespace=namespace or "default",
            )

            # Store the summary and soft-delete originals
            if result.summary_memory is not None:
                embedding = await self._embedder.embed_text(result.summary_memory.content)
                result.summary_memory.embedding = embedding
                await self._store.insert(result.summary_memory)

                for old_id in result.merged_ids:
                    await self._store.delete(old_id, hard=False)

            results.append(result)

        return results

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
