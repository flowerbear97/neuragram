"""Hybrid retrieval engine.

Orchestrates vector search, keyword search, and recency boosting,
then fuses results via RRF and deduplicates.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from neuragram.core.filters import MemoryFilter
from neuragram.core.models import ScoreExplanation, ScoredMemory
from neuragram.processing.embeddings import BaseEmbeddingProvider, NullEmbeddingProvider
from neuragram.retrieval.scoring import (
    apply_recency_boost,
    deduplicate,
    reciprocal_rank_fusion,
)
from neuragram.store.base import BaseMemoryStore


class RetrievalEngine:
    """Hybrid retrieval combining vector, keyword, and recency signals.

    Pipeline:
        1. Vector search (if embedding provider is not Null)
        2. Keyword search (FTS5)
        3. RRF fusion of the two ranked lists
        4. Recency boost
        5. Deduplication
        6. Top-K truncation

    Args:
        store: The memory store backend.
        embedding_provider: Provider for computing query embeddings.
        vector_weight: RRF weight for vector search results.
        keyword_weight: RRF weight for keyword search results.
        recency_weight: Weight for recency boost in final scoring.
        recency_half_life_days: Half-life for recency exponential decay.
        dedup_threshold: Cosine/Jaccard threshold for deduplication.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        embedding_provider: BaseEmbeddingProvider,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.3,
        recency_weight: float = 0.2,
        recency_half_life_days: float = 7.0,
        dedup_threshold: float = 0.95,
    ) -> None:
        self._store = store
        self._embedder = embedding_provider
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._recency_weight = recency_weight
        self._recency_half_life_days = recency_half_life_days
        self._dedup_threshold = dedup_threshold

    async def search(
        self,
        query: str,
        filters: MemoryFilter,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """Execute a hybrid search and return the top-K most relevant memories.

        Args:
            query: Natural language query string.
            filters: Filtering criteria (user_id, namespace, types, etc.).
            top_k: Maximum number of results to return.

        Returns:
            Sorted list of ScoredMemory (highest score first).
        """
        ranked_lists: list[list[ScoredMemory]] = []
        weights: list[float] = []

        # 1. Vector search (skip if using NullEmbeddingProvider)
        use_vector = not isinstance(self._embedder, NullEmbeddingProvider)
        if use_vector:
            query_embedding = await self._embedder.embed_text(query)
            vector_results = await self._store.vector_search(
                embedding=query_embedding,
                filters=filters,
                top_k=top_k * 3,  # over-fetch for fusion
            )
            if vector_results:
                ranked_lists.append(vector_results)
                weights.append(self._vector_weight)

        # 2. Keyword search (always attempted)
        keyword_results = await self._store.keyword_search(
            query=query,
            filters=filters,
            top_k=top_k * 3,
        )
        if keyword_results:
            ranked_lists.append(keyword_results)
            weights.append(self._keyword_weight)

        # If no results from either path, return empty
        if not ranked_lists:
            return []

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)

        # 4. Recency boost
        if self._recency_weight > 0:
            fused = apply_recency_boost(
                fused,
                half_life_days=self._recency_half_life_days,
                weight=self._recency_weight,
            )

        # 5. Deduplication
        fused = deduplicate(fused, threshold=self._dedup_threshold)

        # 6. Top-K truncation
        results = fused[:top_k]

        # Touch accessed memories (update access stats)
        for scored in results:
            try:
                await self._store.touch(scored.memory.id)
            except Exception:
                pass  # non-critical

        return results

    async def explain(
        self,
        query: str,
        filters: MemoryFilter,
        top_k: int = 10,
    ) -> list[ScoreExplanation]:
        """Execute a hybrid search and return detailed score explanations.

        Same pipeline as search(), but captures intermediate scoring details
        for each result so users can understand why memories were ranked
        the way they were.

        Args:
            query: Natural language query string.
            filters: Filtering criteria.
            top_k: Maximum number of results to explain.

        Returns:
            List of ScoreExplanation with full scoring breakdown.
        """
        use_vector = not isinstance(self._embedder, NullEmbeddingProvider)
        rrf_k = 60

        # 1. Run searches
        vector_results: list[ScoredMemory] = []
        keyword_results: list[ScoredMemory] = []

        if use_vector:
            query_embedding = await self._embedder.embed_text(query)
            vector_results = await self._store.vector_search(
                embedding=query_embedding,
                filters=filters,
                top_k=top_k * 3,
            )

        keyword_results = await self._store.keyword_search(
            query=query,
            filters=filters,
            top_k=top_k * 3,
        )

        # 2. Build rank maps
        vector_rank_map: dict[str, int] = {}
        for rank, scored in enumerate(vector_results):
            vector_rank_map[scored.memory.id] = rank

        keyword_rank_map: dict[str, int] = {}
        for rank, scored in enumerate(keyword_results):
            keyword_rank_map[scored.memory.id] = rank

        # 3. Compute RRF with tracking
        all_memory_ids: set[str] = set(vector_rank_map.keys()) | set(keyword_rank_map.keys())
        memory_map: dict[str, ScoredMemory] = {}
        for scored in vector_results + keyword_results:
            mid = scored.memory.id
            if mid not in memory_map or scored.score > memory_map[mid].score:
                memory_map[mid] = scored

        explanations_map: dict[str, ScoreExplanation] = {}

        for mid in all_memory_ids:
            vector_rank = vector_rank_map.get(mid)
            keyword_rank = keyword_rank_map.get(mid)

            vector_rrf = (self._vector_weight / (rrf_k + vector_rank + 1)) if vector_rank is not None else 0.0
            keyword_rrf = (self._keyword_weight / (rrf_k + keyword_rank + 1)) if keyword_rank is not None else 0.0
            rrf_score = vector_rrf + keyword_rrf

            explanations_map[mid] = ScoreExplanation(
                memory_id=mid,
                vector_rank=vector_rank,
                vector_rrf_contribution=round(vector_rrf, 6),
                keyword_rank=keyword_rank,
                keyword_rrf_contribution=round(keyword_rrf, 6),
                rrf_score=round(rrf_score, 6),
            )

        # 4. Compute recency details
        reference_time = datetime.now(timezone.utc)
        decay_lambda = math.log(2) / max(self._recency_half_life_days, 0.001)

        for mid, explanation in explanations_map.items():
            if mid not in memory_map:
                continue
            memory = memory_map[mid].memory
            accessed = memory.last_accessed_at
            if accessed.tzinfo is None:
                accessed = accessed.replace(tzinfo=timezone.utc)

            age_seconds = max((reference_time - accessed).total_seconds(), 0)
            age_days = age_seconds / 86400.0
            recency_factor = math.exp(-decay_lambda * age_days)

            recency_contribution = self._recency_weight * recency_factor
            final_score = (1.0 - self._recency_weight) * explanation.rrf_score + recency_contribution

            explanation.age_days = round(age_days, 2)
            explanation.recency_factor = round(recency_factor, 4)
            explanation.recency_contribution = round(recency_contribution, 6)
            explanation.final_score = round(final_score, 6)

            # Build human-readable summary
            parts = []
            if explanation.vector_rank is not None:
                parts.append(f"vector rank #{explanation.vector_rank + 1} (+{explanation.vector_rrf_contribution:.4f})")
            if explanation.keyword_rank is not None:
                parts.append(f"keyword rank #{explanation.keyword_rank + 1} (+{explanation.keyword_rrf_contribution:.4f})")
            parts.append(f"recency {explanation.recency_factor:.2f} (age {explanation.age_days:.1f}d)")
            explanation.summary = " | ".join(parts)

        # 5. Sort by final score and truncate
        sorted_explanations = sorted(
            explanations_map.values(),
            key=lambda e: e.final_score,
            reverse=True,
        )
        return sorted_explanations[:top_k]
