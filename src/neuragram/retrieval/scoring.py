"""Scoring utilities for hybrid retrieval.

Implements:
- Reciprocal Rank Fusion (RRF) for combining ranked lists
- Recency boost with exponential decay
- Deduplication by embedding cosine similarity
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from neuragram.core.models import ScoredMemory


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredMemory]],
    weights: list[float] | None = None,
    rrf_k: int = 60,
) -> list[ScoredMemory]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ weight_i / (k + rank_i(d))

    Args:
        ranked_lists: Each inner list is sorted by descending relevance.
        weights: Per-list weight multipliers. Defaults to uniform.
        rrf_k: Smoothing constant (standard default = 60).

    Returns:
        A single merged list sorted by fused score (descending).
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    # Accumulate RRF scores by memory ID
    fused_scores: dict[str, float] = {}
    memory_map: dict[str, ScoredMemory] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, scored_memory in enumerate(ranked_list):
            memory_id = scored_memory.memory.id
            rrf_score = weight / (rrf_k + rank + 1)
            fused_scores[memory_id] = fused_scores.get(memory_id, 0.0) + rrf_score

            # Keep the ScoredMemory with the highest original score
            if memory_id not in memory_map or scored_memory.score > memory_map[memory_id].score:
                memory_map[memory_id] = scored_memory

    # Build result with fused scores
    results = [
        ScoredMemory(memory=memory_map[mid].memory, score=fused_score)
        for mid, fused_score in fused_scores.items()
    ]
    results.sort(reverse=True)
    return results


def apply_recency_boost(
    scored_memories: list[ScoredMemory],
    half_life_days: float = 7.0,
    weight: float = 0.2,
    reference_time: datetime | None = None,
) -> list[ScoredMemory]:
    """Boost scores of recent memories using exponential decay.

    recency_factor = exp(-λ * age_days)  where λ = ln(2) / half_life_days
    final_score = (1 - weight) * original_score + weight * recency_factor

    Args:
        scored_memories: Input list (will not be mutated).
        half_life_days: Time for recency factor to halve.
        weight: How much recency influences the final score [0, 1].
        reference_time: "Now" for age calculation. Defaults to UTC now.

    Returns:
        New list with adjusted scores, sorted descending.
    """
    if not scored_memories or weight <= 0:
        return list(scored_memories)

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    decay_lambda = math.log(2) / max(half_life_days, 0.001)

    boosted: list[ScoredMemory] = []
    for scored in scored_memories:
        accessed = scored.memory.last_accessed_at
        if accessed.tzinfo is None:
            accessed = accessed.replace(tzinfo=timezone.utc)

        age_seconds = max((reference_time - accessed).total_seconds(), 0)
        age_days = age_seconds / 86400.0
        recency_factor = math.exp(-decay_lambda * age_days)

        new_score = (1.0 - weight) * scored.score + weight * recency_factor
        boosted.append(ScoredMemory(memory=scored.memory, score=new_score))

    boosted.sort(reverse=True)
    return boosted


def deduplicate(
    scored_memories: list[ScoredMemory],
    threshold: float = 0.95,
) -> list[ScoredMemory]:
    """Remove near-duplicate memories based on content similarity.

    Uses a simple character-level approach when embeddings are not available,
    and cosine similarity when embeddings are present.

    Args:
        scored_memories: Input list sorted by descending score.
        threshold: Similarity threshold above which a memory is considered duplicate.

    Returns:
        Deduplicated list preserving the higher-scored memory.
    """
    if len(scored_memories) <= 1:
        return list(scored_memories)

    kept: list[ScoredMemory] = []

    for candidate in scored_memories:
        is_duplicate = False
        for existing in kept:
            similarity = _content_similarity(
                candidate.memory, existing.memory
            )
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)

    return kept


def _content_similarity(memory_a: object, memory_b: object) -> float:
    """Compute similarity between two memories.

    Prefers cosine similarity of embeddings when available;
    falls back to Jaccard similarity of content tokens.
    """
    from neuragram.core.models import Memory

    if not isinstance(memory_a, Memory) or not isinstance(memory_b, Memory):
        return 0.0

    # Try embedding cosine similarity first
    if (
        memory_a.embedding
        and memory_b.embedding
        and len(memory_a.embedding) == len(memory_b.embedding)
        and any(v != 0.0 for v in memory_a.embedding)
        and any(v != 0.0 for v in memory_b.embedding)
    ):
        return _cosine_similarity(memory_a.embedding, memory_b.embedding)

    # Fallback: Jaccard similarity on word tokens
    tokens_a = set(memory_a.content.lower().split())
    tokens_b = set(memory_b.content.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
