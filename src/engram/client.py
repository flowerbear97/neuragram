"""AgentMemory — the primary public API for Engram.

This is the single entry point that users interact with. It assembles
all internal components (store, embedding, retrieval, lifecycle) and
exposes a clean, minimal API surface:

    mem = AgentMemory(db_path="./memory.db")
    mem.remember("user prefers concise answers", user_id="u1", type="preference")
    results = mem.recall("what style does the user prefer?", user_id="u1")
    mem.close()

Both synchronous and async APIs are provided. The sync versions
internally bridge to async via _run().
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import datetime
from typing import Any

from engram.core.access import AccessDeniedError, AccessLevel, AccessPolicy
from engram.core.config import EngramConfig
from engram.core.filters import MemoryFilter
from engram.core.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    ScoredMemory,
    StoreStats,
)
from engram.core.telemetry import traced_operation
from engram.lifecycle.decay import DecayManager, DecayResult
from engram.lifecycle.forgetting import ForgettingManager, ForgetResult
from engram.processing.embeddings import (
    BaseEmbeddingProvider,
    create_embedding_provider,
)
from engram.processing.llm import BaseLLMProvider
from engram.retrieval.engine import RetrievalEngine
from engram.store.base import BaseMemoryStore
from engram.store.registry import create_store


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous code.

    Handles the tricky case where we might already be inside an event loop
    (e.g. Jupyter notebooks, nested async frameworks) by using a thread pool.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already in an event loop — run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class AgentMemory:
    """The primary interface to Engram's memory system.

    Provides both synchronous and async (``a``-prefixed) methods.

    Args:
        store: Backend name ("sqlite") or a pre-built BaseMemoryStore instance.
        db_path: Database path (for SQLite backend).
        embedding: Embedding provider name ("none", "local", "openai") or instance.
        embedding_model: Model identifier for the embedding provider.
        embedding_dimension: Vector dimensionality.
        config: Full EngramConfig (overrides individual params if provided).
        **kwargs: Additional options passed to config / providers.
    """

    def __init__(
        self,
        store: str | BaseMemoryStore = "sqlite",
        db_path: str = "./engram.db",
        embedding: str | BaseEmbeddingProvider = "none",
        embedding_model: str = "",
        embedding_dimension: int = 384,
        llm: str | BaseLLMProvider | None = None,
        llm_model: str = "",
        access_policy: AccessPolicy | None = None,
        config: EngramConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Build config
        if config is not None:
            self._config = config
        else:
            self._config = EngramConfig(
                store=store if isinstance(store, str) else "sqlite",
                db_path=db_path,
                embedding=embedding if isinstance(embedding, str) else "none",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in EngramConfig.__dataclass_fields__
                },
            )
        self._config.validate()

        # Build store
        if isinstance(store, BaseMemoryStore):
            self._store = store
        else:
            self._store = create_store(
                backend=self._config.store,
                dimension=self._config.embedding_dimension,
                db_path=self._config.db_path,
            )

        # Build embedding provider
        if isinstance(embedding, BaseEmbeddingProvider):
            self._embedder = embedding
        else:
            self._embedder = create_embedding_provider(
                provider_name=self._config.embedding,
                dimension=self._config.embedding_dimension,
                model=self._config.embedding_model,
                **self._config.embedding_options,
            )

         # Build LLM provider (optional, for smart features)
        if isinstance(llm, BaseLLMProvider):
            self._llm: BaseLLMProvider | None = llm
        elif isinstance(llm, str):
            self._llm = create_llm_provider(
                provider_name=llm,
                model=llm_model,
            )
        else:
            self._llm = None

        # Build retrieval engine
        self._retrieval = RetrievalEngine(
            store=self._store,
            embedding_provider=self._embedder,
            vector_weight=self._config.retrieval_vector_weight,
            keyword_weight=self._config.retrieval_keyword_weight,
            recency_weight=self._config.retrieval_recency_weight,
            recency_half_life_days=self._config.recency_half_life_days,
            dedup_threshold=self._config.dedup_threshold,
        )

        # Build lifecycle managers
        self._decay_manager = DecayManager(
            store=self._store,
            max_age_days=self._config.decay_max_age_days,
            ttl_enabled=self._config.decay_ttl_enabled,
        )
        self._forgetting_manager = ForgettingManager(store=self._store)

        # Access control
        self._access_policy = access_policy or AccessPolicy()

        self._initialized = False

    # ── Initialization ──────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._store.initialize()
            self._initialized = True

    # ── Remember (Store) ────────────────────────────────────────────

    async def aremember(
        self,
        content: str,
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
        type: str | MemoryType = "fact",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "",
        expires_at: datetime | None = None,
    ) -> str:
        """Store a new memory (async).

        Args:
            content: The text content to remember.
            user_id: Owner of this memory.
            agent_id: The agent creating this memory.
            namespace: Logical grouping.
            type: Memory type (fact, episode, preference, procedure, plan_state).
            tags: Free-form labels.
            metadata: Arbitrary key-value pairs.
            importance: Importance score [0, 1].
            confidence: Confidence score [0, 1].
            source: Provenance information.
            expires_at: Optional TTL expiration.

        Returns:
            The ID of the newly created memory.
        """
        await self._ensure_initialized()

        memory_type = MemoryType(type) if isinstance(type, str) else type

        # Compute embedding
        embedding = await self._embedder.embed_text(content)

        memory = Memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            tags=tags or [],
            metadata=metadata or {},
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            source=source,
            expires_at=expires_at,
        )

        return await self._store.insert(memory)

    def remember(self, content: str, **kwargs: Any) -> str:
        """Store a new memory (sync). See ``aremember`` for full signature."""
        return _run_async(self.aremember(content, **kwargs))

    # ── Recall (Retrieve) ──────────────────────────────────────────

    async def arecall(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        namespace: str | None = None,
        types: list[str | MemoryType] | None = None,
        top_k: int | None = None,
        min_confidence: float | None = None,
        min_importance: float | None = None,
    ) -> list[ScoredMemory]:
        """Retrieve relevant memories (async).

        Args:
            query: Natural language query.
            user_id: Filter by user.
            agent_id: Filter by agent (for multi-agent memory isolation).
            namespace: Filter by namespace.
            types: Filter by memory types.
            top_k: Max results (defaults to config value).
            min_confidence: Minimum confidence threshold.
            min_importance: Minimum importance threshold.

        Returns:
            List of ScoredMemory sorted by relevance (highest first).
        """
        await self._ensure_initialized()

        parsed_types = [
            MemoryType(t) if isinstance(t, str) else t for t in (types or [])
        ]

        filters = MemoryFilter(
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            types=parsed_types,
            min_confidence=min_confidence,
            min_importance=min_importance,
        )

        return await self._retrieval.search(
            query=query,
            filters=filters,
            top_k=top_k or self._config.retrieval_top_k,
        )

    def recall(self, query: str, **kwargs: Any) -> list[ScoredMemory]:
        """Retrieve relevant memories (sync). See ``arecall`` for full signature."""
        return _run_async(self.arecall(query, **kwargs))

    # ── Get ─────────────────────────────────────────────────────────

    async def aget(self, memory_id: str) -> Memory | None:
        """Get a single memory by ID (async)."""
        await self._ensure_initialized()
        return await self._store.get(memory_id)

    def get(self, memory_id: str) -> Memory | None:
        """Get a single memory by ID (sync)."""
        return _run_async(self.aget(memory_id))

    # ── Update ──────────────────────────────────────────────────────

    async def aupdate(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        expires_at: datetime | None = None,
        source: str | None = None,
    ) -> Memory:
        """Update an existing memory (async).

        Automatically versions the previous state. Only non-None fields
        are applied.

        Args:
            memory_id: ID of the memory to update.
            content: New content (triggers re-embedding).
            metadata: New metadata dict.
            tags: New tags list.
            confidence: New confidence score.
            importance: New importance score.
            expires_at: New expiration time.
            source: New source/provenance.

        Returns:
            The updated Memory.

        Raises:
            MemoryNotFoundError: If the memory doesn't exist.
        """
        await self._ensure_initialized()

        update_fields: dict[str, Any] = {}

        if content is not None:
            update_fields["content"] = content
            # Re-compute embedding for new content
            new_embedding = await self._embedder.embed_text(content)
            update_fields["embedding"] = new_embedding

        if metadata is not None:
            update_fields["metadata"] = metadata
        if tags is not None:
            update_fields["tags"] = tags
        if confidence is not None:
            update_fields["confidence"] = confidence
        if importance is not None:
            update_fields["importance"] = importance
        if expires_at is not None:
            update_fields["expires_at"] = expires_at
        if source is not None:
            update_fields["source"] = source

        return await self._store.update(memory_id, **update_fields)

    def update(self, memory_id: str, **kwargs: Any) -> Memory:
        """Update an existing memory (sync). See ``aupdate`` for full signature."""
        return _run_async(self.aupdate(memory_id, **kwargs))

    # ── Forget (Delete) ─────────────────────────────────────────────

    async def aforget(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        hard: bool = False,
    ) -> int:
        """Delete memories (async).

        Provide either memory_id (single) or user_id (all user memories).

        Args:
            memory_id: Delete a specific memory.
            user_id: Delete all memories for this user (GDPR).
            hard: Physical deletion if True, soft delete if False.

        Returns:
            Number of memories deleted.
        """
        await self._ensure_initialized()

        if memory_id is not None:
            success = await self._forgetting_manager.forget_memory(
                memory_id, hard=hard
            )
            return 1 if success else 0

        if user_id is not None:
            result = await self._forgetting_manager.forget_user(
                user_id, hard=hard
            )
            return result.deleted_count

        return 0

    def forget(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        hard: bool = False,
    ) -> int:
        """Delete memories (sync). See ``aforget`` for full signature."""
        return _run_async(self.aforget(memory_id=memory_id, user_id=user_id, hard=hard))

    # ── History (Versions) ──────────────────────────────────────────

    async def ahistory(self, memory_id: str) -> list[MemoryVersion]:
        """Get version history for a memory (async)."""
        await self._ensure_initialized()
        return await self._store.get_versions(memory_id)

    def history(self, memory_id: str) -> list[MemoryVersion]:
        """Get version history for a memory (sync)."""
        return _run_async(self.ahistory(memory_id))

    # ── Decay (Lifecycle) ───────────────────────────────────────────

    async def adecay(self, max_age_days: int | None = None) -> dict[str, int]:
        """Run memory decay cycle (async).

        Expires TTL'd memories and archives inactive ones.

        Args:
            max_age_days: Override the configured max age for archival.

        Returns:
            Dict with "expired" and "archived" counts.
        """
        await self._ensure_initialized()

        if max_age_days is not None:
            self._decay_manager._max_age_days = max_age_days

        result: DecayResult = await self._decay_manager.run_decay()
        return {"expired": result.expired, "archived": result.archived}

    def decay(self, max_age_days: int | None = None) -> dict[str, int]:
        """Run memory decay cycle (sync). See ``adecay`` for full signature."""
        return _run_async(self.adecay(max_age_days=max_age_days))

    # ── List ────────────────────────────────────────────────────────

    async def alist(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        namespace: str | None = None,
        types: list[str | MemoryType] | None = None,
        statuses: list[str | MemoryStatus] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories with filtering and pagination (async).

        Args:
            user_id: Filter by user.
            agent_id: Filter by agent (for multi-agent memory isolation).
            namespace: Filter by namespace.
            types: Filter by memory types.
            statuses: Filter by statuses (defaults to [ACTIVE]).
            limit: Max results per page.
            offset: Pagination offset.

        Returns:
            List of Memory objects.
        """
        await self._ensure_initialized()

        parsed_types = [
            MemoryType(t) if isinstance(t, str) else t for t in (types or [])
        ]
        parsed_statuses = [
            MemoryStatus(s) if isinstance(s, str) else s
            for s in (statuses or [MemoryStatus.ACTIVE])
        ]

        filters = MemoryFilter(
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            types=parsed_types,
            statuses=parsed_statuses,
        )

        return await self._store.list_memories(filters, limit=limit, offset=offset)

    def list(self, **kwargs: Any) -> list[Memory]:
        """List memories (sync). See ``alist`` for full signature."""
        return _run_async(self.alist(**kwargs))

    # ── Stats ───────────────────────────────────────────────────────

    async def astats(self) -> StoreStats:
        """Get store statistics (async)."""
        await self._ensure_initialized()
        return await self._store.stats()

    def stats(self) -> StoreStats:
        """Get store statistics (sync)."""
        return _run_async(self.astats())

    # ── Lifecycle ───────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Release all resources (async)."""
        await self._store.close()

    def close(self) -> None:
        """Release all resources (sync)."""
        _run_async(self.aclose())

    # ── Context Manager ─────────────────────────────────────────────

    async def __aenter__(self) -> AgentMemory:
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def __enter__(self) -> AgentMemory:
        _run_async(self._ensure_initialized())
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Smart Remember ────────────────────────────────────────────────

    async def asmart_remember(
        self,
        content: str,
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
        detect_conflicts: bool = True,
        auto_classify: bool = True,
    ) -> list[str]:
        """Intelligently store a memory with auto-classification and conflict detection (async).

        Unlike ``aremember()``, this method:
        1. Auto-classifies the memory type, importance, and confidence
        2. Detects conflicts with existing memories
        3. Resolves conflicts automatically (keeps newest by default)

        Works without LLM (uses rule-based heuristics). With LLM, provides
        higher accuracy classification and smarter conflict resolution.

        Args:
            content: The text content to remember.
            user_id: Owner of this memory.
            agent_id: The agent creating this memory.
            namespace: Logical grouping.
            detect_conflicts: Whether to check for conflicting memories.
            auto_classify: Whether to auto-classify type/importance/confidence.

        Returns:
            List of stored memory IDs (usually 1, but may differ after merge).
        """
        await self._ensure_initialized()

        from engram.processing.classifier import MemoryClassifier
        from engram.processing.conflict import ConflictDetector

        # Step 1: Auto-classify
        memory_type = MemoryType.FACT
        importance = 0.5
        confidence = 1.0
        tags: list[str] = []

        if auto_classify:
            classifier = MemoryClassifier(llm_provider=self._llm)
            classification = await classifier.classify(content)
            memory_type = classification.memory_type
            importance = classification.importance
            confidence = classification.confidence
            tags = classification.tags

        # Step 2: Compute embedding
        embedding = await self._embedder.embed_text(content)

        # Step 3: Build memory
        memory = Memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            tags=tags,
            source="smart_remember",
        )

        # Step 4: Conflict detection
        if detect_conflicts:
            detector = ConflictDetector(
                store=self._store,
                embedding_provider=self._embedder,
                llm_provider=self._llm,
            )
            conflicts = await detector.detect(memory)
            if conflicts:
                resolution = await detector.resolve(memory, conflicts)
                if resolution.resulting_memory is None:
                    return []  # KEEP_OLDEST strategy — discard new memory
                memory = resolution.resulting_memory
                # Re-compute embedding if content changed (e.g. after merge)
                if memory.embedding is None:
                    memory.embedding = await self._embedder.embed_text(memory.content)

        # Step 5: Store
        memory_id = await self._store.insert(memory)
        return [memory_id]

    def smart_remember(self, content: str, **kwargs: Any) -> list[str]:
        """Intelligently store a memory (sync). See ``asmart_remember``."""
        return _run_async(self.asmart_remember(content, **kwargs))

    # ── Process Conversation ──────────────────────────────────────────

    async def aprocess_conversation(
        self,
        messages: list[dict[str, str]],
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
        detect_conflicts: bool = True,
    ) -> list[str]:
        """Extract and store memories from a conversation (async).

        Requires an LLM provider. Analyzes the conversation, extracts
        structured memories, checks for conflicts, and stores them.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}.
            user_id: Owner of the extracted memories.
            agent_id: Agent that participated.
            namespace: Logical grouping.
            detect_conflicts: Whether to check for conflicts.

        Returns:
            List of stored memory IDs.

        Raises:
            EngramError: If no LLM provider is configured.
        """
        await self._ensure_initialized()

        if self._llm is None:
            from engram.core.exceptions import EngramError

            raise EngramError(
                "process_conversation() requires an LLM provider. "
                "Initialize with llm='openai' or llm='ollama'."
            )

        from engram.processing.extraction import MemoryExtractor
        from engram.processing.conflict import ConflictDetector

        # Step 1: Extract memories from conversation
        extractor = MemoryExtractor(self._llm)
        extraction = await extractor.extract_from_conversation(
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
        )

        if not extraction.memories:
            return []

        # Step 2: Compute embeddings for all extracted memories
        contents = [m.content for m in extraction.memories]
        embeddings = await self._embedder.embed_batch(contents)
        for memory, embedding in zip(extraction.memories, embeddings):
            memory.embedding = embedding

        # Step 3: Conflict detection and storage
        stored_ids: list[str] = []
        detector = ConflictDetector(
            store=self._store,
            embedding_provider=self._embedder,
            llm_provider=self._llm,
        ) if detect_conflicts else None

        for memory in extraction.memories:
            if detector is not None:
                conflicts = await detector.detect(memory)
                if conflicts:
                    resolution = await detector.resolve(memory, conflicts)
                    if resolution.resulting_memory is None:
                        continue
                    memory = resolution.resulting_memory
                    if memory.embedding is None:
                        memory.embedding = await self._embedder.embed_text(memory.content)

            memory_id = await self._store.insert(memory)
            stored_ids.append(memory_id)

        return stored_ids

    def process_conversation(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> list[str]:
        """Extract and store memories from a conversation (sync). See ``aprocess_conversation``."""
        return _run_async(self.aprocess_conversation(messages, **kwargs))

    # ── Consolidate ───────────────────────────────────────────────────

    async def aconsolidate(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
        similarity_threshold: float = 0.80,
    ) -> dict[str, int]:
        """Consolidate similar memories into summaries (async).

        Finds groups of similar memories and merges each group into a
        single summary. Original memories are soft-deleted.

        Works best with an LLM provider for high-quality summaries.
        Without LLM, keeps the highest-importance memory from each group.

        Args:
            user_id: Filter by user.
            namespace: Filter by namespace.
            similarity_threshold: Minimum similarity to group together.

        Returns:
            Dict with "groups_merged" and "memories_consolidated" counts.
        """
        await self._ensure_initialized()

        from engram.processing.merger import MemoryMerger

        merger = MemoryMerger(
            store=self._store,
            embedding_provider=self._embedder,
            llm_provider=self._llm,
        )

        results = await merger.merge_similar(
            user_id=user_id,
            namespace=namespace,
            similarity_threshold=similarity_threshold,
        )

        total_groups = sum(r.groups_processed for r in results)
        total_consolidated = sum(r.memories_consolidated for r in results)

        return {
            "groups_merged": total_groups,
            "memories_consolidated": total_consolidated,
        }

    def consolidate(self, **kwargs: Any) -> dict[str, int]:
        """Consolidate similar memories (sync). See ``aconsolidate``."""
        return _run_async(self.aconsolidate(**kwargs))

    # ── Classify ──────────────────────────────────────────────────────

    async def aclassify(self, content: str) -> dict[str, Any]:
        """Classify text content without storing it (async).

        Useful for previewing how a memory would be classified.

        Args:
            content: Text to classify.

        Returns:
            Dict with "type", "importance", "confidence", "tags", "method".
        """
        from engram.processing.classifier import MemoryClassifier

        classifier = MemoryClassifier(llm_provider=self._llm)
        result = await classifier.classify(content)
        return {
            "type": result.memory_type.value,
            "importance": result.importance,
            "confidence": result.confidence,
            "tags": result.tags,
            "method": result.method,
            "reasoning": result.reasoning,
        }

    def classify(self, content: str) -> dict[str, Any]:
        """Classify text content (sync). See ``aclassify``."""
        return _run_async(self.aclassify(content))

    # ── Explain ───────────────────────────────────────────────────────

    async def aexplain(
        self,
        query: str,
        user_id: str | None = None,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Explain how retrieval scores were computed (async).

        Returns the same results as recall(), but with a detailed
        breakdown of each score component (vector rank, keyword rank,
        RRF contribution, recency factor).

        Args:
            query: Natural language search query.
            user_id: Filter by user ID.
            namespace: Filter by namespace.
            top_k: Maximum number of results to explain.

        Returns:
            List of dicts with score breakdown for each result.
        """
        await self._ensure_initialized()

        filters = MemoryFilter(
            user_id=user_id,
            namespace=namespace,
            statuses=[MemoryStatus.ACTIVE],
        )

        explanations = await self._retrieval.explain(
            query=query,
            filters=filters,
            top_k=top_k,
        )

        return [
            {
                "memory_id": exp.memory_id,
                "final_score": exp.final_score,
                "vector_rank": exp.vector_rank,
                "vector_rrf_contribution": exp.vector_rrf_contribution,
                "keyword_rank": exp.keyword_rank,
                "keyword_rrf_contribution": exp.keyword_rrf_contribution,
                "rrf_score": exp.rrf_score,
                "recency_factor": exp.recency_factor,
                "recency_contribution": exp.recency_contribution,
                "age_days": exp.age_days,
                "summary": exp.summary,
            }
            for exp in explanations
        ]

    def explain(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Explain retrieval scores (sync). See ``aexplain``."""
        return _run_async(self.aexplain(query, **kwargs))

    # ── Access Control ────────────────────────────────────────────────

    @property
    def access_policy(self) -> AccessPolicy:
        """Get the access control policy for this instance.

        Use this to grant/revoke permissions and enforce access control::

            mem.access_policy.enable()
            mem.access_policy.grant("reader_agent", AccessLevel.READ)
            mem.access_policy.enforce("reader_agent", AccessLevel.WRITE, "remember")
        """
        return self._access_policy

    # ── Lifecycle Worker ──────────────────────────────────────────────

    def create_worker(
        self,
        interval_seconds: float = 3600,
        enable_expiration: bool = True,
        enable_archival: bool = True,
        enable_consolidation: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a background lifecycle worker for this memory instance.

        The worker periodically runs maintenance tasks (expiration,
        archival, consolidation) in the background.

        Args:
            interval_seconds: Seconds between maintenance cycles.
            enable_expiration: Run TTL expiration each cycle.
            enable_archival: Run inactivity archival each cycle.
            enable_consolidation: Run memory consolidation (requires LLM).
            **kwargs: Additional options passed to LifecycleWorker.

        Returns:
            A LifecycleWorker instance (call .start() to begin).

        Example::

            worker = mem.create_worker(interval_seconds=3600)
            await worker.start()
            # ... later ...
            await worker.stop()
        """
        from engram.lifecycle.worker import LifecycleWorker

        return LifecycleWorker(
            memory=self,
            interval_seconds=interval_seconds,
            enable_expiration=enable_expiration,
            enable_archival=enable_archival,
            enable_consolidation=enable_consolidation,
            **kwargs,
        )

    # ── Namespace Stats ──────────────────────────────────────────────

    async def anamespace_stats(
        self,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Get statistics scoped to a namespace (async).

        If namespace is None, returns stats grouped by namespace.

        Args:
            namespace: Specific namespace to query, or None for all.

        Returns:
            Dict with namespace-level statistics.
        """
        await self._ensure_initialized()

        if namespace is not None:
            filters = MemoryFilter(namespace=namespace)
            memories = await self._store.list_memories(filters, limit=0)
            active_filters = MemoryFilter(
                namespace=namespace,
                statuses=[MemoryStatus.ACTIVE],
            )
            active = await self._store.list_memories(active_filters, limit=10000)
            return {
                "namespace": namespace,
                "total_memories": len(active),
                "memory_types": self._count_types(active),
            }

        # All namespaces: get overall stats and list unique namespaces
        all_memories = await self._store.list_memories(
            MemoryFilter(statuses=[MemoryStatus.ACTIVE]),
            limit=10000,
        )
        namespaces: dict[str, int] = {}
        for mem in all_memories:
            ns = mem.namespace
            namespaces[ns] = namespaces.get(ns, 0) + 1

        return {
            "namespaces": namespaces,
            "total_namespaces": len(namespaces),
            "total_memories": len(all_memories),
        }

    def namespace_stats(self, **kwargs: Any) -> dict[str, Any]:
        """Get namespace statistics (sync). See ``anamespace_stats``."""
        return _run_async(self.anamespace_stats(**kwargs))

    @staticmethod
    def _count_types(memories: list[Memory]) -> dict[str, int]:
        """Count memories by type."""
        counts: dict[str, int] = {}
        for mem in memories:
            type_name = mem.memory_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    # ── Repr ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AgentMemory(store={self._config.store!r}, "
            f"embedding={self._config.embedding!r}, "
            f"db_path={self._config.db_path!r})"
        )
