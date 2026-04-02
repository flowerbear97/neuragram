"""SQLite + sqlite-vec + FTS5 memory store implementation.

This is the default storage backend. It uses:
- SQLite for relational data (memories, versions)
- sqlite-vec for vector similarity search
- FTS5 for full-text keyword search
- aiosqlite for async I/O

All three indexes (relational, vector, FTS) are kept in sync within
transactions to guarantee consistency.
"""

from __future__ import annotations

import json
import struct
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import aiosqlite

from engram.core.exceptions import MemoryNotFoundError, StoreError
from engram.core.filters import MemoryFilter
from engram.core.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    ScoredMemory,
    StoreStats,
)
from engram.store.base import BaseMemoryStore

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    memory_type     TEXT NOT NULL DEFAULT 'fact',
    status          TEXT NOT NULL DEFAULT 'active',
    user_id         TEXT NOT NULL DEFAULT '',
    agent_id        TEXT NOT NULL DEFAULT '',
    namespace       TEXT NOT NULL DEFAULT 'default',
    tags            TEXT NOT NULL DEFAULT '[]',
    metadata        TEXT NOT NULL DEFAULT '{}',
    confidence      REAL NOT NULL DEFAULT 1.0,
    importance      REAL NOT NULL DEFAULT 0.5,
    version         INTEGER NOT NULL DEFAULT 1,
    source          TEXT NOT NULL DEFAULT '',
    expires_at      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at);

CREATE TABLE IF NOT EXISTS memory_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions(memory_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    content,
    tokenize='porter unicode61'
);
"""


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(blob: bytes, dimension: int) -> list[float]:
    """Unpack bytes back into a float list."""
    return list(struct.unpack(f"{dimension}f", blob))


def _dt_to_str(dt: datetime) -> str:
    """Serialize a datetime to ISO-8601 UTC string."""
    return dt.isoformat()


def _str_to_dt(value: str) -> datetime:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_memory(row: aiosqlite.Row) -> Memory:
    """Convert a database row to a Memory dataclass."""
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=MemoryType(row["memory_type"]),
        status=MemoryStatus(row["status"]),
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        namespace=row["namespace"],
        tags=json.loads(row["tags"]),
        metadata=json.loads(row["metadata"]),
        confidence=row["confidence"],
        importance=row["importance"],
        version=row["version"],
        source=row["source"],
        expires_at=_str_to_dt(row["expires_at"]) if row["expires_at"] else None,
        created_at=_str_to_dt(row["created_at"]),
        updated_at=_str_to_dt(row["updated_at"]),
        last_accessed_at=_str_to_dt(row["last_accessed_at"]),
        access_count=row["access_count"],
        embedding=None,  # not stored in relational table
    )


def _build_where_clause(
    filters: MemoryFilter,
) -> tuple[str, list[object]]:
    """Build a SQL WHERE clause from a MemoryFilter."""
    conditions: list[str] = []
    params: list[object] = []

    if filters.user_id is not None:
        conditions.append("user_id = ?")
        params.append(filters.user_id)

    if filters.agent_id is not None:
        conditions.append("agent_id = ?")
        params.append(filters.agent_id)

    if filters.namespace is not None:
        conditions.append("namespace = ?")
        params.append(filters.namespace)

    if filters.types:
        placeholders = ",".join("?" for _ in filters.types)
        conditions.append(f"memory_type IN ({placeholders})")
        params.extend(t.value for t in filters.types)

    if filters.statuses:
        placeholders = ",".join("?" for _ in filters.statuses)
        conditions.append(f"status IN ({placeholders})")
        params.extend(s.value for s in filters.statuses)

    if filters.tags:
        for tag in filters.tags:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

    if filters.created_after is not None:
        conditions.append("created_at > ?")
        params.append(_dt_to_str(filters.created_after))

    if filters.created_before is not None:
        conditions.append("created_at < ?")
        params.append(_dt_to_str(filters.created_before))

    if filters.min_confidence is not None:
        conditions.append("confidence >= ?")
        params.append(filters.min_confidence)

    if filters.min_importance is not None:
        conditions.append("importance >= ?")
        params.append(filters.min_importance)

    where = " AND ".join(conditions) if conditions else "1=1"
    return where, params


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite-backed memory store with vector search and full-text search.

    Args:
        db_path: Path to the SQLite database file, or ":memory:" for in-memory.
        dimension: Dimensionality of embedding vectors.
    """

    def __init__(
        self,
        db_path: str = "./engram.db",
        dimension: int = 384,
        cache_size: int = 256,
    ) -> None:
        self._db_path = db_path
        self._dimension = dimension
        self._db: aiosqlite.Connection | None = None
        self._closed = False

        # LRU cache for hot memories (get by ID)
        self._cache_max_size = cache_size
        self._cache: OrderedDict[str, Memory] = OrderedDict()

    def _ensure_open(self) -> aiosqlite.Connection:
        if self._closed or self._db is None:
            raise StoreError("Store is closed")
        return self._db

    def _cache_put(self, memory: Memory) -> None:
        """Add or update a memory in the LRU cache."""
        if memory.id in self._cache:
            self._cache.move_to_end(memory.id)
        self._cache[memory.id] = memory
        while len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)

    def _cache_get(self, memory_id: str) -> Memory | None:
        """Get a memory from cache, returning None on miss."""
        if memory_id in self._cache:
            self._cache.move_to_end(memory_id)
            return self._cache[memory_id]
        return None

    def _cache_invalidate(self, memory_id: str) -> None:
        """Remove a memory from cache."""
        self._cache.pop(memory_id, None)

    def _cache_clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        # Performance PRAGMAs
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA cache_size=-64000")  # 64MB cache
        await self._db.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        await self._db.execute("PRAGMA temp_store=MEMORY")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.executescript(_SCHEMA_SQL)

        # Initialize sqlite-vec virtual table (graceful degradation)
        self._vec_available = False
        try:
            import sqlite_vec

            await self._db.enable_load_extension(True)
            sqlite_vec.load(self._db._conn)  # load into the underlying connection
            await self._db.enable_load_extension(False)

            vec_table_sql = (
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
                f"USING vec0(id TEXT PRIMARY KEY, embedding float[{self._dimension}])"
            )
            await self._db.execute(vec_table_sql)
            self._vec_available = True
        except (ImportError, AttributeError, Exception):
            # ImportError: sqlite-vec not installed
            # AttributeError: enable_load_extension not available (macOS system Python)
            # Other: vec0 module not loadable
            pass

        await self._db.commit()

    async def insert(self, memory: Memory) -> str:
        db = self._ensure_open()

        # Check for duplicate ID
        cursor = await db.execute(
            "SELECT 1 FROM memories WHERE id = ?", (memory.id,)
        )
        if await cursor.fetchone():
            raise StoreError(f"Memory with id {memory.id} already exists")

        now = _utcnow()
        await db.execute(
            """INSERT INTO memories
               (id, content, memory_type, status, user_id, agent_id, namespace,
                tags, metadata, confidence, importance, version, source,
                expires_at, created_at, updated_at, last_accessed_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.status.value,
                memory.user_id,
                memory.agent_id,
                memory.namespace,
                json.dumps(memory.tags),
                json.dumps(memory.metadata),
                memory.confidence,
                memory.importance,
                memory.version,
                memory.source,
                _dt_to_str(memory.expires_at) if memory.expires_at else None,
                _dt_to_str(memory.created_at),
                _dt_to_str(now),
                _dt_to_str(now),
                memory.access_count,
            ),
        )

        # FTS index
        await db.execute(
            "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
            (memory.id, memory.content),
        )

        # Vector index
        if memory.embedding and any(v != 0.0 for v in memory.embedding):
            try:
                await db.execute(
                    "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                    (memory.id, _serialize_embedding(memory.embedding)),
                )
            except Exception:
                pass  # vec0 not available

        await db.commit()
        return memory.id

    async def batch_insert(self, memories: list[Memory]) -> list[str]:
        ids = []
        for memory in memories:
            mid = await self.insert(memory)
            ids.append(mid)
        return ids

    async def get(self, memory_id: str) -> Memory | None:
        # Check LRU cache first
        cached = self._cache_get(memory_id)
        if cached is not None:
            return cached

        db = self._ensure_open()
        cursor = await db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        memory = _row_to_memory(row)
        self._cache_put(memory)
        return memory

    async def list_memories(
        self,
        filters: MemoryFilter,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        db = self._ensure_open()
        where, params = _build_where_clause(filters)
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_memory(row) for row in rows]

    async def vector_search(
        self,
        embedding: list[float],
        filters: MemoryFilter,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        db = self._ensure_open()

        # Use sqlite-vec for ANN search, then post-filter
        try:
            vec_sql = """
                SELECT id, distance
                FROM memories_vec
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """
            cursor = await db.execute(
                vec_sql,
                (_serialize_embedding(embedding), top_k * 5),
            )
            vec_rows = await cursor.fetchall()
        except Exception:
            return []  # vec0 not available

        if not vec_rows:
            return []

        # Build score map from vector results
        score_map: dict[str, float] = {}
        for vec_row in vec_rows:
            distance = vec_row["distance"]
            score_map[vec_row["id"]] = 1.0 / (1.0 + distance)

        # Batch fetch and filter all candidate memories in one query
        candidate_ids = list(score_map.keys())
        where, params = _build_where_clause(filters)
        placeholders = ",".join("?" for _ in candidate_ids)
        batch_sql = (
            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND {where}"
        )
        cursor = await db.execute(batch_sql, [*candidate_ids, *params])
        rows = await cursor.fetchall()

        results: list[ScoredMemory] = []
        for row in rows:
            memory = _row_to_memory(row)
            results.append(ScoredMemory(memory=memory, score=score_map[memory.id]))

        # Sort by score descending and truncate
        results.sort(reverse=True)
        return results[:top_k]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query string for FTS5 MATCH syntax.

        FTS5 treats certain characters as operators. We strip them and
        split the query into individual tokens joined by OR.
        """
        import re

        # Remove FTS5 special characters
        cleaned = re.sub(r'[^\w\s]', ' ', query, flags=re.UNICODE)
        tokens = cleaned.split()
        if not tokens:
            return '""'
        # Join tokens with OR for broader matching
        return " OR ".join(f'"{token}"' for token in tokens if token.strip())

    async def keyword_search(
        self,
        query: str,
        filters: MemoryFilter,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        db = self._ensure_open()

        sanitized_query = self._sanitize_fts_query(query)
        if sanitized_query == '""':
            return []

        # FTS5 search
        fts_sql = """
            SELECT id, rank
            FROM memories_fts
            WHERE content MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        try:
            cursor = await db.execute(fts_sql, (sanitized_query, top_k * 3))
            fts_rows = await cursor.fetchall()
        except Exception:
            return []

        if not fts_rows:
            return []

        # Build score map from FTS results
        score_map: dict[str, float] = {}
        for fts_row in fts_rows:
            rank = fts_row["rank"]
            score_map[fts_row["id"]] = 1.0 / (1.0 + abs(rank))

        # Batch fetch and filter all candidate memories in one query
        candidate_ids = list(score_map.keys())
        where, params = _build_where_clause(filters)
        placeholders = ",".join("?" for _ in candidate_ids)
        batch_sql = (
            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND {where}"
        )
        cursor = await db.execute(batch_sql, [*candidate_ids, *params])
        rows = await cursor.fetchall()

        results: list[ScoredMemory] = []
        for row in rows:
            memory = _row_to_memory(row)
            results.append(ScoredMemory(memory=memory, score=score_map[memory.id]))

        # Sort by score descending and truncate
        results.sort(reverse=True)
        return results[:top_k]

    async def update(self, memory_id: str, **fields: object) -> Memory:
        db = self._ensure_open()

        # Fetch current memory
        current = await self.get(memory_id)
        if current is None:
            raise MemoryNotFoundError(memory_id)

        # Save current version to history
        await db.execute(
            """INSERT INTO memory_versions (memory_id, version, content, metadata, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                memory_id,
                current.version,
                current.content,
                json.dumps(current.metadata),
                _dt_to_str(current.updated_at),
            ),
        )

        # Build update SET clause
        now = _utcnow()
        new_version = current.version + 1
        set_parts = ["version = ?", "updated_at = ?"]
        set_params: list[object] = [new_version, _dt_to_str(now)]

        content_changed = False
        new_content = current.content

        for key, value in fields.items():
            if key == "content" and value is not None:
                set_parts.append("content = ?")
                set_params.append(value)
                content_changed = True
                new_content = str(value)
            elif key == "memory_type" and value is not None:
                set_parts.append("memory_type = ?")
                set_params.append(value.value if isinstance(value, MemoryType) else str(value))
            elif key == "metadata" and value is not None:
                set_parts.append("metadata = ?")
                set_params.append(json.dumps(value))
            elif key == "tags" and value is not None:
                set_parts.append("tags = ?")
                set_params.append(json.dumps(value))
            elif key == "confidence" and value is not None:
                set_parts.append("confidence = ?")
                set_params.append(value)
            elif key == "importance" and value is not None:
                set_parts.append("importance = ?")
                set_params.append(value)
            elif key == "expires_at" and value is not None:
                set_parts.append("expires_at = ?")
                set_params.append(_dt_to_str(value) if isinstance(value, datetime) else value)
            elif key == "source" and value is not None:
                set_parts.append("source = ?")
                set_params.append(value)
            elif key == "embedding" and value is not None:
                # Update vector index
                embedding_list = list(value)  # type: ignore[arg-type]
                try:
                    await db.execute(
                        "DELETE FROM memories_vec WHERE id = ?", (memory_id,)
                    )
                    if any(v != 0.0 for v in embedding_list):
                        await db.execute(
                            "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                            (memory_id, _serialize_embedding(embedding_list)),
                        )
                except Exception:
                    pass

        set_clause = ", ".join(set_parts)
        set_params.append(memory_id)

        await db.execute(
            f"UPDATE memories SET {set_clause} WHERE id = ?",
            set_params,
        )

        # Update FTS if content changed
        if content_changed:
            await db.execute(
                "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
            )
            await db.execute(
                "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
                (memory_id, new_content),
            )

        await db.commit()
        self._cache_invalidate(memory_id)

        updated = await self.get(memory_id)
        assert updated is not None
        return updated

    async def touch(self, memory_id: str) -> None:
        db = self._ensure_open()
        now = _dt_to_str(_utcnow())
        cursor = await db.execute(
            "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            (now, memory_id),
        )
        if cursor.rowcount == 0:
            raise MemoryNotFoundError(memory_id)
        await db.commit()
        self._cache_invalidate(memory_id)

    async def delete(self, memory_id: str, hard: bool = False) -> bool:
        db = self._ensure_open()

        if hard:
            # Physical delete from all tables
            cursor = await db.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            if cursor.rowcount == 0:
                return False
            await db.execute(
                "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
            )
            try:
                await db.execute(
                    "DELETE FROM memories_vec WHERE id = ?", (memory_id,)
                )
            except Exception:
                pass
            await db.execute(
                "DELETE FROM memory_versions WHERE memory_id = ?", (memory_id,)
            )
        else:
            # Soft delete
            cursor = await db.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                (MemoryStatus.DELETED.value, _dt_to_str(_utcnow()), memory_id),
            )
            if cursor.rowcount == 0:
                return False

        await db.commit()
        self._cache_invalidate(memory_id)
        return True

    async def delete_by_filter(
        self, filters: MemoryFilter, hard: bool = False
    ) -> int:
        db = self._ensure_open()
        where, params = _build_where_clause(filters)

        if hard:
            # Get IDs first for cascading cleanup
            cursor = await db.execute(
                f"SELECT id FROM memories WHERE {where}", params
            )
            rows = await cursor.fetchall()
            ids = [row["id"] for row in rows]

            if not ids:
                return 0

            placeholders = ",".join("?" for _ in ids)
            await db.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", ids
            )
            await db.execute(
                f"DELETE FROM memories_fts WHERE id IN ({placeholders})", ids
            )
            try:
                await db.execute(
                    f"DELETE FROM memories_vec WHERE id IN ({placeholders})", ids
                )
            except Exception:
                pass
            await db.execute(
                f"DELETE FROM memory_versions WHERE memory_id IN ({placeholders})",
                ids,
            )
            await db.commit()
            for mid in ids:
                self._cache_invalidate(mid)
            return len(ids)
        else:
            now = _dt_to_str(_utcnow())
            cursor = await db.execute(
                f"UPDATE memories SET status = ?, updated_at = ? WHERE {where}",
                [MemoryStatus.DELETED.value, now, *params],
            )
            await db.commit()
            self._cache_clear()  # Bulk update — clear entire cache
            return cursor.rowcount  # type: ignore[return-value]

    async def expire_stale(self) -> int:
        db = self._ensure_open()
        now = _dt_to_str(_utcnow())
        cursor = await db.execute(
            """UPDATE memories
               SET status = ?, updated_at = ?
               WHERE expires_at IS NOT NULL
                 AND expires_at <= ?
                 AND status = ?""",
            (
                MemoryStatus.EXPIRED.value,
                now,
                now,
                MemoryStatus.ACTIVE.value,
            ),
        )
        await db.commit()
        self._cache_clear()  # Bulk status change — clear cache
        return cursor.rowcount  # type: ignore[return-value]

    async def archive_inactive(self, max_age_days: int) -> int:
        db = self._ensure_open()
        now = _utcnow()
        cutoff = _dt_to_str(now - timedelta(days=max_age_days))
        cursor = await db.execute(
            """UPDATE memories
               SET status = ?, updated_at = ?
               WHERE last_accessed_at < ?
                 AND status = ?""",
            (
                MemoryStatus.ARCHIVED.value,
                _dt_to_str(now),
                cutoff,
                MemoryStatus.ACTIVE.value,
            ),
        )
        await db.commit()
        self._cache_clear()  # Bulk status change — clear cache
        return cursor.rowcount  # type: ignore[return-value]

    async def get_versions(self, memory_id: str) -> list[MemoryVersion]:
        db = self._ensure_open()
        cursor = await db.execute(
            """SELECT memory_id, version, content, metadata, updated_at
               FROM memory_versions
               WHERE memory_id = ?
               ORDER BY version ASC""",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        return [
            MemoryVersion(
                memory_id=row["memory_id"],
                version=row["version"],
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                updated_at=_str_to_dt(row["updated_at"]),
            )
            for row in rows
        ]

    async def stats(self) -> StoreStats:
        db = self._ensure_open()

        total = (await (await db.execute("SELECT COUNT(*) FROM memories")).fetchone())[0]
        active = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM memories WHERE status = ?",
                    (MemoryStatus.ACTIVE.value,),
                )
            ).fetchone()
        )[0]
        archived = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM memories WHERE status = ?",
                    (MemoryStatus.ARCHIVED.value,),
                )
            ).fetchone()
        )[0]
        expired = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM memories WHERE status = ?",
                    (MemoryStatus.EXPIRED.value,),
                )
            ).fetchone()
        )[0]
        deleted = (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM memories WHERE status = ?",
                    (MemoryStatus.DELETED.value,),
                )
            ).fetchone()
        )[0]
        users = (
            await (
                await db.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM memories WHERE user_id != ''"
                )
            ).fetchone()
        )[0]
        namespaces = (
            await (
                await db.execute("SELECT COUNT(DISTINCT namespace) FROM memories")
            ).fetchone()
        )[0]

        return StoreStats(
            total_memories=total,
            active_memories=active,
            archived_memories=archived,
            expired_memories=expired,
            deleted_memories=deleted,
            total_users=users,
            total_namespaces=namespaces,
            embedding_dimensions=self._dimension,
            store_backend="sqlite",
        )

    async def ping(self) -> bool:
        try:
            db = self._ensure_open()
            await db.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._db is not None and not self._closed:
            await self._db.close()
            self._closed = True
            self._db = None
            self._cache_clear()
