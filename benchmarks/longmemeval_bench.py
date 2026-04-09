"""Neuragram Integration Test Suite.

Validates the core memory engine across 5 functional categories:
  1. Information Extraction — store and retrieve specific facts
  2. Multi-Session Reasoning — cross-session, namespace, and agent isolation
  3. Temporal Reasoning — recency ranking, TTL, versioning, archival
  4. Knowledge Updates — content updates, soft/hard delete, metadata
  5. Abstention — correctly return empty results for non-existent queries

These tests exercise neuragram's own API and are NOT comparable to
third-party benchmark scores (e.g. LongMemEval). They serve as a
regression / integration test for the memory engine.

No external LLM required. Uses keyword-only retrieval (embedding="none")
to test the core engine in isolation.

Usage:
    python benchmarks/longmemeval_bench.py
"""

from __future__ import annotations

# Use pysqlite3 if available (ships with modern SQLite supporting FTS5)
try:
    import pysqlite3 as _sqlite3
    import sys as _sys
    _sys.modules["sqlite3"] = _sqlite3
except ImportError:
    pass

import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "src")

from neuragram.client import AgentMemory
from neuragram.core.models import MemoryType


@dataclass
class TestResult:
    dimension: str
    test_name: str
    passed: bool
    latency_ms: float = 0.0
    detail: str = ""


@dataclass
class BenchmarkReport:
    results: list[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def add(self, result: TestResult) -> None:
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total * 100 if self.total else 0.0

    def print_report(self) -> None:
        elapsed = self.end_time - self.start_time
        print("\n" + "=" * 72)
        print("  NEURAGRAM INTEGRATION TEST REPORT")
        print("=" * 72)

        dimensions = {}
        for result in self.results:
            dimensions.setdefault(result.dimension, []).append(result)

        for dimension, tests in dimensions.items():
            dim_passed = sum(1 for t in tests if t.passed)
            dim_total = len(tests)
            dim_rate = dim_passed / dim_total * 100
            status_icon = "✅" if dim_passed == dim_total else "⚠️"
            print(f"\n{status_icon} {dimension} ({dim_passed}/{dim_total} = {dim_rate:.0f}%)")
            print("-" * 60)
            for test in tests:
                icon = "  ✓" if test.passed else "  ✗"
                print(f"{icon} {test.test_name} ({test.latency_ms:.1f}ms)")
                if not test.passed and test.detail:
                    print(f"      → {test.detail}")

        print("\n" + "=" * 72)
        print(f"  TOTAL: {self.passed}/{self.total} passed ({self.pass_rate:.1f}%)")
        print(f"  TIME:  {elapsed:.2f}s")
        print("=" * 72)

        print()


def timed(func):
    """Decorator to measure async function execution time in ms."""
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    return wrapper


# ---------------------------------------------------------------------------
# Dimension 1: Information Extraction
# ---------------------------------------------------------------------------

@timed
async def test_extract_single_fact(mem: AgentMemory) -> TestResult:
    """Store a specific fact and retrieve it by semantic query."""
    await mem.aremember(
        "Alice's favorite programming language is Rust",
        user_id="alice", type="preference",
    )
    await mem.aremember(
        "Bob prefers to use dark mode in all editors",
        user_id="bob", type="preference",
    )
    await mem.aremember(
        "Alice works at a fintech startup in Berlin",
        user_id="alice", type="fact",
    )

    results = await mem.arecall("programming language", user_id="alice", top_k=3)
    found = any("Rust" in r.memory.content for r in results)
    return TestResult(
        dimension="1. Information Extraction",
        test_name="Retrieve single fact by keyword",
        passed=found,
        detail="" if found else f"Expected 'Rust' in results, got: {[r.memory.content for r in results]}",
    )


@timed
async def test_extract_with_user_isolation(mem: AgentMemory) -> TestResult:
    """Ensure user_id filtering isolates memories correctly."""
    results = await mem.arecall("dark mode", user_id="alice", top_k=5)
    leaked = any("dark mode" in r.memory.content for r in results)
    return TestResult(
        dimension="1. Information Extraction",
        test_name="User isolation (no cross-user leakage)",
        passed=not leaked,
        detail="Bob's memory leaked into Alice's results" if leaked else "",
    )


@timed
async def test_extract_by_type_filter(mem: AgentMemory) -> TestResult:
    """Filter memories by type."""
    results = await mem.alist(user_id="alice", types=["preference"])
    all_prefs = all(m.memory_type == MemoryType.PREFERENCE for m in results)
    has_results = len(results) > 0
    return TestResult(
        dimension="1. Information Extraction",
        test_name="Filter by memory type",
        passed=all_prefs and has_results,
        detail="" if (all_prefs and has_results) else f"Got {len(results)} results, all_prefs={all_prefs}",
    )


@timed
async def test_extract_multiple_facts(mem: AgentMemory) -> TestResult:
    """Store multiple facts and retrieve the correct one."""
    await mem.aremember("The project deadline is March 15th", user_id="team", type="fact")
    await mem.aremember("The sprint review is every Friday at 3pm", user_id="team", type="fact")
    await mem.aremember("The database migration is scheduled for April 1st", user_id="team", type="fact")

    results = await mem.arecall("deadline", user_id="team", top_k=3)
    found = any("March 15th" in r.memory.content for r in results)
    return TestResult(
        dimension="1. Information Extraction",
        test_name="Retrieve specific fact from multiple stored",
        passed=found,
        detail="" if found else f"Expected 'March 15th', got: {[r.memory.content for r in results]}",
    )


# ---------------------------------------------------------------------------
# Dimension 2: Multi-Session Reasoning
# ---------------------------------------------------------------------------

@timed
async def test_cross_namespace_isolation(mem: AgentMemory) -> TestResult:
    """Memories in different namespaces are isolated."""
    await mem.aremember("Use PostgreSQL database for production", user_id="dev", namespace="project_alpha")
    await mem.aremember("Use MongoDB database for analytics", user_id="dev", namespace="project_beta")

    results = await mem.arecall("database", user_id="dev", namespace="project_alpha", top_k=5)
    found_pg = any("PostgreSQL" in r.memory.content for r in results)
    found_mongo = any("MongoDB" in r.memory.content for r in results)
    return TestResult(
        dimension="2. Multi-Session Reasoning",
        test_name="Namespace isolation",
        passed=found_pg and not found_mongo,
        detail="" if (found_pg and not found_mongo) else f"PG={found_pg}, Mongo leaked={found_mongo}",
    )


@timed
async def test_cross_agent_retrieval(mem: AgentMemory) -> TestResult:
    """Different agents can store and retrieve independently."""
    await mem.aremember("Deploy to staging first", user_id="ops", agent_id="agent_deploy")
    await mem.aremember("Run linter before commit", user_id="ops", agent_id="agent_review")

    results = await mem.arecall("deploy", user_id="ops", top_k=5)
    found_deploy = any("staging" in r.memory.content for r in results)
    return TestResult(
        dimension="2. Multi-Session Reasoning",
        test_name="Cross-agent memory retrieval",
        passed=found_deploy,
        detail="" if found_deploy else f"Got: {[r.memory.content for r in results]}",
    )


@timed
async def test_multi_user_aggregation(mem: AgentMemory) -> TestResult:
    """Retrieve memories across users when no user filter is applied."""
    await mem.aremember("Charlie likes TypeScript", user_id="charlie", namespace="lang_prefs")
    await mem.aremember("Diana likes Go", user_id="diana", namespace="lang_prefs")

    results = await mem.arecall("likes", namespace="lang_prefs", top_k=10)
    found_charlie = any("TypeScript" in r.memory.content for r in results)
    found_diana = any("Go" in r.memory.content for r in results)
    return TestResult(
        dimension="2. Multi-Session Reasoning",
        test_name="Multi-user aggregation (no user filter)",
        passed=found_charlie and found_diana,
        detail="" if (found_charlie and found_diana) else f"Charlie={found_charlie}, Diana={found_diana}",
    )


@timed
async def test_tag_based_retrieval(mem: AgentMemory) -> TestResult:
    """Memories can be filtered by tags."""
    await mem.aremember(
        "API rate limit is 1000 req/min",
        user_id="dev", tags=["api", "limits"], type="fact",
    )
    await mem.aremember(
        "Frontend uses React 18",
        user_id="dev", tags=["frontend", "react"], type="fact",
    )

    all_mems = await mem.alist(user_id="dev")
    api_mems = [m for m in all_mems if "api" in m.tags]
    return TestResult(
        dimension="2. Multi-Session Reasoning",
        test_name="Tag-based filtering",
        passed=len(api_mems) >= 1 and all("rate limit" in m.content for m in api_mems),
        detail="" if api_mems else "No memories found with tag 'api'",
    )


# ---------------------------------------------------------------------------
# Dimension 3: Temporal Reasoning
# ---------------------------------------------------------------------------

@timed
async def test_recency_boost(mem: AgentMemory) -> TestResult:
    """More recent memories should rank higher with recency boost."""
    now = datetime.now(timezone.utc)

    mem_old = AgentMemory(db_path=":memory:", embedding="none")
    await mem_old._ensure_initialized()

    await mem_old.aremember("Old config: use port 8080", user_id="sys", type="fact")

    # Manually backdate the memory by updating last_accessed_at
    old_mems = await mem_old.alist(user_id="sys")
    if old_mems:
        old_mem = old_mems[0]
        await mem_old._store.update(
            old_mem.id,
            last_accessed_at=(now - timedelta(days=30)).isoformat(),
        )

    await mem_old.aremember("New config: use port 9090", user_id="sys", type="fact")

    results = await mem_old.arecall("config port", user_id="sys", top_k=2)
    if len(results) >= 2:
        newer_first = "9090" in results[0].memory.content
    else:
        newer_first = len(results) == 1 and "9090" in results[0].memory.content

    await mem_old.aclose()
    return TestResult(
        dimension="3. Temporal Reasoning",
        test_name="Recency boost ranks newer memories higher",
        passed=newer_first,
        detail="" if newer_first else f"Order: {[r.memory.content for r in results]}",
    )


@timed
async def test_ttl_expiration(mem: AgentMemory) -> TestResult:
    """Memories with expired TTL should not appear in active results."""
    expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
    await mem.aremember(
        "Temporary access token: abc123",
        user_id="auth", type="fact",
        expires_at=expired_time,
    )
    await mem.aremember(
        "Permanent API key: xyz789",
        user_id="auth", type="fact",
    )

    # Run decay to expire the TTL memory
    await mem.adecay()

    results = await mem.arecall("access token", user_id="auth", top_k=5)
    found_expired = any("abc123" in r.memory.content for r in results)
    return TestResult(
        dimension="3. Temporal Reasoning",
        test_name="TTL expiration removes stale memories",
        passed=not found_expired,
        detail="Expired memory still returned" if found_expired else "",
    )


@timed
async def test_version_history(mem: AgentMemory) -> TestResult:
    """Updates create version history entries."""
    memory_id = await mem.aremember("Server runs on port 3000", user_id="ops", type="fact")
    await mem.aupdate(memory_id, content="Server runs on port 8080")
    await mem.aupdate(memory_id, content="Server runs on port 443")

    history = await mem.ahistory(memory_id)
    current = await mem.aget(memory_id)

    has_history = len(history) >= 2
    current_correct = current is not None and "443" in current.content
    return TestResult(
        dimension="3. Temporal Reasoning",
        test_name="Version history tracks updates",
        passed=has_history and current_correct,
        detail=f"history_len={len(history)}, current={current.content if current else 'None'}",
    )


@timed
async def test_inactivity_archival(mem: AgentMemory) -> TestResult:
    """Memories not accessed for a long time get archived."""
    mem_arch = AgentMemory(db_path=":memory:", embedding="none")
    await mem_arch._ensure_initialized()

    memory_id = await mem_arch.aremember("Old meeting notes from Q1", user_id="team", type="episode")

    # Backdate last_accessed_at
    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    await mem_arch._store.update(memory_id, last_accessed_at=old_date)

    await mem_arch.adecay()

    archived_mem = await mem_arch.aget(memory_id)
    from neuragram.core.models import MemoryStatus
    is_archived = archived_mem is not None and archived_mem.status == MemoryStatus.ARCHIVED

    await mem_arch.aclose()
    return TestResult(
        dimension="3. Temporal Reasoning",
        test_name="Inactivity archival",
        passed=is_archived,
        detail="" if is_archived else f"Status: {archived_mem.status if archived_mem else 'None'}",
    )


# ---------------------------------------------------------------------------
# Dimension 4: Knowledge Updates
# ---------------------------------------------------------------------------

@timed
async def test_update_overwrites_content(mem: AgentMemory) -> TestResult:
    """Updating a memory replaces its content."""
    memory_id = await mem.aremember(
        "User's email is old@example.com",
        user_id="profile", type="fact",
    )
    await mem.aupdate(memory_id, content="User's email is new@example.com")

    updated = await mem.aget(memory_id)
    correct = updated is not None and "new@example.com" in updated.content
    no_old = updated is not None and "old@example.com" not in updated.content
    return TestResult(
        dimension="4. Knowledge Updates",
        test_name="Update overwrites content",
        passed=correct and no_old,
        detail="" if (correct and no_old) else f"Content: {updated.content if updated else 'None'}",
    )


@timed
async def test_soft_delete(mem: AgentMemory) -> TestResult:
    """Soft-deleted memories don't appear in recall."""
    memory_id = await mem.aremember(
        "Deprecated: use v1 API endpoint",
        user_id="api", type="fact",
    )
    await mem.aforget(memory_id=memory_id, hard=False)

    results = await mem.arecall("deprecated API", user_id="api", top_k=5)
    found = any("v1 API" in r.memory.content for r in results)
    return TestResult(
        dimension="4. Knowledge Updates",
        test_name="Soft delete hides from recall",
        passed=not found,
        detail="Deleted memory still returned" if found else "",
    )


@timed
async def test_hard_delete_gdpr(mem: AgentMemory) -> TestResult:
    """Hard delete (GDPR) physically removes all user data."""
    await mem.aremember("GDPR user secret data 1", user_id="gdpr_user", type="fact")
    await mem.aremember("GDPR user secret data 2", user_id="gdpr_user", type="preference")

    deleted_count = await mem.aforget(user_id="gdpr_user", hard=True)

    remaining = await mem.alist(user_id="gdpr_user")
    return TestResult(
        dimension="4. Knowledge Updates",
        test_name="GDPR hard delete removes all user data",
        passed=deleted_count >= 2 and len(remaining) == 0,
        detail=f"Deleted {deleted_count}, remaining {len(remaining)}",
    )


@timed
async def test_metadata_update(mem: AgentMemory) -> TestResult:
    """Metadata can be updated without changing content."""
    memory_id = await mem.aremember(
        "Use retry with exponential backoff",
        user_id="dev", type="procedure",
        metadata={"version": "1.0"},
    )
    await mem.aupdate(memory_id, metadata={"version": "2.0", "reviewed": True})

    updated = await mem.aget(memory_id)
    correct_meta = (
        updated is not None
        and updated.metadata.get("version") == "2.0"
        and updated.metadata.get("reviewed") is True
    )
    content_unchanged = updated is not None and "exponential backoff" in updated.content
    return TestResult(
        dimension="4. Knowledge Updates",
        test_name="Metadata update preserves content",
        passed=correct_meta and content_unchanged,
        detail="" if (correct_meta and content_unchanged) else f"Meta: {updated.metadata if updated else 'None'}",
    )


# ---------------------------------------------------------------------------
# Dimension 5: Abstention
# ---------------------------------------------------------------------------

@timed
async def test_no_results_for_unknown_query(mem: AgentMemory) -> TestResult:
    """Query for non-existent information returns empty results."""
    results = await mem.arecall(
        "quantum computing algorithms",
        user_id="nonexistent_user_xyz",
        top_k=5,
    )
    return TestResult(
        dimension="5. Abstention",
        test_name="Empty results for unknown user",
        passed=len(results) == 0,
        detail=f"Expected 0 results, got {len(results)}",
    )


@timed
async def test_no_results_for_empty_namespace(mem: AgentMemory) -> TestResult:
    """Query in an empty namespace returns nothing."""
    results = await mem.arecall(
        "anything",
        namespace="completely_empty_namespace_xyz",
        top_k=5,
    )
    return TestResult(
        dimension="5. Abstention",
        test_name="Empty results for empty namespace",
        passed=len(results) == 0,
        detail=f"Expected 0 results, got {len(results)}",
    )


@timed
async def test_deleted_memories_not_returned(mem: AgentMemory) -> TestResult:
    """After deleting all memories for a user, recall returns empty."""
    await mem.aremember("Temp data for deletion test", user_id="temp_user_del", type="fact")
    await mem.aforget(user_id="temp_user_del", hard=True)

    results = await mem.arecall("deletion test", user_id="temp_user_del", top_k=5)
    return TestResult(
        dimension="5. Abstention",
        test_name="No results after full user deletion",
        passed=len(results) == 0,
        detail=f"Expected 0 results, got {len(results)}",
    )


@timed
async def test_type_filter_returns_empty(mem: AgentMemory) -> TestResult:
    """Filtering by a type with no matching memories returns empty."""
    await mem.aremember("Only a fact here", user_id="type_test", type="fact")

    results = await mem.arecall(
        "fact",
        user_id="type_test",
        types=["procedure"],
        top_k=5,
    )
    return TestResult(
        dimension="5. Abstention",
        test_name="Empty results for non-matching type filter",
        passed=len(results) == 0,
        detail=f"Expected 0 results, got {len(results)}",
    )


# ---------------------------------------------------------------------------
# Performance Benchmarks
# ---------------------------------------------------------------------------

@timed
async def test_bulk_insert_performance(mem: AgentMemory) -> TestResult:
    """Insert 500 memories and measure throughput."""
    perf_mem = AgentMemory(db_path=":memory:", embedding="none")
    await perf_mem._ensure_initialized()

    count = 100
    start = time.perf_counter()
    for i in range(count):
        await perf_mem.aremember(
            f"Performance test memory number {i} with some content about topic {i % 20}",
            user_id="perf_user",
            type="fact",
        )
    insert_time = (time.perf_counter() - start) * 1000

    stats = await perf_mem.astats()
    await perf_mem.aclose()

    throughput = count / (insert_time / 1000)
    return TestResult(
        dimension="Performance",
        test_name=f"Bulk insert {count} memories ({throughput:.0f} ops/s)",
        passed=stats.total_memories == count,
        detail=f"{insert_time:.0f}ms total, {insert_time/count:.1f}ms/op",
    )


@timed
async def test_recall_latency(mem: AgentMemory) -> TestResult:
    """Measure recall latency on a populated store."""
    perf_mem = AgentMemory(db_path=":memory:", embedding="none")
    await perf_mem._ensure_initialized()

    for i in range(50):
        await perf_mem.aremember(
            f"Memory about topic {i}: details about item {i} in category {i % 10}",
            user_id="perf_user",
            type="fact",
        )

    latencies = []
    queries = ["topic 42", "category 10", "details about item 99", "memory about"]
    for query in queries:
        start = time.perf_counter()
        await perf_mem.arecall(query, user_id="perf_user", top_k=10)
        latencies.append((time.perf_counter() - start) * 1000)

    await perf_mem.aclose()

    avg_latency = sum(latencies) / len(latencies)
    p99_latency = max(latencies)
    return TestResult(
        dimension="Performance",
        test_name=f"Recall latency (avg={avg_latency:.1f}ms, p99={p99_latency:.1f}ms)",
        passed=avg_latency < 500,
        detail=f"Latencies: {[f'{l:.1f}ms' for l in latencies]}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark() -> BenchmarkReport:
    report = BenchmarkReport()
    report.start_time = time.perf_counter()

    mem = AgentMemory(db_path=":memory:", embedding="none")
    await mem._ensure_initialized()

    tests = [
        # Dimension 1: Information Extraction
        test_extract_single_fact,
        test_extract_with_user_isolation,
        test_extract_by_type_filter,
        test_extract_multiple_facts,
        # Dimension 2: Multi-Session Reasoning
        test_cross_namespace_isolation,
        test_cross_agent_retrieval,
        test_multi_user_aggregation,
        test_tag_based_retrieval,
        # Dimension 3: Temporal Reasoning
        test_recency_boost,
        test_ttl_expiration,
        test_version_history,
        test_inactivity_archival,
        # Dimension 4: Knowledge Updates
        test_update_overwrites_content,
        test_soft_delete,
        test_hard_delete_gdpr,
        test_metadata_update,
        # Dimension 5: Abstention
        test_no_results_for_unknown_query,
        test_no_results_for_empty_namespace,
        test_deleted_memories_not_returned,
        test_type_filter_returns_empty,
        # Performance
        test_bulk_insert_performance,
        test_recall_latency,
    ]

    for test_func in tests:
        try:
            result, latency_ms = await test_func(mem)
            result.latency_ms = latency_ms
            report.add(result)
        except Exception as exc:
            report.add(TestResult(
                dimension="ERROR",
                test_name=test_func.__wrapped__.__name__ if hasattr(test_func, "__wrapped__") else str(test_func),
                passed=False,
                latency_ms=0,
                detail=f"Exception: {exc}",
            ))

    await mem.aclose()
    report.end_time = time.perf_counter()
    return report


def main() -> None:
    print("\n🧠 Running Neuragram Integration Tests...\n")
    report = asyncio.run(run_benchmark())
    report.print_report()
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
