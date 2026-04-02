"""Tests for Phase 4: Access Control, Lifecycle Worker, Telemetry, Namespace Stats."""

from __future__ import annotations

import asyncio
import pytest

from engram import (
    AccessDeniedError,
    AccessLevel,
    AccessPolicy,
    AgentMemory,
    LifecycleWorker,
    WorkerStats,
    is_otel_available,
    traced_operation,
)


# ── Access Control Tests ────────────────────────────────────────────


def test_access_level_ordering():
    """AccessLevel values are ordered: NONE < READ < WRITE < ADMIN."""
    assert AccessLevel.NONE < AccessLevel.READ
    assert AccessLevel.READ < AccessLevel.WRITE
    assert AccessLevel.WRITE < AccessLevel.ADMIN


def test_access_policy_disabled_by_default():
    """AccessPolicy is disabled by default (all operations allowed)."""
    policy = AccessPolicy()
    assert not policy.enabled
    assert policy.check("anyone", AccessLevel.ADMIN)


def test_access_policy_grant_and_check():
    """Grants are correctly checked."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("reader", AccessLevel.READ)
    policy.grant("writer", AccessLevel.WRITE)
    policy.grant("admin", AccessLevel.ADMIN)

    assert policy.check("reader", AccessLevel.READ)
    assert not policy.check("reader", AccessLevel.WRITE)

    assert policy.check("writer", AccessLevel.READ)
    assert policy.check("writer", AccessLevel.WRITE)
    assert not policy.check("writer", AccessLevel.ADMIN)

    assert policy.check("admin", AccessLevel.ADMIN)


def test_access_policy_namespace_scoping():
    """Grants can be scoped to specific namespaces."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("agent1", AccessLevel.WRITE, namespace="project_a")
    policy.grant("agent1", AccessLevel.READ, namespace="project_b")

    assert policy.check("agent1", AccessLevel.WRITE, namespace="project_a")
    assert not policy.check("agent1", AccessLevel.WRITE, namespace="project_b")
    assert policy.check("agent1", AccessLevel.READ, namespace="project_b")


def test_access_policy_user_scoping():
    """Grants can be scoped to specific users."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("agent1", AccessLevel.WRITE, user_id="u1")

    assert policy.check("agent1", AccessLevel.WRITE, user_id="u1")
    assert not policy.check("agent1", AccessLevel.WRITE, user_id="u2")


def test_access_policy_enforce_raises():
    """enforce() raises AccessDeniedError when permission is insufficient."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("reader", AccessLevel.READ)

    with pytest.raises(AccessDeniedError, match="reader"):
        policy.enforce("reader", AccessLevel.WRITE, "remember")


def test_access_policy_revoke():
    """revoke() removes grants for an actor."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("agent1", AccessLevel.ADMIN)
    assert policy.check("agent1", AccessLevel.ADMIN)

    policy.revoke("agent1")
    assert not policy.check("agent1", AccessLevel.READ)


def test_access_policy_list_grants():
    """list_grants() returns all grants."""
    policy = AccessPolicy(enabled=True)
    policy.grant("a1", AccessLevel.READ)
    policy.grant("a2", AccessLevel.WRITE, namespace="ns1")

    all_grants = policy.list_grants()
    assert len(all_grants) == 2

    a1_grants = policy.list_grants(actor_id="a1")
    assert len(a1_grants) == 1
    assert a1_grants[0]["level"] == "READ"


def test_access_policy_default_level():
    """Unregistered actors get the default level."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.READ)
    assert policy.check("unknown_actor", AccessLevel.READ)
    assert not policy.check("unknown_actor", AccessLevel.WRITE)


def test_access_policy_global_grant_covers_all_namespaces():
    """A global grant (namespace=None) covers all namespaces."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
    policy.grant("admin", AccessLevel.ADMIN)

    assert policy.check("admin", AccessLevel.ADMIN, namespace="any_ns")
    assert policy.check("admin", AccessLevel.ADMIN, user_id="any_user")


# ── Client Access Policy Integration Tests ──────────────────────────


@pytest.mark.asyncio
async def test_client_access_policy_property(tmp_path):
    """AgentMemory exposes access_policy property."""
    policy = AccessPolicy(enabled=True, default_level=AccessLevel.READ)
    mem = AgentMemory(
        db_path=str(tmp_path / "access_test.db"),
        access_policy=policy,
    )
    assert mem.access_policy is policy
    assert mem.access_policy.enabled
    mem.close()


# ── Lifecycle Worker Tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_run_once(tmp_path):
    """Worker.run_once() executes a single maintenance cycle."""
    mem = AgentMemory(db_path=str(tmp_path / "worker_test.db"))
    await mem._ensure_initialized()

    worker = LifecycleWorker(memory=mem, interval_seconds=60)
    stats = await worker.run_once()

    assert isinstance(stats, WorkerStats)
    assert stats.cycle_number == 1
    assert stats.completed_at is not None
    assert stats.duration_seconds >= 0
    assert len(stats.errors) == 0

    await mem.aclose()


@pytest.mark.asyncio
async def test_worker_start_stop(tmp_path):
    """Worker can be started and stopped."""
    mem = AgentMemory(db_path=str(tmp_path / "worker_test2.db"))
    await mem._ensure_initialized()

    worker = LifecycleWorker(memory=mem, interval_seconds=0.1)
    assert not worker.is_running

    await worker.start()
    assert worker.is_running

    await asyncio.sleep(0.3)  # Let a few cycles run
    assert worker.cycle_count >= 1

    await worker.stop()
    assert not worker.is_running

    await mem.aclose()


@pytest.mark.asyncio
async def test_worker_context_manager(tmp_path):
    """Worker works as async context manager."""
    mem = AgentMemory(db_path=str(tmp_path / "worker_test3.db"))
    await mem._ensure_initialized()

    async with LifecycleWorker(memory=mem, interval_seconds=0.1) as worker:
        assert worker.is_running
        await asyncio.sleep(0.2)

    assert not worker.is_running
    await mem.aclose()


@pytest.mark.asyncio
async def test_worker_created_from_client(tmp_path):
    """create_worker() returns a properly configured worker."""
    mem = AgentMemory(db_path=str(tmp_path / "worker_test4.db"))
    await mem._ensure_initialized()

    worker = mem.create_worker(interval_seconds=60)
    assert isinstance(worker, LifecycleWorker)
    assert not worker.is_running

    stats = await worker.run_once()
    assert stats.cycle_number == 1

    await mem.aclose()


# ── Telemetry Tests ─────────────────────────────────────────────────


def test_otel_availability():
    """is_otel_available() returns a boolean."""
    result = is_otel_available()
    assert isinstance(result, bool)


def test_traced_operation_noop():
    """traced_operation works as no-op when OTel is not installed."""
    with traced_operation("test_op", {"key": "value"}) as span:
        span.set_attribute("result", "ok")
    # Should not raise


def test_traced_operation_records_error():
    """traced_operation handles exceptions gracefully."""
    with pytest.raises(ValueError, match="test error"):
        with traced_operation("failing_op"):
            raise ValueError("test error")


# ── Namespace Stats Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_namespace_stats_specific(tmp_path):
    """namespace_stats() returns stats for a specific namespace."""
    mem = AgentMemory(db_path=str(tmp_path / "ns_test.db"))
    await mem._ensure_initialized()

    await mem.aremember("Fact in ns1", namespace="ns1", type="fact")
    await mem.aremember("Pref in ns1", namespace="ns1", type="preference")
    await mem.aremember("Fact in ns2", namespace="ns2", type="fact")

    stats = await mem.anamespace_stats(namespace="ns1")
    assert stats["namespace"] == "ns1"
    assert stats["total_memories"] == 2
    assert stats["memory_types"]["fact"] == 1
    assert stats["memory_types"]["preference"] == 1

    await mem.aclose()


@pytest.mark.asyncio
async def test_namespace_stats_all(tmp_path):
    """namespace_stats() without namespace returns all namespaces."""
    mem = AgentMemory(db_path=str(tmp_path / "ns_test2.db"))
    await mem._ensure_initialized()

    await mem.aremember("A", namespace="alpha")
    await mem.aremember("B", namespace="alpha")
    await mem.aremember("C", namespace="beta")

    stats = await mem.anamespace_stats()
    assert stats["total_namespaces"] == 2
    assert stats["total_memories"] == 3
    assert stats["namespaces"]["alpha"] == 2
    assert stats["namespaces"]["beta"] == 1

    await mem.aclose()


def test_namespace_stats_sync(tmp_path):
    """Sync namespace_stats() works correctly."""
    mem = AgentMemory(db_path=str(tmp_path / "ns_test3.db"))
    mem.remember("X", namespace="gamma")
    stats = mem.namespace_stats(namespace="gamma")
    assert stats["total_memories"] == 1
    mem.close()
