"""OpenTelemetry integration for Engram observability.

Provides automatic instrumentation of Engram operations with
traces, spans, and metrics. When OpenTelemetry is not installed,
all instrumentation is silently no-op.

Tracked operations:
    - remember / smart_remember: span with memory type, user_id
    - recall: span with query, result count, latency
    - forget: span with deletion count
    - decay: span with expired/archived counts
    - store operations: spans for insert, search, update, delete

Metrics:
    - engram.memories.total: Gauge of total active memories
    - engram.operations.count: Counter of operations by type
    - engram.operations.duration: Histogram of operation latencies
    - engram.recall.results: Histogram of result counts per recall

Usage::

    # Auto-instruments when OpenTelemetry is available
    from engram.core.telemetry import get_tracer, get_meter

    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation"):
        ...

Requires: pip install engram[telemetry]
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

# Try to import OpenTelemetry; fall back to no-ops if unavailable
_OTEL_AVAILABLE = False

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Tracer, Span, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    pass


class _NoOpSpan:
    """No-op span when OpenTelemetry is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpCounter:
    """No-op counter metric."""

    def add(self, amount: int = 1, attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpHistogram:
    """No-op histogram metric."""

    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpGauge:
    """No-op gauge metric."""

    def set(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpMeter:
    """No-op meter when OpenTelemetry is not available."""

    def create_counter(self, name: str, **kwargs: Any) -> _NoOpCounter:
        return _NoOpCounter()

    def create_histogram(self, name: str, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()

    def create_up_down_counter(self, name: str, **kwargs: Any) -> _NoOpCounter:
        return _NoOpCounter()


def get_tracer(name: str = "engram") -> Any:
    """Get an OpenTelemetry tracer, or a no-op tracer if OTel is unavailable.

    Returns:
        A Tracer instance (real or no-op).
    """
    if _OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return _NoOpTracer()


def get_meter(name: str = "engram") -> Any:
    """Get an OpenTelemetry meter, or a no-op meter if OTel is unavailable.

    Returns:
        A Meter instance (real or no-op).
    """
    if _OTEL_AVAILABLE:
        return metrics.get_meter(name)
    return _NoOpMeter()


def is_otel_available() -> bool:
    """Check if OpenTelemetry is installed and available."""
    return _OTEL_AVAILABLE


# ── Pre-built instruments ───────────────────────────────────────────

_tracer = get_tracer("engram")
_meter = get_meter("engram")

# Metrics
operation_counter = _meter.create_counter(
    "engram.operations.count",
    description="Number of Engram operations by type",
)

operation_duration = _meter.create_histogram(
    "engram.operations.duration",
    description="Duration of Engram operations in seconds",
)

recall_results_histogram = _meter.create_histogram(
    "engram.recall.results",
    description="Number of results returned per recall operation",
)

memory_gauge = _meter.create_up_down_counter(
    "engram.memories.total",
    description="Total number of active memories",
)


@contextmanager
def traced_operation(
    operation_name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager that creates a span and records metrics for an operation.

    Usage::

        with traced_operation("remember", {"user_id": "u1"}) as span:
            # do work
            span.set_attribute("memory_id", mid)

    Args:
        operation_name: Name of the operation (e.g., "remember", "recall").
        attributes: Initial span attributes.

    Yields:
        The active span (real or no-op).
    """
    start_time = time.monotonic()
    span = _tracer.start_as_current_span(
        f"engram.{operation_name}",
    )

    try:
        with span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

            operation_counter.add(1, {"operation": operation_name, "status": "success"})
    except Exception as exc:
        operation_counter.add(1, {"operation": operation_name, "status": "error"})
        if _OTEL_AVAILABLE:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
        raise
    finally:
        duration = time.monotonic() - start_time
        operation_duration.record(duration, {"operation": operation_name})
