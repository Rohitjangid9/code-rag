"""F33 — OpenTelemetry tracer factory.

Tracing is gated by the ``CCE_OTEL_ENDPOINT`` environment variable (or
``CCE_OTEL_ENABLED=true``).  When neither is set the module returns a
no-op tracer so zero cost is incurred in dev/test.

Usage::

    from cce.telemetry import get_tracer
    tracer = get_tracer()

    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("query", query)
        ...

Environment variables::

    CCE_OTEL_ENDPOINT   OTLP/gRPC endpoint (e.g. http://localhost:4317)
    CCE_OTEL_SERVICE    Service name (default: code-context-engine)
    CCE_OTEL_ENABLED    Set to "true" to enable with a no-op exporter (testing)
"""

from __future__ import annotations

import os
from functools import lru_cache

_SERVICE_NAME = os.getenv("CCE_OTEL_SERVICE", "code-context-engine")


@lru_cache(maxsize=1)
def get_tracer():
    """Return a cached OpenTelemetry tracer, or a no-op tracer if OTel is off."""
    endpoint = os.getenv("CCE_OTEL_ENDPOINT", "")
    enabled = os.getenv("CCE_OTEL_ENABLED", "false").lower() in ("1", "true", "yes")

    if not endpoint and not enabled:
        # Return a no-op tracer — zero import overhead in dev/test
        return _NoOpTracer()

    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError:
        return _NoOpTracer()

    resource = Resource.create({"service.name": _SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass  # opentelemetry-exporter-otlp not installed

    trace.set_tracer_provider(provider)
    return trace.get_tracer(_SERVICE_NAME)


# ── No-op tracer (used when OTel is not configured) ──────────────────────────

class _NoOpSpan:
    """Minimal span stub that satisfies ``with tracer.start_as_current_span(...)``."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value) -> None:
        pass

    def record_exception(self, exc) -> None:
        pass

    def set_status(self, *args, **kwargs) -> None:
        pass


class _NoOpTracer:
    """Tracer that produces _NoOpSpan on every call."""

    def start_as_current_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()
