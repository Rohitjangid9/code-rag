"""F30 — Server auth + rate limit middleware.

Two independent Starlette middleware layers:

1. **APIKeyMiddleware** — checks the ``X-API-Key`` header against
   ``ServerSettings.api_keys``.  A 401 is returned when the key is absent or
   invalid.  Bypassed entirely when ``api_keys`` is empty (dev mode).

2. **RateLimitMiddleware** — token-bucket limiter keyed on client IP.
   Bucket refills at ``rate_limit_rpm / 60`` tokens per second; capacity equals
   ``rate_limit_rpm``.  Returns 429 when the bucket is empty.  Bypassed when
   ``rate_limit_rpm == 0``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from cce.config import get_settings
from cce.logging import get_logger

log = get_logger(__name__)

# Paths that never require auth (health, docs)
_AUTH_BYPASS: frozenset[str] = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate ``X-API-Key`` header on every request (F30)."""

    async def dispatch(self, request: Request, call_next):
        cfg = get_settings().server
        if not cfg.api_keys:
            return await call_next(request)  # dev mode — no keys configured
        if request.url.path in _AUTH_BYPASS:
            return await call_next(request)
        key = request.headers.get("X-API-Key", "")
        if key not in cfg.api_keys:
            log.warning("Rejected request — invalid or missing API key from %s", request.client)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header."},
            )
        return await call_next(request)


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class _Bucket:
    """Single-client token bucket."""
    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def consume(self, capacity: float, rate: float) -> bool:
        """Refill then attempt to consume one token.  Returns True on success."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(capacity, self.tokens + elapsed * rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_BUCKETS: dict[str, _Bucket] = defaultdict()
_BUCKET_LOCK = Lock()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter keyed on client IP (F30).

    Capacity = ``rate_limit_rpm``; refill rate = ``rate_limit_rpm / 60`` t/s.
    """

    async def dispatch(self, request: Request, call_next):
        cfg = get_settings().server
        rpm = cfg.rate_limit_rpm
        if rpm <= 0:
            return await call_next(request)  # unlimited
        if request.url.path in _AUTH_BYPASS:
            return await call_next(request)

        client_ip = (request.client.host if request.client else "unknown")
        rate = rpm / 60.0  # tokens per second
        capacity = float(rpm)

        with _BUCKET_LOCK:
            bucket = _BUCKETS.setdefault(client_ip, _Bucket(capacity))
            allowed = bucket.consume(capacity, rate)

        if not allowed:
            log.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {rpm} requests/min."},
                headers={"Retry-After": str(int(60 / rate))},
            )
        return await call_next(request)
