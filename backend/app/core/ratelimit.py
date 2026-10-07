"""
In-process rate limiting for the paid endpoints.

A token bucket per (scope, API key), refilling continuously. In-process is the
right shape here because the app is pinned to a single worker (see
``app.core.runtime``): the limit is about bounding one caller's spend, not about
coordinating a fleet.

The bucket starts full, so a legitimate first use is never blocked. A rejected
request gets a 429 and a ``Retry-After`` header, matching the shape the UI
already understands (``frontend/app.py::retry_hint``).
"""

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from app.core.config import settings
from app.core.security import API_KEY_HEADER


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class RateLimiter:
    """A token bucket keyed by scope and identity."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, scope: str, identity: str, limit_per_hour: int) -> float | None:
        """
        Consume one token.

        Returns ``None`` when the call is allowed, otherwise the number of
        seconds the caller should wait before retrying.
        """
        if limit_per_hour <= 0:
            return None

        capacity = float(limit_per_hour)
        refill_per_second = capacity / 3600.0

        now = self._clock()
        key = (scope, identity)

        with self._lock:
            bucket = self._buckets.get(key)

            if bucket is None:
                bucket = _Bucket(tokens=capacity, updated_at=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(capacity, bucket.tokens + elapsed * refill_per_second)
                bucket.updated_at = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0

                return None

            deficit = 1.0 - bucket.tokens

            return deficit / refill_per_second

    def reset(self) -> None:
        """Drop all buckets (tests, and a clean slate at startup)."""
        with self._lock:
            self._buckets.clear()


limiter = RateLimiter()


def reset_rate_limits() -> None:
    """Public hook so tests and startup do not reach into the limiter."""
    limiter.reset()


def _enforce(
    scope: str,
    limit_per_hour: int,
    x_api_key: str | None,
    request: Request,
) -> None:
    identity = x_api_key or (request.client.host if request.client else "unknown")

    retry_after = limiter.check(scope, identity, limit_per_hour)

    if retry_after is None:
        return

    wait = max(1, int(math.ceil(retry_after)))

    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded for {scope}. Try again in about {wait} seconds.",
        headers={"Retry-After": str(wait)},
    )


async def upload_rate_limit(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """Bound how many documents one caller may ingest per hour."""
    _enforce("upload", settings.UPLOAD_RATE_LIMIT_PER_HOUR, x_api_key, request)


async def chat_rate_limit(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """Bound how many questions one caller may ask per hour."""
    _enforce("chat", settings.CHAT_RATE_LIMIT_PER_HOUR, x_api_key, request)
