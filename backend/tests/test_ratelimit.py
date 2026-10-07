"""Tests for the in-process token-bucket rate limiter."""

from app.core.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_capacity_then_denies():
    limiter = RateLimiter(clock=FakeClock())

    for _ in range(3):
        assert limiter.check("chat", "key", 3) is None

    wait = limiter.check("chat", "key", 3)

    assert wait is not None
    assert wait > 0


def test_refills_over_time():
    clock = FakeClock()
    limiter = RateLimiter(clock=clock)

    for _ in range(2):
        assert limiter.check("chat", "key", 2) is None

    assert limiter.check("chat", "key", 2) is not None

    # Half an hour at 2/hour is one token back.
    clock.now += 1800

    assert limiter.check("chat", "key", 2) is None


def test_does_not_refill_beyond_capacity():
    clock = FakeClock()
    limiter = RateLimiter(clock=clock)

    clock.now += 10_000  # far longer than a full refill

    for _ in range(2):
        assert limiter.check("chat", "key", 2) is None

    # Still denied: the bucket is capped at its capacity, not overfilled.
    assert limiter.check("chat", "key", 2) is not None


def test_buckets_are_isolated_by_identity_and_scope():
    limiter = RateLimiter(clock=FakeClock())

    assert limiter.check("chat", "key-a", 1) is None
    assert limiter.check("chat", "key-a", 1) is not None

    # A different identity has its own bucket.
    assert limiter.check("chat", "key-b", 1) is None

    # A different scope has its own bucket too.
    assert limiter.check("upload", "key-a", 1) is None


def test_zero_limit_is_unlimited():
    limiter = RateLimiter(clock=FakeClock())

    for _ in range(100):
        assert limiter.check("chat", "key", 0) is None


def test_reset_clears_every_bucket():
    limiter = RateLimiter(clock=FakeClock())

    assert limiter.check("chat", "key", 1) is None
    assert limiter.check("chat", "key", 1) is not None

    limiter.reset()

    assert limiter.check("chat", "key", 1) is None
