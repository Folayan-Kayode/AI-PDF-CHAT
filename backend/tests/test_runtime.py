"""Tests for the single-worker topology guard (B4)."""

import pytest

from app.core.config import settings
from app.core.runtime import (
    UnsupportedTopologyError,
    assert_single_worker,
    worker_count,
)


def test_a_single_worker_starts(monkeypatch):
    monkeypatch.setattr(settings, "WEB_CONCURRENCY", 1)

    assert_single_worker()  # must not raise

    assert worker_count() == 1


def test_more_than_one_worker_is_refused(monkeypatch):
    # This is the accidental case: Render and Heroku set WEB_CONCURRENCY for
    # you, and embedded Chroma plus the process-wide ingest lock cannot take it.
    monkeypatch.setattr(settings, "WEB_CONCURRENCY", 2)
    monkeypatch.setattr(settings, "MULTI_WORKER_ACK", False)

    with pytest.raises(UnsupportedTopologyError, match="WEB_CONCURRENCY=2"):
        assert_single_worker()


def test_multiple_workers_are_allowed_with_an_explicit_ack(monkeypatch):
    monkeypatch.setattr(settings, "WEB_CONCURRENCY", 2)
    monkeypatch.setattr(settings, "MULTI_WORKER_ACK", True)

    assert_single_worker()  # must not raise

    assert worker_count() == 2


def test_zero_or_negative_workers_reports_one(monkeypatch):
    monkeypatch.setattr(settings, "WEB_CONCURRENCY", 0)

    assert worker_count() == 1
