"""Tests for the daily ingest budget (B9).

The guard is checked before embedding and charged after, so a rejected ingest
never costs anything and a failed one does not consume budget.
"""

import pytest

from app.core.config import settings
from app.core.exceptions import IngestBudgetExceededError
from app.database.registry import DocumentRegistry
from app.services import pdf_service


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    instance = DocumentRegistry(path=tmp_path / "registry.sqlite3")

    monkeypatch.setattr(pdf_service, "get_registry", lambda: instance)

    return instance


def test_under_budget_is_allowed(monkeypatch, registry):
    monkeypatch.setattr(settings, "INGEST_DAILY_EMBEDDING_BATCH_BUDGET", 10)

    registry.add_usage(pdf_service._utc_day(), "embedding_batches", 4)

    pdf_service.PDFService._enforce_ingest_budget(3)  # 4 + 3 <= 10, no raise


def test_exceeding_budget_is_refused_with_retry_after(monkeypatch, registry):
    monkeypatch.setattr(settings, "INGEST_DAILY_EMBEDDING_BATCH_BUDGET", 10)

    registry.add_usage(pdf_service._utc_day(), "embedding_batches", 9)

    with pytest.raises(IngestBudgetExceededError) as raised:
        pdf_service.PDFService._enforce_ingest_budget(3)

    assert raised.value.status_code == 429
    assert int(raised.value.headers["Retry-After"]) >= 1


def test_charging_records_the_batches(monkeypatch, registry):
    monkeypatch.setattr(settings, "INGEST_DAILY_EMBEDDING_BATCH_BUDGET", 10)

    pdf_service.PDFService._charge_ingest_budget(4)
    pdf_service.PDFService._charge_ingest_budget(3)

    assert registry.usage(pdf_service._utc_day(), "embedding_batches") == 7


def test_a_zero_budget_disables_the_ceiling(monkeypatch, registry):
    monkeypatch.setattr(settings, "INGEST_DAILY_EMBEDDING_BATCH_BUDGET", 0)

    pdf_service.PDFService._enforce_ingest_budget(10**9)  # no raise

    pdf_service.PDFService._charge_ingest_budget(10**9)

    assert registry.usage(pdf_service._utc_day(), "embedding_batches") == 0


def test_an_unreadable_registry_does_not_block_the_ingest(monkeypatch):
    def explode():
        raise RuntimeError("registry is down")

    monkeypatch.setattr(pdf_service, "get_registry", explode)
    monkeypatch.setattr(settings, "INGEST_DAILY_EMBEDDING_BATCH_BUDGET", 1)

    # The budget is a guardrail, not the product: a broken registry must not
    # break ingestion.
    pdf_service.PDFService._enforce_ingest_budget(10**9)


def test_seconds_until_midnight_is_positive_and_under_a_day():
    seconds = pdf_service._seconds_until_utc_midnight()

    assert 0 < seconds <= 86400
