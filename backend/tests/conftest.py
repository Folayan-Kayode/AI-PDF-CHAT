"""
Shared pytest fixtures.

The environment is configured before any application module is imported,
because settings are read at import time. Without this, the tests would
write to the real uploads/ and chroma_db/ directories and would need live
API keys.
"""

import os
import tempfile
from pathlib import Path

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="ai-pdf-chat-tests-"))

os.environ["GOOGLE_API_KEY"] = "test-google-key"
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
os.environ["UPLOAD_DIRECTORY"] = str(_TMP_ROOT / "uploads")
os.environ["CHROMA_DIRECTORY"] = str(_TMP_ROOT / "chroma_db")
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["EMBEDDING_BATCH_DELAY_SECONDS"] = "0"
os.environ["REGISTRY_PATH"] = str(_TMP_ROOT / "registry.sqlite3")

# Retries must not make the suite sleep for real.
os.environ["EMBEDDING_RETRY_BASE_SECONDS"] = "0"
os.environ["CHROMA_WRITE_RETRY_BASE_SECONDS"] = "0"

# Query rewriting and reranking each spend a model call. Tests that care about
# them enable them explicitly with fakes injected; everything else runs
# without touching the network.
os.environ["QUERY_REWRITE_ENABLED"] = "false"
os.environ["RERANK_ENABLED"] = "false"
os.environ["DOCUMENT_SUMMARY_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory holding committed sample documents."""
    return FIXTURES_DIR


@pytest.fixture()
def sample_pdf_bytes(fixtures_dir: Path) -> bytes:
    """Bytes of the committed sample PDF."""
    return (fixtures_dir / "sample.pdf").read_bytes()


@pytest.fixture()
def client() -> TestClient:
    """Test client wired to the real app."""
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def neutral_retrieval_defaults(monkeypatch):
    """
    Pin the tunable retrieval knobs for unit tests.

    These values are tuned from the evaluation set and will change again, so
    tests must not depend on whichever number currently ships. Tests that care
    about a threshold set it themselves.
    """
    from app.core.config import settings

    # Selection is relative now; widen it so a unit test that is not about
    # filtering sees every candidate regardless of the tuned margins.
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 10.0)
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 10.0)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 10.0)
    # 0 means "rerank whenever it is enabled", the previous behaviour.
    monkeypatch.setattr(settings, "RERANK_SKIP_RATIO", 0.0)
    # Unit tests that are not about the token budget should not hit it.
    monkeypatch.setattr(settings, "MAX_CONTEXT_TOKENS", 10**6)


@pytest.fixture(autouse=True)
def clear_singletons():
    """
    Drop cached clients between tests.

    The app deliberately caches its database, embedding model, generator and
    pipeline for the life of the process; tests need a clean slate.
    """
    yield

    from app.database.chroma import get_database
    from app.rag.embeddings import get_embedding_model
    from app.rag.generator import get_generator
    from app.rag.pipeline import get_pipeline

    for cached in (
        get_database,
        get_embedding_model,
        get_generator,
        get_pipeline,
    ):
        cached.cache_clear()
