"""Tests for configuration handling."""

import re
from pathlib import Path

import pytest

from app.core import config
from app.core.config import Settings, settings

BACKEND_ROOT = Path(__file__).resolve().parents[1]

REPO_ROOT = BACKEND_ROOT.parent

CONFIG_SOURCE = BACKEND_ROOT / "app" / "core" / "config.py"

ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Any call that declares an environment variable name.
_DECLARED_PATTERN = re.compile(r'(?:_env\w*|_require)\(\s*"([A-Z0-9_]+)"')

# Uncommented KEY=value lines.
_DOCUMENTED_PATTERN = re.compile(r"^([A-Z0-9_]+)=", re.MULTILINE)


def test_env_returns_value(monkeypatch):
    monkeypatch.setenv("CONFIG_TEST_VALUE", "hello")

    assert config._env("CONFIG_TEST_VALUE", "fallback") == "hello"


def test_env_trims_whitespace(monkeypatch):
    monkeypatch.setenv("CONFIG_TEST_VALUE", "  hello  ")

    assert config._env("CONFIG_TEST_VALUE", "fallback") == "hello"


def test_blank_env_falls_back_to_default(monkeypatch):
    # `MODEL=` in .env should behave like it was never set, otherwise the
    # default never applies and the app breaks in a confusing way.
    monkeypatch.setenv("CONFIG_TEST_VALUE", "   ")

    assert config._env("CONFIG_TEST_VALUE", "fallback") == "fallback"


def test_unset_env_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("CONFIG_TEST_VALUE", raising=False)

    assert config._env("CONFIG_TEST_VALUE", "fallback") == "fallback"


def test_env_int_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("CONFIG_TEST_INT", "not-a-number")

    with pytest.raises(RuntimeError, match="CONFIG_TEST_INT"):
        config._env_int("CONFIG_TEST_INT", 1)


def test_env_float_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("CONFIG_TEST_FLOAT", "not-a-number")

    with pytest.raises(RuntimeError, match="CONFIG_TEST_FLOAT"):
        config._env_float("CONFIG_TEST_FLOAT", 1.0)


def test_require_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("CONFIG_TEST_KEY", raising=False)

    with pytest.raises(RuntimeError, match="CONFIG_TEST_KEY"):
        Settings._require("CONFIG_TEST_KEY")


def test_require_rejects_blank_key(monkeypatch):
    monkeypatch.setenv("CONFIG_TEST_KEY", "   ")

    with pytest.raises(RuntimeError, match="CONFIG_TEST_KEY"):
        Settings._require("CONFIG_TEST_KEY")


def test_require_strips_key(monkeypatch):
    monkeypatch.setenv("CONFIG_TEST_KEY", "  secret  ")

    assert Settings._require("CONFIG_TEST_KEY") == "secret"


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


def test_limits_are_sane():
    assert settings.MAX_UPLOAD_SIZE_BYTES > 0
    assert settings.MAX_PAGES_PER_DOCUMENT > 0
    assert settings.MAX_CHUNKS_PER_DOCUMENT > 0
    assert settings.MAX_FILENAME_LENGTH > 0
    assert settings.MAX_CONTEXT_CHARS > 0
    assert settings.RETRIEVAL_TOP_K > 0
    assert settings.RETRIEVAL_CANDIDATES >= settings.RETRIEVAL_TOP_K
    assert settings.RETRIEVAL_MAX_DISTANCE > 0
    assert settings.QUERY_REWRITE_MAX_CHARS > 0
    assert settings.RERANK_SKIP_DISTANCE >= 0
    assert settings.RERANK_SNIPPET_CHARS > 0
    assert settings.EMBEDDING_BATCH_SIZE > 0
    assert settings.EMBEDDING_MAX_RETRIES >= 1
    assert settings.EMBEDDING_REQUESTS_PER_MINUTE > 0
    assert settings.EMBEDDING_RETRY_BASE_SECONDS >= 0
    assert settings.CHROMA_WRITE_BATCH_SIZE > 0
    assert settings.CHROMA_WRITE_MAX_RETRIES >= 1
    assert settings.CHROMA_WRITE_RETRY_BASE_SECONDS >= 0
    assert settings.INGESTION_LOCK_TIMEOUT_SECONDS >= 0
    assert settings.HEALTH_DEEP_CACHE_SECONDS >= 0
    assert settings.EMBEDDING_SCHEMA_VERSION >= 1


def test_model_default_is_deepseek():
    assert settings.MODEL_NAME == "deepseek-chat"


def test_shipped_retrieval_defaults_are_sane():
    # The class attributes are the shipped defaults; the unit-test fixture
    # patches the instance, so these are the values a user actually gets.
    assert 0.0 < config.Settings.RETRIEVAL_MAX_DISTANCE <= 1.0
    assert 0.0 <= config.Settings.RERANK_SKIP_DISTANCE <= 1.0
    assert config.Settings.RETRIEVAL_CANDIDATES >= config.Settings.RETRIEVAL_TOP_K
    assert config.Settings.RETRIEVAL_TOP_K > 0


def test_profile_and_retrieval_switches_are_booleans():
    assert isinstance(settings.DOCUMENT_PROFILE_IN_CONTEXT, bool)
    assert isinstance(settings.QUERY_REWRITE_ENABLED, bool)
    assert isinstance(settings.RERANK_ENABLED, bool)
    assert isinstance(settings.DOCUMENT_SUMMARY_ENABLED, bool)


def test_embedding_pacing_follows_the_quota_budget(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_REQUESTS_PER_MINUTE", 60)

    assert settings.EMBEDDING_MIN_DELAY_SECONDS == pytest.approx(1.0)


def test_embedding_pacing_survives_a_zero_budget(monkeypatch):
    # A zero here would otherwise be a division by zero at import time.
    monkeypatch.setattr(settings, "EMBEDDING_REQUESTS_PER_MINUTE", 0)

    assert settings.EMBEDDING_MIN_DELAY_SECONDS > 0


# --------------------------------------------------------------------------
# Documentation parity (B9)
# --------------------------------------------------------------------------


def test_env_example_documents_every_setting():
    declared = set(_DECLARED_PATTERN.findall(CONFIG_SOURCE.read_text(encoding="utf-8")))
    documented = set(_DOCUMENTED_PATTERN.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))

    assert declared, "no settings were detected in config.py"

    assert declared == documented, (
        "missing from .env.example: "
        f"{sorted(declared - documented)}; "
        "not declared in config.py: "
        f"{sorted(documented - declared)}"
    )


def test_env_example_ships_no_real_secrets():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    for key in ("GOOGLE_API_KEY", "DEEPSEEK_API_KEY"):
        match = re.search(rf"^{key}=(.*)$", text, re.MULTILINE)

        assert match is not None, f"{key} is missing from .env.example"
        assert match.group(1).strip() == "", f"{key} must ship empty"
