"""Tests for configuration handling."""

import pytest

from app.core import config
from app.core.config import Settings, settings


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


def test_limits_are_sane():
    assert settings.MAX_UPLOAD_SIZE_BYTES > 0
    assert settings.MAX_PAGES_PER_DOCUMENT > 0
    assert settings.MAX_CHUNKS_PER_DOCUMENT > 0
    assert settings.MAX_CONTEXT_CHARS > 0
    assert settings.RETRIEVAL_TOP_K > 0
    assert settings.EMBEDDING_BATCH_SIZE > 0


def test_model_default_is_deepseek():
    assert settings.MODEL_NAME == "deepseek-chat"
