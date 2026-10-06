import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    """
    Read an environment variable, treating a blank value as unset.

    Without this, a line like `MODEL=` in .env would yield "" instead of
    the default, since os.getenv only falls back when the variable is absent.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))

    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))

    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}.") from exc


class Settings:
    PROJECT_NAME = "AI PDF Chat"

    API_VERSION = "1.0.0"

    # ---------------- Generation: DeepSeek (OpenAI-compatible API) ---------
    MODEL_NAME = _env("MODEL", "deepseek-chat")

    DEEPSEEK_BASE_URL = _env(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )

    GENERATION_TIMEOUT_SECONDS = _env_float(
        "GENERATION_TIMEOUT_SECONDS",
        60.0,
    )

    # ---------------- Embeddings: Google ---------------------------------
    EMBEDDING_MODEL = _env(
        "EMBEDDING_MODEL",
        "gemini-embedding-2",
    )

    # Identifies the vector space the index was built in. Bump it whenever
    # the embedding model or its configuration changes.
    EMBEDDING_SCHEMA_VERSION = _env_int("EMBEDDING_SCHEMA_VERSION", 1)

    # Chunks are embedded in small batches with a pause between them so a
    # large document does not trip the per-minute embedding quota.
    EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 64)

    EMBEDDING_BATCH_DELAY_SECONDS = _env_float(
        "EMBEDDING_BATCH_DELAY_SECONDS",
        0.25,
    )

    # A transient quota error mid-document should not discard the batches
    # that already succeeded, so each batch is retried with backoff.
    EMBEDDING_MAX_RETRIES = _env_int("EMBEDDING_MAX_RETRIES", 4)

    EMBEDDING_RETRY_BASE_SECONDS = _env_float(
        "EMBEDDING_RETRY_BASE_SECONDS",
        1.0,
    )

    EMBEDDING_REQUESTS_PER_MINUTE = _env_int(
        "EMBEDDING_REQUESTS_PER_MINUTE",
        90,
    )

    # ---------------- Storage --------------------------------------------
    UPLOAD_DIRECTORY = _env("UPLOAD_DIRECTORY", "uploads")

    CHROMA_DIRECTORY = _env("CHROMA_DIRECTORY", "chroma_db")

    # A single unbounded write can exceed what one SQLite transaction can
    # commit, so index writes are chunked and retried.
    CHROMA_WRITE_BATCH_SIZE = _env_int("CHROMA_WRITE_BATCH_SIZE", 200)

    CHROMA_WRITE_MAX_RETRIES = _env_int("CHROMA_WRITE_MAX_RETRIES", 3)

    CHROMA_WRITE_RETRY_BASE_SECONDS = _env_float(
        "CHROMA_WRITE_RETRY_BASE_SECONDS",
        0.5,
    )

    # ---------------- Ingestion limits -----------------------------------
    MAX_UPLOAD_SIZE_MB = _env_int("MAX_UPLOAD_SIZE_MB", 25)

    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    MAX_PAGES_PER_DOCUMENT = _env_int("MAX_PAGES_PER_DOCUMENT", 300)

    MAX_CHUNKS_PER_DOCUMENT = _env_int("MAX_CHUNKS_PER_DOCUMENT", 1500)

    MAX_FILENAME_LENGTH = _env_int("MAX_FILENAME_LENGTH", 200)

    # 0 means "reject a concurrent upload immediately" rather than queue it,
    # because ingestion can take minutes.
    INGESTION_LOCK_TIMEOUT_SECONDS = _env_float(
        "INGESTION_LOCK_TIMEOUT_SECONDS",
        0.0,
    )

    # ---------------- Retrieval / prompt budget --------------------------
    RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 5)

    MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 12000)

    # ---------------- Observability --------------------------------------
    LOG_LEVEL = _env("LOG_LEVEL", "INFO")

    HEALTH_DEEP_CACHE_SECONDS = _env_float(
        "HEALTH_DEEP_CACHE_SECONDS",
        30.0,
    )

    def __init__(self):
        self.GOOGLE_API_KEY = self._require("GOOGLE_API_KEY")

        self.DEEPSEEK_API_KEY = self._require("DEEPSEEK_API_KEY")

    @property
    def EMBEDDING_MIN_DELAY_SECONDS(self) -> float:
        """
        Minimum pause between embedding requests for the configured budget.

        The configured delay is a floor, but it cannot be trusted to stay
        under the provider's per-minute request limit on its own: a faster
        region would issue them quicker.
        """
        return 60.0 / max(1, self.EMBEDDING_REQUESTS_PER_MINUTE)

    @staticmethod
    def _require(name: str) -> str:
        value = os.getenv(name)

        if not value or not value.strip():
            raise RuntimeError(f"{name} is not set. Copy .env.example to .env and add your {name}.")

        return value.strip()


settings = Settings()
