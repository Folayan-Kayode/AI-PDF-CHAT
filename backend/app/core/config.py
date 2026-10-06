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


_TRUE_VALUES = {"1", "true", "yes", "on"}

_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()

    if raw in _TRUE_VALUES:
        return True

    if raw in _FALSE_VALUES:
        return False

    raise RuntimeError(f"{name} must be a boolean, got {raw!r}.")


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

    # ---------------- Retrieval ------------------------------------------
    RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 5)

    # How many candidates to pull before reranking. Retrieval is cheap and
    # reranking is not, so it pays to cast a wide net first.
    RETRIEVAL_CANDIDATES = _env_int("RETRIEVAL_CANDIDATES", 20)

    # Chunks further away than this are treated as noise. The useful scale
    # depends on the embedding model and the document, so it is tunable.
    RETRIEVAL_MAX_DISTANCE = _env_float("RETRIEVAL_MAX_DISTANCE", 0.75)

    MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 12000)

    # ---------------- Retrieval quality ----------------------------------
    # Rewriting the question into search terms recovers questions whose
    # phrasing does not match the document's wording.
    QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", True)

    QUERY_REWRITE_MAX_CHARS = _env_int("QUERY_REWRITE_MAX_CHARS", 200)

    # Reranking reorders candidates with the model. It costs one extra call
    # per question, so it is skipped when retrieval is already confident.
    RERANK_ENABLED = _env_bool("RERANK_ENABLED", True)

    RERANK_SKIP_DISTANCE = _env_float("RERANK_SKIP_DISTANCE", 0.35)

    RERANK_SNIPPET_CHARS = _env_int("RERANK_SNIPPET_CHARS", 300)

    # ---------------- Document profile -----------------------------------
    # Questions about the document itself (title, author, publisher) use
    # words that appear nowhere in the question, so the document is asked to
    # describe itself once at ingest time and that profile is indexed.
    DOCUMENT_SUMMARY_ENABLED = _env_bool("DOCUMENT_SUMMARY_ENABLED", True)

    DOCUMENT_SUMMARY_SOURCE_PAGES = _env_int(
        "DOCUMENT_SUMMARY_SOURCE_PAGES",
        10,
    )

    DOCUMENT_SUMMARY_MAX_CHARS = _env_int("DOCUMENT_SUMMARY_MAX_CHARS", 600)

    # The profile is supplied to the model with every question, alongside the
    # retrieved passages, because vector search does not surface it for the
    # questions it exists to answer.
    DOCUMENT_PROFILE_IN_CONTEXT = _env_bool(
        "DOCUMENT_PROFILE_IN_CONTEXT",
        True,
    )

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
