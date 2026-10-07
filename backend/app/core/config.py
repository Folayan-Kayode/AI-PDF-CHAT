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

    # Pre-release: the deployment commit is tagged v1.0.0 once it is live.
    API_VERSION = _env("API_VERSION", "0.9.0")

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

    # The collection is created with an explicit vector space so distances
    # mean the same thing across embedding models and documents. Changing this
    # changes the space, so existing indexes must be re-embedded; the
    # embedding_model / schema_version fingerprint detects and reports that.
    CHROMA_SPACE = _env("CHROMA_SPACE", "cosine")

    # Identifies the vector space the index was built in. Bump it whenever
    # the embedding model *or* the space changes, so a stale index is
    # detected instead of queried.
    EMBEDDING_SCHEMA_VERSION = _env_int("EMBEDDING_SCHEMA_VERSION", 2)

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

    # ---------------- Security -------------------------------------------
    # /upload and /chat spend provider money, so they must not be reachable
    # without a credential. API_KEY is required in __init__ via _require(),
    # which fails closed at startup instead of defaulting to open. /health,
    # /ready and / stay open so a platform health check needs no secret.
    #
    # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"

    # Only used if a browser client calls the API directly. The shipped
    # Streamlit topology is server-to-server, so this defaults to empty and
    # no CORS middleware is added.
    CORS_ORIGINS = _env("CORS_ORIGINS", "")

    # In-process token buckets, keyed by API key. 0 disables a limit.
    # Uploads are bounded by volume (a concurrent one is already a 409, not a
    # rate-limit), chats by request count.
    UPLOAD_RATE_LIMIT_PER_HOUR = _env_int("UPLOAD_RATE_LIMIT_PER_HOUR", 10)

    CHAT_RATE_LIMIT_PER_HOUR = _env_int("CHAT_RATE_LIMIT_PER_HOUR", 120)

    # Ingestion is the expensive path. This is a daily ceiling on embedding
    # *batches* (not chunks), checked before embedding starts, so a single
    # caller cannot run up an unbounded bill. 0 disables the ceiling.
    INGEST_DAILY_EMBEDDING_BATCH_BUDGET = _env_int(
        "INGEST_DAILY_EMBEDDING_BATCH_BUDGET",
        2000,
    )

    # ---------------- Runtime topology -----------------------------------
    # Embedded Chroma plus a process-wide ingestion lock means exactly one
    # worker. Managed platforms set WEB_CONCURRENCY for you, so the app refuses
    # to start when it is >1 unless MULTI_WORKER_ACK says the operator has
    # moved Chroma to server mode (or otherwise accepted the risk).
    WEB_CONCURRENCY = _env_int("WEB_CONCURRENCY", 1)

    MULTI_WORKER_ACK = _env_bool("MULTI_WORKER_ACK", False)

    # ---------------- Storage --------------------------------------------
    UPLOAD_DIRECTORY = _env("UPLOAD_DIRECTORY", "uploads")

    CHROMA_DIRECTORY = _env("CHROMA_DIRECTORY", "chroma_db")

    # One row per ingested document: what the index was built from.
    REGISTRY_PATH = _env("REGISTRY_PATH", "registry.sqlite3")

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

    # Coarse guards, deliberately generous: the real constraints are the
    # upload size, the embedding quota and ingest wall-clock, not a document's
    # shape. These exist to stop a pathological file, not to reject an unusual
    # one, and they match what the evaluation harness ingests.
    MAX_PAGES_PER_DOCUMENT = _env_int("MAX_PAGES_PER_DOCUMENT", 2000)

    MAX_CHUNKS_PER_DOCUMENT = _env_int("MAX_CHUNKS_PER_DOCUMENT", 20000)

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

    # An absolute distance is not meaningful across embedding models and
    # documents: broad questions ("what is this document about?") sit far from
    # every chunk in a way that specific questions do not, so a fixed cut
    # deletes the broad ones entirely. Selection is therefore *relative* to
    # the best candidate for each query.
    #
    # Keep a candidate when its distance is within both the multiplicative
    # margin and the additive slack of the best match. Two terms because a
    # pure multiplier is too tight when the best distance is small.
    RETRIEVAL_RELATIVE_MARGIN = _env_float("RETRIEVAL_RELATIVE_MARGIN", 1.15)

    RETRIEVAL_ABSOLUTE_SLACK = _env_float("RETRIEVAL_ABSOLUTE_SLACK", 0.10)

    # A noise floor, not a selector: its only job is to reject obvious
    # nonsense when even the best match is unrelated. Distances are cosine,
    # so the useful range is [0, 2].
    RETRIEVAL_MAX_DISTANCE = _env_float("RETRIEVAL_MAX_DISTANCE", 1.50)

    MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 12000)

    # A character budget alone silently means something very different for CJK
    # text, where a character is roughly a token. Both budgets are applied.
    MAX_CONTEXT_TOKENS = _env_int("MAX_CONTEXT_TOKENS", 3000)

    # ---------------- Retrieval quality ----------------------------------
    # Rewriting the question into search terms recovers questions whose
    # phrasing does not match the document's wording.
    QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", True)

    QUERY_REWRITE_MAX_CHARS = _env_int("QUERY_REWRITE_MAX_CHARS", 200)

    # Reranking reorders candidates with the model, reading each passage
    # alongside the question. It only ever reorders, never discards, so it
    # cannot cause an abstention.
    #
    # Measured off by default: on the evaluation set reranking *lowered*
    # answer accuracy (0.83 -> 0.79) and abstention precision (0.67 -> 0.60)
    # while tripling cost per question, so it did not earn its extra call.
    # The code is kept and enabled with RERANK_ENABLED=true for corpora where
    # it does pay for itself; see docs/results.md.
    RERANK_ENABLED = _env_bool("RERANK_ENABLED", False)

    # Only consulted when reranking is enabled: skip the call when the best
    # candidate stands out from the rest of the set for this query. Expressed
    # as a ratio against the median candidate distance rather than an absolute
    # constant, for the same reason as the retrieval cut.
    RERANK_SKIP_RATIO = _env_float("RERANK_SKIP_RATIO", 0.60)

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

    DOCUMENT_SUMMARY_MAX_CHARS = _env_int("DOCUMENT_SUMMARY_MAX_CHARS", 1200)

    # The profile is built from the PDF's own metadata and bookmarks, which is
    # free and exact. Set this to also pay for a model summary when the
    # document already has an outline.
    DOCUMENT_SUMMARY_ALWAYS = _env_bool("DOCUMENT_SUMMARY_ALWAYS", False)

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

        # Required for the same reason as the provider keys: an unset value
        # must stop the app, not silently leave the paid endpoints open.
        self.API_KEY = self._require("API_KEY")

    @property
    def CORS_ORIGIN_LIST(self) -> list[str]:
        """Parsed allowlist; empty means no CORS middleware is added."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

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
