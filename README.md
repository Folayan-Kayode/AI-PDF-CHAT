# AI PDF Chat

Single-document RAG chat over an uploaded PDF, built with FastAPI, Streamlit, DeepSeek (generation), Gemini (embeddings), ChromaDB, and LangChain.

## What it does today

**Ingestion**
- Upload one PDF via `POST /upload/` (single-document mode: a new upload replaces the index).
- Validates the `.pdf` extension, the real `%PDF-` header, the file size, the filename length, and normalises the filename identically on every OS.
- Writes the upload to a staging file and only moves it into place once it is proven to be a PDF, so a bad upload cannot overwrite a good copy.
- Detects duplicates by SHA-256 content hash and skips re-embedding.
- Rejects encrypted, empty, scanned/image-only, corrupt, and oversized documents with a specific `4xx` and a readable message (no OCR yet).
- Serialises ingestion: a second concurrent upload is rejected with `409` instead of interleaving and producing a mixed index.
- Embeds in paced batches and retries a transient failure with backoff, honouring the provider's `Retry-After`, so one quota blip does not discard the whole document.
- Chunks the document, then writes the vectors into a staging collection and swaps it in. A failed write leaves the previously indexed document queryable, and the error says so.
- Records the embedding model and schema version in each chunk's metadata.
- Reads the opening pages once and indexes a **document profile** (title, author, publisher, edition, short summary) as its own chunk, then supplies that profile with every question. Questions about the document itself use words the question never contains, so search alone cannot surface it — measured at distance 0.79 for *"what is the title of this book?"*, versus a 0.75 threshold. The profile costs one model call per document, not per question, and a failure simply skips it.

**Answering**
- Chat via `POST /chat/` (`question` non-empty, max 2000 chars).
- Retrieves the top-k chunks with Gemini embeddings and answers with DeepSeek (`deepseek-chat`) using a context-only prompt.
- Searches with the question **and** an LLM-rewritten variant, merging both result sets by best distance, so a weak rewrite cannot lose a chunk the original would have found.
- Reranks the candidate pool by reading each passage alongside the question, which separates chunks that embedding distance alone confuses. Reranking only ever reorders, never discards.
- Drops matches weaker than `RETRIEVAL_MAX_DISTANCE`, so noise is not sent as context.
- Detects an index built with a different embedding model and treats it as empty rather than returning meaningless neighbours.
- Treats document text as untrusted: it is delimited, closing delimiters are escaped, and the model is told to ignore instructions found inside it.
- Truncates context to a character budget and reports only the sources that were actually sent to the model.
- Returns `I couldn't find that information...` with empty `sources` when unanswerable; abstention detection matches the whole answer, so an answer that merely quotes the sentence keeps its sources.

**Operations**
- Maps failures to real status codes as JSON: `409` concurrent ingest, `413` too large, `429` rate limit with `Retry-After`, `502`/`503`/`504` upstream failures.
- Logs every request with an `X-Request-ID` and its latency, and logs the underlying cause of a `5xx` so it can be diagnosed.
- `GET /health` is a cheap liveness check; `GET /ready` reports readiness, `degraded` when the index is empty, and `503` only when the store is unreachable. `GET /ready?deep=true` probes the embedding provider, cached for `HEALTH_DEEP_CACHE_SECONDS`.
- Streamlit UI: file uploader, chat history, conditional Sources expander, clear messages when the backend is unreachable or rate limits, and a retry button for a failed upload.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Project, version, generation model. |
| `GET` | `/health` | Liveness. Cheap, does not touch the vector store. |
| `GET` | `/ready` | Readiness, index size, embedding-model match. `?deep=true` also probes embeddings. |
| `POST` | `/upload/` | Store and index one PDF. |
| `POST` | `/chat/` | Answer a question from the indexed document. |

## Configuration

Copy `.env.example` to `.env`. Both API keys are required and the app fails at startup if either is missing. Every setting below is mirrored in `.env.example`, and a test fails if the two drift apart.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Required. Embeddings. |
| `DEEPSEEK_API_KEY` | — | Required. Generation. |
| `MODEL` | `deepseek-chat` | DeepSeek generation model. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint. |
| `GENERATION_TIMEOUT_SECONDS` | `60` | Timeout for the generation call. |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Google embedding model. |
| `EMBEDDING_SCHEMA_VERSION` | `1` | Bump when the embedding vector space changes. |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks per embedding request. |
| `EMBEDDING_BATCH_DELAY_SECONDS` | `0.25` | Pause between embedding batches. |
| `EMBEDDING_MAX_RETRIES` | `4` | Attempts per embedding batch. |
| `EMBEDDING_RETRY_BASE_SECONDS` | `1.0` | Backoff base between retries. |
| `EMBEDDING_REQUESTS_PER_MINUTE` | `90` | Budget the pacing delay is derived from. |
| `UPLOAD_DIRECTORY` | `uploads` | Where uploads are stored. |
| `CHROMA_DIRECTORY` | `chroma_db` | Vector store location. |
| `CHROMA_WRITE_BATCH_SIZE` | `200` | Chunks per index write. |
| `CHROMA_WRITE_MAX_RETRIES` | `3` | Attempts per index write batch. |
| `CHROMA_WRITE_RETRY_BASE_SECONDS` | `0.5` | Backoff base between retries. |
| `MAX_UPLOAD_SIZE_MB` | `25` | Upload size limit. |
| `MAX_PAGES_PER_DOCUMENT` | `300` | Rejects longer PDFs during loading. |
| `MAX_CHUNKS_PER_DOCUMENT` | `1500` | Rejects documents that chunk too large. |
| `MAX_FILENAME_LENGTH` | `200` | Rejects absurd filenames with `400`. |
| `INGESTION_LOCK_TIMEOUT_SECONDS` | `0` | `0` rejects a concurrent upload immediately. |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per question. |
| `RETRIEVAL_CANDIDATES` | `20` | Candidates pulled before reranking. |
| `RETRIEVAL_MAX_DISTANCE` | `0.75` | Matches weaker than this are treated as noise. Corpus-dependent. |
| `MAX_CONTEXT_CHARS` | `12000` | Prompt context budget. |
| `QUERY_REWRITE_ENABLED` | `true` | Search with an LLM-rewritten query as well. One call per question. |
| `QUERY_REWRITE_MAX_CHARS` | `200` | Length cap on the rewritten query. |
| `RERANK_ENABLED` | `true` | Rerank candidates by reading them with the question. |
| `RERANK_SKIP_DISTANCE` | `0.35` | Skip the rerank call when retrieval is already confident. |
| `RERANK_SNIPPET_CHARS` | `300` | Passage length shown to the reranker. |
| `DOCUMENT_SUMMARY_ENABLED` | `true` | Index a profile chunk built from the opening pages. One call per document. |
| `DOCUMENT_SUMMARY_SOURCE_PAGES` | `10` | Opening pages read to build the profile. |
| `DOCUMENT_SUMMARY_MAX_CHARS` | `600` | Profile length cap. |
| `DOCUMENT_PROFILE_IN_CONTEXT` | `true` | Supply the profile with every question, since search does not surface it. |
| `LOG_LEVEL` | `INFO` | Root log level. |
| `HEALTH_DEEP_CACHE_SECONDS` | `30` | Cache for the `?deep=true` provider probe. |

The frontend reads its own environment: `BACKEND_URL` (default `http://127.0.0.1:8000`), `REQUEST_TIMEOUT_SECONDS` (default `300`), and `UPLOAD_TIMEOUT_SECONDS` (default `1800`, because a large document needs many paced batches).

Paths are resolved relative to the working directory, so start each service from its own folder as shown below.

## Run

Requires Python 3.11+ (CI runs 3.12).

1. Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` (embeddings) and `DEEPSEEK_API_KEY` (generation).
2. Backend: `pip install -r requirements.txt` then `uvicorn app.main:app --reload` from `backend/`.
3. Frontend: `pip install -r requirements.txt` then `streamlit run app.py` from `frontend/`.
4. Open http://127.0.0.1:8501

## Tests and lint

From `backend/`:

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest          # includes a coverage floor of 75% for app/
```

Tests mock the LLM and the embedding provider, so they run without API keys or network access.

## Optional extras

`pypdf` parses the encoding of CFF/Type1 fonts more accurately when `fontTools`
is installed. It is not required — text extraction works without it — but it
silences `fontTools is required to fully parse the encoding...` warnings and can
improve extraction on PDFs that use those fonts:

```bash
pip install fonttools
```


## Known limitations

- **Run a single worker.** Ingestion is serialised with a process-wide lock and ChromaDB is used in embedded (file) mode. Multiple uvicorn workers would each hold their own client over one directory and could interleave resets. Running multiple workers requires moving Chroma to server mode.
- **Changing `EMBEDDING_MODEL` invalidates the index.** Existing vectors are not re-embedded; the mismatch is detected and reported as `degraded` by `/ready` instead of being queried, so re-upload the document.
- **No OCR.** Scanned or image-only PDFs are rejected, not transcribed.
- **Retrieval thresholds are corpus-dependent.** `RETRIEVAL_MAX_DISTANCE` and `RERANK_SKIP_DISTANCE` are tuned against the sample documents here; a different embedding model or document mix needs its own values.
- **A question whose answer words are unknowable from the question alone** (for example "what is the title of this book?") is answered from the document profile chunk, not from a content chunk. If the opening pages do not state the field, the profile says `unknown` and the question abstains.
- **Single document.** A new upload replaces the previous index; there is no document list or per-document selection yet.
- **No Docker yet.** There are no Dockerfiles or compose file; containerisation is planned for a later phase.
- No streaming responses, conversation memory, authentication, or rate limiting for callers.
