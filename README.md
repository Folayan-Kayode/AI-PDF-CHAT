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
- Chunks the whole document as one text and records each chunk's **page span**, rather than splitting per page, so sentences and tables are not cut at page boundaries.
- Preserves line breaks and indentation when normalising extracted text, so tables, forms and lists keep the structure they were read from.
- Records the embedding model, schema version and `document_id` in each chunk's metadata, and one row per document in a SQLite registry.
- Builds a **document profile** from the PDF's own metadata and bookmarks — free, exact and stable — and supplies it with every question. Questions about the document itself use words the question never contains, so search cannot surface it. A model reads the opening pages only when the document has no outline, and placeholder metadata ("untitled", "anonymous") is ignored.

**Answering**
- Chat via `POST /chat/` (`question` non-empty, max 2000 chars).
- Retrieves the top-k chunks with Gemini embeddings and answers with DeepSeek (`deepseek-chat`) using a context-only prompt.
- Searches with the question **and** an LLM-rewritten variant, merging both result sets by best distance, so a weak rewrite cannot lose a chunk the original would have found.
- Selects passages **relative to the best match for that query** rather than against an absolute distance, because a broad question sits far from every chunk in a way a specific one does not. An absolute value survives only as a noise floor, and if a filter would empty the context the best candidates are sent anyway and a warning is logged — silence is worse than slightly off-topic context.
- Detects an index built with a different embedding model, schema version or vector space and treats it as empty rather than returning meaningless neighbours.
- Scopes retrieval and the profile to a `document_id`, so one document's content is never used to answer about another.
- Treats document text as untrusted: it is delimited, closing delimiters are escaped, and the model is told to ignore instructions found inside it.
- Budgets the prompt in both characters and estimated tokens, so a CJK document cannot silently blow it.
- Answers in the language of the question, and detects abstention with a language-neutral sentinel rather than an English sentence, so it works on non-English documents and cannot be tripped by a document that quotes a refusal.
- Reports retrieval telemetry (`retrieved_chunks`, `considered_candidates`, `best_distance`, `profile_used`) with every answer, and the UI says so explicitly when an answer came from the document profile alone.

**Operations**
- Maps failures to real status codes as JSON: `409` concurrent ingest, `413` too large, `429` rate limit with `Retry-After`, `502`/`503`/`504` upstream failures.
- Logs every request with an `X-Request-ID` and its latency, and logs the underlying cause of a `5xx` so it can be diagnosed.
- `GET /health` is a cheap liveness check; `GET /ready` reports readiness **per document**, `degraded` when the index is empty or does not match the configured model / schema version / vector space, and `503` only when the store is unreachable. `GET /ready?deep=true` probes the embedding provider, cached for `HEALTH_DEEP_CACHE_SECONDS`.
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
| `REGISTRY_PATH` | `registry.sqlite3` | One row per ingested document. |
| `CHROMA_SPACE` | `cosine` | Vector space; distances are only comparable within one space. |
| `CHROMA_WRITE_BATCH_SIZE` | `200` | Chunks per index write. |
| `CHROMA_WRITE_MAX_RETRIES` | `3` | Attempts per index write batch. |
| `CHROMA_WRITE_RETRY_BASE_SECONDS` | `0.5` | Backoff base between retries. |
| `MAX_UPLOAD_SIZE_MB` | `25` | Upload size limit — the real resource constraint. |
| `MAX_PAGES_PER_DOCUMENT` | `2000` | Coarse guard, deliberately generous. |
| `MAX_CHUNKS_PER_DOCUMENT` | `20000` | Coarse guard; bounds the embedding call count. |
| `MAX_FILENAME_LENGTH` | `200` | Rejects absurd filenames with `400`. |
| `INGESTION_LOCK_TIMEOUT_SECONDS` | `0` | `0` rejects a concurrent upload immediately. |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per question. |
| `RETRIEVAL_CANDIDATES` | `20` | Candidates pulled before reranking. |
| `RETRIEVAL_RELATIVE_MARGIN` | `1.15` | Keep candidates within this multiple of the best match. |
| `RETRIEVAL_ABSOLUTE_SLACK` | `0.10` | Also keep candidates within this distance of the best match. |
| `RETRIEVAL_MAX_DISTANCE` | `1.50` | Noise floor only, never the primary selector. |
| `MAX_CONTEXT_CHARS` | `12000` | Prompt context budget, in characters. |
| `MAX_CONTEXT_TOKENS` | `3000` | Prompt context budget, in estimated tokens. |
| `QUERY_REWRITE_ENABLED` | `true` | Search with an LLM-rewritten query as well. One call per question. |
| `QUERY_REWRITE_MAX_CHARS` | `200` | Length cap on the rewritten query. |
| `RERANK_ENABLED` | `false` | Rerank candidates by reading them with the question. Measured as not earning its call; see `docs/results.md`. |
| `RERANK_SKIP_RATIO` | `0.60` | Skip reranking when the best match stands out from the median. |
| `RERANK_SNIPPET_CHARS` | `300` | Passage length shown to the reranker. |
| `DOCUMENT_SUMMARY_ENABLED` | `true` | Build a profile chunk. Free from the PDF's own metadata and bookmarks; a model call only when there is no outline. |
| `DOCUMENT_SUMMARY_ALWAYS` | `false` | Also pay for a model summary when the document has an outline. |
| `DOCUMENT_SUMMARY_SOURCE_PAGES` | `10` | Minimum opening pages read when there is no outline; scales with document size. |
| `DOCUMENT_SUMMARY_MAX_CHARS` | `1200` | Profile length cap. |
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

## Evaluation

The retrieval and prompting choices here are measured rather than assumed. The
question set is committed at `backend/eval/questions.jsonl` — 33 questions that
deliberately include content, paraphrased wording, ambiguous multi-page answers,
broad/aggregate questions, document-metadata questions, and questions the
document cannot answer at all.

```bash
cd backend
python -m eval.run_eval --limit 3     # smoke test: 3 questions, one config
python -m eval.run_eval --all         # the ablation: knob values
python -m eval.run_shapes             # across document shapes
```

It reports retrieval hit@5 and MRR, answer accuracy, abstention precision and
recall, citation rate and citation validity, p50 latency, and estimated cost
per question. It runs against the real providers, so it costs money and is
deliberately kept out of the offline test suite.

`eval/run_shapes.py` is the one that matters for generalisation: it ingests a
two-page form, a table-heavy sheet, a German document, a 157-page technical
standard and a 658-page textbook using the shipped settings, and fails loudly
if any question is answered with an empty context.

Results — the ablation, the tuning decisions taken from it, and the
per-document shape results — are in [docs/results.md](docs/results.md).


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
- **Changing `EMBEDDING_MODEL` or `CHROMA_SPACE` invalidates the index.** Existing vectors are not re-embedded; the mismatch is detected and reported as `degraded` by `/ready` instead of being queried, so re-upload the document.
- **No OCR.** Scanned or image-only PDFs are rejected, not transcribed.
- **The relative margins are still tuned values.** Selection no longer depends on an absolute distance, but `RETRIEVAL_RELATIVE_MARGIN` and `RETRIEVAL_ABSOLUTE_SLACK` were chosen against the documents in `docs/shapes.json`; re-run `python -m eval.run_shapes` against your own before trusting them.
- **A document with no PDF metadata and no bookmarks depends on a model call** to build its profile. If that call fails, or the opening pages do not state a field, the profile is thinner and metadata questions may fall back to retrieval.
- **Nothing enforces multi-document isolation in the UI yet.** Retrieval and the profile are scoped by `document_id` and a registry records every document, but ingestion still replaces the index, so only one document is live at a time.
- **No Docker yet.** There are no Dockerfiles or compose file; containerisation is planned for a later phase.
- No streaming responses, conversation memory, authentication, or rate limiting for callers.
