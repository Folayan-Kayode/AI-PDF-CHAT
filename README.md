# AI PDF Chat

Single-document RAG chat over an uploaded PDF, built with FastAPI, Streamlit, DeepSeek (generation), Gemini (embeddings), ChromaDB, and LangChain.

## What it does today
- Upload one PDF via `POST /upload/` (single-document mode: new upload replaces the index).
- Validates upload: `.pdf` extension, `%PDF-` header, empty-file check, size limit, path-traversal-safe filename.
- Detects duplicates by SHA-256 content hash and skips re-embedding.
- Rejects encrypted, empty, scanned/image-only, corrupt, and oversized documents with a specific `4xx` and a readable message (no OCR yet).
- Splits with `RecursiveCharacterTextSplitter` (1000/200), skipping empty chunks.
- Embeds in small batches with a pause between them, so a large document does not trip the provider's per-minute quota.
- Stores `sha256_page_chunk` IDs plus `{page, chunk, document_id}` metadata in a persistent Chroma index.
- Chat via `POST /chat/` (`question` non-empty, max 2000 chars).
- Retrieves the top-k chunks with Gemini embeddings, answers with DeepSeek (`deepseek-chat`) using a context-only prompt.
- Treats document text as untrusted input: it is delimited, and the model is told to ignore instructions found inside it. The context is truncated to a configurable character budget.
- Returns `I couldn't find that information...` with empty `sources` when the question is not answerable from the document.
- Maps upstream failures to real status codes (`503`/`504`) as JSON instead of a bare `500` text response.
- Streamlit UI: file uploader, chat history, conditional Sources expander (`Page X • Chunk Y`), and a clear message when the backend is unreachable.
- Structured logs with a per-request `X-Request-ID` and latency for every request.
- `GET /health` reports the index size and models, and returns `503` if the vector store is unreachable.

## Configuration
Copy `.env.example` to `.env`. Both API keys are required and the app fails at startup if either is missing.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Required. Used for embeddings. |
| `DEEPSEEK_API_KEY` | — | Required. Used for answer generation. |
| `MODEL` | `deepseek-chat` | DeepSeek generation model. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint. |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Google embedding model. |
| `MAX_UPLOAD_SIZE_MB` | `25` | Upload size limit. |
| `MAX_PAGES_PER_DOCUMENT` | `300` | Rejects longer PDFs during loading. |
| `MAX_CHUNKS_PER_DOCUMENT` | `1500` | Rejects documents that chunk too large. |
| `MAX_CONTEXT_CHARS` | `12000` | Prompt context budget. |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per question. |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks per embedding request. |
| `EMBEDDING_BATCH_DELAY_SECONDS` | `0.25` | Pause between embedding batches. |
| `GENERATION_TIMEOUT_SECONDS` | `60` | Timeout for the generation call. |
| `LOG_LEVEL` | `INFO` | Root log level. |

The frontend reads `BACKEND_URL` (default `http://127.0.0.1:8000`) and `REQUEST_TIMEOUT_SECONDS` from its own environment.

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
pytest
```

Tests mock the LLM, so they run without API keys or network access.

## What it does not do yet
- No streaming responses, conversation memory, or multi-document selection.
- No authentication, rate limiting, or per-user isolation.
- No Dockerfiles or working `docker-compose.yml`.
- No OCR for scanned PDFs, and no automated retrieval/answer evaluation yet.
