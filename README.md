# AI PDF Chat

Single-document RAG chat over an uploaded PDF, built with FastAPI, Streamlit, DeepSeek (generation), Gemini (embeddings), ChromaDB, and LangChain.

## What it does today
- Upload one PDF via `POST /upload/` (single-document mode: new upload replaces index).
- Validates upload: `.pdf` extension, `%PDF-` header, empty-file check, 25MB limit (`MAX_UPLOAD_SIZE_MB`), path-traversal-safe filename.
- Detects duplicates by SHA-256 content hash.
- Rejects encrypted, empty, scanned/image-only, and corrupt PDFs with `400` (no OCR yet).
- Splits with `RecursiveCharacterTextSplitter` (1000/200), skips empty chunks.
- Stores `sha256_page_chunk` IDs + `{page, chunk, document_id}` metadata in persistent Chroma (`chroma_db/`).
- Chat via `POST /chat/` (`question` non-empty, max 2000 chars).
- Retrieves top-5 chunks with Gemini embeddings, answers with DeepSeek (`deepseek-chat`) using a context-only prompt.
- Returns `I couldn't find that information...` with empty `sources` when unanswerable.
- Streamlit UI: file uploader, chat input/history, conditional Sources expander (`Page X • Chunk Y`).
- `GET /health`, `GET /` (project/version/model). Fails at startup if `GOOGLE_API_KEY` or `DEEPSEEK_API_KEY` is missing.

## What it does not do yet
- No streaming, no conversation memory, no multi-document select, no auth/rate-limit, no Docker/tests/eval.

## Run
1. Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` (embeddings) and `DEEPSEEK_API_KEY` (generation).
2. Backend: `uvicorn app.main:app --reload` from `backend/`.
3. Frontend: `streamlit run app.py` from `frontend/` (points at `http://127.0.0.1:8000`).
