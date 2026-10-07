# Deployment

How to run this app somewhere other than your laptop, and what to check once it
is up.

## Topology

**Topology A — single host, Docker Compose (recommended).** Both services run
on one host. Only the Streamlit UI is published to the host; the API is
reachable only on the internal compose network. That makes the API key defence
in depth rather than the only control, and it keeps all state on one volume.

```
        internet
           |
     [ :8501 ]  ui (Streamlit)
           |  http://api:8000  (internal, X-API-Key required)
       [ api ]  FastAPI + embedded ChromaDB
           |
   /data/chroma_db  /data/uploads  /data/registry   (named volumes)
```

`compose.yaml` implements exactly this:

```bash
cp .env.example .env        # then set GOOGLE_API_KEY, DEEPSEEK_API_KEY, API_KEY
docker compose up --build
# UI: http://127.0.0.1:8501
```

Topology B (managed UI + managed API) is also viable, but then the API is
public and the key and rate limits are mandatory rather than defence in depth.
Topology C (one managed platform) only works if the platform supports a
long-running container **and** a persistent volume; most free tiers do not.

**Never expose the API publicly without a key.** It spends provider money by
design: `/upload` runs hundreds of paid embedding calls and `/chat` spends a
generation call.

## Environment

The three secrets, all required (the app refuses to start without them):

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini embeddings (ingest and query). |
| `DEEPSEEK_API_KEY` | DeepSeek generation. |
| `API_KEY` | Shared secret for `/upload` and `/chat`. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. The UI sends it as `BACKEND_API_KEY`. |

Everything else has a safe default; `.env.example` documents every variable.
The ones that matter for a deployment:

| Variable | Default | Why it matters |
| --- | --- | --- |
| `WEB_CONCURRENCY` | `1` | Must stay 1. The app refuses to start above one; a managed platform that sets it for you would otherwise corrupt the index. |
| `CHROMA_DIRECTORY` / `UPLOAD_DIRECTORY` / `REGISTRY_PATH` | local paths | Point all three into the mounted volume (compose uses `/data/...`). |
| `MAX_UPLOAD_SIZE_MB` | `25` | The application's own limit; align the reverse proxy with it (below). |
| `UPLOAD_TIMEOUT_SECONDS` (frontend) | `1800` | How long the UI waits for an ingest; align the proxy read timeout with it. |
| `API_VERSION` | `0.9.0` | Reported by `/health`; bump per release. |
| `UPLOAD_RATE_LIMIT_PER_HOUR` / `CHAT_RATE_LIMIT_PER_HOUR` | `10` / `120` | Per-key token buckets. `0` disables. |
| `INGEST_DAILY_EMBEDDING_BATCH_BUDGET` | `2000` | Daily embedding-batch ceiling, persisted in the registry. |

## Start command

The container runs (from `backend/Dockerfile`):

```bash
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
```

`${PORT:-8000}` honours the port a managed platform injects; `--workers 1` is
mandatory, not a performance choice. On a platform with a start-command field,
use the same line.

## State and volumes

All state is three paths:

| Path | Contents | If lost |
| --- | --- | --- |
| `CHROMA_DIRECTORY` | the vector index | questions return nothing; `/ready` says `degraded` |
| `UPLOAD_DIRECTORY` | the original PDFs | re-upload required |
| `REGISTRY_PATH` | one row per document (and daily usage) | `/ready` cannot name documents |

Mount a **directory** for the registry (compose uses `/data/registry`), not a
single-file bind mount, which is fragile across Docker versions. Back up by
snapshotting the volume: `chroma_db/` plus `registry.sqlite3` is the whole
state. There is no other database.

`/ready` reports `registered_documents` (from the registry) next to the index
count, so "the volume did not persist" is distinguishable from "nothing was
uploaded": rows in the registry with an empty index means state was lost.

## Reverse proxy alignment

If a proxy sits in front, its limits must be **looser** than the application's,
or a large upload fails before FastAPI ever sees it — with the same silent
"it does not upload and does not tell me" symptom the project fixed once:

```nginx
client_max_body_size 30m;     # > MAX_UPLOAD_SIZE_MB (25)
proxy_read_timeout  1800s;    # >= UPLOAD_TIMEOUT_SECONDS (1800)
proxy_send_timeout  1800s;
```

If the proxy is managed (Render, Fly, Cloudflare), look up its body-size and
timeout limits and **lower** `MAX_UPLOAD_SIZE_MB` or `UPLOAD_TIMEOUT_SECONDS`
to match. Always document the smaller of the two.

## Health checks

- **Liveness → `/health`.** Cheap, never touches the vector store. Point the
  platform's restart probe here.
- **Readiness → `/ready`.** Returns **HTTP 200 while `degraded`** on purpose:
  an empty index is not a fault, it is a fresh deployment. Using `/ready` as a
  liveness probe causes restart loops on an empty index. `/ready` is a human or
  monitoring check (per-document counts, model/schema/space match).

A fresh deployment starts `degraded` until a document is uploaded, because
`EMBEDDING_SCHEMA_VERSION=2` and cosine invalidated every older index by design.

## Logs

Logs go to stdout with a request id, latency and, for every chat request,
retrieval telemetry (`retrieved_chunks`, `best_distance`,
`considered_candidates`, `profile_used`, `empty_context`). Use the platform's
log drain, or redirect the container's stdout to a file on the volume. The one
signal to watch is `empty_context=True`: it means an answer rested on the
document profile alone, which is the retrieval regression this project fixed.

## Post-deploy runbook

Run in order after the first deploy and record the outcome.

1. `GET /health` → 200, model names correct, `workers: 1`.
2. `GET /ready` → `degraded`, `indexed_chunks: 0`. Correct on a fresh volume.
3. Upload a small document (~5–20 pages) through the UI. Expect 200 with page
   and chunk counts.
4. `GET /ready` → `ready`, correct chunk count, `embedding_model_match` /
   `schema_version_match` / `vector_space_match` all true.
5. Ask a specific question → the answer cites `[p.N]`.
6. Ask a broad question ("what is this about?") → the answer does **not** cite
   only `[document]`, and the logs show `retrieved_chunks > 0`.
7. Ask an unanswerable question → abstains with no sources.
8. Restart the service, then repeat step 4 and step 5. Persistence proven.
9. `curl` `/upload/` with no key → 401. `curl` with the key works. A burst of
   chats → 429 with `Retry-After`.
10. Upload a near-limit document → succeeds, or fails with a readable message,
    never a silent hang.
11. Check the logs for the request id and latency on each call, and for any
    `empty_context=True` line.

## Rollback and incidents

- Every retrieval default is an environment variable, so a bad value is reverted
  without redeploying code. If the empty-context rate rises, restore
  `RETRIEVAL_RELATIVE_MARGIN` and `RETRIEVAL_ABSOLUTE_SLACK`, re-run
  `python -m eval.run_shapes`, then redeploy.
- Snapshot the volume before changing `EMBEDDING_MODEL`, `CHROMA_SPACE` or
  `EMBEDDING_SCHEMA_VERSION`; those invalidate the index by design.
- Rotating a provider key is zero-downtime: add the new key, deploy, revoke the
  old.
- Suspected corrupt index: stop the API, move the volume aside, start fresh,
  re-upload. `/ready` says `degraded` until then — the intended signal rather
  than a silent wrong answer.

## Not done here

- **TLS.** Terminate it at the host proxy or a Caddy/Traefik sidecar; the
  compose network is plain HTTP.
- **Horizontal scaling.** Requires moving Chroma to server mode, which removes
  the single-worker constraint.
