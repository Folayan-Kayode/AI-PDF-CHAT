# Release checklist

Run these before tagging a release or changing a retrieval default. They exist
because the most expensive bug in this project's history passed every test that
was in place at the time: an absolute distance threshold removed every passage
for broad questions, answers were generated from the document profile alone, and
hit@5, accuracy and cost all looked healthy because none of them can see an empty
context.

## 1. The offline suites, including the invariants

From `backend/`:

```bash
ruff check .
ruff format --check .
pytest
```

From `frontend/`:

```bash
ruff check app.py backend_client.py tests
pytest
```

The backend suite includes `tests/test_retrieval_invariants.py` and
`tests/test_document_isolation.py`. Deleting one of those tests to make a change
pass is a release blocker; they encode the failure above.

## 2. No default is changed without a measurement

Any change to a value in `backend/app/core/config.py` must come with a row in
`docs/results.md` (or a linked run) that measured it, and the commitment that the
new default is safe for a document the harness has not seen.

Do not publish evaluation numbers from a configuration the product rejects.
Ingestion limits in the harness are the production limits; raising them to make a
run finish means the run is measuring something the app will not accept.

## 3. Retrieval is exercised against the documents, not just the unit tests

```bash
python -m eval.diagnose --questions            # committed question set
python -m eval.run_shapes                      # five document shapes
```

`eval/diagnose.py` exits non-zero if any question produces zero passages, so it
can gate a release. `eval/run_shapes.py` fails loudly if any question is answered
with an empty context, and also exits non-zero if a document is missing, so a
quietly smaller run cannot pass as a full one (pass `--allow-missing` to accept a
partial run deliberately).

## 4. The index is compatible with the configuration

`GET /ready` must report `ok` per document, not `degraded`. It checks the
embedding model, `EMBEDDING_SCHEMA_VERSION` and the vector space (`CHROMA_SPACE`).
A mismatch is reported rather than queried, so the answer is "re-upload the
document", not "return meaningless neighbours".

## 5. The deployment surface is closed

- `API_KEY` is set and `/upload` and `/chat` return `401` without it; `/health`,
  `/ready` and `/` stay open.
- The rate limits and the ingest budget return `429` with `Retry-After` when
  exceeded.
- `WEB_CONCURRENCY` is 1 (the app refuses to start otherwise).
- All three state paths are on a persistent volume; a container restart and an
  image rebuild both keep the document (see `docs/deployment.md`).
- `docker compose build && docker compose up` brings up UI and API with only a
  populated `.env` as a manual step. Verify `.env` is **not** in the image:
  `docker run --rm -it <image> ls -a /app`.

## 6. The evidence is versioned and reproducible

Raw outputs (`docs/results.json`, `docs/results-repeats.json`,
`docs/shapes.json`, `docs/results-regression.json`) carry a `schema_version`, a
document identity and a code revision, so a future reader can tell which corpus
and which commit a file describes. Files measured under superseded settings are
marked historical rather than deleted.

The shapes and the regression reproduction ingest only committed, permissively
licensed documents (`backend/eval/documents/README.md`), so a fresh clone can
re-run them. A copyrighted document may only be used through the gitignored
`--local` / `--document` overlay, never committed.

## 7. Honesty sections are intact

`docs/results.md` keeps its "Limitations" section and `docs/decisions.md` keeps
its reversal record. When a conclusion is withdrawn, the entry is corrected or
marked as withdrawn; it is not silently deleted.
