# Release checklist

Run these before tagging a release or changing a retrieval default. They exist
because the most expensive bug in this project's history passed every test that
was in place at the time: an absolute distance threshold removed every passage
for broad questions, answers were generated from the document profile alone,
and hit@5, accuracy and cost all looked healthy because none of them can see an
empty context.

## 1. The offline suite, including the invariants

From `backend/`:

```bash
ruff check .
ruff format --check .
pytest
```

The suite includes `tests/test_retrieval_invariants.py` and
`tests/test_document_isolation.py`. Deleting one of those tests to make a change
pass is a release blocker; they encode the failure above.

## 2. No default is changed without a measurement

Any change to a value in `backend/app/core/config.py` must come with a row in
`docs/results.md` (or a linked run) that measured it, and the commitment that
the new default is safe for a document the harness has not seen.

Do not publish evaluation numbers from a configuration the product rejects.
Ingestion limits in the harness are the production limits; raising them to make
a run finish means the run is measuring something the app will not accept.

## 3. Retrieval is exercised against the document, not just the unit tests

```bash
python -m eval.diagnose --questions            # committed question set
python -m eval.run_shapes                      # five document shapes
```

`eval/diagnose.py` exits non-zero if any question produces zero passages, so it
can gate a release. `eval/run_shapes.py` fails loudly if any question is
answered with an empty context.

## 4. The index is compatible with the configuration

`GET /ready` must report `ok` per document, not `degraded`. It checks the
embedding model, `EMBEDDING_SCHEMA_VERSION` and the vector space
(`CHROMA_SPACE`). A mismatch is reported rather than queried, so the answer is
"re-upload the document", not "return meaningless neighbours".

## 5. The evidence is versioned

Raw outputs (`docs/results.json`, `docs/results-repeats.json`,
`docs/shapes.json`, `docs/results-regression.json`) carry a `schema_version`, a
document identity and a code revision, so a future reader can tell which corpus
and which commit a file describes. Files measured under superseded settings are
marked historical rather than deleted.

## 6. Honesty sections are intact

`docs/results.md` keeps its "Limitations" section and `docs/decisions.md` keeps
its reversal record. When a conclusion is withdrawn, the entry is corrected or
marked as withdrawn; it is not silently deleted.
