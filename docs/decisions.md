# Decisions and reversals

A short record of the retrieval choices that were not obvious, including the
ones that were later reversed. Reversals are kept rather than deleted: a wrong
turn that was caught by measurement is more informative than a decision that
merely looks correct in hindsight.

Each entry is: the decision, why, the evidence, and whether anything was
reversed.

---

## Absolute distance threshold → relative selection (reversed)

**Was.** `RETRIEVAL_MAX_DISTANCE=0.60`: keep chunks within a fixed cosine/L2
distance of the query, tuned on one 658-page textbook.

**Why it looked right.** In the ablation it matched the accuracy of looser
thresholds at lower cost, and it was the cheapest accurate configuration
measured. The cost saving was real: a tighter threshold shrinks the prompt that
generation is billed for.

**What went wrong.** The saving came partly from *deleting the context*. On a
157-page technical standard the broad questions ("what is this document about?",
"list the sections") had best distances of 0.68–0.90, above 0.60, so the filter
removed every candidate. Retrieval returned nothing, and the answer was
generated from the document profile alone, cited entirely as `[document]`. Every
metric the ablation reported still looked healthy, because hit@5, accuracy and
cost cannot see an empty context. The failure was found by hand on a second
document, not by the test set, which was anchored to specific pages.

**Now.** Selection is relative to the best match for *that query*
(`max(best × 1.15, best + 0.10)`, capped by a 1.50 noise floor). The absolute
value survives only as a noise floor. An absolute cut is a statement about all
documents; a relative margin is a statement about one query's candidates.

**And if it still empties.** The never-empty invariant: if filtering would
remove everything, the best candidates are sent anyway and a `WARNING` is
logged, because a slightly off-topic passage is better than a confident answer
from the profile with no indication that retrieval failed. The pipeline logs
`cut_distance` alongside the observed distances so the threshold in force is
visible.

**Evidence.** `docs/results-regression.json` and the "Generalising across
document shapes" section of `docs/results.md`. The old rule is reconstructible
for comparison as `LegacyRetriever` in `backend/eval/reproduce_regression.py`.

---

## Keep the document profile in context (kept)

**Decision.** Build a profile from the PDF's own metadata and bookmarks at
ingest, and supply it with every question (`DOCUMENT_PROFILE_IN_CONTEXT=true`).

**Why.** Metadata questions ("who published this book, and in what year?") use
words the document never contains, so vector search cannot surface the profile
chunk reliably. Supplying it directly makes those answers guaranteed rather than
a coin flip.

**Evidence.** Across three repetitions, the profile configuration answered all
four metadata questions every time while the baseline missed one consistently
and another intermittently. `DOCUMENT_SUMMARY_ALWAYS=false` keeps the cost to
one call per *document*, and only when there is no bookmark outline.

---

## Reranking off by default (kept)

**Decision.** `RERANK_ENABLED=false`.

**Why.** Across repetitions it improved ranking (MRR 0.66–0.68 against 0.65) and
answer accuracy not at all (mean 0.90, identical to baseline) while costing
2.5×. It is a robustness purchase that did not convert into better answers on
the measured corpus. The code is kept and enabled by one flag for corpora where
ranking converts into correctness.

---

## Explicit cosine space, schema version bumped (kept)

**Decision.** Set `hnsw:space=cosine` on the collection and bump
`EMBEDDING_SCHEMA_VERSION` from 1 to 2.

**Why.** The distance values in the ablation were L2 distances; "0.60" means
something different in each space. An explicit space makes distances portable
across providers and documents, and the schema bump makes an old index detectable:
`/ready` reports the mismatch as `degraded` instead of querying an index whose
distances mean something else.

---

## Whole-document chunking with page spans (kept)

**Was.** One split per page.

**Why changed.** Splitting per page cuts sentences and tables at page
boundaries. The whole document is now split as one text, with each chunk
recording `page_start` and `page_end`, so a table that straddles two pages is
retrievable and a citation can name both.

---

## Language-neutral abstention sentinel (kept)

**Was.** Detect abstention by matching an English refusal sentence.

**Why changed.** A German document was answered with a German refusal, which the
English matcher did not recognise, and a document quoting a refusal could trip
it. Abstention is now a single sentinel token (`__NOT_FOUND__`), which is
language-neutral and impossible for document text to fake by accident.

---

## Production ingestion limits raised to match the evaluation (kept)

**Was.** `MAX_PAGES_PER_DOCUMENT=300`, `MAX_CHUNKS_PER_DOCUMENT=1500`, with the
evaluation raising them so its own document would fit.

**Why changed.** The product rejected the document the evaluation used, so every
published number described a configuration the product would not accept. The
defaults are now 2000 pages and 20000 chunks, the harness overrides removed, and
the limits recorded in each results file.
