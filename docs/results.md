# Evaluation results

> **Scope.** The knob ablation (below) was measured in the old L2 space with
> per-page chunking, at ingestion limits the product no longer uses, and is
> **historical**: it documents how each *knob* behaved, not the shipped
> defaults. The current defaults come from the generalisation work in
> "Generalising across document shapes" and "The regression, reproduced". Only
> rows marked with repetitions should be trusted for close comparisons. Read
> "Limitations" before quoting any number, and see
> [decisions.md](decisions.md) for the choices and their reversals.

These numbers decide the retrieval defaults in `backend/app/core/config.py`.
They exist because the project had accumulated five retrieval knobs that were
set by judgement rather than measurement.

**Read "Repetition and variance" before trusting any single row.** Running the
same configuration twice changed the baseline by 13 accuracy points, which is
the most useful thing this exercise produced.

## What was measured

| | |
| --- | --- |
| Document | *Principles of Information Security*, 4th ed. (658 pages) |
| Index | 2,835 chunks at `chunk_size=1000`, embedded with `gemini-embedding-2` |
| Question set | `backend/eval/questions.jsonl` — 33 questions, committed so every configuration is measured against an unchanged set |
| Answers | DeepSeek `deepseek-chat`, one call per question |
| Runs | one pass per configuration for the ablation; four headline configurations repeated three times |

The ablation's source document is a copyrighted textbook and is **not
committed**, so those rows cannot be re-run exactly and are historical evidence.
The reproducible evaluations below (document shapes, and the regression
reproduction) use committed, permissively licensed substitutes — see
[`backend/eval/documents/README.md`](../backend/eval/documents/README.md) — and
run on a fresh clone.

The question set deliberately contains six kinds of question:

- **12 content** — the answer is stated on a page, in the document's own words.
- **5 paraphrase** — the answer exists but the question does not use the document's vocabulary (these exercise query rewriting).
- **3 ambiguous** — several pages could answer, so ranking matters more than presence (these exercise reranking).
- **3 broad** — aggregate questions ("what is this document about?") with no single source page. These are the ones an absolute distance threshold deletes.
- **4 metadata** — about the document itself (title, author, publisher, edition). These have no gold page and are answered from the document profile, so they are excluded from the retrieval metrics.
- **6 unanswerable** — the document cannot answer them, which measures abstention rather than recall.

## How the metrics are defined

| Metric | Definition |
| --- | --- |
| hit@5 | a gold page appears among the five pages supplied to the model |
| MRR | mean reciprocal rank of the first gold page, so putting the right page first scores higher than merely including it |
| Answer accuracy | over the answerable questions: every expected keyword appears in the answer and the answer is not an abstention |
| Abstention precision / recall | abstention is treated as the positive class: a refusal on an answerable question is a false positive, the failure mode users notice most |
| Cited | share of answers that cite anything |
| Citations valid | share of citing answers where every `[p.N]` matches a page that was actually supplied, and `[document]` is used only when a profile was supplied |
| Empty context rate | among answered questions, the share where **zero** passages reached the prompt |
| Profile-only rate | among answered questions, the share whose answer cites nothing but `[document]` |
| p50 | median end-to-end latency per question |
| $/question | estimated from characters at published list prices; an estimate, not a bill |

The last two columns were added after a regression they would have caught. Both
were 0 by construction on the single page-anchored question set the ablation
used, so the failure had no column to appear in: retrieval returned nothing,
the profile alone answered, and every other metric looked fine. They are now the
primary signal that a configuration is answering from the profile rather than
the document, and `eval/run_shapes.py` fails loudly if empty-context rate is
nonzero.

Answer correctness is checked by keyword matching rather than by a model
judge, on purpose: an ablation comparison must not be skewed by the variance
of a model grading its own output.

## Ablation across knob values

> **Historical.** These rows were measured in the old L2 space with per-page
> chunking, an absolute 0.60 distance threshold, and the 30-question set that
> predates the three broad questions, before the generalisation work. They
> document how each *knob* behaved; the shipped defaults have since changed, and
> distances below are not comparable with the cosine distances in the next
> section.

One pass per configuration, same 30 questions, same index. `all` is every
feature on; `tuned (shipped)` was what the project shipped at the time.

| Configuration | hit@5 | MRR | Answer acc. | Abstain P | Abstain R | Cited | Citations valid | p50 s | $/question | calls/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0.80 | 0.65 | 0.96 | 0.86 | 1.00 | 1.00 | 0.96 | 1.8 | 0.00040 | 2.0 |
| `+threshold` | 0.80 | 0.65 | 0.92 | 0.75 | 1.00 | 1.00 | 0.95 | 1.8 | 0.00035 | 1.8 |
| `+rewrite` | 0.80 | 0.67 | 0.92 | 0.86 | 1.00 | 1.00 | 0.96 | 3.3 | 0.00041 | 4.0 |
| `+rerank` | 0.80 | 0.63 | 0.92 | 0.86 | 1.00 | 1.00 | 0.96 | 2.6 | 0.00101 | 2.6 |
| `+profile` | 0.80 | 0.65 | 0.96 | 0.86 | 1.00 | 1.00 | 1.00 | 2.0 | 0.00045 | 2.0 |
| `all` | 0.85 | 0.67 | 0.96 | 0.86 | 1.00 | 1.00 | 1.00 | 3.5 | 0.00085 | 4.3 |
| `tuned (shipped)` | 0.80 | 0.65 | 0.96 | 0.86 | 1.00 | 1.00 | 1.00 | 3.2 | 0.00036 | 4.0 |
| `top_k=3` | 0.70 | 0.59 | 0.83 | 0.75 | 1.00 | 1.00 | 1.00 | 3.6 | 0.00065 | 4.3 |
| `top_k=8` | 0.90 | 0.73 | 0.92 | 0.86 | 1.00 | 1.00 | 1.00 | 3.6 | 0.00098 | 4.3 |
| `max_distance=0.60` | 0.80 | 0.65 | 0.96 | 0.86 | 1.00 | 1.00 | 1.00 | 3.3 | 0.00044 | 4.0 |
| `max_distance=0.90` | 0.85 | 0.66 | 0.96 | 0.86 | 1.00 | 1.00 | 1.00 | 4.0 | 0.00119 | 4.6 |
| `skip_distance=0.20` | 0.85 | 0.68 | 0.92 | 0.86 | 1.00 | 1.00 | 1.00 | 4.1 | 0.00148 | 4.7 |
| `skip_distance=0.50` | 0.85 | 0.63 | 0.96 | 0.86 | 1.00 | 1.00 | 1.00 | 3.3 | 0.00073 | 4.2 |
| `chunk=500` | 0.85 | 0.64 | 0.92 | 0.75 | 1.00 | 1.00 | 1.00 | 3.1 | 0.00031 | 4.0 |
| `chunk=1500` | 0.85 | 0.70 | 0.88 | 0.75 | 1.00 | 1.00 | 1.00 | 3.7 | 0.00151 | 4.4 |

The `+threshold` row's 1.8 calls per question is **a defect indicator, not an
efficiency win**. A stricter threshold emptied the context for some questions,
the pipeline then abstained without calling the model, and the cost column is
lower because fewer questions were answered. It is the same failure that later
broke the technical standard.

These historical rows predate the empty-context and profile-only columns, so
they cannot show how often the context was emptied; the reconstructed before/after
measurement below does. Treat the accuracy column with care too: it is a single
pass, and the repetitions below show that one pass can be 13 points off.

## Repetition and variance

Four configurations were run three times each, same index, same questions.
Retrieval metrics were identical across repetitions for configurations with no
model in the retrieval path — baseline and `+profile` both scored hit@5 0.80
and MRR 0.65 three times out of three. Configurations that put a model in the
retrieval path vary slightly, because the rewritten query and the reranking are
themselves generated (tuned MRR 0.65–0.67, `+rerank` MRR 0.66–0.68).

**So: retrieval numbers from a single pass are trustworthy; accuracy
differences of one question are not.** One question is worth 4.2 points on 24
answerable questions, and answer generation is not deterministic.

| Configuration | Answer accuracy (3 runs) | Mean | Abstention precision | hit@5 | MRR | Citations valid | $/question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0.92, 0.88, 0.92 | 0.90 | 0.67–0.75 | 0.80 | 0.65 | 0.95–1.00 | 0.00039 |
| `+profile` | **0.96, 0.96, 0.96** | **0.96** | **0.86** | 0.80 | 0.65 | 1.00 | 0.00044 |
| `+rerank` | 0.88, 0.92, 0.92 | 0.90 | 0.75 | 0.80–0.85 | 0.66–0.68 | 0.95–1.00 | 0.00097 |
| `tuned (shipped)` | **0.96, 0.96, 0.96** | **0.96** | **0.86** | 0.80 | 0.65–0.67 | 1.00 | **0.00036** |

The baseline and profile ranges do not overlap (0.88–0.92 against 0.96–0.96),
and the mechanism is visible question by question: the baseline misses q23
("who published this book, and in what year?") in all three runs, while every
profile configuration answers all four metadata questions in all three runs.
That is the difference between a metadata answer being guaranteed and being a
coin flip on whether vector search happens to surface the profile chunk.

The single-pass table's baseline row (0.96) was therefore the outlier, and the
apparent 13-point swing in run 2 of the earlier comparison was that outlier
rather than evidence against the profile.

## What the data says

**Guaranteeing the metadata answers is worth one extra call per question at
ingest.** The profile removes a consistent miss and a flaky one, and it is the
only change that moved accuracy with non-overlapping ranges across
repetitions. Its cost is one model call per *document*, not per question:
`+profile` measured $0.00044 and p50 1.8s against the baseline's $0.00039 and
1.7s, so the latency is essentially unchanged.

**Reranking does not earn its call.** Across repetitions it improved retrieval
ranking consistently (MRR 0.66–0.68 against the baseline's 0.65, hit@5 up to
0.85) and improved answer accuracy *not at all* (mean 0.90, identical to the
baseline) while costing 2.5× per question. It is a robustness purchase, not an
accuracy one, so `RERANK_ENABLED` is now false and the setting is one flag
away for corpora where better ranking converts into better answers.

**Withdrawn: "a 0.60 distance threshold is not worse than 0.75 and costs
less."** This was concluded from the ablation above and it is wrong as a general
statement. The tighter threshold did shrink the prompt generation is billed for,
and `tuned` was the cheapest accurate configuration *on this one book*. On a
second document the same threshold deleted every passage for broad questions, and
the "cost saving" was partly the cost of not answering. It is withdrawn, not
softened: an absolute distance is a claim about every document, and one
document cannot support it. The replacement is below.

**`chunk_size=1000` and `top_k=5` are confirmed, not guessed.** 500 is the
cheapest to run and answers worse; 1500 is the most expensive and also answers
worse. `top_k=3` is clearly too few (0.83 in the ablation).

**Citation validity is near-perfect wherever a threshold is used.** Only
configurations that passed unfiltered context produced an invented citation,
which is exactly what the verifier exists to catch.

## Generalising across document shapes

The ablation above used **one** 658-page English textbook, and an absolute
distance threshold tuned on it (0.60) shipped as a global default. On a
different document it deleted every passage for broad questions: retrieval
returned nothing, and the answer was generated from the document profile alone,
cited entirely as `[document]`. A 33-question set anchored to specific pages
could not observe that, because nearly every question it contained had a nearby
page.

`eval/run_shapes.py` ingests five documents with the shipped settings and fails
loudly if any question is answered with an empty context:

| Document | Shape | Questions | Answered | Abstained correctly | Empty ctx rate | Profile-only rate | Fewest passages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sheet` | 2-page form, no bookmarks | 3 | 2/2 | 1/1 | 0.00 | 0.00 | 2 |
| `table` | table of sensor readings | 3 | 2/2 | 1/1 | 0.00 | 0.00 | 2 |
| `german` | 1-page German document | 2 | 1/1 | 1/1 | 0.00 | 0.00 | 2 |
| `standard` | 157-page technical standard | 4 | 3/3 | 1/1 | 0.00 | 0.67 | 5 |
| `book` | 658-page textbook | 4 | 3/3 | 1/1 | 0.00 | 0.33 | 5 |

Every row's empty-context rate is 0.00, which is the invariant this runner
exists to enforce. The profile-only rates are **not** zero and are not meant to
be: a metadata question is supposed to be answered from the profile and cites
`[document]` by design, and on the standard the aggregate question is answered
from the profile outline for the same reason. Empty-context rate is the clean
regression signal; profile-only rate is read beside it.

Sixteen of sixteen questions behaved correctly and **no question was ever
answered with an empty context**, including the questions that used to fail:

| Question | Document | Passages | Best distance |
| --- | --- | --- | --- |
| "What is this document about?" | standard | 5 | 0.349 |
| "What is the title of this document?" | standard | 5 | 0.346 |
| "List the sections of this document." | standard | 5 | 0.331 |
| "What is this book about?" | book | 5 | 0.331 |
| "List the chapters of this book." | book | 5 | 0.315 |
| "What was the temperature on 3 March 2024?" | table | 2 | 0.278 |
| "Wie oft erfolgt die Wartung der Heizungsanlage?" | german | 2 | 0.274 |

Four things this demonstrates that the knob ablation could not:

- **Broad questions reach the model.** "What is this document about?" and "list
  the sections" are aggregate questions with no single source page. They are now
  answered from the profile outline, which is built from the document's own
  bookmarks and supplied with every question.
- **Aggregate questions cannot be answered by a larger top-k.** No top-5 (or
  top-50) of a 157-page standard contains its own table of contents, which is
  why the outline is built at ingest rather than searched for.
- **A table cell is retrievable.** The sensor reading for a specific date
  survives extraction because line structure is preserved instead of being
  flattened into one line per page.
- **A non-English document works end to end.** The German question is answered
  from the German document, because abstention is detected by a language-neutral
  sentinel rather than by matching an English sentence.

Distances are now **cosine** (identical text is 0.0, orthogonal is 1.0), set
explicitly on the collection. The values above sit between 0.19 and 0.55. The
same questions had best **L2** distances of 0.66–0.95, which is why the old
absolute 0.60 cut deleted them; the next section reconstructs it.

## The regression, reproduced

The plan asked for a "before" snapshot before the fix landed. It was not taken,
so `eval/reproduce_regression.py` reconstructs the pre-fix state — **L2 space,
absolute 0.60, no fallback** — and runs three arms over the same two documents.
The reconstruction is an explicit subclass, not a reverted release, and it
separates the two changes that shipped together, the vector space and the
selection rule:

| Document | Arm | Space | Correct | Empty context rate | Profile-only rate |
| --- | --- | --- | --- | --- | --- |
| `standard` (157 pp) | **before:** absolute 0.60, no fallback | L2 | 3/6 | **0.67** | 0.67 |
| `standard` | *middle:* absolute 0.60, no fallback | cosine | 5/6 | 0.00 | 0.20 |
| `standard` | **after:** relative selection | cosine | 5/6 | **0.00** | 0.40 |
| `book` (658 pp) | **before:** absolute 0.60, no fallback | L2 | 3/4 | **1.00** | 1.00 |
| `book` | *middle:* absolute 0.60, no fallback | cosine | 4/4 | 0.00 | 0.33 |
| `book` | **after:** relative selection | cosine | 4/4 | **0.00** | 0.33 |

The before arm's own warnings pin the mechanism to the distance. On the
standard, "what is this document about?" returned **9 candidates with a best L2
distance of 0.699** — none within 0.60 — so all nine were discarded and the
pipeline answered from the profile. The same happened on the book, where every
answered question had zero passages.

Three things this shows:

- **The shipped configuration answered from the profile, not the document.**
  With L2 distances of 0.66–0.95 and a 0.60 cut, the book's empty-context rate
  was **1.00**: no answered question received a single passage. On the standard
  it was 0.67. The old ablation could not see this — the column did not exist,
  and its questions were anchored to pages that do have a nearby match.
- **The space change alone is not the lesson.** Cosine distances are smaller, so
  in the middle arm the same absolute 0.60 happens to pass these two documents
  (empty rate 0.00). That is luck of scale, not safety: a fixed value still
  cannot know whether it is right for a document it has not seen. The relative
  rule plus the never-empty invariant are what make the outcome hold rather than
  happen — and if the filter still empties, the best passages are sent anyway
  with a warning.
- **Profile-only rate is not meant to reach zero.** A metadata question is
  *supposed* to be answered from the profile and cites `[document]`; that is why
  the after arm sits at 0.40 and 0.33 rather than 0. Empty-context rate is the
  clean signal, and profile-only rate is read beside it, not instead of it.

The reproduction is committed as `results-regression.json`, and
`eval/run_shapes.py` is the code path that now fails loudly if empty-context
rate moves off zero.

## Decisions taken

Among the knobs (from the ablation above):

| Setting | Was | Now | Why |
| --- | --- | --- | --- |
| `RERANK_ENABLED` | `true` | **`false`** | no accuracy gain across repetitions at 2.5× the cost |
| `RERANK_SKIP_DISTANCE` | `0.35` | → `RERANK_SKIP_RATIO=0.60` | relative to the query's own candidates |

From the generalisation work (current defaults):

| Setting | Was | Now | Why |
| --- | --- | --- | --- |
| `RETRIEVAL_MAX_DISTANCE` | `0.60` selector | **`1.50` noise floor** | an absolute cut deleted the broad questions on a second document |
| `RETRIEVAL_RELATIVE_MARGIN` | — | **`1.15`** | select against the best match for *this* query |
| `RETRIEVAL_ABSOLUTE_SLACK` | — | **`0.10`** | a pure multiplier is too tight when the best distance is small |
| `CHROMA_SPACE` | unset (L2) | **`cosine`** | distances must mean the same thing across models and documents |
| `EMBEDDING_SCHEMA_VERSION` | `1` | **`2`** | the space change invalidates existing indexes; detected and reported |
| `MAX_PAGES_PER_DOCUMENT` | `300` | **`2000`** | the product rejected the document the evaluation used |
| `MAX_CHUNKS_PER_DOCUMENT` | `1500` | **`20000`** | same, and it bounds the embedding call count |
| `MAX_CONTEXT_TOKENS` | — | **`3000`** | a character budget means something different for CJK text |

`RETRIEVAL_TOP_K=5` and `chunk_size=1000` were confirmed rather than changed.
`RETRIEVAL_CANDIDATES` only applies when reranking is enabled, which is now
off by default.

Every one of these choices, and the reversals behind them, is recorded with its
reasoning in [decisions.md](decisions.md).

## Limitations

- **Neither evaluation is large.** Thirty-three questions on one book, and four
  questions per shape on five documents, is enough to catch a class of failure,
  not to rank close alternatives. The knob ablation's single-pass rows are
  directional; the claims that survive repetition are marked as such.
- **The before/after is reconstructed, not a captured "before".** The plan asked
  for a snapshot before the fix landed; it was not taken, so
  `eval/reproduce_regression.py` re-creates the pre-fix state (L2 space, absolute
  0.60, no fallback) as a subclass and runs both arms over the same documents.
  The emptiness it produces is the property being demonstrated and is robust;
  the accuracy column is a single pass over a handful of questions and is
  directional, like every other single-pass accuracy number here.
- **The relative margins are still tuned values.** `RETRIEVAL_RELATIVE_MARGIN`
  and `RETRIEVAL_ABSOLUTE_SLACK` were chosen so the shape set passes; they have
  not been swept the way the old absolute value was. Re-run
  `python -m eval.run_shapes` against your own documents before trusting them.
- **The shipped profile prefers the document's own metadata and bookmarks.**
  With `DOCUMENT_SUMMARY_ALWAYS=false`, a document that has a bookmark outline
  skips the model call, so fields the PDF does not record (a publisher, for
  example) are not in the profile. Set it to `true` to pay for a model reading
  of the opening pages as well.
- **One question fails in every configuration.** q15, "what stops an attacker
  pretending to be a company in order to steal login details?" (phishing, pages
  105–107), was missed in every run, while the direct question "what is
  phishing?" passes. It is left in the set deliberately: a test set containing
  only questions the system answers is not a test set.
- **OCR is still out of scope.** Scanned and image-only PDFs are rejected with a
  clear message rather than transcribed.
- **Cost is estimated from characters**, not read from provider billing, so
  treat `$` columns as relative comparisons rather than invoices.
- **Answer accuracy is keyword-based.** It cannot detect a confidently wrong
  answer that happens to contain the expected terms, so read it alongside the
  abstention and citation columns.
- **The baseline in the ablation still had a profile chunk in the index.**
  Reranking, rewriting, threshold and profile-in-context are switchable at query
  time, but the profile chunk is created at ingest, so the baseline could not
  un-index it.

## Reproducing

```bash
cd backend
python -m eval.run_eval --limit 3                 # smoke test: 3 questions
python -m eval.run_eval --config baseline         # one configuration
python -m eval.run_eval --all                     # the whole ablation
python -m eval.run_eval --config baseline --config "+profile" \
    --config "+rerank" --config "tuned (shipped)" --repeat 3
python -m eval.run_shapes                         # across document shapes
python -m eval.reproduce_regression               # before/after, both documents
python -m eval.diagnose --questions               # per-question retrieval detail
```

The harness runs the real providers, so it costs money and needs both API
keys. It is deliberately not part of the offline test suite; `pytest` remains
network-free.

`eval/diagnose.py` is the smallest tool here and the one to reach for first: it
prints each question's candidates, their sorted distances, the threshold in
force and how much context would be sent, and **exits non-zero if any question
yields zero passages**, so it can gate a release. See
[release-checklist.md](release-checklist.md).

Raw per-question output, including every answer, retrieved page and retrieval
telemetry, is committed alongside this file as `results.json` (the ablation),
`results-repeats.json` (three repetitions of the four headline configurations),
`shapes.json` (the five document shapes) and `results-regression.json` (the
reconstructed before/after). Each file carries a `schema_version`, the document
identity it was measured against and the code revision, and files measured under
superseded settings are marked `historical` rather than deleted.
