# Evaluation results

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
| Question set | `backend/eval/questions.jsonl` — 30 questions, committed so every configuration is measured against an unchanged set |
| Answers | DeepSeek `deepseek-chat`, one call per question |
| Runs | one pass per configuration for the ablation; four headline configurations repeated three times |

The question set deliberately contains five kinds of question:

- **12 content** — the answer is stated on a page, in the document's own words.
- **5 paraphrase** — the answer exists but the question does not use the document's vocabulary (these exercise query rewriting).
- **3 ambiguous** — several pages could answer, so ranking matters more than presence (these exercise reranking).
- **4 metadata** — about the document itself (title, author, publisher, edition). These have no gold page and are answered from the document profile, so they are excluded from the retrieval metrics.
- **6 unanswerable** — the document cannot answer them, which measures abstention rather than recall.

## How the metrics are defined

| Metric | Definition |
| --- | --- |
| hit@5 | a gold page appears among the five pages supplied to the model |
| MRR | mean reciprocal rank of the first gold page, so putting the right page first scores higher than merely including it |
| Answer accuracy | over the 24 answerable questions: every expected keyword appears in the answer and the answer is not an abstention |
| Abstention precision / recall | abstention is treated as the positive class: a refusal on an answerable question is a false positive, the failure mode users notice most |
| Cited | share of answers that cite anything |
| Citations valid | share of citing answers where every `[p.N]` matches a page that was actually supplied, and `[document]` is used only when a profile was supplied |
| p50 | median end-to-end latency per question |
| $/question | estimated from characters at published list prices; an estimate, not a bill |

Answer correctness is checked by keyword matching rather than by a model
judge, on purpose: an ablation comparison must not be skewed by the variance
of a model grading its own output.

## Ablation across knob values

> **Historical.** These rows were measured in the old L2 space with per-page
> chunking and an absolute 0.60 distance threshold, before the generalisation
> work. They document how each *knob* behaved; the shipped defaults have since
> changed, and distances below are not comparable with the cosine distances in
> the next section.

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

`+threshold` scores 1.8 calls per question rather than 2.0 because a stricter
threshold empties the context for some questions, and the pipeline then
abstains without calling the model at all.

Treat the accuracy column above with care: it is a single pass, and the
repetitions below show that one pass can be 13 points off.

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

**A 0.60 distance threshold is not worse than 0.75 and costs less.** The
tighter threshold shrinks the prompt that generation is billed for, and the
`tuned` configuration is the cheapest accurate one measured ($0.00036 per
question against $0.00065–0.00151 for equally accurate configurations).

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
cited entirely as `[document]`. A 30-question set anchored to specific pages
could not observe that, because every question it contained had a nearby page.

`eval/run_shapes.py` ingests five documents with the shipped settings and fails
loudly if any question is answered with an empty context:

| Document | Shape | Questions | Answered | Abstained correctly | Questions with no passages | Fewest passages |
| --- | --- | --- | --- | --- | --- | --- |
| `sheet` | 2-page form, no bookmarks | 3 | 2/2 | 1/1 | 0 | 2 |
| `table` | table of sensor readings | 3 | 2/2 | 1/1 | 0 | 2 |
| `german` | 1-page German document | 2 | 1/1 | 1/1 | 0 | 2 |
| `standard` | 157-page technical standard | 4 | 3/3 | 1/1 | 0 | 5 |
| `book` | 658-page textbook | 4 | 3/3 | 1/1 | 0 | 5 |

Sixteen of sixteen questions behaved correctly and **no question was ever
answered with an empty context**, including the questions that used to fail:

| Question | Document | Passages | Best distance |
| --- | --- | --- | --- |
| "What is this document about?" | standard | 5 | 0.349 |
| "What is the title of this document?" | standard | 5 | 0.346 |
| "List the sections of this document." | standard | 5 | 0.331 |
| "What is this book about?" | book | 5 | 0.331 |
| "List the chapters of this book." | book | 5 | 0.296 |
| "What was the temperature on 3 March 2024?" | table | 2 | 0.278 |
| "Wie oft erfolgt die Wartung der Heizungsanlage?" | german | 2 | 0.266 |

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
explicitly on the collection. The values above sit between 0.19 and 0.55, so
the old absolute 0.60 cut would have removed the top of that range outright.



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

## Limitations

- **Neither evaluation is large.** Thirty-three questions on one book, and four
  questions per shape on five documents, is enough to catch a class of failure,
  not to rank close alternatives. The knob ablation's single-pass rows are
  directional; the claims that survive repetition are marked as such.
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
```

The harness runs the real providers, so it costs money and needs both API
keys. It is deliberately not part of the offline test suite; `pytest` remains
network-free.

Raw per-question output, including every answer, retrieved page and retrieval
telemetry, is committed alongside this file as `results.json` (the ablation),
`results-repeats.json` (three repetitions of the four headline configurations)
and `shapes.json` (the five document shapes).
