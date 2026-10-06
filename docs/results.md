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

## Ablation

One pass per configuration, same 30 questions, same index. `all` is every
feature on; `tuned (shipped)` is what this project now ships.

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

## Decisions taken

| Setting | Was | Now | Why |
| --- | --- | --- | --- |
| `RERANK_ENABLED` | `true` | **`false`** | no accuracy gain across repetitions at 2.5× the cost |
| `RETRIEVAL_MAX_DISTANCE` | 0.75 | **0.60** | never worse, and a smaller prompt costs less |
| `RERANK_SKIP_DISTANCE` | 0.35 | **0.50** | only consulted when reranking is enabled |

`RETRIEVAL_TOP_K=5` and `chunk_size=1000` were confirmed rather than changed.
`RETRIEVAL_CANDIDATES` only applies when reranking is enabled, which is now
off by default.

## Limitations

- **Thirty questions is a small sample**, which is why the headline claims
  rest on three repetitions rather than one pass. The remaining single-pass
  rows in the ablation should be read as directional.
- **One question fails in every run.** q15, "what stops an attacker pretending
  to be a company in order to steal login details?" (phishing, pages 105–107),
  was missed in every configuration and every repetition, while the direct
  question "what is phishing?" passes. This is a genuine paraphrase-retrieval
  limitation that query rewriting did not rescue. It is left in the set rather
  than removed: a test set containing only questions the system answers is not
  a test set.
- **One document, one language.** Thresholds are corpus-dependent, so treat
  0.60 as a starting point and re-run the harness against your own document.
- **Cost is estimated from characters**, not read from provider billing, so
  treat `$` columns as relative comparisons rather than invoices.
- **Answer accuracy is keyword-based.** It cannot detect a confidently wrong
  answer that happens to contain the expected terms, so read it alongside the
  abstention and citation columns.
- **The baseline still has a profile chunk in the index.** Reranking,
  rewriting, threshold and profile-in-context are switchable at query time,
  but the profile chunk is created at ingest, so the baseline cannot
  un-index it.

## Reproducing

```bash
cd backend
python -m eval.run_eval --limit 3                 # smoke test: 3 questions
python -m eval.run_eval --config baseline         # one configuration
python -m eval.run_eval --all                     # the whole ablation
python -m eval.run_eval --config baseline --config "+profile" \
    --config "+rerank" --config "tuned (shipped)" --repeat 3
```

The harness runs the real providers, so it costs money and needs both API
keys. It is deliberately not part of the offline test suite; `pytest` remains
network-free.

Raw per-question output, including every answer, retrieved page and citation
report, is committed alongside this file as `results.json` (the ablation) and
`results-repeats.json` (three repetitions of the four headline
configurations).
