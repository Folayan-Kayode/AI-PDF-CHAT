"""
Evaluation harness (A2) and ablation runner (A4).

Runs the committed question set through the real pipeline -- real embeddings
and real generation -- so it costs money and needs both API keys. It is
therefore kept out of the unit test suite, which stays offline.

Usage, from backend/:

    python -m eval.run_eval --limit 3                 # smoke test
    python -m eval.run_eval --config baseline         # one configuration
    python -m eval.run_eval --all                     # the whole matrix
    python -m eval.run_eval --all --out ../docs/results.json \\
        --markdown ../docs/results.md

The question set is committed at eval/questions.jsonl so every run, and every
configuration, is measured against an unchanged set.
"""

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.database.chroma import get_database
from app.rag.citations import PROFILE_PAGE
from app.rag.pipeline import is_abstention
from eval.metrics import CallCounter, answer_matches, estimate_cost, summarise

logging.basicConfig(level=logging.WARNING)

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")

DEFAULT_DOCUMENT = Path(r"C:\Users\Davel\Documents\Python\Whitman.pdf")


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------

# Everything the baseline turns off is a retrieval feature added by judgement;
# the ablation exists to find out which of them earn their extra calls.
BASELINE: dict[str, Any] = {
    "QUERY_REWRITE_ENABLED": False,
    "RERANK_ENABLED": False,
    "DOCUMENT_PROFILE_IN_CONTEXT": False,
    "RETRIEVAL_MAX_DISTANCE": 10.0,  # threshold disabled
    "RETRIEVAL_TOP_K": 5,
    "chunk_size": 1000,
}


def _with(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return {**base, **overrides}


ALL_ON: dict[str, Any] = _with(
    BASELINE,
    QUERY_REWRITE_ENABLED=True,
    RERANK_ENABLED=True,
    DOCUMENT_PROFILE_IN_CONTEXT=True,
    RETRIEVAL_MAX_DISTANCE=0.75,
)

#: What this project actually ships (see the configuration table in README.md).
#: Chosen from the matrix below, not by intuition: the 0.60 threshold matched
#: 0.75's accuracy at roughly half the cost, and reranking was left off.
TUNED: dict[str, Any] = _with(
    BASELINE,
    QUERY_REWRITE_ENABLED=True,
    DOCUMENT_PROFILE_IN_CONTEXT=True,
    RETRIEVAL_MAX_DISTANCE=0.60,
)

CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": BASELINE,
    "+threshold": _with(BASELINE, RETRIEVAL_MAX_DISTANCE=0.75),
    "+rewrite": _with(BASELINE, QUERY_REWRITE_ENABLED=True),
    "+rerank": _with(BASELINE, RERANK_ENABLED=True),
    "+profile": _with(BASELINE, DOCUMENT_PROFILE_IN_CONTEXT=True),
    "all": ALL_ON,
    "tuned (shipped)": TUNED,
    "top_k=3": _with(ALL_ON, RETRIEVAL_TOP_K=3),
    "top_k=8": _with(ALL_ON, RETRIEVAL_TOP_K=8),
    "max_distance=0.60": _with(ALL_ON, RETRIEVAL_MAX_DISTANCE=0.60),
    "max_distance=0.90": _with(ALL_ON, RETRIEVAL_MAX_DISTANCE=0.90),
    "skip_distance=0.20": _with(ALL_ON, RERANK_SKIP_DISTANCE=0.20),
    "skip_distance=0.50": _with(ALL_ON, RERANK_SKIP_DISTANCE=0.50),
    "chunk=500": _with(ALL_ON, chunk_size=500),
    "chunk=1500": _with(ALL_ON, chunk_size=1500),
}


class CountingProxy:
    """
    Passes calls through to whichever counter is current.

    The pipeline is instrumented once per configuration, but cost and call
    attribution are needed per question, so the proxy's target is swapped for
    each question.
    """

    def __init__(self) -> None:
        self.current = CallCounter()

    def record(self, kind: str, input_text: str = "", output_text: str = "") -> None:
        self.current.record(kind, input_text=input_text, output_text=output_text)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Read the committed question set."""
    questions = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line:
            questions.append(json.loads(line))

    return questions


def apply_config(config: dict[str, Any]) -> None:
    """Point the live settings at a configuration."""
    for name, value in config.items():
        if name != "chunk_size":
            setattr(settings, name, value)


def clear_caches() -> None:
    """Drop cached clients so a configuration cannot inherit another's state."""
    from app.rag.embeddings import get_embedding_model
    from app.rag.generator import get_generator
    from app.rag.llm import get_chat_client
    from app.rag.pipeline import get_pipeline

    for cached in (get_pipeline, get_database, get_embedding_model, get_generator):
        cached.cache_clear()

    get_chat_client.cache_clear()


def ingest(document: Path, chunk_size: int) -> None:
    """
    Re-index the document at a given chunk size.

    The collection is dropped first because ingestion short-circuits on
    duplicate content, which would otherwise leave the previous chunking in
    place and silently invalidate the comparison.
    """
    from app.services import pdf_service

    original_splitter = pdf_service.TextSplitter

    def splitter() -> Any:
        return original_splitter(
            chunk_size=chunk_size,
            chunk_overlap=max(1, chunk_size // 5),
        )

    pdf_service.TextSplitter = splitter

    try:
        get_database().reset()

        started = time.perf_counter()

        # The production ingestion limits are used as they ship: a harness that
        # has to raise them is measuring a configuration the product rejects.
        result = pdf_service.PDFService.process(document)

        print(
            f"    indexed {len(result['chunks'])} chunks at chunk_size={chunk_size} "
            f"in {time.perf_counter() - started:.0f}s",
            flush=True,
        )
    finally:
        pdf_service.TextSplitter = original_splitter


def build_pipeline(proxy: CountingProxy) -> Any:
    """A pipeline whose provider calls are counted for cost and attribution."""
    from app.rag.pipeline import get_pipeline

    clear_caches()

    pipeline = get_pipeline()

    retriever = pipeline.retriever

    original_embed_query = retriever.embedding_model.embed_query

    def counted_embed_query(text: str) -> Any:
        result = original_embed_query(text)
        proxy.record("embedding", input_text=text)
        return result

    retriever.embedding_model.embed_query = counted_embed_query

    original_rewrite = retriever.query_rewriter.rewrite

    def counted_rewrite(question: str) -> Any:
        result = original_rewrite(question)

        # A disabled feature makes no provider call, so counting it would
        # inflate calls-per-question for every configuration that has it off.
        if settings.QUERY_REWRITE_ENABLED:
            proxy.record("rewrite", input_text=question, output_text=result or "")

        return result

    retriever.query_rewriter.rewrite = counted_rewrite

    original_rerank = retriever.reranker.rerank

    def counted_rerank(question: str, passages: list[str]) -> Any:
        result = original_rerank(question, passages)

        if settings.RERANK_ENABLED:
            proxy.record(
                "rerank",
                input_text=question + "".join(passages),
                output_text=str(result or ""),
            )

        return result

    retriever.reranker.rerank = counted_rerank

    original_generate = pipeline.generator.generate

    def counted_generate(prompt: str) -> str:
        answer = original_generate(prompt)
        proxy.record("generate", input_text=prompt, output_text=answer or "")
        return answer

    pipeline.generator.generate = counted_generate

    return pipeline


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def run_one(question: dict[str, Any], counter: CallCounter, pipeline: Any) -> dict[str, Any]:
    """Ask one question and collect everything the metrics need."""
    started = time.perf_counter()

    result = pipeline.ask(question["question"])

    latency = time.perf_counter() - started

    answer = result.get("answer") or ""

    retrieved_pages = [
        source.get("page")
        for source in (result.get("sources") or [])
        if source
        and source.get("kind") != "document_summary"
        and source.get("page") is not None
        and source.get("page") != PROFILE_PAGE
    ]

    abstained = is_abstention(answer)

    if question["answerable"]:
        correct = not abstained and answer_matches(answer, question["expected_keywords"])
    else:
        correct = abstained

    calls = counter.as_dict()

    citations = result.get("citations") or {}
    retrieval = result.get("retrieval") or {}

    return {
        "id": question["id"],
        "category": question["category"],
        "answerable": question["answerable"],
        "question": question["question"],
        "answer": answer,
        "expected_pages": question["expected_pages"],
        "retrieved_pages": retrieved_pages,
        "abstained": abstained,
        "correct": correct,
        "citations": citations,
        "latency_seconds": latency,
        "calls": calls,
        "cost_usd": estimate_cost(calls),
        # The fields that make a silent retrieval failure visible. Without
        # these, a configuration can score well on accuracy and cost while
        # sending no context at all.
        "passages_supplied": retrieval.get("retrieved_chunks", 0),
        "candidates_returned": retrieval.get("considered_candidates", 0),
        "best_distance": retrieval.get("best_distance"),
        "cut_distance": retrieval.get("cut_distance"),
        "profile_supplied": retrieval.get("profile_used", False),
        "cited_document_only": bool(citations.get("document_cited"))
        and not citations.get("cited_pages"),
        "candidate_distances": retrieval.get("candidate_distances") or [],
    }


def run_config(
    name: str,
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    document: Path,
    indexed: dict[str, int],
    repetition: int = 1,
) -> dict[str, Any]:
    """Run every question under one configuration."""
    label = name if repetition == 1 else f"{name} (run {repetition})"

    print(f"\n=== {label} ===", flush=True)

    apply_config(config)

    chunk_size = config["chunk_size"]

    if indexed.get("chunk_size") != chunk_size:
        ingest(document, chunk_size)
        indexed["chunk_size"] = chunk_size

    proxy = CountingProxy()
    pipeline = build_pipeline(proxy)

    results = []

    for index, question in enumerate(questions, start=1):
        proxy.current = CallCounter()

        row = run_one(question, proxy.current, pipeline)

        results.append(row)

        mark = "ok  " if row["correct"] else "MISS"

        # flush so a long run is observable while it is still going: stdout is
        # block buffered when it is piped, which hides progress entirely.
        print(
            f"  [{index:2}/{len(questions)}] {mark} {row['id']} "
            f"{row['latency_seconds']:.1f}s {question['category']}",
            flush=True,
        )

    summary = summarise(results)

    print(
        f"  -> hit@5 {summary['retrieval_hit_rate']:.2f}  "
        f"mrr {summary['mrr']:.2f}  "
        f"accuracy {summary['answer_accuracy']:.2f}  "
        f"abstain P/R {summary['abstention_precision']:.2f}/"
        f"{summary['abstention_recall']:.2f}  "
        f"citation {summary['citation_rate']:.2f}/"
        f"{summary['citation_validity']:.2f}  "
        f"p50 {summary['latency_p50_seconds']:.1f}s  "
        f"${summary['cost_per_question_usd']:.5f}/q",
        flush=True,
    )

    return {
        "config": name,
        "repeat": repetition,
        "settings": dict(config),
        "summary": summary,
        "results": results,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def to_markdown(runs: list[dict[str, Any]]) -> str:
    """Render the results table."""
    header = (
        "| Configuration | hit@5 | MRR | Answer acc. | Abstain P | Abstain R | "
        "Cited | Citations valid | Empty ctx | Profile only | p50 s | $/question | calls/q |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )

    rows = []

    for run in runs:
        summary = run["summary"]

        rows.append(
            f"| `{run['config']}` "
            f"| {summary['retrieval_hit_rate']:.2f} "
            f"| {summary['mrr']:.2f} "
            f"| {summary['answer_accuracy']:.2f} "
            f"| {summary['abstention_precision']:.2f} "
            f"| {summary['abstention_recall']:.2f} "
            f"| {summary['citation_rate']:.2f} "
            f"| {summary['citation_validity']:.2f} "
            f"| {summary.get('context_empty_rate', 0.0):.2f} "
            f"| {summary.get('profile_only_rate', 0.0):.2f} "
            f"| {summary['latency_p50_seconds']:.1f} "
            f"| {summary['cost_per_question_usd']:.5f} "
            f"| {summary['calls_per_question']:.1f} |"
        )

    return header + "\n".join(rows) + "\n"


# --------------------------------------------------------------------------
# Output versioning
# --------------------------------------------------------------------------

# Bump when the shape of a results file changes, so a reader can tell whether
# it understands the file rather than guessing from the fields present.
RESULTS_SCHEMA_VERSION = 1


def git_revision() -> str | None:
    """The commit a run was produced from, if this is a git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def document_identity(path: Path) -> dict[str, Any]:
    """
    Identify the corpus a run describes.

    The filename alone is not enough: the same book can be re-exported, and an
    evaluation that cannot say which bytes it measured is not evidence.
    """
    if not path.exists():
        return {"path": str(path), "exists": False}

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256_16": digest.hexdigest()[:16],
        "exists": True,
    }


def run_limits() -> dict[str, Any]:
    """The settings that materially change a result, recorded alongside it."""
    return {
        "max_pages_per_document": settings.MAX_PAGES_PER_DOCUMENT,
        "max_chunks_per_document": settings.MAX_CHUNKS_PER_DOCUMENT,
        "retrieval_max_distance": settings.RETRIEVAL_MAX_DISTANCE,
        "retrieval_relative_margin": settings.RETRIEVAL_RELATIVE_MARGIN,
        "retrieval_absolute_slack": settings.RETRIEVAL_ABSOLUTE_SLACK,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "rerank_enabled": settings.RERANK_ENABLED,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_schema_version": settings.EMBEDDING_SCHEMA_VERSION,
        "chroma_space": settings.CHROMA_SPACE,
        "document_summary_always": settings.DOCUMENT_SUMMARY_ALWAYS,
    }


def build_envelope(
    runs: list[dict[str, Any]],
    document: Path,
    questions: Path | None = None,
    kind: str = "ablation",
) -> dict[str, Any]:
    """Wrap a run list with the provenance a future reader needs."""
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "kind": kind,
        "generated_at": datetime.now(UTC).isoformat(),
        "revision": git_revision(),
        "document": document_identity(document),
        "questions": {
            "path": str(questions) if questions else None,
            "count": sum(len(run.get("results", [])) for run in runs),
        },
        "limits": run_limits(),
        "runs": runs,
    }


def load_runs(path: Path | str) -> list[dict[str, Any]]:
    """Read the runs from a versioned envelope or from a bare list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    return data.get("runs", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")

    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run each configuration this many times. Answer generation is not "
            "deterministic, so a single pass cannot distinguish a real "
            "difference from noise on a small question set."
        ),
    )
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)

    args = parser.parse_args()

    names = list(CONFIGS) if args.all else (args.config or ["baseline"])

    unknown = [name for name in names if name not in CONFIGS]

    if unknown:
        raise SystemExit(f"unknown configuration(s): {unknown}")

    questions = load_questions(args.questions)

    if args.limit:
        questions = questions[: args.limit]

    print(f"document  : {args.document}")
    print(f"questions : {len(questions)}")

    indexed: dict[str, int] = {}
    runs = []

    for name in names:
        for repetition in range(1, args.repeat + 1):
            runs.append(
                run_config(
                    name,
                    CONFIGS[name],
                    questions,
                    args.document,
                    indexed,
                    repetition=repetition,
                )
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                build_envelope(runs, args.document, args.questions),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(runs), encoding="utf-8")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
