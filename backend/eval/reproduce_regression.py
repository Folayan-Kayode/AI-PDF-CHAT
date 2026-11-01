"""
Reproduce the retrieval regression and show the fix against it.

What shipped: an absolute 0.60 distance threshold, in L2 space, with no
fallback when the filter removed everything. On the book it was tuned on it
looked fine. On a second document it deleted every passage for broad questions:
retrieval returned nothing, the answer was generated from the document profile
alone, and the phrasing gave no hint. hit@5, accuracy and cost all still looked
healthy, because none of them can see an empty context.

The remediation plan wanted a "before" snapshot captured before the fix landed.
It was not, so this reconstructs the pre-fix state explicitly and runs the arms
over the same documents. The old rule is a subclass, not a setting: it is
evidence tooling, not a supported mode.

Two things changed together, so the arms separate them:

    1. before  -- L2 space, absolute 0.60, no fallback   (what shipped)
    2. middle  -- cosine space, absolute 0.60, no fallback
    3. after   -- cosine space, relative selection        (what ships now)

Each arm's index is built in that arm's vector space, because a distance is
only meaningful within one space.

Usage, from backend/:

    python -m eval.reproduce_regression
    python -m eval.reproduce_regression --out ../docs/results-regression.json
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from eval.metrics import context_empty_rate, profile_only_rate
from eval.run_eval import clear_caches, git_revision, ingest, run_limits

logging.basicConfig(level=logging.WARNING)

HERE = Path(__file__).parent

# Permissively licensed substitutes, committed so the reproduction runs on a
# fresh clone (see eval/documents/README.md). The original copyrighted
# documents can be supplied locally with --document, never committed.
DOCUMENTS = {
    "standard": HERE / "documents" / "standard_http.pdf",
    "book": HERE / "documents" / "book_mobydick.pdf",
}

# The reproduction set, per document: the broad/aggregate questions that an
# absolute threshold deletes, a metadata question, and one the document cannot
# answer at all. Questions are document-specific on purpose -- a question about
# audio equipment asked of a novel scores zero for reasons that have nothing to
# do with retrieval.
QUESTIONS: dict[str, list[dict[str, Any]]] = {
    "standard": [
        {
            "question": "What is this document about?",
            "expect": ["http"],
            "category": "broad",
        },
        {
            "question": "List the sections of this document.",
            "expect": ["introduction"],
            "category": "broad",
        },
        {
            "question": "What is the title of this document?",
            "expect": ["http semantics"],
            "category": "metadata",
        },
        {
            "question": "What does the document say about methods?",
            "expect": ["method"],
            "category": "specific",
        },
        {
            "question": "What is the recipe for a chocolate cake?",
            "answerable": False,
            "category": "unanswerable",
        },
    ],
    "book": [
        {
            "question": "What is this book about?",
            "expect": ["whale"],
            "category": "broad",
        },
        {
            "question": "List the chapters of this book.",
            "expect": ["loomings"],
            "category": "broad",
        },
        {
            "question": "What is the title of this book?",
            "expect": ["moby"],
            "category": "metadata",
        },
        {
            "question": "What is the capital of Mongolia?",
            "answerable": False,
            "category": "unanswerable",
        },
    ],
}


class LegacyRetriever(Retriever):
    """
    The pre-fix selection: keep candidates within an absolute distance, and if
    that removes everything, send nothing.

    The second half is the part that turned a retrieval miss into a confident
    answer instead of an abstention.
    """

    def __init__(self, *args: Any, absolute_cut: float = 0.60, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.absolute_cut = absolute_cut

    def _select(self, ordered: list[Any], top_k: int) -> list[Any]:
        return [candidate for candidate in ordered if candidate.distance <= self.absolute_cut]


ARMS: list[dict[str, Any]] = [
    {
        "label": "before: L2, absolute 0.60, no fallback",
        "space": "l2",
        "retriever": lambda: LegacyRetriever(absolute_cut=0.60),
    },
    {
        "label": "middle: cosine, absolute 0.60, no fallback",
        "space": "cosine",
        "retriever": lambda: LegacyRetriever(absolute_cut=0.60),
    },
    {
        "label": "after: cosine, relative selection",
        "space": "cosine",
        "retriever": lambda: Retriever(),
    },
]


def run_arm(
    label: str,
    space: str,
    make_retriever: Any,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the question set through one retriever."""
    clear_caches()

    pipeline = RAGPipeline(retriever=make_retriever())

    rows = []

    for entry in questions:
        result = pipeline.ask(entry["question"])

        retrieval = result.get("retrieval") or {}
        citations = result.get("citations") or {}
        answer = result["answer"]
        abstained = bool(result.get("abstained"))

        answerable = entry.get("answerable", True)

        if answerable:
            correct = not abstained and all(
                keyword.lower() in answer.lower() for keyword in entry.get("expect", [])
            )
        else:
            correct = abstained

        rows.append(
            {
                "question": entry["question"],
                "category": entry["category"],
                "answerable": answerable,
                "answer": answer.strip().replace("\n", " "),
                "abstained": abstained,
                "correct": correct,
                "passages_supplied": retrieval.get("retrieved_chunks", 0),
                "candidates_returned": retrieval.get("considered_candidates", 0),
                "best_distance": retrieval.get("best_distance"),
                "cut_distance": retrieval.get("cut_distance"),
                "profile_supplied": retrieval.get("profile_used", False),
                "cited_document_only": bool(citations.get("document_cited"))
                and not citations.get("cited_pages"),
                "candidate_distances": retrieval.get("candidate_distances") or [],
            }
        )

    print(f"\n  --- {label} ---", flush=True)

    for row in rows:
        best = row["best_distance"]

        print(
            f"    passages={row['passages_supplied']:>2} "
            f"best={'n/a' if best is None else f'{best:.3f}'} "
            f"profile_only={row['cited_document_only']} "
            f"correct={row['correct']} | {row['question'][:52]}",
            flush=True,
        )

    summary = {
        "label": label,
        "space": space,
        "questions": len(rows),
        "answered": sum(1 for row in rows if not row["abstained"]),
        "correct": sum(1 for row in rows if row["correct"]),
        "context_empty_rate": context_empty_rate(rows),
        "profile_only_rate": profile_only_rate(rows),
    }

    print(
        f"    => empty ctx {summary['context_empty_rate']:.2f}  "
        f"profile-only {summary['profile_only_rate']:.2f}  "
        f"correct {summary['correct']}/{summary['questions']}",
        flush=True,
    )

    return {"summary": summary, "results": rows}


def run_document(name: str, document: Path) -> dict[str, Any]:
    """
    Run every arm against one document.

    The index is rebuilt whenever the arm changes the vector space, because a
    distance is only meaningful within one space.
    """
    print(f"\n=== {name} ({document.name}) ===", flush=True)

    if not document.exists():
        print("    SKIPPED: document not available", flush=True)

        return {"document": name, "skipped": True, "arms": []}

    original_space = settings.CHROMA_SPACE

    arms = []

    indexed_space: str | None = None

    try:
        for arm in ARMS:
            settings.CHROMA_SPACE = arm["space"]

            if indexed_space != arm["space"]:
                clear_caches()

                ingest(document, 1000)

                indexed_space = arm["space"]

            arms.append(run_arm(arm["label"], arm["space"], arm["retriever"], QUESTIONS[name]))
    finally:
        settings.CHROMA_SPACE = original_space

    return {
        "document": name,
        "file": str(document),
        "skipped": False,
        "arms": arms,
    }


def to_markdown(runs: list[dict[str, Any]]) -> str:
    header = (
        "| Document | Arm | Space | Correct | Empty context rate | "
        "Profile-only rate |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )

    rows = []

    for run in runs:
        if run.get("skipped"):
            continue

        for arm in run["arms"]:
            summary = arm["summary"]

            rows.append(
                f"| `{run['document']}` | {summary['label']} "
                f"| {summary['space']} "
                f"| {summary['correct']}/{summary['questions']} "
                f"| {summary['context_empty_rate']:.2f} "
                f"| {summary['profile_only_rate']:.2f} |"
            )

    return header + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Override a document by shape name, for documents that must not be "
            "committed (e.g. --document book=C:\\path\\textbook.pdf)."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Accept a run where some documents are absent instead of failing.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)

    args = parser.parse_args()

    documents = dict(DOCUMENTS)

    for override in args.document:
        if "=" not in override:
            parser.error(f"--document must be NAME=PATH, got {override!r}")

        name, _, path = override.partition("=")

        if name not in documents:
            parser.error(f"unknown document {name!r}; expected one of {sorted(documents)}")

        documents[name] = Path(path)

    names = args.only or list(documents)

    runs = [run_document(name, documents[name]) for name in names]

    print()
    print(to_markdown(runs))

    skipped = [run for run in runs if run.get("skipped")]

    print(f"{len(runs) - len(skipped)} of {len(runs)} documents ran.", flush=True)

    for run in skipped:
        print(f"  MISSING: {run['document']} ({run.get('file')})", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "regression-reproduction",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "revision": git_revision(),
                    "documents": [run["document"] for run in runs],
                    "documents_ran": len(runs) - len(skipped),
                    "documents_total": len(runs),
                    "missing": [run["document"] for run in skipped],
                    "limits": run_limits(),
                    "note": (
                        "The 'before' arm reconstructs the pre-fix state (L2 "
                        "space, absolute 0.60, no fallback) rather than "
                        "reverting code; the original before-snapshot was not "
                        "captured. The 'middle' arm separates the vector-space "
                        "change from the selection-rule change."
                    ),
                    "runs": runs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(runs), encoding="utf-8")
        print(f"wrote {args.markdown}")

    if skipped and not args.allow_missing:
        print(
            "Refusing to report a quietly smaller run. Provide the missing "
            "documents, or pass --allow-missing to accept the partial run.",
            flush=True,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
