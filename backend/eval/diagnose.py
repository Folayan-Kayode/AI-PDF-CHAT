"""
Diagnose one retrieval failure at a time.

The failure this exists for: a question comes back with no passages, the
answer is generated from the document profile alone, and the phrasing gives no
hint that retrieval returned nothing. By the time that reaches a metrics table
it is one column among many. Here it is the whole output.

It prints, per question, the candidates the index returned, their sorted cosine
distances, the threshold actually applied and how much context would be sent.
It exits non-zero if any question yields zero passages, so it can gate a
release.

Usage, from backend/:

    python -m eval.diagnose --question "What is the document about?"
    python -m eval.diagnose --question "..." --question "..." --document-id abc123
    python -m eval.diagnose --questions            # the committed question set
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from app.rag.retriever import Retriever
from eval.run_eval import QUESTIONS_PATH, load_questions


def describe(retriever: Retriever, question: str, top_k: int | None) -> dict[str, Any]:
    """One question's retrieval, with nothing aggregated away."""
    result = retriever.retrieve(question, n_results=top_k)

    documents = result.get("documents") or []

    return {
        "question": question,
        "candidates_returned": result.get("considered_candidates", 0),
        "best_distance": result.get("best_distance"),
        "cut_distance": result.get("cut_distance"),
        "candidate_distances": result.get("candidate_distances") or [],
        "passages_supplied": len(documents),
        "context_chars": sum(len(document) for document in documents),
    }


def report(row: dict[str, Any]) -> None:
    distances = row["candidate_distances"]

    best = row["best_distance"]
    cut = row["cut_distance"]

    print(f"\n{row['question']}", flush=True)
    print(
        f"  candidates returned : {row['candidates_returned']}",
        flush=True,
    )
    print(
        f"  distances           : "
        f"{', '.join(f'{distance:.3f}' for distance in distances[:12])}"
        f"{' ...' if len(distances) > 12 else ''}",
        flush=True,
    )
    print(
        f"  best / cut          : "
        f"{'n/a' if best is None else f'{best:.3f}'} / "
        f"{'n/a' if cut is None else f'{cut:.3f}'}",
        flush=True,
    )
    print(f"  passages supplied   : {row['passages_supplied']}", flush=True)
    print(f"  context chars       : {row['context_chars']}", flush=True)

    if row["passages_supplied"] == 0:
        print(
            "  >> FAIL: zero passages. The answer would come from the profile alone.",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose retrieval for a question.")

    parser.add_argument("--question", action="append", default=[])
    parser.add_argument(
        "--questions",
        action="store_true",
        help="Use the committed question set instead of --question.",
    )
    parser.add_argument("--questions-path", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()

    questions = list(args.question)

    if args.questions:
        questions.extend(entry["question"] for entry in load_questions(args.questions_path))

    if not questions:
        parser.error("give at least one --question, or --questions")

    retriever = Retriever(document_id=args.document_id)

    status = retriever.index_status()

    print(f"index status : {status}", flush=True)

    if status.get("count") == 0:
        print(
            "\nThe index is empty. Upload the document first; nothing below can pass.",
            flush=True,
        )

    rows = [describe(retriever, question, args.top_k) for question in questions]

    for row in rows:
        report(row)

    failures = [row for row in rows if row["passages_supplied"] == 0]

    print(
        f"\n{len(rows) - len(failures)}/{len(rows)} questions produced passages.",
        flush=True,
    )

    if failures:
        print(
            f"{len(failures)} produced none. That is the failure this tool checks for.",
            flush=True,
        )

        return 1

    return 0


if __name__ == "__main__":
    # get_database is process-wide; nothing to close explicitly here.
    sys.exit(main())
