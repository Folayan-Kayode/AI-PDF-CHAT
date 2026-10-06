"""
Run the evaluation across document shapes, not just knob values.

The knob ablation in run_eval.py used one 658-page English textbook. That is
how an absolute distance threshold tuned to 0.60 shipped and then emptied the
context for broad questions on a completely different document.

This runner ingests each shape with the *production* settings and checks, per
document, that questions are actually answered from the document body.

Usage, from backend/:

    python -m eval.run_shapes
    python -m eval.run_shapes --only standard
    python -m eval.run_shapes --out ../docs/shapes.json --markdown <path>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from app.rag.pipeline import RAGPipeline
from eval.run_eval import clear_caches, ingest

logging.basicConfig(level=logging.WARNING)

HERE = Path(__file__).parent

SHAPES_PATH = HERE / "shapes.jsonl"


def load_shapes(path: Path = SHAPES_PATH) -> list[dict[str, Any]]:
    """Read the shape definitions, resolving relative document paths."""
    shapes = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        shape = json.loads(line)

        document = Path(shape["path"])

        shape["document"] = document if document.is_absolute() else HERE / document

        shapes.append(shape)

    return shapes


def run_shape(shape: dict[str, Any], chunk_size: int = 1000) -> dict[str, Any]:
    """Ingest one document shape and ask its questions."""
    print(f"\n=== {shape['name']} ({shape['description']}) ===", flush=True)

    if not shape["document"].exists():
        print(f"    SKIPPED: {shape['document']} is not available", flush=True)

        return {
            "name": shape["name"],
            "description": shape["description"],
            "skipped": True,
            "reason": "document not present",
            "results": [],
        }

    ingest(shape["document"], chunk_size)

    clear_caches()

    pipeline = RAGPipeline()

    results = []

    for entry in shape["questions"]:
        result = pipeline.ask(entry["question"])

        retrieval = result.get("retrieval") or {}

        answer = result["answer"]
        abstained = bool(result.get("abstained"))

        answerable = entry.get("answerable", True)

        if answerable:
            correct = not abstained and all(
                keyword.lower() in answer.lower() for keyword in entry["expect"]
            )
        else:
            correct = abstained

        row = {
            "question": entry["question"],
            "answer": answer.strip().replace("\n", " "),
            "answerable": answerable,
            "abstained": abstained,
            "correct": correct,
            "retrieved_chunks": retrieval.get("retrieved_chunks", 0),
            "considered_candidates": retrieval.get("considered_candidates", 0),
            "best_distance": retrieval.get("best_distance"),
            "profile_used": retrieval.get("profile_used", False),
        }

        results.append(row)

        mark = "ok  " if correct else "MISS"

        distance = row["best_distance"]

        print(
            f"  {mark} chunks={row['retrieved_chunks']:>3} "
            f"best={distance if distance is None else round(distance, 3)} "
            f"profile={row['profile_used']} | {entry['question'][:58]}",
            flush=True,
        )

    return {
        "name": shape["name"],
        "description": shape["description"],
        "skipped": False,
        "results": results,
    }


def summarise(shape: dict[str, Any]) -> dict[str, Any]:
    """Per-document summary row."""
    results = shape["results"]

    if not results:
        return {
            "document": shape["name"],
            "description": shape["description"],
            "questions": 0,
            "answered": "n/a",
            "empty_context": "n/a",
            "min_chunks": "n/a",
        }

    answerable = [row for row in results if row["answerable"]]

    empty = [row for row in results if row["retrieved_chunks"] == 0]

    return {
        "document": shape["name"],
        "description": shape["description"],
        "questions": len(results),
        "answered": (
            f"{sum(1 for row in answerable if row['correct'])}/{len(answerable)}"
            if answerable
            else "n/a"
        ),
        "abstained_correctly": (
            f"{sum(1 for row in results if not row['answerable'] and row['correct'])}"
            f"/{sum(1 for row in results if not row['answerable'])}"
        ),
        "empty_context": len(empty),
        "min_chunks": min(row["retrieved_chunks"] for row in results),
    }


def to_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Document | Shape | Questions | Answered | Abstained correctly | "
        "Questions with no passages | Fewest passages |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )

    body = []

    for row in rows:
        body.append(
            f"| `{row['document']}` | {row['description']} | {row['questions']} "
            f"| {row['answered']} | {row.get('abstained_correctly', 'n/a')} "
            f"| {row['empty_context']} | {row['min_chunks']} |"
        )

    return header + "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate across document shapes.")

    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--shapes", type=Path, default=SHAPES_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)

    args = parser.parse_args()

    shapes = load_shapes(args.shapes)

    if args.only:
        shapes = [shape for shape in shapes if shape["name"] in args.only]

    runs = [run_shape(shape) for shape in shapes]

    rows = [summarise(run) for run in runs]

    print()
    print(to_markdown(rows))

    violations = [row for row in rows if row["empty_context"] not in ("n/a", 0)]

    if violations:
        print("NEVER-EMPTY INVARIANT VIOLATED:", flush=True)

        for row in violations:
            print(f"  {row['document']}: {row['empty_context']} questions", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(rows), encoding="utf-8")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
