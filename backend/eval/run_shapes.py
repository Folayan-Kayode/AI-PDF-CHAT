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
    python -m eval.run_shapes --local shapes.local.jsonl     # optional overlay

A shape whose document is missing is reported and, by default, makes the run
exit non-zero: a quietly smaller run is how a reproducibility gap survives.
Pass --allow-missing to accept a partial run deliberately.

The committed shapes use permissively licensed documents (see
documents/README.md). An optional gitignored overlay (--local) can point at
documents that must not be committed.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.rag.pipeline import RAGPipeline
from eval.run_eval import clear_caches, git_revision, ingest, run_limits

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


def merge_shapes(
    base: list[dict[str, Any]],
    overlay: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay shapes by name, so a local file can replace a committed one."""
    merged = {shape["name"]: shape for shape in base}

    for shape in overlay:
        merged[shape["name"]] = shape

    return list(merged.values())


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
            "path": str(shape["document"]),
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
            "cut_distance": retrieval.get("cut_distance"),
            "profile_used": retrieval.get("profile_used", False),
            # The two fields the blindness hid: how much context was actually
            # supplied, and whether the answer came only from the profile.
            "passages_supplied": retrieval.get("retrieved_chunks", 0),
            "candidates_returned": retrieval.get("considered_candidates", 0),
            "profile_supplied": retrieval.get("profile_used", False),
            "cited_document_only": bool((result.get("citations") or {}).get("document_cited"))
            and not (result.get("citations") or {}).get("cited_pages"),
            "candidate_distances": retrieval.get("candidate_distances") or [],
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
    from eval.metrics import context_empty_rate, profile_only_rate

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
        "context_empty_rate": context_empty_rate(results),
        "profile_only_rate": profile_only_rate(results),
    }


def to_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Document | Shape | Questions | Answered | Abstained correctly | "
        "Empty ctx rate | Profile-only rate | Fewest passages |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )

    body = []

    for row in rows:
        empty_rate = row.get("context_empty_rate", "n/a")
        profile_rate = row.get("profile_only_rate", "n/a")

        body.append(
            f"| `{row['document']}` | {row['description']} | {row['questions']} "
            f"| {row['answered']} | {row.get('abstained_correctly', 'n/a')} "
            f"| {empty_rate if empty_rate == 'n/a' else f'{empty_rate:.2f}'} "
            f"| {profile_rate if profile_rate == 'n/a' else f'{profile_rate:.2f}'} "
            f"| {row['min_chunks']} |"
        )

    return header + "\n".join(body) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate across document shapes.")

    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--shapes", type=Path, default=SHAPES_PATH)
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help=(
            "Optional gitignored overlay of shapes (matched by name). Use it for "
            "documents that must not be committed."
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

    shapes = load_shapes(args.shapes)

    if args.local:
        if not args.local.exists():
            parser.error(f"--local file not found: {args.local}")

        shapes = merge_shapes(shapes, load_shapes(args.local))

    if args.only:
        shapes = [shape for shape in shapes if shape["name"] in args.only]

    runs = [run_shape(shape) for shape in shapes]

    rows = [summarise(run) for run in runs]

    print()
    print(to_markdown(rows))

    skipped = [run for run in runs if run.get("skipped")]

    print(f"{len(runs) - len(skipped)} of {len(runs)} shapes ran.", flush=True)

    for run in skipped:
        print(f"  MISSING: {run['name']} ({run.get('path')})", flush=True)

    violations = [row for row in rows if row["empty_context"] not in ("n/a", 0)]

    if violations:
        print("NEVER-EMPTY INVARIANT VIOLATED:", flush=True)

        for row in violations:
            print(f"  {row['document']}: {row['empty_context']} questions", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "shapes",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "revision": git_revision(),
                    "documents": [run["name"] for run in runs],
                    "shapes_ran": len(runs) - len(skipped),
                    "shapes_total": len(runs),
                    "missing": [run["name"] for run in skipped],
                    "limits": run_limits(),
                    "runs": runs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(rows), encoding="utf-8")
        print(f"wrote {args.markdown}")

    if skipped and not args.allow_missing:
        print(
            "Refusing to report a quietly smaller run. Provide the missing "
            "documents, or pass --allow-missing to accept the partial run.",
            flush=True,
        )

        return 1

    if violations:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
