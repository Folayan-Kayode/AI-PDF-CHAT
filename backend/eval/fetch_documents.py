"""
Build the reproducible long documents used by the shape evaluation.

The original evidence used a CEDIA standard and a Pearson/Cengage textbook.
Both are copyrighted and must not be committed, which meant `run_shapes.py` and
`reproduce_regression.py` pointed at absolute paths on one developer's machine
and could not be re-run by anyone else. This script builds permissively licensed
substitutes instead:

  - book_mobydick.pdf  : Herman Melville, *Moby Dick; Or, The Whale*, from
                         Project Gutenberg (public domain in the US). Rendered
                         WITH a PDF bookmark per chapter (the "book" shape).
  - standard_http.pdf  : RFC 9110, *HTTP Semantics* (IETF; freely
                         redistributable). Rendered WITHOUT bookmarks (the
                         "standard" shape, whose profile needs a model read of
                         the opening pages).

The generated PDFs are committed, so a fresh clone has every shape and the
evaluation reproduces. Run this only to regenerate them.

Uses ``reportlab``, which is deliberately NOT a project dependency: this is a
development tool, not part of the application. Install it separately:

    pip install reportlab
    python -m eval.fetch_documents

Source URLs, licence and a content hash are written to documents/README.md.
"""

import hashlib
import io
import re
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = Path(__file__).with_name("documents")

PAGE_WIDTH, PAGE_HEIGHT = A4

LEFT = 25 * mm
RIGHT = PAGE_WIDTH - 25 * mm
TOP = PAGE_HEIGHT - 25 * mm
BOTTOM = 25 * mm

FONT = "Helvetica"
FONT_SIZE = 9.5
LEADING = 12.5
WRAP_WIDTH = 104

USER_AGENT = "ai-pdf-chat-eval/1.0 (+https://github.com/Folayan-Kayode/AI-PDF-CHAT)"

SOURCES: dict[str, dict[str, Any]] = {
    "book_mobydick.pdf": {
        "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "title": "Moby Dick; Or, The Whale",
        "author": "Herman Melville",
        "licence": "Public domain in the United States (Project Gutenberg)",
        "outline": True,
        "heading": re.compile(r"^CHAPTER\s+\d+\.[^\n]*$"),
    },
    "standard_http.pdf": {
        "url": "https://www.rfc-editor.org/rfc/rfc9110.txt",
        "title": "RFC 9110: HTTP Semantics",
        "author": "Internet Engineering Task Force",
        "licence": "IETF Trust; freely redistributable (BCP 78 / RFC copyright)",
        "outline": False,
        "heading": None,
    },
}


def fetch(url: str) -> str:
    """Download a UTF-8 text document."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        raw = response.read()

    return raw.decode("utf-8", errors="replace")


def body_start(text: str, url: str) -> int:
    """
    Where the machine-readable body begins.

    Moby Dick carries a table of contents before the chapters, so the first
    "CHAPTER 1." is in the contents. Starting at "CHAPTER 1." after the
    front-matter markers keeps the outline pointing at real chapter pages.
    """
    if "gutenberg" not in url.lower():
        return 0

    marker = text.rfind("\nCHAPTER 1.")

    return marker if marker > 0 else 0


def render(text: str, config: dict[str, Any]) -> Path:
    """Render text to a PDF, optionally with one bookmark per heading."""
    path = OUT / config["filename"]

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4, pageCompression=1)

    pdf.setTitle(config["title"])
    pdf.setAuthor(config["author"])

    start = body_start(text, config["url"])
    heading = config["heading"]

    pdf.setFont(FONT, FONT_SIZE)

    y = TOP
    outline_index = 0

    def new_page() -> None:
        nonlocal y
        pdf.showPage()
        pdf.setFont(FONT, FONT_SIZE)
        y = TOP

    for raw_line in text[start:].splitlines():
        line = raw_line.rstrip()

        if config["outline"] and heading and heading.match(line.strip()):
            # Bookmark the page this chapter starts on. Unique keys because
            # the same title can appear more than once.
            outline_index += 1
            key = f"chapter-{outline_index}"
            pdf.bookmarkPage(key)
            pdf.addOutlineEntry(line.strip()[:80], key, level=0)

        wrapped = textwrap.wrap(line, width=WRAP_WIDTH) or [""]

        for piece in wrapped:
            if y <= BOTTOM:
                new_page()

            pdf.drawString(LEFT, y, piece)
            y -= LEADING

        # A blank source line becomes a small paragraph gap.
        if not line:
            y -= LEADING / 2

    pdf.showPage()
    pdf.save()

    path.write_bytes(buffer.getvalue())

    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    for filename, config in SOURCES.items():
        config = {**config, "filename": filename}

        print(f"fetching {config['url']} ...", flush=True)

        text = fetch(config["url"])

        print(f"  {len(text):,} characters; rendering {filename} ...", flush=True)

        path = render(text, config)

        rows.append(
            {
                "file": filename,
                "url": config["url"],
                "title": config["title"],
                "author": config["author"],
                "licence": config["licence"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

        print(f"  wrote {path} ({path.stat().st_size:,} bytes)", flush=True)

    readme = OUT / "README.md"
    readme.write_text(render_readme(rows), encoding="utf-8")

    print(f"wrote {readme}")


def render_readme(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Evaluation documents",
        "",
        "The documents `run_shapes.py` and `reproduce_regression.py` ingest, so",
        "the evidence can be re-run on a fresh clone. They are the *substitutes*",
        "for the original copyrighted CEDIA standard and Pearson/Cengage textbook,",
        "which are not redistributable and are never committed.",
        "",
        "Generated by `python -m eval.fetch_documents` (needs `reportlab`, which is",
        "not a project dependency). Do not edit by hand.",
        "",
        "| File | Source | Title | Author | Licence | sha256 (first 16) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f"| `{row['file']}` | {row['url']} | {row['title']} | {row['author']} "
            f"| {row['licence']} | `{row['sha256'][:16]}` |"
        )

    lines += [
        "",
        "The small synthetic documents (`sheet.pdf`, `table.pdf`, `german.pdf`) are",
        "generated by `eval/make_documents.py` and are original to this project.",
        "",
        "To evaluate against the original copyrighted documents instead, put them",
        "outside the repository and pass the paths explicitly (see",
        "`eval/shapes.local.jsonl.example` and the `--local` / `--document` flags).",
        "That is a local overlay, never committed.",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
