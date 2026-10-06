"""
Text chunking.

Splitting is done across the whole document rather than per page, because a
page boundary is a layout artefact, not a semantic one: per-page splitting cuts
sentences in half, tears tables that span two pages, and turns a slide-style
PDF with ten words per page into degenerate chunks. Each chunk records the page
span it covers.
"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

PAGE_SEPARATOR = "\n\n"


class TextSplitter:
    """Splits document text into overlapping chunks with page spans."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split_document(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Split the document as one text, recording each chunk's page span.

        Chunks never start or end merely because a page did, and a chunk that
        straddles a boundary reports both pages.
        """
        combined, spans = _combine(pages)

        if not combined.strip():
            return []

        chunks: list[dict[str, Any]] = []

        cursor = 0

        for index, chunk in enumerate(self.splitter.split_text(combined)):
            text = chunk.strip()

            if not text:
                continue

            start = _locate(combined, text, cursor, self.chunk_overlap)

            end = start + len(text)

            cursor = end

            page_start, page_end = _pages_for(spans, start, end)

            chunks.append(
                {
                    "text": text,
                    # `page` is the citation anchor and is kept for
                    # compatibility with code written before spans existed.
                    "page": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk": index + 1,
                }
            )

        return chunks

    def split_pages(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Split each page on its own, for callers that want page-local chunks."""
        chunks: list[dict[str, Any]] = []

        for page in pages:
            split = self.splitter.split_text(page["text"])

            for index, chunk in enumerate(split):
                text = chunk.strip()

                if not text:
                    continue

                chunks.append(
                    {
                        "text": text,
                        "page": page["page"],
                        "page_start": page["page"],
                        "page_end": page["page"],
                        "chunk": index + 1,
                    }
                )

        return chunks


def _combine(pages: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join page texts, returning the text and each page's character offsets."""
    parts: list[str] = []

    spans: list[tuple[int, int, int]] = []

    offset = 0

    for page in pages:
        text = (page.get("text") or "").strip()

        if not text:
            continue

        if parts:
            offset += len(PAGE_SEPARATOR)

        spans.append((offset, offset + len(text), page["page"]))

        parts.append(text)

        offset += len(text)

    return PAGE_SEPARATOR.join(parts), spans


def _locate(combined: str, chunk: str, cursor: int, overlap: int) -> int:
    """Find a chunk's offset, tolerating the overlap with the previous one."""
    start = combined.find(chunk, max(0, cursor - overlap))

    if start < 0:
        start = combined.find(chunk)

    if start < 0:
        # Should not happen -- the splitter returns substrings -- but a wrong
        # offset beats a crash.
        return max(0, cursor - overlap)

    return start


def _pages_for(
    spans: list[tuple[int, int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    """The first and last page a character range covers."""
    if not spans:
        return 1, 1

    overlapping = [
        page for (span_start, span_end, page) in spans if span_start < end and span_end > start
    ]

    if overlapping:
        return overlapping[0], overlapping[-1]

    preceding = [page for (span_start, _, page) in spans if span_start <= start]

    page = preceding[-1] if preceding else spans[0][2]

    return page, page
