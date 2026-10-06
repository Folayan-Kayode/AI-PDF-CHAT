"""
Parse and verify the citations an answer makes.

Answers are expected to cite the page a statement came from, as ``[p.N]``,
and to cite the document profile as ``[document]``. Verification matters
because a citation the model invented is worse than no citation at all: it
looks checkable while being wrong.
"""

import re
from collections.abc import Iterable
from typing import Any

PAGE_CITATION_PATTERN = re.compile(r"\[\s*p\.?\s*(\d+)\s*\]", re.IGNORECASE)

DOCUMENT_CITATION_PATTERN = re.compile(r"\[\s*document\s*\]", re.IGNORECASE)

#: Sentinel page value used by the document profile chunk, which is not a
#: real page of the document.
PROFILE_PAGE = 0


def cited_pages(answer: str) -> list[int]:
    """Page numbers cited in the answer, in order, without duplicates."""
    pages: list[int] = []

    for match in PAGE_CITATION_PATTERN.finditer(answer or ""):
        page = int(match.group(1))

        if page not in pages:
            pages.append(page)

    return pages


def cites_document_profile(answer: str) -> bool:
    """Whether the answer cites the document profile."""
    return bool(DOCUMENT_CITATION_PATTERN.search(answer or ""))


def verify_citations(
    answer: str,
    allowed_pages: Iterable[int],
    profile_available: bool = False,
) -> dict[str, Any]:
    """
    Check an answer's citations against the pages it was given.

    ``allowed_pages`` are the pages of the passages supplied to the model; a
    citation to any other page was invented. ``[document]`` is only valid when
    a document profile was actually supplied.
    """
    allowed = set(allowed_pages)

    pages = cited_pages(answer)
    invalid_pages = [page for page in pages if page not in allowed]

    document_cited = cites_document_profile(answer)
    document_invalid = document_cited and not profile_available

    has_citation = bool(pages) or document_cited

    return {
        "cited_pages": pages,
        "invalid_pages": invalid_pages,
        "document_cited": document_cited,
        "document_citation_invalid": document_invalid,
        "has_citation": has_citation,
        "all_valid": has_citation and not invalid_pages and not document_invalid,
    }
