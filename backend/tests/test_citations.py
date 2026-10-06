"""Tests for citation parsing and verification."""

import pytest

from app.rag.citations import (
    cited_pages,
    cites_document_profile,
    verify_citations,
)


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("The answer is 42 [p.12].", [12]),
        ("See [p.3] and [p.7].", [3, 7]),
        ("Repeated [p.3] and [p.3] again.", [3]),
        ("Also written [P.5] or [p 5] or [p.5].", [5]),
        ("No citation here.", []),
        ("", []),
        (None, []),
    ],
)
def test_cited_pages(answer, expected):
    assert cited_pages(answer) == expected


def test_cited_pages_keeps_order():
    assert cited_pages("[p.9] then [p.2] then [p.4]") == [9, 2, 4]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("From the profile [document].", True),
        ("From the profile [Document].", True),
        ("From the profile [ document ].", True),
        ("No profile citation.", False),
    ],
)
def test_cites_document_profile(answer, expected):
    assert cites_document_profile(answer) is expected


def test_valid_page_citations():
    report = verify_citations("The rate is 5 [p.3].", allowed_pages=[1, 3])

    assert report["cited_pages"] == [3]
    assert report["invalid_pages"] == []
    assert report["all_valid"] is True


def test_invented_page_citation_is_flagged():
    report = verify_citations("The rate is 5 [p.99].", allowed_pages=[1, 3])

    assert report["invalid_pages"] == [99]
    assert report["all_valid"] is False


def test_mixed_citations_report_only_the_bad_ones():
    report = verify_citations("A [p.1] and B [p.42].", allowed_pages=[1, 2])

    assert report["cited_pages"] == [1, 42]
    assert report["invalid_pages"] == [42]
    assert report["all_valid"] is False


def test_uncited_answer_is_not_valid():
    report = verify_citations("The rate is 5.", allowed_pages=[1, 3])

    assert report["has_citation"] is False
    assert report["all_valid"] is False


def test_document_citation_is_valid_when_a_profile_was_supplied():
    report = verify_citations(
        "The title is X [document].",
        allowed_pages=[5],
        profile_available=True,
    )

    assert report["document_cited"] is True
    assert report["document_citation_invalid"] is False
    assert report["all_valid"] is True


def test_document_citation_is_invalid_without_a_profile():
    report = verify_citations(
        "The title is X [document].",
        allowed_pages=[5],
        profile_available=False,
    )

    assert report["document_citation_invalid"] is True
    assert report["all_valid"] is False


def test_page_and_document_citations_together():
    report = verify_citations(
        "The title is X [document] and it covers security [p.22].",
        allowed_pages=[22],
        profile_available=True,
    )

    assert report["all_valid"] is True
    assert report["document_cited"] is True
    assert report["cited_pages"] == [22]
