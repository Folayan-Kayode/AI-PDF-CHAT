"""Tests for the text splitter."""

from app.rag.splitter import TextSplitter


def test_chunks_keep_page_and_sequential_number():
    pages = [{"page": 7, "text": "Sentence one. " * 300}]

    chunks = TextSplitter().split_pages(pages)

    assert len(chunks) > 1
    assert all(chunk["page"] == 7 for chunk in chunks)
    assert [c["chunk"] for c in chunks] == list(range(1, len(chunks) + 1))


def test_empty_pages_produce_no_chunks():
    pages = [{"page": 1, "text": ""}, {"page": 2, "text": "   "}]

    assert TextSplitter().split_pages(pages) == []


def test_whitespace_only_chunks_are_dropped():
    pages = [{"page": 3, "text": "Real content here."}, {"page": 4, "text": ""}]

    chunks = TextSplitter().split_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 3
    assert chunks[0]["text"] == "Real content here."


def test_chunk_size_is_respected():
    pages = [{"page": 1, "text": "word " * 800}]

    chunks = TextSplitter(chunk_size=200, chunk_overlap=0).split_pages(pages)

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 200 for chunk in chunks)


def test_each_page_is_split_independently():
    pages = [
        {"page": 1, "text": "First page content."},
        {"page": 2, "text": "Second page content."},
    ]

    chunks = TextSplitter().split_pages(pages)

    assert [chunk["page"] for chunk in chunks] == [1, 2]


# --------------------------------------------------------------------------
# Whole-document splitting with page spans
# --------------------------------------------------------------------------


def test_document_is_split_as_one_text_with_page_spans():
    pages = [
        {"page": 1, "text": "First page content. " * 40},
        {"page": 2, "text": "Second page content. " * 40},
    ]

    chunks = TextSplitter(chunk_size=200, chunk_overlap=0).split_document(pages)

    assert chunks
    assert all("page_start" in chunk and "page_end" in chunk for chunk in chunks)
    assert all(chunk["page_start"] <= chunk["page_end"] for chunk in chunks)
    assert chunks[0]["page_start"] == 1


def test_a_sentence_is_not_cut_merely_because_a_page_ended():
    # A short page followed by a short page should not each become a chunk
    # when the pair fits comfortably in one.
    pages = [
        {"page": 1, "text": "The rule continues on the next page and"},
        {"page": 2, "text": "finishes here."},
    ]

    chunks = TextSplitter(chunk_size=1000, chunk_overlap=0).split_document(pages)

    assert len(chunks) == 1, "per-page splitting would have produced two"
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 2
    assert "finishes here." in chunks[0]["text"]


def test_a_chunk_that_straddles_pages_reports_both():
    pages = [
        {"page": 3, "text": "alpha " * 60},
        {"page": 4, "text": "beta " * 60},
        {"page": 5, "text": "gamma " * 60},
    ]

    # Big enough that the first two pages are merged into one chunk.
    chunks = TextSplitter(chunk_size=800, chunk_overlap=0).split_document(pages)

    straddling = [chunk for chunk in chunks if chunk["page_end"] > chunk["page_start"]]

    assert straddling, "a boundary-crossing chunk is the expected outcome"
    assert (straddling[0]["page_start"], straddling[0]["page_end"]) == (3, 4)
    assert "beta" in straddling[0]["text"]
    assert all(chunk["page"] == chunk["page_start"] for chunk in chunks)


def test_blank_pages_are_skipped_without_breaking_spans():
    pages = [
        {"page": 1, "text": ""},
        {"page": 2, "text": "Real content."},
        {"page": 3, "text": "   "},
    ]

    chunks = TextSplitter().split_document(pages)

    assert len(chunks) == 1
    assert chunks[0]["page_start"] == 2


def test_empty_document_produces_no_chunks():
    assert TextSplitter().split_document([]) == []
    assert TextSplitter().split_document([{"page": 1, "text": ""}]) == []
