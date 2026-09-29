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
