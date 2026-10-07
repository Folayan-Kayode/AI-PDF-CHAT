"""
Tests for the shape runner's bookkeeping, without ingesting anything.

The first test is the point: every shape the runner will use must be present in
the repository, so a fresh clone does not silently run a smaller evaluation.
"""

from eval.run_shapes import load_shapes, merge_shapes, summarise, to_markdown


def test_the_committed_corpus_covers_all_five_shapes():
    names = {shape["name"] for shape in load_shapes()}

    assert names == {"sheet", "table", "german", "standard", "book"}


def test_every_committed_shape_document_exists():
    # This is what makes the evaluation reproducible: no absolute paths, no
    # missing documents. A missing one fails here rather than at run time.
    for shape in load_shapes():
        assert shape["document"].exists(), (
            f"{shape['name']} points at a missing document: {shape['document']}"
        )


def test_merge_overrides_by_name_and_keeps_the_rest():
    base = [{"name": "a", "x": 1}, {"name": "b", "x": 2}]
    overlay = [{"name": "a", "x": 9}, {"name": "c", "x": 3}]

    merged = {shape["name"]: shape for shape in merge_shapes(base, overlay)}

    assert merged["a"]["x"] == 9
    assert merged["b"]["x"] == 2
    assert merged["c"]["x"] == 3


def test_summarise_counts_an_empty_context_as_a_failure():
    run = {
        "name": "x",
        "description": "d",
        "results": [
            {
                "answerable": True,
                "correct": True,
                "retrieved_chunks": 0,
                "abstained": False,
                "cited_document_only": True,
            }
        ],
    }

    row = summarise(run)

    assert row["empty_context"] == 1
    assert row["context_empty_rate"] == 1.0


def test_markdown_renders_a_rate_of_not_applicable():
    rows = [
        {
            "document": "x",
            "description": "d",
            "questions": 0,
            "answered": "n/a",
            "abstained_correctly": "n/a",
            "empty_context": "n/a",
            "min_chunks": "n/a",
        }
    ]

    assert "n/a" in to_markdown(rows)
