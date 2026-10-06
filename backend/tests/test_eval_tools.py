"""
Tests for the evaluation tooling: output versioning and the diagnoser.

These are offline: the diagnoser's retriever is replaced with a fake, so the
test asserts the tool's contract (it exits non-zero on an empty context)
without touching a provider or an index.
"""

import json

from app.core.config import settings
from eval import diagnose
from eval.run_eval import (
    RESULTS_SCHEMA_VERSION,
    build_envelope,
    document_identity,
    load_runs,
)

# --------------------------------------------------------------------------
# Output versioning
# --------------------------------------------------------------------------


def test_document_identity_records_a_hash_and_size(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 hello")

    identity = document_identity(path)

    assert identity["exists"] is True
    assert identity["name"] == "doc.pdf"
    assert identity["bytes"] == len(b"%PDF-1.4 hello")
    assert len(identity["sha256_16"]) == 16


def test_document_identity_does_not_pretend_a_missing_file_exists(tmp_path):
    identity = document_identity(tmp_path / "nope.pdf")

    assert identity["exists"] is False


def test_build_envelope_carries_the_provenance_a_reader_needs(tmp_path):
    document = tmp_path / "Whitman.pdf"
    document.write_bytes(b"x")

    runs = [{"config": "baseline", "summary": {}, "results": [{}, {}]}]

    envelope = build_envelope(runs, document, tmp_path / "questions.jsonl")

    assert envelope["schema_version"] == RESULTS_SCHEMA_VERSION
    assert envelope["kind"] == "ablation"
    assert envelope["document"]["name"] == "Whitman.pdf"
    assert envelope["questions"]["count"] == 2
    assert envelope["limits"]["retrieval_max_distance"] == settings.RETRIEVAL_MAX_DISTANCE
    assert envelope["limits"]["embedding_schema_version"] == settings.EMBEDDING_SCHEMA_VERSION
    assert envelope["generated_at"]
    assert envelope["runs"] == runs


def test_load_runs_reads_a_bare_list_and_an_envelope(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"a": 1}]), encoding="utf-8")

    enveloped = tmp_path / "enveloped.json"
    enveloped.write_text(json.dumps({"runs": [{"b": 2}]}), encoding="utf-8")

    assert load_runs(bare) == [{"a": 1}]
    assert load_runs(enveloped) == [{"b": 2}]


# --------------------------------------------------------------------------
# The diagnoser
# --------------------------------------------------------------------------


def _result(**overrides):
    base = {
        "documents": ["a passage"],
        "metadata": [{"page": 1}],
        "distances": [0.2],
        "considered_candidates": 4,
        "best_distance": 0.2,
        "cut_distance": 0.3,
        "candidate_distances": [0.2, 0.25, 0.9, 1.1],
    }

    base.update(overrides)

    return base


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def index_status(self):
        return {"count": 4}

    def retrieve(self, question, n_results=None):
        self.calls.append((question, n_results))

        return self.result


def test_describe_reports_the_threshold_and_the_context_size():
    row = diagnose.describe(
        FakeRetriever(_result(documents=["abcd", "ef"])),
        "what is this about?",
        None,
    )

    assert row["passages_supplied"] == 2
    assert row["context_chars"] == 6
    assert row["best_distance"] == 0.2
    assert row["cut_distance"] == 0.3
    assert row["candidate_distances"] == [0.2, 0.25, 0.9, 1.1]


def test_diagnose_exits_nonzero_when_a_question_produces_no_passages(
    monkeypatch,
    capsys,
):
    class EmptyRetriever:
        def __init__(self, document_id=None):
            pass

        def index_status(self):
            return {"count": 5}

        def retrieve(self, question, n_results=None):
            return _result(documents=[], metadata=[], distances=[])

    monkeypatch.setattr(diagnose, "Retriever", EmptyRetriever)
    monkeypatch.setattr("sys.argv", ["diagnose", "--question", "anything"])

    assert diagnose.main() == 1

    assert "FAIL: zero passages" in capsys.readouterr().out


def test_diagnose_exits_zero_when_every_question_produces_passages(
    monkeypatch,
    capsys,
):
    class WorkingRetriever:
        def __init__(self, document_id=None):
            pass

        def index_status(self):
            return {"count": 5}

        def retrieve(self, question, n_results=None):
            return _result()

    monkeypatch.setattr(diagnose, "Retriever", WorkingRetriever)
    monkeypatch.setattr("sys.argv", ["diagnose", "--question", "anything"])

    assert diagnose.main() == 0
