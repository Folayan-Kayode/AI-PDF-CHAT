"""Tests for the evaluation metrics. Pure functions, no network."""

import pytest

from eval.metrics import (
    CallCounter,
    abstention_stats,
    answer_matches,
    citation_stats,
    context_empty_rate,
    estimate_cost,
    estimate_tokens,
    hit_at_k,
    normalise,
    profile_only_rate,
    reciprocal_rank,
    summarise,
)

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retrieved, expected, k, result",
    [
        ([5, 8, 9], [8], 5, True),
        ([5, 8, 9], [8], 1, False),
        ([5, 8, 9], [12], 5, False),
        ([], [8], 5, False),
        ([5, 8], [], 5, False),
    ],
)
def test_hit_at_k(retrieved, expected, k, result):
    assert hit_at_k(retrieved, expected, k) is result


@pytest.mark.parametrize(
    "retrieved, expected, score",
    [
        ([8, 5, 1], [8], 1.0),
        ([5, 8, 1], [8], 0.5),
        ([5, 1, 8], [8], 1 / 3),
        ([5, 1, 2], [8], 0.0),
        ([5, 8], [8, 9], 0.5),
        ([5], [], 0.0),
    ],
)
def test_reciprocal_rank(retrieved, expected, score):
    assert reciprocal_rank(retrieved, expected) == pytest.approx(score)


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


def test_normalise_collapses_case_and_whitespace():
    assert normalise("  The   TITLE\nis Here ") == "the title is here"


@pytest.mark.parametrize(
    "answer, keywords, matches",
    [
        (
            "The title is Principles of Information Security",
            ["principles of information security"],
            True,
        ),
        ("cengage learning published it", ["Cengage Learning"], True),
        ("Fourth edition, 2012", ["fourth edition", "2012"], True),
        ("Fourth edition", ["fourth edition", "2012"], False),
        ("anything", [], False),
    ],
)
def test_answer_matches(answer, keywords, matches):
    assert answer_matches(answer, keywords) is matches


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------


def _result(answerable, abstained):
    return {"answerable": answerable, "abstained": abstained}


def test_abstention_stats_count_all_four_outcomes():
    stats = abstention_stats(
        [
            _result(answerable=False, abstained=True),  # true positive
            _result(answerable=False, abstained=False),  # false negative
            _result(answerable=True, abstained=True),  # false positive
            _result(answerable=True, abstained=False),  # true negative
        ]
    )

    assert stats["abstention_true_positives"] == 1
    assert stats["abstention_false_negatives"] == 1
    assert stats["abstention_false_positives"] == 1
    assert stats["abstention_precision"] == pytest.approx(0.5)
    assert stats["abstention_recall"] == pytest.approx(0.5)


def test_abstention_stats_without_any_abstention():
    stats = abstention_stats([_result(answerable=True, abstained=False)])

    assert stats["abstention_precision"] == 0.0
    assert stats["abstention_recall"] == 0.0


def test_perfect_abstention():
    stats = abstention_stats(
        [
            _result(answerable=False, abstained=True),
            _result(answerable=True, abstained=False),
        ]
    )

    assert stats["abstention_precision"] == pytest.approx(1.0)
    assert stats["abstention_recall"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


def _cited(abstained=False, has_citation=True, all_valid=True, invalid_pages=None):
    return {
        "abstained": abstained,
        "citations": {
            "has_citation": has_citation,
            "all_valid": all_valid,
            "invalid_pages": invalid_pages or [],
        },
    }


def test_citation_stats_separate_uncited_from_invalid():
    stats = citation_stats(
        [
            _cited(),  # cited and valid
            _cited(has_citation=False, all_valid=False),  # uncited
            _cited(all_valid=False, invalid_pages=[99]),  # cited but wrong
            _cited(abstained=True),  # abstention is excluded
        ]
    )

    assert stats["answered"] == 3
    assert stats["cited_answers"] == 2
    assert stats["valid_cited_answers"] == 1
    assert stats["citation_rate"] == pytest.approx(2 / 3)
    assert stats["citation_validity"] == pytest.approx(0.5)
    assert stats["invalid_citation_count"] == 1


def test_citation_stats_with_no_answers():
    stats = citation_stats([])

    assert stats["citation_rate"] == 0.0
    assert stats["citation_validity"] == 0.0


# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------


def test_estimate_tokens_rounds_down_and_never_zero_for_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("a" * 400) == 100


def test_estimate_cost_uses_input_and_output_prices():
    counter = {
        "input_chars": {"generate": 4_000_000},
        "output_chars": {"generate": 1_000_000},
    }

    # 1M in * 0.28 + 0.25M out * 0.42
    assert estimate_cost(counter) == pytest.approx(0.28 + 0.105)


def test_estimate_cost_sums_across_call_kinds():
    counter = {
        "input_chars": {"generate": 4_000_000, "rewrite": 4_000_000},
        "output_chars": {"generate": 0, "rewrite": 0},
    }

    assert estimate_cost(counter) == pytest.approx(0.28 * 2)


def test_estimate_cost_without_calls_is_zero():
    assert estimate_cost({}) == 0.0


def test_call_counter_records_calls_and_characters():
    counter = CallCounter()

    counter.record("generate", "hello", "hi")
    counter.record("generate", "again")
    counter.record("embedding", "query")

    payload = counter.as_dict()

    assert payload["calls"] == {"generate": 2, "embedding": 1}
    assert payload["total_calls"] == 3
    assert payload["input_chars"]["generate"] == 10
    assert payload["output_chars"]["generate"] == 2


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _question(
    answerable=True,
    retrieved_pages=None,
    expected_pages=None,
    correct=True,
    abstained=False,
    citations=None,
    latency=1.0,
    cost=0.0,
    calls=None,
    passages_supplied=5,
    cited_document_only=False,
):
    return {
        "answerable": answerable,
        "retrieved_pages": retrieved_pages or [1, 2, 3, 4, 5],
        # An empty list is meaningful (a metadata question has no gold page),
        # so it must not fall back to the default.
        "expected_pages": [1] if expected_pages is None else expected_pages,
        "correct": correct,
        "abstained": abstained,
        "citations": citations or {"has_citation": True, "all_valid": True, "invalid_pages": []},
        "latency_seconds": latency,
        "cost_usd": cost,
        "calls": calls or {"calls": {"generate": 3}},
        "passages_supplied": passages_supplied,
        "cited_document_only": cited_document_only,
    }


# --------------------------------------------------------------------------
# Context emptiness: the regression signal
# --------------------------------------------------------------------------


def test_context_empty_rate_counts_answered_questions_with_no_passages():
    results = [
        _question(passages_supplied=5),
        _question(passages_supplied=0),
        _question(passages_supplied=0),
        _question(passages_supplied=0, abstained=True),
    ]

    # The abstention is excluded: no passages is expected there.
    assert context_empty_rate(results) == pytest.approx(2 / 3)


def test_context_empty_rate_is_zero_when_nothing_was_answered():
    assert context_empty_rate([_question(abstained=True)]) == 0.0
    assert context_empty_rate([]) == 0.0


def test_context_empty_rate_is_zero_when_every_answer_had_passages():
    assert context_empty_rate([_question(), _question()]) == 0.0


def test_profile_only_rate_counts_answers_citing_only_the_document():
    results = [
        _question(cited_document_only=False),
        _question(passages_supplied=0, cited_document_only=True),
        _question(passages_supplied=0, cited_document_only=True, abstained=True),
    ]

    assert profile_only_rate(results) == pytest.approx(0.5)


def test_profile_only_rate_is_zero_when_nothing_was_answered():
    assert profile_only_rate([]) == 0.0


def test_summarise_reports_both_emptiness_rates():
    results = [
        _question(),
        _question(passages_supplied=0, cited_document_only=True),
    ]

    summary = summarise(results)

    assert summary["context_empty_rate"] == pytest.approx(0.5)
    assert summary["profile_only_rate"] == pytest.approx(0.5)


def test_summarise_returns_empty_for_no_results():
    assert summarise([]) == {}


def test_summarise_aggregates_every_column():
    results = [
        _question(latency=1.0, cost=0.001, calls={"calls": {"generate": 3}}),
        _question(latency=3.0, cost=0.003, calls={"calls": {"generate": 1}}),
        _question(
            answerable=False,
            expected_pages=[],
            retrieved_pages=[9],
            abstained=True,
            correct=False,
            latency=2.0,
            cost=0.002,
            calls={"calls": {"generate": 2}},
        ),
    ]

    summary = summarise(results)

    assert summary["questions"] == 3
    assert summary["answerable_questions"] == 2
    assert summary["retrieval_questions"] == 2
    assert summary["retrieval_hit_rate"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx(1.0)
    assert summary["answer_accuracy"] == pytest.approx(1.0)
    assert summary["latency_p50_seconds"] == pytest.approx(2.0)
    assert summary["cost_total_usd"] == pytest.approx(0.006)
    assert summary["cost_per_question_usd"] == pytest.approx(0.002)
    assert summary["calls_per_question"] == pytest.approx(2.0)
    assert summary["calls_by_kind"] == {"generate": 6}
    assert summary["abstention_recall"] == pytest.approx(1.0)


def test_metadata_questions_are_excluded_from_retrieval_metrics():
    # A metadata question has no gold page, because the document profile
    # answers it. Counting it as a retrieval miss would understate retrieval.
    results = [
        _question(),
        _question(
            expected_pages=[],
            retrieved_pages=[1, 2],
            calls={"calls": {"generate": 1}},
        ),
    ]

    summary = summarise(results)

    assert summary["answerable_questions"] == 2
    assert summary["retrieval_questions"] == 1
    assert summary["retrieval_hit_rate"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx(1.0)


def test_summarise_counts_a_missed_retrieval():
    results = [_question(retrieved_pages=[7, 8], expected_pages=[1])]

    summary = summarise(results)

    assert summary["retrieval_hit_rate"] == 0.0
    assert summary["mrr"] == 0.0
