"""
Metric computations for the evaluation harness.

Everything here is a pure function over plain data so it can be unit tested
without a model, a network, or an API key. The harness in run_eval.py is the
only thing that talks to providers.
"""

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Any

# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------

#: USD per million tokens. Approximate list prices, kept here so a result can
#: be recomputed when pricing changes. Costs reported by the harness are
#: estimates derived from characters, not billed amounts.
_CHAT_PRICE = (0.28, 0.42)  # DeepSeek chat: input, output

DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "generate": _CHAT_PRICE,
    "rewrite": _CHAT_PRICE,
    "rerank": _CHAT_PRICE,
    "embedding": (0.15, 0.0),  # Gemini embeddings: input, output
}

#: Rough English characters per token. Used only for cost/latency estimates.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a string."""
    if not text:
        return 0

    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class CallCounter:
    """Counts provider calls and the characters they consumed."""

    calls: Counter = field(default_factory=Counter)

    input_chars: Counter = field(default_factory=Counter)

    output_chars: Counter = field(default_factory=Counter)

    def record(self, kind: str, input_text: str = "", output_text: str = "") -> None:
        self.calls[kind] += 1
        self.input_chars[kind] += len(input_text or "")
        self.output_chars[kind] += len(output_text or "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "input_chars": dict(self.input_chars),
            "output_chars": dict(self.output_chars),
            "total_calls": sum(self.calls.values()),
        }


def estimate_cost(
    counter: dict[str, Any],
    prices: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Estimate the USD cost of the calls a question made."""
    prices = prices or DEFAULT_PRICES

    total = 0.0

    for kind, (input_price, output_price) in prices.items():
        input_chars = (counter.get("input_chars") or {}).get(kind, 0)
        output_chars = (counter.get("output_chars") or {}).get(kind, 0)

        total += estimate_tokens("x" * input_chars) / 1_000_000 * input_price
        total += estimate_tokens("x" * output_chars) / 1_000_000 * output_price

    return total


# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------


def hit_at_k(
    retrieved_pages: list[int],
    expected_pages: list[int],
    k: int = 5,
) -> bool:
    """Whether a gold page appears in the first k retrieved pages."""
    if not expected_pages:
        return False

    return bool(set(retrieved_pages[:k]) & set(expected_pages))


def reciprocal_rank(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    """
    Reciprocal of the rank of the first gold page (0 when none was retrieved).

    Rewards putting the right page first rather than merely somewhere in the
    top k.
    """
    if not expected_pages:
        return 0.0

    wanted = set(expected_pages)

    for index, page in enumerate(retrieved_pages, start=1):
        if page in wanted:
            return 1.0 / index

    return 0.0


# --------------------------------------------------------------------------
# Answer metrics
# --------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant comparisons."""
    return " ".join((text or "").lower().split())


def answer_matches(answer: str, expected_keywords: list[str]) -> bool:
    """
    Whether the answer contains every expected keyword.

    Keyword matching is deliberate: it is deterministic, so an ablation
    comparison cannot be skewed by the variance of a model judging itself.
    """
    if not expected_keywords:
        return False

    haystack = normalise(answer)

    return all(normalise(keyword) in haystack for keyword in expected_keywords)


def abstention_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Abstention precision and recall, treating "abstained" as the positive class.

    Correct abstention on an unanswerable question is a true positive; a
    refusal on an answerable question is a false positive, which is the
    failure mode users notice most.
    """
    true_positive = false_positive = false_negative = 0

    for result in results:
        abstained = bool(result.get("abstained"))

        if result.get("answerable"):
            if abstained:
                false_positive += 1
        elif abstained:
            true_positive += 1
        else:
            false_negative += 1

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    return {
        "abstention_true_positives": true_positive,
        "abstention_false_positives": false_positive,
        "abstention_false_negatives": false_negative,
        "abstention_precision": (
            true_positive / precision_denominator if precision_denominator else 0.0
        ),
        "abstention_recall": (true_positive / recall_denominator if recall_denominator else 0.0),
    }


def context_empty_rate(results: list[dict[str, Any]]) -> float:
    """
    Share of answered questions where zero passages reached the prompt.

    The primary regression signal. Zero passages means the answer was generated
    from the document profile alone, which reads as confidently grounded
    because nothing in the answer says otherwise. A 0.60 absolute distance
    threshold scored well on hit@5, accuracy and cost while this was 100% on
    broad questions, which is why it now has a name and a column.
    """
    answered = [result for result in results if not result.get("abstained")]

    if not answered:
        return 0.0

    empty = sum(1 for result in answered if not result.get("passages_supplied"))

    return empty / len(answered)


def profile_only_rate(results: list[dict[str, Any]]) -> float:
    """
    Share of answered questions citing nothing but [document].

    The same failure seen from the answer's side rather than retrieval's.
    """
    answered = [result for result in results if not result.get("abstained")]

    if not answered:
        return 0.0

    profile_only = sum(1 for result in answered if result.get("cited_document_only"))

    return profile_only / len(answered)


def citation_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    How often answers cite, and how often those citations check out.

    An uncited answer is not counted as invalid -- it is counted as uncited,
    because the two problems need different fixes.
    """
    answered = [result for result in results if not result.get("abstained")]

    cited = [result for result in answered if (result.get("citations") or {}).get("has_citation")]

    valid = [result for result in cited if (result.get("citations") or {}).get("all_valid")]

    invalid_citations = sum(
        len((result.get("citations") or {}).get("invalid_pages") or []) for result in results
    )

    return {
        "answered": len(answered),
        "cited_answers": len(cited),
        "valid_cited_answers": len(valid),
        "invalid_citation_count": invalid_citations,
        "citation_rate": len(cited) / len(answered) if answered else 0.0,
        "citation_validity": len(valid) / len(cited) if cited else 0.0,
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def summarise(results: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Aggregate per-question results into one row of the results table."""
    if not results:
        return {}

    answerable = [result for result in results if result.get("answerable")]

    # Metadata questions are answered from the document profile rather than
    # from retrieved passages, so they carry no gold page and would distort a
    # retrieval metric if they were counted as misses.
    retrievable = [result for result in answerable if result.get("expected_pages")]

    latencies = [result.get("latency_seconds", 0.0) for result in results]

    costs = [result.get("cost_usd", 0.0) for result in results]

    calls: Counter = Counter()

    for result in results:
        calls.update((result.get("calls") or {}).get("calls") or {})

    summary: dict[str, Any] = {
        "questions": len(results),
        "answerable_questions": len(answerable),
        "retrieval_questions": len(retrievable),
        "retrieval_hit_rate": (
            sum(
                hit_at_k(result["retrieved_pages"], result["expected_pages"], k)
                for result in retrievable
            )
            / len(retrievable)
            if retrievable
            else 0.0
        ),
        "mrr": (
            sum(
                reciprocal_rank(result["retrieved_pages"], result["expected_pages"])
                for result in retrievable
            )
            / len(retrievable)
            if retrievable
            else 0.0
        ),
        "answer_accuracy": (
            sum(1 for result in answerable if result.get("correct")) / len(answerable)
            if answerable
            else 0.0
        ),
        "latency_p50_seconds": median(latencies) if latencies else 0.0,
        "latency_total_seconds": sum(latencies),
        "cost_per_question_usd": sum(costs) / len(results),
        "cost_total_usd": sum(costs),
        "calls_per_question": sum(calls.values()) / len(results),
        "calls_by_kind": dict(calls),
    }

    summary.update(abstention_stats(results))
    summary.update(citation_stats(results))

    summary["context_empty_rate"] = context_empty_rate(results)
    summary["profile_only_rate"] = profile_only_rate(results)

    return summary
