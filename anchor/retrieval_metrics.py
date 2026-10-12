"""
Deterministic retrieval metrics. No model calls, no randomness — same
input always produces the same number, so these are safe to run in CI
as a regression gate.

All functions take:
    retrieved_ids: List[str]   — chunk ids in ranked order, best first
    relevant_ids:  Set[str]    — the ground-truth relevant chunk ids
    k:             int         — cutoff (ignored by MRR / MAP, which use
                                  the full ranked list)

Repeated ids in a ranking (a retriever returning the same chunk twice) are
counted once, at their first position. Without that, a duplicate hit would
count twice and push nDCG or MAP above 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .schema import RetrievalMetrics


def _unique(ids: Sequence[str]) -> list[str]:
    seen = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    retrieved_ids = _unique(retrieved_ids)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    """Share of the top-k slots that hold a relevant chunk. The denominator
    is k, so returning fewer than k chunks is not rewarded."""
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    return len(set(_unique(retrieved_ids)[:k]) & relevant) / k


def hit_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    """1.0 if any relevant chunk is in the top k, else 0.0 (the mean is the "hit rate")."""
    relevant = set(relevant_ids)
    return 1.0 if set(_unique(retrieved_ids)[:k]) & relevant else 0.0


def mrr(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    retrieved_ids = _unique(retrieved_ids)
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    retrieved_ids = _unique(retrieved_ids)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    def dcg(ids: Sequence[str]) -> float:
        score = 0.0
        for i, doc_id in enumerate(ids[:k], start=1):
            rel = 1.0 if doc_id in relevant else 0.0
            score += rel / math.log2(i + 1)
        return score

    actual = dcg(retrieved_ids)
    ideal_ids = list(relevant) + [d for d in retrieved_ids if d not in relevant]
    ideal = dcg(ideal_ids)
    return actual / ideal if ideal > 0 else 0.0


def map_score(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    retrieved_ids = _unique(retrieved_ids)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant) if hits else 0.0


def all_retrieval_metrics(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5
) -> RetrievalMetrics:
    """Convenience wrapper used by the runner — computes every metric at once."""
    return RetrievalMetrics(
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        hit_at_k=hit_at_k(retrieved_ids, relevant_ids, k),
        mrr=mrr(retrieved_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k),
        map_score=map_score(retrieved_ids, relevant_ids),
        k=k,
    )
