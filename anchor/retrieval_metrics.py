"""
Deterministic retrieval metrics. No model calls, no randomness — same
input always produces the same number, so these are safe to run in CI
as a regression gate.

All functions take:
    retrieved_ids: List[str]   — chunk ids in ranked order, best first
    relevant_ids:  Set[str]    — the ground-truth relevant chunk ids
    k:             int         — cutoff (ignored by MRR / MAP, which use
                                  the full ranked list)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

from .schema import RetrievalMetrics


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def mrr(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
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
        mrr=mrr(retrieved_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k),
        map_score=map_score(retrieved_ids, relevant_ids),
        k=k,
    )
