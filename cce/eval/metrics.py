"""Phase 12 — Retrieval evaluation metrics: MRR@k, Recall@k, nDCG@k.

All functions are pure: (ranked_list, relevant_set, k) → float.
No external dependencies.
"""

from __future__ import annotations

import math


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant item; 0 if none found."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mrr_at_k(results: list[tuple[list[str], set[str]]], k: int) -> float:
    """Mean Reciprocal Rank @ k over a list of (ranked_list, relevant_set) pairs."""
    if not results:
        return 0.0
    total = sum(reciprocal_rank(ranked[:k], rel) for ranked, rel in results)
    return total / len(results)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant:
        return 1.0
    found = sum(1 for item in ranked[:k] if item in relevant)
    return found / len(relevant)


def mean_recall_at_k(results: list[tuple[list[str], set[str]]], k: int) -> float:
    """Mean Recall@k over a list of (ranked_list, relevant_set) pairs."""
    if not results:
        return 0.0
    return sum(recall_at_k(ranked, rel, k) for ranked, rel in results) / len(results)


def dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain @ k (binary relevance)."""
    return sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(ranked[:k])
        if item in relevant
    )


def ideal_dcg_at_k(relevant: set[str], k: int) -> float:
    """Ideal DCG: all relevant items at the top."""
    n = min(len(relevant), k)
    return sum(1.0 / math.log2(rank + 2) for rank in range(n))


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalised DCG @ k."""
    ideal = ideal_dcg_at_k(relevant, k)
    if ideal == 0:
        return 1.0
    return dcg_at_k(ranked, relevant, k) / ideal


def mean_ndcg_at_k(results: list[tuple[list[str], set[str]]], k: int) -> float:
    """Mean nDCG@k over a list of (ranked_list, relevant_set) pairs."""
    if not results:
        return 0.0
    return sum(ndcg_at_k(ranked, rel, k) for ranked, rel in results) / len(results)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / k
