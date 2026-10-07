"""
Explicit Reciprocal Rank Fusion (RRF).

Standard unweighted form (Cormack et al.):

    RRF_score(d) = Σ_r  1 / (k + rank_r(d))

where ``rank_r(d)`` is the 1-based rank of document ``d`` in ranked list ``r``,
and ``k`` (commonly 60) dampens the impact of very high ranks.

Optional per-list ``weights`` yield weighted RRF (same idea as LangChain's
``EnsembleRetriever.weighted_reciprocal_rank``):

    score(d) = Σ_r  w_r / (k + rank_r(d))

This module is intentionally small and dependency-light so it can be explained
in interviews and unit-tested without LangChain.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from langchain_core.documents import Document


def _doc_key(doc: Document, id_key: str | None) -> str:
    if id_key is not None:
        meta = doc.metadata or {}
        if id_key in meta:
            return str(meta[id_key])
    return (doc.page_content or "").strip()


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    id_key: str | None = None,
    top_n: int | None = None,
) -> list[Document]:
    """
    Fuse multiple ranked document lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: One ranked list per retriever (best first).
        k: RRF constant (default 60).
        weights: Optional weight per list (same length as ``ranked_lists``).
            If omitted, each list has weight 1.0 (classic RRF).
        id_key: Metadata key used to identify the same chunk across lists.
            If None, ``page_content`` is used (matches EnsembleRetriever default).
        top_n: If set, return only the top ``top_n`` fused documents.

    Returns:
        Documents sorted by descending RRF score (duplicates collapsed).
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
        )

    scores: dict[str, float] = defaultdict(float)
    first_doc: dict[str, Document] = {}

    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc, id_key)
            if not key:
                continue
            scores[key] += float(weight) / (k + rank)
            if key not in first_doc:
                first_doc[key] = doc

    ordered_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    fused = [first_doc[key] for key in ordered_keys]
    if top_n is not None:
        return fused[: max(0, top_n)]
    return fused
