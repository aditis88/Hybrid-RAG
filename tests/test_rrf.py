"""Unit tests for explicit Reciprocal Rank Fusion (no models / network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document

from retrieval.rrf import reciprocal_rank_fusion


def _doc(text: str, chunk_id: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


def test_rrf_prefers_docs_ranked_high_in_both_lists():
    a = _doc("alpha", "a")
    b = _doc("beta", "b")
    c = _doc("gamma", "c")

    # a is #1 in both → should lead; c only in one list → lower
    bm25 = [a, b, c]
    dense = [a, c, b]
    fused = reciprocal_rank_fusion([bm25, dense], k=60, id_key="chunk_id")
    assert [d.metadata["chunk_id"] for d in fused][0] == "a"


def test_rrf_top_n_and_weights():
    a = _doc("alpha", "a")
    b = _doc("beta", "b")
    fused = reciprocal_rank_fusion(
        [[a, b], [b, a]],
        k=60,
        weights=[1.0, 0.0],
        id_key="chunk_id",
        top_n=1,
    )
    assert len(fused) == 1
    assert fused[0].metadata["chunk_id"] == "a"


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


if __name__ == "__main__":
    test_rrf_prefers_docs_ranked_high_in_both_lists()
    test_rrf_top_n_and_weights()
    test_rrf_empty()
    print("test_rrf: OK")
