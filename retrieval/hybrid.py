"""
Hybrid retrieval: BM25 + dense → explicit RRF → optional cross-encoder rerank.

This is the interview-facing pipeline. The older LangChain ``EnsembleRetriever``
path remains available for weight-sweep demos in ``hybrid_rag_financial.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document

from retrieval.rrf import reciprocal_rank_fusion


@dataclass
class HybridRRFRetriever:
    """
    Retrieve with BM25 and dense retrievers, fuse with RRF, optionally rerank.

    Args:
        bm25_retriever: Retriever with a ``.invoke(query)`` / ``get_relevant_documents``.
        dense_retriever: Same interface (e.g. Chroma ``as_retriever``).
        rrf_k: RRF constant (default 60).
        rrf_top_n: How many fused candidates to keep before reranking (20–30).
        reranker: Object with ``rerank(query, docs, top_k=...)`` or None.
        rerank_top_k: Final number of chunks after reranking (default 5).
        rrf_weights: Optional [bm25_weight, dense_weight] for weighted RRF.
    """

    bm25_retriever: Any
    dense_retriever: Any
    rrf_k: int = 60
    rrf_top_n: int = 20
    reranker: Any | None = None
    rerank_top_k: int = 5
    rrf_weights: Sequence[float] | None = None
    id_key: str | None = "chunk_id"

    def _invoke_retriever(self, retriever: Any, query: str) -> list[Document]:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        else:
            raise TypeError(f"Unsupported retriever type: {type(retriever)}")
        return list(docs or [])

    def retrieve(
        self,
        query: str,
        *,
        use_reranker: bool | None = None,
    ) -> list[Document]:
        """
        Full hybrid path. If ``use_reranker`` is None, rerank when a reranker
        is attached; set False to return RRF-only candidates (for evaluation).
        """
        bm25_docs = self._invoke_retriever(self.bm25_retriever, query)
        dense_docs = self._invoke_retriever(self.dense_retriever, query)

        fused = reciprocal_rank_fusion(
            [bm25_docs, dense_docs],
            k=self.rrf_k,
            weights=self.rrf_weights,
            id_key=self.id_key,
            top_n=self.rrf_top_n,
        )

        should_rerank = (
            (use_reranker is True)
            or (use_reranker is None and self.reranker is not None)
        )
        if should_rerank and self.reranker is not None and fused:
            return self.reranker.rerank(
                query, fused, top_k=self.rerank_top_k
            )
        return fused

    def invoke(self, query: str) -> list[Document]:
        """LangChain-style alias used by simple callers."""
        return self.retrieve(query)


def build_hybrid_rrf_pipeline(
    bm25_retriever: Any,
    dense_retriever: Any,
    *,
    reranker: Any | None = None,
    rrf_k: int = 60,
    rrf_top_n: int = 20,
    rerank_top_k: int = 5,
    rrf_weights: Sequence[float] | None = None,
    id_key: str | None = "chunk_id",
) -> HybridRRFRetriever:
    """Construct the BM25 + dense → RRF → (rerank) pipeline."""
    return HybridRRFRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        rrf_k=rrf_k,
        rrf_top_n=rrf_top_n,
        reranker=reranker,
        rerank_top_k=rerank_top_k,
        rrf_weights=rrf_weights,
        id_key=id_key,
    )
