"""
Local cross-encoder reranker.

Scores (query, document) pairs and returns the top-K passages.
Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — lightweight (~22M
params), MS MARCO–trained for passage ranking, suitable for a personal laptop.
"""

from __future__ import annotations

import os
from typing import Sequence

from langchain_core.documents import Document

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Rerank candidate chunks with a local SentenceTransformers CrossEncoder.

    Flow:
      1. Build (query, chunk_text) pairs
      2. Score each pair
      3. Sort by score descending
      4. Return top ``top_k`` documents (scores stored in metadata under
         ``rerank_score`` when ``attach_scores`` is True)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        kwargs: dict = {"max_length": max_length}
        if device:
            kwargs["device"] = device
        self._model = CrossEncoder(model_name, **kwargs)

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int = 5,
        attach_scores: bool = True,
    ) -> list[Document]:
        if not documents:
            return []
        if top_k <= 0:
            return []

        pairs = [(query, (d.page_content or "")) for d in documents]
        scores = self._model.predict(pairs)
        # predict may return numpy array
        scored = list(zip(documents, scores))
        scored.sort(key=lambda item: float(item[1]), reverse=True)

        out: list[Document] = []
        for doc, score in scored[:top_k]:
            if attach_scores:
                meta = dict(doc.metadata or {})
                meta["rerank_score"] = float(score)
                out.append(Document(page_content=doc.page_content, metadata=meta))
            else:
                out.append(doc)
        return out


def build_cross_encoder_reranker(
    model_name: str | None = None,
) -> CrossEncoderReranker:
    """Factory: model from arg, ``CROSS_ENCODER_MODEL`` env, or default MiniLM-L-6."""
    name = (
        model_name
        or os.environ.get("CROSS_ENCODER_MODEL")
        or DEFAULT_CROSS_ENCODER_MODEL
    ).strip()
    print(f"Reranker: local cross-encoder — {name}")
    return CrossEncoderReranker(model_name=name)
