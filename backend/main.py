"""
FastAPI server for the Hybrid RAG demo. Serves JSON for a React (or any) UI.

Run (from project root, with venv active):
  uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

Then start the React app (see frontend/package.json scripts).
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# Project root must be on path so ``hybrid_rag_financial`` resolves; that module
# uses BASE_DIR next to the PDF and .env at repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import hybrid_rag_financial as rag


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    bm25_weight: float = Field(0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(0.5, ge=0.0, le=1.0)


class RAGAppState:
    """Holds loaded retrievers, LLM, and chains (hybrid rebuilt when weights change)."""

    def __init__(self) -> None:
        self.ready: bool = False
        self.error: Optional[str] = None
        self.chunks: list = []
        self.keyword_retriever = None
        self.vectorstore = None
        self.vectorstore_retriever = None
        self.llm = None
        self.bm25_chain = None
        self.vector_chain = None
        self.hybrid_chain = None
        self._hybrid_weights: tuple[float, float] = (-1.0, -1.0)

    def load(self, ensemble_weights: list[float]) -> None:
        all_docs = rag.build_all_documents()
        self.chunks = rag.chunk_documents(all_docs)
        self.keyword_retriever = rag.build_bm25_retriever(self.chunks)
        self.vectorstore = rag.build_vectorstore(self.chunks)
        self.vectorstore_retriever = rag.build_vectorstore_retriever(self.vectorstore)
        self.llm = rag.build_llm()
        self.bm25_chain = rag.build_retrieval_qa(self.llm, self.keyword_retriever)
        self.vector_chain = rag.build_retrieval_qa(self.llm, self.vectorstore_retriever)
        self._set_hybrid_weights(tuple(ensemble_weights))

    def _set_hybrid_weights(self, w: tuple[float, float]) -> None:
        if w == self._hybrid_weights and self.hybrid_chain is not None:
            return
        ensemble = rag.build_ensemble_retriever(
            self.keyword_retriever,
            self.vectorstore_retriever,
            weights=[w[0], w[1]],
        )
        self.hybrid_chain = rag.build_retrieval_qa(self.llm, ensemble)
        self._hybrid_weights = w


_state = RAGAppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    try:
        weights = list(rag.DEFAULT_ENSEMBLE_WEIGHTS)
        await asyncio.to_thread(_state.load, weights)
        _state.ready = True
    except Exception as e:
        _state.error = str(e)
        _state.ready = False
    yield


app = FastAPI(title="Hybrid RAG Financial API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ready": _state.ready,
        "error": _state.error,
        "chunks_indexed": len(_state.chunks) if _state.ready else 0,
    }


@app.post("/query")
def run_query(body: QueryBody) -> dict[str, Any]:
    if not _state.ready:
        raise HTTPException(
            status_code=503,
            detail=_state.error or "RAG pipeline not ready yet.",
        )
    w = (body.bm25_weight, body.vector_weight)
    if w[0] + w[1] <= 0:
        raise HTTPException(status_code=400, detail="BM25 and vector weights must sum to a positive value.")
    try:
        _state._set_hybrid_weights(w)
        out = rag.compare_three_results(
            body.query.strip(),
            _state.bm25_chain,
            _state.vector_chain,
            _state.hybrid_chain,
        )
        out["weights"] = {"bm25": w[0], "vector": w[1]}
        out["hybrid_insight"] = rag.hybrid_insight_for_results(
            out["query"],
            out["bm25"],
            out["vector"],
            out["hybrid"],
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
