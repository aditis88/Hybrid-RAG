"""
Gemini generation for Hybrid RAG.

Uses the official Google Gen AI SDK (``google-genai``), not the legacy
``google-generativeai`` package.

Gemini is used **only** after local retrieval + RRF + reranking.
It must never receive the full corpus — only a small top-K context.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from langchain_core.documents import Document

# Cost/latency-friendly default; override with GEMINI_MODEL in .env.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024

SYSTEM_INSTRUCTION = """You are a careful financial / banking assistant answering from retrieved evidence only.

Rules:
1. Answer ONLY using the supplied CONTEXT passages.
2. Do not invent facts, rates, dates, or policy decisions that are not in CONTEXT.
3. If the evidence is insufficient or ambiguous, say so clearly and do not guess.
4. Cite sources using the chunk ids and source labels provided (e.g. [FAQ::chunk_12] or [source=RBI Policy]).
5. Distinguish evidence (what the passages state) from any brief inference; mark inferences explicitly.
6. Keep answers concise and grounded. Prefer quoting or paraphrasing CONTEXT over speculation.
"""


def _gemini_api_key() -> str:
    """Read key from env. Never log this value."""
    return (
        (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")
        .strip()
    )


def format_context_blocks(documents: Sequence[Document], *, max_chars_per_chunk: int = 800) -> str:
    """Build a compact context string for Gemini (top chunks only)."""
    blocks: list[str] = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        chunk_id = meta.get("chunk_id", f"chunk_{i}")
        source = meta.get("source", "unknown")
        text = (doc.page_content or "").strip().replace("\n", " ")
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "…"
        score = meta.get("rerank_score")
        score_line = f" rerank_score={float(score):.4f}" if isinstance(score, (int, float)) else ""
        blocks.append(
            f"[{i}] id={chunk_id} source={source}{score_line}\n{text}"
        )
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


def build_user_prompt(query: str, documents: Sequence[Document]) -> str:
    context = format_context_blocks(documents)
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query.strip()}\n\n"
        "Answer using only CONTEXT. Cite chunk ids. "
        "If evidence is insufficient, say so."
    )


class GeminiGenerator:
    """Thin wrapper around ``google.genai.Client`` for grounded RAG answers."""

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        api_key: str | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        key = (api_key or _gemini_api_key()).strip()
        if not key:
            raise EnvironmentError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env for Gemini generation."
            )
        self.model = (
            model
            or (os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL)
        ).strip()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=key)
        self._types = types
        # Do not print or log the API key.
        print(f"LLM: Gemini — {self.model} (temp={self.temperature})")

    def generate(self, query: str, documents: Sequence[Document]) -> dict[str, Any]:
        """
        Generate a grounded answer from ``documents`` (already retrieved/reranked).

        Returns dict with ``answer``, ``model``, ``num_context_chunks``.
        """
        if not documents:
            return {
                "answer": (
                    "I do not have enough retrieved evidence to answer this question. "
                    "No relevant passages were found in the local index."
                ),
                "model": self.model,
                "num_context_chunks": 0,
                "sources": [],
            }

        user_prompt = build_user_prompt(query, documents)
        config = self._types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            text = (
                "The model returned an empty response. "
                "Try again or check Gemini API quotas / safety filters."
            )

        sources: list[dict[str, str]] = []
        for doc in documents:
            meta = doc.metadata or {}
            sources.append(
                {
                    "chunk_id": str(meta.get("chunk_id", "")),
                    "source": str(meta.get("source", "unknown")),
                    "preview": (doc.page_content or "").replace("\n", " ").strip()[:200],
                }
            )

        return {
            "answer": text,
            "model": self.model,
            "num_context_chunks": len(documents),
            "sources": sources,
        }


def build_gemini_generator(**kwargs: Any) -> GeminiGenerator:
    """Factory for GeminiGenerator."""
    return GeminiGenerator(**kwargs)
