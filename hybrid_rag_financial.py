"""
Hybrid RAG for financial / banking use case.

Demonstrates BM25 vs dense vector vs Ensemble (hybrid) retrieval with LangChain,
Chroma, HuggingFace Inference embeddings (BGE), and an LLM via **Anthropic Claude** (API),
**Hugging Face** router chat, or local **Zephyr 7B** 4-bit — see ``build_llm``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import textwrap
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic import Field

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Windows consoles often default to cp1252; RBI text includes ₹ and other Unicode.
def _configure_stdio_utf8() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()
RBI_POLICY_PDF = BASE_DIR / "Monetary Policy Report.pdf"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"
LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# When local 4-bit Zephyr is unavailable, HF router chat (see ``build_llm``).
HF_LLM_MODEL_DEFAULT = "meta-llama/Llama-3.2-1B-Instruct"
# Anthropic Messages API (https://docs.anthropic.com/). Override with CLAUDE_MODEL in .env.
CLAUDE_MODEL_DEFAULT = "claude-3-5-sonnet-20241022"
TOP_K = 3

# Hugging Face Inference API: one huge embed_documents() call returns an error JSON dict;
# Chroma then mis-indexes and raises KeyError. Batch instead (override with EMBEDDING_BATCH_SIZE).
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

# Ensemble default weights: [BM25, vector]
DEFAULT_ENSEMBLE_WEIGHTS: list[float] = [0.5, 0.5]


# ---------------------------------------------------------------------------
# 1. Document loading
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _disable_unstructured_pdf_complexity_skip():
    """
    Large RBI PDFs can be flagged as 'too complex' by unstructured, which skips pdfminer
    extraction and yields empty text under strategy='fast'. Forcing the check off keeps
    UnstructuredPDFLoader while still using the fast/pdfminer path.
    """
    import unstructured.partition.pdf as unstructured_pdf

    original = unstructured_pdf.is_pdf_too_complex
    unstructured_pdf.is_pdf_too_complex = lambda **kwargs: False
    try:
        yield
    finally:
        unstructured_pdf.is_pdf_too_complex = original


def load_rbi_policy_documents(pdf_path: Path) -> list[Document]:
    """Load RBI monetary policy PDF and tag each document with source metadata."""
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"RBI policy PDF not found at {pdf_path}. Place the RBI Monetary Policy PDF "
            "in the project folder or set RBI_POLICY_PDF."
        )
    with _disable_unstructured_pdf_complexity_skip():
        loader = UnstructuredPDFLoader(
            str(pdf_path),
            strategy="fast",
            languages=["eng"],
            mode="single",
        )
        docs = loader.load()
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = "RBI Policy"
    return docs


def load_faq_and_article_documents() -> list[Document]:
    """
    FAQ and article as explicit strings (per requirements).
    Representative banking / monetary-policy material for demos.
    """
    faq_items = [
        (
            "What is EMI? EMI (Equated Monthly Installment) is a fixed payment you make "
            "each month on a loan. It has principal and interest parts. When interest rates "
            "rise, the interest part grows, so EMI goes up unless the lender lengthens tenure."
        ),
        (
            "Why does my EMI increase? EMI increases when the lender raises your loan's "
            "interest rate—often after RBI increases the repo rate, which makes bank funding "
            "costlier. Higher rates mean more interest per rupee borrowed."
        ),
        (
            "How does inflation affect loans? Inflation often leads central banks to raise "
            "policy rates to cool demand. Higher policy rates usually pass through to lending "
            "rates, making new loans costlier and sometimes pushing floating EMIs higher."
        ),
        (
            "What is repo rate? Repo rate is the rate at which RBI lends short-term money to "
            "banks. It is a key signal for market interest rates and bank lending rates."
        ),
        (
            "What is CRR? Cash Reserve Ratio is the fraction of deposits banks must keep with "
            "RBI as cash. Raising CRR reduces lendable funds and can tighten liquidity."
        ),
    ]
    faq_docs = [
        Document(page_content=text, metadata={"source": "FAQ"}) for text in faq_items
    ]

    article_text = """
    Monetary transmission and household debt.

    When the Reserve Bank of India changes the repo rate, banks adjust their marginal cost
    of funds and often revise their external benchmark-linked lending rates. For floating-rate
    home loans, this can change the EMI or the loan tenure depending on contract terms.

    The cash reserve ratio (CRR) affects how much deposits banks can lend out. A higher CRR
    locks more reserves at the central bank, which can tighten liquidity in the banking
    system and influence short-term rates.

    Borrowers care about the pass-through from policy rates to retail loan rates. Delays or
    incomplete pass-through mean EMIs may not move immediately after a repo rate change, but
    over time, sustained policy tightening usually raises borrowing costs.
    """.strip()

    article_docs = [Document(page_content=article_text, metadata={"source": "Article"})]
    return faq_docs + article_docs


def build_all_documents() -> list[Document]:
    """Combine RBI PDF, FAQ strings, and article into one list."""
    return load_rbi_policy_documents(RBI_POLICY_PDF) + load_faq_and_article_documents()


# ---------------------------------------------------------------------------
# 2. Text chunking
# ---------------------------------------------------------------------------
def chunk_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(list(documents))


# ---------------------------------------------------------------------------
# 3. Embeddings & Chroma vector store
# ---------------------------------------------------------------------------
def make_embeddings() -> Embeddings:
    """
    Remote embeddings via Hugging Face Inference (router); BGE base English v1.5.

    Uses ``langchain_huggingface.HuggingFaceEndpointEmbeddings`` (``huggingface_hub``
    ``InferenceClient``), not the deprecated ``api-inference.huggingface.co`` HTTP helper.

    Set ``HUGGINGFACEHUB_API_TOKEN`` or ``HF_TOKEN`` in ``.env``.
    """
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) for HuggingFaceEndpointEmbeddings."
        )
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_ID,
        task="feature-extraction",
        huggingfacehub_api_token=token,
    )


def _prepare_chunks_for_chroma(chunks: list[Document]) -> list[Document]:
    """Drop empty strings; simplify metadata so Chroma accepts every field."""
    from langchain_community.vectorstores.utils import filter_complex_metadata

    non_empty = [
        Document(
            page_content=(d.page_content or "").strip(),
            metadata=dict(d.metadata or {}),
        )
        for d in chunks
        if (d.page_content or "").strip()
    ]
    return filter_complex_metadata(non_empty)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Index in small batches: HF Inference API rejects or errors on very large ``inputs`` arrays,
    and ``response.json()`` then becomes a dict, which breaks Chroma's embedding alignment.
    """
    embedding_fn = make_embeddings()
    prepared = _prepare_chunks_for_chroma(chunks)
    if not prepared:
        raise ValueError("No non-empty chunks to embed.")

    vs = Chroma(
        collection_name="financial_hybrid_rag",
        embedding_function=embedding_fn,
    )
    n = len(prepared)
    bs = max(1, EMBEDDING_BATCH_SIZE)
    for start in range(0, n, bs):
        batch = prepared[start : start + bs]
        print(f"  Embedding batch {start // bs + 1}/{(n + bs - 1) // bs} ({len(batch)} chunks)…")
        vs.add_documents(batch)
    return vs


def build_vectorstore_retriever(vectorstore: Chroma):
    """Dense retriever: top_k = 3."""
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ---------------------------------------------------------------------------
# 4. BM25 retriever
# ---------------------------------------------------------------------------
def build_bm25_retriever(chunks: list[Document]):
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = TOP_K
    return keyword_retriever


# ---------------------------------------------------------------------------
# 5. Hybrid (ensemble) retriever
# ---------------------------------------------------------------------------
def build_ensemble_retriever(
    bm25_retriever,
    vectorstore_retriever,
    weights: Sequence[float] | None = None,
):
    w = list(weights) if weights is not None else list(DEFAULT_ENSEMBLE_WEIGHTS)
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore_retriever],
        weights=w,
    )


# ---------------------------------------------------------------------------
# 6. LLM — local Zephyr 4-bit, or HF Inference chat fallback (router)
# ---------------------------------------------------------------------------
class HuggingFaceRouterChatLLM(LLM):
    """
    Calls ``huggingface_hub.InferenceClient.chat_completion`` (router), for providers
    that expose conversational models only (no legacy ``text_generation`` route).
    """

    model_id: str = Field(default=HF_LLM_MODEL_DEFAULT)
    max_new_tokens: int = 256
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "huggingface_router_chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN for HuggingFaceRouterChatLLM."
            )
        client = InferenceClient(model=self.model_id, token=token)
        out = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        text = ""
        if out.choices:
            msg = out.choices[0].message
            text = (getattr(msg, "content", None) or "") if msg is not None else ""
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s, 1)[0]
        return text


def _anthropic_api_key() -> str:
    """Anthropic accepts ``ANTHROPIC_API_KEY`` (official) or ``CLAUDE_API_KEY`` (common alias)."""
    return (
        (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or "")
        .strip()
    )


def _resolved_llm_backend() -> str:
    """``LLM_BACKEND`` overrides ``LLM_PROVIDER`` (e.g. ``LLM_PROVIDER=claude``)."""
    b = (os.environ.get("LLM_BACKEND") or "").strip().lower()
    if b:
        return b
    prov = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if prov == "claude":
        return "claude"
    if prov in ("hf", "huggingface", "endpoint", "router"):
        return "endpoint"
    return "auto"


def build_claude_llm() -> BaseLanguageModel:
    """Anthropic Claude via ``langchain_anthropic.ChatAnthropic`` (Messages API)."""
    from langchain_anthropic import ChatAnthropic

    api_key = _anthropic_api_key()
    if not api_key:
        raise EnvironmentError(
            "Set ANTHROPIC_API_KEY or CLAUDE_API_KEY for Claude, or use LLM_BACKEND=endpoint "
            "with a Hugging Face token."
        )
    model = (os.environ.get("CLAUDE_MODEL") or CLAUDE_MODEL_DEFAULT).strip()
    print(f"LLM: Anthropic Claude — {model}")
    return ChatAnthropic(
        model=model,
        temperature=0.2,
        max_tokens=1024,
        anthropic_api_key=api_key,
    )


def build_llm() -> BaseLanguageModel:
    """
    ``LLM_BACKEND``:

    - ``auto`` (default): Claude if ``ANTHROPIC_API_KEY`` is set; else local Zephyr when
      ``bitsandbytes`` is available; else Hugging Face Inference chat (needs HF token).
    - ``claude`` / ``anthropic``: Claude only (requires ``ANTHROPIC_API_KEY``).
    - ``local``: 4-bit Zephyr only (GPU + ``bitsandbytes``).
    - ``endpoint`` / ``api`` / ``router`` / ``hf``: HF router chat only (``HF_LLM_MODEL``).

    Env: ``ANTHROPIC_API_KEY`` / ``CLAUDE_API_KEY``, ``LLM_PROVIDER``, ``CLAUDE_MODEL``, ``HF_LLM_MODEL``, HF token for embeddings.
    """
    backend = _resolved_llm_backend()
    has_anthropic_key = bool(_anthropic_api_key())

    if backend in ("claude", "anthropic"):
        return build_claude_llm()
    if backend == "local":
        print(f"LLM: local 4-bit Zephyr — {LLM_MODEL_ID}")
        return build_zephyr_llm()
    if backend in ("endpoint", "api", "router", "hf"):
        mid = (os.environ.get("HF_LLM_MODEL") or HF_LLM_MODEL_DEFAULT).strip()
        print(f"LLM: Hugging Face Inference (chat) — {mid}")
        return HuggingFaceRouterChatLLM(model_id=mid)

    # ---- auto ----
    if has_anthropic_key:
        return build_claude_llm()

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        mid = (os.environ.get("HF_LLM_MODEL") or HF_LLM_MODEL_DEFAULT).strip()
        print(
            "LLM: auto — no Anthropic key and no bitsandbytes; using HF Inference chat "
            f"({mid}). Add ANTHROPIC_API_KEY or CLAUDE_API_KEY to use Claude, or set LLM_BACKEND=claude / endpoint."
        )
        return HuggingFaceRouterChatLLM(model_id=mid)

    print(f"LLM: auto — local 4-bit Zephyr — {LLM_MODEL_ID}")
    return build_zephyr_llm()


def build_zephyr_llm() -> HuggingFacePipeline:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    if os.environ.get("SKIP_LOCAL_LLM", "").lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "SKIP_LOCAL_LLM is set; unset it to run the quantized Zephyr model locally."
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for 4-bit loading. On Windows, use a CUDA-enabled "
            "Python on Linux/WSL or install a compatible bitsandbytes build."
        ) from exc

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.05,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ---------------------------------------------------------------------------
# 7. Retrieval QA chains (BM25, Vector, Hybrid)
# ---------------------------------------------------------------------------
def build_retrieval_qa(llm: BaseLanguageModel, retriever) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


# ---------------------------------------------------------------------------
# 8–10. Query helpers — answers + retrieved chunk preview
# ---------------------------------------------------------------------------
def preview_doc(doc: Document, max_chars: int = 200) -> str:
    text = (doc.page_content or "").replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    src = (doc.metadata or {}).get("source", "unknown")
    return f"[source={src}] {text}"


def print_retrieved(label: str, documents: Sequence[Document]) -> None:
    print(f"\n--- Retrieved chunks ({label}) ---")
    for i, doc in enumerate(documents, start=1):
        print(f"  [{i}] {preview_doc(doc)}")


def run_chain_with_trace(qa: RetrievalQA, query: str, method_label: str) -> tuple[str, list[Document]]:
    result: dict[str, Any] = qa.invoke({"query": query})
    answer = (result.get("result") or "").strip()
    src_docs = result.get("source_documents", []) or []
    print_retrieved(method_label, src_docs)
    return answer, src_docs


def run_chain_result(qa: RetrievalQA, query: str, preview_chars: int = 200) -> dict[str, Any]:
    """Same as ``run_chain_with_trace`` but JSON-serializable (for HTTP APIs / UIs)."""
    result: dict[str, Any] = qa.invoke({"query": query})
    answer = (result.get("result") or "").strip()
    src_docs = result.get("source_documents", []) or []
    chunks: list[dict[str, str]] = []
    for doc in src_docs:
        chunks.append(
            {
                "source": str((doc.metadata or {}).get("source", "unknown")),
                "preview": preview_doc(doc, preview_chars),
            }
        )
    return {"answer": answer, "chunks": chunks}


def compare_three_results(
    query: str,
    bm25_chain: RetrievalQA,
    vector_chain: RetrievalQA,
    hybrid_chain: RetrievalQA,
) -> dict[str, Any]:
    """BM25, vector, and hybrid answers + retrieved chunk previews (no printing)."""
    return {
        "query": query,
        "bm25": run_chain_result(bm25_chain, query),
        "vector": run_chain_result(vector_chain, query),
        "hybrid": run_chain_result(hybrid_chain, query),
    }


def hybrid_insight_for_results(
    query: str,
    bm25: dict[str, Any],
    vector: dict[str, Any],
    hybrid: dict[str, Any],
    *,
    max_bullets: int = 5,
) -> dict[str, Any]:
    """
    Build a short, query-specific explanation of why hybrid (BM25 + vector fusion)
    is valuable for *this* comparison — from sources, chunk overlap, and answer shape.

    Used by the API / UI; no extra LLM call.
    """
    def chunk_sources(block: dict[str, Any]) -> list[str]:
        return [str(c.get("source", "unknown")) for c in (block.get("chunks") or [])]

    def unique_sources(block: dict[str, Any]) -> list[str]:
        return sorted(set(chunk_sources(block)))

    def preview_fingerprints(block: dict[str, Any]) -> set[str]:
        out: set[str] = set()
        for c in block.get("chunks") or []:
            p = (c.get("preview") or "").strip().lower()
            out.add(p[:120] if p else "")
        out.discard("")
        return out

    ub, uv, uh = unique_sources(bm25), unique_sources(vector), unique_sources(hybrid)
    nb, nv, nh = (
        len(bm25.get("chunks") or []),
        len(vector.get("chunks") or []),
        len(hybrid.get("chunks") or []),
    )
    fb, fv, fh = preview_fingerprints(bm25), preview_fingerprints(vector), preview_fingerprints(hybrid)

    bullets: list[str] = []

    if len(uh) > max(len(ub), len(uv)):
        bullets.append(
            f"For this query, hybrid retrieved chunks tagged from {len(uh)} source types "
            f"({', '.join(uh)}), while BM25 alone covered {len(ub)} ({', '.join(ub) or '—'}) and "
            f"vector alone {len(uv)} ({', '.join(uv) or '—'}), so the blended context is broader."
        )

    only_vs_bm25 = fh - fb
    only_vs_vec = fh - fv
    if only_vs_bm25 or only_vs_vec:
        bullets.append(
            "Rank fusion merged BM25 and dense rankings: some passages in the hybrid list do not "
            "appear in both single-method top‑k lists, so the model can see evidence that neither "
            "retriever would have supplied on its own."
        )

    if nh > max(nb, nv):
        bullets.append(
            f"Ensemble retrieval combined lists here (hybrid: {nh} chunks vs BM25: {nb}, vector: {nv}), "
            "widening what the LLM can cite in one shot."
        )

    def word_count(s: str) -> int:
        return len((s or "").split())

    ab, av, ah = (
        word_count(str(bm25.get("answer", ""))),
        word_count(str(vector.get("answer", ""))),
        word_count(str(hybrid.get("answer", ""))),
    )

    def looks_weak_answer(text: str) -> bool:
        t = (text or "").lower().strip()
        if len(t) < 35:
            return True
        return any(
            x in t
            for x in (
                "don't know",
                "do not know",
                "not enough",
                "cannot answer",
                "no information",
                "not provided",
            )
        )

    hb, hv, hh = bm25.get("answer", ""), vector.get("answer", ""), hybrid.get("answer", "")
    if (looks_weak_answer(hb) or looks_weak_answer(hv)) and not looks_weak_answer(hh):
        bullets.append(
            "With this retrieval draw, at least one single-method answer looks thin or evasive, "
            "while the hybrid context still supported a more grounded reply—typical when one retriever "
            "misses the “right” passage."
        )
    elif ah > max(ab, av) * 1.2 and ah >= 12:
        bullets.append(
            "The hybrid-backed answer is noticeably longer here, which often means the fused passages "
            "gave the model enough material to synthesize instead of hedging."
        )

    qlow = query.lower()
    has_policy = any(
        t in qlow for t in ("crr", "repo", "rbi", "reserve bank", "liquidity", "policy rate", "banking system")
    )
    has_retail = any(
        t in qlow for t in ("emi", "loan", "borrow", "inflation", "interest rate", "mortgage", "tenure")
    )
    if has_policy and has_retail:
        bullets.append(
            "Your wording mixes policy/institutional terms with borrower outcomes—where keyword search "
            "(BM25) and paraphrase-friendly vector search complement each other; hybrid keeps both signals."
        )
    elif has_policy:
        bullets.append(
            "Policy-heavy queries hit exact tokens (CRR, repo, RBI) where BM25 shines; dense search still "
            "adds related passages BM25 might rank lower, which hybrid can promote via fusion."
        )
    elif has_retail:
        bullets.append(
            "Borrower-focused questions align well with FAQ/article-style chunks from vector search; "
            "hybrid still retains any strong lexical hits from RBI text in the same context."
        )

    if not bullets:
        bullets.append(
            "Hybrid uses weighted reciprocal rank fusion so top lexical and semantic hits reinforce "
            "each other in one ranked list—robust when either method alone would drop a useful chunk."
        )

    # De-dupe similar lines (fingerprints can make first two both fire; keep variety)
    seen: set[str] = set()
    unique_bullets: list[str] = []
    for b in bullets:
        key = b[:80]
        if key not in seen:
            seen.add(key)
            unique_bullets.append(b)

    caveat = (
        "Always compare the three panels above: hybrid is a design choice for robustness, not a guarantee "
        "that every answer beats BM25 or vector on every run."
    )

    return {
        "headline": "Why hybrid retrieval fits this query",
        "bullets": unique_bullets[:max_bullets],
        "caveat": caveat,
    }


def compare_three_chains(
    query: str,
    bm25_chain: RetrievalQA,
    vector_chain: RetrievalQA,
    hybrid_chain: RetrievalQA,
) -> None:
    print("\n" + "=" * 80)
    print(f"Query: {query}")

    print("\n--- BM25 RESULT ---")
    bm25_answer, _ = run_chain_with_trace(bm25_chain, query, "BM25")
    print(bm25_answer)

    print("\n--- VECTOR RESULT ---")
    vector_answer, _ = run_chain_with_trace(vector_chain, query, "Vector")
    print(vector_answer)

    print("\n--- HYBRID RESULT ---")
    hybrid_answer, _ = run_chain_with_trace(hybrid_chain, query, "Hybrid")
    print(hybrid_answer)


# ---------------------------------------------------------------------------
# Bonus: different ensemble weights
# ---------------------------------------------------------------------------
def demo_weight_sweep(
    llm,
    bm25_retriever,
    vectorstore_retriever,
    query: str,
    weight_pairs: Sequence[tuple[float, float]],
) -> None:
    print("\n" + "#" * 80)
    print("BONUS: EnsembleRetriever weight sweep (BM25 weight, Vector weight)")
    print("#" * 80)
    for w_bm25, w_vec in weight_pairs:
        ensemble_retriever = build_ensemble_retriever(
            bm25_retriever,
            vectorstore_retriever,
            weights=[w_bm25, w_vec],
        )
        chain = build_retrieval_qa(llm, ensemble_retriever)
        print(f"\nWeights [BM25={w_bm25}, Vector={w_vec}]  |  Query: {query}")
        ans, _docs = run_chain_with_trace(chain, query, f"Hybrid w=({w_bm25},{w_vec})")
        print(ans)
        print("-" * 40)


def parse_weights(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG financial demo")
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default="0.5,0.5",
        help="Comma-separated weights for EnsembleRetriever: BM25,Vector (e.g. 0.6,0.4)",
    )
    args = parser.parse_args()
    ensemble_weights = parse_weights(args.ensemble_weights)
    if len(ensemble_weights) != 2:
        raise SystemExit("--ensemble-weights must have exactly two values: BM25,Vector")

    print(f"EnsembleRetriever weights (BM25, Vector): {ensemble_weights}")
    print("Loading and chunking documents…")
    all_docs = build_all_documents()
    chunks = chunk_documents(all_docs)
    print(f"Total leaf documents: {len(all_docs)}  |  Chunks: {len(chunks)}")

    print("Building BM25 and vector retrievers…")
    keyword_retriever = build_bm25_retriever(chunks)
    vectorstore = build_vectorstore(chunks)
    vectorstore_retriever = build_vectorstore_retriever(vectorstore)
    ensemble_retriever = build_ensemble_retriever(
        keyword_retriever,
        vectorstore_retriever,
        weights=ensemble_weights,
    )

    print("Initializing language model…")
    llm = build_llm()

    bm25_chain = build_retrieval_qa(llm, keyword_retriever)
    vector_chain = build_retrieval_qa(llm, vectorstore_retriever)
    hybrid_chain = build_retrieval_qa(llm, ensemble_retriever)

    test_queries = [
        # Keyword-oriented
        "CRR rate India",
        "Repo rate current",
        # Conceptual
        "Why does EMI increase?",
        "How inflation affects loans?",
        # Hybrid-style
        "Impact of repo rate on EMI",
        "CRR effect on banking system",
    ]

    intro = textwrap.dedent(
        """
        Expected behaviour (retrieval):
        - BM25: strong on exact phrases (e.g. CRR, repo rate, policy keywords in RBI text).
        - Vector: strong on paraphrases / concepts (EMI, inflation, transmission).
        - Hybrid: merges both rank lists (RRF), usually most robust overall.
        """
    ).strip()
    print(intro)

    for q in test_queries:
        compare_three_chains(q, bm25_chain, vector_chain, hybrid_chain)

    demo_weight_sweep(
        llm,
        keyword_retriever,
        vectorstore_retriever,
        query="Impact of repo rate on EMI",
        weight_pairs=[(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)],
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
