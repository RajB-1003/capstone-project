"""
langchain_wrapper.py — LangChain-compatible retriever for the Semantic FAQ System
==================================================================================

This module wraps the existing run_pipeline() function into a LangChain-style
retriever interface, enabling integration with:

  * LangChain RAG pipelines  (e.g., RetrievalQA, ConversationalRetrievalChain)
  * LangChain Agents         (as a Tool with a retriever backend)
  * LangChain Chains         (e.g., StuffDocumentsChain, create_retrieval_chain)

DESIGN PRINCIPLES
-----------------
  * Zero duplication  — all retrieval logic lives in the existing modules.
  * Zero side effects — no existing file is modified.
  * Graceful detection — works with or without LangChain / langchain_core installed.
  * Fixtures are loaded once (module-level cache) on first call.

USAGE (standalone / no LangChain)
----------------------------------
    from langchain_wrapper import FAQRetriever

    retriever = FAQRetriever()
    docs = retriever.get_relevant_documents("What happens if I miss an exam?")
    for doc in docs:
        print(doc.page_content)      # answer text
        print(doc.metadata)          # question, score, source, category, tags, rank

USAGE (LangChain RAG chain)
----------------------------
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    from langchain_wrapper import FAQRetriever

    retriever = FAQRetriever()
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    answer = qa.invoke("What happens if I miss an exam?")

USAGE (LangChain Agent tool)
------------------------------
    from langchain.tools.retriever import create_retriever_tool
    from langchain_wrapper import FAQRetriever

    tool = create_retriever_tool(FAQRetriever(), "faq_search", "Search the FAQ database")
"""

from __future__ import annotations

import os
import sys
from typing import List

# ── Ensure faq_system root is on sys.path ────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ════════════════════════════════════════════════════════════════════
# DOCUMENT CLASS
# Prefer real LangChain Document; fall back to a compatible stub.
# ════════════════════════════════════════════════════════════════════

try:
    from langchain_core.documents import Document          # LangChain ≥ 0.1 (installed)
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document              # LangChain < 0.1 (legacy)
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False

        class Document:                                    # type: ignore[no-redef]
            """
            Minimal LangChain-compatible Document stub.
            Behaviorally identical to langchain_core.documents.Document.
            Used automatically when LangChain is not installed.
            """
            __slots__ = ("page_content", "metadata")

            def __init__(self, page_content: str, metadata: dict | None = None):
                self.page_content: str  = page_content
                self.metadata:     dict = metadata or {}

            def __repr__(self) -> str:
                preview = self.page_content[:80].replace("\n", " ")
                return (
                    f"Document(page_content={preview!r}..., "
                    f"metadata={self.metadata!r})"
                )


# ════════════════════════════════════════════════════════════════════════════════
# FIXTURE LOADER
# Loaded once at module level; never reloads unless explicitly reset.
# ════════════════════════════════════════════════════════════════════════════════

_FIXTURES: tuple | None = None


def _load_fixtures() -> tuple:
    """
    Load all pipeline fixtures required by run_pipeline().

    Returns:
        (model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings)
    """
    from modules.embedder        import load_embedding_model, load_and_embed_faqs
    from modules.keyword_search  import build_bm25_index
    from modules.router_tier1    import load_regex_patterns
    from modules.router_tier2    import embed_intents
    from modules.embedding_store import load_embeddings, save_embeddings

    DATA_DIR       = os.path.join(_HERE, "data")
    FAQS_PATH      = os.path.join(DATA_DIR, "faqs.json")
    PATTERNS_PATH  = os.path.join(DATA_DIR, "regex_patterns.json")
    EXEMPLARS_PATH = os.path.join(DATA_DIR, "intent_exemplars.json")
    EMBED_PATH     = os.path.join(DATA_DIR, "corpus_embeddings.npy")

    model             = load_embedding_model()
    corpus_embeddings = load_embeddings(EMBED_PATH)
    faq_docs, fresh   = load_and_embed_faqs(FAQS_PATH, model)
    if corpus_embeddings is None:
        corpus_embeddings = fresh
        save_embeddings(corpus_embeddings, EMBED_PATH)

    bm25_index        = build_bm25_index(faq_docs)
    patterns          = load_regex_patterns(PATTERNS_PATH)
    intent_embeddings = embed_intents(EXEMPLARS_PATH, model)

    return model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings


def _get_fixtures() -> tuple:
    """Return cached fixtures, loading them on first access."""
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = _load_fixtures()
    return _FIXTURES


# ════════════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════════════

def _result_to_document(result: dict) -> Document:
    """Convert one pipeline result dict into a LangChain Document."""
    return Document(
        page_content=result.get("answer", ""),
        metadata={
            "question": result.get("question", ""),
            "score":    result.get("score",    0.0),
            "source":   result.get("source",   ""),   # "semantic" | "keyword" | "both"
            "category": result.get("category", ""),
            "tags":     result.get("tags",     []),
            "rank":     result.get("rank",     0),
        },
    )


# ════════════════════════════════════════════════════════════════════════════════
# RETRIEVER CLASS
# Implements the LangChain retriever interface without inheriting from the
# Pydantic-based BaseRetriever (which would require field declarations and
# an __init__ rewrite).  Duck-typing works here because LangChain chains
# only call get_relevant_documents / _get_relevant_documents / aget_relevant_documents.
# ════════════════════════════════════════════════════════════════════════════════

class FAQRetriever:
    """
    LangChain-compatible retriever wrapping the Semantic FAQ pipeline.

    Works as a drop-in retriever for any LangChain chain or agent.
    No retrieval logic is duplicated — all calls pass through run_pipeline().

    Args:
        top_k:    Maximum number of documents to return (default 5).
        fixtures: Optional pre-loaded fixture tuple
                  (model, corpus_embeddings, faq_docs, bm25_index,
                   patterns, intent_embeddings).
                  If None, fixtures are loaded lazily on the first call.
                  Pass Streamlit's @st.cache_resource result to reuse
                  already-loaded fixtures without extra disk I/O.

    Examples:
        # Basic usage
        retriever = FAQRetriever()
        docs = retriever.get_relevant_documents("What happens if I miss an exam?")

        # With pre-loaded Streamlit fixtures
        model, corpus_embeddings, faq_docs, bm25_index, patterns, ie = load_pipeline_fixtures()
        retriever = FAQRetriever(fixtures=(model, corpus_embeddings, faq_docs, bm25_index, patterns, ie))

        # LangChain RAG
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=FAQRetriever())
    """

    # Required by some LangChain internals that check retriever type
    search_type:  str = "similarity"
    search_kwargs: dict = {}

    def __init__(self, top_k: int = 5, fixtures: tuple | None = None):
        self._top_k    = top_k
        self._fixtures = fixtures

    # ── Core pipeline call ──────────────────────────────────────────────────

    def _run_pipeline(self, query: str) -> dict:
        """Invoke run_pipeline() with correct fixtures. Returns full response dict."""
        from modules.pipeline import run_pipeline

        model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings = (
            self._fixtures if self._fixtures is not None else _get_fixtures()
        )
        return run_pipeline(
            query             = query,
            model             = model,
            corpus_embeddings = corpus_embeddings,
            faq_docs          = faq_docs,
            bm25_index        = bm25_index,
            patterns          = patterns,
            intent_embeddings = intent_embeddings,
            top_k             = self._top_k,
        )

    # ── LangChain retriever interface ───────────────────────────────────────

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Primary LangChain retriever method.

        Calls the full pipeline and converts results to Documents.

        Args:
            query: Natural-language question or keyword query.

        Returns:
            List of Document objects ordered by relevance (rank 1 = best).
            page_content = answer text.
            metadata     = question, score, source, category, tags, rank.
        """
        response = self._run_pipeline(query)
        return [_result_to_document(r) for r in response.get("results", [])]

    # LangChain ≥ 0.1 calls _get_relevant_documents internally
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """LangChain ≥ 0.1 internal hook — delegates to get_relevant_documents."""
        return self.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async variant — delegates to synchronous implementation."""
        return self.get_relevant_documents(query)

    # Required by some LangChain chain types
    def as_retriever(self, **kwargs):
        """Return self (already a retriever). Mirrors BaseRetriever.as_retriever()."""
        return self

    # ── Extended interface ──────────────────────────────────────────────────

    def get_relevant_documents_with_metadata(self, query: str) -> dict:
        """
        Return the complete pipeline response for debugging or advanced use.

        Includes: route_decision, latency_ms, rationale, confidence flags,
        query_type, and the full results list.

        Args:
            query: Natural-language question or keyword query.

        Returns:
            Full pipeline response dict (same schema as run_pipeline()).
        """
        return self._run_pipeline(query)

    def __repr__(self) -> str:
        src = "pre-loaded" if self._fixtures is not None else "lazy"
        lc  = "with langchain_core" if _LANGCHAIN_AVAILABLE else "stub mode"
        return f"FAQRetriever(top_k={self._top_k}, fixtures={src}, langchain={lc})"
