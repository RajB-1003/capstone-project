"""
langchain_wrapper.py — Feature 4: Minimal LangChain Integration

Wraps the existing semantic search as a LangChain-compatible retriever.
Degrades gracefully when langchain is NOT installed — the class still
works as a standalone retriever.

Design choices:
    - Does NOT replace existing search logic
    - Does NOT require langchain to be installed
    - get_relevant_documents() always works
    - as_langchain_retriever() returns None if langchain is absent

Public API:
    LANGCHAIN_AVAILABLE: bool
    SemanticFAQRetriever(model, corpus_embeddings, faq_docs, top_k=5)
        .get_relevant_documents(query) -> list[dict]     (always works)
        .as_langchain_retriever()      -> BaseRetriever | None
"""

import numpy as np

# Optional LangChain dependency — degrade gracefully if absent
try:
    from langchain.schema import BaseRetriever, Document  # type: ignore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain_core.retrievers import BaseRetriever  # type: ignore
        from langchain_core.documents  import Document       # type: ignore
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False


class SemanticFAQRetriever:
    """
    Semantic FAQ retriever with optional LangChain interface.

    Works standalone (returns list[dict]) and can optionally expose a
    LangChain-compatible BaseRetriever via as_langchain_retriever().

    Args:
        model:             Loaded SentenceTransformer model.
        corpus_embeddings: (N, 384) float32 numpy array.
        faq_docs:          List of FAQ dicts.
        top_k:             Number of results to return.
    """

    def __init__(
        self,
        model,
        corpus_embeddings: np.ndarray,
        faq_docs:          list[dict],
        top_k:             int = 5,
    ):
        self.model             = model
        self.corpus_embeddings = corpus_embeddings
        self.faq_docs          = faq_docs
        self.top_k             = top_k

    def get_relevant_documents(self, query: str) -> list[dict]:
        """
        Retrieve relevant FAQs for a query.

        Returns a list of FAQ dicts (extracted from the search_semantic state dict).
        """
        from modules.semantic_search import search_semantic
        state = search_semantic(
            query,
            self.faq_docs,
            self.corpus_embeddings,
            self.model,
            top_k=self.top_k,
        )
        return state.get("retrieved_docs", [])

    def as_langchain_retriever(self):
        """
        Return a LangChain BaseRetriever wrapping this retriever.

        Returns:
            A BaseRetriever instance if langchain is installed, else None.

        The returned retriever converts FAQ results to LangChain Document
        objects with the full FAQ as metadata.
        """
        if not LANGCHAIN_AVAILABLE:
            return None

        # Capture closure variables
        _get_docs = self.get_relevant_documents

        class _LangChainFAQRetriever(BaseRetriever):
            """Inner LangChain-compatible retriever (auto-generated)."""

            def _get_relevant_documents(
                self, query: str, *, run_manager=None
            ) -> list:
                results = _get_docs(query)
                return [
                    Document(
                        page_content=r.get("answer", ""),
                        metadata={
                            "id":       r.get("id", ""),
                            "question": r.get("question", ""),
                            "category": r.get("category", ""),
                            "score":    r.get("score", 0.0),
                            "tags":     r.get("tags", []),
                        },
                    )
                    for r in results
                ]

        return _LangChainFAQRetriever()
