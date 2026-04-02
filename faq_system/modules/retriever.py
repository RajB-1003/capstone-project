"""
retriever.py — Feature 8: FAQRetriever Abstraction

Provides a clean, reusable class interface over the three retrieval
strategies. Designed for future RAG integration and as the shared
foundation for comparison.py.

Does NOT duplicate logic — every method delegates directly to the
existing search_* functions.

Public API:
    FAQRetriever(model, corpus_embeddings, faq_docs, bm25_index, top_k=5)
        .retrieve_semantic(query) -> list[dict]
        .retrieve_keyword(query)  -> list[dict]
        .retrieve_hybrid(query)   -> list[dict]
"""

import numpy as np

from modules.semantic_search import search_semantic
from modules.keyword_search  import search_keyword
from modules.hybrid_search   import reciprocal_rank_fusion


class FAQRetriever:
    """
    Unified retrieval interface for semantic, keyword, and hybrid search.

    Wraps the existing pure-function modules with a class API that is
    easy to compose, pass around, and extend (e.g., for LangChain RAG).

    Args:
        model:             Loaded SentenceTransformer model.
        corpus_embeddings: (N, 384) float32 matrix.
        faq_docs:          List of FAQ dicts.
        bm25_index:        Pre-built BM25Okapi index.
        top_k:             Maximum results per retriever (default 5).
    """

    def __init__(
        self,
        model,
        corpus_embeddings: np.ndarray,
        faq_docs:          list[dict],
        bm25_index,
        top_k: int = 5,
    ):
        self.model             = model
        self.corpus_embeddings = corpus_embeddings
        self.faq_docs          = faq_docs
        self.bm25_index        = bm25_index
        self.top_k             = top_k

    def _normalise(self, raw_docs: list[dict], score_key: str) -> list[dict]:
        """
        Copy docs and add a unified 'score' field from the given score_key.
        Adds 'rank' based on position. Step 8: reads 'tags' directly (no metadata fallback).
        """
        out = []
        for i, doc in enumerate(raw_docs, 1):
            d = doc.copy()
            d["score"] = round(float(d.get(score_key, 0.0)), 4)
            d["rank"]  = i
            d.setdefault("tags", [])   # Step 8: guarantee tags field present
            out.append(d)
        return out

    def _apply_filters(
        self,
        docs: list[dict],
        categories: list[str] | None,
        tags: list[str] | None,
    ) -> list[dict]:
        """
        Step 7: Filter a result list by category and/or tags BEFORE ranking.

        Applied inside retrieve_* so the filter reduces the ranked set,
        not just the display.  If both filters are empty the list is returned
        unchanged.

        Tag lookup checks the standardised 'tags' field (Step 8).
        """
        if not categories and not tags:
            return docs
        out = []
        for d in docs:
            if categories and d.get("category", "") not in categories:
                continue
            if tags:
                dtags = set(d.get("tags", []))
                if not any(t in dtags for t in tags):
                    continue
            out.append(d)
        return out

    def retrieve_semantic(
        self,
        query:      str,
        categories: list[str] | None = None,
        tags:       list[str] | None = None,
    ) -> list[dict]:
        """
        Run semantic similarity search.
        Returns up to top_k results sorted by cosine similarity (desc).
        Filters applied pre-ranking (Step 7).
        """
        state = search_semantic(
            query, self.faq_docs, self.corpus_embeddings,
            self.model, top_k=self.top_k,
        )
        docs = self._apply_filters(state.get("retrieved_docs", []), categories, tags)
        # Fix 2: re-sort after filter and enforce top_k
        docs = sorted(docs, key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        docs = docs[: self.top_k]
        return self._normalise(docs, "similarity_score")

    def retrieve_keyword(
        self,
        query:      str,
        categories: list[str] | None = None,
        tags:       list[str] | None = None,
    ) -> list[dict]:
        """
        Run BM25 keyword search.
        Returns up to top_k results sorted by BM25 score (desc).
        Filters applied pre-ranking (Step 7).
        """
        state = search_keyword(
            query, self.faq_docs, self.bm25_index, top_k=self.top_k,
        )
        docs = self._apply_filters(state.get("retrieved_docs", []), categories, tags)
        # Fix 2: re-sort after filter and enforce top_k
        docs = sorted(docs, key=lambda x: x.get("bm25_score", 0.0), reverse=True)
        docs = docs[: self.top_k]
        return self._normalise(docs, "bm25_score")

    def retrieve_hybrid(
        self,
        query:       str,
        sem_results: list[dict] | None = None,
        kw_results:  list[dict] | None = None,
        categories:  list[str] | None = None,
        tags:        list[str] | None = None,
    ) -> list[dict]:
        """
        Run hybrid RRF fusion of semantic + keyword results.

        Step 2 fix: accepts pre-computed sem_results / kw_results to avoid
        recomputation when the caller (comparison.py) already has them.

        Adds source attribution:
            "both"     — appeared in both lists (highest confidence)
            "semantic" — only in semantic results
            "keyword"  — only in keyword results

        Returns up to top_k results sorted by RRF score (desc).
        """
        if sem_results is None:
            sem_results = self.retrieve_semantic(query, categories, tags)
        if kw_results is None:
            kw_results = self.retrieve_keyword(query, categories, tags)

        # reciprocal_rank_fusion expects list[list[dict]] where each dict has "id"
        fused = reciprocal_rank_fusion([sem_results, kw_results], k=60)
        # Fix 2: RRF already outputs desc order; apply top_k explicitly
        fused = fused[: self.top_k]

        # Source attribution
        sem_ids = {r.get("id") for r in sem_results}
        kw_ids  = {r.get("id") for r in kw_results}
        for i, doc in enumerate(fused, 1):
            did = doc.get("id")
            doc["source"] = (
                "both"     if (did in sem_ids and did in kw_ids) else
                "semantic" if did in sem_ids                     else
                "keyword"
            )
            # Ensure score and rank are present
            doc.setdefault("score", round(doc.get("rrf_score", 0.0), 4))
            doc["rank"] = i

        return fused
