"""
comparison.py — Feature 2: Side-by-Side Retrieval Comparison (with latency)

Runs all three retrieval strategies on the same query and returns
a structured dict for side-by-side display, including per-retriever
latency measurements.

Step 3 fix: each retriever is independently timed.

Public API:
    compare_retrieval(query, model, corpus_embeddings, faq_docs,
                      bm25_index, top_k=5)
        -> {
             "semantic": {"results": list[dict], "latency_ms": float},
             "keyword":  {"results": list[dict], "latency_ms": float},
             "hybrid":   {"results": list[dict], "latency_ms": float},
           }
"""

import time
from modules.retriever import FAQRetriever


def compare_retrieval(
    query:             str,
    model,
    corpus_embeddings,
    faq_docs:          list[dict],
    bm25_index,
    top_k:             int = 5,
) -> dict:
    """
    Execute semantic, keyword, and hybrid retrieval on the same query.
    Each retriever is timed independently (wall-clock ms).

    Step 2 optimisation: hybrid reuses sem + keyword results computed above
    so there is zero duplicate computation.

    Args:
        query:             User query string.
        model:             Loaded SentenceTransformer model.
        corpus_embeddings: (N, 384) float32 matrix.
        faq_docs:          List of FAQ dicts.
        bm25_index:        Pre-built BM25Okapi index.
        top_k:             Results per retriever column.

    Returns:
        dict with keys "semantic", "keyword", "hybrid".
        Each value is {"results": list[dict], "latency_ms": float}.
    """
    retriever = FAQRetriever(
        model             = model,
        corpus_embeddings = corpus_embeddings,
        faq_docs          = faq_docs,
        bm25_index        = bm25_index,
        top_k             = top_k,
    )

    # ── Semantic ──────────────────────────────────────────────
    t0  = time.perf_counter()
    sem = retriever.retrieve_semantic(query)
    sem_ms = (time.perf_counter() - t0) * 1000

    # ── Keyword ───────────────────────────────────────────────
    t0  = time.perf_counter()
    kw  = retriever.retrieve_keyword(query)
    kw_ms = (time.perf_counter() - t0) * 1000

    # ── Hybrid (reuse sem + kw — no recomputation) ────────────
    t0  = time.perf_counter()
    hyb = retriever.retrieve_hybrid(query, sem_results=sem, kw_results=kw)
    hyb_ms = (time.perf_counter() - t0) * 1000

    return {
        "semantic": {"results": sem, "latency_ms": round(sem_ms,  2)},
        "keyword":  {"results": kw,  "latency_ms": round(kw_ms,   2)},
        "hybrid":   {"results": hyb, "latency_ms": round(hyb_ms,  2)},
    }
