"""
semantic_search.py — Phase 1: Semantic Retrieval

Responsibilities:
  - Accept a natural language query + pre-computed FAQ embeddings
  - Compute dot product similarity (= cosine similarity for L2-normed vectors)
  - Return a structured state object matching the system data contract

Architecture mandate (Phased Agent Prompt Chain Design.docx):
  - Use normalized dot product via NumPy vectorized ops (BLAS-optimized)
  - Do NOT use raw cosine division — vectors are pre-normalized in embedder.py
  - Return the shared data contract structure, not raw lists
  - top_k is configurable; function is stateless and independently testable

Data Contract Output:
  {
      "query":            str,
      "detected_entities": list,    # empty at Phase 1 — set by router
      "route_decision":   str,      # "semantic" (set here)
      "retrieved_docs":   list[dict],  # top-k FAQ records + similarity score
      "scores":           list[float], # similarity scores in descending order
      "rationale":        str       # human-readable explanation
  }

Public API:
  search_semantic(query, faq_docs, embeddings, model, top_k=5) -> dict
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from modules.embedder import embed_single


# ──────────────────────────────────────────────────────────────
# Helper: vectorized dot-product similarity
# ──────────────────────────────────────────────────────────────
def _compute_dot_similarity(
    query_vec: np.ndarray,
    doc_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute similarity between one query vector and all document vectors.

    Since both query_vec and each row in doc_embeddings are L2-normalized (done
    in embedder.py), the dot product equals cosine similarity.

    Uses NumPy matmul — a single BLAS SGEMV call, O(N·D) with very low constant.

    Args:
        query_vec:      shape (D,)   — L2-normalized query embedding.
        doc_embeddings: shape (N, D) — L2-normalized FAQ embeddings matrix.

    Returns:
        similarities: shape (N,) float32 — similarity score for every FAQ.
    """
    # doc_embeddings @ query_vec  →  (N,) via BLAS
    return doc_embeddings @ query_vec


# ──────────────────────────────────────────────────────────────
# Main public function
# ──────────────────────────────────────────────────────────────
def search_semantic(
    query: str,
    faq_docs: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> dict:
    """
    Retrieve the top-k most semantically relevant FAQ documents for a query.

    Steps:
      1. Embed the query (L2-normalized, 1-D vector).
      2. Compute dot-product similarity against all pre-computed FAQ embeddings.
      3. Rank documents by descending similarity score.
      4. Pack results into the shared data contract structure.

    Args:
        query:      Natural language query string.
        faq_docs:   List of raw FAQ dicts loaded from faqs.json.
        embeddings: np.ndarray (N, 384) — pre-computed, L2-normalized FAQ corpus.
        model:      Loaded SentenceTransformer for query encoding.
        top_k:      Number of top results to return (default 5).

    Returns:
        dict matching the shared data contract:
        {
            "query":             str,
            "detected_entities": list,
            "route_decision":    str,       # always "semantic" from this function
            "retrieved_docs":    list[dict],
            "scores":            list[float],
            "rationale":         str
        }
    """
    # ── Step 1: embed the query ────────────────────────────────
    query_vec = embed_single(query, model)   # shape (384,), float32, L2-normed

    # ── Step 2: vectorized dot-product similarity ──────────────
    similarities = _compute_dot_similarity(query_vec, embeddings)  # shape (N,)

    # ── Step 3: rank by descending similarity ──────────────────
    actual_k = min(top_k, len(faq_docs))
    # np.argpartition is O(N) for top-k; then sort only the k elements
    top_indices = np.argpartition(similarities, -actual_k)[-actual_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    # ── Step 4: build structured result list ──────────────────
    retrieved_docs = []
    scores = []
    for idx in top_indices:
        doc = faq_docs[idx].copy()
        score = float(similarities[idx])
        doc["similarity_score"] = round(score, 4)
        retrieved_docs.append(doc)
        scores.append(round(score, 4))

    # ── Step 5: build rationale string (Phase 6 will enrich this) ─
    top_score = scores[0] if scores else 0.0
    rationale = (
        f"Semantic search activated. "
        f"Top result: '{retrieved_docs[0]['question'][:60]}...' "
        f"with similarity score {top_score:.4f}. "
        f"Retrieved {len(retrieved_docs)} documents using normalized dot product."
        if retrieved_docs
        else "Semantic search returned no results."
    )

    # ── Step 6: return shared data contract ───────────────────
    return {
        "query":             query,
        "detected_entities": [],          # populated by Tier-1 router (Phase 3)
        "route_decision":    "semantic",
        "retrieved_docs":    retrieved_docs,
        "scores":            scores,
        "rationale":         rationale,
    }
