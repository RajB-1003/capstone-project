"""
hybrid_search.py — Phase 5: Reciprocal Rank Fusion (RRF) Hybrid Search

Responsibilities:
  - Accept pre-computed results from search_semantic() and search_keyword()
  - Combine them using Reciprocal Rank Fusion (rank-based, no raw score mixing)
  - Return a fused, deduplicated ranked list in the shared data contract format

Architecture mandate (Phased Agent Prompt Chain Design.docx):
  - RRF is the ONLY valid fusion method: raw BM25 and cosine scores are
    mathematically incompatible (BM25: unbounded; cosine: [-1, 1])
  - k = 60 smoothing constant (per architecture specification)
  - Documents appearing in BOTH lists receive a higher combined RRF score
    than documents appearing in only one list (the core benefit of fusion)
  - This module does NOT call embedding model or BM25 index
  - This module does NOT recompute similarity scores
  - All it receives are already-ranked output lists from upstream modules

RRF Formula (per-document across all retrieval lists):
    rrf_score(doc) = Σ  1 / (k + rank_i(doc))
                    all lists i where doc appears
    rank is 1-indexed (top result = rank 1)

Why RRF instead of score averaging?
    - BM25 scores range 0 → ∞ (term statistics)
    - Cosine scores range -1 → 1 (geometric)
    - Averaging mixes incommensurable scales → biases toward BM25 magnitudes
    - RRF discards all raw scores and uses only ordinal rank positions
    - Result is scale-invariant and mathematically sound

Design decisions:
  - Deduplication key: doc["id"] field (unique per FAQ entry in faqs.json)
  - When a doc appears in multiple lists, its RRF contributions are SUMMED
  - The rrf_score field is injected into each returned doc dict
  - search_hybrid() receives full result dicts and unpacks retrieved_docs itself
  - No imports from embedder, semantic_search, or keyword_search (only uses their output)

Function signatures (defined before implementation — architecture mandate):

  reciprocal_rank_fusion(results_list, k=60)
      -> list[dict]
      Pure rank-based fusion; returns deduplicated docs sorted by RRF score.

  search_hybrid(state, semantic_results, keyword_results, top_k=5)
      -> dict
      Orchestrates fusion and packs result into shared data contract.

Public API:
  reciprocal_rank_fusion(results_list, k=60)                         -> list[dict]
  search_hybrid(state, semantic_results, keyword_results, top_k=5)   -> dict
"""

import copy


# ──────────────────────────────────────────────────────────────
# RRF constant (architecture mandate: k=60)
# ──────────────────────────────────────────────────────────────
RRF_K = 60


# ──────────────────────────────────────────────────────────────
# 1. Reciprocal Rank Fusion core algorithm
# ──────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    The algorithm is completely agnostic to the original scoring method of
    each retriever. It only uses the ordinal position (rank) of each document
    within its respective list.

    Algorithm:
        For each result list r_i (i = 0, 1, ...):
            For each document at position rank (1-indexed, rank=1 is best):
                rrf_scores[doc_id] += 1 / (k + rank)

        Sort all docs by rrf_scores descending.
        Deduplicate by doc["id"] — each FAQ ID appears exactly once.

        Why k=60?
            k is a smoothing constant that prevents top-ranked docs from
            dominating with excessively large 1/rank scores. k=60 is the
            standard value from the original RRF paper (Cormack et al., 2009)
            and is confirmed by the architecture doc.

    Args:
        results_list: List of ranked document lists. Each list is ordered
                      from highest to lowest relevance (index 0 = best).
                      Each doc must have an "id" field for deduplication.
                      Docs may also carry "similarity_score" or "bm25_score"
                      from upstream modules — these are preserved but NOT used
                      in RRF calculation.
        k:            RRF smoothing constant (default 60, per architecture).

    Returns:
        list[dict]: Deduplicated docs sorted by descending RRF score.
                    Each doc has a new "rrf_score" field (float, 4 dp).
                    Original score fields (similarity_score, bm25_score)
                    are preserved for downstream explainability (Phase 6).
    """
    # rrf_scores:    doc_id → accumulated RRF score
    rrf_scores:      dict[str, float] = {}
    # doc_store:     doc_id → doc dict (first occurrence wins for base fields)
    doc_store:       dict[str, dict]  = {}
    # semantic_rank: doc_id → rank in the FIRST list (semantic) for tie-breaking
    #   A lower semantic_rank means the semantic retriever was more confident.
    #   Used as secondary sort key when RRF scores are equal.
    semantic_rank:   dict[str, int]   = {}
    # source_map:    doc_id → which list(s) it appeared in
    source_map:      dict[str, str]   = {}   # "semantic" | "keyword" | "both"

    # Convention: results_list[0] = semantic, results_list[1] = keyword
    SOURCE_LABELS = ["semantic", "keyword"]

    for list_idx, result_list in enumerate(results_list):
        list_label = SOURCE_LABELS[list_idx] if list_idx < len(SOURCE_LABELS) else f"list{list_idx}"

        for rank, doc in enumerate(result_list, start=1):  # rank is 1-indexed (CRITICAL)
            doc_id = doc.get("id")
            if doc_id is None:
                # Fallback: use question text as dedup key if id missing
                doc_id = doc.get("question", f"__unknown_{rank}__")

            # Accumulate RRF score: documents in multiple lists get SUMMED
            contribution = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

            # Store first-seen version of each doc (preserves upstream scores)
            if doc_id not in doc_store:
                doc_store[doc_id] = copy.copy(doc)

            # Track semantic rank for tie-breaking (only from list 0 = semantic)
            if list_idx == 0 and doc_id not in semantic_rank:
                semantic_rank[doc_id] = rank

            # Source attribution: track which list(s) each doc appeared in
            if doc_id not in source_map:
                source_map[doc_id] = list_label
            elif source_map[doc_id] != list_label:
                source_map[doc_id] = "both"   # appeared in at least 2 lists

    # Assign a high semantic rank to docs that only appear in keyword results
    # (so they sort after semantic results when RRF scores are tied)
    _MAX_SEMANTIC_RANK = len(results_list[0]) + 1 if results_list else 9999
    for doc_id in rrf_scores:
        if doc_id not in semantic_rank:
            semantic_rank[doc_id] = _MAX_SEMANTIC_RANK

    # Sort: primary = rrf_score descending, secondary = semantic_rank ascending
    # This is a STABLE sort — equal-RRF, equal-semantic-rank docs preserve insertion order
    sorted_ids = sorted(
        rrf_scores,
        key=lambda did: (-rrf_scores[did], semantic_rank[did]),
    )

    # Build final list — inject rrf_score + source fields, preserve original fields
    fused_docs: list[dict] = []
    for doc_id in sorted_ids:
        doc = copy.copy(doc_store[doc_id])
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        doc["source"]    = source_map.get(doc_id, "unknown")   # "semantic"|"keyword"|"both"
        fused_docs.append(doc)

    return fused_docs


# ──────────────────────────────────────────────────────────────
# 2. Hybrid search orchestrator
# ──────────────────────────────────────────────────────────────

def search_hybrid(
    state: dict,
    semantic_results: dict,
    keyword_results: dict,
    top_k: int = 5,
) -> dict:
    """
    Combine semantic and keyword search results via RRF and update state.

    This function does NOT call search_semantic() or search_keyword() —
    it receives their already-computed output dicts and operates purely
    on the retrieved_docs lists within them.

    Fusion behavior:
        - Both retrieved_docs lists are passed to reciprocal_rank_fusion()
        - Docs in both lists receive summed RRF contributions (higher rank)
        - Docs in only one list receive a single contribution
        - Docs with zero overlap across both lists still get ranked by their
          single-list contribution

    Args:
        state:            Shared state dict (data contract). Must have "query".
        semantic_results: Output dict from search_semantic() — must contain
                          "retrieved_docs" key with ranked list.
        keyword_results:  Output dict from search_keyword() — must contain
                          "retrieved_docs" key with ranked list.
        top_k:            Maximum number of fused documents to return.

    Returns:
        dict: Updated state copy conforming to data contract:
            route_decision  = "hybrid"
            retrieved_docs  = top_k fused docs (RRF-ranked)
            scores          = list of rrf_score values (same order as retrieved_docs)
            rationale       = explanation of fusion behavior
    """
    query = state.get("query", "")

    # ── Extract ranked doc lists ───────────────────────────────
    sem_docs  = semantic_results.get("retrieved_docs", [])
    kw_docs   = keyword_results.get("retrieved_docs", [])

    # ── Apply RRF fusion (tie-breaking + source attribution inside) ────
    fused_all = reciprocal_rank_fusion([sem_docs, kw_docs], k=RRF_K)

    # ── Explicit top-k enforcement AFTER full sort ─────────────
    top_docs       = fused_all[:top_k]
    rrf_scores_out = [doc["rrf_score"] for doc in top_docs]

    # ── Diagnostic counts for enhanced rationale ───────────────
    sem_ids       = {d.get("id") for d in sem_docs}
    kw_ids        = {d.get("id") for d in kw_docs}
    overlap_count = len(sem_ids & kw_ids)            # docs that appear in BOTH lists

    # Count sources in the fused top-k
    source_counts = {"semantic": 0, "keyword": 0, "both": 0}
    for doc in top_docs:
        src = doc.get("source", "unknown")
        if src in source_counts:
            source_counts[src] += 1

    # ── Enhanced rationale (includes per-source counts) ────────
    rationale = (
        f"Hybrid search applied using RRF (k={RRF_K}). "
        f"{len(sem_docs)} semantic result(s), {len(kw_docs)} keyword result(s), "
        f"{overlap_count} overlapping document(s) boosted. "
        f"Top-{len(top_docs)} breakdown: "
        f"sem={source_counts['semantic']}, "
        f"kw={source_counts['keyword']}, "
        f"both={source_counts['both']}. "
        f"Top result source: '{top_docs[0].get('source', '?')}' — "
        f"\"{top_docs[0]['question'][:60]}\"."
        if top_docs else
        f"Hybrid search applied using RRF (k={RRF_K}). "
        f"No results returned ({len(sem_docs)} semantic, {len(kw_docs)} keyword)."
    )

    # ── Build output state (deepcopy — never mutate input) ─────
    new_state = copy.deepcopy(state)
    new_state["route_decision"] = "hybrid"
    new_state["retrieved_docs"] = top_docs
    new_state["scores"]         = rrf_scores_out
    new_state["rationale"]      = rationale

    return new_state
