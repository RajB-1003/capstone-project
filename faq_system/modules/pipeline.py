"""
pipeline.py — Phase 7: Full Pipeline Orchestration with Caching and Profiling

Responsibilities:
  - Orchestrate all Phase 1–6 modules in the correct order
  - Apply two-level cache check (exact → semantic) before running pipeline
  - Profile each stage's latency and attach `latency_ms` to the response
  - Update rationale with cache hit provenance when result is served from cache
  - Return the complete final response dict with latency info appended

Architecture:
  - This module DOES NOT modify any upstream module or data contract
  - It is the single integration point for the full pipeline
  - All downstream modules remain pure functions; this module wraps them

Execution order (cache-miss path):
  1. Exact cache lookup             O(1)
  2. embed_single()                 ~12ms  [used for semantic cache + semantic search]
  3. Semantic cache lookup          O(N), N≤20
  4. Tier 1 router (regex)          <1ms
  5. Tier 2 router (intent, sem.)   ~12ms
  6a. [keyword route]  search_keyword()       <2ms   ONLY
  6b. [semantic route] search_semantic()      ~12ms  ONLY
  6c. [hybrid route]   search_semantic()      ~12ms
                       search_keyword()       <2ms
                       search_hybrid() (RRF)  <1ms
  7. build_final_response()         <1ms
  8. Store in exact + semantic cache
  9. Attach latency_ms and return

Cache hit paths:
  Exact hit → skip steps 2–10, return immediately  (<1ms)
  Semantic hit → skip steps 4–10, return with updated rationale (~13ms)

Performance budget (architecture doc):
  Total < 2000ms
  Ideal < 300ms
  Cache hit < 5ms

Public API:
  run_pipeline(query, model, corpus_embeddings, faq_docs,
               bm25_index, patterns, intent_embeddings, top_k=5)
      -> dict   (final response + latency_ms field)
"""

import copy
import concurrent.futures

from modules.embedder        import embed_single
from modules.semantic_search import search_semantic
from modules.keyword_search  import search_keyword
from modules.hybrid_search   import search_hybrid
from modules.router_tier1    import tier1_route
from modules.router_tier2    import tier2_route
from modules.explainability  import build_final_response
from modules.cache           import (
    cache_lookup,
    cache_store,
    semantic_cache_lookup,
    semantic_cache_store,
)
from modules.profiler    import LatencyProfiler
from modules.confidence  import get_query_type, check_confidence


def run_pipeline(
    query: str,
    model,
    corpus_embeddings,
    faq_docs: list[dict],
    bm25_index,
    patterns: dict,
    intent_embeddings: dict,
    top_k: int = 5,
) -> dict:
    """
    Execute the full Semantic FAQ retrieval pipeline with two-level caching
    and per-stage latency profiling.

    Cache-first strategy:
        Exact hit  → return immediately (output unchanged, rationale updated)
        Semantic hit → return immediately (output unchanged, rationale updated)
        Miss       → run full pipeline → store in both caches → return

    Args:
        query:             User query string.
        model:             Loaded SentenceTransformer model instance.
        corpus_embeddings: (N, 384) float32 matrix, pre-built from FAQS.
        faq_docs:          List of FAQ dicts (loaded from faqs.json).
        bm25_index:        Pre-built BM25Okapi index.
        patterns:          Compiled regex patterns dict (from load_regex_patterns).
        intent_embeddings: Dict of intent→(M, 384) matrices (from embed_intents).
        top_k:             Number of documents to retrieve per search (default 5).

    Returns:
        dict: Final response conforming to build_final_response() schema,
              with two extra fields:
                "latency_ms": {stage: ms, ..., "total": ms}
                (rationale updated to mention cache provenance if applicable)
    """
    profiler = LatencyProfiler()

    # ── Step 1: Exact cache check ─────────────────────────────
    exact_hit = cache_lookup(query)
    if exact_hit is not None:
        exact_hit = copy.deepcopy(exact_hit)
        exact_hit["rationale"] = (
            "Result served from exact cache (fast retrieval). "
            + exact_hit.get("rationale", "")
        )
        exact_hit.setdefault("query_type", get_query_type(exact_hit.get("route_decision", "")))
        low_conf, warn_msg = check_confidence(exact_hit.get("results", []), exact_hit.get("route_decision", ""))
        exact_hit["low_confidence"] = low_conf
        exact_hit["confidence_warning"] = warn_msg
        exact_hit["latency_ms"] = {"cache": "exact", "total": profiler.elapsed_ms()}
        return exact_hit

    # ── Step 2: Embed query once (used for semantic cache + semantic search) ──
    with profiler.measure("embedding"):
        query_vec = embed_single(query, model)

    # ── Step 3: Semantic cache check ──────────────────────────
    semantic_hit = semantic_cache_lookup(query_vec)
    if semantic_hit is not None:
        result, similarity, matched_query = semantic_hit
        result = copy.deepcopy(result)
        result["rationale"] = (
            f"Result served from semantic cache (similarity {similarity:.4f} ≥ 0.98 "
            f"with cached query \"{matched_query[:60]}\"). "
            + result.get("rationale", "")
        )
        # Fix 1: apply confidence check (was missing from semantic cache path)
        _route = result.get("route_decision", "")
        result.setdefault("query_type", get_query_type(_route))
        _low_conf, _warn_msg = check_confidence(result.get("results", []), _route)
        result["low_confidence"]     = _low_conf
        result["confidence_warning"] = _warn_msg
        result["latency_ms"] = {
            "embedding": profiler._stages.get("embedding", 0),
            "cache":     "semantic",
            "total":     profiler.elapsed_ms(),
        }
        return result

    # ── Step 4: Build initial state ───────────────────────────
    state = {
        "query":             query,
        "detected_entities": [],
        "route_decision":    "",
        "retrieved_docs":    [],
        "scores":            [],
        "rationale":         "",
    }

    # ── Step 5: Tier 1 router (regex entity detection) ────────
    with profiler.measure("router_tier1"):
        state = tier1_route(state, patterns)

    # ── Step 6: Tier 2 router (semantic intent, only if T1 did not fire) ──
    if not state["route_decision"]:
        with profiler.measure("router_tier2"):
            state = tier2_route(state, intent_embeddings, model)

    route = state["route_decision"]

    # ── Step 7: Route-enforced retrieval ─────────────────────────
    # Only run the retriever(s) that the router selected.
    # keyword → BM25 only  |  semantic → embedding only  |  hybrid → both + RRF

    if route == "keyword":
        with profiler.measure("keyword_search"):
            kw_results  = search_keyword(query, faq_docs, bm25_index, top_k=top_k)
        state = kw_results   # kw_results is already a state dict
        state["route_decision"] = "keyword"  # preserve routing decision

    elif route == "semantic":
        with profiler.measure("semantic_search"):
            sem_results = search_semantic(
                query, faq_docs, corpus_embeddings, model, top_k=top_k
            )
        state = sem_results  # sem_results is already a state dict
        state["route_decision"] = "semantic"

    else:  # hybrid (default / fallback)
        state["route_decision"] = "hybrid"

        # Step 9: Run semantic + keyword in parallel (safe — both are GIL-releasing)
        def _run_semantic():
            return search_semantic(query, faq_docs, corpus_embeddings, model, top_k=top_k)

        def _run_keyword():
            return search_keyword(query, faq_docs, bm25_index, top_k=top_k)

        with profiler.measure("semantic_search"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f_sem = ex.submit(_run_semantic)
                f_kw  = ex.submit(_run_keyword)
                sem_results = f_sem.result()
                kw_results  = f_kw.result()
            # keyword latency captured inside the same block (wall-clock overlap)
            profiler._stages["keyword_search"] = profiler._stages.get("semantic_search", 0)

        # ── Hybrid RRF fusion ──────────────────────────────────
        with profiler.measure("hybrid_fusion"):
            state = search_hybrid(state, sem_results, kw_results, top_k=top_k)

    # ── Step 10: Explainability ───────────────────────────────
    with profiler.measure("explainability"):
        response = build_final_response(state)

    # ── Step 10b: Attach query_type and confidence metadata ───
    response["query_type"]         = get_query_type(response.get("route_decision", ""))
    low_conf, warn_msg             = check_confidence(response.get("results", []), response.get("route_decision", ""))
    response["low_confidence"]     = low_conf
    response["confidence_warning"] = warn_msg

    # ── Step 11: Store in both caches ─────────────────────────
    cache_store(query, response)
    semantic_cache_store(query_vec, query, response)

    # ── Step 12: Attach latency and return ────────────────────
    response = copy.deepcopy(response)
    profile  = profiler.get_profile()
    response["latency_ms"] = profile

    return response
