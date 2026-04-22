"""
pipeline.py — Phase 7: Full Pipeline Orchestration with Caching and Profiling
            (Composable signal-based routing added)

Responsibilities:
  - Orchestrate all Phase 1–6 modules in the correct order
  - Apply two-level cache check (exact → semantic) before running pipeline
  - Profile each stage’s latency and attach `latency_ms` to the response
  - Update rationale with cache hit provenance when result is served from cache
  - Return the complete final response dict with latency info appended

Routing model (composable, post-fix):
  Tier 1 → provides entity_signal   (bool: structured entity detected?)
  Tier 2 → provides semantic_score   (float: best intent similarity score)
  _fuse_signals() → combines both signals using existing thresholds:

      entity + score ≥ HIGH  → HYBRID   (both signals present — composable)
      entity + score <  HIGH  → KEYWORD  (entity dominates, semantic weak)
      no entity + score ≥ HIGH → SEMANTIC
      no entity + score ≥ LOW  → HYBRID
      no entity + score <  LOW  → HYBRID  (uncertain fallback)

  This removes the early-exit pattern where Tier 1 finalized routing
  and Tier 2 was skipped. Both tiers now ALWAYS run.

Architecture:
  - This module DOES NOT modify any upstream module or data contract
  - router_tier1 and router_tier2 are fully unchanged
  - _fuse_signals() is a module-private function (not exported)
  - routing_debug field is a non-breaking addition to all responses

Execution order (cache-miss path):
  1. Exact cache lookup             O(1)
  2. embed_single()                 ~12ms  [used for semantic cache + semantic search]
  3. Semantic cache lookup          O(N), N≤20
  4. Tier 1 router (regex)          <1ms   → produces entity_signal
  5. Tier 2 router (intent, sem.)   ~12ms  → ALWAYS runs, produces semantic_score
  6. _fuse_signals()                <1ms   → final route from combined signals
  7a. [keyword route]  search_keyword()       <2ms
  7b. [semantic route] search_semantic()      ~12ms
  7c. [hybrid route]   search_semantic()      ~12ms
                       search_keyword()       <2ms
                       search_hybrid() (RRF)  <1ms
  8. build_final_response()         <1ms
  9. Store in exact + semantic cache
  10. Attach latency_ms, routing_debug and return

Public API:
  run_pipeline(query, model, corpus_embeddings, faq_docs,
               bm25_index, patterns, intent_embeddings, top_k=5)
      -> dict   (final response + latency_ms + routing_debug fields)
"""

import copy
import concurrent.futures

from modules.embedder        import embed_single
from modules.semantic_search import search_semantic
from modules.keyword_search  import search_keyword
from modules.hybrid_search   import search_hybrid
from modules.router_tier1    import tier1_route
from modules.router_tier2    import tier2_route
from modules.constants       import THRESHOLD_HIGH, THRESHOLD_LOW

from modules.explainability  import build_final_response
from modules.cache           import (
    cache_lookup,
    cache_store,
    semantic_cache_lookup,
    semantic_cache_store,
)
from modules.profiler        import LatencyProfiler
from modules.confidence      import get_query_type, check_confidence
from modules.feedback_store  import apply_feedback_reranking
from modules.query_filter    import is_valid_unanswered_query   # Phase 2
from modules.db              import store_unanswered_query       # Phase 2
from modules.db              import log_query                    # Phase 5: analytics

import logging as _logging
_log = _logging.getLogger(__name__)


def _track_unanswered(query: str, low_confidence: bool) -> None:
    """
    Fire-and-forget helper: if the response was low-confidence and the
    query passes the quality gate, store it in unanswered_queries.

    Wrapped in try/except so a DB error never affects the response.
    """
    if not low_confidence:
        return
    try:
        if is_valid_unanswered_query(query):
            store_unanswered_query(query)
            # Fetch current count for the debug log (best-effort).
            from modules.db import get_connection as _gc
            conn = _gc()
            try:
                row = conn.execute(
                    "SELECT count FROM unanswered_queries WHERE query = ?",
                    (query.strip().lower(),),
                ).fetchone()
                cnt = row["count"] if row else "?"
            finally:
                conn.close()
            _log.debug(
                '[unanswered_query] stored: "%s" (count=%s)', query.strip(), cnt
            )
    except Exception as exc:  # noqa: BLE001
        _log.warning("[unanswered_query] storage error: %s", exc)


# ──────────────────────────────────────────────────────────────
# Signal fusion (module-private)
# Combines Tier 1 entity signal + Tier 2 semantic score into final route.
# Uses only the existing THRESHOLD_HIGH / THRESHOLD_LOW constants.
# Does NOT hardcode any language, entity name, or route string.
# ──────────────────────────────────────────────────────────────

def _fuse_signals(entity_detected: bool, semantic_score: float) -> tuple[str, str]:
    """
    Produce a final routing decision by combining both router signals.

    This is the ONLY place where the two tiers' outputs are reconciled.
    Neither tier finalizes the route on its own — they each contribute a
    signal; this function merges them.

    Decision table (uses existing THRESHOLD_HIGH=0.82, THRESHOLD_LOW=0.65):

        entity=True  + score >= LOW   →  hybrid
            A meaningful semantic signal exists alongside the structured entity.
            The query likely has intent beyond a pure lookup (e.g. mixed query).
            Both retrievers should contribute.

        entity=True  + score <  LOW   →  keyword
            Entity is the dominant signal; semantic similarity is too weak to
            add value — structured lookup alone is sufficient.

        entity=False + score >= HIGH  →  semantic
            No structured entity; high conceptual similarity — semantic only.

        entity=False + score >= LOW   →  hybrid
            Moderate semantic signal without an entity — ambiguous range,
            use both retrievers for coverage.

        entity=False + score <  LOW   →  semantic
            No entity and weak semantic signal.  Hybrid would add nothing
            useful here (no keyword anchor from an entity).  Route to semantic
            as the lowest-noise fallback; downstream confidence check will flag
            low-confidence results if retrieval quality is poor.

    Args:
        entity_detected: True if Tier 1 found at least one structured entity.
        semantic_score:  Best intent similarity score from Tier 2 (float in [0,1]).

    Returns:
        (route, reason) — route is one of "keyword", "semantic", "hybrid";
        reason is a human-readable explanation string for routing_debug.
    """
    if entity_detected:
        if semantic_score >= THRESHOLD_LOW:
            return (
                "hybrid",
                f"Fusion: entity detected + semantic score {semantic_score:.4f} "
                f">= {THRESHOLD_LOW} (LOW threshold) — entity + meaningful semantic "
                f"signal present; hybrid retrieval engaged.",
            )
        else:
            return (
                "keyword",
                f"Fusion: entity detected, semantic score {semantic_score:.4f} "
                f"< {THRESHOLD_LOW} — semantic signal too weak to contribute; "
                f"structured keyword lookup dominates.",
            )

    # No entity detected — routing driven purely by semantic score
    if semantic_score >= THRESHOLD_HIGH:
        return (
            "semantic",
            f"Fusion: no entity, semantic score {semantic_score:.4f} "
            f">= {THRESHOLD_HIGH} (HIGH threshold) — strong conceptual intent; "
            f"semantic retrieval only.",
        )

    if semantic_score >= THRESHOLD_LOW:
        return (
            "hybrid",
            f"Fusion: no entity, semantic score {semantic_score:.4f} in ambiguous "
            f"range [{THRESHOLD_LOW}, {THRESHOLD_HIGH}) — moderate semantic signal; "
            f"hybrid retrieval for coverage.",
        )

    # Below both thresholds, no entity — semantic fallback (not hybrid)
    # Hybrid would add no value here: BM25 has no keyword anchor without an entity.
    return (
        "semantic",
        f"Fusion: no entity, semantic score {semantic_score:.4f} "
        f"< {THRESHOLD_LOW} — weak signal, semantic fallback; "
        f"downstream confidence check will flag low-quality results.",

    )


def run_pipeline(
    query: str,
    user_id: int | None,
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
    exact_hit = cache_lookup(query, user_id)
    if exact_hit is not None:
        exact_hit = copy.deepcopy(exact_hit)
        exact_hit["rationale"] = (
            "[⚡ Served from your cache] Original routing decision reused. "
            + exact_hit.get("rationale", "")
        )
        exact_hit.setdefault("query_type", get_query_type(exact_hit.get("route_decision", "")))
        low_conf, warn_msg = check_confidence(exact_hit.get("results", []), exact_hit.get("route_decision", ""))
        exact_hit["low_confidence"] = low_conf
        exact_hit["confidence_warning"] = warn_msg
        exact_hit["latency_ms"]      = {"cache": "exact", "total": profiler.elapsed_ms()}
        exact_hit["cache_type"]      = "exact"
        exact_hit["cache_similarity"] = 1.0
        # Phase 2: reapply feedback reranking so stale cache rank reflects
        # votes cast AFTER the original query was cached.
        _cached_sem_score = (
            exact_hit.get("routing_debug", {}).get("semantic_score", 1.0)
        )
        exact_hit["results"] = apply_feedback_reranking(
            exact_hit.get("results", []),
            semantic_score=_cached_sem_score,
        )
        # Phase 2: track low-confidence queries even on cache hits
        _track_unanswered(query, exact_hit.get("low_confidence", False))
        # Phase 5: log analytics (cache hit path)
        _total_latency = exact_hit.get("latency_ms", {}).get("total", 0.0)
        if not isinstance(_total_latency, (int, float)):
            _total_latency = 0.0
        _top_conf = exact_hit.get("results", [{}])[0].get("score", 0.0) if exact_hit.get("results") else 0.0
        log_query(
            query=query,
            route="cached",
            confidence=_top_conf,
            latency=_total_latency,
            cache_hit=True,
        )
        return exact_hit

    # ── Step 2: Embed query once (used for semantic cache + semantic search) ──
    with profiler.measure("embedding"):
        query_vec = embed_single(query, model)

    # ── Step 3: Semantic cache check ──────────────────────────
    semantic_hit = semantic_cache_lookup(query_vec, user_id)
    if semantic_hit is not None:
        result, similarity, matched_query = semantic_hit
        result = copy.deepcopy(result)
        result["rationale"] = (
            f"[⚡ Served from your cache] Original routing decision reused. "
            f"(Semantic similarity {similarity:.4f} matched cached query \"{matched_query[:60]}\"). "
            + result.get("rationale", "")
        )
        _route = result.get("route_decision", "")
        result.setdefault("query_type", get_query_type(_route))
        _low_conf, _warn_msg = check_confidence(result.get("results", []), _route)
        result["low_confidence"]      = _low_conf
        result["confidence_warning"]  = _warn_msg
        result["latency_ms"]          = {
            "embedding": profiler._stages.get("embedding", 0),
            "cache":     "semantic",
            "total":     profiler.elapsed_ms(),
        }
        result["cache_type"]      = "semantic"
        result["cache_similarity"] = round(similarity, 4)
        # Phase 2: reapply feedback reranking so stale cache rank reflects
        # votes cast AFTER the original query was cached.
        _cached_sem_score = (
            result.get("routing_debug", {}).get("semantic_score", 1.0)
        )
        result["results"] = apply_feedback_reranking(
            result.get("results", []),
            semantic_score=_cached_sem_score,
        )
        # Phase 2: track low-confidence queries even on semantic cache hits
        _track_unanswered(query, result.get("low_confidence", False))
        # Phase 5: log analytics (semantic cache hit path)
        _total_latency = result.get("latency_ms", {}).get("total", 0.0)
        if not isinstance(_total_latency, (int, float)):
            _total_latency = 0.0
        _top_conf = result.get("results", [{}])[0].get("score", 0.0) if result.get("results") else 0.0
        log_query(
            query=query,
            route="cached",
            confidence=_top_conf,
            latency=_total_latency,
            cache_hit=True,
        )
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

    # Capture Tier 1 signal (entity presence) BEFORE Tier 2 runs.
    # tier1_route() sets route_decision="keyword" when entities are found;
    # we record this as a boolean signal without using it as a final decision.
    entity_detected: bool = bool(state.get("detected_entities"))

    # Clear route_decision so Tier 2 always runs unconditionally.
    # This removes the early-exit that previously skipped Tier 2 when T1 fired.
    state["route_decision"] = ""

    # ── Step 6: Tier 2 router (ALWAYS runs — produces semantic_score) ──
    with profiler.measure("router_tier2"):
        state = tier2_route(state, intent_embeddings, model)

    # Capture Tier 2 semantic score from the internal diagnostic field.
    semantic_score: float = float(state.get("_tier2_score", 0.0))

    # ── Step 6b: Signal fusion — final route from both signals ────
    route, decision_reason = _fuse_signals(entity_detected, semantic_score)
    state["route_decision"] = route

    # Compose rationale: preserve T1/T2 diagnostic rationale, then append fusion.
    t1_rationale = (
        f"Tier 1: entities detected: {state.get('detected_entities', [])}. "
        if entity_detected else "Tier 1: no structured entities. "
    )
    t2_rationale = state.get("rationale", "")
    state["rationale"] = t1_rationale + t2_rationale + " | " + decision_reason

    # Build routing_debug — non-breaking addition (does not alter any existing fields)
    routing_signals = {
        "entity_detected":  entity_detected,
        "semantic_score":   round(semantic_score, 4),
        "decision_reason":  decision_reason,
    }

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

    # ── Step 10c: Feedback reranking (Phase 2 — additive layer) ────────
    # Passes semantic_score for FIX 7 gate (skip adjustment on weak matches).
    # Does NOT modify routing, retrieval, or thresholds.
    # Adds F7/F8 debug fields: feedback_score, feedback_avg, feedback_count,
    # adjustment, adjusted_score, low_quality.
    response["results"] = apply_feedback_reranking(
        response.get("results", []),
        semantic_score=semantic_score,       # FIX 7: confidence gate
    )

    # ── Step 10b: Attach query_type, confidence, and routing_debug ───
    # Build routing_debug with full trace (Upgrade 6)
    routing_signals = {
        "entity_detected":  entity_detected,
        "semantic_score":   round(semantic_score, 4),
        "thresholds":       {"low": THRESHOLD_LOW, "high": THRESHOLD_HIGH},
        "final_route":      route,
        "decision_reason":  decision_reason,
    }
    response["query_type"]         = get_query_type(response.get("route_decision", ""))
    low_conf, warn_msg             = check_confidence(
        response.get("results", []),
        response.get("route_decision", ""),
        routing_debug=routing_signals,           # Upgrade 4: pass full trace
    )
    response["low_confidence"]     = low_conf
    response["confidence_warning"] = warn_msg
    response["routing_debug"]      = routing_signals   # non-breaking new field

    # ── Phase 2: Unanswered query tracking ────────────────────────────────
    # Runs AFTER confidence is determined; never blocks or alters response.
    _track_unanswered(query, low_conf)

    # ── Step 11: Store in both caches ─────────────────────────
    cache_store(query, response, user_id)
    semantic_cache_store(query_vec, query, response, user_id)

    # ── Step 12: Attach latency, cache debug info, and return ─
    response = copy.deepcopy(response)
    profile  = profiler.get_profile()
    response["latency_ms"]       = profile
    response["cache_type"]       = "miss"  # debug: full pipeline was executed
    response["cache_similarity"] = 0.0     # debug: no cache match

    # ── Phase 5: Log analytics (full pipeline path) ────────────
    _p5_total   = profile.get("total", 0.0)
    _p5_results = response.get("results", [])
    _p5_conf    = _p5_results[0].get("score", 0.0) if _p5_results else 0.0
    log_query(
        query=query,
        route=response.get("route_decision", route),
        confidence=_p5_conf,
        latency=_p5_total,
        cache_hit=False,
    )

    return response
