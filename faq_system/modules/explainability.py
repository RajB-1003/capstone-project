"""
explainability.py — Phase 6: Explainability Layer

Responsibilities:
  - Generate transparent, human-readable explanations for routing decisions
    and individual retrieval results — using ONLY actual system outputs
  - Enhance each retrieved document with an "explanation" field
  - Produce a clean, structured final response ready for UI or API consumption

Architecture mandates (Phased Agent Prompt Chain Design.docx):
  - Explanations are derived exclusively from state fields:
      route_decision, scores, detected_entities, source, _tier2_intent, _tier2_score
  - No LLM reasoning, no attention maps, no token highlighting
  - Deterministic: identical state → identical explanation every time
  - Read-only: this module NEVER modifies upstream modules or the source state

Design decisions:
  - generate_rationale() produces a NEW user-facing rationale from scratch.
    It reads state["rationale"] as context but rewrites it in cleaner English.
    The upstream rationale (set by search modules) is kept for debugging.
  - explain_results() infers "source" for pure-route docs (semantic/keyword)
    when the source field is absent (only hybrid results carry it from Phase 5).
  - build_final_response() is the single public entry point for Phase 7+.
    It calls generate_rationale() and explain_results() internally.
  - All explanation strings are template-driven — no string concatenation loops.

Explanation templates (deterministic, grounded in actual signal):
    KEYWORD route:  entity names + bypass rationale
    SEMANTIC route: intent label + similarity score + threshold
    HYBRID route:   score range + both-retriever rationale + overlap count

Function signatures (defined before implementation — architecture mandate):

  generate_rationale(state: dict) -> str
      Build a clean routing explanation from scored state fields.

  explain_results(state: dict) -> dict
      Return state copy with "explanation" added to each retrieved_doc.

  build_final_response(state: dict) -> dict
      Produce the final structured output combining route + enriched results.

Public API:
  generate_rationale(state)    -> str
  explain_results(state)       -> dict
  build_final_response(state)  -> dict
"""

import copy


# ──────────────────────────────────────────────────────────────
# Threshold constants (must match router_tier2.py values)
# ──────────────────────────────────────────────────────────────
_THRESHOLD_HIGH = 0.82
_THRESHOLD_LOW  = 0.65


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _infer_doc_source(doc: dict, route_decision: str) -> str:
    """
    Return the source label for a document.
    Phase 5 (hybrid) sets doc["source"] explicitly.
    Pure semantic / keyword routes don't set it — infer from route.
    """
    src = doc.get("source")
    if src in ("semantic", "keyword", "both"):
        return src
    # Infer from route for pure-route results
    if route_decision == "semantic":
        return "semantic"
    if route_decision == "keyword":
        return "keyword"
    return "unknown"


def _best_score(doc: dict, fallback: float = 0.0) -> tuple[float, str]:
    """
    Return the most informative per-doc score and its label.
    Priority: similarity_score (semantic) > bm25_score (keyword) > rrf_score (hybrid).
    """
    if "similarity_score" in doc:
        return round(float(doc["similarity_score"]), 4), "similarity"
    if "bm25_score" in doc:
        return round(float(doc["bm25_score"]), 4), "BM25"
    if "rrf_score" in doc:
        return round(float(doc["rrf_score"]), 6), "RRF"
    return round(fallback, 4), "score"


# ──────────────────────────────────────────────────────────────
# 1. Routing rationale generator
# ──────────────────────────────────────────────────────────────

def generate_rationale(state: dict) -> str:
    """
    Generate a clean, user-facing explanation of the routing decision.

    Reads ONLY from state fields — no LLM, no inference, no invention.
    Deterministic: same state dict → same output string every time.

    Routing logic covered:
        "keyword"  → Tier 1 regex triggered; entity names listed
        "semantic" → Tier 2 high-similarity; intent + score cited
        "hybrid"   → Tier 2 mid-range score; both retrievers used
        other/""   → Fallback / unresolved routing

    Retrieval summary appended for all routes:
        - Top score from the result list
        - Source distribution (how many semantic / keyword / both)
        - Overlap count (for hybrid)

    Args:
        state: Shared data contract dict (must have route_decision).

    Returns:
        str: Clear, concise rationale string (2–4 sentences).
             Never empty, never contains placeholder text.
    """
    route      = state.get("route_decision", "")
    entities   = state.get("detected_entities", [])
    scores     = state.get("scores", [])
    docs       = state.get("retrieved_docs", [])

    # Tier 2 diagnostic fields (set by router_tier2.py when it ran)
    t2_intent  = state.get("_tier2_intent", "")
    t2_score   = state.get("_tier2_score",  None)

    top_score  = round(scores[0], 4) if scores else None

    # ── Part A: Routing explanation ────────────────────────────

    if route == "keyword":
        entity_str = ", ".join(entities) if entities else "structured identifier(s)"
        routing_text = (
            f"Keyword search activated. "
            f"Structured identifier(s) detected in query: [{entity_str}]. "
            f"Tier 1 regex router bypassed semantic evaluation to ensure "
            f"exact token matching for course codes and identifiers."
        )

    elif route == "semantic":
        if t2_intent and t2_score is not None:
            routing_text = (
                f"Semantic search activated. "
                f"Query matched '{t2_intent}' intent with similarity {t2_score:.4f} "
                f"(threshold: {_THRESHOLD_HIGH}). "
                f"Tier 2 intent router directed the query to the embedding-based retriever."
            )
        elif top_score is not None:
            routing_text = (
                f"Semantic search activated. "
                f"Query similarity {top_score:.4f} exceeded the high-confidence threshold "
                f"({_THRESHOLD_HIGH}). "
                f"Embedding-based retrieval used for conceptual matching."
            )
        else:
            routing_text = (
                "Semantic search activated via Tier 2 intent classification. "
                "Query matched a known conceptual or informational intent pattern."
            )

    elif route == "hybrid":
        if t2_score is not None:
            score_context = (
                f"Query similarity {t2_score:.4f} falls in the ambiguous range "
                f"[{_THRESHOLD_LOW}, {_THRESHOLD_HIGH}). "
            )
        else:
            score_context = (
                "Query did not meet the high-confidence semantic threshold. "
            )
        routing_text = (
            score_context +
            "Hybrid search deployed: both semantic and keyword retrievers were used. "
            "Results combined using Reciprocal Rank Fusion (RRF, k=60) for "
            "rank-based score fusion without mixing raw similarity and BM25 values."
        )

    else:
        routing_text = (
            "Routing decision not set. "
            "Query may require manual review or a fallback retrieval strategy."
        )

    # ── Part B: Retrieval summary ──────────────────────────────

    if not docs:
        retrieval_text = "No documents were retrieved for this query."
    else:
        # Count sources across all retrieved docs
        source_counts = {"semantic": 0, "keyword": 0, "both": 0, "unknown": 0}
        for doc in docs:
            src = _infer_doc_source(doc, route)
            source_counts[src] = source_counts.get(src, 0) + 1

        overlap_count = source_counts.get("both", 0)

        # Build source breakdown string
        breakdown_parts = []
        if source_counts["both"]     > 0:
            breakdown_parts.append(f"{source_counts['both']} from both retrievers (boosted)")
        if source_counts["semantic"] > 0:
            breakdown_parts.append(f"{source_counts['semantic']} semantic-only")
        if source_counts["keyword"]  > 0:
            breakdown_parts.append(f"{source_counts['keyword']} keyword-only")

        breakdown_str = "; ".join(breakdown_parts) if breakdown_parts else f"{len(docs)} result(s)"

        score_label = f"Top score: {top_score:.4f}. " if top_score is not None else ""

        if route == "hybrid" and overlap_count > 0:
            retrieval_text = (
                f"{score_label}"
                f"Retrieved {len(docs)} document(s): {breakdown_str}. "
                f"{overlap_count} document(s) appeared in both retrieval lists, "
                f"indicating strong relevance."
            )
        else:
            retrieval_text = (
                f"{score_label}"
                f"Retrieved {len(docs)} document(s): {breakdown_str}."
            )

    return f"{routing_text} {retrieval_text}"


# ──────────────────────────────────────────────────────────────
# 2. Per-document explainer
# ──────────────────────────────────────────────────────────────

def explain_results(state: dict) -> dict:
    """
    Enhance each retrieved document with an "explanation" field.

    Explanation is grounded in:
        - source: how the document was retrieved
        - rank position: relative confidence ordering
        - score: numeric retrieval signal (similarity / BM25 / RRF)
        - category: domain hint from FAQ metadata

    Never modifies the input state. Returns a new state copy.

    Per-source explanation templates (deterministic):
        "semantic" → "Retrieved via semantic similarity — content matches
                       query intent conceptually (rank #N, score: X.XXXX)."
        "keyword"  → "Retrieved via keyword match — document contains query
                       terms or identifiers directly (rank #N, BM25: X.XXXX)."
        "both"     → "Strong match in both semantic and keyword search
                       (rank #N, RRF: X.XXXXXX). Appeared in both retrieval
                       lists — highest confidence result."
        "unknown"  → "Retrieved with unresolved source attribution
                       (rank #N, score: X.XXXX)."

    Args:
        state: Shared data contract dict with retrieved_docs.

    Returns:
        dict: Deep copy of state with "explanation" and "source" fields
              injected into every doc in retrieved_docs.
    """
    new_state = copy.deepcopy(state)
    route     = state.get("route_decision", "")
    scores    = state.get("scores", [])
    docs      = new_state.get("retrieved_docs", [])

    for rank, doc in enumerate(docs, start=1):
        source = _infer_doc_source(doc, route)
        doc["source"] = source    # Ensure source field is always present

        # Best per-doc score — use fallback from state scores list if needed
        fallback = scores[rank - 1] if rank - 1 < len(scores) else 0.0
        score_val, score_label = _best_score(doc, fallback=fallback)

        category = doc.get("category", "")
        cat_hint = f" [{category}]" if category else ""

        if source == "semantic":
            explanation = (
                f"Retrieved via semantic similarity — content matches query intent "
                f"conceptually{cat_hint} (rank #{rank}, {score_label}: {score_val})."
            )
        elif source == "keyword":
            explanation = (
                f"Retrieved via keyword match — document contains query terms "
                f"or identifiers directly{cat_hint} (rank #{rank}, {score_label}: {score_val})."
            )
        elif source == "both":
            explanation = (
                f"Strong match in both semantic and keyword search{cat_hint} "
                f"(rank #{rank}, {score_label}: {score_val}). "
                f"Appeared in both retrieval lists — highest confidence result."
            )
        else:
            explanation = (
                f"Retrieved with unresolved source attribution{cat_hint} "
                f"(rank #{rank}, {score_label}: {score_val})."
            )

        doc["explanation"] = explanation

    new_state["retrieved_docs"] = docs
    return new_state


# ──────────────────────────────────────────────────────────────
# 3. Final response builder
# ──────────────────────────────────────────────────────────────

def build_final_response(state: dict) -> dict:
    """
    Produce the final structured output, combining routing rationale and
    enriched, explained result documents.

    This is the single public entry point for UI / API consumption.
    Internally calls generate_rationale() and explain_results().

    Output schema:
        {
            "query":          str,
            "route_decision": str,
            "rationale":      str,    # clean user-facing explanation
            "results": [
                {
                    "rank":        int,
                    "question":    str,
                    "answer":      str,
                    "category":    str,
                    "score":       float,
                    "source":      str,   # "semantic" | "keyword" | "both"
                    "explanation": str    # why this doc was retrieved
                },
                ...
            ]
        }

    Rules:
        - Only fields listed above are included in results (no leaking of
          internal fields like rrf_score, bm25_score to final output)
        - score = most informative per-doc score (similarity / BM25 / RRF)
        - Explanation is never empty
        - rationale is generated fresh, not taken from state["rationale"]

    Args:
        state: Shared data contract dict (may be output of any Phase 1-5 module).

    Returns:
        dict: Final structured response ready for Streamlit / API layer.
    """
    # ── Step 1: enrich docs with explanations ─────────────────
    enriched_state = explain_results(state)

    # ── Step 2: generate clean routing rationale ───────────────
    rationale = generate_rationale(enriched_state)

    # ── Step 3: build clean results list ──────────────────────
    results = []
    scores  = enriched_state.get("scores", [])

    for rank, doc in enumerate(enriched_state.get("retrieved_docs", []), start=1):
        route = enriched_state.get("route_decision", "")
        score_val, _ = _best_score(
            doc,
            fallback=scores[rank - 1] if rank - 1 < len(scores) else 0.0,
        )
        results.append({
            "rank":        rank,
            "question":    doc.get("question", ""),
            "answer":      doc.get("answer", ""),
            "category":    doc.get("category", ""),
            "tags":        doc.get("tags", []),          # Step 8: standardised field
            "score":       score_val,
            "source":      doc.get("source", _infer_doc_source(doc, route)),
            "explanation": doc.get("explanation", ""),
        })

    # ── Step 4: return final structured response ───────────────
    return {
        "query":          enriched_state.get("query", ""),
        "route_decision": enriched_state.get("route_decision", ""),
        "rationale":      rationale,
        "results":        results,
    }
