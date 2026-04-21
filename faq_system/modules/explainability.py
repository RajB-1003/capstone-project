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

from modules.constants import THRESHOLD_HIGH, THRESHOLD_LOW



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
# Upgrade 1 helper: Structured reasoning trace
# ──────────────────────────────────────────────────────────────

def build_reasoning_trace(routing_debug: dict, state: dict) -> dict:
    """
    Build a structured reasoning trace explaining WHY the chosen route
    is optimal vs the alternatives.

    Returns:
        {
          "decision":  str,         # final route (KEYWORD / SEMANTIC / HYBRID)
          "reasoning": list[str],   # 3 ordered reasoning bullets
        }

    All text is dynamically constructed from routing_debug signals and
    threshold constants. No phrases are hardcoded.
    """
    route           = state.get("route_decision", "").upper()
    entities        = state.get("detected_entities", [])
    routing_debug   = routing_debug or {}
    entity_detected = routing_debug.get("entity_detected", bool(entities))
    semantic_score  = routing_debug.get("semantic_score", 0.0)

    reasoning: list[str] = []

    # ── Bullet 1: Entity signal ────────────────────────────────
    if entity_detected:
        entity_str = ", ".join(f"'{e}'" for e in entities) if entities else "a structured identifier"
        reasoning.append(
            f"Entity {entity_str} detected → enables precise keyword matching "
            f"by anchoring the query to a specific structured token"
        )
    else:
        reasoning.append(
            f"No structured entity detected → routing relies entirely on "
            f"semantic similarity signal ({semantic_score:.4f})"
        )

    # ── Bullet 2: Semantic signal interpretation ───────────────
    if semantic_score >= THRESHOLD_HIGH:
        level = "high"
        interp = (f"score {semantic_score:.4f} ≥ {THRESHOLD_HIGH} "
                  f"→ strong conceptual match with a known intent category")
    elif semantic_score >= THRESHOLD_LOW:
        level = "medium"
        interp = (f"score {semantic_score:.4f} in [{THRESHOLD_LOW}, {THRESHOLD_HIGH}) "
                  f"→ moderate conceptual intent, too ambiguous for pure semantic")
    else:
        level = "low"
        interp = (f"score {semantic_score:.4f} < {THRESHOLD_LOW} "
                  f"→ weak semantic signal, no reliable intent match")
    reasoning.append(f"Semantic similarity: {level} — {interp}")

    # ── Bullet 3: Why this route, not the alternatives ─────────
    route_lower = route.lower()
    if route_lower == "hybrid":
        if entity_detected:
            reasoning.append(
                f"HYBRID chosen: entity provides keyword anchor + semantic score "
                f"{semantic_score:.4f} ≥ {THRESHOLD_LOW} adds conceptual coverage — "
                f"pure KEYWORD would miss related conceptual matches; "
                f"pure SEMANTIC would ignore the entity's exact lookup value"
            )
        else:
            reasoning.append(
                f"HYBRID chosen: ambiguous range [{THRESHOLD_LOW}, {THRESHOLD_HIGH}) — "
                f"pure SEMANTIC risks missing keyword-anchored FAQs; "
                f"pure KEYWORD has no entity anchor — both retrievers provide coverage"
            )
    elif route_lower == "keyword":
        reasoning.append(
            f"KEYWORD chosen: entity present but semantic score {semantic_score:.4f} "
            f"< {THRESHOLD_LOW} — semantic signal too weak to add value; "
            f"SEMANTIC alone would retrieve unrelated conceptual matches"
        )
    else:  # semantic
        if semantic_score >= THRESHOLD_HIGH:
            reasoning.append(
                f"SEMANTIC chosen: score {semantic_score:.4f} ≥ {THRESHOLD_HIGH} — "
                f"high-confidence conceptual intent; no entity, so KEYWORD would "
                f"produce no relevant BM25 anchor matches"
            )
        else:
            reasoning.append(
                f"SEMANTIC fallback: score {semantic_score:.4f} < {THRESHOLD_LOW} — "
                f"HYBRID would add no value (no keyword anchor); "
                f"downstream confidence check will flag if results are poor"
            )

    return {"decision": route, "reasoning": reasoning}


def generate_rationale(state: dict) -> str:
    """
    Generate a clean, user-facing explanation of the routing decision.

    Upgrade 1: Appends a 'Reasoning' block with 3 structured bullets
    explaining entity signal, semantic signal interpretation, and why
    the chosen route is optimal vs alternatives.  All text is derived
    dynamically from routing_debug and threshold constants.

    Reads ONLY from state/response fields — no LLM, no inference, no invention.
    Deterministic: same state dict → same output string every time.

    Args:
        state: Shared data contract dict (must have route_decision).
               routing_debug is optional — falls back gracefully.

    Returns:
        str: Clear, concise rationale string with structured reasoning.
             Never empty, never contains placeholder text.
    """
    route    = state.get("route_decision", "")
    entities = state.get("detected_entities", [])
    scores   = state.get("scores", [])
    docs     = state.get("retrieved_docs", [])

    # Tier 2 diagnostic fields (set by router_tier2.py when it ran)
    t2_intent = state.get("_tier2_intent", "")
    t2_score  = state.get("_tier2_score", None)

    top_score = round(scores[0], 4) if scores else None

    # routing_debug carries the actual fusion signals (entity + semantic_score)
    # so the explanation can mirror _fuse_signals() logic exactly.
    routing_debug   = state.get("routing_debug", {})
    entity_detected = routing_debug.get("entity_detected", bool(entities))
    semantic_score  = routing_debug.get("semantic_score",  t2_score)

    # ── Part A: Routing explanation (unchanged from previous version) ───────
    if entity_detected:
        entity_str = ", ".join(entities) if entities else "structured identifier(s)"
        if semantic_score is not None and semantic_score >= THRESHOLD_LOW:
            routing_text = (
                f"Hybrid search activated. "
                f"Structured identifier(s) detected: [{entity_str}]. "
                f"Semantic similarity {semantic_score:.4f} \u2265 {THRESHOLD_LOW} (meaningful signal) \u2014 "
                f"both keyword and semantic retrievers used to cover exact and conceptual matches."
            )
        else:
            score_note = (
                f" Semantic similarity {semantic_score:.4f} is below {THRESHOLD_LOW}"
                f" \u2014 structured lookup dominates."
                if semantic_score is not None else ""
            )
            routing_text = (
                f"Keyword search activated. "
                f"Structured identifier(s) detected in query: [{entity_str}]."
                f"{score_note} "
                f"Tier 1 regex router directed this query to exact token matching."
            )
    else:
        if semantic_score is not None and semantic_score >= THRESHOLD_HIGH:
            intent_note = f"'{t2_intent}' intent " if t2_intent else ""
            routing_text = (
                f"Semantic search activated. "
                f"Query matched {intent_note}with similarity {semantic_score:.4f} "
                f"\u2265 {THRESHOLD_HIGH} (high-confidence threshold). "
                f"Embedding-based retriever used for conceptual matching."
            )
        elif semantic_score is not None and semantic_score >= THRESHOLD_LOW:
            intent_note = f"'{t2_intent}' " if t2_intent else ""
            routing_text = (
                f"Hybrid search activated. "
                f"Semantic similarity {semantic_score:.4f} falls in the ambiguous range "
                f"[{THRESHOLD_LOW}, {THRESHOLD_HIGH}) for {intent_note}intent. "
                f"Both keyword and semantic retrievers used; results fused via RRF."
            )
        else:
            score_note = (
                f" Similarity {semantic_score:.4f} is below {THRESHOLD_LOW}."
                if semantic_score is not None else ""
            )
            routing_text = (
                f"Semantic search activated (low-signal fallback). "
                f"No structured entity detected."
                f"{score_note} "
                f"Semantic retriever used; downstream confidence check applies."
            )

    # ── Part B: Retrieval summary ───────────────────────────────
    if not docs:
        retrieval_text = "No documents were retrieved for this query."
    else:
        source_counts = {"semantic": 0, "keyword": 0, "both": 0, "unknown": 0}
        for doc in docs:
            src = _infer_doc_source(doc, route)
            source_counts[src] = source_counts.get(src, 0) + 1

        overlap_count = source_counts.get("both", 0)

        breakdown_parts = []
        if source_counts["both"]     > 0:
            breakdown_parts.append(f"{source_counts['both']} from both retrievers (boosted)")
        if source_counts["semantic"] > 0:
            breakdown_parts.append(f"{source_counts['semantic']} semantic-only")
        if source_counts["keyword"]  > 0:
            breakdown_parts.append(f"{source_counts['keyword']} keyword-only")

        breakdown_str = "; ".join(breakdown_parts) if breakdown_parts else f"{len(docs)} result(s)"
        score_label   = f"Top score: {top_score:.4f}. " if top_score is not None else ""

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

    # ── Part C: Structured reasoning block (Upgrade 1) ─────────────────
    trace = build_reasoning_trace(routing_debug, state)
    reasoning_lines = "\n".join(f"  • {r}" for r in trace["reasoning"])
    reasoning_block = f"\n\n**Reasoning:**\n{reasoning_lines}"

    return f"{routing_text} {retrieval_text}{reasoning_block}"


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
            "faq_id":      doc.get("id", ""),            # Phase 2: needed by feedback reranking
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
