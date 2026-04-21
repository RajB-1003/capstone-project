"""
confidence.py — Feature 3: Low Confidence Detection
               Upgrade 4: Actionable confidence messages

Wraps pipeline results to detect ambiguous or low-quality retrieval.
Does NOT change any upstream module behaviour.

Issue 6 fix: Low-confidence threshold aligned to THRESHOLD_LOW (0.65)
  semantic route → threshold 0.65  (must import from constants — no magic numbers)
  hybrid   route → threshold 0.40  (fused RRF rank naturally produces lower scores)
  keyword  route → no threshold    (BM25 score scale differs; never flagged)

Issue 2 fix: Imports THRESHOLD_LOW from constants.py instead of duplicating 0.65.

Upgrade 4: Low-confidence warning now includes specific, query-derived suggestions:
  - "Add course code" — shown when entity_detected=False (from routing_debug)
  - "Specify a topic" — always shown (generic but accurate)
  - "Rephrase as a question" — shown when score < THRESHOLD_LOW / 2 (very weak)
  Suggestions are not hardcoded per-query; they derive from signal characteristics.

Public API:
    check_confidence(results, route_decision, routing_debug=None) -> (bool, str)
    get_query_type(route_decision)            -> str
"""

from modules.constants import THRESHOLD_LOW

# Per-route adaptive thresholds
# semantic uses THRESHOLD_LOW (0.65) — aligns with routing signal boundary.
# hybrid uses 0.40 — RRF rank-fusion produces lower raw scores by design.
# keyword is None — BM25 scores are unbounded and not comparable to [0,1] scale.
_THRESHOLDS = {
    "semantic": THRESHOLD_LOW,
    "hybrid":   0.40,
    "keyword":  None,   # keyword route: never trigger threshold warning
}

# Maps route_decision -> human-readable query type
_QUERY_TYPE_MAP = {
    "keyword":  "code",
    "semantic": "conceptual",
    "hybrid":   "hybrid",
}


def check_confidence(
    results: list[dict],
    route_decision: str,
    routing_debug: dict | None = None,
) -> tuple[bool, str]:
    """
    Determine whether the retrieval result set has low confidence.

    Args:
        results:        List of result dicts from build_final_response().
        route_decision: Route string ("keyword", "semantic", "hybrid").
        routing_debug:  Optional routing signal dict (entity_detected,
                        semantic_score). When provided, suggestions in the
                        warning message are tailored to signal characteristics.

    Returns:
        (is_low_confidence: bool, warning_message: str)
        If is_low_confidence is False, warning_message is "".
    """
    if not results:
        return True, (
            "⚠️ **No results found.** Try rephrasing your query or "
            "using specific course codes (e.g., CS-202)."
        )

    # Keyword route: BM25 scores are unbounded — never apply a threshold
    route = route_decision.lower()
    threshold = _THRESHOLDS.get(route)
    if threshold is None:
        return False, ""

    top_score = float(results[0].get("score", 0))

    if top_score < threshold:
        # ── Build dynamic suggestions from signal characteristics ─
        debug         = routing_debug or {}
        entity_detected = debug.get("entity_detected", False)
        semantic_score  = debug.get("semantic_score", top_score)

        suggestions = []

        # Suggestion 1: course code — only when no entity was detected
        if not entity_detected:
            suggestions.append(
                "• Include a course code (e.g., CS-202, CS-301) to enable "
                "precise keyword matching"
            )

        # Suggestion 2: topic specification — always useful
        suggestions.append(
            "• Specify a concrete topic (attendance, exam, eligibility, "
            "hostel fee, scholarship)"
        )

        # Suggestion 3: rephrase — when signal is very weak (< half threshold)
        if semantic_score < (THRESHOLD_LOW / 2):
            suggestions.append(
                "• Rephrase as a complete question for better semantic matching "
                "(e.g., \"What is the attendance policy?\")"
            )

        suggestion_block = "\n" + "\n".join(suggestions) if suggestions else ""

        return True, (
            f"⚠️ **Low confidence** — score **{top_score:.4f}** is below "
            f"the **{threshold}** threshold for **{route}** retrieval.\n"
            f"**Suggestions:**{suggestion_block}"
        )

    return False, ""


def get_query_type(route_decision: str) -> str:
    """
    Map routing decision to a human-readable query type label.

    Returns: "code" | "conceptual" | "hybrid"
    """
    return _QUERY_TYPE_MAP.get(route_decision.lower(), "hybrid")
