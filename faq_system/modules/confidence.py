"""
confidence.py — Feature 3: Low Confidence Detection

Wraps pipeline results to detect ambiguous or low-quality retrieval.
Does NOT change any upstream module behaviour.

Step 4 fix: Route-adaptive thresholds replace flat 0.5 constant.
  semantic route → threshold 0.6  (embedding similarity must be strong)
  hybrid   route → threshold 0.4  (fused rank naturally lower)
  keyword  route → no threshold   (BM25 score scale differs; never flagged)

Public API:
    check_confidence(results, route_decision) -> (bool, str)
    get_query_type(route_decision)            -> str
"""

# Per-route adaptive thresholds (Step 4)
_THRESHOLDS = {
    "semantic": 0.6,
    "hybrid":   0.4,
    "keyword":  None,   # keyword route: never trigger threshold warning
}

# Maps route_decision -> human-readable query type
_QUERY_TYPE_MAP = {
    "keyword":  "code",
    "semantic": "conceptual",
    "hybrid":   "hybrid",
}


def check_confidence(results: list[dict],
                     route_decision: str) -> tuple[bool, str]:
    """
    Determine whether the retrieval result set has low confidence.

    Args:
        results:        List of result dicts from build_final_response().
        route_decision: Route string ("keyword", "semantic", "hybrid").

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
        return True, (
            f"⚠️ **Low confidence** — top score **{top_score:.4f}** is below the "
            f"**{threshold}** threshold for **{route}** retrieval. "
            f"Try rephrasing your query or using specific course codes."
        )

    return False, ""


def get_query_type(route_decision: str) -> str:
    """
    Map routing decision to a human-readable query type label.

    Returns: "code" | "conceptual" | "hybrid"
    """
    return _QUERY_TYPE_MAP.get(route_decision.lower(), "hybrid")
