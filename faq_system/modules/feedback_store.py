"""
feedback_store.py — Phase 2: Feedback System (Refinement pass)

Refinements applied (all additive — no upstream modules modified):

  FIX 1: Normalised scoring
      avg_feedback = weighted_sum / max(count, 1)
      adjustment   = avg_feedback * weight
      → prevents runaway scores from high event counts

  FIX 2: Score clamping
      adjusted_score = max(0.0, min(1.0, adjusted_score))
      → keeps scores in valid [0, 1] range, ensures ranking stability

  FIX 3: Differentiated not_helpful penalty
      not_helpful events carry NOT_HELPFUL_MULTIPLIER (1.5×) when building
      weighted_sum; low_quality also uses weighted_sum so detection
      accelerates when not_helpful dominates.

  FIX 4: Improved tag extraction
      Tokens must be: length > 3, non-numeric, not in stopwords,
      not already present in the FAQ's own text, deduplicated.
      Returns top-5 max.

  FIX 5: Query-level feedback memory
      modules/query_feedback.json tracks (query, faq_id, feedback, count)
      for repeated-failure detection.

  FIX 6: Feedback decay (simplified window)
      Only the last DECAY_WINDOW (20) entries per FAQ are considered.
      Older feedback is excluded, so recent signals dominate.

  FIX 7: Feedback gate on semantic confidence
      apply_feedback_reranking() accepts semantic_score parameter.
      If semantic_score < THRESHOLD_LOW (0.65), adjustment is skipped;
      feedback must not amplify a weak/wrong match.

  FIX 8: Extended debug output per result
      { feedback_score, feedback_avg, feedback_count,
        adjustment, adjusted_score, low_quality }

Public API (backward-compatible)
---------------------------------
  store_feedback(faq_id, query, route, score, feedback)  -> None
  get_aggregated_scores()                                -> dict[str, dict]
  get_feedback_score(faq_id)                             -> int
  get_poor_faqs(score_threshold, count_threshold)        -> list[str]
  is_low_quality(faq_id)                                 -> bool
  get_suggested_tags(faq_id, top_k)                      -> list[str]
  apply_feedback_reranking(results, weight,
                           semantic_score)               -> list[dict]
  store_query_feedback(query, faq_id, feedback)          -> None   [FIX 5]
  get_query_feedback(faq_id, query)                      -> list   [FIX 5]
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Path constants
# ──────────────────────────────────────────────────────────────

_MODULES_DIR      = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR         = os.path.join(_MODULES_DIR, "..", "data")

FEEDBACK_LOG_PATH    = os.path.join(_DATA_DIR, "feedback_log.jsonl")
QUERY_FEEDBACK_PATH  = os.path.join(_MODULES_DIR, "query_feedback.json")   # FIX 5

# ──────────────────────────────────────────────────────────────
# Scoring constants (FIX 3)
# ──────────────────────────────────────────────────────────────

SCORE_MAP: dict[str, int] = {
    "up":          +1,
    "down":        -1,
    "not_helpful": -2,
}

NOT_HELPFUL_MULTIPLIER: float = 1.5
"""
not_helpful events are multiplied by this factor when building weighted_sum.
This accelerates both rank penalty and low_quality detection for completely
wrong answers, without changing the raw SCORE_MAP values used externally.
"""

# ──────────────────────────────────────────────────────────────
# Detection thresholds (F5)
# ──────────────────────────────────────────────────────────────

POOR_SCORE_THRESHOLD: float = -5.0   # weighted_sum must be below this
POOR_COUNT_THRESHOLD: int   =  3     # AND count >= this (>= for FIX 3 acceleration)

# ──────────────────────────────────────────────────────────────
# Reranking (FIX 1, FIX 2)
# ──────────────────────────────────────────────────────────────

RERANK_WEIGHT: float = 0.1
"""
Multiplier applied to avg_feedback when computing adjustment.
Temporarily raised to 0.1 (from 0.05) for visible ranking validation.
Revert to 0.05 once feedback effect is confirmed in production.
Small by design: feedback nudges rank, never overrides retrieval.
"""

# ──────────────────────────────────────────────────────────────
# Decay window (FIX 6)
# ──────────────────────────────────────────────────────────────

DECAY_WINDOW: int = 20
"""
Maximum number of recent feedback entries considered per FAQ.
Entries beyond this window are ignored so old signals fade out naturally.
"""

# ──────────────────────────────────────────────────────────────
# Routing gate (FIX 7)
# ──────────────────────────────────────────────────────────────

# Imported lazily inside apply_feedback_reranking to avoid circular imports.
# The actual value is 0.65 — matches THRESHOLD_LOW in constants.py.
_THRESHOLD_LOW_FALLBACK: float = 0.65

# ──────────────────────────────────────────────────────────────
# Tag suggestion constants (FIX 4)
# ──────────────────────────────────────────────────────────────

TAG_TOP_K: int = 5
TAG_MIN_LENGTH: int = 4   # FIX 4: length > 3 means minimum 4 chars

_STOPWORDS: frozenset[str] = frozenset({
    "what", "how", "when", "where", "why", "who", "which", "can", "cannot",
    "does", "will", "would", "should", "could", "may", "must", "need",
    "have", "has", "had", "being", "been", "give", "given", "takes",
    "the", "this", "that", "these", "those", "there", "here",
    "your", "our", "their", "about", "with", "from", "into", "onto",
    "also", "just", "only", "more", "than", "then", "some", "like",
    "were", "well", "much", "many", "such", "each", "both", "off",
    "me", "my", "we", "you", "it", "its", "him", "her", "his",
    "they", "them", "and", "or", "but", "if", "for", "on", "in",
    "at", "to", "of", "by", "not", "no", "so", "any", "all",
    "get", "got", "do", "did", "do", "are", "is", "be", "was",
    "a", "an", "i",
})


# ──────────────────────────────────────────────────────────────
# Internal I/O helpers
# ──────────────────────────────────────────────────────────────

def _ensure_log_exists() -> None:
    """Create the feedback log file and its parent directory if not present."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.exists(FEEDBACK_LOG_PATH):
        open(FEEDBACK_LOG_PATH, "w", encoding="utf-8").close()


def _read_log() -> list[dict]:
    """
    Read all entries from the append-only JSONL log.
    Malformed lines are skipped silently.
    """
    _ensure_log_exists()
    entries: list[dict] = []
    with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _get_faq_text(faq_id: str, min_freq: int = 2) -> frozenset[str]:
    """
    Return the set of lowercase word-tokens that appear >= min_freq times
    in the FAQ's question + answer text.

    FIX 4: Only high-frequency words are excluded from tag suggestions.
    Single-occurrence words are kept as candidates because the user's query
    may legitimately reference them as novel pain-points.  Words that appear
    repeatedly in the FAQ text are assumed 'already covered' and excluded.

    Args:
        faq_id:   FAQ identifier to look up.
        min_freq: Minimum occurrence count to be excluded (default: 2).

    Returns:
        frozenset of excluded word-tokens (empty on any I/O error).
    """
    try:
        faq_path = os.path.join(_DATA_DIR, "faqs.json")
        with open(faq_path, "r", encoding="utf-8") as fh:
            faqs = json.load(fh)
        for faq in faqs:
            if faq.get("id") == faq_id:
                combined = (
                    faq.get("question", "") + " " + faq.get("answer", "")
                ).lower()
                words = re.sub(r"[^a-z\s]", " ", combined).split()
                freq: dict[str, int] = {}
                for w in words:
                    freq[w] = freq.get(w, 0) + 1
                return frozenset(w for w, cnt in freq.items() if cnt >= min_freq)
    except Exception:
        pass
    return frozenset()


# ──────────────────────────────────────────────────────────────
# F2 — Feedback Storage
# ──────────────────────────────────────────────────────────────

def store_feedback(
    faq_id:   str,
    query:    str,
    route:    str,
    score:    float,
    feedback: str,
    user_id:  int | None = None,
) -> None:
    """
    Append one feedback event to the append-only JSONL log (FIX 2: F2).
    Also updates the query-level memory (FIX 5).

    Feedback is stored as a **global** signal: all aggregation functions
    (get_aggregated_scores, apply_feedback_reranking) consider every entry
    regardless of user_id.  The user_id field is recorded for audit purposes
    only and does not affect ranking.

    Args:
        faq_id:   FAQ identifier (e.g. "faq_001")
        query:    Original English query that produced this result
        route:    Routing decision ("keyword" | "semantic" | "hybrid")
        score:    Retrieval score for this FAQ in this query
        feedback: "up" | "down" | "not_helpful"
        user_id:  Optional integer ID of the user who submitted the feedback.
                  None for guest users or pre-existing log entries.

    Raises:
        ValueError: if feedback is not one of the recognised types.
    """
    if feedback not in SCORE_MAP:
        raise ValueError(
            f"Invalid feedback type: {feedback!r}. "
            f"Valid options: {sorted(SCORE_MAP.keys())}"
        )

    _ensure_log_exists()
    entry = {
        "faq_id":    faq_id,
        "query":     query,
        "route":     route,
        "score":     round(float(score), 4),
        "feedback":  feedback,
        "user_id":   user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # FIX 5: also persist query-level memory
    store_query_feedback(query, faq_id, feedback)


# ──────────────────────────────────────────────────────────────
# FIX 5 — Query-level Feedback Memory
# ──────────────────────────────────────────────────────────────

def _read_query_feedback() -> dict:
    """Load query_feedback.json; returns empty dict on any I/O error."""
    if not os.path.exists(QUERY_FEEDBACK_PATH):
        return {}
    try:
        with open(QUERY_FEEDBACK_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_query_feedback(data: dict) -> None:
    """Persist query feedback dict atomically."""
    with open(QUERY_FEEDBACK_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def store_query_feedback(query: str, faq_id: str, feedback: str) -> None:
    """
    Upsert a query-level feedback record.

    Storage schema (modules/query_feedback.json):
        { "<query>|||<faq_id>": { query, faq_id, feedback, count }, ... }

    The count increments on each event; feedback is updated to the latest type.
    This enables detecting which (query, FAQ) pairs consistently fail.
    """
    data = _read_query_feedback()
    key  = f"{query}|||{faq_id}"
    rec  = data.get(key, {"query": query, "faq_id": faq_id, "feedback": feedback, "count": 0})
    rec["count"]    += 1
    rec["feedback"]  = feedback          # track most recent type
    data[key]        = rec
    _write_query_feedback(data)


def get_query_feedback(
    faq_id: str | None = None,
    query:  str | None = None,
) -> list[dict]:
    """
    Return query-level feedback records, optionally filtered.

    Args:
        faq_id: If provided, only return records for this FAQ.
        query:  If provided, only return records matching this exact query.

    Returns:
        List of { query, faq_id, feedback, count } dicts.
    """
    data = _read_query_feedback()
    recs = list(data.values())
    if faq_id:
        recs = [r for r in recs if r.get("faq_id") == faq_id]
    if query:
        recs = [r for r in recs if r.get("query") == query]
    return recs


# ──────────────────────────────────────────────────────────────
# F3 + FIX 1 + FIX 6 — Aggregated Scoring (normalised, windowed)
# ──────────────────────────────────────────────────────────────

def get_aggregated_scores() -> dict[str, dict]:
    """
    Compute per-FAQ aggregated feedback metrics with all refinements applied.

    Refinements:
      FIX 6: Only the most recent DECAY_WINDOW entries per FAQ are used.
      FIX 3: not_helpful events are multiplied by NOT_HELPFUL_MULTIPLIER (1.5×)
             in weighted_sum, accelerating rank penalty and low_quality detection.
      FIX 1: avg_feedback = weighted_sum / count (normalised, not raw sum).

    Returns:
        {
          "faq_id": {
              "score":        int,    # raw sum (backward compat for UI)
              "count":        int,    # events in decay window
              "weighted_sum": float,  # sum with not_helpful × 1.5
              "avg_feedback": float,  # weighted_sum / count
          },
          ...
        }
    """
    # Step 1: group all valid entries per FAQ (FIX 6: keep insertion order)
    raw: dict[str, list[dict]] = {}
    for entry in _read_log():
        fid = entry.get("faq_id", "")
        fb  = entry.get("feedback", "")
        if not fid or fb not in SCORE_MAP:
            continue
        raw.setdefault(fid, []).append(entry)

    # Step 2: apply decay window and compute metrics per FAQ
    agg: dict[str, dict] = {}
    for fid, entries in raw.items():
        # FIX 6: only consider the last DECAY_WINDOW entries
        window     = entries[-DECAY_WINDOW:]
        raw_score  = 0
        w_sum      = 0.0
        for e in window:
            fb         = e["feedback"]
            delta      = SCORE_MAP[fb]
            multiplier = NOT_HELPFUL_MULTIPLIER if fb == "not_helpful" else 1.0
            raw_score += delta
            w_sum     += delta * multiplier

        count        = len(window)
        avg_feedback = w_sum / count if count > 0 else 0.0

        agg[fid] = {
            "score":        raw_score,          # backward compat (UI reads this)
            "count":        count,
            "weighted_sum": round(w_sum, 4),
            "avg_feedback": round(avg_feedback, 6),
        }

    return agg


def get_feedback_score(faq_id: str) -> int:
    """Return the raw aggregated feedback score for a single FAQ (0 if none)."""
    return get_aggregated_scores().get(faq_id, {}).get("score", 0)


# ──────────────────────────────────────────────────────────────
# F5 — Poor FAQ Detection (FIX 3: uses weighted_sum)
# ──────────────────────────────────────────────────────────────

def get_poor_faqs(
    score_threshold: float = POOR_SCORE_THRESHOLD,
    count_threshold: int   = POOR_COUNT_THRESHOLD,
) -> list[str]:
    """
    Return faq_ids flagged as low quality.

    Conditions (both required):
      weighted_sum < score_threshold (default: -5.0)
      count >= count_threshold       (default: 3)

    FIX 3: Uses weighted_sum (not raw score) so not_helpful events (×1.5)
    accelerate detection — 3 not_helpful events give weighted_sum = -9 (< -5),
    reaching the threshold with just 3 events.

    count uses >= (not >) so the acceleration achieves effect at exactly
    count_threshold events rather than requiring count_threshold + 1.
    """
    agg = get_aggregated_scores()
    return [
        fid for fid, data in agg.items()
        if data["weighted_sum"] < score_threshold
        and data["count"] >= count_threshold
    ]


def is_low_quality(faq_id: str) -> bool:
    """Return True if this FAQ is currently flagged as low quality."""
    return faq_id in get_poor_faqs()


# ──────────────────────────────────────────────────────────────
# FIX 4 — Improved Tag Extraction
# ──────────────────────────────────────────────────────────────

def _extract_tokens(
    text:     str,
    faq_text: frozenset[str] = frozenset(),
) -> list[str]:
    """
    Extract clean keyword tokens from a query string.

    FIX 4 rules (all must hold):
      1. Lowercase alpha characters only (regex strip).
      2. Length > 3 (minimum 4 characters).
      3. Not purely numeric (isnumeric guard).
      4. Not in the English stopword set.
      5. Not already present in the FAQ's own text (faq_text set).
      6. Deduplicated (seen-set ensures no duplicates in output list).

    Args:
        text:     Query string to tokenise.
        faq_text: Word set from the FAQ's question + answer (FIX 4, rule 5).

    Returns:
        Ordered list of unique clean tokens from this query.
    """
    normalised = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    seen: set[str] = set()
    tokens: list[str] = []
    for t in normalised.split():
        if (
            len(t) > TAG_MIN_LENGTH - 1      # length > 3  → len >= 4
            and not t.isnumeric()             # not purely numeric
            and t not in _STOPWORDS           # not a stopword
            and t not in faq_text             # not already in FAQ text (FIX 4)
            and t not in seen                 # no duplicates (FIX 4)
        ):
            seen.add(t)
            tokens.append(t)
    return tokens


def get_suggested_tags(faq_id: str, top_k: int = TAG_TOP_K) -> list[str]:
    """
    Suggest tags for a FAQ based on repeated negative-feedback query patterns.

    FIX 4 improvements:
      - Tokens excluded if already in the FAQ's own question/answer text.
      - Minimum token length raised to 4 (length > 3).
      - Deduplication within each query's token list.

    Algorithm:
      1. Load FAQ text as a word set (for exclusion filtering).
      2. Scan log for entries with this faq_id AND feedback ∈ (down, not_helpful).
      3. Tokenise each query with improved _extract_tokens(text, faq_text).
      4. Count token frequency across all qualifying entries.
      5. Return top-K by frequency.

    Returns:
        List of suggested tag strings (empty if no negative feedback exists).
    """
    faq_text = _get_faq_text(faq_id)   # FIX 4: load once per call

    freq: dict[str, int] = {}
    for entry in _read_log():
        if entry.get("faq_id") != faq_id:
            continue
        if entry.get("feedback") not in ("down", "not_helpful"):
            continue
        for token in _extract_tokens(entry.get("query", ""), faq_text):
            freq[token] = freq.get(token, 0) + 1

    if not freq:
        return []

    return sorted(freq, key=lambda t: freq[t], reverse=True)[:top_k]


# ──────────────────────────────────────────────────────────────
# FIX 1 + 2 + 7 — Ranking Adjustment (normalised, clamped, gated)
# ──────────────────────────────────────────────────────────────

def apply_feedback_reranking(
    results:       list[dict],
    weight:        float = RERANK_WEIGHT,
    semantic_score: float = 1.0,
) -> list[dict]:
    """
    Apply normalised, clamped, confidence-gated feedback reranking.

    FIX 1 — Normalised formula:
        avg_feedback  = weighted_sum / max(count, 1)
        adjustment    = avg_feedback * weight
        adjusted_score = base_score + adjustment

    FIX 2 — Clamping:
        adjusted_score = max(0.0, min(1.0, adjusted_score))

    FIX 3 — Differentiated penalty:
        not_helpful events contribute ×1.5 to weighted_sum (via
        get_aggregated_scores), accelerating negative adjustment.

    FIX 7 — Semantic gate:
        If semantic_score < THRESHOLD_LOW (0.65), no adjustment is applied.
        Feedback must not amplify a weak or mismatched retrieval signal.

    FIX 8 — Extended debug fields per result:
        feedback_score, feedback_avg, feedback_count,
        adjustment, adjusted_score, low_quality

    Args:
        results:        Result dicts from build_final_response().
        weight:         Multiplier for avg_feedback (default: 0.05).
        semantic_score: Tier 2 semantic similarity score (FIX 7 gate).

    Returns:
        New list sorted by adjusted_score descending, ranks updated,
        debug fields injected.  Returns input unchanged if empty.
    """
    if not results:
        return results

    # FIX 7: load routing threshold (lazy import avoids circular dependency)
    try:
        from modules.constants import THRESHOLD_LOW
    except ImportError:
        THRESHOLD_LOW = _THRESHOLD_LOW_FALLBACK

    apply_adjustment = semantic_score >= THRESHOLD_LOW

    # Debug: log gate decision (Fix 5 — skip log)
    if not apply_adjustment:
        logger.debug(
            "[feedback_reranking] GATE SKIPPED: semantic_score=%.4f < "
            "THRESHOLD_LOW=%.2f — no adjustment applied to any result.",
            semantic_score, THRESHOLD_LOW,
        )

    agg      = get_aggregated_scores()
    poor_set = set(get_poor_faqs())

    enriched: list[dict] = []
    for result in results:
        fid        = result.get("faq_id", "")
        orig_score = float(result.get("score", 0.0))
        data       = agg.get(fid, {})

        w_sum      = data.get("weighted_sum", 0.0)
        count      = data.get("count", 0)
        avg_fb     = data.get("avg_feedback", 0.0)
        raw_score  = data.get("score", 0)

        if apply_adjustment and count > 0:
            adjustment = avg_fb * weight
            adj_score  = max(0.0, min(1.0, round(orig_score + adjustment, 6)))
        else:
            adjustment = 0.0
            adj_score  = orig_score

        # Fix 3: per-result debug log
        logger.debug(
            "[feedback_reranking] faq_id=%-10s base=%.4f  "
            "avg_fb=%+.3f  count=%2d  adjustment=%+.4f  adj=%.4f  "
            "low_quality=%s  gate=%s",
            fid or "(none)", orig_score,
            avg_fb, count, adjustment, adj_score,
            fid in poor_set, "ON" if apply_adjustment else "OFF",
        )

        enriched.append({
            **result,
            "feedback_score":  raw_score,
            "feedback_avg":    round(avg_fb, 4),
            "feedback_count":  count,
            "adjustment":      round(adjustment, 6),
            "adjusted_score":  adj_score,
            "low_quality":     fid in poor_set,
        })

    enriched.sort(key=lambda r: r["adjusted_score"], reverse=True)
    for i, r in enumerate(enriched, start=1):
        r["rank"] = i

    return enriched
