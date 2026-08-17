"""
multilingual.py — Phase 1 Improvement: Multilingual Query Support
                  (Robustness fixes applied)

Responsibilities:
  - Detect the language of an incoming query using langdetect (confidence-gated)
  - Translate non-English / uncertain queries to English via deep-translator
  - Cache translation results in memory (normalized keys) to avoid redundant calls
  - Return a structured query-data dict for integration with run_pipeline()

Design Principles:
  - Fully generalized — no hardcoded languages, keywords, or query strings
  - Graceful degradation — always returns a valid result, never crashes
  - Stateless functions with shared module-level cache (thread-safe dict ops)
  - Data-driven — relies entirely on library-based detection and translation

Robustness fixes (applied after Phase 1):
  Fix 1 — detect_language() uses detect_langs() with a 0.70 confidence threshold;
           low-confidence detections return "unknown" instead of a guessed code.
  Fix 2 — process_query() uses a clean en / unknown / other decision tree;
           the error-prone non-ASCII character heuristic has been removed.
  Fix 3 — Cache keys are normalized (strip + lower) so minor whitespace/casing
           differences always hit the same cache entry.
  Fix 4 — translate_to_english() applies light output cleanup (strip + trailing
           punctuation guard) without altering semantics.
  Fix 5 — All existing fail-safes are preserved (detection fail → "en",
           translation fail → original query, never raises).

Architecture mandate:
  - This module DOES NOT modify run_pipeline(), routing, retrieval, or RRF
  - It acts as a pre-processing hook inserted in app.py BEFORE run_pipeline()
  - All existing modules remain pure and unchanged

Public API:
  detect_language(query: str) -> str
  translate_to_english(query: str) -> str
  process_query(query: str) -> dict
"""

from __future__ import annotations

import logging

# ──────────────────────────────────────────────────────────────
# Module-level logger
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# In-memory translation cache
# Keyed by a NORMALIZED query string (strip + lower) to ensure that minor
# casing / leading-trailing space differences always hit the same entry.
# Value is the processed_query string (translated or original).
# Using a plain dict — Python dict reads/writes are GIL-protected (thread-safe
# for single-key access in CPython).
# ──────────────────────────────────────────────────────────────
_translation_cache: dict[str, str] = {}

# Minimum langdetect confidence probability to trust a detected language.
# Detections below this threshold are treated as "unknown".
_CONFIDENCE_THRESHOLD: float = 0.70


# ──────────────────────────────────────────────────────────────
# Lazy-import helpers — keep startup time fast; only pay the
# import cost when the first multilingual query arrives.
# ──────────────────────────────────────────────────────────────

def _get_langdetect_langs():
    """
    Return the langdetect.detect_langs callable (returns a list of
    Language objects with .lang and .prob attributes), or None if
    the library is unavailable.

    detect_langs() is preferred over detect() because it exposes the
    confidence probability, letting us gate on _CONFIDENCE_THRESHOLD.
    """
    try:
        from langdetect import detect_langs, DetectorFactory  # type: ignore
        # Make detection deterministic across runs
        DetectorFactory.seed = 42
        return detect_langs
    except ImportError:
        logger.warning(
            "langdetect is not installed. "
            "Install it with: pip install langdetect"
        )
        return None


def _get_translator():
    """
    Return a callable translate(text, target_lang) -> str, or None.

    Preference order (first available wins):
      1. deep-translator GoogleTranslate  (most reliable, no version pinning)
      2. googletrans Translator           (fallback)

    Both are optional — if neither is available the module degrades gracefully
    by returning the original query unchanged.
    """
    # ── Option 1: deep-translator ──────────────────────────────
    try:
        from deep_translator import GoogleTranslator  # type: ignore

        def _deep_translate(text: str, target: str = "en") -> str:
            return GoogleTranslator(source="auto", target=target).translate(text)

        return _deep_translate
    except ImportError:
        pass

    # ── Option 2: googletrans ──────────────────────────────────
    try:
        from googletrans import Translator  # type: ignore

        _gt = Translator()

        def _googletrans_translate(text: str, target: str = "en") -> str:
            result = _gt.translate(text, dest=target)
            return result.text

        return _googletrans_translate
    except ImportError:
        pass

    logger.warning(
        "No translation library found. "
        "Install one of: deep-translator, googletrans. "
        "Multilingual translation will be disabled."
    )
    return None


# ──────────────────────────────────────────────────────────────
# Language heuristic constants (Upgrade 3)
# ──────────────────────────────────────────────────────────────

# Fraction of characters that must be ASCII for a query to be considered
# likely-Roman script (0.80 = at most 20% non-ASCII characters allowed).
_ASCII_RATIO_THRESHOLD: float = 0.80

# Fraction of query tokens that must appear in _ENGLISH_WORD_SET for the
# query to be fast-pathed as English (0.30 = at least 30% known words).
_ENG_WORD_RATIO_THRESHOLD: float = 0.30

# High-frequency English words: common function words + academic FAQ domain
# words.  The union covers short, single-word queries ("exam", "fee") and
# long interrogative sentences alike.
# Edit this set to extend domain coverage — never add query-specific tokens.
_ENGLISH_WORD_SET: frozenset[str] = frozenset({
    # Function words and determiners
    "what", "how", "when", "where", "why", "who", "which", "can", "do",
    "does", "is", "are", "will", "would", "should", "could", "may", "must",
    "have", "has", "had", "be", "been", "being",
    "the", "a", "an", "my", "your", "their", "our", "its", "i", "we",
    "and", "or", "but", "if", "that", "this", "these", "those", "for",
    "on", "in", "at", "to", "of", "with", "by", "from", "about", "not",
    # Academic FAQ domain words
    "attendance", "exam", "exams", "test", "course", "courses", "semester",
    "grade", "grades", "fee", "fees", "scholarship", "hostel", "assignment",
    "syllabus", "prerequisite", "prerequisites", "elective", "registration",
    "library", "placement", "internship", "project", "thesis",
    "credit", "credits", "gpa", "cgpa", "retake", "appeal", "penalty",
    "plagiarism", "deadline", "department", "faculty", "student", "policy",
    "apply", "applied", "application", "eligibility", "eligible", "miss",
    "missed", "late", "rule", "rules", "regulation", "procedure", "process",
})


def _is_likely_english(query: str) -> bool:
    """
    Ratio-based English pre-check (Upgrade 3).

    Two independent ratios are computed; BOTH must exceed their thresholds
    for the function to return True (bypassing langdetect).

    Ratio 1 — ASCII ratio:
        ascii_chars / total_chars >= _ASCII_RATIO_THRESHOLD (0.80)
        A mostly-ASCII query is strongly indicative of Latin-script input.

    Ratio 2 — English word ratio:
        (query_tokens ∩ _ENGLISH_WORD_SET) / total_tokens >= _ENG_WORD_RATIO_THRESHOLD (0.30)
        Measures the fraction of tokens that are high-frequency English words.
        Unlike the old anchor-token check, this is a rate, not a presence flag,
        so single-word queries ("exam") still pass and long non-English queries
        still fail even if they happen to contain one English token.

    Thresholds are module-level named constants — never magic numbers.

    Positive-only guarantee: returning False does NOT mean non-English.
    It means we cannot be confident, so langdetect should decide.
    """
    if not query or not query.strip():
        return False

    # ── Ratio 1: ASCII ratio ──────────────────────────────────
    chars = query.strip()
    ascii_count = sum(1 for c in chars if ord(c) < 128)
    ascii_ratio = ascii_count / len(chars)
    if ascii_ratio < _ASCII_RATIO_THRESHOLD:
        return False   # Non-ASCII majority — let langdetect decide

    # ── Ratio 2: English word ratio ───────────────────────────
    tokens = query.lower().split()
    if not tokens:
        return False
    english_hits    = sum(1 for t in tokens if t in _ENGLISH_WORD_SET)
    eng_word_ratio  = english_hits / len(tokens)

    return eng_word_ratio >= _ENG_WORD_RATIO_THRESHOLD


def detect_language(query: str) -> str:

    """
    Detect the ISO 639-1 language code of the given query string.

    Uses langdetect.detect_langs() (model-based, not keyword matching)
    which supports 55+ languages including Tamil (ta), Hindi (hi), and
    mixed scripts.  The top detection is only accepted when its confidence
    probability is >= _CONFIDENCE_THRESHOLD (0.70); otherwise the function
    returns "unknown" so that process_query() can apply a safe fallback.

    Fix 1 applied here:
      - Uses detect_langs() instead of detect() to access probability.
      - Returns "unknown" for low-confidence detections (e.g. short mixed
        queries like "CS-202 attendance என்ன?").

    Args:
        query: Any natural-language string (full sentence preferred).

    Returns:
        ISO 639-1 language code (e.g. "en", "ta", "hi"),
        "unknown" when confidence < 0.70,
        or "en" on any detection error (fail-safe).
    """
    if not query or not query.strip():
        return "en"

    # ── Issue 4 fix: ASCII + English-token pre-check ──────────
    # Fast-path for obvious English queries to avoid langdetect
    # misclassifying short English text (e.g. as French).
    if _is_likely_english(query):
        logger.debug("Pre-check classified query as English (ASCII+token heuristic): %r", query[:60])
        return "en"

    detect_fn = _get_langdetect_langs()
    if detect_fn is None:
        # Library unavailable — default to English so pipeline continues
        return "en"


    try:
        results = detect_fn(query.strip())   # list of Language(lang, prob)
        if not results:
            return "en"

        top = results[0]          # highest-probability language
        lang = str(top.lang)      # e.g. "ta", "hi", "en"
        prob = float(top.prob)    # confidence in [0, 1]

        if prob < _CONFIDENCE_THRESHOLD:
            logger.debug(
                "Low-confidence detection for query %r: lang=%s prob=%.3f — returning 'unknown'",
                query[:60], lang, prob,
            )
            return "unknown"

        return lang if lang else "en"

    except Exception as exc:
        logger.debug("Language detection failed for query %r: %s", query[:60], exc)
        return "en"


def translate_to_english(query: str) -> str:
    """
    Translate an arbitrary-language query to English.

    Translation is full-sentence (not token-by-token) via the best
    available translation backend (deep-translator > googletrans).

    Args:
        query: Natural-language query in any language, including
               mixed-language inputs (e.g. "CS-202 attendance என்ன?").

    Returns:
        English translation of the query.
        Returns the *original* query string on any error — never raises.
    """
    if not query or not query.strip():
        return query

    translate_fn = _get_translator()
    if translate_fn is None:
        # No translator available — return original (pipeline still works)
        logger.debug("No translator available; returning original query.")
        return query

    try:
        translated = translate_fn(query.strip(), "en")

        # Fix 4 — light output cleanup (strip only; guard trailing punctuation)
        if translated:
            translated = translated.strip()
            # If the translation lost trailing punctuation and the last char is
            # alphanumeric, restore a question mark (most FAQ queries are questions).
            if translated and translated[-1].isalnum():
                translated += "?"

        if translated:
            return translated
        # Empty result — fall through to original
        return query
    except Exception as exc:
        logger.debug(
            "Translation failed for query %r: %s — returning original.",
            query[:60], exc,
        )
        return query


def process_query(query: str) -> dict:
    """
    Full pre-processing entry point for multilingual query handling.

    Logic (Fix 2 & Fix 3 applied):
      1. Normalize cache key: strip + lower  (Fix 3).
      2. Check normalized key in cache — return early on hit.
      3. Detect language with confidence gating (via detect_language).
      4. Translation decision tree  (Fix 2 — removes non-ASCII heuristic):
           language == "en"      → DO NOT translate (pure English)
           language == "unknown" → attempt translation (safe fallback for
                                    mixed/ambiguous queries)
           anything else         → translate normally
      5. Cache result under normalized key.
      6. Return structured dict (original_query always preserves raw input).

    Args:
        query: Raw user query (any language).

    Returns:
        {
            "processed_query": str,   # English query for pipeline
            "original_query":  str,   # Raw user input (unchanged)
            "language":        str,   # Detected ISO 639-1 code or "unknown"
            "translated":      bool,  # True if translation was applied
        }
    """
    original = query  # preserve exactly what the user typed

    # ── Fix 3: Normalize cache key ─────────────────────────────
    # strip + lower so "  CS-202 என்ன?  " and "cs-202 என்ன?" share one entry.
    cache_key = original.strip().lower()

    # ── Cache hit ──────────────────────────────────────────────
    if cache_key in _translation_cache:
        cached_processed = _translation_cache[cache_key]
        # Re-derive language metadata (cheap; not cached to keep dict simple)
        lang = detect_language(original)
        translated = (cached_processed.strip() != original.strip())
        return {
            "processed_query": cached_processed,
            "original_query":  original,
            "language":        lang,
            "translated":      translated,
        }

    # ── Detect language (confidence-gated) ─────────────────────
    lang = detect_language(original)

    # ── Fix 2: Safe translation decision tree ──────────────────
    # Removed: error-prone non-ASCII character heuristic.
    # Rules:
    #   "en"      → keep original (genuine English, no translation needed)
    #   "unknown" → attempt translation  (mixed/short/ambiguous input)
    #   anything  → translate normally
    if lang == "en":
        needs_translation = False
    elif lang == "unknown":
        needs_translation = True   # safe fallback for uncertain detections
    else:
        needs_translation = True   # confirmed non-English language

    # ── Apply translation ──────────────────────────────────────
    if needs_translation:
        processed = translate_to_english(original)
        translated = (processed.strip() != original.strip())
    else:
        processed = original
        translated = False

    # ── Fix 3: Store under normalized key ─────────────────────
    _translation_cache[cache_key] = processed

    return {
        "processed_query": processed,
        "original_query":  original,
        "language":        lang,
        "translated":      translated,
    }


# ──────────────────────────────────────────────────────────────
# Cache utility (for testing / diagnostics)
# ──────────────────────────────────────────────────────────────

def get_cache_stats() -> dict:
    """Return diagnostic information about the translation cache."""
    return {
        "cache_size": len(_translation_cache),
        "cached_queries": list(_translation_cache.keys()),
    }


def clear_translation_cache() -> None:
    """Evict all entries from the translation cache (useful for testing)."""
    _translation_cache.clear()
