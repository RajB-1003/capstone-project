"""
query_filter.py — Input quality gate for unanswered_queries table (Phase 2).

Determines whether a low-confidence query is meaningful enough to store.
All rules are regex-based — no hardcoded topic keywords.

Public API
----------
    is_valid_unanswered_query(query: str) -> bool
"""

import re

# ── Compiled patterns (module-level — compiled once at import time) ────────────

# At least one Unicode letter (covers Latin, Devanagari, Tamil, Arabic, CJK…)
_RE_HAS_LETTER = re.compile(r"\p{L}", re.UNICODE) if False else None

# Fallback without the `regex` library — use the built-in `re` with a broad
# character range that covers the scripts used in the target system.
# [a-zA-Z] covers ASCII letters; [\u0080-\uFFFF] covers virtually all
# non-ASCII Unicode letters without requiring the third-party `regex` library.
_RE_HAS_ALPHA = re.compile(
    r"[a-zA-Z\u0080-\uFFFF]",
    re.UNICODE,
)

# A "word" for our purposes: any run of alphanumeric/Unicode characters
# (splits on whitespace and punctuation).
_RE_WORD_SPLIT = re.compile(r"[\s\W]+", re.UNICODE)

# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_CHARS:  int = 10   # Step 2 rule 4: reject if total char length < 10
_MIN_WORDS:  int = 3    # Step 2 rule 1: must have at least 3 words


# ── Public function ───────────────────────────────────────────────────────────

def is_valid_unanswered_query(query: str) -> bool:
    """
    Return True if ``query`` is meaningful enough to store as unanswered.

    Rules (Step 2 spec — all regex, no hardcoded keywords):

    1. Minimum 3 words — rejects single-word and two-word queries that
       are too vague to act on ("hostel?", "fees what").
    2. Must contain at least 1 alphabetic character — ensures the query
       expresses a concept in some human language.
    3. Reject if only symbols or numbers — pure noise ("@@@###$$$",
       "12345 67890 000").
    4. Reject very short queries (total char count < 10) — catches
       "hostel?" and other near-trivial inputs.

    All rules operate on the stripped input; no additional normalisation
    is applied here (normalisation to lowercase happens in db.py before
    storage).

    Args:
        query: Raw (or lightly normalised) user query string.

    Returns:
        True  → query should be stored in unanswered_queries.
        False → query is noise and should be silently discarded.

    Examples:
        >>> is_valid_unanswered_query("how to apply for internship outside university")
        True
        >>> is_valid_unanswered_query("@@@###$$$")
        False
        >>> is_valid_unanswered_query("hostel?")
        False
        >>> is_valid_unanswered_query("12345 67890 000")
        False
    """
    if not query:
        return False

    stripped = query.strip()

    # Rule 4: minimum character length
    if len(stripped) < _MIN_CHARS:
        return False

    # Tokenise into words (split on whitespace + punctuation)
    words = [w for w in _RE_WORD_SPLIT.split(stripped) if w]

    # Rule 1: minimum 3 words
    if len(words) < _MIN_WORDS:
        return False

    # Rule 2 + 3: must contain at least one alphabetic character
    # This simultaneously rejects purely-numeric and purely-symbolic input.
    if not _RE_HAS_ALPHA.search(stripped):
        return False

    return True
