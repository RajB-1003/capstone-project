"""
constants.py — Shared routing threshold constants + system versioning

This is the SINGLE SOURCE OF TRUTH for the routing thresholds used throughout
the system. Every module that references 0.82 or 0.65 MUST import from here.

Also defines cache versioning tokens so the cache can self-invalidate when the
dataset or embedding model changes.

Defined once, imported everywhere:
    modules/router_tier2.py   — classify_intent thresholds
    modules/pipeline.py       — _fuse_signals thresholds + routing trace
    modules/explainability.py — generate_rationale display
    modules/confidence.py     — low-confidence detection
    modules/cache.py          — version-aware cache key
    app.py                    — sidebar route legend

DO NOT duplicate these values in any other module.
"""

# ──────────────────────────────────────────────────────────────
# Routing thresholds (architecture mandate — do not change without
# updating the full routing decision table in pipeline._fuse_signals)
# ──────────────────────────────────────────────────────────────

THRESHOLD_HIGH: float = 0.82
"""Minimum semantic score for a pure-semantic routing decision (no entity)."""

THRESHOLD_LOW: float = 0.65
"""Minimum semantic score considered meaningful.
  - With entity: routes HYBRID instead of KEYWORD
  - Without entity: routes HYBRID instead of SEMANTIC fallback
"""

# ──────────────────────────────────────────────────────────────
# Cache versioning (Upgrade 2)
# Increment CORPUS_VERSION whenever faqs.json is edited.
# Update EMBEDDING_VERSION whenever the embedding model changes.
# Both tokens are folded into the cache key so stale entries are
# automatically bypassed without requiring a manual cache clear.
# ──────────────────────────────────────────────────────────────

CORPUS_VERSION: str = "1.0"
"""Increment this string whenever the FAQ dataset (faqs.json) changes."""

EMBEDDING_VERSION: str = "all-MiniLM-L6-v2"
"""Update this string whenever the sentence-transformer model changes."""
