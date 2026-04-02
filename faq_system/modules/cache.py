"""
cache.py — Phase 7: LRU Exact Cache + Semantic Near-Duplicate Cache

Responsibilities:
  - Exact LRU cache: O(1) lookup/insert using OrderedDict; max 50 entries
  - Semantic cache: dot-product similarity on L2-normalized query embeddings;
    max 20 entries; threshold 0.98 (near-duplicate queries only)
  - Module-level singletons so cache persists for the lifetime of the process
  - All public functions operate on the singletons — callers never touch internals

Design decisions:
  - Cache ONLY the final build_final_response() output (complete result dict).
    Partial results (state, BM25 scores, intermediate embeddings) are NOT cached.
  - Exact cache uses normalized (stripped, lowercased) query string as key so
    "CS-202 Prerequisites " and "cs-202 prerequisites" both hit the same entry.
  - Semantic cache uses FIFO eviction (pop index 0) — the oldest entry is
    removed when the cache is full. This keeps implementation O(N) per store
    with N ≤ 20 (negligible cost relative to embedding computation).
  - Semantic cache entries store the original (un-normalized) query string for
    rationale messages, and a deep copy of the result dict to avoid aliasing.
  - Thread safety: not implemented — this is a single-process system.

Public API:
  cache_lookup(query)                    -> dict | None
  cache_store(query, result)
  semantic_cache_lookup(embedding, threshold=0.98)  -> dict | None
  semantic_cache_store(embedding, query, result)
  clear_all_caches()                     (testing / reset)
  cache_stats()                          -> dict
"""

import copy
from collections import OrderedDict

import numpy as np


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
EXACT_CACHE_MAX_SIZE    = 50
SEMANTIC_CACHE_MAX_SIZE = 20
SEMANTIC_THRESHOLD_DEFAULT = 0.98


# ──────────────────────────────────────────────────────────────
# 1. LRU exact-match cache
# ──────────────────────────────────────────────────────────────

class _LRUCache:
    """
    OrderedDict-based LRU cache for exact query → result mappings.

    Strategy:
        - On lookup HIT:  move key to end (most recently used).
        - On store:       insert at end; if over capacity, pop first (LRU).
        - Normalisation:  key = query.strip().lower() for case/space tolerance.

    Complexity:
        - lookup:  O(1)
        - store:   O(1) amortized
        - evict:   O(1) via popitem(last=False)
    """

    def __init__(self, max_size: int = EXACT_CACHE_MAX_SIZE):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self.max_size = max_size
        self.hits   = 0
        self.misses = 0
        self.stores = 0
        self.evictions = 0

    @staticmethod
    def _normalize(query: str) -> str:
        return query.strip().lower()

    def lookup(self, query: str) -> dict | None:
        key = self._normalize(query)
        if key not in self._store:
            self.misses += 1
            return None
        # Move to end (most-recently used)
        self._store.move_to_end(key)
        self.hits += 1
        return copy.deepcopy(self._store[key])   # deep copy — never return reference

    def store(self, query: str, result: dict) -> None:
        key = self._normalize(query)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = copy.deepcopy(result)
        else:
            if len(self._store) >= self.max_size:
                self._store.popitem(last=False)   # evict LRU (first = oldest)
                self.evictions += 1
            self._store[key] = copy.deepcopy(result)
        self.stores += 1

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = self.stores = self.evictions = 0

    def size(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        return list(self._store.keys())


# ──────────────────────────────────────────────────────────────
# 2. Semantic near-duplicate cache
# ──────────────────────────────────────────────────────────────

class _SemanticCache:
    """
    Vector similarity cache for near-duplicate query detection.

    Each entry stores: (L2-normalized embedding, query_str, result_dict).

    Lookup:
        Stack all stored embeddings into (N, 384) matrix.
        Compute dot products against query_vec → (N,) similarities.
        If max similarity ≥ threshold → return cached result.

    Eviction: FIFO — oldest entry (index 0) removed when full.

    Why threshold = 0.98?
        0.98 ensures only near-duplicate queries (paraphrases with identical
        meaning) reuse cached results. Semantically similar-but-different
        queries (e.g. "hostel fee" vs "hostel rules") won't reach 0.98.

    Complexity:
        - lookup:  O(N * D) BLAS matmul, N ≤ 20, D = 384 → negligible
        - store:   O(N) for FIFO check, N ≤ 20
    """

    def __init__(
        self,
        max_size: int = SEMANTIC_CACHE_MAX_SIZE,
        threshold: float = SEMANTIC_THRESHOLD_DEFAULT,
    ):
        self.max_size  = max_size
        self.threshold = threshold
        # Each entry: (embedding: np.ndarray (384,), query: str, result: dict)
        self._entries: list[tuple[np.ndarray, str, dict]] = []
        self.hits      = 0
        self.misses    = 0
        self.stores    = 0
        self.evictions = 0

    def lookup(
        self,
        query_embedding: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[dict, float, str] | None:
        """
        Returns (result_copy, similarity_score, matched_query) or None.
        Returns a tuple so callers can include the matched query in rationale.
        """
        effective_threshold = threshold if threshold is not None else self.threshold
        if not self._entries:
            self.misses += 1
            return None

        # Stack stored embeddings: (N, 384)
        stored_matrix = np.stack([e[0] for e in self._entries], axis=0)
        similarities  = stored_matrix @ query_embedding   # (N,), BLAS matmul
        best_idx      = int(np.argmax(similarities))
        best_score    = float(similarities[best_idx])

        if best_score >= effective_threshold:
            self.hits += 1
            matched_query  = self._entries[best_idx][1]
            matched_result = copy.deepcopy(self._entries[best_idx][2])
            return matched_result, round(best_score, 4), matched_query

        self.misses += 1
        return None

    def store(
        self,
        embedding: np.ndarray,
        query: str,
        result: dict,
    ) -> None:
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)   # FIFO: remove oldest
            self.evictions += 1
        self._entries.append((
            embedding.copy().astype(np.float32),   # ensure normalized float32
            query,
            copy.deepcopy(result),
        ))
        self.stores += 1

    def clear(self) -> None:
        self._entries.clear()
        self.hits = self.misses = self.stores = self.evictions = 0

    def size(self) -> int:
        return len(self._entries)

    def stored_queries(self) -> list[str]:
        return [e[1] for e in self._entries]


# ──────────────────────────────────────────────────────────────
# 3. Module-level singletons
# ──────────────────────────────────────────────────────────────

_exact_cache    = _LRUCache(max_size=EXACT_CACHE_MAX_SIZE)
_semantic_cache = _SemanticCache(
    max_size=SEMANTIC_CACHE_MAX_SIZE,
    threshold=SEMANTIC_THRESHOLD_DEFAULT,
)


# ──────────────────────────────────────────────────────────────
# 4. Public API functions
# ──────────────────────────────────────────────────────────────

def cache_lookup(query: str) -> dict | None:
    """
    Exact LRU cache lookup.

    Args:
        query: Raw query string (normalized internally).

    Returns:
        Deep copy of cached result dict, or None on miss.
    """
    return _exact_cache.lookup(query)


def cache_store(query: str, result: dict) -> None:
    """
    Store a final response in the exact LRU cache.

    Args:
        query:  Raw query string.
        result: Final response dict from build_final_response().
                Must be the COMPLETE final result — no partial caching.
    """
    _exact_cache.store(query, result)


def semantic_cache_lookup(
    query_embedding: np.ndarray,
    threshold: float = SEMANTIC_THRESHOLD_DEFAULT,
) -> tuple[dict, float, str] | None:
    """
    Semantic near-duplicate cache lookup using dot-product similarity.

    Args:
        query_embedding: L2-normalized 384-dim float32 vector.
        threshold:       Minimum cosine similarity to accept (default 0.98).

    Returns:
        Tuple (result_copy, similarity_score, matched_query) if hit,
        or None if no entry exceeds the threshold.
    """
    return _semantic_cache.lookup(query_embedding, threshold=threshold)


def semantic_cache_store(
    query_embedding: np.ndarray,
    query: str,
    result: dict,
) -> None:
    """
    Store a final response in the semantic cache alongside its embedding.

    Args:
        query_embedding: L2-normalized 384-dim float32 vector from embed_single().
        query:           Original query string (for rationale messages).
        result:          Final response dict from build_final_response().
    """
    _semantic_cache.store(query_embedding, query, result)


def clear_all_caches() -> None:
    """Reset both caches — primarily for testing between test runs."""
    _exact_cache.clear()
    _semantic_cache.clear()


def cache_stats() -> dict:
    """
    Return current cache statistics for monitoring and debugging.

    Returns:
        dict with exact and semantic cache hit/miss/eviction counts and sizes.
    """
    return {
        "exact": {
            "size":       _exact_cache.size(),
            "max_size":   _exact_cache.max_size,
            "hits":       _exact_cache.hits,
            "misses":     _exact_cache.misses,
            "stores":     _exact_cache.stores,
            "evictions":  _exact_cache.evictions,
            "keys":       _exact_cache.keys(),
        },
        "semantic": {
            "size":         _semantic_cache.size(),
            "max_size":     _semantic_cache.max_size,
            "threshold":    _semantic_cache.threshold,
            "hits":         _semantic_cache.hits,
            "misses":       _semantic_cache.misses,
            "stores":       _semantic_cache.stores,
            "evictions":    _semantic_cache.evictions,
            "stored_queries": _semantic_cache.stored_queries(),
        },
    }
