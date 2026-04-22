"""
cache.py — Phase 7: LRU Exact Cache + Semantic Near-Duplicate Cache
           Upgrade 2: Version-aware cache keys

Responsibilities:
  - Exact LRU cache: O(1) lookup/insert using OrderedDict; max 50 entries
  - Semantic cache: dot-product similarity on L2-normalized query embeddings;
    max 20 entries; threshold 0.90 (semantic similarity queries)
  - Both caches embed CORPUS_VERSION + EMBEDDING_VERSION into keys/metadata
    so stale entries are silently bypassed when the dataset or model changes
  - Module-level singletons so cache persists for the lifetime of the process
  - All public functions operate on the singletons — callers never touch internals

Design decisions:
  - Exact cache key = sha256(normalized_query + version_tag)[:16] hex string.
    The full query is still stored for display; the hash is used for lookup only.
  - Semantic cache entries carry a version_tag string; mismatched entries are
    skipped during lookup (treated as misses) without evicting them immediately.
  - Public API (cache_lookup, cache_store, semantic_cache_lookup,
    semantic_cache_store, clear_all_caches, cache_stats) is UNCHANGED.
    Versioning is entirely internal — no caller changes required.

Public API:
  cache_lookup(query)                    -> dict | None
  cache_store(query, result)
  semantic_cache_lookup(embedding, threshold=0.90)  -> tuple | None
  semantic_cache_store(embedding, query, result)
  clear_all_caches()                     (testing / reset)
  cache_stats()                          -> dict
"""

import copy
import hashlib
from collections import OrderedDict

import numpy as np

from modules.constants import CORPUS_VERSION, EMBEDDING_VERSION


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
EXACT_CACHE_MAX_SIZE       = 50
SEMANTIC_CACHE_MAX_SIZE    = 20
SEMANTIC_THRESHOLD_DEFAULT = 0.90   # ≥ 0.90 cosine similarity → semantic cache hit

# Current version tag — computed once at import time from constants.
# Changing CORPUS_VERSION or EMBEDDING_VERSION in constants.py automatically
# invalidates all existing cache entries on next process start.
_CACHE_VERSION_TAG: str = f"{CORPUS_VERSION}:{EMBEDDING_VERSION}"


def _make_exact_key(query: str, user_id: int | None) -> str:
    """
    Build a version-aware cache key for the exact LRU cache.

    key = sha256(str(user_id) + "|" + normalized_query + "|" + version_tag)[:16]

    The sha256 prefix is used purely for key uniqueness — the original
    query is stored separately in the result dict for display purposes.
    A 16-hex-char prefix (64-bit key space) is collision-free for N ≤ 50.
    """
    payload = f"{user_id}|{query.strip().lower()}|{_CACHE_VERSION_TAG}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────
# 1. LRU exact-match cache
# ──────────────────────────────────────────────────────────────

class _LRUCache:
    """
    OrderedDict-based LRU cache for exact query → result mappings.

    Strategy:
        - On lookup HIT:  move key to end (most recently used).
        - On store:       insert at end; if over capacity, pop first (LRU).
        - Key:            sha256(normalized_query + version_tag)[:16]
                          so entries from a different dataset/model never match.

    Complexity:
        - lookup:  O(1)
        - store:   O(1) amortized
        - evict:   O(1) via popitem(last=False)
    """

    def __init__(self, max_size: int = EXACT_CACHE_MAX_SIZE):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self.max_size  = max_size
        self.hits      = 0
        self.misses    = 0
        self.stores    = 0
        self.evictions = 0

    def lookup(self, query: str, user_id: int | None) -> dict | None:
        key = _make_exact_key(query, user_id)   # version-aware and user-aware key
        if key not in self._store:
            self.misses += 1
            return None
        # Move to end (most-recently used)
        self._store.move_to_end(key)
        self.hits += 1
        return copy.deepcopy(self._store[key])   # deep copy — never return reference

    def store(self, query: str, result: dict, user_id: int | None) -> None:
        key = _make_exact_key(query, user_id)   # version-aware and user-aware key
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
    Vector similarity cache for semantic query matching.

    Each entry stores: (L2-normalized embedding, query_str, result_dict,
                        version_tag).

    Lookup:
        Filter entries to current version_tag first.
        Stack remaining embeddings into (M, 384) matrix.
        Compute dot products against query_vec → (M,) similarities.
        If max similarity ≥ threshold → return cached result.

    Eviction: FIFO — oldest entry (index 0) removed when full.

    Why version_tag check?
        When CORPUS_VERSION or EMBEDDING_VERSION changes, entries from the
        previous run are silently skipped without requiring a cache flush.
        They are evicted naturally via FIFO as new entries are added.

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
        # Each entry: (embedding, query, result, version_tag, user_id)
        self._entries: list[tuple[np.ndarray, str, dict, str, int | None]] = []
        self.hits      = 0
        self.misses    = 0
        self.stores    = 0
        self.evictions = 0

    def lookup(
        self,
        query_embedding: np.ndarray,
        user_id: int | None,
        threshold: float | None = None,
    ) -> tuple[dict, float, str] | None:
        """
        Returns (result_copy, similarity_score, matched_query) or None.
        Version-mismatched entries and cross-user entries are silently skipped.
        """
        effective_threshold = threshold if threshold is not None else self.threshold

        # Filter to current version AND current user only
        valid_entries = [e for e in self._entries if e[3] == _CACHE_VERSION_TAG and e[4] == user_id]
        if not valid_entries:
            self.misses += 1
            return None

        # Stack valid embeddings: (M, 384)
        stored_matrix = np.stack([e[0] for e in valid_entries], axis=0)
        similarities  = stored_matrix @ query_embedding   # (M,), BLAS matmul
        best_idx      = int(np.argmax(similarities))
        best_score    = float(similarities[best_idx])

        if best_score >= effective_threshold:
            self.hits += 1
            matched_query  = valid_entries[best_idx][1]
            matched_result = copy.deepcopy(valid_entries[best_idx][2])
            return matched_result, round(best_score, 4), matched_query

        self.misses += 1
        return None

    def store(
        self,
        embedding: np.ndarray,
        query: str,
        result: dict,
        user_id: int | None,
    ) -> None:
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)   # FIFO: remove oldest
            self.evictions += 1
        self._entries.append((
            embedding.copy().astype(np.float32),   # ensure normalized float32
            query,
            copy.deepcopy(result),
            _CACHE_VERSION_TAG,                    # version tag for invalidation
            user_id,                               # user isolation
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

def cache_lookup(query: str, user_id: int | None) -> dict | None:
    """
    Exact LRU cache lookup.

    Args:
        query: Raw query string (normalized internally).
        user_id: ID of the querying user.

    Returns:
        Deep copy of cached result dict, or None on miss.
    """
    return _exact_cache.lookup(query, user_id)


def cache_store(query: str, result: dict, user_id: int | None) -> None:
    """
    Store a final response in the exact LRU cache.

    Args:
        query:  Raw query string.
        result: Final response dict from build_final_response().
                Must be the COMPLETE final result — no partial caching.
        user_id: ID of the querying user.
    """
    _exact_cache.store(query, result, user_id)


def semantic_cache_lookup(
    query_embedding: np.ndarray,
    user_id: int | None,
    threshold: float = SEMANTIC_THRESHOLD_DEFAULT,
) -> tuple[dict, float, str] | None:
    """
    Semantic near-duplicate cache lookup using dot-product similarity.

    Args:
        query_embedding: L2-normalized 384-dim float32 vector.
        user_id:         ID of the querying user.
        threshold:       Minimum cosine similarity to accept (default 0.98).

    Returns:
        Tuple (result_copy, similarity_score, matched_query) if hit,
        or None if no entry exceeds the threshold.
    """
    return _semantic_cache.lookup(query_embedding, user_id, threshold=threshold)


def semantic_cache_store(
    query_embedding: np.ndarray,
    query: str,
    result: dict,
    user_id: int | None,
) -> None:
    """
    Store a final response in the semantic cache alongside its embedding.

    Args:
        query_embedding: L2-normalized 384-dim float32 vector from embed_single().
        query:           Original query string (for rationale messages).
        result:          Final response dict from build_final_response().
        user_id:         ID of the querying user.
    """
    _semantic_cache.store(query_embedding, query, result, user_id)


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
        "version": _CACHE_VERSION_TAG,   # corpus:embedding version tag
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
            "size":           _semantic_cache.size(),
            "max_size":       _semantic_cache.max_size,
            "threshold":      _semantic_cache.threshold,
            "hits":           _semantic_cache.hits,
            "misses":         _semantic_cache.misses,
            "stores":         _semantic_cache.stores,
            "evictions":      _semantic_cache.evictions,
            "stored_queries": _semantic_cache.stored_queries(),
        },
    }
