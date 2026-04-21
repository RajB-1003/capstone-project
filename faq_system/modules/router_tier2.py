"""
router_tier2.py — Phase 4: Tier 2 Semantic Intent Router

Responsibilities:
  - Load intent exemplar queries from external JSON (data/intent_exemplars.json)
  - Embed all exemplar queries into L2-normalized matrices (one per intent)
  - At query time: embed the query and compute dot-product similarity against
    every exemplar matrix using vectorized NumPy operations (BLAS)
  - Apply dual thresholds to determine routing decision (no hard-coded keywords)
  - ONLY activates if Tier 1 has NOT already set route_decision

Architecture mandate (Phased Agent Prompt Chain Design.docx):
  - Uses sentence-transformers/all-MiniLM-L6-v2 embeddings (same model as Phase 1)
  - Similarity: dot product on pre-normalized vectors (NOT raw cosine division)
  - Threshold HIGH  = 0.82 → route_decision = "semantic"
  - Threshold LOW   = 0.65 → route_decision = "hybrid"
  - Below LOW threshold → route_decision left unchanged (truly uncertain)
  - Execution budget: <50ms (embed query: ~100-300ms first call, <10ms thereafter)
  - All intent logic comes from exemplar data — no hardcoded intent-to-route mappings

Design decisions:
  - embed_intents() returns per-intent matrices stacked as (N_i, 384) float32 arrays
    so classify_intent() uses a single BLAS matmul per intent (no Python loops)
  - classify_intent() takes the MAX similarity across all exemplars per intent,
    then selects the intent with the global highest max-score
  - tier2_route() guards: if state["route_decision"] is non-empty, returns unchanged
  - Returns a new state dict copy (NEVER mutates the input state)

Function signatures (defined before implementation — architecture mandate):

  load_intent_exemplars(path: str)
      -> dict[str, list[str]]
      Loads raw exemplar query lists per intent from JSON.

  embed_intents(exemplars: dict[str, list[str]], model: SentenceTransformer)
      -> dict[str, np.ndarray]
      Embeds each intent's exemplar list; returns {intent: (N_i, 384) L2-normed matrix}.

  classify_intent(query: str, intent_embeddings: dict[str, np.ndarray], model)
      -> tuple[str, float]
      Embeds query; computes max cosine-equivalent similarity per intent;
      returns (best_intent_name, best_score).

  tier2_route(state: dict, intent_embeddings: dict[str, np.ndarray], model)
      -> dict
      Applies dual-threshold routing; updates route_decision and rationale.

Public API:
  load_intent_exemplars(path)                        -> dict[str, list[str]]
  embed_intents(exemplars, model)                    -> dict[str, np.ndarray]
  classify_intent(query, intent_embeddings, model)   -> tuple[str, float]
  tier2_route(state, intent_embeddings, model)       -> dict
"""

import json
import copy
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from modules.embedder import embed_single, embed_texts
from modules.constants import THRESHOLD_HIGH, THRESHOLD_LOW



# ──────────────────────────────────────────────────────────────
# 1. Exemplar loader
# ──────────────────────────────────────────────────────────────

def load_intent_exemplars(path: str) -> dict[str, list[str]]:
    """
    Load intent exemplar queries from a JSON file.

    Expected JSON format:
        {
            "conceptual":   ["query1", "query2", ...],
            "informational": ["query1", ...],
            "procedural":   ["query1", ...]
        }
    Keys starting with "_" (comment keys) are ignored.

    Args:
        path: Absolute or relative path to intent_exemplars.json.

    Returns:
        dict[str, list[str]]: Maps intent_name → list of exemplar query strings.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If any intent's value is not a list of strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    exemplars: dict[str, list[str]] = {}

    for intent, queries in raw.items():
        if intent.startswith("_"):
            continue   # skip comment/metadata keys
        if not isinstance(queries, list):
            raise ValueError(
                f"Intent '{intent}' must map to a list of strings, "
                f"got {type(queries).__name__}."
            )
        exemplars[intent] = [str(q) for q in queries]

    return exemplars


# ──────────────────────────────────────────────────────────────
# 2. Intent embedding builder
# ──────────────────────────────────────────────────────────────

def embed_intents(
    exemplars: dict[str, list[str]],
    model: SentenceTransformer,
) -> dict[str, np.ndarray]:
    """
    Embed all intent exemplar queries and return per-intent L2-normalized matrices.

    Why matrices instead of individual vectors?
        Storing exemplars as (N_i, 384) matrices allows classify_intent() to
        compute ALL exemplar similarities in ONE BLAS matmul call per intent:
            similarities = intent_matrix @ query_vec   # shape (N_i,)
        This is vastly faster than a Python loop over individual vectors.

    Normalization:
        Each row vector is L2-normalized so that dot product == cosine similarity.
        This matches the normalization applied in Phase 1 (embedder.py).

    Args:
        exemplars: Dict of {intent_name: [query_str, ...]}, from load_intent_exemplars().
        model:     Loaded SentenceTransformer (all-MiniLM-L6-v2).

    Returns:
        dict[str, np.ndarray]: Maps intent_name → (N_i, 384) float32 matrix.
        Each row is an L2-normalized embedding of one exemplar query.
    """
    intent_embeddings: dict[str, np.ndarray] = {}

    for intent, query_list in exemplars.items():
        if not query_list:
            raise ValueError(f"Intent '{intent}' has an empty exemplar list.")

        # embed_texts returns (N, 384) float32, L2-normalized — from Phase 1
        matrix = embed_texts(query_list, model)   # shape: (N_i, 384)
        intent_embeddings[intent] = matrix

    return intent_embeddings


# ──────────────────────────────────────────────────────────────
# 3. Intent classifier
# ──────────────────────────────────────────────────────────────

def classify_intent(
    query: str,
    intent_embeddings: dict[str, np.ndarray],
    model: SentenceTransformer,
) -> tuple[str, float]:
    """
    Classify the semantic intent of a query using dot-product similarity.

    Algorithm:
        1. Embed query → query_vec (shape: (384,), L2-normalized).
        2. For each intent:
               similarities = intent_matrix @ query_vec   # shape (N_i,) — BLAS matmul
               intent_score = float(similarities.max())   # best exemplar match
        3. Return the intent with the highest intent_score and that score.

    Why MAX (not mean)?
        A query only needs to closely match ONE exemplar in the intent group
        to be classified as that intent. Mean would dilute the signal when
        only a subset of exemplars are relevant to the specific query.

    Architecture note:
        Vectors are pre-normalized → dot product equals cosine similarity.
        No division required. Single BLAS call per intent (not a Python loop).

    Args:
        query:             Raw query string from the user.
        intent_embeddings: Output of embed_intents() — per-intent normed matrices.
        model:             Loaded SentenceTransformer for query encoding.

    Returns:
        tuple[str, float]:
            best_intent — name of the intent with highest max-exemplar similarity.
            best_score  — float in [-1, 1], typically [0.6, 1.0] for real queries.
    """
    # ── Step 1: embed query ────────────────────────────────────
    query_vec = embed_single(query, model)   # shape (384,), L2-normed

    # ── Step 2: compute max similarity per intent ──────────────
    intent_scores: dict[str, float] = {}

    for intent, matrix in intent_embeddings.items():
        # BLAS matmul: (N_i, 384) @ (384,) → (N_i,) similarities
        similarities = matrix @ query_vec          # vectorized, no Python loop
        intent_scores[intent] = float(similarities.max())

    # ── Step 3: select best intent ────────────────────────────
    best_intent = max(intent_scores, key=lambda k: intent_scores[k])
    best_score  = intent_scores[best_intent]

    return best_intent, best_score


# ──────────────────────────────────────────────────────────────
# 4. Tier 2 routing function
# ──────────────────────────────────────────────────────────────

def tier2_route(
    state: dict,
    intent_embeddings: dict[str, np.ndarray],
    model: SentenceTransformer,
) -> dict:
    """
    Apply Tier 2 semantic intent routing to the shared AgentState.

    Guard clause:
        If state["route_decision"] is already set (non-empty string), return
        state unchanged. This is the Tier 1 hand-off contract:
        Tier 1 sets route_decision="keyword" → Tier 2 must not override it.

    Routing thresholds (from architecture doc):
        score >= THRESHOLD_HIGH (0.82) → route_decision = "semantic"
            Rationale: "Conceptual search activated based on high similarity
                        ({score:.4f}) to {intent}-related queries."

        THRESHOLD_LOW (0.65) <= score < THRESHOLD_HIGH → route_decision = "hybrid"
            Rationale: "Hybrid search deployed — moderate similarity ({score:.4f})
                        to {intent} intent; using both retrievers."

        score < THRESHOLD_LOW (0.65) → route_decision left unchanged
            Rationale: "Intent uncertain (score {score:.4f} below threshold 0.65).
                        Query will be handled by fallback."

    Args:
        state:             Shared state dict conforming to data contract.
        intent_embeddings: Output of embed_intents().
        model:             Loaded SentenceTransformer.

    Returns:
        dict: Updated state copy. Input state is NEVER mutated.
              Modified fields (when routing occurs):
                  route_decision — "semantic" | "hybrid" | unchanged
                  rationale      — human-readable explanation
              Diagnostic fields always written:
                  _tier2_intent        — best matching intent name
                  _tier2_score         — similarity score (float)
                  _tier2_latency_ms    — classification time in ms
    """
    # ── Guard: Tier 1 already routed — do not override ────────
    existing_route = state.get("route_decision", "")
    if existing_route:
        return state   # fast path — no copy needed, state returned as-is

    # ── Time the full classification ───────────────────────────
    t_start = time.perf_counter()

    query = state.get("query", "")
    best_intent, score = classify_intent(query, intent_embeddings, model)

    t_elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ── Build updated state copy ───────────────────────────────
    new_state = copy.deepcopy(state)
    new_state["_tier2_intent"]      = best_intent
    new_state["_tier2_score"]       = round(score, 4)
    new_state["_tier2_latency_ms"]  = round(t_elapsed_ms, 2)

    # ── Apply dual-threshold routing ───────────────────────────
    if score >= THRESHOLD_HIGH:
        new_state["route_decision"] = "semantic"
        new_state["rationale"] = (
            f"Tier 2 (Semantic): Conceptual search activated based on high "
            f"similarity ({score:.4f}) to '{best_intent}'-related queries. "
            f"Threshold met: {score:.4f} >= {THRESHOLD_HIGH}. "
            f"Classification latency: {t_elapsed_ms:.2f}ms."
        )

    elif score >= THRESHOLD_LOW:
        new_state["route_decision"] = "hybrid"
        new_state["rationale"] = (
            f"Tier 2 (Semantic): Hybrid search deployed — moderate similarity "
            f"({score:.4f}) to '{best_intent}' intent. "
            f"Score in ambiguous range [{THRESHOLD_LOW}, {THRESHOLD_HIGH}). "
            f"Both retrievers will be used. "
            f"Classification latency: {t_elapsed_ms:.2f}ms."
        )

    else:
        # Below both thresholds — intent is uncertain; leave route_decision unset
        new_state["rationale"] = (
            f"Tier 2 (Semantic): Intent uncertain. "
            f"Best match: '{best_intent}' with score {score:.4f} "
            f"(below threshold {THRESHOLD_LOW}). "
            f"Route decision deferred. "
            f"Classification latency: {t_elapsed_ms:.2f}ms."
        )
        # route_decision remains "" — caller handles fallback

    return new_state
