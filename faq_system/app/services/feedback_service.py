"""
evaluation.py — Upgrade 5: Lightweight Evaluation Module

Responsibilities:
  - Accept a list of test queries with expected answer patterns
  - Run each query through the live pipeline
  - Compute Top-1 Accuracy and MRR (Mean Reciprocal Rank)
  - Return a structured evaluation report

Design decisions:
  - Matching is substring-based on the question field (case-insensitive).
    This is intentional: exact FAQ IDs are fragile to data edits, while
    question substrings are stable and evaluator-readable.
  - The built-in test set covers all three route types (keyword, semantic,
    hybrid) plus edge cases; it does not hard-code expected route values
    because routing is a function of the live signal scores.
  - No external datasets required — test set is embedded in this module and
    can be extended by editing the TEST_CASES list.
  - The pipeline is called through a callable argument so this module never
    imports pipeline.py at module load time (avoids model loading on import).

Public API:
  run_evaluation(pipeline_fn, pipeline_kwargs, test_cases=None) -> dict
  TEST_CASES   — default built-in test set (list of dicts)
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Built-in test set (12 queries, covers keyword / semantic / hybrid)
#
# Each entry:
#   query                      — the input query string
#   expected_question_fragment — a substring that MUST appear (case-insensitive)
#                                in the question field of a top-5 result to
#                                count as a "hit".
# ──────────────────────────────────────────────────────────────

TEST_CASES: list[dict] = [
    # ── Keyword route (course-code entity) ──────────────────────
    {
        "query": "What are the prerequisites for CS-202?",
        "expected_question_fragment": "prerequisite",
    },
    {
        "query": "CS-301 exam schedule",
        "expected_question_fragment": "exam",
    },
    {
        "query": "CS-101 attendance policy",
        "expected_question_fragment": "attendance",
    },
    # ── Semantic route (conceptual intent, no course code) ───────
    {
        "query": "What happens if I miss a final exam?",
        "expected_question_fragment": "miss",
    },
    {
        "query": "How can I appeal a grade?",
        "expected_question_fragment": "appeal",
    },
    {
        "query": "Is there a penalty for plagiarism?",
        "expected_question_fragment": "plagiarism",
    },
    {
        "query": "How do I apply for a scholarship?",
        "expected_question_fragment": "scholarship",
    },
    # ── Hybrid route (entity + conceptual intent) ────────────────
    {
        "query": "CS-202 attendance and exam eligibility",
        "expected_question_fragment": "attendance",
    },
    {
        "query": "hostel fee CS-301",
        "expected_question_fragment": "hostel",
    },
    # ── Edge cases ───────────────────────────────────────────────
    {
        "query": "library working hours",
        "expected_question_fragment": "library",
    },
    {
        "query": "internship placement process",
        "expected_question_fragment": "placement",
    },
    {
        "query": "elective course registration deadline",
        "expected_question_fragment": "registration",
    },
]


# ──────────────────────────────────────────────────────────────
# Matching helper
# ──────────────────────────────────────────────────────────────

def _matches(result: dict, fragment: str) -> bool:
    """
    Return True if the result's question field contains the expected fragment
    (case-insensitive). Falls back to checking the answer field.
    """
    fragment_lower = fragment.lower()
    question = result.get("question", "").lower()
    answer   = result.get("answer",   "").lower()
    return fragment_lower in question or fragment_lower in answer


# ──────────────────────────────────────────────────────────────
# Core evaluation function
# ──────────────────────────────────────────────────────────────

def run_evaluation(
    pipeline_fn,
    pipeline_kwargs: dict,
    test_cases: list[dict] | None = None,
) -> dict:
    """
    Run evaluation over a test set and return accuracy + MRR.

    Args:
        pipeline_fn:      Callable — the run_pipeline function.
        pipeline_kwargs:  kwargs to pass to pipeline_fn (model, corpus_embeddings, etc.)
                          Do NOT include `query` — it is provided per test case.
        test_cases:       List of {"query": str, "expected_question_fragment": str}.
                          Defaults to the built-in TEST_CASES if None.

    Returns:
        {
            "accuracy":    float,   # Top-1 Accuracy (fraction of queries where rank-1 hit)
            "mrr":         float,   # Mean Reciprocal Rank (first hit in top-5)
            "num_queries": int,
            "per_query":   [
                {
                    "query":      str,
                    "route":      str,
                    "top_result": str,   # question text of rank-1 result
                    "hit":        bool,  # True if expected fragment found in top-5
                    "rank":       int,   # 1-indexed rank of first hit (0 if not found)
                    "latency_ms": float,
                },
                ...
            ]
        }
    """
    if test_cases is None:
        test_cases = TEST_CASES

    hits       = 0
    rr_sum     = 0.0  # sum of reciprocal ranks for MRR
    per_query  = []

    for tc in test_cases:
        query    = tc["query"]
        fragment = tc["expected_question_fragment"]

        t_start = time.perf_counter()
        try:
            resp = pipeline_fn(query=query, **pipeline_kwargs)
        except Exception as exc:
            logger.warning("Evaluation pipeline error for query %r: %s", query, exc)
            per_query.append({
                "query":      query,
                "route":      "error",
                "top_result": "",
                "hit":        False,
                "rank":       0,
                "latency_ms": 0.0,
            })
            continue
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        results    = resp.get("results", [])
        route      = resp.get("route_decision", "")
        top_result = results[0].get("question", "") if results else ""

        # Find first matching rank (1-indexed; 0 = not found in top-5)
        first_rank = 0
        for rank, r in enumerate(results, start=1):
            if _matches(r, fragment):
                first_rank = rank
                break

        hit = first_rank > 0
        if hit:
            hits    += 1
            rr_sum  += 1.0 / first_rank

        per_query.append({
            "query":      query,
            "route":      route,
            "top_result": top_result,
            "hit":        hit,
            "rank":       first_rank,
            "latency_ms": round(elapsed_ms, 2),
        })

    n         = len(test_cases)
    accuracy  = round(hits / n, 4) if n > 0 else 0.0
    mrr       = round(rr_sum / n, 4) if n > 0 else 0.0

    return {
        "accuracy":    accuracy,
        "mrr":         mrr,
        "num_queries": n,
        "per_query":   per_query,
    }
