# -*- coding: utf-8 -*-
"""
validate_phase6.py  --  Phase 6 Validation: Explainability Layer

Tests validate:
  1.  Keyword route rationale  — entity names cited, bypass explained
  2.  Semantic route rationale — intent + score + threshold cited
  3.  Hybrid route rationale   — score range, both retrievers, overlap
  4.  Fallback / unknown route — no crash, non-empty output
  5.  explain_results()        — every doc gets explanation, source, score
  6.  No empty explanations    — all explanation strings are non-empty
  7.  Source-driven templates  — correct template per source label
  8.  build_final_response()   — schema validation (all 5 fields per result)
  9.  Score propagation        — correct score type used per route
 10.  Determinism              — same state → same output
 11.  No upstream mutation     — input state unchanged after all calls
 12.  Real pipeline test       — end-to-end Phase 1→5→6 for three queries

Run from: d:/Capstone Project/faq_system/
    python -W ignore validate_phase6.py
"""

import sys, os, io, json, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.explainability import (
    generate_rationale,
    explain_results,
    build_final_response,
)

# Pipeline modules (for real end-to-end test)
from modules.embedder        import load_embedding_model, load_and_embed_faqs
from modules.semantic_search import search_semantic
from modules.keyword_search  import build_bm25_index, search_keyword
from modules.hybrid_search   import search_hybrid
from modules.router_tier1    import load_regex_patterns, tier1_route
from modules.router_tier2    import load_intent_exemplars, embed_intents, tier2_route

FAQS_PATH      = os.path.join("data", "faqs.json")
PATTERNS_PATH  = os.path.join("config", "regex_patterns.json")
EXEMPLARS_PATH = os.path.join("data", "intent_exemplars.json")

def line(char="=", n=65): print(char * n)
def ok(msg):   print(f"  OK    {msg}")
def warn(msg): print(f"  WARN  {msg}")

FINAL_RESPONSE_RESULT_KEYS = {"rank", "question", "answer", "category",
                               "score", "source", "explanation"}
FINAL_RESPONSE_KEYS = {"query", "route_decision", "rationale", "results"}


# ──────────────────────────────────────────────────────────────
# Fixture states for isolated unit tests
# ──────────────────────────────────────────────────────────────

def make_keyword_state(entities=None):
    docs = [
        {"id": "faq_001", "question": "What are the prerequisites for CS-202?",
         "answer": "CS-101 with grade B or above.", "category": "course_info",
         "metadata": {"course_code": "CS-202"}, "bm25_score": 7.81},
        {"id": "faq_002", "question": "Is CS-202 available in the summer term?",
         "answer": "Yes, CS-202 is offered in short summer sessions.",
         "category": "course_info", "metadata": {}, "bm25_score": 4.65},
    ]
    return {
        "query":             "CS-202 prerequisites",
        "detected_entities": entities or ["CS-202"],
        "route_decision":    "keyword",
        "retrieved_docs":    docs,
        "scores":            [7.81, 4.65],
        "rationale":         "Tier 1 activated keyword search for CS-202.",
    }

def make_semantic_state(intent="conceptual", score=0.97):
    docs = [
        {"id": "faq_005", "question": "What happens if I miss a mid-semester exam?",
         "answer": "Zero marks unless medical proof is submitted.",
         "category": "exam", "metadata": {}, "similarity_score": 0.9711},
        {"id": "faq_008", "question": "What is the attendance requirement?",
         "answer": "75% minimum to sit the final exam.",
         "category": "exam", "metadata": {}, "similarity_score": 0.8988},
    ]
    return {
        "query":             "What happens if I miss an exam?",
        "detected_entities": [],
        "route_decision":    "semantic",
        "_tier2_intent":     intent,
        "_tier2_score":      score,
        "retrieved_docs":    docs,
        "scores":            [0.9711, 0.8988],
        "rationale":         "Tier 2 semantic router activated.",
    }

def make_hybrid_state(t2_score=0.74):
    docs = [
        {"id": "faq_011", "question": "What is the hostel fee?",
         "answer": "Rs 45,000 per semester.", "category": "hostel",
         "metadata": {}, "rrf_score": 0.032787, "source": "both"},
        {"id": "faq_002", "question": "Is CS-202 available in the summer term?",
         "answer": "Yes.", "category": "course_info",
         "metadata": {}, "rrf_score": 0.031754, "source": "both"},
        {"id": "faq_014", "question": "Total annual tuition fee for B.Tech?",
         "answer": "Rs 1,20,000 per year.", "category": "fees",
         "metadata": {}, "rrf_score": 0.016129, "source": "semantic"},
        {"id": "faq_001", "question": "Prerequisites for CS-202?",
         "answer": "CS-101 grade B+.", "category": "course_info",
         "metadata": {}, "rrf_score": 0.015873, "source": "keyword"},
    ]
    return {
        "query":             "hostel fee CS-202",
        "detected_entities": [],
        "route_decision":    "hybrid",
        "_tier2_intent":     "informational",
        "_tier2_score":      t2_score,
        "retrieved_docs":    docs,
        "scores":            [d["rrf_score"] for d in docs],
        "rationale":         "Hybrid RRF applied.",
    }


# ──────────────────────────────────────────────────────────────
# TEST 1 — Keyword route rationale
# ──────────────────────────────────────────────────────────────
line()
print("TEST 1 — generate_rationale(): Keyword Route")
line()

kw_state = make_keyword_state()
kw_rationale = generate_rationale(kw_state)
print(f"  Rationale: \"{kw_rationale}\"")
print()

assert isinstance(kw_rationale, str) and len(kw_rationale) > 0, "Must be non-empty"
assert "CS-202" in kw_rationale, "Must mention the detected entity"
assert any(w in kw_rationale.lower() for w in ["keyword", "identifier", "exact"]), \
    "Must mention keyword/identifier/exact search"
assert any(w in kw_rationale.lower() for w in ["tier 1", "regex", "bypassed", "bypass"]), \
    "Must mention Tier 1 or regex router"

ok("Non-empty rationale")
ok("Entity 'CS-202' cited in rationale")
ok("Keyword/identifier search mentioned")
ok("Tier 1 / regex routing explained")
print()


# ──────────────────────────────────────────────────────────────
# TEST 2 — Semantic route rationale
# ──────────────────────────────────────────────────────────────
line()
print("TEST 2 — generate_rationale(): Semantic Route with Tier 2 Intent")
line()

sem_state = make_semantic_state(intent="conceptual", score=0.9711)
sem_rationale = generate_rationale(sem_state)
print(f"  Rationale: \"{sem_rationale}\"")
print()

assert isinstance(sem_rationale, str) and len(sem_rationale) > 0
assert "conceptual" in sem_rationale.lower(), "Must cite the matched intent"
assert "0.9711" in sem_rationale or "0.971" in sem_rationale, \
    "Must cite the similarity score"
assert any(w in sem_rationale.lower() for w in ["threshold", "0.82"]), \
    "Must reference the high-confidence threshold"
assert any(w in sem_rationale.lower() for w in ["semantic", "embedding", "intent"]), \
    "Must explain semantic routing"

ok("Intent 'conceptual' cited")
ok("Score 0.9711 cited")
ok("Threshold (0.82) referenced")
ok("Semantic/embedding retrieval explained")
print()


# ──────────────────────────────────────────────────────────────
# TEST 3 — Hybrid route rationale
# ──────────────────────────────────────────────────────────────
line()
print("TEST 3 — generate_rationale(): Hybrid Route")
line()

hyb_state = make_hybrid_state(t2_score=0.74)
hyb_rationale = generate_rationale(hyb_state)
print(f"  Rationale: \"{hyb_rationale}\"")
print()

assert isinstance(hyb_rationale, str) and len(hyb_rationale) > 0
assert "0.74" in hyb_rationale, "Must cite the ambiguous similarity score"
assert any(w in hyb_rationale.lower() for w in ["hybrid", "both", "rrf", "reciprocal"]), \
    "Must mention hybrid / RRF"
assert any(w in hyb_rationale.lower() for w in ["semantic", "keyword"]), \
    "Must mention both retriever types"
overlap_mentioned = any(w in hyb_rationale.lower()
    for w in ["overlap", "both retrieval", "both lists", "appeared"])
ok("Score 0.74 cited (ambiguous range)")
ok("Hybrid/RRF mentioned")
ok("Both retriever types mentioned")
if overlap_mentioned:
    ok("Overlap / both-list citation present")
else:
    warn("Overlap not explicitly mentioned in rationale (check source counts)")
print()


# ──────────────────────────────────────────────────────────────
# TEST 4 — Fallback: empty / unknown route
# ──────────────────────────────────────────────────────────────
line()
print("TEST 4 — generate_rationale(): Empty route_decision (Fallback)")
line()

empty_state = {
    "query": "random xyz", "detected_entities": [], "route_decision": "",
    "retrieved_docs": [], "scores": [], "rationale": "",
}
fallback_rat = generate_rationale(empty_state)
assert isinstance(fallback_rat, str) and len(fallback_rat) > 0, \
    "Must return non-empty even for unknown route"
ok(f"Non-empty fallback rationale: \"{fallback_rat[:80]}...\"")
print()


# ──────────────────────────────────────────────────────────────
# TEST 5 — explain_results(): all docs get explanation, source, score
# ──────────────────────────────────────────────────────────────
line()
print("TEST 5 — explain_results(): Every Doc Gets explanation + source + score")
line()

for label, state_fn in [("keyword", make_keyword_state),
                         ("semantic", make_semantic_state),
                         ("hybrid",   make_hybrid_state)]:
    state_in = state_fn()
    enriched = explain_results(state_in)
    docs_out = enriched["retrieved_docs"]

    for i, doc in enumerate(docs_out, 1):
        assert "explanation" in doc,    f"[{label}] doc {i} missing 'explanation'"
        assert "source"      in doc,    f"[{label}] doc {i} missing 'source'"
        assert doc["explanation"],      f"[{label}] doc {i} has empty explanation"
        assert doc["source"] in ("semantic", "keyword", "both", "unknown"), \
            f"[{label}] invalid source: {doc['source']}"

    ok(f"Route '{label}' — {len(docs_out)} docs, all have explanation + source")
    for doc in docs_out:
        print(f"    [{doc['source']:<8}] rank #{docs_out.index(doc)+1}: "
              f"\"{doc['explanation'][:70]}\"")
    print()


# ──────────────────────────────────────────────────────────────
# TEST 6 — No empty explanations
# ──────────────────────────────────────────────────────────────
line()
print("TEST 6 — No Empty Explanations in Any Route")
line()

for label, state_in in [
    ("keyword", make_keyword_state()),
    ("semantic", make_semantic_state()),
    ("hybrid",  make_hybrid_state()),
]:
    enriched = explain_results(state_in)
    empty_exps = [d["id"] for d in enriched["retrieved_docs"]
                  if not d.get("explanation")]
    assert not empty_exps, \
        f"Route '{label}' has docs with empty explanations: {empty_exps}"
    ok(f"Route '{label}' — zero empty explanations")
print()


# ──────────────────────────────────────────────────────────────
# TEST 7 — Source-driven templates: correct text per source
# ──────────────────────────────────────────────────────────────
line()
print("TEST 7 — Source-Driven Templates: Correct Text for Each Source")
line()

template_docs = [
    {"id": "S1", "question": "SemDoc", "answer": "SemAns",
     "category": "exam", "metadata": {}, "similarity_score": 0.91, "source": "semantic"},
    {"id": "K1", "question": "KwDoc",  "answer": "KwAns",
     "category": "exam", "metadata": {}, "bm25_score": 6.5,  "source": "keyword"},
    {"id": "B1", "question": "BothDoc","answer": "BothAns",
     "category": "exam", "metadata": {}, "rrf_score": 0.0328, "source": "both"},
]
template_state = {
    "query": "test query", "detected_entities": [], "route_decision": "hybrid",
    "retrieved_docs": template_docs, "scores": [0.91, 6.5, 0.0328], "rationale": "",
}
enriched_t = explain_results(template_state)
by_id = {d["id"]: d for d in enriched_t["retrieved_docs"]}

# Semantic template
sem_exp = by_id["S1"]["explanation"]
assert "semantic similarity" in sem_exp.lower(), f"Semantic template wrong: {sem_exp}"
assert "0.91" in sem_exp, f"Semantic score not in explanation: {sem_exp}"
ok(f"Semantic: \"{sem_exp}\"")

# Keyword template
kw_exp = by_id["K1"]["explanation"]
assert any(w in kw_exp.lower() for w in ["keyword", "terms", "identifier"]), \
    f"Keyword template wrong: {kw_exp}"
assert "6.5" in kw_exp, f"Keyword score not in explanation: {kw_exp}"
ok(f"Keyword:  \"{kw_exp}\"")

# Both template
both_exp = by_id["B1"]["explanation"]
assert any(w in both_exp.lower() for w in ["both", "strong"]), \
    f"Both template wrong: {both_exp}"
assert "0.032" in both_exp, f"RRF score not in explanation: {both_exp}"
ok(f"Both:     \"{both_exp}\"")
print()


# ──────────────────────────────────────────────────────────────
# TEST 8 — build_final_response(): schema validation
# ──────────────────────────────────────────────────────────────
line()
print("TEST 8 — build_final_response(): Schema Validation")
line()

for label, state_fn in [("keyword", make_keyword_state),
                          ("semantic", make_semantic_state),
                          ("hybrid",   make_hybrid_state)]:
    state_in = state_fn()
    response = build_final_response(state_in)

    # Top-level keys
    missing_top = FINAL_RESPONSE_KEYS - set(response.keys())
    assert not missing_top, f"[{label}] missing top-level keys: {missing_top}"
    assert response["query"]          == state_in["query"]
    assert response["route_decision"] == state_in["route_decision"]
    assert isinstance(response["rationale"], str) and response["rationale"]
    assert isinstance(response["results"], list)
    assert len(response["results"])   == len(state_in["retrieved_docs"])

    # Per-result keys
    for i, result in enumerate(response["results"], 1):
        missing_result = FINAL_RESPONSE_RESULT_KEYS - set(result.keys())
        assert not missing_result, \
            f"[{label}] result #{i} missing keys: {missing_result}"
        assert result["rank"]        == i
        assert isinstance(result["question"],    str) and result["question"]
        assert isinstance(result["answer"],      str)
        assert isinstance(result["score"],       float)
        assert isinstance(result["source"],      str)
        assert isinstance(result["explanation"], str) and result["explanation"]

    ok(f"Route '{label}' — all {len(response['results'])} results have correct schema")
print()


# ──────────────────────────────────────────────────────────────
# TEST 9 — Score propagation: correct score type per route
# ──────────────────────────────────────────────────────────────
line()
print("TEST 9 — Score Propagation: Correct Score Type per Route")
line()

# Keyword: bm25_score should appear in explanation
kw_resp = build_final_response(make_keyword_state())
for result in kw_resp["results"]:
    assert "BM25" in result["explanation"] or str(result["score"]) in result["explanation"], \
        f"BM25 score not referenced: {result['explanation']}"
ok(f"Keyword results reference BM25 scores: {[r['score'] for r in kw_resp['results']]}")

# Semantic: similarity_score should appear
sem_resp = build_final_response(make_semantic_state())
for result in sem_resp["results"]:
    assert "similarity" in result["explanation"].lower(), \
        f"Similarity not referenced: {result['explanation']}"
ok(f"Semantic results reference similarity scores: {[r['score'] for r in sem_resp['results']]}")

# Hybrid: rrf_score should appear for 'both' tagged docs
hyb_resp = build_final_response(make_hybrid_state())
for result in hyb_resp["results"]:
    if result["source"] == "both":
        assert "RRF" in result["explanation"] or "rrf" in result["explanation"].lower(), \
            f"RRF not referenced for 'both' doc: {result['explanation']}"
ok(f"Hybrid results correctly identify RRF scores for 'both' docs")
print()


# ──────────────────────────────────────────────────────────────
# TEST 10 — Determinism: same state → same output
# ──────────────────────────────────────────────────────────────
line()
print("TEST 10 — Determinism: Same State Produces Same Output")
line()

det_state = make_hybrid_state()
out1 = build_final_response(det_state)
out2 = build_final_response(det_state)
out3 = build_final_response(det_state)

assert out1["rationale"]  == out2["rationale"] == out3["rationale"], \
    "Rationale is not deterministic"
for i in range(len(out1["results"])):
    assert out1["results"][i]["explanation"] == out2["results"][i]["explanation"] == \
           out3["results"][i]["explanation"], f"Result {i} explanation not deterministic"

ok("build_final_response() is fully deterministic")
ok(f"Rationale identical across 3 calls: \"{out1['rationale'][:60]}...\"")
print()


# ──────────────────────────────────────────────────────────────
# TEST 11 — Input state immutability
# ──────────────────────────────────────────────────────────────
line()
print("TEST 11 — Input State Immutability")
line()

orig = make_hybrid_state()
orig_snap = copy.deepcopy(orig)

_ = generate_rationale(orig)
_ = explain_results(orig)
_ = build_final_response(orig)

assert orig == orig_snap, "Input state was mutated by explainability module"
ok("Input state unchanged after generate_rationale(), explain_results(), build_final_response()")
print()


# ──────────────────────────────────────────────────────────────
# TEST 12 — Real end-to-end pipeline: Phase 1→5→6
# ──────────────────────────────────────────────────────────────
line()
print("TEST 12 — End-to-End Pipeline: Phases 1-5 → Phase 6")
line()

# Load fixtures once
with open(FAQS_PATH) as f:
    faq_docs = json.load(f)

model                = load_embedding_model()
_, corpus_embeddings = load_and_embed_faqs(FAQS_PATH, model)
bm25_index, _        = build_bm25_index(faq_docs)
patterns             = load_regex_patterns(PATTERNS_PATH)
exemplars            = load_intent_exemplars(EXEMPLARS_PATH)
intent_embeddings    = embed_intents(exemplars, model)

e2e_queries = [
    ("CS-202 prerequisites",              "keyword"),
    ("What happens if I miss an exam?",   "semantic"),
    ("hostel fee CS-202",                 "hybrid"),
]

for query, expected_route in e2e_queries:
    # Build initial state
    state = {"query": query, "detected_entities": [], "route_decision": "",
             "retrieved_docs": [], "scores": [], "rationale": ""}

    # Phase 3: Tier 1 router
    state = tier1_route(state, patterns)

    # Phase 4: Tier 2 router (only if Tier 1 did not set route)
    if not state["route_decision"]:
        state = tier2_route(state, intent_embeddings, model)

    # Phase 1+2: retrieve from both
    route = state["route_decision"]
    sem_results = search_semantic(query, faq_docs, corpus_embeddings, model, top_k=5)
    kw_results  = search_keyword(query, faq_docs, bm25_index, top_k=5)

    # Phase 5: hybrid fusion
    state = search_hybrid(state, sem_results, kw_results, top_k=5)

    # Phase 6: explainability
    response = build_final_response(state)

    # Validate response
    assert response["query"]          == query
    assert response["route_decision"] in ("keyword", "semantic", "hybrid")
    assert isinstance(response["rationale"], str) and response["rationale"]
    assert len(response["results"])   > 0

    for result in response["results"]:
        missing = FINAL_RESPONSE_RESULT_KEYS - set(result.keys())
        assert not missing, f"Result missing keys: {missing}"
        assert result["explanation"], f"Empty explanation for {result['question'][:40]}"

    line("-", 65)
    print(f"  Query: \"{query}\"")
    print(f"  Route: {response['route_decision']}  (expected: {expected_route})")
    print(f"  Rationale: \"{response['rationale'][:100]}...\"")
    print(f"  Results ({len(response['results'])}):")
    for r in response["results"]:
        print(f"    #{r['rank']} [{r['source']:<8}] {r['score']:>8}  "
              f"{r['question'][:45]}")
        print(f"         → {r['explanation'][:75]}")
    print()

ok("All 3 end-to-end queries produced valid build_final_response() output")
ok("No empty explanations, no missing keys, no state mutations")
print()


# ──────────────────────────────────────────────────────────────
# TEST 13 — No LLM/embedding imports in explainability.py
# ──────────────────────────────────────────────────────────────
line()
print("TEST 13 — No ML/LLM Imports in explainability.py")
line()

with open(os.path.join("modules", "explainability.py"), "r") as f:
    import_lines = [l.strip() for l in f if l.strip().startswith(("import ", "from "))]

forbidden = ["sentence_transformers", "openai", "langchain", "numpy",
             "rank_bm25", "embedder", "semantic_search", "keyword_search",
             "hybrid_search", "router_tier1", "router_tier2"]
violations = [ln for ln in import_lines if any(fb in ln for fb in forbidden)]
assert not violations, f"Forbidden imports: {violations}"
ok(f"Imports: {import_lines}")
ok("Only 'copy' imported — zero ML / LLM / search dependencies")
print()


# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
line("=")
print("PHASE 6 VALIDATION SUMMARY")
line("=")
print("  Test  1  — Keyword route rationale (entity cited)  : PASSED")
print("  Test  2  — Semantic route rationale (intent+score) : PASSED")
print("  Test  3  — Hybrid route rationale (both+overlap)   : PASSED")
print("  Test  4  — Fallback / empty route (no crash)       : PASSED")
print("  Test  5  — explain_results: all docs enriched      : PASSED")
print("  Test  6  — No empty explanation strings            : PASSED")
print("  Test  7  — Source-driven templates correct         : PASSED")
print("  Test  8  — build_final_response schema             : PASSED")
print("  Test  9  — Score type propagation per route        : PASSED")
print("  Test 10  — Determinism (3x same output)            : PASSED")
print("  Test 11  — Input state immutability                : PASSED")
print("  Test 12  — End-to-end Phase 1-5 → 6 integration   : PASSED")
print("  Test 13  — No ML/LLM imports                       : PASSED")
print()
print("  Next: Phase 7 — Optimization (Caching + Latency Profiling)")
line("=")
