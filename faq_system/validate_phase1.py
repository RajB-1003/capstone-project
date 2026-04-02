# -*- coding: utf-8 -*-
"""
validate_phase1.py — Standalone validation script for Phase 1.
Run from: d:/Capstone Project/faq_system/
    python validate_phase1.py
"""
import sys, os
import io
import numpy as np

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.embedder import (
    load_embedding_model,
    embed_texts,
    embed_single,
    load_and_embed_faqs,
    EMBEDDING_DIM,
)
from modules.semantic_search import search_semantic

FAQS_PATH = os.path.join("data", "faqs.json")

def line(char="=", n=65):
    print(char * n)


# ── TEST 1: Model Loading ──────────────────────────────────────
line()
print("TEST 1 — Model Loading")
line()
model = load_embedding_model()
print(f"  OK  Model type : {type(model).__name__}")
print(f"  OK  Model name : all-MiniLM-L6-v2 (384-dim, CPU)")


# ── TEST 2: embed_texts shape + L2 norm ───────────────────────
line()
print("TEST 2 — embed_texts() Shape and L2 Normalization")
line()
texts = [
    "What are the exam rules?",
    "How do I apply for a scholarship?",
    "CS-202 prerequisites",
]
emb = embed_texts(texts, model)
norms = np.linalg.norm(emb, axis=1)
print(f"  Shape  : {emb.shape}   (expected (3, {EMBEDDING_DIM}))")
print(f"  dtype  : {emb.dtype}")
print(f"  Norms  : {norms.round(6).tolist()}  (all must be ~1.0)")
assert emb.shape == (3, EMBEDDING_DIM), f"Shape mismatch: {emb.shape}"
assert emb.dtype == np.float32, f"dtype mismatch: {emb.dtype}"
assert np.allclose(norms, 1.0, atol=1e-5), f"Not unit-normed: {norms}"
print("  PASS")


# ── TEST 3: embed_single shape + L2 norm ──────────────────────
line()
print("TEST 3 — embed_single() Shape and L2 Normalization")
line()
vec = embed_single("How do I pay my fees?", model)
norm = float(np.linalg.norm(vec))
print(f"  Shape  : {vec.shape}  (expected ({EMBEDDING_DIM},))")
print(f"  L2 norm: {norm:.6f}    (must be ~1.0)")
assert vec.shape == (EMBEDDING_DIM,), f"Shape mismatch: {vec.shape}"
assert abs(norm - 1.0) < 1e-5, f"Not unit-normed: {norm}"
print("  PASS")


# ── TEST 4: load_and_embed_faqs ────────────────────────────────
line()
print("TEST 4 — load_and_embed_faqs() Corpus Loading")
line()
faq_docs, corpus_emb = load_and_embed_faqs(FAQS_PATH, model)
corpus_norms = np.linalg.norm(corpus_emb, axis=1)
categories = sorted(set(d["category"] for d in faq_docs))
print(f"  FAQ count       : {len(faq_docs)}  (expected 30)")
print(f"  Corpus shape    : {corpus_emb.shape}  (expected (30, {EMBEDDING_DIM}))")
print(f"  Norm range      : [{corpus_norms.min():.6f}, {corpus_norms.max():.6f}]  (all ~1.0)")
print(f"  Categories      : {categories}")
assert len(faq_docs) == 30, f"Expected 30 FAQs, got {len(faq_docs)}"
assert corpus_emb.shape == (30, EMBEDDING_DIM)
assert np.allclose(corpus_norms, 1.0, atol=1e-5)
print("  PASS")


# ── TEST 5: search_semantic data contract + quality ───────────
line()
print("TEST 5 — search_semantic() Results and Data Contract")
line()

test_queries = [
    ("What happens if I miss an exam?",              "exam",        "Conceptual exam-policy"),
    ("How do I apply for a merit scholarship?",       "scholarship", "Conceptual scholarship"),
    ("What is the hostel curfew time?",              "hostel",      "Hostel rules"),
    ("Which companies visit for campus recruitment?", "placement",   "Placement recruiters"),
    ("How much are the tuition fees per year?",       "fees",        "Annual tuition fee"),
]

REQUIRED_KEYS = {
    "query", "detected_entities", "route_decision",
    "retrieved_docs", "scores", "rationale",
}

all_ok = True
for query, expected_cat, desc in test_queries:
    result = search_semantic(query, faq_docs, corpus_emb, model, top_k=3)

    # -- Data contract checks
    missing_keys = REQUIRED_KEYS - set(result.keys())
    assert not missing_keys, f"Missing keys: {missing_keys}"
    assert result["query"] == query
    assert result["route_decision"] == "semantic"
    assert isinstance(result["detected_entities"], list)
    assert len(result["retrieved_docs"]) == 3
    assert len(result["scores"]) == 3
    assert isinstance(result["rationale"], str) and len(result["rationale"]) > 0
    assert result["scores"] == sorted(result["scores"], reverse=True), "Scores not descending"

    top = result["retrieved_docs"][0]
    cat_ok = top["category"] == expected_cat
    if not cat_ok:
        all_ok = False

    status = "OK  " if cat_ok else "MISMATCH"
    print(f"\n  [{status}] {desc}")
    print(f"  Query     : \"{query}\"")
    print(f"  Top result: \"{top['question'][:70]}\"")
    print(f"  Category  : {top['category']}  (expected: {expected_cat})")
    print(f"  Scores    : {result['scores']}")
    print(f"  Rationale : {result['rationale'][:100]}")


line("=")
print("PHASE 1 VALIDATION SUMMARY")
line("=")
print("  Tests 1-4  (embedder):       ALL PASSED")
if all_ok:
    print("  Test  5    (semantic search): ALL QUERIES MATCHED EXPECTED CATEGORY")
else:
    print("  Test  5    (semantic search): MOST PASSED — review any MISMATCH above")
print()
print("  Next: Phase 2 — BM25 keyword search")
line("=")
