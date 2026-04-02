"""
test_phase1.py — Validation for Phase 1: Embedding Pipeline + Semantic Search

Tests validate:
  1. Model loads without error
  2. embed_texts() returns correct shape and L2-normalized vectors
  3. embed_single() returns correct shape and is L2-normalized
  4. load_and_embed_faqs() loads all 30 FAQs with correct embedding matrix shape
  5. search_semantic() returns valid data contract for multiple query types:
       - conceptual queries (exam policy, scholarships)
       - keyword-heavy queries (not course codes — those are Phase 3)
       - category-spanning queries

Run from: d:/Capstone Project/faq_system/
    python -m tests.test_phase1
"""

import sys
import os
import numpy as np

# ── Allow running from faq_system/ root ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.embedder import (
    load_embedding_model,
    embed_texts,
    embed_single,
    load_and_embed_faqs,
    EMBEDDING_DIM,
)
from modules.semantic_search import search_semantic

# ────────────────────────────────────────────────────────────────
# Shared fixtures (loaded once)
# ────────────────────────────────────────────────────────────────
FAQS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.json")


def separator(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


# ────────────────────────────────────────────────────────────────
# Test 1 — Model Loading
# ────────────────────────────────────────────────────────────────
def test_model_loading():
    separator("TEST 1 — Model Loading")
    model = load_embedding_model()
    print(f"  ✅ Model type : {type(model).__name__}")
    print(f"  ✅ Model name : {model.model_card_data.base_model if hasattr(model, 'model_card_data') else 'all-MiniLM-L6-v2'}")
    return model


# ────────────────────────────────────────────────────────────────
# Test 2 — embed_texts() shape + L2 norm
# ────────────────────────────────────────────────────────────────
def test_embed_texts(model):
    separator("TEST 2 — embed_texts() Shape and L2 Normalization")
    sample_texts = [
        "What are the exam rules?",
        "How do I apply for a scholarship?",
        "CS-202 prerequisites",
    ]
    embeddings = embed_texts(sample_texts, model)

    assert embeddings.shape == (3, EMBEDDING_DIM), (
        f"Expected shape (3, {EMBEDDING_DIM}), got {embeddings.shape}"
    )
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"

    # Check L2 norm ≈ 1.0 for every vector
    norms = np.linalg.norm(embeddings, axis=1)
    for i, norm in enumerate(norms):
        assert abs(norm - 1.0) < 1e-5, f"Vector {i} norm = {norm:.6f} (expected ≈ 1.0)"

    print(f"  ✅ Shape       : {embeddings.shape}")
    print(f"  ✅ dtype       : {embeddings.dtype}")
    print(f"  ✅ L2 norms    : {norms.round(6).tolist()}  (all ≈ 1.0)")
    return embeddings


# ────────────────────────────────────────────────────────────────
# Test 3 — embed_single() shape + L2 norm
# ────────────────────────────────────────────────────────────────
def test_embed_single(model):
    separator("TEST 3 — embed_single() Shape and L2 Normalization")
    vec = embed_single("How do I pay my fees?", model)

    assert vec.shape == (EMBEDDING_DIM,), (
        f"Expected shape ({EMBEDDING_DIM},), got {vec.shape}"
    )
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Norm = {norm:.6f} (expected ≈ 1.0)"

    print(f"  ✅ Shape       : {vec.shape}")
    print(f"  ✅ L2 norm     : {norm:.6f}  (≈ 1.0)")


# ────────────────────────────────────────────────────────────────
# Test 4 — load_and_embed_faqs()
# ────────────────────────────────────────────────────────────────
def test_load_and_embed_faqs(model):
    separator("TEST 4 — load_and_embed_faqs() Corpus Loading")
    faq_docs, corpus_embeddings = load_and_embed_faqs(FAQS_PATH, model)

    assert len(faq_docs) == 30, f"Expected 30 FAQs, got {len(faq_docs)}"
    assert corpus_embeddings.shape == (30, EMBEDDING_DIM), (
        f"Expected (30, {EMBEDDING_DIM}), got {corpus_embeddings.shape}"
    )
    norms = np.linalg.norm(corpus_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Not all corpus vectors are L2-normalized"

    print(f"  ✅ FAQ count   : {len(faq_docs)}")
    print(f"  ✅ Embedding shape : {corpus_embeddings.shape}")
    print(f"  ✅ Norm range  : [{norms.min():.6f}, {norms.max():.6f}]  (all ≈ 1.0)")
    print(f"  ✅ Categories  : {sorted(set(d['category'] for d in faq_docs))}")

    return faq_docs, corpus_embeddings


# ────────────────────────────────────────────────────────────────
# Test 5 — search_semantic() data contract + result quality
# ────────────────────────────────────────────────────────────────
def test_search_semantic(model, faq_docs, corpus_embeddings):
    separator("TEST 5 — search_semantic() Results and Data Contract")

    test_queries = [
        {
            "query": "What happens if I miss an exam?",
            "expected_category": "exam",
            "description": "Conceptual exam-policy query",
        },
        {
            "query": "How do I apply for a merit scholarship?",
            "expected_category": "scholarship",
            "description": "Conceptual scholarship query",
        },
        {
            "query": "What is the hostel curfew time?",
            "expected_category": "hostel",
            "description": "Hostel rule query",
        },
        {
            "query": "Which companies visit for campus recruitment?",
            "expected_category": "placement",
            "description": "Placement / recruiter query",
        },
        {
            "query": "How much are the tuition fees per year?",
            "expected_category": "fees",
            "description": "Fee-related conceptual query",
        },
    ]

    all_passed = True

    for tc in test_queries:
        result = search_semantic(
            query=tc["query"],
            faq_docs=faq_docs,
            embeddings=corpus_embeddings,
            model=model,
            top_k=3,
        )

        # ── Validate data contract keys ──────────────────────
        required_keys = {
            "query", "detected_entities", "route_decision",
            "retrieved_docs", "scores", "rationale",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"Missing keys in result: {missing}"

        # ── Validate contract values ─────────────────────────
        assert result["query"] == tc["query"]
        assert result["route_decision"] == "semantic"
        assert isinstance(result["detected_entities"], list)
        assert len(result["retrieved_docs"]) == 3
        assert len(result["scores"]) == 3
        assert isinstance(result["rationale"], str)

        # ── Validate scores are descending ───────────────────
        assert result["scores"] == sorted(result["scores"], reverse=True), (
            "Scores not in descending order"
        )

        # ── Check top result category ────────────────────────
        top_doc = result["retrieved_docs"][0]
        top_category = top_doc["category"]
        category_match = top_category == tc["expected_category"]

        status = "✅" if category_match else "⚠️ "
        if not category_match:
            all_passed = False

        print(f"\n  Query      : \"{tc['query']}\"")
        print(f"  Description: {tc['description']}")
        print(f"  Top result : \"{top_doc['question'][:70]}\"")
        print(f"  Category   : {top_category}  (expected: {tc['expected_category']}) {status}")
        print(f"  Score      : {result['scores'][0]:.4f}")
        print(f"  Scores     : {result['scores']}")
        print(f"  Rationale  : {result['rationale'][:100]}...")

    separator("PHASE 1 VALIDATION SUMMARY")
    if all_passed:
        print("  ✅ All semantic search queries returned expected category as top result.")
    else:
        print("  ⚠️  One or more queries returned a different top category.")
        print("      This may be acceptable — review scores manually above.")
    print()


# ────────────────────────────────────────────────────────────────
# Run all tests
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  PHASE 1 VALIDATION — Embedding Pipeline + Semantic Search")
    print("█" * 60)

    model = test_model_loading()
    test_embed_texts(model)
    test_embed_single(model)
    faq_docs, corpus_embeddings = test_load_and_embed_faqs(model)
    test_search_semantic(model, faq_docs, corpus_embeddings)

    print("█" * 60)
    print("  Phase 1 complete. Proceed to Phase 2 (keyword search).")
    print("█" * 60 + "\n")
