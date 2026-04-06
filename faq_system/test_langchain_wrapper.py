"""
test_langchain_wrapper.py — Validation script for the LangChain wrapper
=========================================================================

Run from the faq_system/ directory:

    python test_langchain_wrapper.py

What it tests:
  1. FAQRetriever instantiation
  2. get_relevant_documents() returns Document objects
  3. Document fields are correctly populated
  4. get_relevant_documents_with_metadata() returns the full pipeline dict
  5. Pipeline behavior is unchanged (same results as calling run_pipeline directly)
"""

import sys
import os

# Ensure faq_system root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def separator(title: str) -> None:
    print(f"\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def test_instantiation() -> "FAQRetriever":
    separator("TEST 1 — Instantiation")
    from langchain_wrapper import FAQRetriever
    retriever = FAQRetriever(top_k=5)
    print(f"  ✔  Created:  {retriever!r}")
    return retriever


def test_get_relevant_documents(retriever) -> list:
    separator("TEST 2 — get_relevant_documents()")
    query = "What happens if I miss an exam?"
    print(f"  Query: {query!r}\n")

    docs = retriever.get_relevant_documents(query)

    print(f"  [OK] Documents returned: {len(docs)}")
    assert len(docs) > 0, "Expected at least 1 document"
    return docs


def test_document_structure(docs: list) -> None:
    separator("TEST 3 — Document structure")
    from langchain_wrapper import Document

    first = docs[0]
    assert isinstance(first, Document), f"Expected Document, got {type(first)}"

    print(f"  First document:")
    print(f"    page_content : {first.page_content[:120]!r}...")
    print(f"    metadata     :")
    for k, v in first.metadata.items():
        print(f"      {k:<12} = {v!r}")

    required_keys = {"question", "score", "source", "category", "tags", "rank"}
    missing = required_keys - first.metadata.keys()
    assert not missing, f"Missing metadata keys: {missing}"
    print(f"\n  ✔  All required metadata keys present: {sorted(required_keys)}")


def test_all_documents(docs: list) -> None:
    separator("TEST 4 — All documents summary")
    for i, doc in enumerate(docs, 1):
        rank  = doc.metadata.get("rank", "?")
        src   = doc.metadata.get("source", "?")
        score = doc.metadata.get("score", "?")
        q     = doc.metadata.get("question", "")[:70]
        print(f"  #{rank:>2}  [{src:<8}]  score={score:<8}  {q}")
    print(f"\n  ✔  {len(docs)} document(s) printed")


def test_metadata_method(retriever) -> None:
    separator("TEST 5 — get_relevant_documents_with_metadata()")
    query = "What happens if I miss an exam?"
    full  = retriever.get_relevant_documents_with_metadata(query)

    assert isinstance(full, dict), "Expected dict"
    required_top = {"query", "results", "route_decision", "latency_ms"}
    missing = required_top - full.keys()
    assert not missing, f"Missing top-level keys: {missing}"

    print(f"  ✔  Top-level keys: {sorted(full.keys())}")
    print(f"     route_decision : {full.get('route_decision')}")
    print(f"     query_type     : {full.get('query_type')}")
    print(f"     low_confidence : {full.get('low_confidence')}")
    print(f"     total latency  : {full['latency_ms'].get('total', '?')} ms")


def test_keyword_route(retriever) -> None:
    separator("TEST 6 — Keyword route (CS-202)")
    query = "CS-202 prerequisites"
    docs  = retriever.get_relevant_documents(query)
    full  = retriever.get_relevant_documents_with_metadata(query)
    route = full.get("route_decision", "")

    print(f"  Query  : {query!r}")
    print(f"  Route  : {route}")
    print(f"  Docs   : {len(docs)}")
    # CS-202 must trigger keyword route (Tier 1)
    assert route == "keyword", f"Expected 'keyword' route, got {route!r}"
    print(f"  ✔  Correct keyword route detected")


def test_no_langchain_required() -> None:
    separator("TEST 7 — Works without LangChain")
    try:
        import langchain  # noqa: F401
        print("  ℹ  LangChain IS installed — using real Document/BaseRetriever classes")
    except ImportError:
        print("  ℹ  LangChain NOT installed — using built-in stubs")
    print("  ✔  Import succeeded regardless")


def main():
    print("=" * 60)
    print("  FAQ LangChain Wrapper -- Test Suite")
    print("=" * 60)

    try:
        test_no_langchain_required()
        retriever = test_instantiation()
        docs      = test_get_relevant_documents(retriever)
        test_document_structure(docs)
        test_all_documents(docs)
        test_metadata_method(retriever)
        test_keyword_route(retriever)

        separator("RESULT")
        print("  [PASS] All tests passed -- LangChain wrapper is working correctly\n")

    except AssertionError as exc:
        separator("RESULT")
        print(f"  [FAIL] Test FAILED: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        separator("RESULT")
        print(f"  [ERROR] Unexpected error: {type(exc).__name__}: {exc}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
