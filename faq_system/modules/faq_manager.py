"""
faq_manager.py — Feature 1: FAQ CRUD Management

Provides load/add/edit/delete operations on data/faqs.json.
On any mutation, the on-disk embedding cache (.npy) is deleted so
the system recomputes embeddings on next startup.

Data contract:
    Each FAQ must have: id (str), question, answer, category, tags (list)
    The 'tags' field is optional in existing FAQs (defaults to []).
    The 'metadata' field from existing FAQs is preserved unchanged.

Public API:
    load_faqs(path)                             -> list[dict]
    add_faq(question, answer, category, tags)   -> dict   (new FAQ)
    edit_faq(faq_id, updated_fields)            -> dict   (updated FAQ)
    delete_faq(faq_id)                          -> None
    get_categories(path)                        -> list[str]
    get_all_tags(path)                          -> list[str]
"""

import json
import os
import time

FAQS_PATH       = os.path.join("data", "faqs.json")
EMBEDDINGS_PATH = os.path.join("data", "corpus_embeddings.npy")


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _load_raw(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_raw(faqs: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)


def _new_id(faqs: list[dict]) -> str:
    """Generate next sequential FAQ id string like 'faq_031'."""
    existing = []
    for faq in faqs:
        raw = str(faq.get("id", ""))
        numeric = raw.replace("faq_", "")
        if numeric.isdigit():
            existing.append(int(numeric))
        elif raw.isdigit():
            existing.append(int(raw))
    next_num = max(existing, default=0) + 1
    return f"faq_{next_num:03d}"


def _invalidate_embeddings() -> None:
    """Delete persisted embedding cache so it is recomputed on next load. Step 6: safe."""
    try:
        os.remove(EMBEDDINGS_PATH)
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────
# Public CRUD API
# ─────────────────────────────────────────────────────────────

def load_faqs(path: str = FAQS_PATH) -> list[dict]:
    """
    Load FAQs from JSON file.
    Injects 'tags' = [] for any FAQ missing the field (backward compat).
    """
    faqs = _load_raw(path)
    for faq in faqs:
        faq.setdefault("tags", [])
    return faqs


def add_faq(
    question: str,
    answer:   str,
    category: str,
    tags:     list[str] | None = None,
    path:     str = FAQS_PATH,
) -> dict:
    """
    Add a new FAQ entry and persist to disk.

    Args:
        question: Question text.
        answer:   Answer text.
        category: Category string (e.g., "course", "exam").
        tags:     Optional list of keyword tags.
        path:     Path to faqs.json.

    Returns:
        The newly created FAQ dict.
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not answer.strip():
        raise ValueError("Answer must not be empty.")
    if not category.strip():
        raise ValueError("Category must not be empty.")

    faqs = _load_raw(path)
    new_faq = {
        "id":       _new_id(faqs),
        "question": question.strip(),
        "answer":   answer.strip(),
        "category": category.strip().lower(),
        "tags":     [t.strip().lower() for t in (tags or []) if t.strip()],
        "metadata": {"type": "user_added", "created_at": int(time.time())},
    }
    faqs.append(new_faq)
    _save_raw(faqs, path)
    _invalidate_embeddings()
    return new_faq


def edit_faq(
    faq_id:         str,
    updated_fields: dict,
    path:           str = FAQS_PATH,
) -> dict:
    """
    Update fields of an existing FAQ.

    Args:
        faq_id:         The 'id' string of the FAQ to update.
        updated_fields: Dict of fields to merge (e.g., {"answer": "..."}).
        path:           Path to faqs.json.

    Returns:
        The updated FAQ dict.

    Raises:
        ValueError: If the FAQ id is not found.
    """
    # Tags: normalise if provided
    if "tags" in updated_fields:
        updated_fields["tags"] = [
            t.strip().lower()
            for t in updated_fields["tags"]
            if t.strip()
        ]

    faqs = _load_raw(path)
    for faq in faqs:
        if str(faq.get("id")) == str(faq_id):
            faq.update(updated_fields)
            _save_raw(faqs, path)
            _invalidate_embeddings()
            return faq

    raise ValueError(f"FAQ with id='{faq_id}' not found.")


def delete_faq(faq_id: str, path: str = FAQS_PATH) -> None:
    """
    Remove a FAQ by its id and persist to disk.

    Raises:
        ValueError: If the FAQ id is not found.
    """
    faqs = _load_raw(path)
    original_len = len(faqs)
    faqs = [f for f in faqs if str(f.get("id")) != str(faq_id)]

    if len(faqs) == original_len:
        raise ValueError(f"FAQ with id='{faq_id}' not found.")

    _save_raw(faqs, path)
    _invalidate_embeddings()


def get_categories(path: str = FAQS_PATH) -> list[str]:
    """Return sorted list of unique category strings."""
    faqs = _load_raw(path)
    return sorted({f.get("category", "").strip() for f in faqs
                   if f.get("category", "").strip()})


def get_all_tags(path: str = FAQS_PATH) -> list[str]:
    """Return sorted list of all unique tags across all FAQs."""
    faqs   = _load_raw(path)
    all_tags: set[str] = set()
    for faq in faqs:
        all_tags.update(t for t in faq.get("tags", []) if t)
    return sorted(all_tags)
