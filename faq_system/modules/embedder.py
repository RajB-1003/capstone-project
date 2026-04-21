"""
embedder.py — Phase 1: Embedding Pipeline

Responsibilities:
  - Load the sentence-transformer model (all-MiniLM-L6-v2, 384-dim)
  - Convert text into L2-normalized embedding vectors
  - Load + embed the FAQ corpus from faqs.json

Architecture mandate (Phased Agent Prompt Chain Design.docx):
  - Model: sentence-transformers/all-MiniLM-L6-v2
  - Pooling: average pooling (default for MiniLM)
  - Normalization: L2-normalize every vector so dot product == cosine similarity
  - Storage: keep embeddings as float32 NumPy array for BLAS-optimized dot product

Public API:
  load_embedding_model()                    -> SentenceTransformer
  embed_texts(texts, model)                 -> np.ndarray  shape (N, 384), float32, L2-normed
  embed_single(text, model)                 -> np.ndarray  shape (384,),   float32, L2-normed
  load_and_embed_faqs(faqs_path, model)     -> tuple[list[dict], np.ndarray]
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# ──────────────────────────────────────────────────────────────
# 1. Model loader
# ──────────────────────────────────────────────────────────────
def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load and return a SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier.
                    Defaults to all-MiniLM-L6-v2 (architecture mandate).

    Returns:
        SentenceTransformer instance ready for inference.
    """
    model = SentenceTransformer(model_name)
    return model


# ──────────────────────────────────────────────────────────────
# 2. Batch text embedder
# ──────────────────────────────────────────────────────────────
def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed a list of texts and return L2-normalized float32 vectors.

    Why normalize?
        When both query and document vectors have unit L2 norm, the dot product
        equals the cosine similarity — allowing NumPy BLAS routines (much faster
        than computing cos manually with divisions).

    Args:
        texts: List of strings to embed.
        model: Loaded SentenceTransformer model.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype float32, L2-normalized.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    # model.encode returns an ndarray; normalize_embeddings=True applies L2 norm.
    # dtype kwarg is NOT supported in sentence-transformers v2+, so we cast afterwards.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # architecture mandate: unit-norm vectors
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 3. Single-query embedder
# ──────────────────────────────────────────────────────────────
def embed_single(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a single text string and return a 1-D L2-normalized vector.

    Args:
        text: Input string (query or document).
        model: Loaded SentenceTransformer model.

    Returns:
        np.ndarray of shape (EMBEDDING_DIM,), dtype float32, L2-normalized.
    """
    embedding = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embedding[0].astype(np.float32)   # flatten to 1-D


# ──────────────────────────────────────────────────────────────
# 4. FAQ corpus loader + embedder
# ──────────────────────────────────────────────────────────────
def load_and_embed_faqs(
    faqs_data: str | list[dict],
    model: SentenceTransformer,
) -> tuple[list[dict], np.ndarray]:
    """
    Load FAQ documents from a JSON file or direct list and compute embeddings.

    Args:
        faqs_data: Absolute or relative path to faqs.json OR list of FAQ dicts.
        model:     Loaded SentenceTransformer model.

    Returns:
        faq_docs   — list[dict]
        embeddings — np.ndarray shape (N, EMBEDDING_DIM), float32, L2-normalized.
    """
    if isinstance(faqs_data, str):
        with open(faqs_data, "r", encoding="utf-8") as f:
            faq_docs = json.load(f)
    else:
        faq_docs = faqs_data

    if not faq_docs:
        raise ValueError(f"FAQ data is empty.")

    # Embed the concatenation of question + answer — richer signal than Q alone
    texts = [
        f"{doc['question']} {doc['answer']}"
        for doc in faq_docs
    ]

    embeddings = embed_texts(texts, model)
    return faq_docs, embeddings
