"""
embedding_store.py — Feature 6: Persistent Embedding Cache

Saves and loads the corpus embedding matrix to/from disk so the model
does not need to re-encode all FAQs on every startup.

Safe to use with @st.cache_resource: on FAQ mutation, faq_manager.py
deletes the .npy file, which causes the next startup to recompute and
re-save.

Public API:
    save_embeddings(embeddings, file_path)
    load_embeddings(file_path) -> np.ndarray | None
    embeddings_exist(file_path) -> bool
    delete_embeddings(file_path)
"""

import os
import numpy as np

DEFAULT_PATH = os.path.join("data", "corpus_embeddings.npy")


def save_embeddings(embeddings: np.ndarray,
                    file_path: str = DEFAULT_PATH) -> None:
    """Persist a float32 embedding matrix to disk."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    np.save(file_path, embeddings.astype(np.float32))


def load_embeddings(file_path: str = DEFAULT_PATH) -> "np.ndarray | None":
    """
    Load persisted embeddings using memory-mapping (Fix 3).
    mmap_mode='r' keeps the data memory-mapped (OS-paged, low RSS).
    The astype() copy is skipped when the stored dtype is already float32
    (the normal case — save_embeddings() always writes float32), preserving
    the memory-mapping benefit.  A conversion is performed only for legacy
    files stored in other dtypes.
    Returns None if the file does not exist.
    """
    if os.path.exists(file_path):
        # Removing mmap_mode="r" to prevent WinError 32 (file lock on Windows)
        # since the cache holds a reference to the array preventing file deletion.
        arr = np.load(file_path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr
    return None


def embeddings_exist(file_path: str = DEFAULT_PATH) -> bool:
    return os.path.exists(file_path)


def delete_embeddings(file_path: str = DEFAULT_PATH) -> None:
    """Remove cached embeddings (called after FAQ mutations). Step 6: safe."""
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass
