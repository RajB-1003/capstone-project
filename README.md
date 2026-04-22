# 🎓 Semantic FAQ Router

A **production-grade, 7-phase modular FAQ retrieval system** built for university environments. It intelligently routes natural-language queries through keyword, semantic, and hybrid retrieval strategies using a two-tier agentic architecture — all surfaced through a polished Streamlit interface.

---

## 📌 Project Overview

This system answers university-related FAQs (courses, exams, fees, hostels, scholarships, etc.) by:

- **Tier 1 Router** — Fast regex-based detection of structured patterns (course codes, identifiers)
- **Tier 2 Router** — Intent classification using sentence embeddings + cosine similarity
- **Semantic Search** — Dense retrieval via `sentence-transformers`
- **Keyword Search** — Sparse BM25 retrieval via `rank-bm25`
- **Hybrid RRF Fusion** — Reciprocal Rank Fusion combining both signals
- **Caching** — Exact and semantic (embedding-level) cache layers
- **Explainability** — Per-result rationale and pipeline latency profiling

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Tier 1 Router      │  ← Regex patterns (course codes, identifiers)
│  (router_tier1.py)  │
└────────┬────────────┘
         │ No match
         ▼
┌─────────────────────┐
│  Tier 2 Router      │  ← Intent similarity (sentence embeddings)
│  (router_tier2.py)  │
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
  Semantic  Keyword        ← Or both (Hybrid)
  Search    Search
    │         │
    └────┬────┘
         ▼
    RRF Fusion  →  Confidence Check  →  Explainability  →  Cache  →  Response
```

---

## 📁 Project Structure

```
faq_system/
├── app.py                      # Streamlit UI — Search, Compare, Manage tabs
├── requirements.txt            # Python dependencies
│
├── config/
│   └── regex_patterns.json     # Tier 1 router: regex rules for course codes
│
├── data/
│   ├── faqs.json               # FAQ knowledge base
│   └── intent_exemplars.json   # Tier 2 router: intent training exemplars
│
├── modules/
│   ├── __init__.py
│   ├── embedder.py             # Sentence embedding model loader
│   ├── embedding_store.py      # Embedding persistence (load/save .npy)
│   ├── keyword_search.py       # BM25 sparse retrieval
│   ├── semantic_search.py      # Dense cosine similarity retrieval
│   ├── hybrid_search.py        # RRF fusion of both retrieval signals
│   ├── router_tier1.py         # Regex-based Tier 1 router
│   ├── router_tier2.py         # Intent-based Tier 2 router
│   ├── pipeline.py             # End-to-end query pipeline orchestrator
│   ├── retriever.py            # FAQRetriever class (used by Compare tab)
│   ├── comparison.py           # Side-by-side retrieval comparison
│   ├── confidence.py           # Confidence scoring and low-confidence alerts
│   ├── explainability.py       # Per-result rationale generation
│   ├── faq_manager.py          # FAQ CRUD operations
│   ├── cache.py                # Exact + semantic cache layers
│   ├── profiler.py             # Pipeline stage latency profiler
│   └── langchain_wrapper.py    # LangChain / LangGraph integration wrapper
│
└── tests/
    └── test_phase1.py          # Unit tests for Phase 1 (embeddings)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/RajB-1003/capstone-project.git
cd capstone-project
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r faq_system/requirements.txt
pip install streamlit
```

### 4. (Optional) Install Voice & Authentication Dependencies
If you want to enable the experimental voice transcription and authentication features, run the voice dependency script:
```bash
cd faq_system
python modules/install_voice_deps.py
```

### 5. Run the app
```bash
# Make sure you are in the faq_system directory
cd faq_system
streamlit run app.py
```

The app will initialize the local database automatically and open at `http://localhost:8501`.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🔵 **Keyword Route** | Regex-based Tier 1 detection for course codes (e.g., `CS-202`) |
| 🟢 **Semantic Route** | Dense embedding similarity ≥ 0.82 triggers Tier 2 semantic retrieval |
| 🟣 **Hybrid Route** | Similarity 0.65–0.82 triggers RRF fusion of semantic + keyword results |
| 💾 **Caching** | Exact query cache + semantic embedding-level cache to skip recomputation |
| ⚖️ **Compare Tab** | Side-by-side view of all three retrieval strategies for any query |
| 📋 **Manage Tab** | Full FAQ CRUD — add, edit, delete FAQ entries at runtime |
| 🔧 **Explainability** | Per-result rationale bullets and pipeline latency breakdown |
| 🏷️ **Filters** | Filter results by category and tags in the sidebar |
| ⚠️ **Confidence Alerts** | Warns when top result falls below confidence threshold |

---

## 🧪 Validation Scripts

Each development phase has a dedicated validation script:

```bash
cd faq_system
python validate_phase1.py   # Embedding model + FAQ embedding
python validate_phase2.py   # BM25 keyword search
python validate_phase3.py   # Tier 1 regex router
python validate_phase4.py   # Tier 2 intent router
python validate_phase5.py   # Hybrid RRF fusion
python validate_phase6.py   # Explainability layer
python validate_phase7.py   # Caching layer
```

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Embeddings | `sentence-transformers >= 2.7.0` |
| Keyword Search | `rank-bm25 >= 0.2.2` |
| Agentic Framework | `langchain >= 0.2.0`, `langgraph >= 0.1.0` |
| Vector Operations | `numpy >= 1.26.0` |
| Caching | `cachetools >= 5.3.0` |
| UI | `streamlit` |

---

## 📖 Query Examples

| Query | Route |
|-------|-------|
| `CS-202 prerequisites` | 🔵 Keyword (Tier 1) |
| `What happens if I miss an exam?` | 🟢 Semantic (Tier 2) |
| `What is the attendance requirement?` | 🟢 Semantic (Tier 2) |
| `hostel fee CS-202` | 🟣 Hybrid |
| `scholarship eligibility and course rules` | 🟣 Hybrid |

---

## 👤 Author

**Raj B**  
Capstone Project — Semantic FAQ Retrieval System  
[GitHub](https://github.com/RajB-1003)
