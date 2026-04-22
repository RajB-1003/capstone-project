"""
app.py — Semantic FAQ Router  v3  (Feature-Complete)
Run: streamlit run app.py  (from faq_system/)

Tabs:
  🔍 Search   — main search with filters, confidence alert, query-type label
  ⚖️ Compare  — side-by-side Semantic | Keyword | Hybrid comparison
  📋 Manage   — FAQ CRUD (add / edit / delete)

New features vs v2:
  F1  FAQ CRUD management panel
  F2  Side-by-side retrieval comparison view
  F3  Low confidence alert integrated into search results
  F5  Tags field displayed in result cards; filter by category + tags
  F6  Embedding persistence (load .npy, skip recompute if available)
  F7  Query type label (code / conceptual / hybrid) near route decision
  F8  FAQRetriever class used by comparison tab
"""

import os, sys, json, re, textwrap, logging
import streamlit as st

# Enable debug logging so feedback_reranking trace is visible in terminal.
# Remove or set to INFO once visible ranking is confirmed.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Semantic FAQ Router",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import shared threshold constants (Issue 2 fix — single source of truth)
from modules.constants import THRESHOLD_HIGH, THRESHOLD_LOW

# Phase 1: Auth + DB (additive — no pipeline changes)
from modules.auth import (
    init_session, login_user, logout_user, register_user,
    get_current_user, is_authenticated, is_admin, ensure_default_admin,
)

# Initialise DB tables and Streamlit session keys on every rerun.
init_session()
ensure_default_admin()   # creates admin/admin123 on first launch


# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
# CSS
# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #ffffff; color: #1e293b; font-size: 15px; }
#MainMenu, footer, header  { visibility: hidden; }

/* PAGE HEADER */
.page-header {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #818cf8 100%);
    border-radius: 16px; padding: 24px 32px 20px; margin-bottom: 16px;
    border: 1px solid rgba(99,102,241,0.2);
    box-shadow: 0 8px 40px rgba(99,102,241,0.18);
}
.page-header h1 { margin:0 0 4px; font-size:1.9rem; font-weight:800; color:#ffffff; }
.page-header .sub { margin:0; font-size:0.85rem; color:rgba(255,255,255,0.75); }
.phase-pills { margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; }
.phase-pill  { background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25);
               border-radius:20px; padding:2px 10px; font-size:0.65rem; color:rgba(255,255,255,0.9); }

/* ROUTE HERO */
.route-hero { border-radius:14px; padding:22px 24px 18px; border:2px solid; }
.route-hero-keyword  { background:#eff6ff; border-color:#3b82f6; }
.route-hero-semantic { background:#f0fdf4; border-color:#22c55e; }
.route-hero-hybrid   { background:#faf5ff; border-color:#a855f7; }
.route-hero-cached   { background:#fdf4ff; border-color:#d946ef; }
.route-icon { font-size:2rem; margin-bottom:6px; }
.route-tier-label { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:3px; }
.route-tier-keyword  { color:#2563eb; } .route-tier-semantic { color:#16a34a; }
.route-tier-hybrid   { color:#7c3aed; } .route-tier-cached   { color:#a21caf; }
.route-decision-name { font-size:2.4rem; font-weight:800; line-height:1; letter-spacing:-1px; margin-bottom:10px; }
.route-name-keyword  { color:#1d4ed8; } .route-name-semantic { color:#15803d; }
.route-name-hybrid   { color:#6d28d9; } .route-name-cached   { color:#86198f; }
.route-meta-row { display:flex; align-items:flex-start; gap:8px; margin-bottom:5px; font-size:0.8rem; }
.route-meta-key { font-weight:600; color:#374151; min-width:72px; flex-shrink:0; }
.route-meta-val { color:#4b5563; }
.entity-tag {
    display:inline-block; background:#dbeafe; border:1px solid #93c5fd;
    border-radius:5px; padding:1px 7px; font-size:0.78rem; font-weight:700;
    color:#1d4ed8; font-family:monospace; margin:1px 2px; }

/* QUERY TYPE CHIP */
.qtype-chip {
    display:inline-flex; align-items:center; gap:5px;
    border-radius:20px; padding:3px 12px; margin-left:8px;
    font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; }
.qtype-code       { background:#dbeafe; color:#1d4ed8; border:1.5px solid #93c5fd; }
.qtype-conceptual { background:#dcfce7; color:#15803d; border:1.5px solid #86efac; }
.qtype-hybrid     { background:#faf5ff; color:#6d28d9; border:1.5px solid #c4b5fd; }

/* SOURCE BADGES */
.src-badge { display:inline-flex; align-items:center; gap:4px; font-size:0.7rem; font-weight:700;
             border-radius:20px; padding:3px 10px; letter-spacing:0.4px; text-transform:uppercase; }
.src-both     { background:#f3e8ff; color:#6d28d9; border:1.5px solid #c4b5fd; }
.src-semantic { background:#dcfce7; color:#15803d; border:1.5px solid #86efac; }
.src-keyword  { background:#dbeafe; color:#1d4ed8; border:1.5px solid #93c5fd; }
.src-unknown  { background:#f1f5f9; color:#64748b; border:1.5px solid #cbd5e1; }

/* TAGS */
.tag-chip {
    display:inline-block; background:#f0f9ff; border:1px solid #bae6fd;
    border-radius:4px; padding:1px 7px; font-size:0.65rem; font-weight:600;
    color:#0369a1; margin:1px 2px; }

/* RESULT CARDS */
.result-card { background:#f8fafc; border:1.5px solid #e2e8f0; border-radius:14px;
               padding:18px 20px; margin-bottom:10px; }
.result-card:hover { box-shadow:none; border-color:#94a3b8; }
.result-card-top { background:#fffbeb;
                   border:1.5px solid #f59e0b; border-radius:14px; padding:18px 20px; margin-bottom:10px;
                   box-shadow:none; }
.best-match-banner { display:inline-flex; align-items:center; gap:4px;
                     background:#f59e0b; color:#fff; border-radius:6px; padding:3px 10px;
                     font-size:0.68rem; font-weight:700; text-transform:uppercase; margin-bottom:8px; }
.result-header { display:flex; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap; }
.result-rank   { font-size:0.67rem; font-weight:700; color:#94a3b8; background:#f1f5f9;
                 border-radius:5px; padding:2px 8px; letter-spacing:1px; text-transform:uppercase; }
.result-rank-top { color:#92400e; background:#fef3c7; }
.score-chip     { margin-left:auto; font-size:0.7rem; font-weight:700; color:#64748b;
                  background:#f1f5f9; border-radius:5px; padding:2px 8px; }
.score-chip-top { color:#92400e; background:#fde68a; }
.cat-chip       { font-size:0.65rem; color:#94a3b8; background:#f8fafc; border:1px solid #e2e8f0;
                  border-radius:5px; padding:1px 7px; }
.result-question { font-size:1.1rem; font-weight:700; color:#0f172a; margin-bottom:6px; line-height:1.4; }
.result-answer   { font-size:0.93rem; color:#475569; line-height:1.7; margin-bottom:10px; }
.expl-section { border-top:1px solid #e2e8f0; padding-top:8px; }
.expl-bullet  { display:flex; align-items:flex-start; gap:7px; font-size:0.77rem;
                color:#64748b; line-height:1.5; margin-bottom:2px; }
.expl-checkmark { color:#22c55e; font-weight:700; flex-shrink:0; }

/* FEEDBACK BAR */
.feedback-bar { display:flex; align-items:center; gap:8px; padding:6px 0 4px;
                border-top:1px dashed #e2e8f0; margin-top:4px; flex-wrap:wrap; }
.feedback-label { font-size:0.67rem; color:#94a3b8; font-weight:600;
                  text-transform:uppercase; letter-spacing:0.6px; }
.fb-score-pos   { color:#16a34a; font-weight:700; font-size:0.72rem; }
.fb-score-neg   { color:#dc2626; font-weight:700; font-size:0.72rem; }
.fb-score-neu   { color:#94a3b8; font-weight:700; font-size:0.72rem; }
.low-quality-flag { display:inline-flex; align-items:center; gap:4px;
                    background:#fef2f2; border:1.5px solid #fca5a5;
                    border-radius:6px; padding:2px 9px;
                    font-size:0.68rem; color:#dc2626; font-weight:700; }


.rationale-box { background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #6366f1;
                 border-radius:10px; padding:12px 16px; font-size:0.86rem; color:#374151;
                 line-height:1.8; margin-bottom:18px; }
.rationale-box b { color:#1e293b; }

/* CACHE BANNER */
.cache-banner { background:linear-gradient(90deg,#fdf4ff,#fae8ff); border:1.5px solid #d946ef;
                border-radius:10px; padding:10px 16px; font-size:0.84rem; color:#86198f;
                margin-bottom:16px; display:flex; align-items:center; gap:10px; }

/* SECTION LABELS */
.section-label { font-size:0.63rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:2px; color:#94a3b8; margin-bottom:8px; }

/* SIDEBAR */
.active-route-badge { border-radius:8px; padding:9px 13px; margin-bottom:12px;
                      font-size:0.8rem; font-weight:600; text-align:center; }
.arb-keyword  { background:#eff6ff; color:#1d4ed8; border:1.5px solid #93c5fd; }
.arb-semantic { background:#f0fdf4; color:#15803d; border:1.5px solid #86efac; }
.arb-hybrid   { background:#faf5ff; color:#6d28d9; border:1.5px solid #c4b5fd; }
.arb-none     { background:#f8fafc; color:#94a3b8; border:1.5px solid #e2e8f0; }
.legend-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:0.78rem; color:#4b5563; }
.legend-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.stat-row   { display:flex; justify-content:space-between; font-size:0.76rem; color:#6b7280;
              padding:4px 0; border-bottom:1px solid #f1f5f9; }
.stat-val   { font-weight:700; color:#374151; }

/* COMPARISON TABLE */
.compare-col-header { text-align:center; font-weight:700; font-size:0.85rem; padding:8px; border-radius:8px; margin-bottom:8px; }
.compare-semantic   { background:#dcfce7; color:#15803d; }
.compare-keyword    { background:#dbeafe; color:#1d4ed8; }
.compare-hybrid     { background:#f3e8ff; color:#6d28d9; }
.compare-card { background:#fff; border:1px solid #e2e8f0; border-radius:10px;
                padding:12px 14px; margin-bottom:8px; font-size:0.82rem; }
.compare-card-q { font-weight:700; color:#0f172a; margin-bottom:4px; font-size:0.85rem; }
.compare-card-a { color:#64748b; line-height:1.5; margin-bottom:6px; font-size:0.78rem; }
.compare-score  { font-size:0.7rem; color:#94a3b8; font-weight:600; }

/* EMPTY STATE */
.empty-state { text-align:center; padding:54px 20px; }
.empty-state-icon { font-size:2.8rem; margin-bottom:12px; }
.empty-state h3   { font-size:1.05rem; font-weight:700; color:#374151; margin-bottom:6px; }
.empty-state p    { font-size:0.875rem; color:#94a3b8; }

/* FAQ MANAGE */
.manage-faq-card { background:#fff; border:1.5px solid #e2e8f0; border-radius:10px;
                   padding:14px 16px; margin-bottom:10px; }
.manage-faq-q { font-weight:700; color:#0f172a; font-size:0.9rem; margin-bottom:3px; }
.manage-faq-meta { font-size:0.72rem; color:#94a3b8; }


/* SIDEBAR — force open and ensure toggle arrow is always visible */
section[data-testid="stSidebar"] {
    min-width: 280px !important;
    width: 280px !important;
    transform: translateX(0) !important;
    visibility: visible !important;
}
/* Always show the sidebar toggle arrow */
button[data-testid="baseButton-header"],
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)


# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
# CONSTANTS
# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
ROUTE_CONFIG = {
    "keyword":  {"icon": "🔵", "tier": "Tier 1 — Regex Detection",  "cls": "keyword",  "label": "KEYWORD"},
    "semantic": {"icon": "🟢", "tier": "Tier 2 — Intent Router",    "cls": "semantic", "label": "SEMANTIC"},
    "hybrid":   {"icon": "🟣", "tier": "Tier 2 — Intent Router",    "cls": "hybrid",   "label": "HYBRID"},
}
SOURCE_BADGES = {
    "both":     '<span class="src-badge src-both">🟣 Both</span>',
    "semantic": '<span class="src-badge src-semantic">🟢 Semantic</span>',
    "keyword":  '<span class="src-badge src-keyword">🔵 Keyword</span>',
}
QTYPE_CLASS = {
    "code":        "qtype-code",
    "conceptual":  "qtype-conceptual",
    "hybrid":      "qtype-hybrid",
}
EXAMPLE_QUERIES = {
    "🔵 Keyword Route (Tier 1)": [
        "CS-202 prerequisites", "ENG-404 syllabus", "What is CS-202?",
    ],
    "🟢 Semantic Route (Tier 2)": [
        "What happens if I miss an exam?",
        "What is the attendance requirement?",
        "Can I retake a failed course?",
        "What is the penalty for academic plagiarism?",
    ],
    "🟣 Hybrid Route": [
        "hostel fee CS-202",
        "scholarship eligibility and course rules",
        "ENG-404 and CS-202 eligibility",
    ],
}
STAGE_META = [
    ("embedding",       "🔵 Embedding"),
    ("router_tier1",    "🔸 Tier 1 Router"),
    ("router_tier2",    "🔶 Tier 2 Router"),
    ("semantic_search", "🟢 Semantic Search"),
    ("keyword_search",  "🔹 Keyword Search"),
    ("hybrid_fusion",   "🟣 RRF Fusion"),
    ("explainability",  "🔮 Explainability"),
]
FAQ_CATEGORIES = ["course", "exam", "hostel", "fees", "scholarship",
                  "placement", "administration", "wellbeing", "other"]


# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
# PIPELINE FIXTURE LOADER
# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
@st.cache_resource(show_spinner=False)
def load_pipeline_fixtures():
    from modules.db import get_all_faqs
    from modules.embedder        import load_embedding_model, load_and_embed_faqs
    from modules.keyword_search  import build_bm25_index
    from modules.router_tier1    import load_regex_patterns
    from modules.router_tier2    import load_intent_exemplars, embed_intents
    from modules.embedding_store import load_embeddings, save_embeddings

    PATTERNS_PATH  = os.path.join("config", "regex_patterns.json")
    EXEMPLARS_PATH = os.path.join("data", "intent_exemplars.json")

    faq_docs = get_all_faqs()

    model = load_embedding_model()

    # Feature 6: load from disk if available, otherwise compute + save
    corpus_embeddings = load_embeddings()
    if corpus_embeddings is None:
        _, corpus_embeddings = load_and_embed_faqs(faq_docs, model)
        save_embeddings(corpus_embeddings)

    bm25_index, _     = build_bm25_index(faq_docs)
    patterns          = load_regex_patterns(PATTERNS_PATH)
    exemplars         = load_intent_exemplars(EXEMPLARS_PATH)
    intent_embeddings = embed_intents(exemplars, model)

    return model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings


# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
# SESSION STATE INIT
# =========================================================================
for key, default in [
    ("response",      None),
    ("total_queries", 0),
    ("pending_query", ""),
    ("compare_result", None),
    ("edit_faq_id",   None),
    ("active_page",   "search"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
# HELPERS
# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
def _bold_numbers(text: str) -> str:
    text = re.sub(r'\b(\d+\.\d+)\b', r'<b>\1</b>', text)
    text = re.sub(r'\b(\d+)\s*(ms|result|doc|document|overlap\b)', r'<b>\1</b> \2', text)
    return text


def _explanation_bullets(source: str, rank: int, score: float) -> str:
    bullets = []
    if source == "both":
        bullets.append("Strong match in both semantic and keyword search")
    elif source == "semantic":
        bullets.append("Retrieved via semantic similarity to query intent")
    elif source == "keyword":
        bullets.append("Retrieved via direct keyword / identifier match")
    else:
        bullets.append("Retrieved by retrieval pipeline")

    conf = "highest" if rank == 1 else "strong" if rank <= 2 else "moderate"
    bullets.append(f"Rank #{rank} — {conf} confidence (score: {score})")

    if source == "both":
        bullets.append("Appeared in both retrieval lists — boosted by RRF")
    elif source == "semantic":
        bullets.append("Embedding similarity matched query concept")
    elif source == "keyword":
        bullets.append("BM25 term match with structured token preservation")

    return "".join(
        f'<div class="expl-bullet"><span class="expl-checkmark">✔</span>'
        f'<span>{b}</span></div>'
        for b in bullets
    )


def _tags_html(tags: list[str]) -> str:
    if not tags:
        return ""
    return "".join(f'<span class="tag-chip">{t}</span>' for t in tags)


def _apply_filters(results: list[dict],
                   sel_categories: list[str],
                   sel_tags: list[str]) -> list[dict]:
    """Filter result list by selected categories and tags (client-side).
    Step 8: reads 'tags' field only — no metadata fallback."""
    out = []
    for r in results:
        if sel_categories and r.get("category", "") not in sel_categories:
            continue
        if sel_tags:
            rtags = set(r.get("tags", []))  # Step 8: standardised field only
            if not any(t in rtags for t in sel_tags):
                continue
        out.append(r)
    return out


# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═
# RENDER FUNCTIONS
# â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═â•═

def render_navbar(active_response=None, cur_user=None):
    from modules.cache       import cache_stats, clear_all_caches
    from modules.faq_manager import get_categories, get_all_tags

    # ── Navbar ─────────────────────────────────────────────
    nl, nc, nr = st.columns([2, 6, 2])
    
    with nl:
        st.markdown("<h3 style='margin:0; padding-top:0px; color:#4f46e5; font-weight:800;'>🎓 Univ FAQ</h3>", unsafe_allow_html=True)

    if cur_user:
        role = cur_user.get("role", "user")
        if role == "admin":
            nav_items = [
                ("compare",         "⚖️ Compare"),
                ("admin_dashboard", "📊 Admin Dashboard"),
                ("manage_faq",      "⚙️ Manage FAQ"),
            ]
        else:
            nav_items = [
                ("search", "🔍 Search"),
                ("rag",    "🤖 RAG"),
            ]
    else:
        nav_items = []
        
    with nc:
        btn_cols = st.columns(len(nav_items))
        active_page = st.session_state.get("active_page", "search")
        for idx, (page_id, label) in enumerate(nav_items):
            is_active = (active_page == page_id)
            with btn_cols[idx]:
                if st.button(
                    label,
                    key=f"nav_{page_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_page"] = page_id
                    # Persist last_page in DB so refresh returns user here
                    _token = st.session_state.get("auth_token")
                    if _token:
                        from modules.db import update_token_last_page
                        update_token_last_page(_token, page_id)
                    st.rerun()

    with nr:
        if cur_user:
            role  = cur_user.get("role", "user")
            uname = cur_user.get("username", "")
            role_badge = "🔺" if role == "admin" else "👤"
            
            user_col, logout_col = st.columns([0.6, 0.4])
            with user_col:
                st.markdown(
                    f"<div style='text-align:right; font-size:0.85rem; padding-top:8px;'>"
                    f"<b>{role_badge} {uname}</b></div>",
                    unsafe_allow_html=True
                )
            with logout_col:
                if st.button("Out", key="btn_logout", use_container_width=True):
                    from modules.auth import logout_user
                    logout_user()
                    st.rerun()

    st.markdown("<hr style='margin:0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

    # ── Global Tools Expander ─────────────────────────────────────────
    with st.expander("⚙️ Tools & Filters", expanded=False):
        t1, t2, t3 = st.columns(3)
        
        with t1:
            st.markdown("**🔍 Filters**")
            categories = get_categories()
            all_tags   = get_all_tags()
            sel_cats   = st.multiselect("Category", categories, key="filter_cats", placeholder="All categories")
            sel_tags   = st.multiselect("Tags", all_tags, key="filter_tags", placeholder="All tags") if all_tags else []

            # Example queries
            st.markdown("**💡 Examples**")
            for group, examples in EXAMPLE_QUERIES.items():
                if st.button(f"{group}...", key=f"exg_{group}", use_container_width=True, help="Load an example"):
                    st.session_state.pending_query = examples[0]
                    st.session_state["active_page"] = "search"
                    st.rerun()

        with t2:
            st.markdown("**Route Information**")
            if active_response:
                route = active_response.get("route_decision", "")
                lms   = active_response.get("latency_ms", {})
                total = lms.get("total", 0)
                cache = lms.get("cache", "")
                qtype = active_response.get("query_type", "")
                icons = {"keyword": "🔵", "semantic": "🟢", "hybrid": "🟣"}
                arb   = f"arb-{route}" if route in icons else "arb-none"
                cache_note = f" · {cache} cache" if cache else ""
                qtype_note = f" · {qtype}" if qtype else ""
                st.markdown(
                    f'<div class="active-route-badge {arb}">'
                    f'{icons.get(route,"⚡")} <strong>{route.upper()}</strong>'
                    f'<br><span style="font-size:0.7rem;opacity:0.75">{total:.1f} ms{cache_note}{qtype_note}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="active-route-badge arb-none">No query yet</div>', unsafe_allow_html=True)

            # Route legend
            st.markdown(f"""
            <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1.8px;color:#94a3b8;margin:8px 0 7px;">Route Logic</div>
            <div class="legend-row"><div class="legend-dot" style="background:#3b82f6;"></div><span><b style="color:#1d4ed8">Keyword</b> — course codes</span></div>
            <div class="legend-row"><div class="legend-dot" style="background:#22c55e;"></div><span><b style="color:#15803d">Semantic</b> — high similarity</span></div>
            <div class="legend-row"><div class="legend-dot" style="background:#a855f7;"></div><span><b style="color:#6d28d9">Hybrid</b> — semantic + keyword fallback</span></div>
            """, unsafe_allow_html=True)

        with t3:
            st.markdown("**Cache & History**")
            stats = cache_stats()
            st.markdown(f"""
            <div class="stat-row"><span>Exact cache</span><span class="stat-val">{stats['exact']['size']}</span></div>
            <div class="stat-row"><span>Semantic cache</span><span class="stat-val">{stats['semantic']['size']}</span></div>
            <div class="stat-row"><span>Exact hits</span><span class="stat-val">{stats['exact']['hits']}</span></div>
            <div class="stat-row"><span>Queries run</span><span class="stat-val">{st.session_state.total_queries}</span></div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Clear Caches", use_container_width=True, type="secondary"):
                clear_all_caches()
                st.rerun()
                
            from modules.db import get_history
            
            _cur = get_current_user()
            if _cur:
                user_hist = get_history(_cur["id"], limit=3)
                if user_hist:
                    st.markdown("**Recent**")
                    for q, route, ms in user_hist:
                        icon = {"keyword": "🔵", "semantic": "🟢", "hybrid": "🟣"}.get(route, "⚡")
                        st.markdown(
                            f"<div style='font-size:0.76rem;color:#6b7280;padding:2px 0'>"
                            f"{icon} {q[:25]} <span style='color:#d1d5db'>({ms:.0f}ms)</span></div>",
                            unsafe_allow_html=True,
                        )

    return sel_cats if 'sel_cats' in dir() else [], sel_tags if 'sel_tags' in dir() else []

def render_feedback_buttons(result: dict, query: str, route: str) -> None:
    """
    Render 👍 / 👎 / 🚫 feedback buttons below a result card (Feature 1).
    Clicking any button:
      - Persists the event via store_feedback()          (Feature 2)
      - Refreshes, showing a confirmation + live score   (Feature 3)
    Low-quality flag (Feature 5) and adjusted-score debug (Feature 7)
    are also surfaced here.
    """
    faq_id = result.get("faq_id", "")
    if not faq_id:
        return

    # Resolve current user_id for feedback record (global signal, user tracked)
    _cur     = get_current_user()
    _user_id = _cur["id"] if _cur else None

    # Unique per-FAQ-per-query key so state is isolated across results.
    fb_done_key = f"fb_done_{faq_id}_{abs(hash(query))}"

    # ── Feature 5: low-quality flag ───────────────────────────
    if result.get("low_quality"):
        st.markdown(
            '<div class="low-quality-flag">⚠️ Frequently marked unhelpful</div>',
            unsafe_allow_html=True,
        )

    # ── FIX 8: extended debug caption ─────────────────────────
    adj      = result.get("adjusted_score")
    orig     = result.get("score", 0.0)
    if adj is not None and abs(adj - orig) > 1e-9:
        fb_avg   = result.get("feedback_avg", 0.0)
        fb_count = result.get("feedback_count", 0)
        fb_adj   = result.get("adjustment", 0.0)
        st.caption(
            f"🔧 Feedback-adjusted: **{adj:.4f}** "
            f"(original {orig:.4f} · avg {fb_avg:+.3f} × {fb_count} vote"
            f"{'s' if fb_count != 1 else ''} · Δ {fb_adj:+.4f})"
        )

    # ── Feature 1 + 3: show buttons or confirmation ───────────
    if st.session_state.get(fb_done_key):
        agg   = get_aggregated_scores()
        score = agg.get(faq_id, {}).get("score", 0)
        count = agg.get(faq_id, {}).get("count", 0)
        sc_cls = (
            "fb-score-pos" if score > 0
            else "fb-score-neg" if score < 0
            else "fb-score-neu"
        )
        st.markdown(
            f'<div class="feedback-bar">'
            f'<span class="feedback-label">✅ Feedback recorded</span>'
            f'&nbsp;<span class="{sc_cls}">'
            f'score {score:+d} &middot; {count} vote{"s" if count != 1 else ""}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        # Feature 6: suggested tags (only when negative feedback exists)
        suggested = get_suggested_tags(faq_id)
        if suggested:
            tags_html = " ".join(
                f'<span class="tag-chip" style="background:#fef2f2;color:#dc2626;border-color:#fca5a5">'
                f'{t}</span>'
                for t in suggested
            )
            st.markdown(
                f'<div style="margin-top:4px;font-size:0.7rem;color:#94a3b8">'
                f'💡 Suggested tags: {tags_html}</div>',
                unsafe_allow_html=True,
            )
        return

    # Buttons row —
    st.markdown(
        '<div style="font-size:0.75rem;color:#64748b;margin-bottom:8px;'
        'font-weight:600;">Was this helpful?</div>',
        unsafe_allow_html=True,
    )
    c_up, c_dn, c_not = st.columns([1, 1, 2])
    with c_up:
        if st.button(
            "👍",
            key=f"fb_up_{faq_id}_{abs(hash(query))}",
            help="Helpful — this answered my question",
            use_container_width=True
        ):
            store_feedback(faq_id, query, route, result.get("score", 0.0), "up", user_id=_user_id)
            st.session_state[fb_done_key] = True
            st.rerun()
    with c_dn:
        if st.button(
            "👎",
            key=f"fb_dn_{faq_id}_{abs(hash(query))}",
            help="Not what I needed",
            use_container_width=True
        ):
            store_feedback(faq_id, query, route, result.get("score", 0.0), "down", user_id=_user_id)
            st.session_state[fb_done_key] = True
            st.rerun()
    with c_not:
        if st.button(
            "🚫 Not Helpful",
            key=f"fb_not_{faq_id}_{abs(hash(query))}",
            help="Completely unhelpful — wrong topic",
            use_container_width=True
        ):
            store_feedback(faq_id, query, route, result.get("score", 0.0), "not_helpful", user_id=_user_id)
            st.session_state[fb_done_key] = True
            st.rerun()


def render_results(results: list[dict], query: str = "", route: str = ""):
    if not results:
        st.warning("No documents retrieved for this query.")
        return
    st.markdown("#### Retrieved Documents")
    st.markdown("<br>", unsafe_allow_html=True)

    # Route badge colour map
    _BADGE_STYLES = {
        "semantic": "background:#dcfce7; color:#15803d; border:1.5px solid #86efac;",
        "keyword":  "background:#dbeafe; color:#1d4ed8; border:1.5px solid #93c5fd;",
        "hybrid":   "background:#f3e8ff; color:#6d28d9; border:1.5px solid #c4b5fd;",
        "cached":   "background:#fff7ed; color:#c2410c; border:1.5px solid #fdba74;",
    }
    _BADGE_ICONS = {
        "semantic": "🟢", "keyword": "🔵", "hybrid": "🟣", "cached": "🟠",
    }

    route_key = route.lower() if route else "semantic"
    badge_style = _BADGE_STYLES.get(route_key, _BADGE_STYLES["semantic"])
    badge_icon  = _BADGE_ICONS.get(route_key, "⚡")

    for i, r in enumerate(results):
        q     = r.get("question", "")
        a     = r.get("answer", "")
        score = r.get("score", 0.0)
        tags  = r.get("tags", [])

        st.markdown(
            '<div style="padding:1.5rem 1.75rem; border:1.5px solid #e2e8f0; '
            'border-radius:12px; margin-bottom:1.5rem; background:#ffffff;">',
            unsafe_allow_html=True,
        )

        # ── Route badge + confidence row (ABOVE question) ──────────────
        conf_color = "#16a34a" if score >= 0.65 else "#d97706"
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">'
            f'<span style="{badge_style} padding:5px 14px; border-radius:8px; '
            f'font-weight:700; font-size:14px; display:inline-flex; align-items:center; gap:5px;">'
            f'{badge_icon} {route_key.upper()}</span>'
            f'<span style="font-size:13px; color:{conf_color}; font-weight:600;">'
            f'Confidence: {score:.2f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Question ───────────────────────────────────────────────────
        st.markdown(
            f'<div style="font-size:1.15rem; font-weight:700; color:#0f172a; '
            f'margin-bottom:8px; line-height:1.4;">{q}</div>',
            unsafe_allow_html=True,
        )

        # ── Answer ─────────────────────────────────────────────────────
        st.markdown(
            f'<div style="font-size:0.95rem; color:#334155; line-height:1.7; '
            f'margin-bottom:1rem;">{a}</div>',
            unsafe_allow_html=True,
        )

        # ── Feedback buttons ───────────────────────────────────────────
        if query:
            render_feedback_buttons(r, query, route)

        # ── Divider ────────────────────────────────────────────────────
        st.markdown(
            "<hr style='margin:14px 0 10px 0; border:none; border-top:1px solid #e2e8f0;'>",
            unsafe_allow_html=True,
        )

        # ── Tags (tertiary, below divider) ─────────────────────────────
        if tags:
            tags_html = " ".join(
                f'<span style="background:#f0f9ff; border:1px solid #bae6fd; '
                f'border-radius:4px; padding:1px 8px; font-size:0.72rem; '
                f'color:#0369a1; font-weight:600;">{t}</span>'
                for t in tags
            )
            st.markdown(
                f'<div style="font-size:0.78rem; color:#94a3b8;">'
                f'Tags: {tags_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)


def render_compare_column(results: list[dict], label: str, css_cls: str):
    st.markdown(f"#### {label}")
    if not results:
        st.info("No results")
        return
    for r in results:
        q = r.get("question", "")
        a = textwrap.shorten(r.get("answer", ""), 160)
        score = r.get("score", 0.0)
        
        st.markdown(f"**{q}**")
        st.markdown(f"{a}")
        st.caption(f"Confidence: {score:.4f}")
        st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)


# ========================================================================================================================
# LOGIN PAGE  (Phase 1 — additive)
# ========================================================================================================================

def render_login_page() -> None:
    """
    Full-page login / register UI shown to unauthenticated visitors.
    On successful login st.session_state is updated and st.rerun() is called,
    which causes the rest of the app to render the main UI.
    """
    # Centre the card
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
                    border-radius:20px;padding:40px 36px 32px;margin-top:60px;
                    box-shadow:0 20px 60px rgba(99,102,241,0.35);">
            <div style="text-align:center;margin-bottom:28px;">
                <div style="font-size:2.8rem;margin-bottom:8px;">&#127891;</div>
                <div style="font-size:1.55rem;font-weight:800;color:#fff;
                            letter-spacing:-0.5px;">Semantic FAQ Router</div>
                <div style="font-size:0.82rem;color:rgba(255,255,255,0.75);
                            margin-top:4px;">University Knowledge Base System</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # Tab switcher: Login | Register
        login_tab, reg_tab = st.tabs(["Sign In", "Create Account"])

        # ── Login tab ──────────────────────────────────────────
        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                st.markdown("#### Welcome back")
                uname = st.text_input(
                    "Username", placeholder="Enter your username",
                    key="login_username"
                )
                pwd = st.text_input(
                    "Password", type="password",
                    placeholder="Enter your password",
                    key="login_password"
                )
                submitted = st.form_submit_button(
                    "Sign In →", use_container_width=True, type="primary"
                )

            if submitted:
                if not uname or not pwd:
                    st.error("Please enter both username and password.")
                else:
                    try:
                        user = login_user(uname, pwd)
                        role = user.get("role", "user")
                        if role == "admin":
                            st.session_state["active_page"] = "admin_dashboard"
                        else:
                            st.session_state["active_page"] = "search"
                        
                        st.success(
                            f"Welcome, **{user['username']}**! "
                            f"({'Admin' if role == 'admin' else 'User'} account)"
                        )
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        # ── Register tab ───────────────────────────────────────
        with reg_tab:
            with st.form("register_form", clear_on_submit=True):
                st.markdown("#### Create account")
                new_uname = st.text_input(
                    "Username", placeholder="Choose a username",
                    key="reg_username"
                )
                new_pwd = st.text_input(
                    "Password", type="password",
                    placeholder="Min. 6 characters",
                    key="reg_password"
                )
                new_pwd2 = st.text_input(
                    "Confirm Password", type="password",
                    placeholder="Repeat password",
                    key="reg_password2"
                )
                reg_submitted = st.form_submit_button(
                    "Create Account", use_container_width=True, type="primary"
                )

            if reg_submitted:
                if new_pwd != new_pwd2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        register_user(new_uname, new_pwd, role="user")
                        st.success(
                            "Account created! Switch to **Sign In** to log in."
                        )
                    except ValueError as exc:
                        st.error(str(exc))

        st.markdown(
            "<p style='text-align:center;font-size:0.72rem;color:#94a3b8;"
            "margin-top:20px;'>Default admin credentials on first launch: "
            "<code>admin / admin123</code></p>",
            unsafe_allow_html=True,
        )


# ========================================================================================================================
# PAGE LAYOUT
# ========================================================================================================================

# ── Auth gate ─────────────────────────────────────────────────
if not is_authenticated():
    render_login_page()
    st.stop()

_cur_user = get_current_user()

# ── Load fixtures ─────────────────────────────────────────────
with st.spinner("⏳ Loading embedding model..."):
    fixtures = load_pipeline_fixtures()

model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings = fixtures
from modules.pipeline      import run_pipeline
from modules.comparison    import compare_retrieval
from modules.faq_manager   import load_faqs, add_faq, edit_faq, delete_faq, get_categories
from modules.rag_demo      import generate_rag_answer
from modules.multilingual  import process_query as multilingual_process_query
from modules.feedback_store import (
    store_feedback,
    get_aggregated_scores,
    get_feedback_score,
    get_suggested_tags,
)
from modules.admin_dashboard import render_admin_dashboard

PIPELINE_ARGS = dict(
    model=model, corpus_embeddings=corpus_embeddings, faq_docs=faq_docs,
    bm25_index=bm25_index, patterns=patterns, intent_embeddings=intent_embeddings, top_k=5,
)

# ── Top Navbar ────────────────────────────────────────────────
role = _cur_user.get("role", "user") if _cur_user else "user"
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "admin_dashboard" if role == "admin" else "search"

st.session_state.setdefault("response", None)
st.session_state.setdefault("pending_query", "")
sel_categories, sel_tags_filter = render_navbar(st.session_state.response, _cur_user)

# ── Page routing ──────────────────────────────────────────────
active_page = st.session_state.get("active_page")

# Guard: Role-based Access Control
if role == "user" and active_page not in ["search", "rag"]:
    st.error("Access denied")
    st.stop()

if role == "admin" and active_page not in ["compare", "admin_dashboard", "manage_faq"]:
    st.error("Access denied")
    st.stop()


# ─────────────────────────────────────────────────────────────
# PAGE: SEARCH
# ─────────────────────────────────────────────────────────────
if active_page == "search":
    st.markdown(
        "<h3 style='margin-bottom:4px;'>🔍 Search</h3>"
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:16px;'>"
        "Ask any university-related question.</p>",
        unsafe_allow_html=True,
    )

    # Handle pending query from sidebar buttons
    if st.session_state.pending_query:
        st.session_state.query_input = st.session_state.pending_query
        st.session_state.pending_query = ""

    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        
        search_col, voice_col = st.columns([0.88, 0.12])
        
        with voice_col:
            from modules.voice_utils import get_voice_input, normalize_voice_query
            voice_text = get_voice_input()
            if voice_text is not None:
                if not voice_text.strip():
                    st.warning("No voice input detected")
                else:
                    clean_query = normalize_voice_query(voice_text)
                    if clean_query:
                        st.session_state.query_input = clean_query
                        st.session_state.last_voice_text = voice_text
                        
        with search_col:
            query = st.text_input(
                "University FAQ Query",
                label_visibility="collapsed",
                placeholder="Ask a question — 'CS-202 prerequisites', 'What happens if I miss an exam?'",
                key="query_input",
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Ask →", type="primary", use_container_width=True)
        
        if st.session_state.get("last_voice_text"):
            st.info(f"🎤 Transcribed text: {st.session_state.last_voice_text}")
            if search_clicked:
                st.session_state.last_voice_text = ""
                
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Execute pipeline ────────────────────────────────────────
    if search_clicked and query.strip():
        with st.spinner("🔍 Running pipeline..."):
            try:
                query_data      = multilingual_process_query(query.strip())
                processed_query = query_data["processed_query"]
                
                _cur = get_current_user()
                _user_id = _cur["id"] if _cur else None
                
                resp = run_pipeline(processed_query, _user_id, **PIPELINE_ARGS)
                resp["original_query"] = query_data["original_query"]
                resp["language"]       = query_data["language"]
                resp["translated"]     = query_data["translated"]
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()
        st.session_state.response = resp
        st.session_state.total_queries += 1
        lms = resp.get("latency_ms", {})
        
        if _user_id:
            from modules.db import store_query
            store_query(
                _user_id, 
                query.strip(), 
                resp.get("route_decision", "?"), 
                lms.get("total", 0)
            )
            
        st.rerun()

    # ── Render response ─────────────────────────────────────────
    response = st.session_state.response

    if response is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <h3>Ready to Answer Your Questions</h3>
            <p>Type a query above or pick an example from the sidebar.</p>
            <p style="margin-top:10px;font-size:0.78rem">
                🔵 <b>CS-202 prerequisites</b> → keyword &nbsp;|&nbsp;
                🟢 <b>What happens if I miss an exam?</b> → semantic &nbsp;|&nbsp;
                🟣 <b>hostel fee CS-202</b> → hybrid
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        lms        = response.get("latency_ms", {})
        cache_type = response.get("cache_type", "miss")
        cache_sim  = response.get("cache_similarity", 0.0)
        route      = response.get("route_decision", "")
        query_type = response.get("query_type", "")
        rationale  = response.get("rationale", "")

        # ── Multilingual badge ──────────────────────────────────
        _translated = response.get("translated", False)
        _lang       = response.get("language", "en")
        _orig_q     = response.get("original_query", "")
        if _translated:
            st.info(
                f"🌐 **Translated from** `{_lang}` **→ English** "
                f"*(original: \u201c{_orig_q[:80]}{'...' if len(_orig_q) > 80 else ''}\u201d)*"
            )
        else:
            st.caption(f"🌐 Language detected: **{_lang.upper() if _lang else 'English'}**")

        # ── Low confidence alert ────────────────────────────────
        if response.get("low_confidence"):
            warn = response.get("confidence_warning", "⚠️ Low confidence result.")
            st.warning(warn)

        # ── Cache hit ───────────────────────────────────────────
        if cache_type in ("exact", "semantic"):
            st.success("⚡ Served from cache")

        # ── Filters + Results ───────────────────────────────────
        all_results = response.get("results", [])
        filtered    = _apply_filters(all_results, sel_categories, sel_tags_filter)

        if sel_categories or sel_tags_filter:
            st.caption(
                f"Showing {len(filtered)} of {len(all_results)} results "
                f"(category={sel_categories or 'all'}, tags={sel_tags_filter or 'all'})"
            )

        _, res_col, _ = st.columns([1, 3, 1])
        with res_col:
            render_results(
                filtered if (sel_categories or sel_tags_filter) else all_results,
                query=response.get("query", ""),
                route=response.get("route_decision", ""),
            )

            with st.expander("🔧 Developer View (Raw JSON)", expanded=False):
                st.json({k: v for k, v in response.items() if k != "latency_ms"})
                st.json({"latency_ms": lms})



# ─────────────────────────────────────────────────────────────
# PAGE: COMPARE
# ─────────────────────────────────────────────────────────────
elif active_page == "compare":
    st.markdown(
        "<h3 style='margin-bottom:4px;'>⚖️ Compare</h3>"
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:16px;'>"
        "Run the same query through Keyword, Semantic, and Hybrid retrievers simultaneously.</p>",
        unsafe_allow_html=True,
    )

    cq_in, cq_btn = st.columns([0.82, 0.18])
    with cq_in:
        cmp_query = st.text_input(
            "Comparison Query",
            label_visibility="collapsed",
            placeholder="Enter a query to compare all three retrieval methods…",
            key="compare_query",
        )
    with cq_btn:
        cmp_clicked = st.button("Compare →", type="primary", use_container_width=True)

    if cmp_clicked and cmp_query.strip():
        with st.spinner("⚖️ Running all three retrievers…"):
            try:
                st.session_state.compare_result = compare_retrieval(
                    cmp_query.strip(),
                    model=model,
                    corpus_embeddings=corpus_embeddings,
                    faq_docs=faq_docs,
                    bm25_index=bm25_index,
                    top_k=5,
                )
            except Exception as e:
                st.error(f"Comparison error: {e}")

    cmp = st.session_state.compare_result
    if cmp:
        c_kw, c_sem, c_hyb = st.columns(3)
        with c_kw:
            kw_data = cmp.get("keyword", {})
            render_compare_column(
                kw_data.get("results", []) if isinstance(kw_data, dict) else kw_data,
                "Keyword Results",
                "keyword",
            )
        with c_sem:
            sem_data = cmp.get("semantic", {})
            render_compare_column(
                sem_data.get("results", []) if isinstance(sem_data, dict) else sem_data,
                "Semantic Results",
                "semantic",
            )
        with c_hyb:
            hyb_data = cmp.get("hybrid", {})
            render_compare_column(
                hyb_data.get("results", []) if isinstance(hyb_data, dict) else hyb_data,
                "Hybrid Results",
                "hybrid",
            )
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">⚖️</div>
            <h3>No comparison yet</h3>
            <p>Enter a query above to see how each retriever responds.</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: RAG ANSWER
# ─────────────────────────────────────────────────────────────
elif active_page == "rag":
    st.markdown(
        "<h3 style='margin-bottom:4px;'>🧠 RAG Answer</h3>"
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:16px;'>"
        "Retrieval-augmented generation — retrieves context then generates an answer.</p>",
        unsafe_allow_html=True,
    )
    rag_query = st.text_input("Ask a question for generated answer...", key="rag_input")
    if st.button("Generate Answer", type="primary", key="rag_btn"):
        if not rag_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                rag_resp = generate_rag_answer(rag_query, top_k=3, fixtures=fixtures)
                ans  = rag_resp["generated_answer"]
                docs = rag_resp["source_documents"]

                st.markdown(
                    f'<div style="font-size:1.1rem;padding:15px;border-radius:8px;'
                    f'background:#f8fafc;border:1px solid #e2e8f0;color:#0f172a;margin-bottom:20px;">'
                    f'{ans}</div>',
                    unsafe_allow_html=True,
                )

                if len(docs) == 0 or "I could not find a precise answer" in ans:
                    st.warning("Low confidence: answer may be incomplete.")

                st.divider()
                st.markdown("#### Sources used:")
                for doc in docs:
                    q = doc.get("question", "")
                    a = doc.get("answer", "")
                    st.markdown(
                        '<div style="margin-bottom:10px;padding:10px;border-left:3px solid #6366f1;'
                        'background:#ffffff;border-radius:0 4px 4px 0;box-shadow:0 1px 2px rgba(0,0,0,0.05);">'
                        f'<strong style="color:#1e293b;">Q: {q}</strong><br>'
                        f'<span style="color:#475569;font-size:0.9rem;">{a[:150]}...</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────
# PAGE: ADMIN DASHBOARD (admin only)
# ─────────────────────────────────────────────────────────────
elif active_page == "admin_dashboard" and _cur_user["role"] == "admin":
    render_admin_dashboard()


# ─────────────────────────────────────────────────────────────
# PAGE: MANAGE FAQ (admin only)
# ─────────────────────────────────────────────────────────────
elif active_page == "manage_faq" and _cur_user["role"] == "admin":
    st.markdown(
        "<h3 style='margin-bottom:4px;'>⚙️ Manage FAQ</h3>"
        "<p style='color:#64748b;font-size:0.85rem;margin-bottom:16px;'>"
        "Add, edit, or delete FAQs. Changes persist to faqs.json.</p>",
        unsafe_allow_html=True,
    )

    try:
        current_faqs = load_faqs()
    except Exception as e:
        st.error(f"Could not load FAQs: {e}")
        current_faqs = []

    st.markdown("""
        <style>
        .manage-form-container {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
        }
        .manage-faq-card { 
            background:#fff; 
            border:1px solid #e2e8f0; 
            border-radius:12px;
            padding:16px 20px; 
            margin-bottom:12px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease-in-out;
        }
        .manage-faq-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
            border-color: #c7d2fe;
            transform: translateY(-2px);
        }
        .manage-faq-q { font-weight:700; color:#0f172a; font-size:0.95rem; margin-bottom:6px; line-height: 1.4; }
        .manage-faq-meta { font-size:0.75rem; color:#64748b; display: flex; align-items: center; gap: 8px; }
        </style>
    """, unsafe_allow_html=True)

    m_add, m_list = st.columns([0.42, 0.58], gap="large")

    with m_add:
        edit_id = st.session_state.edit_faq_id
        if edit_id:
            edit_target = next((f for f in current_faqs if str(f.get("id")) == str(edit_id)), None)
            st.markdown(f"<h3 style='color:#4f46e5;font-size:1.4rem;margin-bottom:16px;'>✏️ Edit FAQ: {edit_id}</h3>", unsafe_allow_html=True)
        else:
            edit_target = None
            st.markdown("<h3 style='color:#4f46e5;font-size:1.4rem;margin-bottom:16px;'>➕ Add New FAQ</h3>", unsafe_allow_html=True)

        st.markdown('<div class="manage-form-container">', unsafe_allow_html=True)
        with st.form("faq_form", clear_on_submit=True, border=False):
            f_question = st.text_area(
                "Question *",
                value=edit_target.get("question", "") if edit_target else "",
                height=80, key=f"f_q_{edit_id}" if edit_target else "f_q_new",
            )
            f_answer = st.text_area(
                "Answer *",
                value=edit_target.get("answer", "") if edit_target else "",
                height=120, key=f"f_a_{edit_id}" if edit_target else "f_a_new",
            )
            cats     = get_categories()
            all_cats = sorted(set(FAQ_CATEGORIES + cats))
            cat_def  = edit_target.get("category", "other") if edit_target else "course"
            cat_idx  = all_cats.index(cat_def) if cat_def in all_cats else 0
            f_category = st.selectbox("Category *", all_cats, index=cat_idx, key=f"f_cat_{edit_id}" if edit_target else "f_cat_new")
            f_tags = st.text_input(
                "Tags (comma-separated)",
                value=", ".join(edit_target.get("tags", [])) if edit_target else "",
                key=f"f_tags_{edit_id}" if edit_target else "f_tags_new", placeholder="e.g. prerequisite, grades, semester",
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            c_sub, c_cancel = st.columns(2)
            with c_sub:
                submitted = st.form_submit_button(
                    "💾 Update FAQ" if edit_target else "➕ Add FAQ",
                    type="primary", use_container_width=True,
                )
            with c_cancel:
                if st.form_submit_button("✕ Cancel", use_container_width=True):
                    st.session_state.edit_faq_id = None
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            tags_list = [t.strip() for t in f_tags.split(",") if t.strip()]
            try:
                if edit_target:
                    edit_faq(edit_id, {"question": f_question, "answer": f_answer,
                                       "category": f_category, "tags": tags_list})
                    st.success(f"✅ FAQ **{edit_id}** updated!")
                else:
                    new = add_faq(f_question, f_answer, f_category, tags_list)
                    st.success(f"✅ FAQ **{new['id']}** added!")
                load_pipeline_fixtures.clear()
                st.session_state.edit_faq_id = None
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    with m_list:
        st.markdown(f"<h3 style='color:#4f46e5;font-size:1.4rem;margin-bottom:16px;'>📄 All FAQs ({len(current_faqs)})</h3>", unsafe_allow_html=True)
        
        st.markdown('<div class="manage-form-container" style="padding: 16px; margin-bottom: 20px; background: #f8fafc; border: 1px solid #e2e8f0; box-shadow: none;">', unsafe_allow_html=True)
        filt_cat_manage = st.selectbox(
            "Filter by category", ["(all)"] + get_categories(), key="manage_cat_filter",
        )
        search_q = st.text_input(
            "Search FAQs", placeholder="Filter by question text…",
            key="manage_search", label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        shown = current_faqs
        if filt_cat_manage != "(all)":
            shown = [f for f in shown if f.get("category") == filt_cat_manage]
        if search_q:
            shown = [f for f in shown if search_q.lower() in f.get("question", "").lower()]
        
        st.markdown(f"<p style='color:#64748b; font-size:0.85rem; margin-bottom: 12px;'>Showing {len(shown)} of {len(current_faqs)}</p>", unsafe_allow_html=True)

        for faq in shown:
            fid    = faq.get("id", "?")
            cat    = faq.get("category", "")
            tags   = faq.get("tags", [])
            tags_h = _tags_html(tags) if tags else ""
            cat_h  = f'<span class="cat-chip">{cat}</span>' if cat else ""

            col_card, col_edit, col_del = st.columns([7.4, 1.3, 1.3])
            with col_card:
                st.markdown(f"""
                <div class="manage-faq-card">
                    <div class="manage-faq-q">{faq.get("question","")}</div>
                    <div class="manage-faq-meta"><span style="color:#94a3b8; font-weight: 500;">{fid}</span> &nbsp;&bull;&nbsp; {cat_h}&nbsp;{tags_h}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_edit:
                st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)
                if st.button("✏️", key=f"edit_{fid}", help="Edit this FAQ",
                             use_container_width=True):
                    st.session_state.edit_faq_id = fid
                    st.rerun()
            with col_del:
                st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{fid}", help="Delete this FAQ",
                             use_container_width=True):
                    try:
                        delete_faq(fid)
                        load_pipeline_fixtures.clear()
                        st.success(f"Deleted {fid}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
