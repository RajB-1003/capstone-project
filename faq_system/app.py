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

import os, sys, json, re, textwrap
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Semantic FAQ Router",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header  { visibility: hidden; }

/* PAGE HEADER */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 16px; padding: 24px 32px 20px; margin-bottom: 16px;
    border: 1px solid rgba(148,163,184,0.1);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
}
.page-header h1 { margin:0 0 4px; font-size:1.9rem; font-weight:800; color:#f8fafc; }
.page-header .sub { margin:0; font-size:0.85rem; color:#64748b; }
.phase-pills { margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; }
.phase-pill  { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.09);
               border-radius:20px; padding:2px 10px; font-size:0.65rem; color:#94a3b8; }

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
.result-card:hover { box-shadow:0 4px 14px rgba(0,0,0,0.07); border-color:#cbd5e1; }
.result-card-top { background:linear-gradient(145deg,#fffbeb,#fef9c3);
                   border:2.5px solid #f59e0b; border-radius:14px; padding:18px 20px; margin-bottom:10px;
                   box-shadow:0 4px 20px rgba(245,158,11,0.12); }
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
.result-question { font-size:0.97rem; font-weight:700; color:#0f172a; margin-bottom:5px; line-height:1.4; }
.result-answer   { font-size:0.85rem; color:#475569; line-height:1.7; margin-bottom:10px; }
.expl-section { border-top:1px solid #e2e8f0; padding-top:8px; }
.expl-bullet  { display:flex; align-items:flex-start; gap:7px; font-size:0.77rem;
                color:#64748b; line-height:1.5; margin-bottom:2px; }
.expl-checkmark { color:#22c55e; font-weight:700; flex-shrink:0; }

/* RATIONALE */
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
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
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
    ("router_tier1",    "🔷 Tier 1 Router"),
    ("router_tier2",    "💠 Tier 2 Router"),
    ("semantic_search", "🟢 Semantic Search"),
    ("keyword_search",  "🔹 Keyword Search"),
    ("hybrid_fusion",   "🟣 RRF Fusion"),
    ("explainability",  "🔮 Explainability"),
]
FAQ_CATEGORIES = ["course", "exam", "hostel", "fees", "scholarship",
                  "placement", "administration", "wellbeing", "other"]


# ═══════════════════════════════════════════════════════════════
# PIPELINE FIXTURE LOADER
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline_fixtures():
    from modules.embedder        import load_embedding_model, load_and_embed_faqs
    from modules.keyword_search  import build_bm25_index
    from modules.router_tier1    import load_regex_patterns
    from modules.router_tier2    import load_intent_exemplars, embed_intents
    from modules.embedding_store import load_embeddings, save_embeddings

    FAQS_PATH      = os.path.join("data", "faqs.json")
    PATTERNS_PATH  = os.path.join("config", "regex_patterns.json")
    EXEMPLARS_PATH = os.path.join("data", "intent_exemplars.json")

    with open(FAQS_PATH) as f:
        faq_docs = json.load(f)

    model = load_embedding_model()

    # Feature 6: load from disk if available, otherwise compute + save
    corpus_embeddings = load_embeddings()
    if corpus_embeddings is None:
        _, corpus_embeddings = load_and_embed_faqs(FAQS_PATH, model)
        save_embeddings(corpus_embeddings)

    bm25_index, _     = build_bm25_index(faq_docs)
    patterns          = load_regex_patterns(PATTERNS_PATH)
    exemplars         = load_intent_exemplars(EXEMPLARS_PATH)
    intent_embeddings = embed_intents(exemplars, model)

    return model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════
for key, default in [
    ("response",      None),
    ("query_history", []),
    ("total_queries", 0),
    ("pending_query", ""),
    ("compare_result", None),
    ("edit_faq_id",   None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def render_sidebar(active_response=None):
    from modules.cache      import cache_stats, clear_all_caches
    from modules.faq_manager import get_categories, get_all_tags

    st.markdown("## 🗺️ System Guide")

    # ── Active route indicator ─────────────────────────────────
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
        st.markdown(
            '<div class="active-route-badge arb-none">No query yet</div>',
            unsafe_allow_html=True,
        )

    # ── Route legend ────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1.8px;color:#94a3b8;margin-bottom:7px;">Route Logic</div>
    <div class="legend-row"><div class="legend-dot" style="background:#3b82f6;"></div><span><b style="color:#1d4ed8">Keyword</b> — course codes (Tier 1)</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#22c55e;"></div><span><b style="color:#15803d">Semantic</b> — similarity ≥ 0.82 (Tier 2)</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#a855f7;"></div><span><b style="color:#6d28d9">Hybrid</b> — 0.65–0.82, RRF fusion</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#d946ef;"></div><span><b style="color:#86198f">Cached</b> — exact or semantic hit</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Filters (Feature 5) ─────────────────────────────────────
    st.markdown("### 🔍 Filters")
    categories = get_categories()
    all_tags   = get_all_tags()
    sel_cats   = st.multiselect("Category", categories, key="filter_cats",
                                placeholder="All categories")
    sel_tags   = st.multiselect("Tags", all_tags, key="filter_tags",
                                placeholder="All tags") if all_tags else []

    st.markdown("---")

    # ── Example queries ──────────────────────────────────────────
    st.markdown("### 💡 Try These Queries")
    for group, examples in EXAMPLE_QUERIES.items():
        with st.expander(group, expanded=False):
            for q in examples:
                if st.button(q, key=f"ex_{q}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

    st.markdown("---")

    # ── Cache stats ──────────────────────────────────────────────
    stats = cache_stats()
    st.markdown("""<div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1.8px;color:#94a3b8;margin-bottom:7px;">Cache Status</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row"><span>Exact cache</span><span class="stat-val">{stats['exact']['size']} / {stats['exact']['max_size']}</span></div>
    <div class="stat-row"><span>Semantic cache</span><span class="stat-val">{stats['semantic']['size']} / {stats['semantic']['max_size']}</span></div>
    <div class="stat-row"><span>Exact hits</span><span class="stat-val">{stats['exact']['hits']}</span></div>
    <div class="stat-row"><span>Queries run</span><span class="stat-val">{st.session_state.total_queries}</span></div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️ Clear All Caches", use_container_width=True, type="secondary"):
        clear_all_caches()
        st.success("Caches cleared.")
        st.rerun()

    # ── Query history ────────────────────────────────────────────
    if st.session_state.query_history:
        with st.expander("📋 History", expanded=False):
            for q, route, ms in reversed(st.session_state.query_history[-10:]):
                icon = {"keyword": "🔵", "semantic": "🟢", "hybrid": "🟣"}.get(route, "⚡")
                st.markdown(
                    f"<div style='font-size:0.76rem;color:#6b7280;padding:2px 0'>"
                    f"{icon} {q[:38]} <span style='color:#d1d5db'>({ms:.0f}ms)</span></div>",
                    unsafe_allow_html=True,
                )

    return sel_cats, sel_tags


def render_route_panel(response: dict, cache_type):
    route    = response.get("route_decision", "")
    rationale = response.get("rationale", "")
    query_type = response.get("query_type", "")
    cfg      = ROUTE_CONFIG.get(route, ROUTE_CONFIG["hybrid"])
    cls      = "cached" if cache_type else cfg["cls"]
    icon     = "💾" if cache_type else cfg["icon"]
    label    = "CACHED" if cache_type else cfg["label"]
    tier     = f"from {cache_type} cache" if cache_type else cfg["tier"]

    intent_m = re.search(r"matched '(\w+)'\s+intent", rationale)
    score_m  = re.search(r"similarity\s+([\d.]+)", rationale)
    intent   = intent_m.group(1).capitalize() if intent_m else None
    score    = float(score_m.group(1)) if score_m else None

    meta = ""
    if cache_type:
        meta += f'<div class="route-meta-row"><span class="route-meta-key">Source</span><span class="route-meta-val">{cache_type} cache</span></div>'
    entities = response.get("detected_entities", [])
    if entities:
        tags = " ".join(f'<span class="entity-tag">{e}</span>' for e in entities)
        meta += f'<div class="route-meta-row"><span class="route-meta-key">Entities</span><span class="route-meta-val">{tags}</span></div>'
    if intent:
        meta += f'<div class="route-meta-row"><span class="route-meta-key">Intent</span><span class="route-meta-val">{intent}</span></div>'
    if score:
        meta += f'<div class="route-meta-row"><span class="route-meta-key">Confidence</span><span class="route-meta-val"><b>{score:.4f}</b></span></div>'
    meta += f'<div class="route-meta-row"><span class="route-meta-key">Trigger</span><span class="route-meta-val">{tier}</span></div>'

    # Query type chip
    qtype_html = ""
    if query_type:
        qcls = QTYPE_CLASS.get(query_type, "qtype-hybrid")
        qtype_html = f'<span class="qtype-chip {qcls}">{query_type}</span>'

    st.markdown(f"""
    <div class="route-hero route-hero-{cls}">
        <div class="route-icon">{icon}</div>
        <div class="route-tier-label route-tier-{cls}">Routing Decision</div>
        <div class="route-decision-name route-name-{cls}">{label} {qtype_html}</div>
        {meta}
    </div>
    """, unsafe_allow_html=True)


def render_latency_panel(lms: dict):
    total_ms   = lms.get("total", 0)
    cache_type = lms.get("cache", "")

    c_met, c_note = st.columns([0.45, 0.55])
    with c_met:
        st.metric("⚡ Total Latency", f"{total_ms:.1f} ms")
    with c_note:
        if cache_type:
            st.info(f"💾 **{cache_type.capitalize()} cache hit** — pipeline bypassed")
        else:
            budget = min(total_ms / 2000, 1.0)
            st.progress(budget, text=f"{budget*100:.0f}% of 2 000 ms budget")

    stages = [(s, lbl, lms[s]) for s, lbl in STAGE_META if s in lms]
    if not stages:
        return
    max_ms = max(ms for _, _, ms in stages) or 1.0
    for _, label, ms in stages:
        c1, c2, c3 = st.columns([3, 6, 1])
        with c1:
            st.markdown(f"<div style='font-size:0.74rem;color:#6b7280;padding-top:4px'>{label}</div>", unsafe_allow_html=True)
        with c2:
            st.progress(min(ms / max_ms, 1.0))
        with c3:
            st.markdown(f"<div style='font-size:0.73rem;color:#9ca3af;padding-top:4px;text-align:right'>{ms:.0f}ms</div>", unsafe_allow_html=True)


def render_rationale(rationale: str):
    if not rationale:
        return
    st.markdown('<div class="section-label" style="margin-top:16px">System Rationale</div>', unsafe_allow_html=True)
    sentences = re.split(r'(?<=[.!?])\s+', rationale.strip())
    short = _bold_numbers(" ".join(sentences[:3]))
    st.markdown(f'<div class="rationale-box">{short}</div>', unsafe_allow_html=True)


def render_result_card(result: dict, is_top: bool = False):
    rank   = result.get("rank", 0)
    source = result.get("source", "unknown")
    score  = result.get("score", 0.0)
    cat    = result.get("category", "")
    tags   = result.get("tags", [])  # Step 8: standardised field, guaranteed by pipeline
    q      = result.get("question", "")
    a      = result.get("answer", "")

    badge  = SOURCE_BADGES.get(source, f'<span class="src-badge src-unknown">{source}</span>')
    cat_h  = f'<span class="cat-chip">{cat}</span>' if cat else ""
    tags_h = _tags_html(tags) if isinstance(tags, list) else ""
    bullets = _explanation_bullets(source, rank, score)

    if is_top:
        tags_row = f'<div style="margin-bottom:6px">{tags_h}</div>' if tags_h else ""
        html = (
            f'<div class="result-card-top">'
            f'<div class="best-match-banner">🏆 Best Match</div>'
            f'<div class="result-header">'
            f'<span class="result-rank result-rank-top">#{rank}</span>'
            f'{badge} {cat_h}'
            f'<span class="score-chip score-chip-top">score: {score}</span>'
            f'</div>'
            f'{tags_row}'
            f'<div class="result-question">{q}</div>'
            f'<div class="result-answer">{a}</div>'
            f'<div class="expl-section">{bullets}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    else:
        tags_row = f'<div style="margin-bottom:6px">{tags_h}</div>' if tags_h else ""
        html = (
            f'<div class="result-card">'
            f'<div class="result-header">'
            f'<span class="result-rank">#{rank}</span>'
            f'{badge} {cat_h}'
            f'<span class="score-chip">score: {score}</span>'
            f'</div>'
            f'{tags_row}'
            f'<div class="result-question">{q}</div>'
            f'<div class="result-answer">{a}</div>'
            f'<div class="expl-section">{bullets}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)


def render_results(results: list[dict]):
    if not results:
        st.warning("No documents retrieved for this query.")
        return
    st.markdown(
        f'<div class="section-label">Retrieved Documents '
        f'<span style="font-weight:400;text-transform:none;letter-spacing:0;color:#94a3b8">'
        f'— {len(results)} result{"s" if len(results)!=1 else ""}</span></div>',
        unsafe_allow_html=True,
    )
    for i, r in enumerate(results):
        render_result_card(r, is_top=(i == 0))


def render_compare_column(results: list[dict], label: str, css_cls: str):
    st.markdown(
        f'<div class="compare-col-header compare-{css_cls}">{label}</div>',
        unsafe_allow_html=True,
    )
    if not results:
        st.info("No results")
        return
    for r in results:
        score  = r.get("score", 0.0)
        tags   = r.get("tags") or []
        source = r.get("source", "")
        q      = r.get("question", "")
        a      = textwrap.shorten(r.get("answer", ""), 160)

        # Build badge + tags as plain text to avoid raw HTML in f-string
        src_label  = source.upper() if source else ""
        src_colour = {"both": "#6d28d9", "semantic": "#15803d", "keyword": "#1d4ed8"}.get(source, "#64748b")
        src_bg     = {"both": "#f3e8ff", "semantic": "#dcfce7", "keyword": "#dbeafe"}.get(source, "#f1f5f9")
        tags_str   = " · ".join(tags) if tags else ""

        card_parts = [
            f'<div class="compare-card">',
            f'<div class="compare-card-q">{q}</div>',
            f'<div class="compare-card-a">{a}</div>',
            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:4px">',
        ]
        if source:
            card_parts.append(
                f'<span style="background:{src_bg};color:{src_colour};border:1.5px solid {src_colour}4d;'
                f'border-radius:20px;padding:2px 9px;font-size:0.67rem;font-weight:700;'
                f'text-transform:uppercase">{src_label}</span>'
            )
        if tags_str:
            card_parts.append(
                f'<span style="font-size:0.67rem;color:#0369a1">{tags_str}</span>'
            )
        card_parts.append(
            f'<span style="margin-left:auto;font-size:0.7rem;color:#94a3b8;font-weight:600">score: {score}</span>'
        )
        card_parts.append('</div></div>')
        st.markdown("".join(card_parts), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    sel_categories, sel_tags_filter = render_sidebar(st.session_state.response)

# ── Page header ────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🎓 Semantic FAQ Router</h1>
    <p class="sub">7-phase modular retrieval — agentic routing · hybrid RRF · explainability · caching · management</p>
    <div class="phase-pills">
        <span class="phase-pill">Phase 1 · Embeddings</span>
        <span class="phase-pill">Phase 2 · BM25 Keyword</span>
        <span class="phase-pill">Phase 3 · Tier 1 Router</span>
        <span class="phase-pill">Phase 4 · Tier 2 Router</span>
        <span class="phase-pill">Phase 5 · Hybrid RRF</span>
        <span class="phase-pill">Phase 6 · Explainability</span>
        <span class="phase-pill">Phase 7 · Caching</span>
        <span class="phase-pill">F1 · FAQ CRUD</span>
        <span class="phase-pill">F2 · Compare</span>
        <span class="phase-pill">F6 · Embed Store</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load fixtures ──────────────────────────────────────────────
with st.spinner("⏳ Loading embedding model..."):
    fixtures = load_pipeline_fixtures()

model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings = fixtures
from modules.pipeline    import run_pipeline
from modules.comparison  import compare_retrieval
from modules.faq_manager import load_faqs, add_faq, edit_faq, delete_faq, get_categories

PIPELINE_ARGS = dict(
    model=model, corpus_embeddings=corpus_embeddings, faq_docs=faq_docs,
    bm25_index=bm25_index, patterns=patterns, intent_embeddings=intent_embeddings, top_k=5,
)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_search, tab_compare, tab_manage = st.tabs([
    "🔍 Search", "⚖️ Compare", "📋 Manage FAQs"
])


# ─────────────────────────────────────────────────────────────
# TAB 1: SEARCH
# ─────────────────────────────────────────────────────────────
with tab_search:
    # Handle pending query from sidebar buttons
    default_val = st.session_state.pending_query
    if st.session_state.pending_query:
        st.session_state.pending_query = ""

    cin, cbtn = st.columns([0.85, 0.15])
    with cin:
        query = st.text_input(
            "University FAQ Query",
            value=default_val,
            label_visibility="collapsed",
            placeholder="Ask a question — 'CS-202 prerequisites', 'What happens if I miss an exam?'",
            key="query_input",
        )
    with cbtn:
        search_clicked = st.button("Ask →", type="primary", use_container_width=True)

    # ── Execute pipeline ────────────────────────────────────────
    if search_clicked and query.strip():
        with st.spinner("🔍 Running pipeline..."):
            try:
                resp = run_pipeline(query.strip(), **PIPELINE_ARGS)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()
        st.session_state.response = resp
        st.session_state.total_queries += 1
        lms = resp.get("latency_ms", {})
        st.session_state.query_history.append(
            (query.strip(), resp.get("route_decision", "?"), lms.get("total", 0))
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
        cache_type = lms.get("cache", None)
        route      = response.get("route_decision", "")
        query_type = response.get("query_type", "")

        # Feature 3: Low confidence alert
        if response.get("low_confidence"):
            warn = response.get("confidence_warning", "⚠️ Low confidence result.")
            st.warning(warn)

        # Cache banner
        if cache_type:
            icon = "⚡" if cache_type == "exact" else "🔮"
            st.markdown(f"""
            <div class="cache-banner">
                {icon} <span><strong>Cache Hit</strong> — from <strong>{cache_type}</strong>
                cache in <strong>{lms.get('total',0):.2f} ms</strong></span>
            </div>
            """, unsafe_allow_html=True)

        # Route + latency side by side
        c_route, c_lat = st.columns([0.38, 0.62])
        with c_route:
            st.markdown('<div class="section-label">Routing Decision</div>', unsafe_allow_html=True)
            render_route_panel(response, cache_type)
        with c_lat:
            st.markdown('<div class="section-label">Pipeline Latency Profile</div>', unsafe_allow_html=True)
            render_latency_panel(lms)

        render_rationale(response.get("rationale", ""))
        st.divider()

        # Apply filters (Feature 5)
        all_results = response.get("results", [])
        filtered    = _apply_filters(all_results, sel_categories, sel_tags_filter)

        if sel_categories or sel_tags_filter:
            st.caption(f"Showing {len(filtered)} of {len(all_results)} results after filters (category={sel_categories or 'all'}, tags={sel_tags_filter or 'all'})")

        render_results(filtered if (sel_categories or sel_tags_filter) else all_results)

        with st.expander("🔧 Developer View (Raw JSON)", expanded=False):
            st.json({k: v for k, v in response.items() if k != "latency_ms"})
            st.json({"latency_ms": lms})


# ─────────────────────────────────────────────────────────────
# TAB 2: COMPARE (Feature 2)
# ─────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Retrieval Comparison")
    st.caption("Runs the same query through all three retrieval strategies simultaneously.")

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
        c_sem, c_kw, c_hyb = st.columns(3)
        with c_sem:
            sem_data = cmp.get("semantic", {})
            render_compare_column(
                sem_data.get("results", []) if isinstance(sem_data, dict) else sem_data,
                f"🟢 Semantic · {sem_data.get('latency_ms', '?')} ms" if isinstance(sem_data, dict) else "🟢 Semantic",
                "semantic",
            )
        with c_kw:
            kw_data = cmp.get("keyword", {})
            render_compare_column(
                kw_data.get("results", []) if isinstance(kw_data, dict) else kw_data,
                f"🔵 Keyword · {kw_data.get('latency_ms', '?')} ms" if isinstance(kw_data, dict) else "🔵 Keyword",
                "keyword",
            )
        with c_hyb:
            hyb_data = cmp.get("hybrid", {})
            render_compare_column(
                hyb_data.get("results", []) if isinstance(hyb_data, dict) else hyb_data,
                f"🟣 Hybrid · {hyb_data.get('latency_ms', '?')} ms" if isinstance(hyb_data, dict) else "🟣 Hybrid",
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
# TAB 3: MANAGE FAQs (Feature 1)
# ─────────────────────────────────────────────────────────────
with tab_manage:
    st.markdown("### 📋 FAQ Management")
    st.caption("Add, edit, or delete FAQs. Changes are persisted to faqs.json. Reload the app to re-embed.")

    # ── Load current FAQs ──────────────────────────────────────
    try:
        current_faqs = load_faqs()
    except Exception as e:
        st.error(f"Could not load FAQs: {e}")
        current_faqs = []

    m_add, m_list = st.columns([0.42, 0.58])

    # ── ADD / EDIT form ─────────────────────────────────────────
    with m_add:
        edit_id = st.session_state.edit_faq_id
        if edit_id:
            edit_target = next((f for f in current_faqs if str(f.get("id")) == str(edit_id)), None)
            st.subheader(f"✏️ Edit FAQ: {edit_id}")
        else:
            edit_target = None
            st.subheader("➕ Add New FAQ")

        with st.form("faq_form", clear_on_submit=True):
            f_question = st.text_area(
                "Question *",
                value=edit_target.get("question", "") if edit_target else "",
                height=80,
                key="f_q",
            )
            f_answer = st.text_area(
                "Answer *",
                value=edit_target.get("answer", "") if edit_target else "",
                height=120,
                key="f_a",
            )
            cats = get_categories()
            all_cats = sorted(set(FAQ_CATEGORIES + cats))
            cat_def = edit_target.get("category", "other") if edit_target else "course"
            cat_idx = all_cats.index(cat_def) if cat_def in all_cats else 0
            f_category = st.selectbox("Category *", all_cats, index=cat_idx, key="f_cat")
            f_tags = st.text_input(
                "Tags (comma-separated)",
                value=", ".join(edit_target.get("tags", [])) if edit_target else "",
                key="f_tags",
                placeholder="e.g. prerequisite, grades, semester",
            )

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

                # Invalidate cached fixtures so next load re-embeds
                load_pipeline_fixtures.clear()
                st.session_state.edit_faq_id = None
                st.rerun()

            except ValueError as e:
                st.error(str(e))

    # ── FAQ LIST ────────────────────────────────────────────────
    with m_list:
        st.subheader(f"📄 All FAQs ({len(current_faqs)})")

        # Filter by category
        filt_cat_manage = st.selectbox(
            "Filter by category", ["(all)"] + get_categories(),
            key="manage_cat_filter",
        )
        search_q = st.text_input("Search FAQs", placeholder="Filter by question text…",
                                 key="manage_search", label_visibility="collapsed")

        shown = current_faqs
        if filt_cat_manage != "(all)":
            shown = [f for f in shown if f.get("category") == filt_cat_manage]
        if search_q:
            shown = [f for f in shown if search_q.lower() in f.get("question", "").lower()]

        st.caption(f"Showing {len(shown)} of {len(current_faqs)}")

        for faq in shown:
            fid  = faq.get("id", "?")
            cat  = faq.get("category", "")
            tags = faq.get("tags", [])
            tags_h = _tags_html(tags) if tags else ""
            cat_h  = f'<span class="cat-chip">{cat}</span>' if cat else ""

            col_card, col_edit, col_del = st.columns([7, 1.3, 1.3])
            with col_card:
                st.markdown(f"""
                <div class="manage-faq-card">
                    <div class="manage-faq-q">{faq.get("question","")}</div>
                    <div class="manage-faq-meta">{fid} &nbsp;{cat_h}&nbsp;{tags_h}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_edit:
                if st.button("✏️", key=f"edit_{fid}", help="Edit this FAQ",
                             use_container_width=True):
                    st.session_state.edit_faq_id = fid
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{fid}", help="Delete this FAQ",
                             use_container_width=True):
                    try:
                        delete_faq(fid)
                        load_pipeline_fixtures.clear()
                        st.success(f"Deleted {fid}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
