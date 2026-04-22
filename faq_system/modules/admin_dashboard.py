"""
admin_dashboard.py — Phase 3: Read-only Admin Dashboard.

Provides:
  - DB read helpers: get_overview_stats(), get_top_faqs(), get_unanswered_queries()
  - Phase 5 analytics helpers: get_total_queries(), get_avg_latency(),
    get_route_distribution(), get_cache_hit_rate()
  - Streamlit render function: render_admin_dashboard()

Rules:
  - Strictly READ-ONLY — no inserts, updates, or deletes here
  - No changes to retrieval, pipeline, feedback, or multilingual modules
  - Admin-only: caller (app.py) must enforce role check before calling render
"""

from __future__ import annotations

import streamlit as st

from modules.db import get_connection

# ── Step 2: Database read helpers ─────────────────────────────────────────────


def get_overview_stats() -> dict:
    """
    Return aggregate counts from all four tracked tables.

    Returns:
        {
            "total_users":       int,
            "total_faqs":        int,  # rows in DB faq table
            "total_feedback":    int,  # rows in DB feedback table
            "total_unanswered":  int,  # rows in unanswered_queries table
        }

    Note:
        The faq and feedback tables may be empty if JSON-only mode is in use
        (Phase 7 migration is not yet complete).  Counts reflect only what is
        stored in SQLite; JSON-backed counts are shown separately in the UI.
    """
    conn = get_connection()
    try:
        total_users      = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_faqs       = conn.execute("SELECT COUNT(*) FROM faq").fetchone()[0]
        total_feedback   = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        total_unanswered = conn.execute(
            "SELECT COUNT(*) FROM unanswered_queries"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "total_users":      total_users,
        "total_faqs":       total_faqs,
        "total_feedback":   total_feedback,
        "total_unanswered": total_unanswered,
    }


def get_top_faqs(limit: int = 10) -> list[dict]:
    """
    Return the most-discussed FAQs based on feedback event count.

    Queries the DB feedback table (populated when users give thumbs
    up/down/not_helpful via the UI).

    Args:
        limit: Maximum number of FAQs to return (default 10).

    Returns:
        List of { faq_id, question, frequency } dicts, ordered by
        frequency descending.  `question` is "--" if the faq_id has no
        matching row in the faq table yet.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT f.faq_id,
                   COALESCE(q.question, '--') AS question,
                   COUNT(*) AS frequency
            FROM feedback f
            LEFT JOIN faq q ON q.id = f.faq_id
            GROUP BY f.faq_id
            ORDER BY frequency DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_unanswered_queries(limit: int = 20) -> list[dict]:
    """
    Return the most-repeated unanswered queries, newest conflicts first.

    Args:
        limit: Maximum number of rows to return (default 20).

    Returns:
        List of { query, count, last_seen } dicts, ordered by count desc.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT query, count, last_seen
            FROM unanswered_queries
            ORDER BY count DESC, last_seen DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_poor_faqs_from_db() -> list[dict]:
    """
    Return FAQs flagged as low-quality by the feedback system.

    Uses the feedback_store aggregation logic (not the DB feedback table)
    so it stays in sync with the JSON-based feedback log that the pipeline uses.

    Returns:
        List of { faq_id, weighted_sum, avg_feedback, count } dicts.
    """
    try:
        from modules.feedback_store import get_aggregated_scores, POOR_SCORE_THRESHOLD
        agg = get_aggregated_scores()
        poor = []
        for faq_id, data in agg.items():
            if data.get("weighted_sum", 0) <= POOR_SCORE_THRESHOLD:
                poor.append({
                    "faq_id":       faq_id,
                    "weighted_sum": round(data.get("weighted_sum", 0), 2),
                    "avg_feedback": round(data.get("avg_feedback", 0), 3),
                    "count":        data.get("count", 0),
                })
        return sorted(poor, key=lambda x: x["weighted_sum"])
    except Exception:
        return []


# ── Phase 5: System Analytics helpers ────────────────────────────────────────


def get_total_queries() -> int:
    """
    Return the total number of queries logged in query_logs.

    Returns:
        int: Total row count in query_logs (0 if table is empty or missing).
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0
    finally:
        conn.close()


def get_avg_latency() -> float:
    """
    Return the average pipeline latency across all logged queries.

    Returns:
        float: AVG(latency_ms) rounded to 1 decimal place, or 0.0 if no rows.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT AVG(latency_ms) FROM query_logs").fetchone()
        val = row[0] if row and row[0] is not None else 0.0
        return round(float(val), 1)
    except Exception:
        return 0.0
    finally:
        conn.close()


def get_route_distribution() -> dict[str, int]:
    """
    Return query counts grouped by routing decision.

    Returns:
        dict: { route_label: count } — e.g. {"keyword": 3, "semantic": 5, ...}
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT route, COUNT(*) AS cnt FROM query_logs GROUP BY route"
        ).fetchall()
        return {r["route"]: r["cnt"] for r in rows}
    except Exception:
        return {}
    finally:
        conn.close()


def get_cache_hit_rate() -> float:
    """
    Return the fraction of queries served from cache (exact or semantic).

    Returns:
        float: Percentage (0.0–100.0) of cache hits, or 0.0 if no rows.
    """
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
        if not total:
            return 0.0
        hits = conn.execute(
            "SELECT COUNT(*) FROM query_logs WHERE cache_hit = 1"
        ).fetchone()[0]
        return round((hits / total) * 100, 1)
    except Exception:
        return 0.0
    finally:
        conn.close()


# ── Step 3: Streamlit render function ─────────────────────────────────────────


def get_faq_mapping() -> dict[str, str]:
    """Helper to get FAQ ID -> Question mapping from JSON"""
    try:
        import json, os
        faq_path = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.json")
        with open(faq_path, encoding="utf-8") as f:
            faqs = json.load(f)
            return {item["id"]: item.get("question", item["id"]) for item in faqs}
    except Exception:
        return {}


def get_faq_mapping() -> dict[str, str]:
    """Helper to get FAQ ID -> Question mapping from JSON"""
    try:
        import json, os
        faq_path = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.json")
        with open(faq_path, encoding="utf-8") as f:
            faqs = json.load(f)
            return {item["id"]: item.get("question", item["id"]) for item in faqs}
    except Exception:
        return {}


def render_admin_dashboard() -> None:
    """
    Render the insight-driven admin dashboard inside the current Streamlit context.
    """
    
    # Inject minimal custom CSS for background highlighting, animations, and headers
    st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .admin-container {
        animation: fadeIn 0.4s ease-out;
    }
    .admin-header {
        color: #4f46e5;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 12px;
        transition: all 0.2s ease;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .bg-blue { background-color: #f0f9ff; border-color: #bae6fd; }
    .bg-purple { background-color: #faf5ff; border-color: #e9d5ff; }
    .bg-green { background-color: #f0fdf4; border-color: #bbf7d0; }
    .bg-orange { background-color: #fff7ed; border-color: #fed7aa; }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-delta {
        font-size: 0.75rem;
        font-weight: 600;
    }
    .delta-up { color: #16a34a; }
    .delta-down { color: #ef4444; }
    .section-spacing {
        margin-top: 3rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='admin-container'>", unsafe_allow_html=True)
    
    # ── Data Fetching ────────────────────────────────────────────────────────
    stats = get_overview_stats()
    faq_map = get_faq_mapping()
    total_queries = get_total_queries()
    avg_latency   = get_avg_latency()
    cache_rate    = get_cache_hit_rate()
    
    _feedback_log_count = 0
    try:
        import pathlib
        from modules.feedback_store import FEEDBACK_LOG_PATH
        _log = pathlib.Path(FEEDBACK_LOG_PATH)
        if _log.exists():
            _feedback_log_count = sum(1 for _ in _log.open(encoding="utf-8"))
    except Exception:
        pass
    
    total_feedback = _feedback_log_count or stats.get("total_feedback", 0)

    # ── 1. System Analytics (Hero Section) ──────────────────────────────────
    st.markdown("### 📊 System Analytics")

    # Row 1 — Content metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card bg-blue">
            <div class="metric-label">👥 Total Users</div>
            <div class="metric-value">{stats.get("total_users", 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card bg-purple">
            <div class="metric-label">📚 Total FAQs</div>
            <div class="metric-value">{stats.get("total_faqs", 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card bg-green">
            <div class="metric-label">❗ Unanswered Queries</div>
            <div class="metric-value">{stats.get("total_unanswered", 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Row 2 — Activity metrics (centered with empty side columns)
    _, col4, col5, _ = st.columns([0.5, 1, 1, 0.5])

    with col4:
        st.markdown(f"""
        <div class="metric-card bg-orange">
            <div class="metric-label">⚡ Total Queries</div>
            <div class="metric-value">{total_queries}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card bg-blue">
            <div class="metric-label">💬 Feedback Events</div>
            <div class="metric-value">{total_feedback}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── 2. Feedback Distribution + Trend ──────────────────────────────────────
    st.markdown("### 📈 Feedback Insights")
    
    top_faqs = get_top_faqs(limit=50)
    try:
        from modules.feedback_store import get_aggregated_scores
        agg = get_aggregated_scores()
    except Exception:
        agg = {}

    chart_data = []
    if not top_faqs and agg:
        rows = sorted(agg.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:10]
        for fid, d in rows:
            q = faq_map.get(fid, fid)
            if q.startswith("faq_"): q = "Unknown Question"
            q_short = q[:20] + "..." if len(q) > 20 else q
            chart_data.append({"FAQ": q_short, "Events": d.get("count", 0)})
    elif top_faqs:
        sorted_top_faqs = sorted(top_faqs, key=lambda x: x["frequency"], reverse=True)[:10]
        for r in sorted_top_faqs:
            fid = r["faq_id"]
            q = faq_map.get(fid, r["question"])
            if q == "--": q = faq_map.get(fid, fid)
            if q.startswith("faq_"): q = "Unknown Question"
            q_short = q[:20] + "..." if len(q) > 20 else q
            chart_data.append({"FAQ": q_short, "Events": r["frequency"]})

    chart_data = sorted(chart_data, key=lambda x: x["Events"], reverse=True)

    st.markdown("**Feedback Distribution**")
    if chart_data:
        import plotly.graph_objects as go

        # Full questions (untruncated) stored in customdata for hover
        full_questions = [x["FAQ"] for x in chart_data]
        events = [x["Events"] for x in chart_data]
        x_labels = [f"#{i+1}" for i in range(len(events))]   # clean bar numbers
        n = len(events)

        # Gradient palette: indigo → violet → purple
        colors = [
            f"rgba({int(99 + (139-99)*i/(max(n-1,1)))}, "
            f"{int(102 + (92-102)*i/(max(n-1,1)))}, "
            f"{int(241 + (246-241)*i/(max(n-1,1)))}, 0.90)"
            for i in range(n)
        ]

        fig = go.Figure(go.Bar(
            x=x_labels,
            y=events,
            customdata=full_questions,
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.15)", width=1.2),
                cornerradius=8,
            ),
            text=events,
            textposition="outside",
            textfont=dict(size=12, color="#4f46e5", family="Inter, sans-serif"),
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Feedback events: <b>%{y}</b><extra></extra>"
            ),
        ))

        fig.update_layout(
            plot_bgcolor="rgba(248,250,252,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#475569"),
            margin=dict(l=10, r=10, t=30, b=40),
            height=360,
            xaxis=dict(
                tickfont=dict(size=11, color="#94a3b8"),
                showgrid=False,
                showline=True,
                linecolor="#e2e8f0",
                linewidth=1.5,
            ),
            yaxis=dict(
                title=dict(text="Feedback Count", font=dict(size=12, color="#94a3b8")),
                tickfont=dict(size=11, color="#94a3b8"),
                gridcolor="rgba(226,232,240,0.8)",
                gridwidth=1,
                showline=False,
                zeroline=False,
            ),
            hoverlabel=dict(
                bgcolor="#1e293b",
                font_color="#f8fafc",
                font_size=13,
                bordercolor="#334155",
                namelength=0,
            ),
            bargap=0.35,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Not enough feedback data to display chart.")
        
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── 3. Top FAQs ──────────────────────────────────────────────────────────
    st.markdown("### 🔥 Most Asked Questions")
    
    table_data = []
    if not top_faqs and agg:
        rows = sorted(agg.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:15]
        for fid, d in rows:
            q = faq_map.get(fid, fid)
            if q.startswith("faq_"): q = "Unknown Question"
            table_data.append({
                "Question": q,
                "Times Asked": d.get("count", 0)
            })
    elif top_faqs:
        for r in sorted(top_faqs, key=lambda x: x["frequency"], reverse=True)[:15]:
            fid = r["faq_id"]
            q = faq_map.get(fid, r["question"])
            if q == "--": q = faq_map.get(fid, fid)
            if q.startswith("faq_"): q = "Unknown Question"
            table_data.append({
                "Question": q,
                "Times Asked": r["frequency"]
            })
            
    if table_data:
        import pandas as pd
        df = pd.DataFrame(table_data).sort_values(by="Times Asked", ascending=False).reset_index(drop=True)
        max_count = df["Times Asked"].max() or 1

        rank_colors  = ["#f59e0b", "#94a3b8", "#cd7f32"]   # gold, silver, bronze
        rank_bg      = ["#fef3c7", "#f1f5f9", "#fdf4e7"]

        import textwrap
        
        rows_html = ""
        for i, row in df.iterrows():
            rank      = i + 1
            question  = row["Question"]
            count     = int(row["Times Asked"])
            pct       = round((count / max_count) * 100)

            if rank <= 3:
                badge_color = rank_colors[rank - 1]
                badge_bg    = rank_bg[rank - 1]
            else:
                badge_color = "#6366f1"
                badge_bg    = "#eef2ff"

            bar_color = "#6366f1" if rank > 3 else badge_color
            row_bg    = "#ffffff" if rank % 2 == 0 else "#f8fafc"

            rows_html += textwrap.dedent(f"""
            <tr style="background:{row_bg}; transition:background 0.15s;"
                onmouseover="this.style.background='#eef2ff'"
                onmouseout="this.style.background='{row_bg}'">
              <td style="padding:12px 16px; width:48px; text-align:center;">
                <span style="display:inline-flex; align-items:center; justify-content:center;
                             width:30px; height:30px; border-radius:50%;
                             background:{badge_bg}; color:{badge_color};
                             font-weight:700; font-size:0.78rem; border:1.5px solid {badge_color}33;">
                  {rank}
                </span>
              </td>
              <td style="padding:12px 16px; color:#1e293b; font-size:0.9rem; font-weight:500; line-height:1.4;">
                {question}
              </td>
              <td style="padding:12px 24px 12px 16px; width:220px;">
                <div style="display:flex; align-items:center; gap:10px;">
                  <div style="flex:1; background:#e2e8f0; border-radius:999px; height:8px; overflow:hidden;">
                    <div style="width:{pct}%; height:100%; border-radius:999px;
                                background:linear-gradient(90deg, {bar_color}, {bar_color}cc);
                                transition:width 0.4s ease;"></div>
                  </div>
                  <span style="min-width:28px; text-align:right; font-weight:700;
                               font-size:0.88rem; color:{badge_color};">{count}</span>
                </div>
              </td>
            </tr>""")

        table_html = textwrap.dedent(f"""
        <div style="border-radius:14px; overflow:hidden; border:1.5px solid #e2e8f0;
                    box-shadow:0 2px 12px rgba(99,102,241,0.07); margin-top:8px;">
          <table style="width:100%; border-collapse:collapse; font-family:'Inter',sans-serif;">
            <thead>
              <tr style="background:linear-gradient(135deg,#6366f1,#8b5cf6);">
                <th style="padding:14px 16px; color:#fff; font-size:0.75rem; font-weight:600;
                           letter-spacing:0.08em; text-transform:uppercase; width:48px;">#</th>
                <th style="padding:14px 16px; color:#fff; font-size:0.75rem; font-weight:600;
                           letter-spacing:0.08em; text-transform:uppercase; text-align:left;">Question</th>
                <th style="padding:14px 24px 14px 16px; color:#fff; font-size:0.75rem; font-weight:600;
                           letter-spacing:0.08em; text-transform:uppercase; width:220px;">Frequency</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        """)
        
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No queries recorded yet.")
        
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── 4. Poor Performing FAQs ──────────────────────────────────────────────
    st.markdown("### ⚠️ Poor Performing FAQs")
    poor = get_poor_faqs_from_db()
    poor_table = []
    for p in poor:
        fid = p["faq_id"]
        q = faq_map.get(fid, fid)
        if q.startswith("faq_"): q = "Unknown Question"
        poor_table.append({
            "Question": q,
            "Score": p["weighted_sum"],
            "Status": "⚠️ Needs Review"
        })
    
    if poor_table:
        st.dataframe(
            poor_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large")
            }
        )
    else:
        st.success("No FAQs are currently flagged as poor performing.")
        
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── 5. Unanswered Queries ────────────────────────────────────────────────
    unanswered = get_unanswered_queries(limit=20)
    u_count = len(unanswered)
    st.markdown(f"### ❓ Unanswered Queries ({u_count})")

    if not unanswered:
        st.success("No unanswered queries recorded yet.")
    else:
        for r in unanswered:
            q = r["query"]
            with st.container(border=True):
                c1, c2 = st.columns([0.8, 0.2])
                with c1:
                    st.markdown(f"**{q}**")
                    st.caption(f"Times Asked: **{r['count']}**")
                with c2:
                    btn_key_add = f"add_{hash(q)}"
                    btn_key_rem = f"rem_{hash(q)}"
                    form_open_key = f"form_open_{hash(q)}"
                    
                    if st.button("➕ Answer", key=btn_key_add, use_container_width=True):
                        st.session_state[form_open_key] = True
                    if st.button("🗑️ Ignore", key=btn_key_rem, use_container_width=True):
                        from modules.db import delete_unanswered_query
                        delete_unanswered_query(q)
                        st.rerun()

                if st.session_state.get(form_open_key):
                    st.markdown("---")
                    with st.form(key=f"form_{hash(q)}"):
                        st.caption("Convert query to new FAQ")
                        new_ans = st.text_area("Answer", height=100)
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            new_cat = st.text_input("Category (e.g. course, exam)")
                        with cc2:
                            new_tags = st.text_input("Tags (comma separated)")
                        
                        if st.form_submit_button("Submit FAQ"):
                            if not new_ans.strip():
                                st.error("Answer cannot be empty.")
                            else:
                                from modules.db import add_faq, delete_unanswered_query
                                from modules.embedding_store import delete_embeddings
                                add_faq(q, new_ans, new_cat, new_tags)
                                delete_unanswered_query(q)
                                delete_embeddings()
                                st.cache_resource.clear()
                                st.session_state[form_open_key] = False
                                st.success("FAQ added successfully!")
                                st.rerun()
                                
    st.markdown("</div>", unsafe_allow_html=True)
