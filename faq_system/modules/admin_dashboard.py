"""
admin_dashboard.py — Phase 3: Read-only Admin Dashboard.

Provides:
  - DB read helpers: get_overview_stats(), get_top_faqs(), get_unanswered_queries()
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


# ── Step 3: Streamlit render function ─────────────────────────────────────────


def render_admin_dashboard() -> None:
    """
    Render the read-only admin dashboard inside the current Streamlit context.

    Layout:
      Section 1 — Overview metrics (4 st.metric cards)
      Section 2 — Top FAQs by feedback event count
      Section 3 — Unanswered / low-confidence queries
      Section 4 — FAQs flagged as low-quality by the feedback system

    Called by app.py ONLY when the logged-in user's role == "admin".
    """
    st.markdown(
        "<h2 style='margin-bottom:4px;'>📊 Admin Dashboard</h2>"
        "<p style='color:#64748b;font-size:0.85rem;margin-top:0;'>"
        "Read-only insights · auto-refreshed on each page load</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Section 1: Overview metrics ───────────────────────────────────────────
    stats = get_overview_stats()

    # Also pull live JSON-based FAQ count for the metric note
    _json_faq_count = 0
    try:
        import json, os
        _faq_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "faqs.json"
        )
        with open(_faq_path, encoding="utf-8") as _f:
            _json_faq_count = len(json.load(_f))
    except Exception:
        pass

    _feedback_log_count = 0
    try:
        import pathlib
        from modules.feedback_store import FEEDBACK_LOG_PATH
        _log = pathlib.Path(FEEDBACK_LOG_PATH)
        if _log.exists():
            _feedback_log_count = sum(1 for _ in _log.open(encoding="utf-8"))
    except Exception:
        pass

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Users",            stats["total_users"])
    m2.metric(
        "📚 FAQs",
        _json_faq_count or stats["total_faqs"],
        help="Count from faqs.json (live knowledge base); "
             f"DB table has {stats['total_faqs']} rows (migration pending).",
    )
    m3.metric(
        "💬 Feedback Events",
        _feedback_log_count or stats["total_feedback"],
        help="Count from feedback_log.jsonl (live); "
             f"DB table has {stats['total_feedback']} rows.",
    )
    m4.metric("❓ Unanswered Queries", stats["total_unanswered"])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Section 2: Top FAQs by feedback volume ────────────────────────────────
    st.subheader("🔥 Top FAQs by Feedback Volume")
    top_faqs = get_top_faqs(limit=10)

    if not top_faqs:
        st.info(
            "No feedback events in the DB yet.  "
            "Feedback is currently stored in `feedback_log.jsonl`; once DB "
            "sync is enabled (Phase 7) this table will populate.",
            icon="ℹ️",
        )
        # Fallback: show aggregation from JSON feedback log
        try:
            from modules.feedback_store import get_aggregated_scores
            agg = get_aggregated_scores()
            if agg:
                st.caption("Showing from JSON feedback log (live source):")
                rows = sorted(
                    agg.items(),
                    key=lambda kv: kv[1].get("count", 0),
                    reverse=True,
                )[:10]
                table_data = [
                    {
                        "FAQ ID":       fid,
                        "Events":       d.get("count", 0),
                        "Avg Feedback": round(d.get("avg_feedback", 0), 3),
                        "Score Sum":    round(d.get("weighted_sum", 0), 2),
                    }
                    for fid, d in rows
                ]
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception:
            pass
    else:
        st.dataframe(
            [
                {
                    "FAQ ID":    r["faq_id"],
                    "Question":  r["question"],
                    "Events":    r["frequency"],
                }
                for r in top_faqs
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Section 3: Unanswered / low-confidence queries ────────────────────────
    st.subheader("❓ Unanswered Queries")
    st.caption(
        "Queries that triggered a low-confidence result and passed the quality "
        "filter (≥ 3 words, ≥ 10 chars, at least one alphabetic character). "
        "Sorted by recurrence count."
    )
    unanswered = get_unanswered_queries(limit=20)

    if not unanswered:
        st.info("No unanswered queries recorded yet.", icon="ℹ️")
    else:
        for r in unanswered:
            q = r["query"]
            with st.container(border=True):
                st.markdown(f"**{q}** (x{r['count']})")
                c1, c2, _ = st.columns([0.2, 0.2, 0.6])
                
                # Use unique keys based on query hash to avoid duplicates
                btn_key_add = f"add_{hash(q)}"
                btn_key_rem = f"rem_{hash(q)}"
                form_open_key = f"form_open_{hash(q)}"
                
                with c1:
                    if st.button("➕ Add Answer", key=btn_key_add, use_container_width=True):
                        st.session_state[form_open_key] = True
                with c2:
                    if st.button("🗑️ Remove", key=btn_key_rem, use_container_width=True):
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
                                
                                # 1. Insert into faq table
                                add_faq(q, new_ans, new_cat, new_tags)
                                # 2. Remove from unanswered_queries
                                delete_unanswered_query(q)
                                # 3. Invalidate embeddings cache
                                delete_embeddings()
                                st.cache_resource.clear()
                                
                                st.session_state[form_open_key] = False
                                st.success("FAQ added successfully! Embeddings will regenerate.")
                                # Need to use st.rerun instead of st.experimental_rerun in newer streamlit
                                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Section 4: Low-quality FAQs ───────────────────────────────────────────
    st.subheader("⚠️ Low-Quality FAQs")
    st.caption(
        "FAQs flagged by the feedback reranker "
        "(weighted_sum ≤ LOW_QUALITY_THRESHOLD) based on down-votes and "
        "\"not helpful\" events."
    )
    poor = get_poor_faqs_from_db()

    if not poor:
        st.success("No FAQs currently flagged as low-quality.", icon="✅")
    else:
        st.dataframe(
            poor,
            use_container_width=True,
            hide_index=True,
            column_config={
                "faq_id":       st.column_config.TextColumn("FAQ ID"),
                "weighted_sum": st.column_config.NumberColumn("Score Sum"),
                "avg_feedback": st.column_config.NumberColumn("Avg Feedback"),
                "count":        st.column_config.NumberColumn("Events"),
            },
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption(
        f"Dashboard auto-refreshes on each interaction. "
        f"Data sources: SQLite (`data/db.sqlite3`) + live JSON logs."
    )
