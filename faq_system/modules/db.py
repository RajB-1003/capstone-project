"""
db.py — SQLite database layer for Phase 1 (Login + Admin system).

Creates and manages the database at data/db.sqlite3.
Completely independent of all retrieval, routing, and caching modules.

Tables
------
users              — registered accounts with bcrypt-hashed passwords
faq                — FAQ entries (parallel to / mirrors data/faqs.json)
feedback           — user feedback events (mirrors feedback_log.jsonl)
unanswered_queries — low-confidence queries for admin review

Public API
----------
    init_db()                       -> None   — create tables if they don't exist
    get_connection()                -> sqlite3.Connection
    store_unanswered_query(query)   -> None   — upsert normalised query
    delete_unanswered_query(id)     -> None
    get_unanswered_queries()        -> list[dict]
    record_unanswered_query(query)  -> None   — alias for store_unanswered_query
"""

import os
import sqlite3

# ── Path ──────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_BASE_DIR, "..", "data")
DB_PATH    = os.path.join(_DATA_DIR, "db.sqlite3")

# ── DDL statements ────────────────────────────────────────────────────────────

_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    password_hash TEXT  NOT NULL,
    role        TEXT    NOT NULL DEFAULT 'user'
                        CHECK(role IN ('user', 'admin')),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_FAQ = """
CREATE TABLE IF NOT EXISTS faq (
    id          TEXT    PRIMARY KEY,   -- matches faqs.json id field (e.g. "faq_001")
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    category    TEXT    NOT NULL DEFAULT '',
    tags        TEXT    NOT NULL DEFAULT '[]',   -- JSON-encoded list
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    faq_id      TEXT    NOT NULL,
    query       TEXT    NOT NULL,
    feedback    TEXT    NOT NULL
                        CHECK(feedback IN ('up', 'down', 'not_helpful')),
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_UNANSWERED = """
CREATE TABLE IF NOT EXISTS unanswered_queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT    NOT NULL UNIQUE,
    count       INTEGER NOT NULL DEFAULT 1,
    last_seen   TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_ALL_DDL = [_CREATE_USERS, _CREATE_FAQ, _CREATE_FEEDBACK, _CREATE_UNANSWERED]


# ── Public API ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """
    Return a new SQLite connection to the database.

    The caller is responsible for closing the connection (or using it as
    a context manager).  Row factory is set to sqlite3.Row so columns can
    be accessed by name.

    Returns:
        sqlite3.Connection with row_factory = sqlite3.Row.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent Streamlit reruns
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """
    Create all required tables if they do not already exist.

    Safe to call multiple times — all DDL statements use
    CREATE TABLE IF NOT EXISTS.  Called once at app startup.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = get_connection()
    try:
        for ddl in _ALL_DDL:
            conn.execute(ddl)
        conn.commit()
    finally:
        conn.close()


# ── Unanswered query helpers (used by pipeline output layer — Phase 5) ────────

def store_unanswered_query(query: str) -> None:
    """
    Upsert a low-confidence query into unanswered_queries.

    Normalisation (Step 1 spec):
      - Strip leading/trailing whitespace
      - Lowercase
    This ensures case-insensitive deduplication:
      "Hostel room?", "hostel room?" and "  hostel room?  " all map to the
      same row (Step 4 spec).

    Quality gate (Step 2 spec):
      Calls is_valid_unanswered_query() before storing. Invalid/noisy
      queries (too short, no alpha chars, garbage) are silently discarded
      regardless of the call site.

    If the normalised query already exists:
        count = count + 1, last_seen = now
    If new:
        INSERT row with count = 1.

    Args:
        query: Raw user query that received a low-confidence response.
    """
    # Step 2: quality gate — applied unconditionally at write time
    from modules.query_filter import is_valid_unanswered_query
    if not is_valid_unanswered_query(query):
        return

    normalised = query.strip().lower()
    if not normalised:
        return

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, count FROM unanswered_queries WHERE query = ?",
            (normalised,),
        ).fetchone()

        if row:
            conn.execute(
                """
                UPDATE unanswered_queries
                SET count = count + 1, last_seen = datetime('now')
                WHERE id = ?
                """,
                (row["id"],),
            )
        else:
            conn.execute(
                """
                INSERT INTO unanswered_queries (query, count, last_seen)
                VALUES (?, 1, datetime('now'))
                """,
                (normalised,),
            )
        conn.commit()
    finally:
        conn.close()


# Backward-compat alias used by existing Phase 1 code
record_unanswered_query = store_unanswered_query


def delete_unanswered_query(query: str) -> None:
    """Delete an unanswered query record by its exact query text."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM unanswered_queries WHERE query = ?", (query,))
        conn.commit()
    finally:
        conn.close()


def get_unanswered_queries() -> list[dict]:
    """
    Return all unanswered query records ordered by count descending.

    Returns:
        List of dicts: { id, query, count, last_seen }
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, query, count, last_seen "
            "FROM unanswered_queries ORDER BY count DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_faqs() -> list[dict]:
    """Return all FAQs from the SQLite database."""
    import json
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, question, answer, category, tags, created_at FROM faq"
        ).fetchall()
        
        faqs = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d["tags"])
            faqs.append(d)
        return faqs
    finally:
        conn.close()


def add_faq(question: str, answer: str, category: str = "", tags: str = "") -> None:
    """Add a new FAQ to the database and generate a unique faq_id."""
    import json
    import uuid
    conn = get_connection()
    
    faq_id = f"faq_{uuid.uuid4().hex[:8]}"
    
    # ensure tags is a list, parse string if needed
    if isinstance(tags, str):
        tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        tags_list = tags or []
        
    tags_json = json.dumps(tags_list)
    
    try:
        conn.execute(
            """
            INSERT INTO faq (id, question, answer, category, tags, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (faq_id, question, answer, category, tags_json)
        )
        conn.commit()
    finally:
        conn.close()
