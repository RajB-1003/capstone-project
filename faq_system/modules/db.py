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
query_logs         — Phase 5: per-query analytics (route, latency, cache hit)

Public API
----------
    init_db()                       -> None   — create tables if they don't exist
    get_connection()                -> sqlite3.Connection
    store_unanswered_query(query)   -> None   — upsert normalised query
    delete_unanswered_query(id)     -> None
    get_unanswered_queries()        -> list[dict]
    record_unanswered_query(query)  -> None   — alias for store_unanswered_query
    log_query(query, route,         -> None   — Phase 5: insert one analytics row
              confidence, latency,
              cache_hit)
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

_CREATE_QUERY_LOGS = """
CREATE TABLE IF NOT EXISTS query_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT    NOT NULL,
    route       TEXT    NOT NULL DEFAULT '',
    confidence  REAL    NOT NULL DEFAULT 0.0,
    latency_ms  REAL    NOT NULL DEFAULT 0.0,
    cache_hit   INTEGER NOT NULL DEFAULT 0,   -- SQLite boolean: 0/1
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_SESSION_TOKENS = """
CREATE TABLE IF NOT EXISTS session_tokens (
    token       TEXT    PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    last_page   TEXT    NOT NULL DEFAULT 'search',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    expires_at  TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

_CREATE_SEARCH_HISTORY = """
CREATE TABLE IF NOT EXISTS search_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    query       TEXT    NOT NULL,
    route       TEXT    NOT NULL DEFAULT '',
    latency_ms  REAL    NOT NULL DEFAULT 0.0,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

_ALL_DDL = [
    _CREATE_USERS, _CREATE_FAQ, _CREATE_FEEDBACK, 
    _CREATE_UNANSWERED, _CREATE_QUERY_LOGS, _CREATE_SESSION_TOKENS,
    _CREATE_SEARCH_HISTORY
]


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


# ── Phase 5: Query analytics logging ─────────────────────────────────────────

def log_query(
    query: str,
    route: str,
    confidence: float,
    latency: float,
    cache_hit: bool,
) -> None:
    """
    Insert one analytics row into query_logs.

    This is a fire-and-forget function: any DB error is silently swallowed
    so that logging never affects the pipeline response.

    Args:
        query:      Original user query string.
        route:      Routing decision ("keyword", "semantic", "hybrid", "cached").
        confidence: Top-result confidence score (float in [0, 1]).
        latency:    Total pipeline latency in milliseconds.
        cache_hit:  True if the response was served from any cache layer.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO query_logs (query, route, confidence, latency_ms, cache_hit, timestamp)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    query,
                    route,
                    float(confidence),
                    float(latency),
                    1 if cache_hit else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        _log.warning("[query_logs] logging error: %s", exc)


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


# ── Session Token Management ──────────────────────────────────────────────────

_TOKEN_TTL_DAYS = 7


def create_session_token(user_id: int, last_page: str = "search") -> str:
    """
    Create a new session token for the given user and persist it in the DB.

    Automatically sets expires_at = now + TOKEN_TTL_DAYS.
    Any existing tokens for the same user are left in place (multi-session
    support); purge_expired_tokens() cleans stale rows at startup.

    Args:
        user_id:   Integer primary key of the users table.
        last_page: The page the user was on (default: "search").

    Returns:
        The newly created token string (UUID4).
    """
    import uuid
    token = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO session_tokens (token, user_id, last_page, created_at, expires_at)
            VALUES (
                ?,
                ?,
                ?,
                datetime('now'),
                datetime('now', '+{} days')
            )
            """.format(_TOKEN_TTL_DAYS),
            (token, user_id, last_page),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def validate_session_token(token: str) -> dict | None:
    """
    Validate a session token and return the associated user record.

    Checks that the token exists AND has not expired (expires_at > now).

    Args:
        token: The UUID token string stored in the browser URL.

    Returns:
        A dict with keys { id, username, role, created_at, last_page } if the
        token is valid, or None if invalid / expired.
    """
    if not token:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT u.id, u.username, u.role, u.created_at, t.last_page
            FROM session_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token = ?
              AND t.expires_at > datetime('now')
            """,
            (token,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_session_token(token: str) -> None:
    """
    Delete a specific session token from the DB (called on logout).

    Args:
        token: The token string to invalidate.
    """
    if not token:
        return
    conn = get_connection()
    try:
        conn.execute("DELETE FROM session_tokens WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def update_token_last_page(token: str, page: str) -> None:
    """
    Update the last_page column for an active session token.

    Called on every page-navigation button click so that refresh returns
    the user to the page they were on.

    Args:
        token: Active session token.
        page:  Page identifier string (e.g. "search", "admin", "compare").
    """
    if not token or not page:
        return
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE session_tokens SET last_page = ? WHERE token = ?",
            (page, token),
        )
        conn.commit()
    finally:
        conn.close()


def purge_expired_tokens() -> None:
    """
    Delete all expired session token rows from the DB.

    Called once at app startup (inside init_db / init_session) to keep
    the session_tokens table from growing unboundedly.
    """
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM session_tokens WHERE expires_at <= datetime('now')"
        )
        conn.commit()
    finally:
        conn.close()


# ── Search History (Phase 7 Isolation) ────────────────────────────────────────

def store_query(user_id: int, query: str, route: str, latency_ms: float) -> None:
    """
    Store a user's search query in their isolated history log.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO search_history (user_id, query, route, latency_ms)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, query.strip(), route, latency_ms),
        )
        conn.commit()
    finally:
        conn.close()


def get_history(user_id: int, limit: int = 10) -> list[tuple[str, str, float]]:
    """
    Retrieve the most recent queries for a specific user.
    Returns a list of (query, route, latency_ms) matching the UI expectation.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT query, route, latency_ms
            FROM search_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        # Return in chronological order (oldest to newest of the limit window)
        # so `reversed(history[-10:])` in the UI works naturally.
        results = [(r["query"], r["route"], r["latency_ms"]) for r in rows]
        results.reverse()
        return results
    finally:
        conn.close()
