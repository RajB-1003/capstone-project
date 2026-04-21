"""
auth.py — Authentication layer for Phase 1 (Login + Admin system).

Provides user registration and login backed by the SQLite users table
created in db.py.  All passwords are hashed with bcrypt before storage.

Session management uses Streamlit's session_state:
    st.session_state["auth_user"]  — dict { id, username, role } or None
    st.session_state["auth_token"] — opaque session token (UUID) or None

No pipeline, routing, caching, or retrieval modules are touched here.

Public API
----------
    init_session()                    -> None
    register_user(username, password,
                  role="user")        -> dict  (or raises ValueError)
    login_user(username, password)    -> dict  (or raises ValueError)
    logout_user()                     -> None
    get_current_user()                -> dict | None
    is_authenticated()                -> bool
    is_admin()                        -> bool
    require_auth(role=None)           -> dict  (raises if not authed)
"""

import uuid
import bcrypt
import streamlit as st

from modules.db import get_connection, init_db


# ── Role constants ─────────────────────────────────────────────────────────────

ROLE_USER  = "user"
ROLE_ADMIN = "admin"

_VALID_ROLES = {ROLE_USER, ROLE_ADMIN}

# ── Session keys ──────────────────────────────────────────────────────────────

_KEY_USER  = "auth_user"    # dict { id, username, role } once logged in
_KEY_TOKEN = "auth_token"   # UUID string used as a lightweight session token


# ── Session helpers ───────────────────────────────────────────────────────────

def init_session() -> None:
    """
    Initialise authentication keys in Streamlit session_state.

    Call once at the top of the Streamlit script before any auth check.
    Safe to call on every rerun — only sets keys that are not yet present.
    Also ensures the database and its tables exist.
    """
    init_db()   # idempotent — creates tables if missing
    st.session_state.setdefault(_KEY_USER,  None)
    st.session_state.setdefault(_KEY_TOKEN, None)


def get_current_user() -> dict | None:
    """Return the logged-in user dict, or None if not authenticated."""
    return st.session_state.get(_KEY_USER)


def is_authenticated() -> bool:
    """Return True if a user is currently logged in."""
    return get_current_user() is not None


def is_admin() -> bool:
    """Return True if the logged-in user has the 'admin' role."""
    user = get_current_user()
    return user is not None and user.get("role") == ROLE_ADMIN


def logout_user() -> None:
    """Clear the session, effectively logging the user out."""
    st.session_state[_KEY_USER]  = None
    st.session_state[_KEY_TOKEN] = None


# ── Registration ──────────────────────────────────────────────────────────────

def register_user(
    username: str,
    password: str,
    role:     str = ROLE_USER,
) -> dict:
    """
    Register a new user and return the newly created user record.

    Args:
        username: Must be non-empty; case-sensitive; must be unique in DB.
        password: Plain-text password; minimum 6 characters.
        role:     "user" (default) or "admin".

    Returns:
        dict: { id, username, role, created_at }

    Raises:
        ValueError: on invalid input or if the username is already taken.
    """
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if role not in _VALID_ROLES:
        raise ValueError(f"Role must be one of: {sorted(_VALID_ROLES)}")

    # bcrypt hash — work factor 12 is the recommended default
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))

    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, pw_hash.decode("utf-8"), role),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, username, role, created_at FROM users WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row)
    except Exception as exc:
        # SQLite UNIQUE constraint raises sqlite3.IntegrityError
        if "UNIQUE" in str(exc).upper():
            raise ValueError(f"Username '{username}' is already taken.") from exc
        raise
    finally:
        conn.close()


# ── Login ─────────────────────────────────────────────────────────────────────

def login_user(username: str, password: str) -> dict:
    """
    Verify credentials and create a session if valid.

    On success, sets st.session_state[_KEY_USER] and [_KEY_TOKEN].

    Args:
        username: Exact username (case-sensitive).
        password: Plain-text password to verify against stored hash.

    Returns:
        dict: { id, username, role, created_at }

    Raises:
        ValueError: if credentials are invalid (intentionally vague message
                    to avoid username enumeration).
    """
    username = username.strip()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, role, created_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError("Invalid username or password.")

    stored_hash = row["password_hash"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        raise ValueError("Invalid username or password.")

    user_info = {
        "id":         row["id"],
        "username":   row["username"],
        "role":       row["role"],
        "created_at": row["created_at"],
    }

    # Persist in session_state
    st.session_state[_KEY_USER]  = user_info
    st.session_state[_KEY_TOKEN] = str(uuid.uuid4())

    return user_info


# ── Auth guard ────────────────────────────────────────────────────────────────

def require_auth(role: str | None = None) -> dict:
    """
    Return the current user or raise RuntimeError if not authenticated.

    Args:
        role: If provided, also assert the user has this specific role.

    Returns:
        Current user dict.

    Raises:
        RuntimeError: if not logged in, or if role doesn't match.
    """
    user = get_current_user()
    if user is None:
        raise RuntimeError("Not authenticated.")
    if role is not None and user.get("role") != role:
        raise RuntimeError(f"Access denied: requires role '{role}'.")
    return user


# ── Seed admin utility ────────────────────────────────────────────────────────

def ensure_default_admin(
    username: str = "admin",
    password: str = "admin123",
) -> None:
    """
    Create a default admin account if no admin exists yet.

    Called once at app startup so the system is accessible on first run.
    In production, change the default password immediately after first login.

    Args:
        username: Default admin username (default: "admin").
        password: Default admin password (default: "admin123").
    """
    conn = get_connection()
    try:
        exists = conn.execute(
            "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if exists is None:
        try:
            register_user(username, password, role=ROLE_ADMIN)
        except ValueError:
            pass   # already exists — ignore
