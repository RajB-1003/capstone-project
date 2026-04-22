"""
auth.py — Authentication layer (Phase 1 + Session Persistence).

Provides user registration, login, and logout backed by SQLite (users table
in db.py).  Passwords are hashed with bcrypt.

Session Persistence (additive — no pipeline changes):
    On login:
        1. A UUID token is created in the session_tokens table (7-day TTL).
        2. The token is stored in st.session_state AND in st.query_params["token"]
           so it survives browser refreshes.
    On every rerun (init_session):
        1. st.query_params["token"] is read.
        2. validate_session_token() checks the DB — auto-login if valid.
        3. last_page is restored so the user returns to the page they were on.
    On logout:
        1. Token is deleted from DB.
        2. st.session_state is cleared.
        3. st.query_params is cleared (URL cleaned up).

Session-state keys
------------------
    auth_user   — dict { id, username, role, created_at } or None
    auth_token  — UUID token string or None

Public API (backward-compatible)
---------------------------------
    init_session()
    register_user(username, password, role="user")  -> dict | raises ValueError
    login_user(username, password)                  -> dict | raises ValueError
    logout_user()
    get_current_user()                              -> dict | None
    is_authenticated()                              -> bool
    is_admin()                                      -> bool
    require_auth(role=None)                         -> dict | raises RuntimeError
    ensure_default_admin()
"""

import uuid
import logging

import bcrypt
import streamlit as st

from modules.db import (
    get_connection,
    init_db,
    create_session_token,
    validate_session_token,
    delete_session_token,
    purge_expired_tokens,
)

logger = logging.getLogger(__name__)

# ── Role constants ─────────────────────────────────────────────────────────────

ROLE_USER  = "user"
ROLE_ADMIN = "admin"

_VALID_ROLES = {ROLE_USER, ROLE_ADMIN}

# ── Session keys ──────────────────────────────────────────────────────────────

_KEY_USER  = "auth_user"    # dict { id, username, role, created_at } once logged in
_KEY_TOKEN = "auth_token"   # UUID string used as a persistent session token


# ── Session helpers ───────────────────────────────────────────────────────────

def init_session() -> None:
    """
    Initialise authentication keys and attempt token-based auto-login.

    Called on every Streamlit rerun at the top of app.py.

    Steps:
        1. Ensure DB tables exist (idempotent).
        2. Purge expired tokens (housekeeping).
        3. Set default session-state keys if absent.
        4. If already authenticated in session_state → done (fast path).
        5. Otherwise, check st.query_params for a "token" value.
        6. Validate the token against the DB (expiry-aware).
        7. If valid, restore auth_user and active_page from the DB record.
    """
    init_db()            # idempotent — creates tables (including session_tokens)
    purge_expired_tokens()  # clean stale rows once per app start

    st.session_state.setdefault(_KEY_USER,  None)
    st.session_state.setdefault(_KEY_TOKEN, None)

    # Fast path: already authenticated in this server-side session
    if st.session_state[_KEY_USER] is not None:
        return

    # Attempt token-based restore from URL query param
    token = st.query_params.get("token", None)
    if not token:
        return

    user_row = validate_session_token(token)
    if user_row is None:
        # Token invalid or expired — clear the stale URL param
        try:
            del st.query_params["token"]
        except Exception:
            pass
        return

    # Valid token → restore session
    st.session_state[_KEY_USER]  = {
        "id":         user_row["id"],
        "username":   user_row["username"],
        "role":       user_row["role"],
        "created_at": user_row["created_at"],
    }
    st.session_state[_KEY_TOKEN] = token

    # Restore the last active page so refresh returns user to correct page
    last_page = user_row.get("last_page", "search")
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = last_page


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
    """
    Log the current user out.

    1. Delete the session token from the DB (invalidates the URL token).
    2. Clear session_state authentication keys.
    3. Remove the token from st.query_params (clean URL for next visitor).
    """
    token = st.session_state.get(_KEY_TOKEN)
    if token:
        try:
            delete_session_token(token)
        except Exception as exc:
            logger.warning("[auth] logout: token deletion failed: %s", exc)

    st.session_state[_KEY_USER]  = None
    st.session_state[_KEY_TOKEN] = None

    # Clear active page so next login starts fresh
    st.session_state["active_page"] = "search"

    # Clean up URL query params
    try:
        del st.query_params["token"]
    except Exception:
        pass


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
        role:     \"user\" (default) or \"admin\".

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
    Verify credentials, create a DB session token, and persist it in the URL.

    On success:
        - Writes auth_user and auth_token to st.session_state.
        - Creates a row in session_tokens (7-day TTL) in the DB.
        - Writes the token to st.query_params["token"] so it survives refresh.

    Args:
        username: Exact username (case-sensitive).
        password: Plain-text password to verify against stored hash.

    Returns:
        dict: { id, username, role, created_at }

    Raises:
        ValueError: if credentials are invalid.
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

    # Create a persistent DB token (7-day TTL)
    current_page = st.session_state.get("active_page", "search")
    token = create_session_token(user_info["id"], last_page=current_page)

    # Persist in session_state
    st.session_state[_KEY_USER]  = user_info
    st.session_state[_KEY_TOKEN] = token

    # Persist in URL so refresh restores the session
    st.query_params["token"] = token

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
        username: Default admin username (default: \"admin\").
        password: Default admin password (default: \"admin123\").
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
