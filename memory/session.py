"""Redis-backed multi-session history (short-term memory)."""

import json
import uuid
from datetime import datetime

import redis.asyncio as aioredis
from config import REDIS_HOST, REDIS_PORT, MAX_SESSION_MESSAGES

_redis: aioredis.Redis | None = None

# TTLs
_HISTORY_TTL = 30 * 86400   # 30 days — individual session history
_META_TTL    = 90 * 86400   # 90 days — sessions list + current pointer
MAX_SESSIONS = 20            # max sessions kept per user

# Docs
_DOCS_TTL = 30 * 86400
_MAX_DOCS  = 10


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}",
            decode_responses=True,
        )
    return _redis


def _new_sid() -> str:
    return f"s_{uuid.uuid4().hex[:12]}"


# ── Session ID management ──────────────────────────────────────────────────────

async def get_current_session_id(user_id: str) -> str:
    """Return active session_id; auto-creates one if the user has none."""
    r   = await _get_redis()
    sid = await r.get(f"current_session:{user_id}")
    if not sid:
        sid = await create_session(user_id)
    return sid


async def create_session(user_id: str, title: str = "New Session") -> str:
    """
    Create a new session, set it as the active session, add to sessions list.
    Returns the new session_id.
    """
    r   = await _get_redis()
    sid = _new_sid()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    meta     = {"id": sid, "title": title, "created_at": now, "message_count": 0}
    sessions = await get_sessions_list(user_id)
    sessions.insert(0, meta)

    # Drop oldest sessions beyond the cap
    if len(sessions) > MAX_SESSIONS:
        for old in sessions[MAX_SESSIONS:]:
            await r.delete(f"history:{user_id}:{old['id']}")
        sessions = sessions[:MAX_SESSIONS]

    await r.setex(f"sessions:{user_id}",        _META_TTL, json.dumps(sessions))
    await r.setex(f"current_session:{user_id}", _META_TTL, sid)
    return sid


async def switch_session(user_id: str, session_id: str) -> bool:
    """
    Make session_id the active session.
    Returns True if the session exists, False otherwise.
    """
    sessions = await get_sessions_list(user_id)
    if not any(s["id"] == session_id for s in sessions):
        return False
    r = await _get_redis()
    await r.setex(f"current_session:{user_id}", _META_TTL, session_id)
    return True


async def get_sessions_list(user_id: str) -> list[dict]:
    """Return all session metadata for a user, newest first."""
    r    = await _get_redis()
    data = await r.get(f"sessions:{user_id}")
    return json.loads(data) if data else []


async def get_current_session_meta(user_id: str) -> dict | None:
    """Return metadata dict for the currently active session."""
    sid      = await get_current_session_id(user_id)
    sessions = await get_sessions_list(user_id)
    return next((s for s in sessions if s["id"] == sid), None)


# ── History access (public API used by agent.py — signatures unchanged) ────────

async def get_session(user_id: str) -> list[dict]:
    """Return message history for the active session."""
    sid  = await get_current_session_id(user_id)
    r    = await _get_redis()
    data = await r.get(f"history:{user_id}:{sid}")
    return json.loads(data) if data else []


async def save_turn(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """Append a turn to the active session and update session metadata."""
    sid = await get_current_session_id(user_id)
    r   = await _get_redis()

    key     = f"history:{user_id}:{sid}"
    data    = await r.get(key)
    history = json.loads(data) if data else []
    history.append({"role": "user",      "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    if len(history) > MAX_SESSION_MESSAGES:
        history = history[-MAX_SESSION_MESSAGES:]
    await r.setex(key, _HISTORY_TTL, json.dumps(history))

    # Update title from first user message + message count
    sessions = await get_sessions_list(user_id)
    for s in sessions:
        if s["id"] == sid:
            s["message_count"] = len(history) // 2
            if s.get("title") in ("New Session", "") and user_msg.strip():
                s["title"] = user_msg.strip()[:40]
            break
    await r.setex(f"sessions:{user_id}", _META_TTL, json.dumps(sessions))


async def get_session_history(user_id: str, session_id: str) -> list[dict]:
    """Return message history for a specific session by ID."""
    r    = await _get_redis()
    data = await r.get(f"history:{user_id}:{session_id}")
    return json.loads(data) if data else []


async def clear_session(user_id: str) -> None:
    """Delete history of the active session (session stays in the list)."""
    sid = await get_current_session_id(user_id)
    r   = await _get_redis()
    await r.delete(f"history:{user_id}:{sid}")

    sessions = await get_sessions_list(user_id)
    for s in sessions:
        if s["id"] == sid:
            s["message_count"] = 0
            break
    await r.setex(f"sessions:{user_id}", _META_TTL, json.dumps(sessions))


# ── Per-user uploaded document tracking (unchanged) ───────────────────────────

async def get_user_docs(user_id: str) -> list[str]:
    """Return filenames uploaded by this user, most recent first."""
    r     = await _get_redis()
    items = await r.lrange(f"docs:{user_id}", 0, _MAX_DOCS - 1)
    return items


async def add_user_doc(user_id: str, filename: str) -> None:
    """Prepend filename and cap list at _MAX_DOCS."""
    r   = await _get_redis()
    key = f"docs:{user_id}"
    await r.lpush(key, filename)
    await r.ltrim(key, 0, _MAX_DOCS - 1)
    await r.expire(key, _DOCS_TTL)
