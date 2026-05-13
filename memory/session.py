"""Redis-backed session history (short-term memory)."""

import json
import redis.asyncio as aioredis
from config import REDIS_HOST, REDIS_PORT, SESSION_TTL, MAX_SESSION_MESSAGES

_redis: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}",
            decode_responses=True,
        )
    return _redis


async def get_session(user_id: str) -> list[dict]:
    """Return the stored message history for a user (may be empty)."""
    r = await _get_redis()
    data = await r.get(f"session:{user_id}")
    return json.loads(data) if data else []


async def save_turn(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """Append a user/assistant turn and persist with TTL."""
    r = await _get_redis()
    key = f"session:{user_id}"
    history = await get_session(user_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # Rolling window
    if len(history) > MAX_SESSION_MESSAGES:
        history = history[-MAX_SESSION_MESSAGES:]
    await r.setex(key, SESSION_TTL, json.dumps(history))


async def clear_session(user_id: str) -> None:
    r = await _get_redis()
    await r.delete(f"session:{user_id}")


# ── Per-user uploaded document tracking ───────────────────────────────────────
_DOCS_TTL = 30 * 86400   # 30 days
_MAX_DOCS  = 10


async def get_user_docs(user_id: str) -> list[str]:
    """Return filenames uploaded by this user, most recent first."""
    r = await _get_redis()
    items = await r.lrange(f"docs:{user_id}", 0, _MAX_DOCS - 1)
    return items  # already strings (decode_responses=True)


async def add_user_doc(user_id: str, filename: str) -> None:
    """Prepend a filename and keep the list capped at _MAX_DOCS."""
    r = await _get_redis()
    key = f"docs:{user_id}"
    await r.lpush(key, filename)
    await r.ltrim(key, 0, _MAX_DOCS - 1)
    await r.expire(key, _DOCS_TTL)
