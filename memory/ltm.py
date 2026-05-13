"""Mem0-backed long-term memory (Qdrant vectors + Neo4j graph)."""

import asyncio
from config import ltm_memory


async def search_ltm(query: str, user_id: str, limit: int = 5) -> str:
    """Return relevant facts as a newline-separated string, or empty string."""
    results = await asyncio.to_thread(
        ltm_memory.search, query, user_id=user_id, limit=limit
    )
    items = (results or {}).get("results", [])
    if not items:
        return ""
    return "\n".join(f"- {r['memory']}" for r in items)


async def upsert_ltm(user_input: str, assistant_output: str, user_id: str) -> None:
    """Extract and upsert facts from a conversation turn (runs in background)."""
    text = f"User: {user_input}\nAssistant: {assistant_output}"
    await asyncio.to_thread(ltm_memory.add, text, user_id=user_id)
