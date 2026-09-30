"""
tools/langmem_mcp.py — LangMem Memory Tools for the Main LLM Orchestrator.

Three hot-path tools — mounted under namespace "Memory" so final names are:

  Memory_search(query)          → search BOTH Semantic + Episodic Memory in parallel
                                    Qdrant: langmem_semantic + langmem_episodic
  Memory_save(content, namespace) → write to Semantic or Episodic tier
                                    Qdrant: langmem_semantic / langmem_episodic
  Memory_get_rules()              → read Procedural Memory (evolved system rules)
                                    Qdrant: langmem_procedural

The current user_id is bound per-request via set_langmem_context(),
called at the start of process_turn() in agent.py.
"""

import asyncio
import contextvars

from fastmcp import FastMCP

from memory.langmem_store import (
    get_procedural_rules,
    save_episodic,
    save_semantic,
    search_episodic,
    search_semantic,
)

_ctx_user_id = contextvars.ContextVar("langmem_user_id", default="")


def set_langmem_context(user_id: str) -> None:
    """Bind user_id for the current async task (called once per process_turn)."""
    _ctx_user_id.set(user_id)


langmem_mcp = FastMCP("LangMemMemory")


@langmem_mcp.tool()
async def search(query: str) -> str:
    """
    Search long-term memory for anything relevant to the query.
    Always searches BOTH tiers in parallel:
      - Semantic Memory (langmem_semantic): user facts, preferences, personal details
      - Episodic Memory (langmem_episodic): past interaction examples, how questions were handled
    Use whenever the user references anything from previous sessions.
    """
    user_id = _ctx_user_id.get()
    if not user_id:
        return "Memory unavailable (no user context)."

    facts, examples = await asyncio.gather(
        search_semantic(query, user_id, limit=5),
        search_episodic(query, user_id, limit=3),
    )

    parts: list[str] = []
    if facts:
        parts.append("[Semantic Memory — user facts]")
        parts.extend(f"- {f}" for f in facts)
    if examples:
        parts.append("[Episodic Memory — past interactions]")
        parts.extend(f"• {ex}" for ex in examples)
    if not parts:
        return "No relevant memory found."
    return "\n".join(parts)


@langmem_mcp.tool()
async def save(content: str, namespace: str = "semantic") -> str:
    """
    Save an important fact or interaction example to LangMem Memory.
    - namespace='semantic'  (default): user fact, preference, or personal detail
                                        → Qdrant langmem_semantic
    - namespace='episodic': high-quality interaction example (question → tool → answer)
                             → Qdrant langmem_episodic
    Call when the user shares something worth remembering across sessions.
    """
    user_id = _ctx_user_id.get()
    if not user_id:
        return "Memory unavailable — fact not stored."

    if namespace == "episodic":
        key = save_episodic(user_id, content)
        return f"Stored in Episodic Memory (key={key[:16]}): {content[:120]}"

    key = save_semantic(user_id, content)
    return f"Stored in Semantic Memory (key={key[:16]}): {content[:120]}"


@langmem_mcp.tool()
async def get_rules() -> str:
    """
    Read the current Procedural Memory — evolved system instruction rules.
    Qdrant collection: langmem_procedural.
    Use to check what custom instructions have been learned for this user over time.
    """
    user_id = _ctx_user_id.get()
    if not user_id:
        return "Procedural Memory unavailable (no user context)."

    rules = await get_procedural_rules(user_id)
    if not rules:
        return "No custom procedural rules stored yet."
    return f"[Active System Rules]\n{rules}"
