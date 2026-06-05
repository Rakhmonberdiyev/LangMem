"""
tools/langmem_mcp.py — LangMem Memory Tools for the Main LLM Orchestrator.

Four hot-path tools the LLM calls during reasoning:

  Memory_semantic_search(query)   → search Semantic Memory (user facts & preferences)
                                    Qdrant: langmem_semantic
  Memory_episodic_search(query)   → search Episodic Memory (past interaction examples)
                                    Qdrant: langmem_episodic
  Memory_save(content, namespace) → write to Semantic or Episodic tier
                                    Qdrant: langmem_semantic / langmem_episodic
  Memory_get_rules()              → read Procedural Memory (evolved system rules)
                                    Qdrant: langmem_procedural

The current user_id is bound per-request via set_langmem_context(),
called at the start of process_turn() in agent.py.
"""

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
async def Memory_semantic_search(query: str) -> str:
    """
    Search Semantic Memory for stored user facts, preferences, and personal details.
    Qdrant collection: langmem_semantic.
    Use when the user references personal information, preferences, or profile details
    from previous sessions (name, job, language, city, goals, etc.).
    """
    user_id = _ctx_user_id.get()
    if not user_id:
        return "Memory unavailable (no user context)."

    facts = await search_semantic(query, user_id, limit=5)
    if not facts:
        return "No relevant facts found in Semantic Memory."
    return "\n".join(f"- {f}" for f in facts)


@langmem_mcp.tool()
async def Memory_episodic_search(query: str) -> str:
    """
    Search Episodic Memory for past interaction examples relevant to this query.
    Qdrant collection: langmem_episodic.
    Use when you need to recall how a similar question was handled in a past session,
    find a relevant few-shot example, or understand the user's prior interaction patterns.
    """
    user_id = _ctx_user_id.get()
    if not user_id:
        return "Memory unavailable (no user context)."

    examples = await search_episodic(query, user_id, limit=3)
    if not examples:
        return "No relevant interaction examples found in Episodic Memory."
    return "\n".join(f"• {ex}" for ex in examples)


@langmem_mcp.tool()
async def Memory_save(content: str, namespace: str = "semantic") -> str:
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
async def Memory_get_rules() -> str:
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
