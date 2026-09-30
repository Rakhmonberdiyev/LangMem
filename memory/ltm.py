"""
memory/ltm.py — LangMem adapter (eval + legacy compatibility).

Provides upsert_ltm / search_ltm as a thin wrapper over
memory/langmem_store.py so existing eval scripts keep working.
"""

from memory.langmem_store import (
    background_update,
    save_semantic,
    save_episodic,
    search_semantic,
    search_episodic,
)


async def upsert_ltm(
    user_msg: str,
    asst_msg: str,
    user_id: str,
    full_history: list[dict] | None = None,
) -> None:
    """
    Store a conversation turn into LangMem long-term memory.

    Two-tier storage:
      1. Direct raw storage — always runs, no LLM, guarantees data is retrievable.
         Episodic: full exchange text (historical record, never deleted).
         Semantic:  assistant response only (current knowledge — may be overwritten by manager).
      2. background_update — LLM-based extraction for structured semantic facts on top.
    """
    # ── Tier 1: direct raw storage (guaranteed baseline) ────────────────────────
    raw_episode = f"User: {user_msg}  |  Assistant: {asst_msg}"
    save_episodic(user_id, raw_episode)
    save_semantic(user_id, asst_msg)

    # ── Tier 2: LLM-based enrichment ────────────────────────────────────────────
    # Pass only the NEW turn (last 2 messages) — not cumulative history.
    # Passing full_history causes the manager to re-extract already-seen facts
    # on every turn, creating duplicate entries with new UUID keys.
    await background_update(
        [{"role": "user", "content": user_msg}, {"role": "assistant", "content": asst_msg}],
        user_id,
    )


async def search_ltm(query: str, user_id: str, limit: int = 7) -> str:
    """
    Search both Semantic and Episodic memory for relevant facts.
    Semantic  → current state facts (what is true now)
    Episodic  → historical events   (what happened, when, before/after values)
    Returns a single formatted string (one fact / example per line).
    """
    # Give episodic equal weight — temporal reasoning needs historical events
    sem_limit = limit
    epi_limit = limit

    facts, examples = await _gather(query, user_id, sem_limit, epi_limit)

    lines: list[str] = []
    for f in facts:
        lines.append(f"[semantic] {f}")
    for e in examples:
        lines.append(f"[episodic] {e}")
    return "\n".join(lines)


async def _gather(query, user_id, sem_limit, epi_limit):
    import asyncio
    return await asyncio.gather(
        search_semantic(query, user_id, limit=sem_limit),
        search_episodic(query, user_id, limit=epi_limit),
    )
