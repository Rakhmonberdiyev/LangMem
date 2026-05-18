"""Mem0-backed long-term memory (Qdrant vectors + Neo4j graph)."""

import asyncio
import time
from config import ltm_memory, MEM0_CONFIG
import ui

_QDRANT_COLLECTION = MEM0_CONFIG["vector_store"]["config"]["collection_name"]


def _patch_sub_stores() -> tuple[dict, dict, callable]:
    """
    Temporarily monkey-patch Qdrant vector_store.search and Neo4j graph.search
    so we can capture per-store timing and raw results without modifying mem0.

    Returns (timings_dict, raw_dict, restore_fn).
    The caller must call restore_fn() after the search completes.
    """
    timings: dict[str, float] = {}
    raw:     dict[str, object] = {}
    originals: list[tuple] = []   # (obj, attr, original_fn)

    vs = getattr(ltm_memory, "vector_store", None) if ltm_memory else None
    if vs and hasattr(vs, "search"):
        orig_vs = vs.search
        originals.append((vs, "search", orig_vs))
        def _vs_search(*a, **kw):
            t = time.perf_counter()
            r = orig_vs(*a, **kw)
            timings["qdrant"] = time.perf_counter() - t
            raw["qdrant"] = r
            return r
        vs.search = _vs_search

    gs = getattr(ltm_memory, "graph", None) if ltm_memory else None
    if gs and hasattr(gs, "search"):
        orig_gs = gs.search
        originals.append((gs, "search", orig_gs))
        def _gs_search(*a, **kw):
            t = time.perf_counter()
            r = orig_gs(*a, **kw)
            timings["neo4j"] = time.perf_counter() - t
            raw["neo4j"] = r
            return r
        gs.search = _gs_search

    def restore():
        for obj, attr, orig in originals:
            setattr(obj, attr, orig)

    return timings, raw, restore


async def search_ltm(query: str, user_id: str, limit: int = 5) -> str:
    """Return relevant facts as a newline-separated string, or empty string."""
    if ltm_memory is None:
        ui.warn("Mem0 LTM not initialized (Qdrant was down at startup) — skipping LTM search")
        return ""

    ui.console.print()
    ui.console.print(
        f"      [bold white]Mem0 Search[/bold white]"
        f"  [dim]Qdrant:[/dim] [cyan]{_QDRANT_COLLECTION}[/cyan]"
        f"  [dim]Graph:[/dim] [magenta]Neo4j[/magenta]"
    )
    ui.kv("Query →", query[:200])
    ui.kv("User",    user_id)
    ui.kv("Limit",   str(limit))

    timings, _raw, restore = _patch_sub_stores()

    t0 = time.perf_counter()
    try:
        results = await asyncio.to_thread(
            ltm_memory.search, query, user_id=user_id, limit=limit
        )
    except Exception as exc:
        restore()
        ui.warn(f"Mem0 search failed (non-fatal): {exc}")
        ui.kv("LTM injected ≈", "0 tok — search error, skipped")
        return ""
    elapsed = time.perf_counter() - t0
    restore()

    # ── Sub-store timings ──────────────────────────────────────────────────────
    if "qdrant" in timings:
        ui.timing("Qdrant vector search", timings["qdrant"])
    if "neo4j" in timings:
        ui.timing("Neo4j graph search",   timings["neo4j"])
    ui.timing("Mem0 total",               elapsed)

    _MIN_SCORE = 0.5   # discard vector hits below this cosine similarity

    items = (results or {}).get("results", [])
    if not items:
        ui.kv("Results", "none found")
        return ""

    # Distinguish sources: vector results have a float `score`, graph results don't
    # Drop low-confidence vector hits so irrelevant facts don't pollute the prompt
    vector_items = [r for r in items if r.get("score") is not None and r["score"] >= _MIN_SCORE]
    dropped      = [r for r in items if r.get("score") is not None and r["score"] <  _MIN_SCORE]
    graph_items  = [r for r in items if r.get("score") is None]

    if dropped:
        ui.console.print(
            f"      [dim]Dropped {len(dropped)} low-score vector hit(s)"
            f" (score < {_MIN_SCORE}) — irrelevant to query[/dim]"
        )

    kept = len(vector_items) + len(graph_items)
    ui.kv("Results", f"{len(items)} raw → {kept} kept  (vector={len(vector_items)}  graph={len(graph_items)})")

    if vector_items:
        ui.console.print(f"      [bold cyan]── Qdrant / {_QDRANT_COLLECTION}[/bold cyan]")
        for i, r in enumerate(vector_items):
            mem   = r.get("memory", "")
            score = r.get("score", 0.0)
            tok   = len(mem) // 4
            ui.console.print(
                f"        [dim]{i + 1}.[/dim]"
                f"  [dim]collection=[/dim][cyan]{_QDRANT_COLLECTION}[/cyan]"
                f"  score=[cyan]{score:.3f}[/cyan]"
                f"  ≈[cyan]{tok}[/cyan] tok"
            )
            ui.console.print(f"          [dim]{mem[:300]}[/dim]")

    if graph_items:
        ui.console.print("      [bold magenta]── Neo4j graph[/bold magenta]")
        for i, r in enumerate(graph_items):
            mem = r.get("memory", "")
            tok = len(mem) // 4
            ui.console.print(
                f"        [dim]{i + 1}.[/dim]"
                f"  [dim]collection=[/dim][magenta]neo4j[/magenta]"
                f"  ≈[magenta]{tok}[/magenta] tok"
            )
            ui.console.print(f"          [dim]{mem[:300]}[/dim]")

    kept_items = vector_items + graph_items
    if not kept_items:
        ui.kv("LTM injected ≈", "0 tok — all hits below score threshold")
        return ""

    total_tok = sum(len(r.get("memory", "")) for r in kept_items) // 4
    ui.kv("LTM injected ≈", f"{total_tok} tok")

    return "\n".join(f"- {r['memory']}" for r in kept_items)


async def upsert_ltm(user_input: str, assistant_output: str, user_id: str) -> None:
    """Extract and upsert facts from a conversation turn (runs in background)."""
    if ltm_memory is None:
        return
    text = f"User: {user_input}\nAssistant: {assistant_output}"
    await asyncio.to_thread(ltm_memory.add, text, user_id=user_id)
