"""
agent.py — Main entry point (CLI + shared core for Telegram bot).

Full pipeline per turn:

  User Input
      │
      ├─ [Redis]    load session history
      ├─ [Mem0]     search long-term memory        ← parallel
      ├─ [LangMem]  get_procedural_rules()         ← parallel
      │
      ▼
  llm_input_panel  (terminal: shows every source + token breakdown)
      │
      ▼
  Prompt Ingestion & Safety Guard
      │
      ├─ deepthink=False ──► System 1  (fast, direct tool-call loop)
      └─ deepthink=True  ──► System 2  (Strategy → Critique → Synthesis)
                                  │
      ┌───────────────────────────┘
      ▼
  Grounding & Hallucination Filter
      │
      ├─ Redis:    Save Turn             (background)
      ├─ Mem0:     Upsert Facts          (background)
      ├─ LangMem:  background_update()   (background — semantic + episodic)
      └─ LangMem:  optimize_procedural() (background — every 5 turns)
      │
      ▼
  Output Safety Guard → return response
"""

import asyncio
import re
import time
import config

from fastmcp import FastMCP

import ui
from memory.session import get_session, save_turn, get_user_docs, get_current_session_meta
from memory.langmem_store import (
    background_update,
    get_procedural_rules,
    optimize_procedural,
)
from tools.mcp_server import search_mcp, rag_mcp, set_rag_context
from tools.langmem_mcp import langmem_mcp, set_langmem_context
from tools.remote_mcp import (
    deposit_mcp, credit_mcp, pension_mcp,
    card_mcp, admin_mcp, realtime_mcp,
)
from fastmcp import Client
from pipeline.context_ingestion import build_messages
from pipeline import safety
from pipeline import system1, system2
from pipeline.output_processor import ground_and_filter

# ── Deepthink intent detection ────────────────────────────────────────────────
_DEEPTHINK_RE = re.compile(
    r"\b("
    r"deepthink|deep\s*think"
    r"|deep\s*research|deep\s*dive|research\s+this|research\s+thoroughly"
    r"|think\s+carefully|think\s+deeply|think\s+step.by.step"
    r"|reason\s+carefully|reason\s+through"
    r"|analyze\s+carefully|careful\s+analysis|thorough\s+analysis|detailed\s+analysis"
    r"|explain\s+in\s+detail|elaborate|in[\s-]depth"
    r")\b"
    r"|yaxshilab|batafsil|chuqur\s*o['']yla|chuqur\s*tahlil|sinchiklab",
    re.IGNORECASE,
)

def _wants_deepthink(text: str) -> bool:
    return bool(_DEEPTHINK_RE.search(text))


# ── Shared MCP server ──────────────────────────────────────────────────────────
main_mcp = FastMCP("Main")
main_mcp.mount(search_mcp,   namespace="WebSearch")
main_mcp.mount(rag_mcp,      namespace="RAG")
main_mcp.mount(langmem_mcp,  namespace="Memory")
main_mcp.mount(deposit_mcp,  namespace="Deposit")
main_mcp.mount(credit_mcp,   namespace="Credit")
main_mcp.mount(pension_mcp,  namespace="Pension")
main_mcp.mount(card_mcp,     namespace="Card")
main_mcp.mount(admin_mcp,    namespace="Admin")
main_mcp.mount(realtime_mcp, namespace="RealTime")


_FALLBACK_MODEL = "/models/gemma"

async def initialize() -> str:
    """Resolve model ID from the Xazna API. Falls back to hardcoded ID if API is down."""
    try:
        models = await config.llm_client.models.list()
        model_id = models.data[0].id
    except Exception as exc:
        model_id = _FALLBACK_MODEL
        ui.warn(f"Could not fetch model list ({exc}) — using fallback: {model_id}")
    config.MODEL_ID = model_id
    ui.console.print(
        f"\n[dim]Model:[/dim] [bold cyan]{model_id}[/bold cyan]"
        f"  [dim]│[/dim]  [bold cyan]Fast mode[/bold cyan] by default\n"
    )
    return model_id


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def process_turn(
    user_input: str,
    user_id: str,
    deepthink: bool = False,
    stream_callback=None,
    metadata: dict | None = None,
) -> str:
    """Process one user turn through the full pipeline. Returns assistant response."""
    model = config.MODEL_ID

    # Bind per-request context vars so MCP tools know which user they're serving
    set_langmem_context(user_id)
    set_rag_context(user_id)

    t_overall = time.perf_counter()

    ui.blank()
    ui.user_panel(user_input)

    # ── 1. Context Ingestion — all sources loaded in parallel ──────────────────
    session_hist, user_docs, proc_rules = await asyncio.gather(
        get_session(user_id),
        get_user_docs(user_id),
        get_procedural_rules(user_id),
    )

    messages = build_messages(
        user_input,
        session_hist,
        user_docs=user_docs,
        procedural_rules=proc_rules,
    )

    # ── 2. Terminal panel — ordered, no duplicates ─────────────────────────────
    try:
        async with Client(main_mcp) as _c:
            _raw_tools = await _c.list_tools()
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema or {},
                },
            }
            for t in _raw_tools
        ]
    except Exception:
        tool_schemas = []

    ui.llm_input_panel(
        user_id=user_id,
        user_input=user_input,
        messages=messages,
        session_hist=session_hist,
        user_docs=user_docs,
        ltm_facts="",
        procedural_rules=proc_rules,
        tools=tool_schemas,
    )

    # ── 3. Safety Guard (input) ────────────────────────────────────────────────
    is_safe, reason = safety.check_input(user_input)
    if not is_safe:
        ui.err(f"Input BLOCKED — {reason}")
        return "I'm sorry, I can't process that request."
    ui.ok("  Safety: input passed")

    # ── 4. Routing ─────────────────────────────────────────────────────────────
    evidence = ""
    forced = not deepthink and _wants_deepthink(user_input)
    if forced:
        ui.warn("Fast mode override — user requested deep thinking → System 2")

    t_system = time.perf_counter()
    if deepthink or forced:
        ui.stage("System 2", "Deepthink ON  (Strategy → Critique Loop → Synthesis)")
        response, evidence = await system2.run(messages, main_mcp, model, stream_callback=stream_callback)
        ui.total_time("System 2 total", time.perf_counter() - t_system)
        if metadata is not None:
            metadata["evidence"] = evidence
    else:
        ui.stage("System 1", "Fast Mode  (direct tool-call loop)")
        response, _ = await system1.run(messages, main_mcp, model, stream_callback=stream_callback)
        ui.total_time("System 1 total", time.perf_counter() - t_system)
        if metadata is not None:
            metadata["evidence"] = ""

    # ── 5. Grounding & Hallucination Filter ────────────────────────────────────
    t_ground = time.perf_counter()
    response = await ground_and_filter(response, evidence, model)
    ui.timing("Grounding filter", time.perf_counter() - t_ground)

    # ── 6. Output Safety Guard ─────────────────────────────────────────────────
    out_safe, out_reason = safety.check_output(response)
    if not out_safe:
        ui.err(f"Output BLOCKED — {out_reason}")
        response = safety.redact_unsafe_output(response)
    else:
        ui.ok("  Safety: output passed")

    # ── 7. Persist in background ───────────────────────────────────────────────
    resp_toks = len(response) // 4
    ui.console.print(
        f"  [dim]✓  Persisting → Redis + LangMem  [{resp_toks} tok][/dim]"
    )
    asyncio.create_task(_persist(user_input, response, user_id, session_hist, model))

    ui.blank()
    ui.response_panel(response)
    ui.total_time("Overall pipeline", time.perf_counter() - t_overall)
    ui.blank()

    return response


async def _persist(
    user_input: str,
    response: str,
    user_id: str,
    session_hist: list[dict],
    model: str,
) -> None:
    # ── Redis ──────────────────────────────────────────────────────────────────
    try:
        await save_turn(user_id, user_input, response)
        ui.console.print("[dim]  💾 Recall ✓[/dim]")
    except Exception as e:
        ui.warn(f"Persist error: {e}")

    # ── LangMem background consolidation ──────────────────────────────────────
    try:
        full_convo = session_hist + [
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": response},
        ]
        await background_update(full_convo, user_id)
        ui.console.print("[dim]  🧠 LangMem ✓[/dim]")
    except Exception as e:
        ui.warn(f"LangMem background_update error (non-fatal): {e}")

    # ── Procedural Memory optimizer — every 5 turns ────────────────────────────
    try:
        meta = await get_current_session_meta(user_id)
        turn_count = (meta or {}).get("message_count", 0)
        if turn_count >= 5 and turn_count % 5 == 0:
            current_rules = await get_procedural_rules(user_id)
            full_convo = session_hist + [
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": response},
            ]
            await optimize_procedural(full_convo[-20:], user_id, current_rules)
            ui.console.print("[dim]  ⚙️  Procedural Memory updated[/dim]")
    except Exception as e:
        ui.warn(f"Procedural optimizer error (non-fatal): {e}")


# ── CLI loop ───────────────────────────────────────────────────────────────────

async def main() -> None:
    from rich.panel import Panel
    ui.console.print(Panel(
        "[bold]AI Agent[/bold] — architecture: Context → Safety → System1/System2 → Grounding → Output\n"
        "Commands: [cyan]/deepthink on[/cyan] | [cyan]/deepthink off[/cyan] | [cyan]exit[/cyan]",
        border_style="cyan",
        padding=(0, 2),
    ))

    await initialize()

    USER_ID   = "user_001"
    deepthink = False

    while True:
        try:
            raw = await asyncio.to_thread(input, "\n[You] → ")
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[dim]Goodbye.[/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue
        if raw.lower() == "exit":
            ui.console.print("[dim]Goodbye.[/dim]")
            break
        if raw.lower() == "/deepthink off":
            deepthink = False
            ui.warn("Deepthink OFF — using System 1 (fast)")
            continue
        if raw.lower() == "/deepthink on":
            deepthink = True
            ui.ok("Deepthink ON — using System 2")
            continue

        try:
            await process_turn(raw, USER_ID, deepthink=deepthink)
        except Exception as e:
            ui.err(f"Pipeline error: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
