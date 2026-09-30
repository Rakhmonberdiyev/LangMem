"""
agent.py — Core pipeline entry point shared by Telegram bot, FastAPI web server, and CLI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL TURN PIPELINE  —  process_turn(user_input, user_id, deepthink)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 0 — CONTEXT VARIABLE BINDING
──────────────────────────────────
  set_langmem_context(user_id)   → tools/langmem_mcp.py  _ctx_user_id ContextVar
  set_rag_context(user_id)       → tools/mcp_server.py   _rag_user_id ContextVar

  Why: FastMCP tools run as plain functions with no request object. ContextVar
  is the only way to pass user_id into a synchronous MCP tool without changing
  every function signature. Both vars are set once per turn before any tool can run.

  Effect on tools:
    • Memory_search      → searches ns ("user", user_id, "semantic/episodic") only
    • Memory_save        → writes to ns ("user", user_id, "semantic/episodic") only
    • Memory_get_rules   → searches ns ("user", user_id, "procedural") only
    • RAG_rag_search     → Qdrant filter: FieldCondition(key="user_id", value=user_id)
  No cross-user data leakage is possible once both vars are set.


STEP 1 — PARALLEL CONTEXT LOAD  (asyncio.gather)
─────────────────────────────────────────────────
  Three sources loaded simultaneously to minimise latency:

  A) get_session(user_id)          → memory/session.py
     Redis key : history:{user_id}:{session_id}
     Returns   : list[{role, content}]  last ACTIVE_TURNS*2 messages (default 40)
     Used for  : conversation buffer injected into messages array

  B) get_user_docs(user_id)        → memory/session.py
     Redis key : docs:{user_id}  (Redis list, newest-first)
     Returns   : list[str]  filenames only (not content)
     Used for  : document filenames injected into system prompt so LLM knows
                 what to search via RAG_rag_search

  C) get_procedural_rules(user_id) → memory/langmem_store.py
     Qdrant    : langmem_procedural
     Namespace : ("user", user_id, "procedural")
     Query     : "system instructions rules"  limit=10
     Returns   : str — joined rule texts from all matching Qdrant points
     Used for  : injected into system prompt under "## Your learned instructions"
                 section so the LLM adapts its behaviour to this specific user


STEP 2 — MESSAGE ASSEMBLY  (build_messages)
────────────────────────────────────────────
  pipeline/context_ingestion.py builds the final messages array in this order:

  messages[0] = system prompt containing:
    ① HARD RULES 1-5
       Rule 1: bank questions → call bank tool FIRST, never answer from training data
       Rule 2: document questions → call RAG_rag_search FIRST
       Rule 3: personal/preference questions → call Memory_search FIRST
       Rule 4: only answer directly for pure greetings, math, completely new topics
       Rule 5: never state personal facts unless from current conversation or Memory_search

    ② Current date + time (UTC+5 Tashkent) — injected at call time so it is always fresh

    ③ Tool descriptions for all 36 MCP tools (listed with parameters)
       Sections: Memory / Document / Bank / Web

    ④ Procedural rules (from step 1C) — appended if non-empty:
       "## Your learned instructions for this user\n{rules}"
       These are rules the system learned over time: e.g. "this user always wants
       answers in Uzbek", "always ask for district before giving pension dates"

    ⑤ Uploaded doc filenames (from step 1B) — appended if non-empty:
       "[Documents uploaded by this user — searchable via RAG_rag_search]:\n  - file.pdf"
       Hint added: "when user says 'this file' they mean: {user_docs[0]}"

  messages[1..N] = session_history (from step 1A)
                   last ACTIVE_TURNS turn-pairs of {role, content} dicts

  messages[-1]   = {"role": "user", "content": user_input}  — current turn


STEP 3 — TERMINAL PANEL  (llm_input_panel)
──────────────────────────────────────────
  ui.py renders a structured terminal panel showing exactly what enters the LLM:
    • Base instructions (full text, untruncated)
    • Procedural memory (full text, untruncated)
    • Uploaded documents list
    • Conversation buffer (last 3 turn-pairs)
    • All 36 tools with source labels
    • Token breakdown (chars÷4 estimate per section)


STEP 4 — INPUT SAFETY GUARD  (check_input)
───────────────────────────────────────────
  pipeline/safety.py scans user_input against 8 compiled regex patterns:
    "ignore instructions" / "forget everything" / "you are now" / "jailbreak"
    "disregard training" / "new personality" / "override guidelines" / "act as"

  If matched → return blocked message immediately, pipeline stops here.
  If clean   → continue.


STEP 5 — ROUTING
─────────────────
  deepthink parameter OR _DEEPTHINK_RE auto-detection in user_input text.
  Patterns: deepthink / deep think / think carefully / chuqur o'yla /
            batafsil / yaxshilab / detailed analysis / reason through / etc.

  False (default) → System 1  (fast direct tool loop)
  True / matched  → System 2  (strategy → critique → synthesis)


STEP 6A — SYSTEM 1 (fast path)   pipeline/system1.py
──────────────────────────────────────────────────────
  Input  : messages array from step 2 + all 36 tool schemas from main_mcp
  Loop   : max 6 rounds

  Each round:
    1. LLM call → chat.completions.create(model, messages, tools)
       Streaming: _llm_call_stream() sends chunks via stream_callback(chunk)
       Non-stream: _llm_call()
       Retry: ×3  exponential backoff 4s / 8s / 16s on InternalServerError

    2. If response has tool_calls:
       For each tool_call:
         a. Parse arguments: json.loads(tc["function"]["arguments"])
         b. Execute: mcp.call_tool(name, args)  — dispatched to correct namespace:
              Memory_search      → langmem_mcp.search()         → Qdrant search
              Memory_save        → langmem_mcp.save()           → Qdrant upsert
              Memory_get_rules   → langmem_mcp.get_rules()      → Qdrant search
              RAG_rag_search     → mcp_server.rag_search()      → Qdrant query
              WebSearch_web_search → mcp_server.web_search()    → DuckDuckGo
              Deposit_*/Credit_*/Pension_*/Card_*/Admin_*/RealTime_*
                                 → remote_mcp proxy             → Nginx :8080
         c. Append {role:"tool", tool_call_id, name, content} to conversation
         d. Loop back to step 1 with updated conversation

    3. If no tool_calls → return (response_text, appended_messages)

  WHEN MEMORY TOOLS ARE CALLED IN SYSTEM 1:
    Memory_search(query):
      → asyncio.gather(search_semantic, search_episodic) in parallel
      → Qdrant: vector search in langmem_semantic + langmem_episodic
      → Namespace filter: ns = ["user", user_id, "semantic/episodic"]
      → Returns "[Semantic Memory — user facts]\n- fact1\n[Episodic Memory]\n• ex1"
      → LLM sees this as tool result and incorporates into answer

    Memory_save(content, namespace):
      → store.put(("user", user_id, namespace), key, {"kind":"Memory","content":{...}})
      → Qdrant sync upsert: embed content → store in langmem_semantic or langmem_episodic
      → Called when LLM detects a fact worth remembering across sessions

    Memory_get_rules():
      → store.asearch(("user", user_id, "procedural"), query="system instructions rules")
      → Returns current evolved rules for this user


STEP 6B — SYSTEM 2 (deepthink)   pipeline/system2.py
───────────────────────────────────────────────────────
  Input: same messages array + main_mcp

  Phase 1 — Strategy Formulation
    LLM call with _STRATEGY_SYSTEM + _THOUGHT_SIG_PROMPT
    Input: user query
    Output: JSON Thought Signature {goal, approach, confidence 0-10, needs_search, data_needed}

  Phase 2 — Thought Signature parsing
    confidence ≥ 8 AND needs_search=false → SKIP Phase 3 entirely (no tool calls)
    Otherwise → enter Phase 3 critique loop

  Phase 3 — Self-Critique Loop (max 3 rounds)
    Each round:
      a. LLM call with _CRITIQUE_PROMPT (goal + approach + evidence so far + tools list)
         Output: JSON {verdict, confidence, tool, tool_args, search_query}
      b. If verdict = "needs_data":
           → mcp.call_tool(tool, tool_args)  (proactive tool call — same dispatch as System 1)
           → evidence_pieces.append(result)
           → loop
      c. If verdict = "validated" or confidence ≥ 7 → exit loop

  Phase 4 — Final Synthesis
    _llm_with_tools() — full tool loop (max 4 rounds) + streaming
    Input: _SYNTHESIS_SYSTEM + original question + reasoning_trace + evidence_combined
    LLM can still call tools during synthesis if needed
    Output: (final_content, evidence_combined)


STEP 7 — GROUNDING & HALLUCINATION FILTER   pipeline/output_processor.py
──────────────────────────────────────────────────────────────────────────
  Only runs if evidence was gathered in System 2 Phase 3 (evidence != empty).
  System 1 fast-path always skips this (no evidence collected).

  Input : response text + evidence_combined string (capped at 3000 chars)
  LLM call with _GROUNDING_SYSTEM:
    Compare response against evidence.
    If a claim is not supported → append "(unverified)" after that claim.
    If well-grounded → return unchanged.
  Output: response text, possibly with "(unverified)" caveats added.


STEP 8 — OUTPUT SAFETY GUARD  (check_output)
─────────────────────────────────────────────
  Blocklist: "i am now jailbroken" / "i have no restrictions" / "my new instructions are"
  If matched → redact_unsafe_output() returns generic safe fallback.
  If clean   → continue.


STEP 9 — BACKGROUND PERSIST  (_persist — asyncio.create_task)
──────────────────────────────────────────────────────────────
  Runs concurrently after response is returned to the caller.
  Caller does NOT wait for this — it fires and continues.

  9A — Redis save_turn(user_id, user_input, response)
       Key    : history:{user_id}:{session_id}
       Action : append 2 messages → trim to MAX_SESSION_MESSAGES (40) → r.set()
       Also   : update sessions:{user_id} → title (from first user message) + message_count

  9B — LangMem background_update(full_convo, user_id)
       full_convo = session_hist + [user turn] + [assistant turn]

       SEMANTIC MANAGER (every turn):
         _semantic_manager.ainvoke({"messages": lc_msgs}, config={"langgraph_user_id": user_id})
         → LLM reads full conversation
         → Extracts: name, city, job, language preferences, goals, ongoing projects,
           named people, organisations, persistent preferences/dislikes
         → Deduplicates and consolidates with existing facts
         → store.aput(("user", user_id, "semantic"), key, {"kind":"Memory","content":{...}})
         → Qdrant upsert: langmem_semantic collection

       EPISODIC MANAGER (only if len(lc_msgs) >= 6):
         _episodic_manager.ainvoke({"messages": lc_msgs}, config={"langgraph_user_id": user_id})
         → LLM looks for high-quality interaction examples where:
           • user asked complex domain question + received tool-grounded accurate answer
           • user gave explicit positive feedback
           • exchange demonstrates important reasoning pattern
         → Format: compact (question → key facts used → concise answer) triple
         → store.aput(("user", user_id, "episodic"), key, {"kind":"Memory","content":{...}})
         → Qdrant upsert: langmem_episodic collection

  9C — Procedural optimizer  optimize_procedural(convo[-20:], user_id, current_rules)
       ONLY runs when: turn_count >= 5 AND turn_count % 5 == 0
       (i.e. every 5th turn: turn 5, 10, 15, 20, …)

       turn_count = get_current_session_meta(user_id)["message_count"]

       _proc_optimizer.ainvoke({
           "trajectories": [AnnotatedTrajectory(messages=lc_msgs, feedback="positive")],
           "prompt": Prompt(name="system_rules", prompt=current_rules_or_defaults),
       })
       → LangMem metaprompt optimizer reflects on the last 20 conversation messages
       → Produces refined system instruction rules string
       → store.aput(
             ("user", user_id, "procedural"),
             "system_rules",
             {"rule": new_rules, "version": unix_timestamp}
         )
       → Qdrant upsert: langmem_procedural collection
       → These rules are loaded at the START of the NEXT turn (step 1C)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASTMCP TOOL DISPATCH — main_mcp  (36 tools mounted via namespaces)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  main_mcp.mount(search_mcp,   namespace="WebSearch")   → 1 tool
  main_mcp.mount(rag_mcp,      namespace="RAG")         → 1 tool
  main_mcp.mount(langmem_mcp,  namespace="Memory")      → 3 tools
  main_mcp.mount(deposit_mcp,  namespace="Deposit")     → 5 tools
  main_mcp.mount(credit_mcp,   namespace="Credit")      → 8 tools
  main_mcp.mount(pension_mcp,  namespace="Pension")     → 4 tools
  main_mcp.mount(card_mcp,     namespace="Card")        → 4 tools
  main_mcp.mount(admin_mcp,    namespace="Admin")       → 3 tools
  main_mcp.mount(realtime_mcp, namespace="RealTime")    → 3 tools

  WebSearch_web_search(query, max_results=5)
    → DuckDuckGo DDGS.text()  —  returns title + body + href per result

  RAG_rag_search(query, top_k=5)
    → embed query via /models/embedding (2048-dim)
    → Qdrant query_points(knowledge_base, vector, filter=user_id, limit=top_k)
    → returns [score + source + text] per chunk

  Memory_search(query)
    → asyncio.gather(search_semantic, search_episodic) in parallel
    → search_semantic: Qdrant asearch langmem_semantic  ns=(user,uid,semantic) limit=5
    → search_episodic: Qdrant asearch langmem_episodic  ns=(user,uid,episodic) limit=3
    → combined result with section headers

  Memory_save(content, namespace="semantic")
    → namespace="semantic" : store.put(("user",uid,"semantic"), key, value)
    → namespace="episodic" : store.put(("user",uid,"episodic"), key, value)
    → key = f"fact_{uuid4().hex[:12]}"  or  f"ep_{uuid4().hex[:12]}"
    → value = {"kind": "Memory", "content": {"content": content_str}}
    → Sync upsert: embed → Qdrant upsert

  Memory_get_rules()
    → store.asearch(("user",uid,"procedural"), query="system instructions rules", limit=10)
    → extracts value["rule"] from each result → joined string

  Deposit_* / Credit_* / Pension_* / Card_* / Admin_* / RealTime_*
    → FastMCP create_proxy(f"http://localhost:8080/{namespace}/mcp")
    → HTTP request forwarded to Nginx gateway :8080
    → Nginx routes to the actual bank microservice


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QDRANT COLLECTIONS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  knowledge_base      RAG chunks      payload: {text, source, user_id, chunk_idx}
                                      filtered by user_id at query time

  langmem_semantic    per-user facts  ns: ("user", uid, "semantic")
                                      written by: background_update + Memory_save
                                      read by: Memory_search

  langmem_episodic    per-user examples  ns: ("user", uid, "episodic")
                                      written by: background_update (≥6 msgs) + Memory_save
                                      read by: Memory_search

  langmem_procedural  per-user rules  ns: ("user", uid, "procedural")  key: "system_rules"
                                      value: {rule: str, version: unix_ts}
                                      written by: optimize_procedural (every 5 turns)
                                      read by: get_procedural_rules (step 1C, start of turn)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REDIS KEYS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  history:{uid}:{sid}       list[{role,content}]   max 40 messages   no TTL
  sessions:{uid}            list[{id,title,created_at,message_count}] max 20  no TTL
  current_session:{uid}     str — active session pointer              no TTL
  docs:{uid}                Redis list of filenames  max 10           no TTL
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
        # Pass only the new turn — passing full_convo re-processes old messages every
        # turn and creates duplicate episodic entries (new UUID key = new Qdrant point)
        await background_update(
            [{"role": "user", "content": user_input}, {"role": "assistant", "content": response}],
            user_id,
        )
        ui.console.print("[dim]  🧠 LangMem ✓[/dim]")
    except Exception as e:
        ui.warn(f"LangMem background_update error (non-fatal): {e}")

    # ── Procedural Memory optimizer — every 5 turns ────────────────────────────
    try:
        meta = await get_current_session_meta(user_id)
        turn_count = (meta or {}).get("message_count", 0)
        next_trigger = 5 * ((turn_count // 5) + 1) if turn_count % 5 != 0 else turn_count
        ui.console.print(
            f"[dim]  ⚙️  Procedural turn={turn_count}  "
            f"{'RUNNING NOW' if turn_count >= 5 and turn_count % 5 == 0 else f'next trigger @ turn {next_trigger}'}[/dim]"
        )
        if turn_count >= 5 and turn_count % 5 == 0:
            current_rules = await get_procedural_rules(user_id)
            full_convo = session_hist + [
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": response},
            ]
            await optimize_procedural(full_convo[-20:], user_id, current_rules)
            ui.console.print("[dim]  ⚙️  Procedural Memory updated[/dim]")
    except Exception as e:
        import traceback
        ui.warn(f"Procedural optimizer error:\n{traceback.format_exc()}")


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
