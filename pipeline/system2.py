"""
System 2 — Agentic Reasoning Mode (Deepthink, default ON).

Pipeline:
  1. Initial Strategy Formulation
  2. Generate Thought Signature  (structured JSON plan)
  3. Self-Critique & Plan loop   (up to MAX_CRITIQUE_ROUNDS)
       ├─ "needs_data" → Proactive Tool Call → MCP Sandbox → New Evidence → loop
       └─ "validated"  → Final Synthesis
  4. Final Synthesis → clean user-facing answer

Key fix: if the Thought Signature already has confidence ≥ 8 AND needs_search=false,
the critique loop is skipped entirely — avoids wasteful web searches for simple queries.
"""

import asyncio
import json
import re
import time
import openai
from fastmcp import FastMCP, Client
from config import llm_client
import ui

MAX_CRITIQUE_ROUNDS = 3

# ── Prompts ────────────────────────────────────────────────────────────────────

_STRATEGY_SYSTEM = """\
You are a strategic reasoning assistant.
Think carefully before answering. Be accurate and identify what you know vs what needs research."""

_THOUGHT_SIG_PROMPT = """\
Analyze this query and produce a Thought Signature as valid JSON.

Guidelines for needs_search:
- false  → greetings, math, general knowledge, definitions, opinions, creative writing, current date/time (already provided in system prompt)
- true   → latest news, live prices, recent events after your training cutoff, specific facts you are genuinely unsure about

Return ONLY valid JSON (no markdown fences, no extra text):
{{
  "goal": "<concise goal in one sentence>",
  "approach": "<how you will answer>",
  "confidence": <integer 0-10>,
  "needs_search": <true or false>,
  "data_needed": ["<specific gap>"] or []
}}

Query: {query}"""

_CRITIQUE_PROMPT = """\
Evaluate whether you have enough information to answer the question well.

Rules:
- Only set verdict "needs_data" if you CANNOT answer without external lookup.
- For greetings, math, general knowledge → verdict "validated".
- For current events, bank/financial data, schedules, specific facts → verdict "needs_data".
- Prefer specific domain tools (Deposit, Credit, Pension, Card, Admin, RealTime) over web_search when the question is about bank products or services.

Available tools:
{tools_list}

GOAL: {goal}
APPROACH: {approach}
EVIDENCE SO FAR:
{evidence}

Return ONLY valid JSON (no markdown fences):
{{
  "confidence": <integer 0-10>,
  "verdict": "validated" or "needs_data",
  "missing": "<what is still needed, or empty string>",
  "tool": "<exact tool name from the list above>",
  "tool_args": {{<argument key-value pairs matching the tool's parameters, or {{"query": "..."}} for web/rag search>}},
  "search_query": "<for web_search/rag_search only, the search query string>"
}}"""

_SYNTHESIS_SYSTEM = """\
You are a helpful assistant producing a final answer.
Use the provided evidence and reasoning. Be clear, concise, and accurate."""

_SYNTHESIS_PROMPT = """\
Original question: {question}

Reasoning trace:
{reasoning_trace}

Evidence gathered:
{evidence}

Write the final response for the user:"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_reasoning(msg: dict) -> str:
    return msg.get("reasoning_content") or (msg.get("model_extra") or {}).get(
        "reasoning_content", ""
    ) or ""


def _parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


async def _llm(messages: list[dict], model: str, retries: int = 3) -> tuple[str, str, int, int]:
    for attempt in range(retries + 1):
        try:
            resp  = await llm_client.chat.completions.create(model=model, messages=messages)
            msg   = resp.choices[0].message.model_dump()
            usage = resp.usage
            ptok  = usage.prompt_tokens     if usage else 0
            ctok  = usage.completion_tokens if usage else 0
            return msg.get("content") or "", _extract_reasoning(msg), ptok, ctok
        except openai.InternalServerError:
            if attempt < retries:
                wait = 4.0 * (2 ** attempt)
                ui.warn(f"Xazna API unavailable — retry {attempt + 1}/{retries} in {wait:.0f}s…")
                await asyncio.sleep(wait)
            else:
                raise


async def _call_tool(mcp: FastMCP, tool_name: str, args: dict) -> str:
    try:
        result = await mcp.call_tool(tool_name, args)
        return "".join(
            item.text if item.type == "text" else f"[{item.type}]"
            for item in result.content
        )
    except Exception as e:
        return f"Tool error: {e}"


def _build_tools_list(schemas: list[dict]) -> str:
    """Format tool schemas for the critique prompt."""
    lines = []
    for s in schemas:
        fn = s["function"]
        props = fn.get("parameters", {}).get("properties", {})
        params = ", ".join(props.keys()) if props else "no params"
        lines.append(f"  - {fn['name']}: {fn['description']} (args: {params})")
    return "\n".join(lines)


async def _get_tool_schemas(mcp: FastMCP) -> list[dict]:
    async with Client(mcp) as c:
        tools = await c.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


async def _llm_stream_synthesis(
    model: str, messages: list, tools: list, on_chunk,
) -> tuple[dict, None]:
    """Streaming LLM call for synthesis. Invokes on_chunk for text tokens; silently
    accumulates tool calls. Falls back to non-streaming if streaming fails."""
    try:
        stream = await llm_client.chat.completions.create(
            model=model, messages=messages, tools=tools or None, stream=True,
        )
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        has_tool_calls = False

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                has_tool_calls = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc.function.arguments

            if delta.content:
                content_parts.append(delta.content)
                if not has_tool_calls:
                    await on_chunk(delta.content)

        content = "".join(content_parts)
        tool_calls = (
            [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            if tool_calls_acc else None
        )
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}, None

    except Exception:
        # Fall back to non-streaming
        resp = await llm_client.chat.completions.create(
            model=model, messages=messages, tools=tools or None,
        )
        msg = resp.choices[0].message.model_dump()
        if msg.get("content") and not msg.get("tool_calls"):
            await on_chunk(msg["content"])
        return msg, resp.usage


async def _llm_with_tools(
    messages: list[dict],
    mcp: FastMCP,
    model: str,
    max_rounds: int = 4,
    retries: int = 3,
    stream_callback=None,
) -> tuple[str, str]:
    """LLM call with full tool-execution loop (mirrors System 1). Returns (content, reasoning)."""
    tools = await _get_tool_schemas(mcp)
    ui.tools_list(tools)
    conversation = list(messages)
    last_reasoning = ""

    for rnd in range(max_rounds):
        # ── LLM call with retry ────────────────────────────────────────────────
        msg = None
        usage = None
        if stream_callback:
            msg, usage = await _llm_stream_synthesis(model, conversation, tools, stream_callback)
        else:
            for attempt in range(retries + 1):
                try:
                    resp  = await llm_client.chat.completions.create(
                        model=model, messages=conversation, tools=tools,
                    )
                    msg   = resp.choices[0].message.model_dump()
                    usage = resp.usage
                    break
                except openai.InternalServerError:
                    if attempt < retries:
                        wait = 4.0 * (2 ** attempt)
                        ui.warn(f"Xazna API unavailable — retry {attempt + 1}/{retries} in {wait:.0f}s…")
                        await asyncio.sleep(wait)
                    else:
                        raise

        if usage:
            ui.token_usage(f"Phase 4 Synthesis round {rnd+1}", usage.prompt_tokens, usage.completion_tokens)

        reasoning = _extract_reasoning(msg)
        if reasoning:
            last_reasoning = reasoning
            ui.reasoning_block(reasoning, f"Synthesis Round {rnd + 1}")

        conversation.append(msg)

        if not msg.get("tool_calls"):
            ui.no_tools_used()
            return msg.get("content") or "", last_reasoning

        # ── Execute tool calls ─────────────────────────────────────────────────
        ui.stage(f"Synthesis round {rnd + 1} — tool calls: {len(msg['tool_calls'])}")
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            ui.tool_call(name, json.dumps(args, ensure_ascii=False))

            t_tool = time.perf_counter()
            result = await _call_tool(mcp, name, args)
            ui.timing(name, time.perf_counter() - t_tool)
            ui.tool_result(result)

            conversation.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": result,
            })

    for msg in reversed(conversation):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"], last_reasoning
    return "I was unable to generate a response.", last_reasoning

# ── Main entry point ────────────────────────────────────────────────────────────

async def run(
    messages: list[dict],
    mcp: FastMCP,
    model: str,
    stream_callback=None,
) -> tuple[str, str]:
    """
    Returns (final_response_text, evidence_string).
    """
    user_query = messages[-1]["content"]
    reasoning_trace: list[str] = []
    evidence_pieces: list[str] = []

    # ══ Phase 1: Initial Strategy Formulation ══════════════════════════════════
    ui.section("System 2 — Phase 1: Initial Strategy Formulation")
    t_phase12 = time.perf_counter()

    strategy_input = _THOUGHT_SIG_PROMPT.format(query=user_query)
    ui.llm_input("Strategy prompt", f"[system] {_STRATEGY_SYSTEM}\n[user] {strategy_input}")

    sig_content, sig_reasoning, sig_ptok, sig_ctok = await _llm(
        messages=[
            {"role": "system", "content": _STRATEGY_SYSTEM},
            *messages[:-1],
            {"role": "user", "content": strategy_input},
        ],
        model=model,
    )

    ui.timing("Phase 1 (Strategy LLM)", time.perf_counter() - t_phase12)
    ui.token_usage("Phase 1 Strategy", sig_ptok, sig_ctok)
    if sig_reasoning:
        ui.reasoning_block(sig_reasoning, "Strategy Internal Reasoning")

    # ══ Phase 2: Generate Thought Signature ════════════════════════════════════
    ui.section("System 2 — Phase 2: Thought Signature")
    ui.llm_output("Thought Signature (raw)", sig_content)

    sig = _parse_json(sig_content)
    goal        = sig.get("goal", user_query)
    approach    = sig.get("approach", "Direct reasoning")
    confidence  = int(sig.get("confidence", 5))
    needs_srch  = bool(sig.get("needs_search", bool(sig.get("data_needed"))))

    ui.kv("Goal",         goal)
    ui.kv("Approach",     approach)
    ui.kv("Confidence",   f"{confidence}/10")
    ui.kv("Needs search", "Yes" if needs_srch else "No")
    ui.kv("Data needed",  str(sig.get("data_needed") or "none"))

    reasoning_trace.append(f"Strategy: {sig_content}")

    # ══ Phase 3: Self-Critique & Plan loop ════════════════════════════════════
    # Build tool list once — used in critique prompt so LLM knows all available tools
    all_tool_schemas = await _get_tool_schemas(mcp)
    tools_list_str   = _build_tools_list(all_tool_schemas)

    if confidence >= 8 and not needs_srch:
        ui.section("System 2 — Phase 3: Self-Critique Loop")
        ui.ok("Skipped — confidence ≥ 8 and no search needed")
        ui.evidence_state(evidence_pieces)
    else:
        ui.section("System 2 — Phase 3: Self-Critique Loop")
        t_phase3 = time.perf_counter()

        for rnd in range(MAX_CRITIQUE_ROUNDS):
            evidence_str = "\n".join(evidence_pieces) or "None yet."
            ui.stage(f"Critique round {rnd + 1}")
            ui.evidence_state(evidence_pieces)

            critique_prompt = _CRITIQUE_PROMPT.format(
                goal=goal, approach=approach, evidence=evidence_str,
                tools_list=tools_list_str,
            )
            ui.llm_input(f"Critique round {rnd+1}", critique_prompt)

            t_crit = time.perf_counter()
            crit_content, crit_reasoning, crit_ptok, crit_ctok = await _llm(
                messages=[
                    {"role": "system", "content": "Evaluate the plan. Return valid JSON only."},
                    {"role": "user", "content": critique_prompt},
                ],
                model=model,
            )
            ui.timing(f"Critique round {rnd+1} (LLM)", time.perf_counter() - t_crit)
            ui.token_usage(f"Phase 3 Critique round {rnd+1}", crit_ptok, crit_ctok)
            ui.llm_output(f"Critique round {rnd+1} verdict", crit_content)

            if crit_reasoning:
                ui.reasoning_block(crit_reasoning, f"Critique Round {rnd+1}")

            crit         = _parse_json(crit_content)
            verdict      = crit.get("verdict", "validated")
            crit_conf    = int(crit.get("confidence", 7))
            search_query = crit.get("search_query", "").strip()
            mcp_tool     = crit.get("tool", "WebSearch_web_search").strip()
            tool_args    = crit.get("tool_args") or {}

            ui.kv("Verdict",    verdict)
            ui.kv("Confidence", f"{crit_conf}/10")
            ui.kv("Missing",    crit.get("missing", "") or "—")
            ui.kv("Tool",       mcp_tool)

            reasoning_trace.append(
                f"Critique round {rnd+1}: verdict={verdict} confidence={crit_conf}"
            )

            # If no data needed → validate and exit loop
            if verdict != "needs_data" or not mcp_tool:
                if crit_conf >= 7 or verdict == "validated":
                    ui.ok("Plan validated — moving to Final Synthesis")
                    break

            # ── Proactive Tool Call (MCP Sandbox) ──────────────────────────────
            # Use tool_args if provided by LLM; fall back to {"query": search_query}
            if not tool_args and search_query:
                tool_args = {"query": search_query}

            if tool_args:
                ui.stage("Proactive Tool Call → MCP Sandbox")
                ui.kv("Tool",  mcp_tool)
                ui.kv("Args",  json.dumps(tool_args, ensure_ascii=False)[:200])
                ui.tool_call(mcp_tool, json.dumps(tool_args, ensure_ascii=False))

                t_tool = time.perf_counter()
                result = await _call_tool(mcp, mcp_tool, tool_args)
                ui.timing(mcp_tool, time.perf_counter() - t_tool)
                ui.tool_result(result)

                evidence_pieces.append(f"[{mcp_tool} | {json.dumps(tool_args)}]:\n{result}")
                reasoning_trace.append(f"Evidence via {mcp_tool}: {result[:200]}")
                ev_toks       = len(result) // 4
                total_ev_toks = sum(len(p) // 4 for p in evidence_pieces)
                ui.stage("New Evidence accumulated")
                ui.evidence_state(evidence_pieces)
                ui.kv("Evidence tokens", f"≈{ev_toks:,} this result  |  ≈{total_ev_toks:,} total evidence")

        ui.timing("Phase 3 (Critique total)", time.perf_counter() - t_phase3)

    # ══ Phase 4: Final Synthesis ══════════════════════════════════════════════
    ui.section("System 2 — Phase 4: Final Synthesis")
    t_phase4 = time.perf_counter()

    evidence_combined = (
        "\n\n".join(evidence_pieces) if evidence_pieces else "No external data required."
    )
    trace_combined = "\n".join(reasoning_trace)

    synthesis_prompt = _SYNTHESIS_PROMPT.format(
        question=user_query,
        reasoning_trace=trace_combined,
        evidence=evidence_combined,
    )
    ui.llm_input("Synthesis prompt", synthesis_prompt)

    final_content, final_reasoning = await _llm_with_tools(
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            *messages[:-1],
            {"role": "user", "content": synthesis_prompt},
        ],
        mcp=mcp,
        model=model,
        stream_callback=stream_callback,
    )
    ui.timing("Phase 4 (Synthesis)", time.perf_counter() - t_phase4)

    if final_reasoning:
        ui.reasoning_block(final_reasoning, "Final Synthesis Reasoning")

    ui.llm_output("Final answer (pre-grounding)", final_content)

    return final_content, evidence_combined
