"""
System 1 — Main LLM Orchestrator (fast path).

Standard agentic loop: LLM → tool calls → LLM → … → final response.
"""

import asyncio
import json
import time
import openai
from fastmcp import FastMCP, Client
from config import llm_client
import ui

_RETRIES = 3


async def _llm_call(model: str, messages: list, tools: list) -> dict:
    for attempt in range(_RETRIES + 1):
        try:
            resp = await llm_client.chat.completions.create(
                model=model, messages=messages, tools=tools,
            )
            return resp.choices[0].message.model_dump()
        except openai.InternalServerError:
            if attempt < _RETRIES:
                wait = 4.0 * (2 ** attempt)
                ui.warn(f"Xazna API unavailable — retry {attempt + 1}/{_RETRIES} in {wait:.0f}s…")
                await asyncio.sleep(wait)
            else:
                raise


def _extract_reasoning(msg: dict) -> str:
    return msg.get("reasoning_content") or (msg.get("model_extra") or {}).get(
        "reasoning_content", ""
    ) or ""


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


async def run(
    messages: list[dict],
    mcp: FastMCP,
    model: str,
    max_rounds: int = 6,
) -> tuple[str, list[dict]]:
    """
    Returns (final_response_text, new_messages_appended).
    """
    conversation = list(messages)
    appended: list[dict] = []
    tools = await _get_tool_schemas(mcp)

    ui.section("System 1 — Main LLM Orchestrator")
    ui.tools_list(tools)

    for rnd in range(max_rounds):
        t_llm = time.perf_counter()
        msg = await _llm_call(model, conversation, tools)
        ui.timing(f"LLM round {rnd+1}", time.perf_counter() - t_llm)

        reasoning = _extract_reasoning(msg)
        if reasoning:
            ui.reasoning_block(reasoning, f"Round {rnd+1} Reasoning")

        conversation.append(msg)
        appended.append(msg)

        if not msg.get("tool_calls"):
            ui.no_tools_used()
            ui.ok(f"Response generated (round {rnd+1})")
            return msg.get("content") or "", appended

        # Execute tool calls
        ui.stage(f"Round {rnd+1} — tool calls: {len(msg['tool_calls'])}")
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            ui.tool_call(name, json.dumps(args, ensure_ascii=False))

            t_tool = time.perf_counter()
            try:
                result = await mcp.call_tool(name, args)
                content = "".join(
                    item.text if item.type == "text" else f"[{item.type}]"
                    for item in result.content
                )
            except Exception as e:
                content = f"Tool error: {e}"
            ui.timing(name, time.perf_counter() - t_tool)

            ui.tool_result(content)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": content,
            }
            conversation.append(tool_msg)
            appended.append(tool_msg)

    # Fallback: return last text found
    for msg in reversed(appended):
        if msg.get("content"):
            return msg["content"], appended
    return "I was unable to generate a response.", appended
