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


async def _llm_call(model: str, messages: list, tools: list) -> tuple[dict, object]:
    for attempt in range(_RETRIES + 1):
        try:
            resp = await llm_client.chat.completions.create(
                model=model, messages=messages, tools=tools,
            )
            return resp.choices[0].message.model_dump(), resp.usage
        except openai.InternalServerError:
            if attempt < _RETRIES:
                wait = 4.0 * (2 ** attempt)
                ui.warn(f"Xazna API unavailable — retry {attempt + 1}/{_RETRIES} in {wait:.0f}s…")
                await asyncio.sleep(wait)
            else:
                raise


async def _llm_call_stream(
    model: str, messages: list, tools: list, on_chunk,
) -> tuple[dict, None]:
    """Streaming LLM call. Calls on_chunk(str) for text tokens only (skips tool calls).
    Falls back to non-streaming and a single on_chunk call if streaming fails."""
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
        # Streaming unsupported or failed — fall back to non-streaming
        msg, usage = await _llm_call(model, messages, tools)
        if msg.get("content") and not msg.get("tool_calls"):
            await on_chunk(msg["content"])
        return msg, usage


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
    stream_callback=None,
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
        if stream_callback:
            msg, usage = await _llm_call_stream(model, conversation, tools, stream_callback)
        else:
            msg, usage = await _llm_call(model, conversation, tools)
        ui.timing(f"LLM round {rnd+1}", time.perf_counter() - t_llm)
        if usage:
            ui.token_usage(f"round {rnd+1}", usage.prompt_tokens, usage.completion_tokens)

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
