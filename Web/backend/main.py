"""FastAPI web server for the AI Agent Dashboard.

Shares the same Redis (STM) and Mem0 (LTM) instances as the Telegram bot.

Run from the project root:
    uvicorn Web.backend.main:app --reload --port 8000

Or from this directory:
    uvicorn main:app --reload --port 8000
"""

import sys
import os

# Add project root to import path so we can reuse agent, config, memory.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio
import json
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import ui
from agent import process_turn, initialize
from memory.session import (
    get_sessions_list,
    get_current_session_id,
    create_session,
    switch_session,
    get_session_history,
)

app = FastAPI(title="AI Agent Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_initialized = False


@app.on_event("startup")
async def _startup() -> None:
    global _initialized
    if not _initialized:
        await initialize()
        _initialized = True


# ── Session endpoints ──────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions(user_id: str):
    sessions = await get_sessions_list(user_id)
    current  = await get_current_session_id(user_id)
    return {"sessions": sessions, "current_session_id": current}


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str, user_id: str):
    history = await get_session_history(user_id, session_id)
    return {"session_id": session_id, "history": history}


@app.post("/api/sessions")
async def new_session(user_id: str):
    sid = await create_session(user_id)
    return {"session_id": sid}


@app.post("/api/sessions/{session_id}/switch")
async def switch(session_id: str, user_id: str):
    ok = await switch_session(user_id, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# ── SSE chat endpoint ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str | None = None
    deepthink: bool = True


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Process a chat turn and stream tokens via Server-Sent Events.

    Event types:
      {"type": "token",  "content": "..."}          — partial response chunk
      {"type": "done",   "response": "...",
                         "metadata": {...}}          — final full response + metadata
      {"type": "error",  "error": "..."}             — pipeline error
    """
    if req.session_id:
        await switch_session(req.user_id, req.session_id)

    queue: asyncio.Queue = asyncio.Queue()
    metadata: dict = {}
    tool_collector: dict = {"rag": [], "web": [], "tools": []}

    async def _stream_cb(chunk: str) -> None:
        await queue.put({"type": "token", "content": chunk})

    async def _run() -> None:
        ui.set_log_queue(queue)
        ui.set_tool_collector(tool_collector)
        try:
            response = await process_turn(
                req.message,
                req.user_id,
                req.deepthink,
                stream_callback=_stream_cb,
                metadata=metadata,
            )
            full_metadata = {
                **metadata,
                "rag_results": tool_collector["rag"],
                "web_results": tool_collector["web"],
                "tool_calls": tool_collector["tools"],
            }
            await queue.put({"type": "done", "response": response, "metadata": full_metadata})
        except Exception as exc:
            await queue.put({"type": "error", "error": str(exc)})

    async def _event_gen() -> AsyncIterator[str]:
        task = asyncio.create_task(_run())
        try:
            while True:
                item = await queue.get()
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item["type"] in ("done", "error"):
                    break
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
