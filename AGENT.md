# AGENT.md — Complete Project Blueprint

> Give this file to any AI and it will reproduce the entire project exactly, with no bugs.

---

## 1. What This Project Is

A **production-ready AI assistant** with a Telegram interface and a full terminal pipeline display.

**Core features:**
- Two reasoning modes: **System 1** (fast, direct tool loop) and **System 2** (Deepthink: Strategy → Self-Critique → Synthesis)
- **Long-term memory** (Mem0 = Qdrant vectors + Neo4j graph)
- **Short-term memory** (Redis session history, per-user)
- **RAG** — users upload PDFs/TXT/MD in Telegram → indexed into Qdrant → searchable
- **Web search** via DuckDuckGo
- **MCP tool server** (FastMCP) mounting both tools
- **Safety guards** on input and output
- **Grounding / hallucination filter** on every response
- **Rich terminal UI** — every pipeline step, timing, LLM I/O, tool calls, and results printed
- **Auto-escalation** — if user asks for deep analysis while in Fast mode, automatically routes to System 2 for that turn
- **Continuous Telegram typing indicator** — refreshed every 4s while processing

---

## 2. Tech Stack & Versions

| Component | Library | Notes |
|-----------|---------|-------|
| LLM | Xazna API (OpenAI-compatible) | `base_url=https://ai.xazna.uz/llm/v1` |
| LTM memory | `mem0` | Qdrant + Neo4j backend |
| STM memory | `redis.asyncio` | Session history, doc list |
| RAG vector DB | `qdrant-client >=1.17` | **Must use `.query_points()` not `.search()`** |
| Embeddings | OpenAI `text-embedding-3-small` | Used by Mem0 and RAG |
| LLM for Mem0 internals | OpenAI `gpt-4o-mini` | Separate from main LLM |
| MCP tool server | `fastmcp` | Two servers mounted with namespaces |
| Web search | `ddgs` (or `duckduckgo_search`) | Try `ddgs` first, fallback to old name |
| PDF extraction | `pypdf` | |
| Telegram bot | `aiogram v3` | `DefaultBotProperties(parse_mode="HTML")` |
| Terminal UI | `rich` | Console, Panel, Rule, Text, Table |
| Infrastructure | Docker Compose | Redis 7, Qdrant, Neo4j |

---

## 3. Project Structure

```
Mem0_full/
├── docker-compose.yml          # Redis + Qdrant + Neo4j
├── .env                        # Secret keys (see section 4)
├── config.py                   # All config, LLM client, Mem0 client
├── agent.py                    # Main pipeline, CLI loop
├── telegram_bot.py             # Telegram interface (aiogram v3)
├── ui.py                       # Rich terminal display helpers
├── memory/
│   ├── __init__.py
│   ├── session.py              # Redis STM: get/save/clear session + doc tracking
│   └── ltm.py                  # Mem0 LTM: search + upsert
├── pipeline/
│   ├── __init__.py
│   ├── context_ingestion.py    # Merges session + LTM + docs into messages[]
│   ├── system1.py              # Fast path: direct tool-call loop
│   ├── system2.py              # Deepthink: Strategy → Critique → Synthesis
│   ├── safety.py               # Input + output safety guards
│   └── output_processor.py     # Grounding & hallucination filter
└── tools/
    ├── __init__.py
    ├── mcp_server.py           # FastMCP: web_search + rag_search tools
    └── ingestion.py            # PDF/TXT/MD → chunk → embed → Qdrant
```

---

## 4. Environment Variables (`.env`)

```env
OPENAI_API_KEY=sk-...           # For Mem0 internals + embeddings + RAG
NEO4J_PASSWORD=password123      # Must match docker-compose NEO4J_AUTH
TELEGRAM_BOT_TOKEN=...          # BotFather token

# Optional (defaults shown):
# REDIS_HOST=localhost
# REDIS_PORT=6379
```

The Xazna LLM credentials are **hardcoded in `config.py`** (not in `.env`):
```python
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY  = "sk-raximberdi-cmF4aW1iZXJkaQ"
```

---

## 5. Infrastructure (`docker-compose.yml`)

```yaml
services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j_mem0
    ports: ["7474:7474", "7687:7687"]
    environment: [NEO4J_AUTH=neo4j/password123]
    volumes: [neo4j_data:/data]

  qdrant:
    image: qdrant/qdrant
    container_name: qdrant_mem0
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]

  redis:
    image: redis:7-alpine
    container_name: redis_session
    ports: ["6379:6379"]
    command: redis-server --appendonly yes
    volumes: [redis_data:/data]

volumes:
  neo4j_data:
  qdrant_data:
  redis_data:
```

Start with: `docker compose up -d`

---

## 6. Full Source Code

### `config.py`

```python
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mem0 import Memory

load_dotenv()

# --- LLM (Xazna) ---
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY  = "sk-raximberdi-cmF4aW1iZXJkaQ"

llm_client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# Resolved at startup via initialize() in agent.py
MODEL_ID: str = ""

# --- OpenAI (Mem0 internals + embeddings) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Mem0 (LTM): Qdrant vectors + Neo4j graph ---
MEM0_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small", "api_key": OPENAI_API_KEY},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333, "collection_name": "mem0_ltm"},
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD", "password123"),
        },
    },
}

ltm_memory = Memory.from_config(MEM0_CONFIG)

# --- Redis (session history) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SESSION_TTL          = 86400   # 24 hours
MAX_SESSION_MESSAGES = 40      # rolling window

# --- Qdrant RAG ---
QDRANT_HOST   = "localhost"
QDRANT_PORT   = 6333
RAG_COLLECTION = "knowledge_base"
EMBED_MODEL    = "text-embedding-3-small"
```

---

### `memory/session.py`

```python
"""Redis-backed session history (short-term memory)."""

import json
import redis.asyncio as aioredis
from config import REDIS_HOST, REDIS_PORT, SESSION_TTL, MAX_SESSION_MESSAGES

_redis: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}",
            decode_responses=True,
        )
    return _redis


async def get_session(user_id: str) -> list[dict]:
    r = await _get_redis()
    data = await r.get(f"session:{user_id}")
    return json.loads(data) if data else []


async def save_turn(user_id: str, user_msg: str, assistant_msg: str) -> None:
    r = await _get_redis()
    key = f"session:{user_id}"
    history = await get_session(user_id)
    history.append({"role": "user",      "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    if len(history) > MAX_SESSION_MESSAGES:
        history = history[-MAX_SESSION_MESSAGES:]
    await r.setex(key, SESSION_TTL, json.dumps(history))


async def clear_session(user_id: str) -> None:
    r = await _get_redis()
    await r.delete(f"session:{user_id}")


# ── Per-user uploaded document tracking ───────────────────────────────────────
_DOCS_TTL = 30 * 86400   # 30 days
_MAX_DOCS  = 10


async def get_user_docs(user_id: str) -> list[str]:
    """Return filenames uploaded by this user, most recent first."""
    r = await _get_redis()
    items = await r.lrange(f"docs:{user_id}", 0, _MAX_DOCS - 1)
    return items   # decode_responses=True → already str


async def add_user_doc(user_id: str, filename: str) -> None:
    """Prepend filename and cap list at _MAX_DOCS."""
    r = await _get_redis()
    key = f"docs:{user_id}"
    await r.lpush(key, filename)
    await r.ltrim(key, 0, _MAX_DOCS - 1)
    await r.expire(key, _DOCS_TTL)
```

---

### `memory/ltm.py`

```python
"""Mem0-backed long-term memory (Qdrant vectors + Neo4j graph)."""

import asyncio
from config import ltm_memory


async def search_ltm(query: str, user_id: str, limit: int = 5) -> str:
    """Return relevant facts as newline-separated string, or empty string."""
    results = await asyncio.to_thread(
        ltm_memory.search, query, user_id=user_id, limit=limit
    )
    items = (results or {}).get("results", [])
    if not items:
        return ""
    return "\n".join(f"- {r['memory']}" for r in items)


async def upsert_ltm(user_input: str, assistant_output: str, user_id: str) -> None:
    """Extract and upsert facts from a conversation turn (runs in background)."""
    text = f"User: {user_input}\nAssistant: {assistant_output}"
    await asyncio.to_thread(ltm_memory.add, text, user_id=user_id)
```

---

### `tools/mcp_server.py`

```python
"""MCP tools: web_search (DuckDuckGo) and rag_search (Qdrant knowledge_base)."""

from fastmcp import FastMCP
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT, RAG_COLLECTION, EMBED_MODEL

search_mcp = FastMCP("WebSearch")
rag_mcp    = FastMCP("RAG")

_embed_client = OpenAI(api_key=OPENAI_API_KEY)
_qdrant       = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _ensure_rag_collection() -> bool:
    try:
        existing = [c.name for c in _qdrant.get_collections().collections]
        if RAG_COLLECTION not in existing:
            _qdrant.create_collection(
                collection_name=RAG_COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        return True
    except Exception as e:
        print(f"[RAG] Qdrant collection init error: {e}")
        return False


@search_mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for real-time information using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return "No results found."
        lines = []
        for h in hits:
            lines.append(
                f"**{h.get('title', '')}**\n"
                f"{h.get('body', '')}\n"
                f"Source: {h.get('href', '')}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"Web search error: {e}"


@rag_mcp.tool()
def rag_search(query: str, top_k: int = 5) -> str:
    """Search documents and files the user has uploaded (PDFs, CVs, reports, text files).
    Use this when the user references an uploaded file or asks about document content."""
    if not _ensure_rag_collection():
        return "Knowledge base unavailable."
    try:
        resp   = _embed_client.embeddings.create(model=EMBED_MODEL, input=query)
        vector = resp.data[0].embedding

        # IMPORTANT: qdrant-client >=1.17 removed .search() — use .query_points()
        result = _qdrant.query_points(
            collection_name=RAG_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        hits = result.points   # .points not .result

        if not hits:
            return "No relevant documents found in the knowledge base."

        lines = []
        for h in hits:
            payload = h.payload or {}
            text    = payload.get("text") or payload.get("content") or str(payload)
            source  = payload.get("source", "unknown")
            lines.append(f"[score={h.score:.3f}] {source}\n{text}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"RAG search error: {e}"
```

---

### `tools/ingestion.py`

```python
"""
Document ingestion pipeline.
Flow: raw bytes → extract text → chunk → embed → upsert into Qdrant RAG collection.
Supports: PDF, TXT, MD.
"""

import io
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT, RAG_COLLECTION, EMBED_MODEL

_embed_client = OpenAI(api_key=OPENAI_API_KEY)
_qdrant       = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks


def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(file_bytes)
    if name.endswith((".txt", ".md")):
        return file_bytes.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {filename}. Send PDF, TXT, or MD.")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages  = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _ensure_collection() -> None:
    existing = [c.name for c in _qdrant.get_collections().collections]
    if RAG_COLLECTION not in existing:
        _qdrant.create_collection(
            collection_name=RAG_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )


def _embed_batch(texts: list[str]) -> list[list[float]]:
    resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest_document(file_bytes: bytes, filename: str, user_id: str) -> int:
    """Extract → chunk → embed → store in Qdrant. Returns chunk count stored."""
    text = extract_text(file_bytes, filename)
    if not text.strip():
        raise ValueError("Could not extract any text from the document.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no usable chunks.")

    _ensure_collection()

    BATCH = 32
    total = 0
    for i in range(0, len(chunks), BATCH):
        batch   = chunks[i : i + BATCH]
        vectors = _embed_batch(batch)
        points  = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text":      chunk,
                    "source":    filename,
                    "user_id":   user_id,
                    "chunk_idx": i + j,
                },
            )
            for j, (chunk, vec) in enumerate(zip(batch, vectors))
        ]
        _qdrant.upsert(collection_name=RAG_COLLECTION, points=points)
        total += len(points)

    return total
```

---

### `pipeline/context_ingestion.py`

```python
"""Unified context ingestion: merges session history + LTM facts into messages."""

from datetime import datetime


def _system_prompt() -> str:
    # Called fresh every request so date/time is always current
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    return f"""\
You are a helpful AI assistant with long-term memory and access to a personal knowledge base.
Current date: {date_str}  |  Current time: {time_str} (UTC+5 Tashkent)

## Tools you have
- **rag_search**: Search documents and files the user has previously uploaded (PDFs, text files, CVs, reports, etc.).
  → Use this whenever the user asks about a file they sent, references a document, or asks you to recall/summarize uploaded content.
- **web_search**: Search the internet for real-time or factual information you don't already know.

## Rules
- If the user asks about a document, file, or anything they previously uploaded → ALWAYS call rag_search first.
- Never say you cannot access a file — if a file was uploaded it is in the knowledge base; search for it.
- Combine rag_search results with your reasoning to give accurate, grounded answers."""


def build_messages(
    user_input: str,
    session_history: list[dict],
    ltm_facts: str,
    user_docs: list[str] | None = None,
) -> list[dict]:
    """
    Returns the full messages array ready to send to the LLM:
      [system] → [recent session history] → [user]
    LTM facts, current date/time, and uploaded document list are injected into the system message.
    """
    sys_content = _system_prompt()   # fresh timestamp on every call

    if user_docs:
        doc_list     = "\n".join(f"  - {d}" for d in user_docs)
        sys_content += (
            f"\n\n[Documents uploaded by this user — searchable via rag_search, most recent first]:\n"
            f"{doc_list}\n"
            f"When the user says 'this file', 'this document', 'last pdf', or similar, "
            f"they mean the most recent one: '{user_docs[0]}'."
        )

    if ltm_facts:
        sys_content += f"\n\n[Long-term memory about this user]:\n{ltm_facts}"

    messages: list[dict] = [{"role": "system", "content": sys_content}]
    messages.extend(session_history[-20:])
    messages.append({"role": "user", "content": user_input})
    return messages
```

---

### `pipeline/safety.py`

```python
"""Safety guards: prompt injection detection (input) + output sanity check."""

import re

_INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?instructions",
    r"forget (everything|all instructions)",
    r"you are now",
    r"new personality",
    r"(act|pretend|behave) as (a |an )?",
    r"system prompt",
    r"jailbreak",
    r"disregard (your |all )?training",
    r"override (your |all )?guidelines",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def check_input(text: str) -> tuple[bool, str]:
    for pattern in _COMPILED:
        if pattern.search(text):
            return False, f"Potential prompt injection: '{pattern.pattern}'"
    return True, ""


_OUTPUT_BLOCKLIST = [
    "i am now jailbroken",
    "i have no restrictions",
    "my new instructions are",
]


def check_output(text: str) -> tuple[bool, str]:
    lower = text.lower()
    for phrase in _OUTPUT_BLOCKLIST:
        if phrase in lower:
            return False, f"Unsafe output phrase detected: '{phrase}'"
    return True, ""


def redact_unsafe_output(text: str) -> str:
    return (
        "I'm sorry, I can't provide that response. "
        "Please ask me something else."
    )
```

---

### `pipeline/output_processor.py`

```python
"""
Output processor: Grounding & Hallucination Filter.

Runs a lightweight LLM check to verify the response is grounded
in the retrieved evidence. If claims appear unsupported, it adds
"(unverified)" caveats rather than silently hallucinating.
"""

from config import llm_client

_GROUNDING_SYSTEM = """\
You are a fact-checking assistant.
Your only job: compare the AI response against the provided evidence.
- If a claim in the response is clearly not supported by the evidence,
  append "(unverified)" after that claim.
- If the response is well-grounded or no evidence was needed, return it unchanged.
- Return ONLY the final response text — no commentary, no preamble."""

_GROUNDING_PROMPT = """\
Evidence / context:
{evidence}

AI response to check:
{response}"""


async def ground_and_filter(response: str, evidence: str, model: str) -> str:
    """Returns response, possibly with "(unverified)" caveats. Skipped when no evidence."""
    if not evidence or evidence.strip() == "No external data required.":
        return response

    result = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _GROUNDING_SYSTEM},
            {
                "role": "user",
                "content": _GROUNDING_PROMPT.format(
                    evidence=evidence[:3000],
                    response=response,
                ),
            },
        ],
    )
    return result.choices[0].message.content or response
```

---

### `pipeline/system1.py`

```python
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
                wait = 4.0 * (2 ** attempt)   # 4s → 8s → 16s
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
    """Returns (final_response_text, new_messages_appended)."""
    conversation = list(messages)
    appended: list[dict] = []
    tools = await _get_tool_schemas(mcp)

    ui.section("System 1 — Main LLM Orchestrator")
    ui.tools_list(tools)   # Shows tool names, not just count

    for rnd in range(max_rounds):
        t_llm = time.perf_counter()
        msg   = await _llm_call(model, conversation, tools)
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
                result  = await mcp.call_tool(name, args)
                content = "".join(
                    item.text if item.type == "text" else f"[{item.type}]"
                    for item in result.content
                )
            except Exception as e:
                content = f"Tool error: {e}"
            ui.timing(name, time.perf_counter() - t_tool)
            ui.tool_result(content)

            tool_msg = {
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         name,
                "content":      content,
            }
            conversation.append(tool_msg)
            appended.append(tool_msg)

    # Fallback: return last text found
    for msg in reversed(appended):
        if msg.get("content"):
            return msg["content"], appended
    return "I was unable to generate a response.", appended
```

---

### `pipeline/system2.py`

```python
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
- For current events, unknown specific facts → verdict "needs_data".

GOAL: {goal}
APPROACH: {approach}
EVIDENCE SO FAR:
{evidence}

Return ONLY valid JSON (no markdown fences):
{{
  "confidence": <integer 0-10>,
  "verdict": "validated" or "needs_data",
  "missing": "<what is still needed, or empty string>",
  "tool": "web_search" or "rag_search",
  "search_query": "<exact search query, or empty string>"
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


async def _llm(messages: list[dict], model: str, retries: int = 3) -> tuple[str, str]:
    for attempt in range(retries + 1):
        try:
            resp = await llm_client.chat.completions.create(model=model, messages=messages)
            msg  = resp.choices[0].message.model_dump()
            return msg.get("content") or "", _extract_reasoning(msg)
        except openai.InternalServerError:
            if attempt < retries:
                wait = 4.0 * (2 ** attempt)
                ui.warn(f"Xazna API unavailable — retry {attempt + 1}/{retries} in {wait:.0f}s…")
                await asyncio.sleep(wait)
            else:
                raise


async def _call_tool(mcp: FastMCP, tool_name: str, query: str) -> str:
    try:
        result = await mcp.call_tool(tool_name, {"query": query})
        return "".join(
            item.text if item.type == "text" else f"[{item.type}]"
            for item in result.content
        )
    except Exception as e:
        return f"Tool error: {e}"


_TOOL_MAP = {
    "web_search": "WebSearch_web_search",
    "rag_search": "RAG_rag_search",
}


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


async def _llm_with_tools(
    messages: list[dict],
    mcp: FastMCP,
    model: str,
    max_rounds: int = 4,
    retries: int = 3,
) -> tuple[str, str]:
    """LLM call with full tool-execution loop (mirrors System 1). Returns (content, reasoning)."""
    tools = await _get_tool_schemas(mcp)
    ui.tools_list(tools)
    conversation  = list(messages)
    last_reasoning = ""

    for rnd in range(max_rounds):
        msg = None
        for attempt in range(retries + 1):
            try:
                resp = await llm_client.chat.completions.create(
                    model=model, messages=conversation, tools=tools,
                )
                msg = resp.choices[0].message.model_dump()
                break
            except openai.InternalServerError:
                if attempt < retries:
                    wait = 4.0 * (2 ** attempt)
                    ui.warn(f"Xazna API unavailable — retry {attempt + 1}/{retries} in {wait:.0f}s…")
                    await asyncio.sleep(wait)
                else:
                    raise

        reasoning = _extract_reasoning(msg)
        if reasoning:
            last_reasoning = reasoning
            ui.reasoning_block(reasoning, f"Synthesis Round {rnd + 1}")

        conversation.append(msg)

        if not msg.get("tool_calls"):
            ui.no_tools_used()
            return msg.get("content") or "", last_reasoning

        ui.stage(f"Synthesis round {rnd + 1} — tool calls: {len(msg['tool_calls'])}")
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            ui.tool_call(name, json.dumps(args, ensure_ascii=False))

            t_tool = time.perf_counter()
            result = await _call_tool(mcp, name, args.get("query", ""))
            ui.timing(name, time.perf_counter() - t_tool)
            ui.tool_result(result)

            conversation.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         name,
                "content":      result,
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
) -> tuple[str, str]:
    """Returns (final_response_text, evidence_string)."""
    user_query       = messages[-1]["content"]
    reasoning_trace: list[str] = []
    evidence_pieces: list[str] = []

    # ══ Phase 1: Initial Strategy Formulation ═════════════════════════════════
    ui.section("System 2 — Phase 1: Initial Strategy Formulation")
    t_phase12 = time.perf_counter()

    strategy_input = _THOUGHT_SIG_PROMPT.format(query=user_query)
    ui.llm_input("Strategy prompt", f"[system] {_STRATEGY_SYSTEM}\n[user] {strategy_input}")

    sig_content, sig_reasoning = await _llm(
        messages=[
            {"role": "system", "content": _STRATEGY_SYSTEM},
            *messages[:-1],
            {"role": "user", "content": strategy_input},
        ],
        model=model,
    )

    ui.timing("Phase 1 (Strategy LLM)", time.perf_counter() - t_phase12)
    if sig_reasoning:
        ui.reasoning_block(sig_reasoning, "Strategy Internal Reasoning")

    # ══ Phase 2: Generate Thought Signature ════════════════════════════════════
    ui.section("System 2 — Phase 2: Thought Signature")
    ui.llm_output("Thought Signature (raw)", sig_content)

    sig        = _parse_json(sig_content)
    goal       = sig.get("goal", user_query)
    approach   = sig.get("approach", "Direct reasoning")
    confidence = int(sig.get("confidence", 5))
    needs_srch = bool(sig.get("needs_search", bool(sig.get("data_needed"))))

    ui.kv("Goal",         goal)
    ui.kv("Approach",     approach)
    ui.kv("Confidence",   f"{confidence}/10")
    ui.kv("Needs search", "Yes" if needs_srch else "No")
    ui.kv("Data needed",  str(sig.get("data_needed") or "none"))

    reasoning_trace.append(f"Strategy: {sig_content}")

    # ══ Phase 3: Self-Critique & Plan loop ════════════════════════════════════
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
            )
            ui.llm_input(f"Critique round {rnd+1}", critique_prompt)

            t_crit = time.perf_counter()
            crit_content, crit_reasoning = await _llm(
                messages=[
                    {"role": "system", "content": "Evaluate the plan. Return valid JSON only."},
                    {"role": "user",   "content": critique_prompt},
                ],
                model=model,
            )
            ui.timing(f"Critique round {rnd+1} (LLM)", time.perf_counter() - t_crit)
            ui.llm_output(f"Critique round {rnd+1} verdict", crit_content)

            if crit_reasoning:
                ui.reasoning_block(crit_reasoning, f"Critique Round {rnd+1}")

            crit         = _parse_json(crit_content)
            verdict      = crit.get("verdict", "validated")
            crit_conf    = int(crit.get("confidence", 7))
            search_query = crit.get("search_query", "").strip()
            tool_key     = crit.get("tool", "web_search")

            ui.kv("Verdict",    verdict)
            ui.kv("Confidence", f"{crit_conf}/10")
            ui.kv("Missing",    crit.get("missing", "") or "—")

            reasoning_trace.append(
                f"Critique round {rnd+1}: verdict={verdict} confidence={crit_conf}"
            )

            # CRITICAL: check verdict BEFORE breaking — must run tool if needs_data
            if verdict != "needs_data" or not search_query:
                if crit_conf >= 7 or verdict == "validated":
                    ui.ok("Plan validated — moving to Final Synthesis")
                    break

            # ── Proactive Tool Call ────────────────────────────────────────────
            if search_query:
                mcp_tool = _TOOL_MAP.get(tool_key, "WebSearch_web_search")
                ui.stage("Proactive Tool Call → MCP Sandbox")
                ui.kv("Tool",  mcp_tool)
                ui.kv("Query", search_query)
                ui.tool_call(mcp_tool, json.dumps({"query": search_query}))

                t_tool = time.perf_counter()
                result = await _call_tool(mcp, mcp_tool, search_query)
                ui.timing(mcp_tool, time.perf_counter() - t_tool)
                ui.tool_result(result)

                evidence_pieces.append(f"[{tool_key} | '{search_query}']:\n{result}")
                reasoning_trace.append(f"Evidence via {tool_key}: {result[:200]}")
                ui.stage("New Evidence accumulated")
                ui.evidence_state(evidence_pieces)

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

    # Phase 4 MUST use _llm_with_tools (not plain _llm) so the LLM can call
    # tools during synthesis instead of outputting raw <|tool_call|> markup.
    final_content, final_reasoning = await _llm_with_tools(
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            *messages[:-1],
            {"role": "user",   "content": synthesis_prompt},
        ],
        mcp=mcp,
        model=model,
    )
    ui.timing("Phase 4 (Synthesis)", time.perf_counter() - t_phase4)

    if final_reasoning:
        ui.reasoning_block(final_reasoning, "Final Synthesis Reasoning")

    ui.llm_output("Final answer (pre-grounding)", final_content)

    return final_content, evidence_combined
```

---

### `ui.py`

```python
"""Rich-based terminal UI — imported by agent.py and pipeline modules."""

from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text
from rich.rule    import Rule
from rich.theme   import Theme
from rich.table   import Table

_theme = Theme({
    "stage":   "bold cyan",
    "label":   "bold yellow",
    "dimval":  "dim",
    "ok":      "bold green",
    "warn":    "bold yellow",
    "err":     "bold red",
    "tool":    "bold magenta",
    "query":   "cyan",
    "reason":  "dim italic yellow",
    "user":    "bold green",
    "bot":     "bold blue",
    "section": "bold cyan",
})

console = Console(theme=_theme, highlight=False)


def blank() -> None:
    console.print()


def section(title: str) -> None:
    console.rule(f"[section] {title} [/section]", style="cyan dim")


def user_panel(text: str) -> None:
    console.print(Panel(
        Text(text, style="white"),
        title="[user]💬  You[/user]",
        border_style="green",
        padding=(0, 2),
    ))


def response_panel(text: str) -> None:
    console.print(Panel(
        Text(text, style="white"),
        title="[bot]🤖  Assistant[/bot]",
        border_style="blue",
        padding=(0, 2),
    ))


def tools_list(tools: list[dict]) -> None:
    names = [t["function"]["name"] for t in tools]
    console.print(
        f"      [label]{'Tools available':<16}[/label]  [bold]{len(names)}[/bold]  "
        f"[dim]({', '.join(names)})[/dim]"
    )


def stage(name: str, detail: str = "") -> None:
    if detail:
        console.print(f"  [stage]▸ {name}[/stage]  [dimval]{detail}[/dimval]")
    else:
        console.print(f"  [stage]▸ {name}[/stage]")


def kv(label: str, value: str, indent: int = 6) -> None:
    pad = " " * indent
    console.print(f"{pad}[label]{label:<16}[/label]  {value}")


def ok(msg: str) -> None:
    console.print(f"  [ok]✓[/ok]  {msg}")


def warn(msg: str) -> None:
    console.print(f"  [warn]⚠[/warn]   {msg}")


def err(msg: str) -> None:
    console.print(f"  [err]✗[/err]  {msg}")


def tool_call(tool_name: str, args: str) -> None:
    console.print(
        f"  [tool]🔧 TOOL CALL → {tool_name}[/tool]\n"
        f"      [label]args:[/label] [query]{args[:300]}[/query]"
    )


def tool_result(text: str) -> None:
    preview = text[:1500]
    suffix  = f"\n[dim]…+{len(text)-1500} chars truncated[/dim]" if len(text) > 1500 else ""
    console.print(Panel(
        Text(preview + suffix, style="dim"),
        title="[dim]↳ TOOL RESULT (sent back to LLM)[/dim]",
        border_style="dim cyan",
        padding=(0, 1),
    ))


def no_tools_used() -> None:
    console.print("      [dim]  (LLM answered directly — no tools called this round)[/dim]")


def timing(label: str, elapsed: float) -> None:
    ms    = elapsed * 1000
    color = "green" if ms < 500 else ("yellow" if ms < 2000 else "red")
    console.print(f"      [dim]{label:<22}[/dim]  [{color}]⏱ {ms:.0f} ms[/{color}]")


def total_time(label: str, elapsed: float) -> None:
    ms    = elapsed * 1000
    color = "green" if ms < 3000 else ("yellow" if ms < 8000 else "red")
    console.rule(
        f"[bold {color}]⏱  {label}: {ms:.0f} ms[/bold {color}]",
        style=f"{color} dim",
    )


def llm_input(label: str, content: str, max_chars: int = 500) -> None:
    preview = content[:max_chars].replace("\n", " ↵ ")
    if len(content) > max_chars:
        preview += f"  [dim]…+{len(content)-max_chars} chars[/dim]"
    console.print(Panel(
        Text(preview),
        title=f"[bold magenta]→ INPUT  {label}[/bold magenta]",
        border_style="magenta dim",
        padding=(0, 1),
    ))


def llm_output(label: str, content: str, max_chars: int = 600) -> None:
    preview = content[:max_chars]
    if len(content) > max_chars:
        preview += f"\n[dim]…+{len(content)-max_chars} chars[/dim]"
    console.print(Panel(
        Text(preview, style="bold white"),
        title=f"[bold cyan]← OUTPUT  {label}[/bold cyan]",
        border_style="cyan dim",
        padding=(0, 1),
    ))


def evidence_state(pieces: list[str]) -> None:
    if not pieces:
        console.print("      [dim]  Evidence: none yet[/dim]")
        return
    total = sum(len(p) for p in pieces)
    console.print(f"      [yellow]Evidence accumulated:[/yellow]  "
                  f"[bold]{len(pieces)} piece(s)[/bold]  [dim]({total} chars total)[/dim]")
    for i, p in enumerate(pieces):
        preview = p[:200].replace("\n", " ")
        if len(p) > 200:
            preview += "…"
        console.print(f"      [dim][{i+1}] {preview}[/dim]")


def session_dump(history: list[dict]) -> None:
    if not history:
        console.print("      [dim]  (empty session)[/dim]")
        return
    for msg in history:
        role    = msg.get("role", "?")
        content = str(msg.get("content") or "")
        preview = content[:200].replace("\n", " ")
        if len(content) > 200:
            preview += "…"
        color = "green" if role == "user" else "blue"
        console.print(f"      [bold {color}][{role:9}][/bold {color}]  {preview}")


def ltm_dump(facts: str) -> None:
    if not facts:
        console.print("      [dim]  (no LTM facts)[/dim]")
        return
    for line in facts.splitlines():
        console.print(f"      [yellow]•[/yellow] {line.lstrip('- ')}")


def messages_dump(messages: list[dict]) -> None:
    for msg in messages:
        role    = msg.get("role", "?")
        content = str(msg.get("content") or "")
        preview = content[:300].replace("\n", " ↵ ")
        if len(content) > 300:
            preview += "…"
        colors = {"system": "magenta", "user": "green", "assistant": "blue", "tool": "cyan"}
        c = colors.get(role, "white")
        console.print(f"      [bold {c}][{role:9}][/bold {c}]  {preview}")


def save_redis(user_input: str, response: str) -> None:
    u = user_input[:150].replace("\n", " ")
    r = response[:150].replace("\n", " ")
    console.print(f"      [bold green][user     ][/bold green]  {u}")
    console.print(f"      [bold blue][assistant][/bold blue]  {r}")


def save_mem0(user_input: str, response: str) -> None:
    combined = f"User: {user_input[:120]}  |  Assistant: {response[:120]}"
    console.print(f"      [yellow]{combined.replace(chr(10), ' ')}[/yellow]")


def reasoning_block(text: str, label: str = "Reasoning") -> None:
    if not text:
        return
    short = text[:700] + ("…" if len(text) > 700 else "")
    console.print(Panel(
        Text(short, style="dim italic"),
        title=f"[yellow]🤔 {label}[/yellow]",
        border_style="yellow dim",
        padding=(0, 1),
    ))
```

---

### `agent.py`

```python
"""
agent.py — Main entry point (CLI + shared core for Telegram bot).

Full pipeline per turn:

  User Input
      │
      ├─ [Redis] load session history
      ├─ [Mem0]  search long-term memory          ← parallel
      │
      ▼
  Unified Context Ingestion
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
      ├─ Redis: Save Turn   (background)
      └─ Mem0:  Upsert Facts (background)
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
from memory.session import get_session, save_turn, get_user_docs
from memory.ltm     import search_ltm, upsert_ltm
from tools.mcp_server import search_mcp, rag_mcp
from pipeline.context_ingestion import build_messages
from pipeline import safety
from pipeline import system1, system2
from pipeline.output_processor import ground_and_filter

# ── Deepthink intent detection ─────────────────────────────────────────────────
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
main_mcp.mount(search_mcp, namespace="WebSearch")
main_mcp.mount(rag_mcp,    namespace="RAG")


async def initialize() -> str:
    """Resolve model ID from the Xazna API. Must be called once at startup."""
    models   = await config.llm_client.models.list()
    model_id = models.data[0].id
    config.MODEL_ID = model_id
    ui.console.print(
        f"\n[dim]Model:[/dim] [bold cyan]{model_id}[/bold cyan]"
        f"  [dim]│[/dim]  [bold green]Deepthink ON[/bold green] by default\n"
    )
    return model_id


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def process_turn(
    user_input: str,
    user_id:    str,
    deepthink:  bool = True,
) -> str:
    """Process one user turn through the full pipeline. Returns assistant response."""
    model = config.MODEL_ID

    t_overall = time.perf_counter()

    ui.blank()
    ui.user_panel(user_input)

    # ── 1. Context Ingestion ────────────────────────────────────────────────────
    ui.section("Context Ingestion")
    ui.stage("Loading Redis session + Mem0 LTM + user docs in parallel…")

    _times: dict[str, float] = {}

    async def _timed(coro, label: str):
        t      = time.perf_counter()
        result = await coro
        _times[label] = time.perf_counter() - t
        return result

    session_hist, ltm_facts, user_docs = await asyncio.gather(
        _timed(get_session(user_id),            "Redis session"),
        _timed(search_ltm(user_input, user_id), "Mem0 LTM search"),
        _timed(get_user_docs(user_id),          "User docs (Redis)"),
    )

    for label, elapsed in _times.items():
        ui.timing(label, elapsed)

    n_turns = len(session_hist) // 2
    ui.kv("Redis session",  f"{n_turns} turn{'s' if n_turns != 1 else ''}" if n_turns else "empty (new session)")
    ui.session_dump(session_hist[-6:])

    ui.kv("Mem0 LTM facts", f"{len(ltm_facts.splitlines())} facts" if ltm_facts else "none found")
    ui.ltm_dump(ltm_facts)

    ui.kv("Uploaded docs",  ", ".join(user_docs) if user_docs else "none")

    messages = build_messages(user_input, session_hist, ltm_facts, user_docs=user_docs)
    ui.kv("Context window", f"{len(messages)} messages sent to LLM")
    ui.section("Messages → LLM")
    ui.messages_dump(messages)

    # ── 2. Safety Guard (input) ─────────────────────────────────────────────────
    ui.section("Prompt Ingestion & Safety Guard")
    is_safe, reason = safety.check_input(user_input)
    if not is_safe:
        ui.err(f"Input BLOCKED — {reason}")
        return "I'm sorry, I can't process that request."
    ui.ok("Input passed safety check")

    # ── 3. Routing ──────────────────────────────────────────────────────────────
    ui.section("Routing")
    evidence = ""

    forced = not deepthink and _wants_deepthink(user_input)
    if forced:
        ui.warn("Fast mode override — user requested deep thinking → System 2")

    t_system = time.perf_counter()
    if deepthink or forced:
        ui.stage("System 2", "Deepthink ON  (Strategy → Critique Loop → Synthesis)")
        response, evidence = await system2.run(messages, main_mcp, model)
        ui.total_time("System 2 total", time.perf_counter() - t_system)
    else:
        ui.stage("System 1", "Fast Mode  (direct tool-call loop)")
        response, _ = await system1.run(messages, main_mcp, model)
        ui.total_time("System 1 total", time.perf_counter() - t_system)

    # ── 4. Grounding & Hallucination Filter ────────────────────────────────────
    ui.section("Output Processing")
    ui.stage("Grounding & Hallucination Filter")
    t_ground = time.perf_counter()
    response = await ground_and_filter(response, evidence, model)
    ui.timing("Grounding filter", time.perf_counter() - t_ground)
    ui.ok("Grounding complete")

    # ── 5. Output Safety Guard ──────────────────────────────────────────────────
    out_safe, out_reason = safety.check_output(response)
    if not out_safe:
        ui.err(f"Output BLOCKED — {out_reason}")
        response = safety.redact_unsafe_output(response)
    else:
        ui.ok("Output passed safety check")

    # ── 6. Persist in background (Redis + Mem0) ─────────────────────────────────
    ui.section("Persisting")
    ui.stage("Saving to Redis (session turn)")
    ui.save_redis(user_input, response)
    ui.stage("Saving to Mem0 (LTM upsert)")
    ui.save_mem0(user_input, response)
    asyncio.create_task(_persist(user_input, response, user_id))

    ui.blank()
    ui.response_panel(response)
    ui.total_time("Overall pipeline", time.perf_counter() - t_overall)
    ui.blank()

    return response


async def _persist(user_input: str, response: str, user_id: str) -> None:
    try:
        await asyncio.gather(
            save_turn(user_id, user_input, response),
            upsert_ltm(user_input, response, user_id),
        )
        ui.console.print("[dim]  💾 Redis ✓  Mem0 ✓[/dim]")
    except Exception as e:
        ui.warn(f"Persist error: {e}")


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
    deepthink = True

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
```

---

### `telegram_bot.py`

```python
"""
Telegram bot interface.

Features:
  • Mode toggle button: 🧠 Deepthink  ↔  ⚡ Fast
  • New Session button (clears Redis history, keeps Mem0 LTM)
  • Continuous typing indicator (refreshed every 4s while processing)
  • Per-user mode stored in memory
  • Each Telegram user_id maps directly to the agent's user_id
  • Document ingestion: PDF, TXT, MD → Qdrant RAG
"""

import os
import asyncio
import logging
import html as _html
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton, BotCommand,
)
from aiogram.filters import CommandStart, Command
from aiogram.enums   import ChatAction
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest

load_dotenv()
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for _noisy in ("aiogram", "aiohttp", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import config
from agent          import process_turn, initialize
from memory.session import clear_session, add_user_doc
from tools.ingestion import ingest_document
import ui

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ── Per-user settings (in-memory) ─────────────────────────────────────────────
_settings: dict[int, dict] = {}


def _get_deepthink(uid: int) -> bool:
    return _settings.setdefault(uid, {"deepthink": True})["deepthink"]


def _set_deepthink(uid: int, val: bool) -> None:
    _settings.setdefault(uid, {})["deepthink"] = val


# ── Inline keyboard ────────────────────────────────────────────────────────────

def _keyboard(uid: int) -> InlineKeyboardMarkup:
    deepthink    = _get_deepthink(uid)
    toggle_label = "⚡ Switch to Fast"       if deepthink else "🧠 Switch to Deepthink"
    mode_label   = "🧠 Deepthink ✓"         if deepthink else "⚡ Fast ✓"
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=mode_label,    callback_data="noop"),
            InlineKeyboardButton(text=toggle_label,  callback_data="toggle_mode"),
        ],
        [
            InlineKeyboardButton(text="🆕 New Session", callback_data="new_session"),
        ],
    ])


# ── Message helpers ────────────────────────────────────────────────────────────

def _split_4096(text: str) -> list[str]:
    MAX = 4000
    if len(text) <= MAX:
        return [text]
    chunks, buf = [], ""
    for word in text.split(" "):
        if len(buf) + len(word) + 1 > MAX:
            chunks.append(buf.rstrip())
            buf = word + " "
        else:
            buf += word + " "
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# ── Typing indicator loop ──────────────────────────────────────────────────────

async def _keep_typing(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    """Re-send ChatAction.TYPING every 4 s until stop is set (Telegram expires it after ~5 s)."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(stop.wait()), timeout=4.0)
        except asyncio.TimeoutError:
            pass


# ── Handlers ──────────────────────────────────────────────────────────────────

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    uid  = message.from_user.id
    _set_deepthink(uid, True)
    name = message.from_user.first_name or "there"
    await message.answer(
        f"👋 Hi <b>{name}</b>! I'm your AI Assistant.\n\n"
        "🧠 <b>Deepthink</b> — I reason step-by-step, search for evidence, "
        "then synthesize a careful answer.\n"
        "⚡ <b>Fast</b> — Direct response with tool access, no deep reasoning.\n\n"
        "Just send me a message to get started!",
        reply_markup=_keyboard(uid),
    )


@router.message(Command("mode"))
async def cmd_mode(message: Message) -> None:
    uid  = message.from_user.id
    mode = "🧠 Deepthink" if _get_deepthink(uid) else "⚡ Fast"
    await message.answer(f"Current mode: <b>{mode}</b>", reply_markup=_keyboard(uid))


@router.message(Command("newsession"))
async def cmd_newsession(message: Message) -> None:
    uid = message.from_user.id
    await clear_session(str(uid))
    await message.answer(
        "🆕 <b>New session started!</b>\n"
        "Conversation history cleared. Long-term memories are preserved.",
        reply_markup=_keyboard(uid),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "<b>Commands:</b>\n"
        "/start      — Welcome + reset mode\n"
        "/mode       — Show current mode\n"
        "/newsession — Clear conversation history\n"
        "/help       — This message\n\n"
        "<b>Buttons on each reply:</b>\n"
        "🧠 Deepthink / ⚡ Fast  — switch reasoning mode\n"
        "🆕 New Session          — start fresh conversation\n\n"
        "<b>Long-term memory</b> is always active — I remember facts "
        "about you across sessions."
    )


@router.callback_query(F.data == "noop")
async def cb_noop(cb: CallbackQuery) -> None:
    await cb.answer()


@router.callback_query(F.data == "toggle_mode")
async def cb_toggle(cb: CallbackQuery) -> None:
    uid     = cb.from_user.id
    new_val = not _get_deepthink(uid)
    _set_deepthink(uid, new_val)
    label   = "🧠 Deepthink" if new_val else "⚡ Fast"
    await cb.answer(f"Switched to {label} mode")
    try:
        await cb.message.edit_reply_markup(reply_markup=_keyboard(uid))
    except TelegramBadRequest:
        pass


@router.callback_query(F.data == "new_session")
async def cb_new_session(cb: CallbackQuery) -> None:
    uid = cb.from_user.id
    await clear_session(str(uid))
    await cb.answer("Session cleared!")
    await cb.message.answer(
        "🆕 <b>New session started!</b>\n"
        "Previous conversation cleared. Long-term memories preserved.",
        reply_markup=_keyboard(uid),
    )


@router.message(F.document)
async def handle_document(message: Message) -> None:
    uid  = message.from_user.id
    doc  = message.document
    name = doc.file_name or "document"
    ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext not in ("pdf", "txt", "md"):
        await message.answer(
            "⚠️ Unsupported file type. Please send a <b>PDF</b>, <b>TXT</b>, or <b>MD</b> file.",
            reply_markup=_keyboard(uid),
        )
        return

    status = await message.answer(f"📄 <i>Indexing <b>{name}</b>…</i>")
    ui.section(f"Document ingestion — {name}  (user {uid})")

    try:
        file    = await message.bot.get_file(doc.file_id)
        buf     = await message.bot.download_file(file.file_path)
        data    = buf.read()

        n_chunks = await asyncio.to_thread(ingest_document, data, name, str(uid))
        await add_user_doc(str(uid), name)

        ui.ok(f"Indexed {n_chunks} chunks from '{name}'")
        await status.edit_text(
            f"✅ <b>{name}</b> indexed — {n_chunks} chunks added to the knowledge base.\n"
            "You can now ask me questions about this document.",
            reply_markup=_keyboard(uid),
        )
    except Exception as exc:
        ui.err(f"Ingestion error: {exc}")
        safe = _html.escape(str(exc))
        await status.edit_text(
            f"❌ <b>Ingestion failed:</b> {safe}",
            reply_markup=_keyboard(uid),
        )


@router.message(F.text)
async def handle_message(message: Message) -> None:
    uid  = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        return

    deepthink  = _get_deepthink(uid)
    mode_emoji = "🧠" if deepthink else "⚡"
    mode_name  = "Deepthink" if deepthink else "Fast"

    ui.section(f"Telegram → user {uid}  {mode_emoji} {mode_name}")

    # Start continuous typing indicator + status message
    _stop_typing = asyncio.Event()
    _typing_task = asyncio.create_task(
        _keep_typing(message.bot, message.chat.id, _stop_typing)
    )
    status = await message.answer(
        f"<i>{mode_emoji} {mode_name} mode — processing…</i>"
    )

    try:
        response = await process_turn(text, str(uid), deepthink=deepthink)

        await status.delete()

        chunks = _split_4096(response)
        for i, chunk in enumerate(chunks):
            kb = _keyboard(uid) if i == len(chunks) - 1 else None
            try:
                await message.answer(chunk, reply_markup=kb)
            except TelegramBadRequest:
                await message.answer(_html.escape(chunk), reply_markup=kb)

    except Exception as exc:
        logging.exception("process_turn error")
        safe = _html.escape(str(exc))[:300]
        try:
            await status.edit_text(
                f"❌ <b>Error:</b> {safe}",
                reply_markup=_keyboard(uid),
            )
        except TelegramBadRequest:
            await message.answer(f"❌ Error: {safe}", reply_markup=_keyboard(uid))

    finally:
        _stop_typing.set()
        await asyncio.gather(_typing_task, return_exceptions=True)


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "Add it to .env:  TELEGRAM_BOT_TOKEN=<your-token>"
        )

    await initialize()

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    await bot.set_my_commands([
        BotCommand(command="start",      description="Start / reset"),
        BotCommand(command="mode",       description="Show current mode"),
        BotCommand(command="newsession", description="Clear conversation history"),
        BotCommand(command="help",       description="Help & commands"),
    ])

    dp = Dispatcher()
    dp.include_router(router)

    from rich.panel import Panel
    ui.console.print(Panel(
        "[bold green]Telegram bot is live — waiting for messages[/bold green]\n"
        "[dim]Terminal shows the full pipeline for every incoming message.\n"
        "Chat happens in Telegram. No input needed here.[/dim]",
        border_style="green",
        padding=(0, 2),
    ))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Critical Bugs & Their Fixes

These were real bugs encountered during development. Any AI reproducing this project must implement these fixes from the start.

### Bug 1: `QdrantClient` has no attribute `search`
**Cause:** `qdrant-client >= 1.17` removed `.search()`.
**Fix:** Use `.query_points()` and access `.points` (not `.result`):
```python
result = _qdrant.query_points(collection_name=RAG_COLLECTION, query=vector, limit=top_k, with_payload=True)
hits = result.points
```

### Bug 2: System 2 outputs raw `<|tool_call>call:rag_search{...}<tool_call|>` markup
**Cause:** Phase 4 (Final Synthesis) used plain `_llm()` which has no `tools=` parameter, so the LLM emitted tool markup as text instead of executing tools.
**Fix:** Phase 4 must use `_llm_with_tools()` which runs the full tool-execution loop.

### Bug 3: Retry wrapping entire System 2 (phases restart on 504)
**Cause:** The retry wrapper was around `system2.run()`, so a 504 in Phase 4 would restart all 4 phases.
**Fix:** Retry inside each `_llm()` and `_llm_call()` — only the specific failed API call retries.

### Bug 4: Date hallucination (LLM invents wrong date)
**Cause:** No current date in the system prompt.
**Fix:** `_system_prompt()` is a function (not a constant) called fresh each request, injecting:
```
Current date: 2026-05-13  |  Current time: 14:30 (UTC+5 Tashkent)
```

### Bug 5: Phase 3 skips tool calls even when verdict is `needs_data`
**Cause:** `if verdict == "validated" or crit_conf >= 7: break` triggered before the tool call block.
**Fix:**
```python
if verdict != "needs_data" or not search_query:
    if crit_conf >= 7 or verdict == "validated":
        ui.ok("Plan validated — moving to Final Synthesis")
        break
# Tool call happens AFTER the break check
if search_query:
    ...call tool...
```

### Bug 6: `TelegramBadRequest` — can't parse HTML entities
**Cause:** Error messages (504 HTML pages) sent as Telegram HTML without escaping.
**Fix:** Always `_html.escape(str(exc))` before sending errors to Telegram.

### Bug 7: `typing` indicator disappears after ~5 seconds
**Cause:** `send_chat_action(TYPING)` expires after ~5s and was only sent once.
**Fix:** Background task `_keep_typing()` resends it every 4s in a loop with `asyncio.Event` for clean shutdown.

### Bug 8: Uzbek deep-think phrases not triggering System 2 override
**Cause:** Regex only had English phrases.
**Fix:** Added Uzbek keywords: `yaxshilab`, `batafsil`, `sinchiklab`, `chuqur o'yla`, `chuqur tahlil`.

---

## 8. Required Python Packages

```
openai
mem0ai
redis[asyncio]
qdrant-client>=1.17
fastmcp
aiogram>=3.0
rich
pypdf
ddgs
python-dotenv
```

Install: `pip install openai mem0ai "redis[asyncio]" "qdrant-client>=1.17" fastmcp "aiogram>=3.0" rich pypdf ddgs python-dotenv`

---

## 9. How to Run

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Set environment variables
cp .env.example .env   # fill in OPENAI_API_KEY, NEO4J_PASSWORD, TELEGRAM_BOT_TOKEN

# 3. Install dependencies
pip install openai mem0ai "redis[asyncio]" "qdrant-client>=1.17" fastmcp "aiogram>=3.0" rich pypdf ddgs python-dotenv

# 4a. Run Telegram bot (recommended)
python telegram_bot.py

# 4b. Or run CLI
python agent.py
```

---

## 10. Data Flow Diagram

```
User Message (Telegram)
        │
        ▼
  handle_message()
        │
        ├── _keep_typing() [background task, every 4s]
        │
        ▼
  process_turn(user_input, user_id, deepthink)
        │
        ├── asyncio.gather:
        │     ├── get_session(user_id)        → Redis → list[dict]
        │     ├── search_ltm(query, user_id)  → Mem0 → str (facts)
        │     └── get_user_docs(user_id)      → Redis → list[str]
        │
        ├── build_messages()
        │     → [system(date+tools+docs+LTM)] + [session[-20:]] + [user]
        │
        ├── safety.check_input()  [regex, fast]
        │
        ├── ROUTING:
        │     deepthink=True or forced → system2.run()
        │     deepthink=False          → system1.run()
        │
        │   System 1:
        │     loop(LLM → tool_calls → LLM → ... → answer)
        │
        │   System 2:
        │     Phase 1: _llm() → Thought Signature JSON
        │     Phase 2: parse confidence, needs_search
        │     Phase 3: if needs_search:
        │                 loop(_llm() → critique JSON → tool call → evidence)
        │     Phase 4: _llm_with_tools() → final answer
        │
        ├── ground_and_filter(response, evidence)
        │     → LLM checks claims against evidence, adds "(unverified)" if needed
        │
        ├── safety.check_output()
        │
        └── asyncio.create_task(_persist()):
              ├── save_turn() → Redis
              └── upsert_ltm() → Mem0 (Qdrant + Neo4j)
```

---

## 11. MCP Tool Namespacing

Tools are mounted with namespaces so the LLM sees:
- `WebSearch_web_search` (not `web_search`)
- `RAG_rag_search` (not `rag_search`)

This is why `_TOOL_MAP` in system2.py maps short names to full names:
```python
_TOOL_MAP = {
    "web_search": "WebSearch_web_search",
    "rag_search": "RAG_rag_search",
}
```

And in `agent.py`:
```python
main_mcp = FastMCP("Main")
main_mcp.mount(search_mcp, namespace="WebSearch")
main_mcp.mount(rag_mcp,    namespace="RAG")
```

---

## 12. Empty `__init__.py` Files Required

```
memory/__init__.py     — empty
pipeline/__init__.py   — empty
tools/__init__.py      — empty
```
