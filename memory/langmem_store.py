"""
memory/langmem_store.py — LangMem Long-Term Storage Layer.

Tripartite Memory Architecture:
  1. Semantic Memory  — user facts, preferences, JSON profile
  2. Episodic Memory  — few-shot interaction history
  3. Procedural Memory— evolved prompt instructions / rules

Storage backend:  Qdrant — three separate collections per tier:
                    langmem_semantic   — user facts & preferences
                    langmem_episodic   — few-shot interaction examples
                    langmem_procedural — evolved system prompt rules

Background lifecycle:
  background_update()   → LLM-powered extractor: semantic consolidation + episodic distillation
  optimize_procedural() → Prompt optimizer (metaprompt): reflection → new system rules

Hot-path tools (called by the LLM during reasoning):
  search_semantic()      → Top-K semantic facts about this user
  search_episodic()      → Top-K relevant few-shot examples
  get_procedural_rules() → Active system instruction rules

manage_memory is exposed via tools/langmem_mcp.py for real-time edits.
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.store.base import Item, SearchItem
from langmem import create_memory_store_manager
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

import ui
from config import (
    EMBED_DIM,
    EPISODIC_COLLECTION,
    PROCEDURAL_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    SEMANTIC_COLLECTION,
)

# ── Xazna LLM / Embedding config ─────────────────────────────────────────────

_LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
_LLM_API_KEY  = "sk-raximberdi-cmF4aW1iZXJkaQ"
_LLM_MODEL    = "/models/gemma"
_EMBED_MODEL  = "/models/embedding"

_lang_model = ChatOpenAI(
    model=_LLM_MODEL,
    base_url=_LLM_BASE_URL,
    api_key=_LLM_API_KEY,
)

_embeddings = OpenAIEmbeddings(
    model=_EMBED_MODEL,
    base_url=_LLM_BASE_URL,
    api_key=_LLM_API_KEY,
)

# ── Namespace helpers ─────────────────────────────────────────────────────────

def _semantic_ns(user_id: str)   -> tuple: return ("user", user_id, "semantic")
def _episodic_ns(user_id: str)   -> tuple: return ("user", user_id, "episodic")
def _procedural_ns(user_id: str) -> tuple: return ("user", user_id, "procedural")


# ── QdrantLTMStore — LangGraph-compatible store backed by Qdrant ──────────────

class QdrantLTMStore:
    """
    LangGraph-compatible store — three separate Qdrant collections, one per tier:
        langmem_semantic   — user facts and preferences
        langmem_episodic   — few-shot interaction examples
        langmem_procedural — evolved system prompt rules

    Routing is based on the last element of the namespace tuple:
        (..., "semantic")   → langmem_semantic
        (..., "episodic")   → langmem_episodic
        (..., "procedural") → langmem_procedural

    Qdrant payload schema per point:
        ns      : str  — JSON-encoded namespace, e.g. '["user","u1","semantic"]'
        key     : str  — item key
        value   : dict — stored content  {"kind": "Memory", "content": {...}}
        updated : str  — ISO-8601 UTC timestamp
    """

    # namespace-tier → Qdrant collection name
    _TIER_MAP: dict[str, str] = {
        "semantic":   SEMANTIC_COLLECTION,
        "episodic":   EPISODIC_COLLECTION,
        "procedural": PROCEDURAL_COLLECTION,
    }

    def __init__(self) -> None:
        self._sync  = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
        self._async = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    # ── Collection routing ─────────────────────────────────────────────────

    def _collection(self, namespace: tuple) -> str:
        """Return the Qdrant collection for this namespace tier."""
        tier = namespace[-1] if namespace else "semantic"
        return self._TIER_MAP.get(tier, SEMANTIC_COLLECTION)

    # ── Key helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _point_id(namespace: tuple, key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}||{key}"))

    @staticmethod
    def _ns_str(namespace: tuple) -> str:
        return json.dumps(list(namespace))

    @staticmethod
    def _ns_filter(namespace: tuple) -> Filter:
        return Filter(must=[
            FieldCondition(key="ns", match=MatchValue(value=QdrantLTMStore._ns_str(namespace)))
        ])

    @staticmethod
    def _to_item(payload: dict) -> Item:
        """Used by aget — returns a plain Item (no score)."""
        ns = tuple(json.loads(payload["ns"]))
        ts = payload.get("updated", datetime.now(timezone.utc).isoformat())
        dt = datetime.fromisoformat(ts)
        return Item(namespace=ns, key=payload["key"], value=payload["value"],
                    created_at=dt, updated_at=dt)

    @staticmethod
    def _to_search_item(payload: dict, score: float = 1.0) -> SearchItem:
        """Used by asearch — returns a SearchItem with score (required by LangMem)."""
        ns = tuple(json.loads(payload["ns"]))
        ts = payload.get("updated", datetime.now(timezone.utc).isoformat())
        dt = datetime.fromisoformat(ts)
        return SearchItem(namespace=ns, key=payload["key"], value=payload["value"],
                          created_at=dt, updated_at=dt, score=score)

    # ── Sync operations (used by save_semantic / save_episodic) ───────────

    def put(self, namespace: tuple, key: str, value: dict, index=None, *, ttl=None) -> None:
        col    = self._collection(namespace)
        text   = _extract_text(value)
        vector = _embeddings.embed_query(text)
        self._sync.upsert(
            collection_name = col,
            points=[PointStruct(
                id      = self._point_id(namespace, key),
                vector  = vector,
                payload = {
                    "ns":      self._ns_str(namespace),
                    "key":     key,
                    "value":   value,
                    "updated": datetime.now(timezone.utc).isoformat(),
                },
            )],
        )

    # ── Async operations (used by LangMem managers + search functions) ────

    async def aput(self, namespace: tuple, key: str, value: dict, index=None, *, ttl=None) -> None:
        col    = self._collection(namespace)
        text   = _extract_text(value)
        vector = await _embeddings.aembed_query(text)
        await self._async.upsert(
            collection_name = col,
            points=[PointStruct(
                id      = self._point_id(namespace, key),
                vector  = vector,
                payload = {
                    "ns":      self._ns_str(namespace),
                    "key":     key,
                    "value":   value,
                    "updated": datetime.now(timezone.utc).isoformat(),
                },
            )],
        )

    async def asearch(
        self,
        namespace_prefix: tuple,
        *,
        query: Optional[str] = None,
        limit: int = 10,
        **_kwargs: Any,
    ) -> list[SearchItem]:
        col       = self._collection(namespace_prefix)
        ns_filter = self._ns_filter(namespace_prefix)
        if query:
            q_vec  = await _embeddings.aembed_query(query)
            result = await self._async.query_points(
                collection_name = col,
                query           = q_vec,
                query_filter    = ns_filter,
                limit           = limit,
                with_payload    = True,
            )
            return [self._to_search_item(p.payload, p.score)
                    for p in result.points if p.payload]
        else:
            scroll, _ = await self._async.scroll(
                collection_name = col,
                scroll_filter   = ns_filter,
                limit           = limit,
                with_payload    = True,
            )
            return [self._to_search_item(p.payload) for p in scroll if p.payload]

    async def aget(self, namespace: tuple, key: str) -> Optional[Item]:
        pts = await self._async.retrieve(
            collection_name = self._collection(namespace),
            ids             = [self._point_id(namespace, key)],
            with_payload    = True,
        )
        return self._to_item(pts[0].payload) if pts and pts[0].payload else None

    async def adelete(self, namespace: tuple, key: str) -> None:
        await self._async.delete(
            collection_name = self._collection(namespace),
            points_selector = PointIdsList(points=[self._point_id(namespace, key)]),
        )

    # Sync wrappers — delegate to async versions via event loop
    def search(self, namespace_prefix: tuple, *, query: Optional[str] = None,
               limit: int = 10, **kw) -> list[SearchItem]:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an async context (e.g. called from a sync function
            # that is itself called from async code) — run in a thread executor.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    asyncio.run,
                    self.asearch(namespace_prefix, query=query, limit=limit, **kw),
                )
                return future.result()
        return loop.run_until_complete(
            self.asearch(namespace_prefix, query=query, limit=limit, **kw)
        )

    def get(self, namespace: tuple, key: str) -> Optional[Item]:
        return asyncio.get_event_loop().run_until_complete(self.aget(namespace, key))

    def delete(self, namespace: tuple, key: str) -> None:
        asyncio.get_event_loop().run_until_complete(self.adelete(namespace, key))


# ── Singleton store instance ──────────────────────────────────────────────────

try:
    store = QdrantLTMStore()
except Exception as _e:
    ui.warn(f"[langmem] Qdrant unavailable ({_e}) — LTM will not persist until Qdrant is running.")
    store = None  # type: ignore[assignment]


# ── Text extraction helper ────────────────────────────────────────────────────

def _extract_text(value: Any) -> str:
    """
    Extract plain string for embedding from a store item value.

    Handles both LangMem manager format and direct-save format:
      LangMem:  {"kind": "Memory", "content": {"content": "text", ...}}
      Direct:   {"kind": "Memory", "content": {"content": "text"}}   (same after fix)
    """
    if not isinstance(value, dict):
        return str(value)
    # Procedural format: {"rule": "...", "version": N}
    if "rule" in value:
        return str(value["rule"])
    # LangMem manager format: {"kind": "Memory", "content": {"content": "..."}}
    inner = value.get("content", value)
    if isinstance(inner, dict):
        return inner.get("content", str(inner))
    return str(inner)


# ── LangMem background managers ──────────────────────────────────────────────

_semantic_manager = create_memory_store_manager(
    _lang_model,
    namespace=("user", "{langgraph_user_id}", "semantic"),
    instructions=(
        "Extract and store CURRENT STATE facts about the user — things that are true RIGHT NOW. "
        "Track: name, city, job, language preference, billing address, active card type and tier, "
        "credit limit, co-holder names, enabled features (foreign currency, SMS, autopay), income. "
        "CRITICAL: preserve all numeric values exactly as stated (write '7,000,000 UZS' not 'seven million'). "
        "When a fact changes, DELETE the old version — keep only the latest value per concept. "
        "Skip greetings and pure questions that contain no new facts about the user."
    ),
    store=store,
    enable_inserts=True,
    enable_deletes=True,
)

_episodic_manager = create_memory_store_manager(
    _lang_model,
    namespace=("user", "{langgraph_user_id}", "episodic"),
    instructions=(
        "Record WHAT HAPPENED as immutable history entries for temporal reasoning. "
        "For every meaningful exchange store: what the user requested, what changed, before/after values. "
        "Examples: 'Applied for Humo Classic card', 'Switched from Humo Classic to Visa Gold', "
        "'Credit limit set to 10,000,000 UZS', 'Credit limit reduced from 10,000,000 to 7,000,000 UZS', "
        "'Co-holder Nilufar added', 'Co-holder Nilufar removed — card is now sole-holder', "
        "'Billing address: Toshkent, Amir Temur 12', 'Income: 5,000,000 UZS/month', "
        "'Statement language set to Uzbek', 'Foreign currency transactions enabled'. "
        "CRITICAL: preserve all numeric values exactly (7,000,000 not 'seven million'). "
        "For changes always record BOTH old and new values. "
        "NEVER delete entries — history is immutable. Skip only pure greetings."
    ),
    store=store,
    enable_inserts=True,
    enable_deletes=False,
)



# ── Message converter ─────────────────────────────────────────────────────────

def _to_lc_messages(messages: list[dict]) -> list:
    result = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content") or ""
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
    return result


# ── Hot-path search ───────────────────────────────────────────────────────────

async def search_semantic(query: str, user_id: str, limit: int = 5) -> list[str]:
    if store is None:
        return []
    try:
        results = await store.asearch(_semantic_ns(user_id), query=query, limit=limit)
        ui.console.print(
            f"      [dim cyan]Qdrant collection:[/dim cyan] [bold cyan]{SEMANTIC_COLLECTION}[/bold cyan]"
            f"  [dim]({len(results)} hits)[/dim]"
        )
        texts = []
        for item in results:
            if not item.value:
                continue
            text = _extract_text(item.value)
            score = getattr(item, "score", None)
            score_str = f"  [dim]score={score:.3f}[/dim]" if score is not None else ""
            ui.console.print(f"        [dim]key={item.key[:20]}[/dim]{score_str}  {text[:120]}")
            texts.append(text)
        return texts
    except Exception as e:
        ui.warn(f"[langmem] Semantic search error: {e}")
        return []


async def search_episodic(query: str, user_id: str, limit: int = 3) -> list[str]:
    if store is None:
        return []
    try:
        results = await store.asearch(_episodic_ns(user_id), query=query, limit=limit)
        ui.console.print(
            f"      [dim cyan]Qdrant collection:[/dim cyan] [bold cyan]{EPISODIC_COLLECTION}[/bold cyan]"
            f"  [dim]({len(results)} hits)[/dim]"
        )
        texts = []
        for item in results:
            if not item.value:
                continue
            text = _extract_text(item.value)
            score = getattr(item, "score", None)
            score_str = f"  [dim]score={score:.3f}[/dim]" if score is not None else ""
            ui.console.print(f"        [dim]key={item.key[:20]}[/dim]{score_str}  {text[:120]}")
            texts.append(text)
        return texts
    except Exception as e:
        ui.warn(f"[langmem] Episodic search error: {e}")
        return []


async def get_procedural_rules(user_id: str) -> str:
    if store is None:
        return ""
    try:
        results = await store.asearch(
            _procedural_ns(user_id), query="system instructions rules", limit=10
        )
        if results:
            lines = []
            for item in results:
                val = item.value or {}
                rule_text = val.get("rule") or val.get("rules") or _extract_text(val)
                if rule_text:
                    lines.append(rule_text)
            return "\n".join(lines)
    except Exception:
        pass
    return ""


# ── Direct store write helpers (used by MCP tools) ────────────────────────────

def save_semantic(user_id: str, content: str, key: str | None = None) -> str:
    if store is None:
        return key or "no_store"
    # Deterministic key: same content → same Qdrant point → upsert overwrites instead of duplicating
    k = key or "fact_" + hashlib.sha1(content.encode()).hexdigest()[:12]
    store.put(_semantic_ns(user_id), k, {"kind": "Memory", "content": {"content": content}})
    return k


def save_episodic(user_id: str, content: str, key: str | None = None) -> str:
    if store is None:
        return key or "no_store"
    # Deterministic key: same content → same Qdrant point → upsert overwrites instead of duplicating
    k = key or "ep_" + hashlib.sha1(content.encode()).hexdigest()[:12]
    store.put(_episodic_ns(user_id), k, {"kind": "Memory", "content": {"content": content}})
    return k


# ── Background memory lifecycle ───────────────────────────────────────────────

async def background_update(messages: list[dict], user_id: str) -> None:
    if store is None or not messages:
        return

    # Prepend current UTC timestamp so episodic manager can record WHEN events happened
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    timestamped = [{"role": "system", "content": f"Current timestamp: {now_str}"}] + messages

    lc_msgs = _to_lc_messages(timestamped)
    if not lc_msgs:
        return
    cfg = {"configurable": {"langgraph_user_id": user_id}}

    try:
        await _semantic_manager.ainvoke({"messages": lc_msgs}, config=cfg)
        ui.console.print("[dim]  🧠 LangMem Semantic: facts updated[/dim]")
    except Exception as e:
        ui.warn(f"[langmem] Semantic manager error (non-fatal): {e}")

    # Episodic runs for any real exchange (≥2 messages = at least one user+assistant turn)
    if len(messages) >= 2:
        try:
            await _episodic_manager.ainvoke({"messages": lc_msgs}, config=cfg)
            ui.console.print("[dim]  📚 LangMem Episodic: event recorded[/dim]")
        except Exception as e:
            ui.warn(f"[langmem] Episodic manager error (non-fatal): {e}")


_PROC_SYSTEM = (
    "You are improving an AI assistant's behavior rules based on observed conversations. "
    "Analyze the conversation below and produce an improved set of rules. "
    "Return ONLY the updated rules text — no headers, no explanation, no preamble."
)


async def optimize_procedural(
    thread_messages: list[dict],
    user_id: str,
    current_rules: str = "",
) -> None:
    """
    Improve per-user procedural rules using a direct LLM call (no tool calling required).
    Called every 5 turns by agent.py:_persist.
    """
    if store is None or not thread_messages:
        return

    base_rules = current_rules or (
        "Always call the appropriate bank tool before answering from general knowledge. "
        "Respond in the user's language (Uzbek, Russian, or English). "
        "Be concise, factual, and grounded in tool results."
    )

    lc_msgs    = _to_lc_messages(thread_messages)
    convo_text = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {m.content}"
        for m in lc_msgs
        if hasattr(m, "content") and m.content
    )
    if not convo_text.strip():
        return

    prompt_text = (
        f"Current rules:\n{base_rules}\n\n"
        f"Recent conversation:\n{convo_text}\n\n"
        "Write improved rules that address any issues. Keep them concise and actionable."
    )

    try:
        response = await _lang_model.ainvoke([
            {"role": "system", "content": _PROC_SYSTEM},
            {"role": "user",   "content": prompt_text},
        ])
        rule_text = (response.content if hasattr(response, "content") else str(response)).strip()
        if rule_text:
            await store.aput(
                _procedural_ns(user_id),
                "system_rules",
                {
                    "rule":    rule_text,
                    "version": int(datetime.now(timezone.utc).timestamp()),
                },
            )
            ui.console.print("[dim]  ⚙️  LangMem Procedural: rules updated[/dim]")
    except Exception as e:
        import traceback
        ui.warn(f"[langmem] Procedural optimizer error:\n{traceback.format_exc()}")
