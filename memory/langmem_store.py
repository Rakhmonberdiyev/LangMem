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
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.store.base import Item, SearchItem
from langmem import Prompt, create_memory_store_manager, create_prompt_optimizer
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

    def put(self, namespace: tuple, key: str, value: dict) -> None:
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

    async def aput(self, namespace: tuple, key: str, value: dict) -> None:
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
    # LangMem format: value["content"] is a dict with a "content" key
    inner = value.get("content", value)
    if isinstance(inner, dict):
        return inner.get("content", str(inner))
    return str(inner)


# ── LangMem background managers ──────────────────────────────────────────────

_semantic_manager = create_memory_store_manager(
    _lang_model,
    namespace=("user", "{langgraph_user_id}", "semantic"),
    instructions=(
        "Extract and store long-term facts about the user: name, age, city, job, "
        "language preferences, goals, ongoing projects, named people and organisations, "
        "and any persistent preferences or dislikes. "
        "Skip greetings, filler, and one-off requests that won't be relevant in future sessions. "
        "Consolidate and deduplicate — update existing facts rather than creating redundant entries."
    ),
    store=store,
    enable_inserts=True,
    enable_deletes=False,
)

_episodic_manager = create_memory_store_manager(
    _lang_model,
    namespace=("user", "{langgraph_user_id}", "episodic"),
    instructions=(
        "Identify high-quality, reusable interaction examples from this conversation. "
        "Store examples where: the user asked a complex domain question and received a highly "
        "accurate tool-grounded answer; the user gave explicit positive feedback; "
        "or the exchange demonstrates an important reasoning pattern. "
        "Format as a compact (user question → key facts used → concise answer) triple."
    ),
    store=store,
    enable_inserts=True,
    enable_deletes=True,
)

_proc_optimizer = create_prompt_optimizer(_lang_model, kind="metaprompt")


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
        return [_extract_text(item.value) for item in results if item.value]
    except Exception as e:
        ui.warn(f"[langmem] Semantic search error: {e}")
        return []


async def search_episodic(query: str, user_id: str, limit: int = 3) -> list[str]:
    if store is None:
        return []
    try:
        results = await store.asearch(_episodic_ns(user_id), query=query, limit=limit)
        return [_extract_text(item.value) for item in results if item.value]
    except Exception as e:
        ui.warn(f"[langmem] Episodic search error: {e}")
        return []


async def get_procedural_rules(user_id: str) -> str:
    if store is None:
        return ""
    try:
        results = await store.asearch(
            _procedural_ns(user_id), query="system instructions rules", limit=1
        )
        if results:
            val = results[0].value
            # Procedural rules may be stored directly {"rules": "..."} or via LangMem format
            return (val.get("rules", "")
                    or _extract_text(val)
                    or "")
    except Exception:
        pass
    return ""


# ── Direct store write helpers (used by MCP tools) ────────────────────────────

def save_semantic(user_id: str, content: str, key: str | None = None) -> str:
    if store is None:
        return key or "no_store"
    k = key or f"fact_{uuid.uuid4().hex[:12]}"
    # LangMem expects {"kind": "Memory", "content": {"content": "..."}}
    store.put(_semantic_ns(user_id), k, {"kind": "Memory", "content": {"content": content}})
    return k


def save_episodic(user_id: str, content: str, key: str | None = None) -> str:
    if store is None:
        return key or "no_store"
    k = key or f"ep_{uuid.uuid4().hex[:12]}"
    store.put(_episodic_ns(user_id), k, {"kind": "Memory", "content": {"content": content}})
    return k


# ── Background memory lifecycle ───────────────────────────────────────────────

async def background_update(messages: list[dict], user_id: str) -> None:
    if store is None or not messages:
        return
    lc_msgs = _to_lc_messages(messages)
    if not lc_msgs:
        return
    cfg = {"configurable": {"langgraph_user_id": user_id}}
    try:
        await _semantic_manager.ainvoke({"messages": lc_msgs}, config=cfg)
        ui.console.print("[dim]  🧠 LangMem Semantic: consolidated[/dim]")
    except Exception as e:
        ui.warn(f"[langmem] Semantic manager error (non-fatal): {e}")
    if len(lc_msgs) >= 6:
        try:
            await _episodic_manager.ainvoke({"messages": lc_msgs}, config=cfg)
            ui.console.print("[dim]  📚 LangMem Episodic: distilled[/dim]")
        except Exception as e:
            ui.warn(f"[langmem] Episodic manager error (non-fatal): {e}")


async def optimize_procedural(
    thread_messages: list[dict],
    user_id: str,
    current_rules: str = "",
) -> None:
    if store is None or not thread_messages:
        return
    lc_msgs    = _to_lc_messages(thread_messages)
    trajectory = [{"messages": lc_msgs, "label": "positive"}]
    base_rules = current_rules or (
        "Always call the appropriate bank tool before answering from general knowledge. "
        "Respond in the user's language (Uzbek, Russian, or English). "
        "Be concise, factual, and grounded in tool results."
    )
    try:
        new_rules = await _proc_optimizer.ainvoke({
            "trajectories": trajectory,
            "prompt": Prompt(name="system_rules", prompt=base_rules),
        })
        if isinstance(new_rules, str) and new_rules.strip():
            await store.aput(
                _procedural_ns(user_id),
                "rules",
                {"rules": new_rules.strip(), "content": new_rules.strip()},
            )
            ui.console.print("[dim]  ⚙️  LangMem Procedural: rules updated[/dim]")
    except Exception as e:
        ui.warn(f"[langmem] Procedural optimizer error (non-fatal): {e}")
