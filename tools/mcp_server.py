"""MCP tools: web_search (DuckDuckGo) and rag_search (Qdrant knowledge_base)."""

from fastmcp import FastMCP
try:
    from ddgs import DDGS          # new package name
except ImportError:
    from duckduckgo_search import DDGS  # fallback
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT, RAG_COLLECTION, EMBED_MODEL

# --- FastMCP servers (mounted together in agent.py) ---
search_mcp = FastMCP("WebSearch")
rag_mcp = FastMCP("RAG")

_embed_client = OpenAI(api_key=OPENAI_API_KEY)
_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _ensure_rag_collection() -> bool:
    """Create the RAG collection if it doesn't exist. Returns True if ready."""
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
        resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=query)
        vector = resp.data[0].embedding

        result = _qdrant.query_points(
            collection_name=RAG_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        hits = result.points
        if not hits:
            return "No relevant documents found in the knowledge base."

        lines = []
        for h in hits:
            payload = h.payload or {}
            text = payload.get("text") or payload.get("content") or str(payload)
            source = payload.get("source", "unknown")
            lines.append(f"[score={h.score:.3f}] {source}\n{text}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"RAG search error: {e}"
