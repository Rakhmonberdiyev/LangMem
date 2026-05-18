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

from config import LLM_API_KEY, LLM_BASE_URL, QDRANT_HOST, QDRANT_PORT, RAG_COLLECTION, EMBED_MODEL

_embed_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks


# ── Text extraction ────────────────────────────────────────────────────────────

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
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Embedding + Qdrant upsert ──────────────────────────────────────────────────

def _ensure_collection() -> None:
    existing = [c.name for c in _qdrant.get_collections().collections]
    if RAG_COLLECTION not in existing:
        _qdrant.create_collection(
            collection_name=RAG_COLLECTION,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )


def _embed_batch(texts: list[str]) -> list[list[float]]:
    resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_document(file_bytes: bytes, filename: str, user_id: str) -> int:
    """
    Extract → chunk → embed → store in Qdrant.
    Returns the number of chunks stored.
    """
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
        batch = chunks[i : i + BATCH]
        vectors = _embed_batch(batch)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text":     chunk,
                    "source":   filename,
                    "user_id":  user_id,
                    "chunk_idx": i + j,
                },
            )
            for j, (chunk, vec) in enumerate(zip(batch, vectors))
        ]
        _qdrant.upsert(collection_name=RAG_COLLECTION, points=points)
        total += len(points)

    return total
