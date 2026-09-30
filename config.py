import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# --- Qdrant connection ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6335"))

# --- LLM (Xazna) ---
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY = "sk-raximberdi-cmF4aW1iZXJkaQ"

llm_client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# Model ID is resolved at startup via initialize() in agent.py
MODEL_ID: str = ""

# --- Redis (session history) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
SESSION_TTL = 86400
MAX_SESSION_MESSAGES = 40

# --- Qdrant RAG ---
RAG_COLLECTION = "knowledge_base"
EMBED_MODEL = "/models/embedding"

# --- LangMem (three-tier Qdrant collections) ---
EMBED_DIM             = 2048
SEMANTIC_COLLECTION   = "langmem_semantic"
EPISODIC_COLLECTION   = "langmem_episodic"
PROCEDURAL_COLLECTION = "langmem_procedural"

# Pre-create all LangMem collections at startup
try:
    _qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _existing = {c.name for c in _qc.get_collections().collections}
    for _col in (SEMANTIC_COLLECTION, EPISODIC_COLLECTION, PROCEDURAL_COLLECTION):
        if _col not in _existing:
            _qc.create_collection(
                _col,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
    _qc.close()
    del _qc, _existing, _col
except Exception as _e:
    print(f"[config] WARNING: Qdrant unavailable at startup ({_e})")
