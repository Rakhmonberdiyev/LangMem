import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# --- LLM (Xazna) ---
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY = "sk-raximberdi-cmF4aW1iZXJkaQ"

llm_client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# Model ID is resolved at startup via initialize() in agent.py
MODEL_ID: str = ""

# --- Mem0 (LTM): Qdrant vectors + Neo4j graph ---
#
# Two-layer memory architecture:
#   mem0          (Qdrant)  — Active atomic facts, current state only.
#                             Mem0 auto-deduplicates: update/delete old fact
#                             when a new one conflicts. Always the latest "you".
#   mem0_history  (SQLite)  — Change-log archive. Every ADD / UPDATE / DELETE
#                             event is recorded here automatically by Mem0.
#                             Used to answer "when did I change X?" queries.
#                             Stored at mem0_history.db next to this file.
#
# Both LLM and embedder use Xazna — no OpenAI dependency.
#
MEM0_CONFIG = {
    "llm": {
        "provider": "openai",           # openai-compatible protocol
        "config": {
            "model":            "/models/gemma",
            "api_key":          LLM_API_KEY,
            "openai_base_url":  LLM_BASE_URL,
        },
    },
    "embedder": {
        "provider": "openai",           # openai-compatible protocol
        "config": {
            "model":            "/models/embedding",
            "api_key":          LLM_API_KEY,
            "openai_base_url":  LLM_BASE_URL,
            # Do NOT set embedding_dims here — Xazna does not support the
            # OpenAI matryoshka `dimensions` parameter and will return 400.
            # The 2048 dim is declared in vector_store.embedding_model_dims below.
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "mem0", 
            "embedding_model_dims": 2048,         # Active facts — current state
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD", "password123"),
        },
    },
    "history_db_path": os.path.join(os.path.dirname(__file__), "mem0_history.db"),
    "custom_prompt": """\
Extract ONLY meaningful semantic facts from the conversation — things worth remembering long-term.

INCLUDE:
- Personal facts: name, age, job, location, nationality
- Preferences: likes, dislikes, favorite things
- Goals, plans, projects the user is working on
- Skills and expertise the user has
- People, places, and organizations the user mentions
- Facts the user explicitly states about themselves or others

EXCLUDE (do NOT extract these as entities or facts):
- Grammatical words: pronouns (men, siz, I, you), greetings (salom, hello), conjunctions, verbs
- Generic question words or filler words
- Temporary conversational context (e.g. "what's the weather today")
- Facts about the world in general (China history, Tesla specs) — only personal facts

Always output the user's language (Uzbek, English, Russian, etc.) back as-is.
""",
}

# Pre-create the mem0 Qdrant collection at 2048 dims BEFORE Mem0 initializes.
# Wrapped in try/except so the server can start even when Qdrant is temporarily down.
try:
    _qc = QdrantClient(host="localhost", port=6333)
    _existing = {c.name: c for c in _qc.get_collections().collections}
    if "mem0" in _existing:
        _current_dim = _qc.get_collection("mem0").config.params.vectors.size
        if _current_dim != 2048:
            _qc.delete_collection("mem0")
            _qc.create_collection("mem0", vectors_config=VectorParams(size=2048, distance=Distance.COSINE))
    else:
        _qc.create_collection("mem0", vectors_config=VectorParams(size=2048, distance=Distance.COSINE))
    _qc.close()
    del _qc, _existing
except Exception as _e:
    print(f"[config] WARNING: Qdrant unavailable at startup ({_e}) — RAG/LTM will fail until Qdrant is up.")

try:
    ltm_memory = Memory.from_config(MEM0_CONFIG)
except Exception as _e:
    ltm_memory = None
    print(f"[config] WARNING: Mem0 LTM unavailable ({_e}) — long-term memory disabled until Qdrant is up.")

# --- Redis (session history) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SESSION_TTL = 86400          # 24 hours (seconds)
MAX_SESSION_MESSAGES = 40    # rolling window

# --- Qdrant RAG ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
RAG_COLLECTION = "knowledge_base"
EMBED_MODEL = "/models/embedding"
