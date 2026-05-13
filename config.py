import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mem0 import Memory

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

# --- OpenAI (Mem0 internals + embeddings) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Mem0 (LTM): Qdrant vectors + Neo4j graph ---
MEM0_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": OPENAI_API_KEY,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "mem0_ltm",
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
}

ltm_memory = Memory.from_config(MEM0_CONFIG)

# --- Redis (session history) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SESSION_TTL = 86400          # 24 hours (seconds)
MAX_SESSION_MESSAGES = 40    # rolling window

# --- Qdrant RAG ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
RAG_COLLECTION = "knowledge_base"
EMBED_MODEL = "text-embedding-3-small"
