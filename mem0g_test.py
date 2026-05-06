import os
from mem0 import Memory

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": "REDACTED_OPENAI_API_KEY"
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "REDACTED_OPENAI_API_KEY"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password123"
        }
    }
}

memory = Memory.from_config(config)

# Xotiraga qo'shish
memory.add("Mening ismim Raximberdi, Xalq Bankida ML Researcher bo'lib ishlayman", user_id="rakh_001")
print("Saqlandi!")

# Qidirish
results = memory.search("ism va ish joyi", user_id="rakh_001")
for r in results["results"]:
    print(f"Topildi: {r['memory']}")