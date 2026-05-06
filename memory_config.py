from mem0 import Memory
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# xazna_client = OpenAI(
#     base_url="https://ai.xazna.uz/llm/v1",
#     api_key="sk-raximberdi-cmF4aW1iZXJkaQ"
# )

# models = xazna_client.models.list()
# model_id = models.data[0].id

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            # "openai_base_url": "https://ai.xazna.uz/llm/v1",    
            # "api_key": "sk-raximberdi-cmF4aW1iZXJkaQ"        
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