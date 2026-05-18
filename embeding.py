from openai import OpenAI
from openai import OpenAI


openai_api_key = "sk-raximberdi-cmF4aW1iZXJkaQ"
openai_api_base = "https://ai.xazna.uz/llm/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
print("Mavjud modellar ro'yxati:")
print("-" * 30)

for model in models.data:
    print(f"ID: {model.id}")
model = '/models/embedding'
print(f"\nTanlangan model: {model}")

responses = client.embeddings.create(
    input=[
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models"
    ],
    model=model,
)

for data in responses.data:
    print(data.embedding)  # list of float of len 4096