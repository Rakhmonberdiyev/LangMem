from openai import OpenAI

client = OpenAI( 
    base_url="https://ai.xazna.uz/llm/v1",
    api_key="sk-raximberdi-cmF4aW1iZXJkaQ"
)


models = client.models.list()

model_id = models.data[0].id
# model_id = "Qwen3-32B"
print(f"Available models: {len(models.data)}, using the first one: {model_id}")

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Give short answer. What is the capital of France?   "}
    ],
    stream=False,
    temperature=0.7,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

print(response.choices[0].message.content)