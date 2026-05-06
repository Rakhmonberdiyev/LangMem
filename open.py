from openai import OpenAI

client = OpenAI(
    api_key="REDACTED_OPENAI_API_KEY"
)


models = client.models.list()

model_id = models.data[0].id
print(f"Available models: {len(models.data)}, using the first one: {model_id}")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Give short answer. What is the capital of France?   "}
    ],
    stream=False,
    temperature=0.7,
)

print(response.choices[0].message.content)