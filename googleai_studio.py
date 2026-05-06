from google import genai

client = genai.Client(api_key="AIzaSyAoyyO9YZNWR8sUIhkUCTErRaoY5A5ceA4")

print("Checking available models...")
try:
    # This will list all models that support 'generateContent'
    for m in client.models.list():
        if 'generateContent' in m.supported_methods:
            print(f"✅ Found: {m.name}")
except Exception as e:
    print(f"❌ API Key Error: {e}")