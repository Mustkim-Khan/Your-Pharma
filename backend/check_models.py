import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models_to_test = ["gpt-5.2", "gpt-5-mini"]

print("Testing model access...")

for model in models_to_test:
    try:
        print(f"Testing {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print(f"✅ Success: {model} is accessible.")
    except Exception as e:
        print(f"❌ Failed: {model} - {e}")
