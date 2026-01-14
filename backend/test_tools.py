import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

start_model = "gpt-5.2"

tool = {
    "type": "function",
    "function": {
        "name": "say_hello",
        "description": "Say hello",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    }
}

try:
    print(f"Testing {start_model} with tools...")
    response = client.chat.completions.create(
        model=start_model,
        messages=[{"role": "user", "content": "Call the function with name=World"}],
        tools=[tool]
    )
    print(f"✅ Success: {start_model} supports tools.")
    print(response.choices[0].message.tool_calls)
except Exception as e:
    print(f"❌ Failed: {start_model} - {e}")

start_model = "gpt-5-mini"
try:
    print(f"Testing {start_model} with tools...")
    response = client.chat.completions.create(
        model=start_model,
        messages=[{"role": "user", "content": "Call the function with name=World"}],
        tools=[tool]
    )
    print(f"✅ Success: {start_model} supports tools.")
    print(response.choices[0].message.tool_calls)
except Exception as e:
    print(f"❌ Failed: {start_model} - {e}")
