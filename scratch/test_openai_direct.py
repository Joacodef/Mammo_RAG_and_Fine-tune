import os
from dotenv import load_dotenv
from openai import OpenAI

env_path = "E:\\Mammo_RAG_and_Fine-tune\\.env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key: {api_key[:15]}...")

client = OpenAI(api_key=api_key)
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to return JSON."},
            {"role": "user", "content": "Dame una lista de entidades en formato JSON: {\"entities\": []}"}
        ],
        response_format={"type": "json_object"}
    )
    print("Success!")
    print(f"Response: {response}")
    print(f"Content: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
