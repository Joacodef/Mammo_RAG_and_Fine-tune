import os
import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = "E:\\Mammo_RAG_and_Fine-tune\\.env"
load_dotenv(dotenv_path=env_path)

sys.path.append(str(Path(__file__).parent.parent))

from src.llm_services import get_llm_client

try:
    print(f"API KEY: {os.getenv('OPENAI_API_KEY')[:15]}...")
    client = get_llm_client(config_path="configs/rag_ner_gpt_config.yaml")
    prompt = "Dame una lista de entidades en formato JSON: {\"entities\": []}"
    res = client.get_ner_prediction(prompt)
    print(f"Result: {res}")
except Exception as e:
    import traceback
    traceback.print_exc()

