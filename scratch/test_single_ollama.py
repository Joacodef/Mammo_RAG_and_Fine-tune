import os
import sys
import yaml
import json
from pathlib import Path

# Add the project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm_services import get_llm_client
from src.vector_db.sentence_embedder import SentenceEmbedder
from src.vector_db.database_manager import DatabaseManager
from scripts.evaluation.generate_rag_predictions import load_test_data, format_re_prompt

def test_single():
    config_path = "configs/rag_re_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Overrides as done in generate_rag_predictions
    index_path = "output/vector_db/faiss_index_clean.bin"
    source_data_path = "data/processed/train-50/sample-1/train.jsonl"
    
    embedder = SentenceEmbedder(model_name=config['vector_db']['embedding_model'])
    db_manager = DatabaseManager(
        embedder=embedder,
        source_data_path=source_data_path,
        index_path=index_path
    )
    db_manager.build_index()
    
    test_records = load_test_data(config['test_file'])
    record = test_records[0]
    
    # Test with 3 examples
    similar_examples = db_manager.search(
        query_text=record['text'],
        top_k=3
    )
    
    prompt_template_path = config['rag_prompt']['prompt_template_path']
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
        
    prompt = format_re_prompt(
        new_report_text=record['text'],
        entities=record['entities'],
        examples=similar_examples,
        relation_definitions=config['rag_prompt']['relation_labels'],
        prompt_template=prompt_template
    )
    
    print("--- CONSTRUCTED PROMPT PREVIEW ---")
    print(prompt[-1500:])
    print("--- END OF PROMPT PREVIEW ---\n")
    
    client = get_llm_client(config_path)
    
    print("Calling Ollama...")
    raw_response = client._call_api(prompt)
    print("--- RAW OLLAMA RESPONSE ---")
    print(repr(raw_response))
    print("--- END OF RAW OLLAMA RESPONSE ---")
    
    parsed = client.get_re_prediction(prompt)
    print("--- PARSED RESPONSE ---")
    print(parsed)

if __name__ == "__main__":
    test_single()
