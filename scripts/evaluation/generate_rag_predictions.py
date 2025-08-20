import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add the project root to the Python path to allow for absolute imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_services import get_llm_client
from src.vector_db.sentence_embedder import SentenceEmbedder
from src.vector_db.database_manager import DatabaseManager
from src.utils.cost_tracker import CostTracker

def load_test_data(file_path: str) -> list:
    """Loads records from a .jsonl test file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def format_prompt(new_report_text: str, examples: list, entity_definitions: list, prompt_template: str) -> str:
    """
    Constructs the final few-shot prompt for the LLM using a template file.

    Args:
        new_report_text (str): The text of the new mammogram report to analyze.
        examples (list): A list of annotated records retrieved from the vector DB.
        entity_definitions (list): A list of dictionaries defining each entity.
        prompt_template (str): The loaded prompt template string.

    Returns:
        str: The fully formatted, detailed prompt.
    """
    # Format the entity definitions into a string
    entity_definitions_str = ""
    for entity in entity_definitions:
        entity_definitions_str += f"- name: \"{entity['name']}\"\n  description: \"{entity['description']}\"\n"

    # Format the few-shot examples into a string
    examples_str = ""
    for ex in examples:
        # We need to format the entities from the example into the expected JSON output format
        formatted_entities = []
        for e in ex.get("entities", []):
            entity_text = ex['text'][e['start_offset']:e['end_offset']]
            formatted_entities.append({"text": entity_text, "label": e["label"]})
        
        entities_json_str = json.dumps(formatted_entities, ensure_ascii=False)
        examples_str += f"---\nText: {ex['text']}\nOutput: {entities_json_str}\n"

    # Inject the dynamic content into the template's placeholders
    prompt = prompt_template.format(
        entity_definitions=entity_definitions_str.strip(),
        examples=examples_str.strip(),
        new_report_text=new_report_text
    )

    return prompt


def main(config_path: str):
    """
    Main function to run the end-to-end RAG prediction pipeline.
    """
    load_dotenv()
    
    print("--- Starting RAG Prediction Pipeline ---")

    # --- 1. Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from: {config_path}")

    rag_config = config.get('rag_prompt', {})
    db_config = config.get('vector_db', {})
    paths_config = config.get('paths', {})
    test_file_path = paths_config.get('test_file', 'data/processed/test.jsonl')
    prompt_template_path = rag_config.get('prompt_template_path')

    if not prompt_template_path:
        raise ValueError("'prompt_template_path' not found in rag_config.yaml")

    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # --- 2. Initialize Components ---
    print("Initializing components...")
    cost_tracker = CostTracker() # Initialize the cost tracker
    embedder = SentenceEmbedder(model_name=db_config['embedding_model'])
    db_manager = DatabaseManager(
        embedder=embedder,
        source_data_path=db_config['source_data_path'],
        index_path=db_config['index_path']
    )
    db_manager.build_index() # Loads if exists, builds if not

    # Pass the cost_tracker instance to the factory
    llm_client = get_llm_client(config_path, cost_tracker=cost_tracker)
    print("All components initialized successfully.")

    # --- 3. Load Test Data ---
    test_records = load_test_data(test_file_path)
    print(f"Loaded {len(test_records)} records for prediction from {test_file_path}.")

    # --- 4. Process Each Test Record ---
    results = []
    n_examples_to_retrieve = rag_config.get('n_examples', 3)
    entity_definitions = rag_config.get('entity_labels', [])

    progress_bar = tqdm(test_records, desc="Generating Predictions")
    for record in progress_bar:
        similar_examples = db_manager.search(
            query_text=record['text'],
            top_k=n_examples_to_retrieve
        )

        prompt = format_prompt(
            new_report_text=record['text'],
            examples=similar_examples,
            entity_definitions=entity_definitions,
            prompt_template=prompt_template
        )

        predicted_entities = llm_client.get_ner_prediction(prompt)

        results.append({
            "source_text": record['text'],
            "true_entities": record.get('entities', []),
            "predicted_entities": predicted_entities,
            "prompt_used": prompt
        })

    # --- 5. Save Results ---
    output_dir = Path(paths_config.get('output_dir', 'output/rag_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "rag_predictions.jsonl"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"\nPrediction generation complete. Predictions saved to: {results_file}")
    
    # --- 6. Save the Cost and Usage Log ---
    cost_tracker.save_log()
    
    print("--- RAG Prediction Pipeline Finished Successfully ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a full RAG-based prediction pipeline for NER."
    )
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        default='configs/rag_config.yaml',
        help='Path to the RAG configuration YAML file.'
    )
    
    args = parser.parse_args()
    main(args.config_path)