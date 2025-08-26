import argparse
import yaml
import json
import os
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from langfuse import Langfuse
from typing import Any, Optional

# Add the project root to the Python path to allow for absolute imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_services import get_llm_client
from src.vector_db.sentence_embedder import SentenceEmbedder
from src.vector_db.database_manager import DatabaseManager

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

def run_predictions(config_path: str, trace: Optional[Any]):
    """
    Executes the core prediction generation logic. This function is designed
    to be called from within a Langfuse trace context.
    
    Args:
        config_path (str): Path to the RAG configuration file.
        trace (Optional[Any]): The parent Langfuse trace object. If None,
                               tracing for nested generations is skipped.
    """
    logging.info("--- Starting RAG Prediction Pipeline ---")

    # --- 1. Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from: {config_path}")

    rag_config = config.get('rag_prompt', {})
    db_config = config.get('vector_db', {})
    test_file_path = config.get('test_file', 'data/processed/test.jsonl')
    prompt_template_path = rag_config.get('prompt_template_path')

    if not prompt_template_path:
        raise ValueError("'prompt_template_path' not found in rag_config.yaml")

    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # --- 2. Initialize Components ---
    logging.info("Initializing SentenceEmbedder...")
    embedder = SentenceEmbedder(model_name=db_config['embedding_model'])
    
    logging.info("Initializing DatabaseManager...")
    db_manager = DatabaseManager(
        embedder=embedder,
        source_data_path=db_config['source_data_path'],
        index_path=db_config['index_path']
    )
    logging.info("Building or loading vector database index...")
    db_manager.build_index()
    logging.info("Vector database index is ready.")

    logging.info("Initializing LLM client...")
    llm_client = get_llm_client(config_path=config_path)
    logging.info("All components initialized successfully.")

    # --- 3. Load Test Data ---
    test_records = load_test_data(test_file_path)
    logging.info(f"Loaded {len(test_records)} records for prediction from {test_file_path}.")

    # --- 4. Process Each Test Record ---
    results = []
    n_examples_to_retrieve = rag_config.get('n_examples', 3)
    entity_definitions = rag_config.get('entity_labels', [])

    progress_bar = tqdm(test_records, desc="Generating Predictions")
    for i, record in enumerate(test_records):
        logging.info(f"Processing record {i+1}/{len(test_records)}.")
        
        logging.info("Searching for similar examples in the vector database.")
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

        logging.info("Sending prompt to LLM for prediction...")
        predicted_entities = llm_client.get_ner_prediction(prompt, trace=trace)
        logging.info("Received prediction from LLM.")

        results.append({
            "source_text": record['text'],
            "true_entities": record.get('entities', []),
            "predicted_entities": predicted_entities,
            "prompt_used": prompt
        })

    # --- 5. Save Results ---
    output_dir = Path(config.get('output_dir', 'output/rag_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "rag_predictions.jsonl"
    
    logging.info(f"Saving {len(results)} results to {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    logging.info(f"Prediction generation complete. Predictions saved to: {results_file}")
    logging.info("--- RAG Prediction Pipeline Finished Successfully ---")

def main(config_path: str):
    """
    Sets up tracing and logging, then runs the main prediction pipeline.
    """
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        stream=sys.stdout
    )

    # --- Langfuse Tracing (Optional) ---
    langfuse_client = None
    if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
        logging.info("Langfuse environment variables found. Initializing Langfuse client.")
        langfuse_client = Langfuse()
    else:
        logging.info("Langfuse environment variables not set. Proceeding without Langfuse tracing.")

    if langfuse_client:
        # Create a single trace that encompasses the entire script run
        with langfuse_client.start_as_current_span(name="RAG_Prediction_Run") as trace:
            run_predictions(config_path, trace)
        # Ensure all buffered data is sent before the script exits
        langfuse_client.flush()
    else:
        # Execute the main logic without a parent trace
        run_predictions(config_path, None)

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