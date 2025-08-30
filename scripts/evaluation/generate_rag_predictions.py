import argparse
import yaml
import json
import os
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
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

def format_ner_prompt(new_report_text: str, examples: list, entity_definitions: list, prompt_template: str) -> str:
    """
    Constructs the final few-shot prompt for the NER task.

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
        entity_definitions_str += f"- nombre: \"{entity['name']}\"\n  descripcion: \"{entity['description']}\"\n"

    # Create a set of valid label names for efficient lookup
    valid_labels = {entity['name'] for entity in entity_definitions}

    # Format the few-shot examples into a string
    examples_str = ""
    for ex in examples:
        # We need to format the entities from the example into the expected JSON output format
        formatted_entities = []
        for e in ex.get("entities", []):
            # Only include the entity if its label is in the list of valid labels
            if e["label"] in valid_labels:
                entity_text = ex['text'][e['start_offset']:e['end_offset']]
                formatted_entities.append({"text": entity_text, "label": e["label"], "start_offset": e["start_offset"], "end_offset": e["end_offset"]})

        entities_json_str = json.dumps(formatted_entities, ensure_ascii=False)
        examples_str += f"---\nText: {ex['text']}\nOutput: {entities_json_str}\n"

    # Inject the dynamic content into the template's placeholders
    prompt = prompt_template.format(
        entity_definitions=entity_definitions_str.strip(),
        examples=examples_str.strip(),
        new_report_text=new_report_text
    )
    return prompt


def format_re_prompt(
    new_report_text: str, 
    entities: list, 
    examples: list, 
    relation_definitions: list, 
    prompt_template: str
) -> str:
    """
    Constructs the final few-shot prompt for the RE task.
    """
    # Format the relation definitions into a string
    relation_definitions_str = ""
    for rel in relation_definitions:
        relation_definitions_str += f"- nombre: \"{rel['name']}\"\n  descripcion: \"{rel['description']}\"\n"

    # Format the few-shot examples into a string for RE
    examples_str = ""
    for ex in examples:
        # For RE, examples must show the text, the entities, and the resulting relations
        entities_list = ex.get("entities", [])
        relations_list = ex.get("relations", [])
        
        # We need to add the 'text' of the span to the entity object for the prompt
        formatted_entities = []
        for e in entities_list:
            entity_text = ex['text'][e['start_offset']:e['end_offset']]
            formatted_entities.append({
                "id": e["id"],
                "label": e["label"],
                "text": entity_text
            })

        entities_json_str = json.dumps(formatted_entities, ensure_ascii=False, indent=4)
        relations_json_str = json.dumps(relations_list, ensure_ascii=False, indent=4)
        
        examples_str += (
            f"Texto: {ex['text']}\n"
            f"Entidades:\n{entities_json_str}\n"
            f"Respuesta:\n{relations_json_str}\n---\n"
        )
    
    # Format the entities for the new report into a JSON string
    new_entities_formatted = []
    for e in entities:
        entity_text = new_report_text[e['start_offset']:e['end_offset']]
        new_entities_formatted.append({
            "id": e["id"],
            "label": e["label"],
            "text": entity_text
        })
    entities_json_str_new = json.dumps(new_entities_formatted, ensure_ascii=False, indent=4)

    # Inject the dynamic content into the template's placeholders
    prompt = prompt_template.format(
        relation_definitions=relation_definitions_str.strip(),
        examples=examples_str.strip(),
        new_report_text=new_report_text,
        entities_json=entities_json_str_new
    )
    return prompt

def _run_ner_prediction_loop(records_to_process, all_test_texts, results_file, rag_config, db_manager, llm_client, trace):
    """Handles the prediction loop specifically for the NER task."""
    n_examples_to_retrieve = rag_config.get('n_examples', 3)
    entity_definitions = rag_config.get('entity_labels', [])
    valid_entity_labels = {entity['name'] for entity in entity_definitions}
    prompt_template_path = rag_config.get('prompt_template_path')

    if not prompt_template_path:
        raise ValueError("'prompt_template_path' not found in rag_config.yaml")

    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Open the file in append mode to add new results
    with open(results_file, 'a', encoding='utf-8') as f_out:
        progress_bar = tqdm(records_to_process, desc="Generating RAG NER Predictions")
        for record in progress_bar:
            try:
                report_index = all_test_texts.index(record['text'])
            except ValueError:
                report_index = -1 # Fallback in case the record text isn't found
            if n_examples_to_retrieve > 0:
                similar_examples = db_manager.search(
                    query_text=record['text'],
                    top_k=n_examples_to_retrieve
                )
            else:
                similar_examples = []

            prompt = format_ner_prompt(
                new_report_text=record['text'],
                examples=similar_examples,
                entity_definitions=entity_definitions,
                prompt_template=prompt_template
            )
            
            predicted_entities = llm_client.get_ner_prediction(prompt, trace=trace, report_index=report_index)

            # Validate and Clean LLM Output
            validated_entities = []
            for entity in predicted_entities:
                if isinstance(entity, dict) and entity.get("label") in valid_entity_labels:
                    validated_entities.append(entity)
                else:
                    logging.warning(f"Discarding invalid entity from LLM output: {entity}")

            # Reconstruct true entities
            true_entities_decoded = []
            for entity in record.get("entities", []):
                if entity.get("label") in valid_entity_labels:
                    true_entities_decoded.append({
                        "text": record["text"][entity["start_offset"]:entity["end_offset"]],
                        "label": entity["label"]
                    })

            # Format and write the result immediately
            result_line = {
                "source_text": record['text'],
                "true_entities": true_entities_decoded,
                "predicted_entities": validated_entities,
                "prompt_used": prompt
            }
            f_out.write(json.dumps(result_line, ensure_ascii=False) + '\n')

# scripts/evaluation/generate_rag_predictions.py

def _run_re_prediction_loop(records_to_process, all_test_texts, results_file, rag_config, db_manager, llm_client, trace):
    """Handles the prediction loop specifically for the RE task."""
    n_examples_to_retrieve = rag_config.get('n_examples', 3)
    relation_definitions = rag_config.get('relation_labels', [])
    valid_relation_types = {rel['name'] for rel in relation_definitions}
    prompt_template_path = rag_config.get('prompt_template_path')

    if not prompt_template_path:
        raise ValueError("'prompt_template_path' not found in re_prompt config.")

    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Open the file in append mode to add new results
    with open(results_file, 'a', encoding='utf-8') as f_out:
        progress_bar = tqdm(records_to_process, desc="Generating RAG RE Predictions")
        for record in progress_bar:
            # For RE, we use the ground-truth entities from the test set as input
            entities = record.get("entities", [])
            if not entities:
                logging.warning(f"Skipping record with no entities: {record.get('text', 'N/A')}")
                continue

            try:
                report_index = all_test_texts.index(record['text'])
            except ValueError:
                report_index = -1

            if n_examples_to_retrieve > 0:
                similar_examples = db_manager.search(
                    query_text=record['text'],
                    top_k=n_examples_to_retrieve
                )
            else:
                similar_examples = []

            prompt = format_re_prompt(
                new_report_text=record['text'],
                entities=entities,
                examples=similar_examples,
                relation_definitions=relation_definitions,
                prompt_template=prompt_template
            )
            
            predicted_relations = llm_client.get_re_prediction(prompt, trace=trace, report_index=report_index)

            # Validate the structure of the LLM output
            validated_relations = []
            for rel in predicted_relations:
                if (isinstance(rel, dict) and 
                    "from_id" in rel and 
                    "to_id" in rel and 
                    rel.get("type") in valid_relation_types):
                    validated_relations.append(rel)
                else:
                    logging.warning(f"Discarding invalid relation from LLM output: {rel}")

            # Format and write the result immediately
            result_line = {
                "source_text": record['text'],
                "true_relations": record.get("relations", []),
                "predicted_relations": validated_relations,
                "prompt_used": prompt
            }
            f_out.write(json.dumps(result_line, ensure_ascii=False) + '\n')


def run_predictions(config_path: str, output_dir: Path, trace: Optional[Any]):
    """
    Executes the core prediction generation logic with incremental saving and resume capability.
    
    Args:
        config_path (str): Path to the RAG configuration file.
        output_dir (Path): The exact directory where results will be saved.
        trace (Optional[Any]): The parent Langfuse trace object.
    """
    logging.info("--- Starting RAG Prediction Pipeline ---")

    # --- 1. Load Configuration ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from: {config_path}")

    task = config.get("task")
    if not task or task not in ['ner', 're']:
        raise ValueError("Configuration file must specify a 'task': 'ner' or 're'.")

    rag_config = config.get('rag_prompt', {})
    db_config = config.get('vector_db', {})
    test_file_path = config.get('test_file')
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

    # --- 3. Load Test Data and Handle Resume Logic ---
    all_test_records = load_test_data(test_file_path)
    results_file = output_dir / "predictions.jsonl"
    
    completed_texts = set()
    if results_file.exists():
        logging.info(f"Found existing prediction file at: {results_file}. Attempting to resume.")
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_texts.add(json.loads(line)['source_text'])
                except (json.JSONDecodeError, KeyError):
                    continue # Ignore malformed lines
    
    records_to_process = [rec for rec in all_test_records if rec['text'] not in completed_texts]
    
    if completed_texts:
        logging.info(f"Skipping {len(completed_texts)} already processed records.")
    
    logging.info(f"Total records to process in this run: {len(records_to_process)}.")
    all_test_texts = [rec['text'] for rec in all_test_records]

    # --- 4. Process Each Test Record ---
    if task == 'ner':
        _run_ner_prediction_loop(
            records_to_process=records_to_process,
            all_test_texts=all_test_texts,
            results_file=results_file,
            rag_config=rag_config,
            db_manager=db_manager,
            llm_client=llm_client,
            trace=trace
        )
    elif task == 're':
        _run_re_prediction_loop(
            records_to_process=records_to_process,
            all_test_texts=all_test_texts,
            results_file=results_file,
            rag_config=rag_config,
            db_manager=db_manager,
            llm_client=llm_client,
            trace=trace
        )

    logging.info(f"Prediction generation complete. All results saved to: {results_file}")
    logging.info("--- RAG Prediction Pipeline Finished Successfully ---")


def main(config_path: str, resume_dir: Optional[str] = None):
    """
    Sets up tracing, logging, and output directories, then runs the main prediction pipeline.
    """
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        stream=sys.stdout
    )

    # --- Setup Output Directory ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if resume_dir:
        run_output_dir = Path(resume_dir)
        if not run_output_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        print(f"Resuming RAG prediction run in existing directory: {run_output_dir}")
    else:
        task = config.get("task", "ner")
        base_output_dir = Path(config.get('output_dir', 'output/rag_results'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = base_output_dir / task / timestamp
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting new RAG prediction run. Outputs will be saved in: {run_output_dir}")
        # Copy the config only for new runs to avoid overwriting
        shutil.copy(config_path, run_output_dir / "rag_config.yaml")

    # --- Langfuse Tracing (Optional) ---
    langfuse_client = None
    if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
        logging.info("Langfuse environment variables found. Initializing Langfuse client.")
        langfuse_client = Langfuse()
        # Extract details for a more descriptive trace name
        n_examples = config.get('rag_prompt', {}).get('n_examples', 0)
        source_data_path = Path(config.get('vector_db', {}).get('source_data_path', ''))

        # Extract the parent directory name (e.g., 'train-5', 'train-100')
        db_size_name = source_data_path.parent.parent.name if source_data_path else "unknown_db"

        shot_type = f"{n_examples}-shot"

        trace_name = f"RAG Prediction Run - {db_size_name} - {shot_type}"

        trace_metadata = {
            "db_size": db_size_name,
            "shot_type": shot_type,
            "n_examples": n_examples,
            "config_path": config_path
        }
    else:
        logging.info("Langfuse environment variables not set. Proceeding without Langfuse tracing.")

    if langfuse_client:
        # Create a single trace that encompasses the entire script run
        with langfuse_client.start_as_current_span(name=trace_name, metadata=trace_metadata) as trace:
            run_predictions(config_path, run_output_dir, trace)
        # Ensure all buffered data is sent before the script exits
        langfuse_client.flush()
    else:
        # Execute the main logic without a parent trace
        run_predictions(config_path, run_output_dir, None)

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
    
    parser.add_argument(
        '--resume-dir',
        type=str,
        default=None,
        help='Path to an existing run directory to resume an interrupted prediction job.'
    )
    
    args = parser.parse_args()
    main(args.config_path, args.resume_dir)