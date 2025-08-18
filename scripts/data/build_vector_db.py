import argparse
import yaml
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.vector_db.sentence_embedder import SentenceEmbedder
from src.vector_db.database_manager import DatabaseManager

def main(config_path: str, force_rebuild: bool):
    """
    Main function to build the vector database.

    This function loads the configuration, initializes the necessary components
    (SentenceEmbedder, DatabaseManager), and triggers the index build process.

    Args:
        config_path (str): The path to the RAG configuration YAML file.
        force_rebuild (bool): If True, forces the index to be rebuilt even if
                              an existing index file is found.
    """
    print("--- Starting Vector Database Build Process ---")
    
    # --- 1. Load Configuration ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        return

    db_config = config.get('vector_db', {})
    embedding_model = db_config.get('embedding_model')
    source_data_path = db_config.get('source_data_path')
    index_path = db_config.get('index_path')

    if not all([embedding_model, source_data_path, index_path]):
        print("Error: Missing required keys ('embedding_model', 'source_data_path', 'index_path') "
              "in the 'vector_db' section of the config.")
        return

    # --- 2. Initialize Components ---
    embedder = SentenceEmbedder(model_name=embedding_model)
    db_manager = DatabaseManager(
        embedder=embedder,
        source_data_path=source_data_path,
        index_path=index_path
    )

    # --- 3. Build and Save the Index ---
    db_manager.build_index(force_rebuild=force_rebuild)

    print("\n--- Vector Database Build Process Finished Successfully ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build and save a FAISS vector database for RAG."
    )
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        default='configs/rag_config.yaml',
        help='Path to the RAG configuration YAML file.'
    )
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='If set, the index will be rebuilt even if it already exists.'
    )
    
    args = parser.parse_args()
    
    main(args.config_path, args.force_rebuild)