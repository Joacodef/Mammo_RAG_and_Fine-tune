import os
import sys
import subprocess
import glob
from pathlib import Path

def get_latest_dir(pattern):
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def run_cmd(args):
    print(f"\nRunning command: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error executing command. Code: {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    else:
        print("Command completed successfully.")
        print(result.stdout)
    return result.stdout

def main():
    # 1. Clean NER RAG
    print("=================== 1. Running Clean NER RAG (GPT-4o) ===================")
    run_cmd([
        sys.executable, "scripts/evaluation/generate_rag_predictions.py",
        "--config-path", "configs/rag_ner_gpt_config.yaml",
        "--index-path", "output/vector_db/faiss_index_clean.bin",
        "--source-data-path", "data/processed/train-50/sample-1/train.jsonl"
    ])
    
    clean_dir = get_latest_dir("output/rag_results/ner/3-shot/*")
    print(f"Detected output directory: {clean_dir}")
    clean_pred = os.path.join(clean_dir, "predictions.jsonl")
    
    print("\nEvaluating Clean NER RAG (GPT-4o)...")
    run_cmd([
        sys.executable, "scripts/evaluation/calculate_final_metrics.py",
        "--prediction-path", clean_pred,
        "--type", "ner",
        "--config-path", "configs/rag_ner_gpt_config.yaml"
    ])
    
    # 2. Light Noisy NER RAG
    print("\n=================== 2. Running Light Noisy NER RAG (GPT-4o) ===================")
    run_cmd([
        sys.executable, "scripts/evaluation/generate_rag_predictions.py",
        "--config-path", "configs/rag_ner_gpt_config.yaml",
        "--index-path", "output/vector_db/faiss_index_noisy_light.bin",
        "--source-data-path", "data/processed/train-50/sample-1/train_noisy_light.jsonl"
    ])
    
    light_dir = get_latest_dir("output/rag_results/ner/3-shot/*")
    print(f"Detected output directory: {light_dir}")
    light_pred = os.path.join(light_dir, "predictions.jsonl")
    
    print("\nEvaluating Light Noisy NER RAG (GPT-4o)...")
    run_cmd([
        sys.executable, "scripts/evaluation/calculate_final_metrics.py",
        "--prediction-path", light_pred,
        "--type", "ner",
        "--config-path", "configs/rag_ner_gpt_config.yaml"
    ])
    
    # 3. Heavy Noisy NER RAG
    print("\n=================== 3. Running Heavy Noisy NER RAG (GPT-4o) ===================")
    run_cmd([
        sys.executable, "scripts/evaluation/generate_rag_predictions.py",
        "--config-path", "configs/rag_ner_gpt_config.yaml",
        "--index-path", "output/vector_db/faiss_index_noisy_heavy.bin",
        "--source-data-path", "data/processed/train-50/sample-1/train_noisy_heavy.jsonl"
    ])
    
    heavy_dir = get_latest_dir("output/rag_results/ner/3-shot/*")
    print(f"Detected output directory: {heavy_dir}")
    heavy_pred = os.path.join(heavy_dir, "predictions.jsonl")
    
    print("\nEvaluating Heavy Noisy NER RAG (GPT-4o)...")
    run_cmd([
        sys.executable, "scripts/evaluation/calculate_final_metrics.py",
        "--prediction-path", heavy_pred,
        "--type", "ner",
        "--config-path", "configs/rag_ner_gpt_config.yaml"
    ])
    
    print("\n=================== All NER RAG (GPT-4o) prediction and evaluation runs completed successfully! ===================")

if __name__ == '__main__':
    main()
