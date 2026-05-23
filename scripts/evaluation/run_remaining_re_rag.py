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
    # 1. Resume Light Noisy RE RAG prediction run
    print("=================== 2. Resuming Light Noisy RE RAG ===================")
    light_dir = "output/rag_results/re/3-shot/train-50_20260523_130839"
    run_cmd([
        sys.executable, "scripts/evaluation/generate_rag_predictions.py",
        "--config-path", "configs/rag_re_config.yaml",
        "--index-path", "output/vector_db/faiss_index_noisy_light.bin",
        "--source-data-path", "data/processed/train-50/sample-1/train_noisy_light.jsonl",
        "--resume-dir", light_dir
    ])
    
    light_pred = os.path.join(light_dir, "predictions.jsonl")
    
    print("\nEvaluating Light Noisy RE RAG...")
    run_cmd([
        sys.executable, "scripts/evaluation/calculate_final_metrics.py",
        "--prediction-path", light_pred,
        "--type", "re",
        "--config-path", "configs/rag_re_config.yaml"
    ])
    
    # 2. Heavy Noisy RE RAG
    print("\n=================== 3. Running Heavy Noisy RE RAG ===================")
    run_cmd([
        sys.executable, "scripts/evaluation/generate_rag_predictions.py",
        "--config-path", "configs/rag_re_config.yaml",
        "--index-path", "output/vector_db/faiss_index_noisy_heavy.bin",
        "--source-data-path", "data/processed/train-50/sample-1/train_noisy_heavy.jsonl"
    ])
    
    heavy_dir = get_latest_dir("output/rag_results/re/3-shot/*")
    print(f"Detected output directory: {heavy_dir}")
    heavy_pred = os.path.join(heavy_dir, "predictions.jsonl")
    
    print("\nEvaluating Heavy Noisy RE RAG...")
    run_cmd([
        sys.executable, "scripts/evaluation/calculate_final_metrics.py",
        "--prediction-path", heavy_pred,
        "--type", "re",
        "--config-path", "configs/rag_re_config.yaml"
    ])
    
    print("\n=================== All remaining RE RAG runs completed successfully! ===================")

if __name__ == '__main__':
    main()
