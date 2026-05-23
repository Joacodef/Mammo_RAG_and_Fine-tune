import os
import sys
import subprocess
import glob
import time
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

def run_regime(regime_name, train_file_name):
    print(f"\n==================== Starting BETO NER Training/Evaluation for: {regime_name} ====================")
    
    # 1. Train BETO models for the 5 folds
    run_cmd([
        sys.executable, "scripts/training/run_ner_training.py",
        "--config-path", "configs/training_ner_config.yaml",
        "--partition-dir", "data/processed/train-50",
        "--train-file-name", train_file_name
    ])
    
    # Get the latest run output folder in output/models/ner
    model_dir = get_latest_dir("output/models/ner/*")
    print(f"Detected model output directory: {model_dir}")
    if not model_dir:
        raise FileNotFoundError("Could not find newly trained model folder in output/models/ner")
        
    run_folder_name = os.path.basename(model_dir)
    
    # 2. Generate predictions on test set for all 5 folds
    print(f"\nGenerating predictions for BETO NER - {regime_name}...")
    run_cmd([
        sys.executable, "scripts/evaluation/generate_finetuned_predictions.py",
        "--config-path", "configs/inference_ner_config.yaml",
        "--model-dir", model_dir
    ])
    
    # 3. Calculate final aggregate metrics
    print(f"\nCalculating aggregate metrics for BETO NER - {regime_name}...")
    pred_dir = os.path.join("output/finetuned_results/ner", run_folder_name)
    run_cmd([
        sys.executable, "scripts/evaluation/calculate_final_metrics.py",
        "--prediction-dir", pred_dir,
        "--type", "ner"
    ])
    
    print(f"==================== Completed BETO NER for: {regime_name} ====================\n")

def main():
    start_time = time.time()
    
    # Option A: Clean
    run_regime("Clean (Option A)", "train.jsonl")
    
    # Option B: Light Noisy
    run_regime("Light Noisy (Option B)", "train_noisy_light.jsonl")
    
    # Option C: Heavy Noisy
    run_regime("Heavy Noisy (Option C)", "train_noisy_heavy.jsonl")
    
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print(f"\n==================== All BETO NER training and evaluations completed in {elapsed:.2f} minutes! ====================")

if __name__ == '__main__':
    main()
