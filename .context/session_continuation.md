# Session Continuation & Execution Handover

This document summarizes the exact active background processes, completed milestones, and step-by-step instructions to resume our Mammo NLP robustness and baseline experiments in a new chat thread.

---

## 1. Active Background Processes

These processes are currently running asynchronously on your machine and will not be interrupted by starting a new chat thread.

### Task 1: LLaMA 3.2 3B Few-Shot RE RAG Suite
* **Task ID**: `3ec80d6a-8be4-4cd4-8e52-c8bcf01a5d80/task-1386` (or system process ID `9072` / `18392`)
* **Status**: **IN PROGRESS** (Generating sequential predictions for Clean, Light Noisy, and Heavy Noisy regimes)
* **Progress**: Currently at **~79/91 predictions** for the Clean run.
* **Storage Folder**: `output/rag_results/re/3-shot/train-50_20260522_232206`
* **Telemetry Log**: `C:\Users\joaco\.gemini\antigravity-ide\brain\3ec80d6a-8be4-4cd4-8e52-c8bcf01a5d80\.system_generated\tasks\task-1386.log`
* **Command**:
  ```bash
  uv run python scripts/evaluation/run_all_re_rag.py
  ```
* **Notes**: Ollama has successfully transitioned to local GPU acceleration now that the supervised training has completed! Inference has sped up 4.7x (from 7.5 min to **~1.6 min per report**). It will complete the Clean run in ~18 minutes, then proceed sequentially to Light and Heavy Noisy runs.

---

## 2. Completed Milestones

### Named Entity Recognition (NER)
* **Fine-Tuning (Clinical RoBERTa)**: **100% COMPLETE** (Clean, Light, Heavy regimes evaluated and logged).
* **RAG LLaMA 3.2 3B**: **100% COMPLETE** (Clean, Light, Heavy regimes evaluated and logged).
* **NER Robustness Log**: Fully updated in [robustness_benchmark_ner.md](file:///e:/Mammo_RAG_and_Fine-tune/experiments/robustness_benchmark_ner.md).

### Relation Extraction (RE)
* **Supervised Clinical RoBERTa Option A (Clean)**: **100% COMPLETE** (Precision: **98.13% ± 0.56%**, Recall: **98.95% ± 0.59%**, F1: **98.54% ± 0.41%**).
* **Supervised Clinical RoBERTa Option B (Light Noisy)**: **100% COMPLETE** (Precision: **94.96% ± 2.58%**, Recall: **74.21% ± 18.82%**, F1: **81.13% ± 16.07%**).
* **Supervised Clinical RoBERTa Option C (Heavy Noisy)**: **100% COMPLETE** (Precision: **95.67% ± 2.47%**, Recall: **60.75% ± 23.75%**, F1: **71.54% ± 20.23%**).
  * **Model Weights Directory**: `output/models_clinical/re/train-50_20260523_054829`
  * **Predictions Directory**: `output/finetuned_results/re/train-50_20260523_054829`
* **RE Robustness Log**: Fully updated with Supervised options A, B, and C and primary guidelines in [robustness_benchmark_re.md](file:///e:/Mammo_RAG_and_Fine-tune/experiments/robustness_benchmark_re.md).

---

## 3. Resuming Steps in the Next Session

When you start the new thread, copy and paste the handover prompt to direct the agent on the exact outstanding tasks:

### Step 3.1: Monitor & Evaluate LLaMA 3.2 RE RAG
1. Confirm when `task-1386` completes its run (sequential clean, light, and heavy).
2. Once complete, collect the metrics from:
   * Clean: `output/rag_results/re/3-shot/train-50_20260522_232206/final_metrics_predictions.json`
   * Light: `output/rag_results/re/3-shot/<latest_light_noisy>/final_metrics_predictions.json`
   * Heavy: `output/rag_results/re/3-shot/<latest_heavy_noisy>/final_metrics_predictions.json`
3. Log these RAG LLaMA metrics in [robustness_benchmark_re.md](file:///e:/Mammo_RAG_and_Fine-tune/experiments/robustness_benchmark_re.md).

### Step 3.2: Address GPT-4o API Key Quota Block
1. The user's OpenAI API key has run out of funds/quota (`Error code: 429 - insufficient_quota`). 
2. Ask the user if they have completed the credit top-up at [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing).
3. Once the quota block is confirmed resolved:
   * Delete the failed/partial GPT folders:
     * `output/rag_results/ner/3-shot/train-50_20260522_175708`
     * `output/rag_results/ner/3-shot/train-50_20260522_180016`
   * Re-run both GPT-4o RAG suites (NER and RE) across all regimes:
     ```bash
     uv run python scripts/evaluation/run_all_ner_rag_gpt.py
     uv run python scripts/evaluation/run_all_re_rag_gpt.py
     ```
   * Log the final RAG GPT-4o scores in both [robustness_benchmark_ner.md](file:///e:/Mammo_RAG_and_Fine-tune/experiments/robustness_benchmark_ner.md) and [robustness_benchmark_re.md](file:///e:/Mammo_RAG_and_Fine-tune/experiments/robustness_benchmark_re.md).
