# Spanish Mammography NER Robustness & Noise Benchmark

This log documents the quantitative impact of training annotation noise on fine-tuned clinical language models vs. structured few-shot RAG (3 exemplars) under different noise regimes.

- **Pre-trained Model**: `"PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"`
- **Dataset Partition**: `train-50` (5-fold cross-validation)
- **Target Task**: Named Entity Recognition (NER)
- **Target Labels**: `HALL_presente`, `HALL_ausente`, `CARACT`, `CUAD`, `LAT`, `REG`, `DENS`
- **Independent Test Set Size**: 91 clinical reports

---

## 1. Quantitative Performance Summary

The table below compiles test set precision, recall, and F1-score averages computed across all 5 cross-validation samples for the fine-tuned models (mean ± standard deviation), alongside the structured 3-shot RAG baseline:

### Fine-Tuned pre-trained Clinical BERT vs. Structured Few-Shot RAG (3-shot)

| Model Type / Regime | Precision (Mean ± SD / Value) | Recall (Mean ± SD / Value) | F1-Score (Mean ± SD / Value) |
| :--- | :---: | :---: | :---: |
| **Fine-Tuned Clinical RoBERTa: Option A (Clean)** | **71.33% ± 3.38%** | **71.07% ± 2.19%** | **70.92% ± 2.73%** |
| **Fine-Tuned Clinical RoBERTa: Option B (Light Noisy)** | 54.37% ± 5.83% | 28.25% ± 2.31% | 35.33% ± 2.93% |
| **Fine-Tuned Clinical RoBERTa: Option C (Heavy Noisy)** | 50.20% ± 4.66% | 25.27% ± 3.65% | 32.44% ± 4.26% |
| **Fine-Tuned BETO Baseline: Option A (Clean)** | *[Pending]* | *[Pending]* | *[Pending]* |
| **Fine-Tuned BETO Baseline: Option B (Light Noisy)** | *[Pending]* | *[Pending]* | *[Pending]* |
| **Fine-Tuned BETO Baseline: Option C (Heavy Noisy)** | *[Pending]* | *[Pending]* | *[Pending]* |
| **RAG LLaMA 3.2: Option A (Clean Baseline)** | **69.70%** | **45.52%** | **53.23%** |
| **RAG LLaMA 3.2: Option B (Light Noisy)** | 2.23% | 0.73% | 0.96% |
| **RAG LLaMA 3.2: Option C (Heavy Noisy)** | 0.67% | 0.43% | 0.44% |
| **RAG GPT-4o: Option A (Clean Baseline)** | *[Pending]* | *[Pending]* | *[Pending]* |
| **RAG GPT-4o: Option B (Light Noisy)** | *[Pending]* | *[Pending]* | *[Pending]* |
| **RAG GPT-4o: Option C (Heavy Noisy)** | *[Pending]* | *[Pending]* | *[Pending]* |

### Regime Specifics & Parameters:
* **Option A (Clean)**: Standard clean annotation training partition under `data/processed/train-50/sample-{1..5}/train.jsonl`
* **Option B (Light Noisy)**: Perturbed dataset containing **5% label swaps**, **offset shifts**, and **10% relation drops** (`train_noisy_light.jsonl`)
* **Option C (Heavy Noisy)**: Perturbed dataset containing **10% label swaps**, **offset shifts**, and **20% relation drops** (`train_noisy_heavy.jsonl`)

---

## 2. Core Takeaways & Practitioner Guidelines

1. **Extreme Few-Shot RAG Sensitivity to Annotation Offset Noise**: Under noisy regimes, few-shot RAG collapses completely (F1 drops from **53.23%** to **0.96%** and **0.44%**). Because the LLM mimics the exact formatting style of the retrieved exemplars, character offset errors/shifts in the prompt examples cause the LLM to output poorly segmented boundaries (e.g. partial words or lines), leading to a catastrophic recall collapse in strict evaluations.
2. **Fine-Tuning Resilience vs. RAG**: While fine-tuned models are affected by noise (F1 dropping from **70.92%** to **35.33%** and **32.44%**), they still maintain significantly higher F1-scores than RAG under noise. Fine-tuning allows the model to capture the semantic structure of entities across 50 reports, mitigating local offset noise, whereas few-shot RAG is direct-copy sensitive to the 3 retrieved exemplars.
3. **Practitioner Recommendation**: In clinical settings where character offset annotations might contain human errors (e.g., highlighting partial tokens), **fine-tuning pre-trained clinical language models** is vastly more robust than few-shot RAG, unless the few-shot RAG vector database has been manually validated and cleaned with 100% precision.

---

## 3. RAG Prompt Order Permutation Diagnostics

To support prompt engineering robustness, we verified that the structural formatting layout parses cleanly under multiple retrieval permutations:
* **Prompt Length**: Constant at exactly **2,762 characters** regardless of exemplar order.
* **Integrity Constraint Checks**: Checked and passed constraints for missing placeholders (`{examples}`, `{new_report_text}`, `{entity_definitions}`) and preserved clinical terms.
* **Result**: **PASS**. All order variations maintain formatting contracts.

---

## 4. Run Directories & Telemetry Logs

### Fine-Tuned Models
* **Option A Model Folder**: `output/models_clinical/ner/train-50_20260521_220748`
* **Option A Predictions**: `output/finetuned_results/ner/train-50_20260521_220748`
* **Option B Model Folder**: `output/models_clinical/ner/train-50_20260521_221822`
* **Option B Predictions**: `output/finetuned_results/ner/train-50_20260521_221822`
* **Option C Model Folder**: `output/models_clinical/ner/train-50_20260521_223558`
* **Option C Predictions**: `output/finetuned_results/ner/train-50_20260521_223558`

### Few-Shot RAG Runs
* **Option A Predictions & Config**: `output/rag_results/ner/3-shot/train-50_20260522_084232`
* **Option B Predictions & Config**: `output/rag_results/ner/3-shot/train-50_20260522_111218`
* **Option C Predictions & Config**: `output/rag_results/ner/3-shot/train-50_20260522_132211`
