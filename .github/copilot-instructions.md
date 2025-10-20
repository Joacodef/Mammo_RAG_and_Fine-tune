## Purpose

Short guidance for AI coding agents working on the Mammo_RAG_and_Fine-tune repository. Focus on the architecture, core workflows, conventions, and quick examples so an agent can be immediately productive.

## Big-picture architecture (what to know)

- Two main experiment paradigms: fine-tuning (NER and RE) and RAG (retrieval + LLM prompting). See `README.md` for an overview.
- Major components:
  - Data & partitions: `data/raw/all.jsonl` -> `scripts/data/generate_partitions.py` -> `data/processed/train-*/sample-*/train.jsonl`
  - Vector DB / RAG: `scripts/data/build_vector_db.py` and `src/vector_db/` (FAISS index used for retrieval)
  - LLM providers: `src/llm_services/` implements a small adapter layer (look at `base_client.py`, `openai_client.py`, `ollama_client.py`)
  - Models: `src/models/ner_bert.py` and `src/models/re_model.py` contain model definitions and fine-tuning hooks
  - Training loop: `scripts/training/*` call into `src/training/trainer.py`
  - Evaluation: `scripts/evaluation/*` and `src/evaluation/predictor.py` handle predictions and metrics

## Developer workflows (explicit commands)

- Setup (Python 3.11+):
  - Create environment and install: `pip install -r requirements.txt`
  - Copy example env file: `cp .env_example .env` and populate keys (e.g. `OPENAI_API_KEY`) when using API clients

- Common experiment steps (minimal reproducible flow):
  1. Generate partitions: `python scripts/data/generate_partitions.py --config-path configs/data_preparation_config.yaml`
  2. Build vector DB for RAG: `python scripts/data/build_vector_db.py --config-path configs/rag_config.yaml`
  3. Run NER training: `python scripts/training/run_ner_training.py --config-path configs/training_ner_config.yaml --partition-dir data/processed/train-50`
  4. Generate RAG predictions: `python scripts/evaluation/generate_rag_predictions.py --config-path configs/rag_config.yaml`
  5. Calculate metrics: `python scripts/evaluation/calculate_final_metrics.py --prediction-path <pred.jsonl> --type ner --output-path <metrics.json>`

## Project-specific conventions and patterns

- Config-driven experiments: most scripts take `--config-path` pointing to YAML files under `configs/` (there is also `my_configs/` for user overrides). Prefer changes in YAML over changing scripts.
- Partition sampling: training data is arranged under `data/processed/train-*/sample-*` to support repeated experiments. Code expects the partition directory structure.
- Prompt templates: RAG prompts live in `prompts/` (Spanish prompts and strict variants are present; e.g. `rag_ner_prompt_spanish.txt`). Use these when editing RAG behavior.
- Unified LLM client interface: add new LLM providers by implementing `BaseClient` in `src/llm_services/base_client.py` and wiring them like `openai_client.py`.
- Cost tracking: API costs are tracked via `src/utils/cost_tracker.py`; when adding API calls ensure costs are reported similarly.

## Integration points & external dependencies

- External LLMs: OpenAI (via `openai_client.py`) and Ollama (`ollama_client.py`). API keys go in `.env`.
- Vector DB backend: FAISS (created by `build_vector_db.py`); inspect `src/vector_db/` for indexing and retrieval code.
- Outputs and checkpoints: `output/models/ner/`, `output/models/re/`, `output/rag_results/` and logs under `output/finetuned_results/` — follow this structure when saving artifacts.

## Files to inspect for common tasks (examples)

- Add new LLM provider: inspect `src/llm_services/base_client.py`, mirror `openai_client.py`.
- Change model training loop: check `src/training/trainer.py` and the wrapper scripts in `scripts/training/`.
- Alter evaluation metrics or prediction format: look at `src/evaluation/predictor.py` and `scripts/evaluation/calculate_final_metrics.py`.
- Modify vector indexing: `scripts/data/build_vector_db.py` + `src/vector_db/`.

## When making edits — concrete rules for the agent

- Prefer config changes over code changes for experiments. If adding CLI flags, update the corresponding `scripts/*` and their README lines.
- Keep data partitioning format unchanged: scripts and evaluation assume `train.jsonl` under sample folders.
- Tests: run `python -m pytest` locally before opening PR; CI runs unit tests on pushes to `main`.
- Small docs or helper scripts are fine to add; avoid large refactors without an accompanying test plan and smoke test results.

## Quick-check list before creating a PR

- Run unit tests: `python -m pytest -q`, though you need to be in an environment with dependencies installed
- Ensure `.env` secrets are not committed
- Follow output paths under `output/` and do not commit large model files

## If something's unclear, start here

- Read `README.md` (top-level) for the experiment flow
- Inspect `src/llm_services/` and `prompts/` to understand RAG prompting
- Open `scripts/data/generate_partitions.py` to learn how samples are created and named

## Example input report

{"id":"1.2.840.113619.2.373.202306131802469930109650","text":"Ambas mamas son densas y heterogéneas.\nMicrocalcificaciones aisladas.\nNódulo periareolar derecho bien delimitadp de 10mm.\nNódulo calcifcado derecho.\nNo observo microcalcificaciones sospechosas agrupadas ni imágenes espiculadas.\nRegiones axilares sin adenopatías.\nImpresión: Mamas densas y nódulo derecho presuntamente benigno.\nSugiero ecografía mamaria.\nBI-RADS 3 ACR C","Comments":[],"entities":[{"id":4762,"label":"DENS","start_offset":0,"end_offset":37},{"id":4763,"label":"HALL_presente","start_offset":39,"end_offset":59},{"id":4764,"label":"CARACT","start_offset":60,"end_offset":68},{"id":4765,"label":"HALL_presente","start_offset":70,"end_offset":76},{"id":4766,"label":"HALL_presente","start_offset":122,"end_offset":128},{"id":4767,"label":"HALL_ausente","start_offset":160,"end_offset":180},{"id":4769,"label":"HALL_presente","start_offset":289,"end_offset":295},{"id":4770,"label":"REG","start_offset":77,"end_offset":88},{"id":4771,"label":"LAT","start_offset":89,"end_offset":96},{"id":4772,"label":"CARACT","start_offset":97,"end_offset":112},{"id":4773,"label":"CARACT","start_offset":116,"end_offset":120},{"id":4774,"label":"CARACT","start_offset":129,"end_offset":139},{"id":4775,"label":"LAT","start_offset":140,"end_offset":147},{"id":4776,"label":"CARACT","start_offset":181,"end_offset":192},{"id":4777,"label":"CARACT","start_offset":193,"end_offset":202},{"id":4778,"label":"HALL_ausente","start_offset":206,"end_offset":226},{"id":4779,"label":"DENS","start_offset":274,"end_offset":286},{"id":4780,"label":"LAT","start_offset":296,"end_offset":303},{"id":4781,"label":"CARACT","start_offset":304,"end_offset":325},{"id":13446,"label":"HALL_ausente","start_offset":250,"end_offset":261},{"id":13447,"label":"REG","start_offset":228,"end_offset":245}],"relations":[{"id":22,"from_id":4764,"to_id":4763,"type":"describir"},{"id":23,"from_id":4772,"to_id":4765,"type":"describir"},{"id":24,"from_id":4773,"to_id":4765,"type":"describir"},{"id":25,"from_id":4774,"to_id":4766,"type":"describir"},{"id":26,"from_id":4776,"to_id":4767,"type":"describir"},{"id":27,"from_id":4777,"to_id":4767,"type":"describir"},{"id":29,"from_id":4770,"to_id":4765,"type":"ubicar"},{"id":30,"from_id":4775,"to_id":4766,"type":"ubicar"},{"id":31,"from_id":4780,"to_id":4769,"type":"ubicar"},{"id":32,"from_id":4781,"to_id":4769,"type":"describir"},{"id":5254,"from_id":13447,"to_id":13446,"type":"ubicar"}]}


## Example NER predicted entities (not the entire output)

"predicted_entities": [{"label": "DENS", "start_offset": 0, "end_offset": 38, "text": "Ambas mamas son densas y heterogéneas."}, {"label": "HALL_presente", "start_offset": 39, "end_offset": 69, "text": "Microcalcificaciones aisladas."}, {"label": "HALL_ausente", "start_offset": 149, "end_offset": 227, "text": "No observo microcalcificaciones sospechosas agrupadas ni imágenes espiculadas."}, {"label": "REG", "start_offset": 228, "end_offset": 262, "text": "Regiones axilares sin adenopatías."}, {"label": "DENS", "start_offset": 263, "end_offset": 326, "text": "Impresión: Mamas densas y nódulo derecho presuntamente benigno."}, {"label": "HALL_presente", "start_offset": 327, "end_offset": 353, "text": "Sugiero ecografía mamaria."}, {"label": "HALL_presente", "start_offset": 354, "end_offset": 369, "text": "BI-RADS 3 ACR C"}]


## Example RE predicted relations (not the entire output)

"predicted_relations": [{"from_id": 4764, "to_id": 4763, "type": "describir"}, {"from_id": 4770, "to_id": 4769, "type": "ubicar"}, {"from_id": 4771, "to_id": 4769, "type": "ubicar"}, {"from_id": 4772, "to_id": 4765, "type": "describir"}, {"from_id": 4773, "to_id": 4765, "type": "describir"}, {"from_id": 4774, "to_id": 4766, "type": "describir"}, {"from_id": 4775, "to_id": 4766, "type": "ubicar"}, {"from_id": 4776, "to_id": 4767, "type": "describir"}, {"from_id": 4777, "to_id": 4767, "type": "describir"}, {"from_id": 13447, "to_id": 13446, "type": "ubicar"}, {"from_id": 4780, "to_id": 4769, "type": "ubicar"}, {"from_id": 4781, "to_id": 4769, "type": "describir"}]