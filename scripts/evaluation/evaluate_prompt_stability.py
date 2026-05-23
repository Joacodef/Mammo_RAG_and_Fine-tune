# scripts/evaluation/evaluate_prompt_stability.py
import json
import random
import argparse
from pathlib import Path
import yaml

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.evaluation.generate_rag_predictions import format_ner_prompt

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt stability across different exemplar orderings.")
    parser.add_argument("--config-path", type=str, default="configs/rag_ner_config.yaml", help="Path to RAG configuration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ordering.")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. Load config and template
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prompt_template_path = config['rag_prompt']['prompt_template_path']
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # 2. Setup mock clinical examples
    exemplars = [
        {
            "text": "Mamografía bilateral. Mamas con patrón de densidad tipo B. Nódulo denso de 12mm en CSE de mama izquierda.",
            "entities": [
                {"label": "DENS", "start_offset": 36, "end_offset": 42},
                {"label": "HALL_presente", "start_offset": 43, "end_offset": 55},
                {"label": "LAT", "start_offset": 75, "end_offset": 84}
            ]
        },
        {
            "text": "Proyecciones CC y MLO. Calcificaciones benignas bilaterales dispersas. No se observan nódulos.",
            "entities": [
                {"label": "HALL_ausente", "start_offset": 74, "end_offset": 93}
            ]
        },
        {
            "text": "Control evolutivo. Distorsión de la arquitectura en región periareolar derecha. BI-RADS 4.",
            "entities": [
                {"label": "HALL_presente", "start_offset": 19, "end_offset": 47},
                {"label": "REG", "start_offset": 51, "end_offset": 72}
            ]
        }
    ]

    new_report = "Paciente con sospecha clínica. Nódulo espiculado de 15mm en cuadrante superior externo derecho."
    entity_definitions = config['rag_prompt']['entity_labels']

    print("--- Evaluating Prompt Formatting Stability ---")
    print(f"Loaded dynamic prompt template from: {prompt_template_path}")
    print(f"Number of test exemplars: {len(exemplars)}")

    # 3. Generate prompts with different orderings
    orderings = [
        ([0, 1, 2], "Original Order"),
        ([2, 1, 0], "Reversed Order"),
        ([1, 2, 0], "Shuffled Order A"),
        ([0, 2, 1], "Shuffled Order B")
    ]

    prompts = {}
    for indices, name in orderings:
        ordered_examples = [exemplars[i] for i in indices]
        prompt = format_ner_prompt(
            new_report_text=new_report,
            examples=ordered_examples,
            entity_definitions=entity_definitions,
            prompt_template=prompt_template
        )
        prompts[name] = prompt
        print(f"\n[{name}] Prompt Length: {len(prompt)} characters.")
        # Verify placeholders were formatted correctly
        assert "{examples}" not in prompt
        assert "{new_report_text}" not in prompt
        assert "{entity_definitions}" not in prompt
        print("  - Placeholders formatted successfully.")

    # 4. Compare prompt text differences
    # Shuffling order changes text, but structural tokens (e.g. definitions and report texts) should remain exact.
    print("\nVerification of Prompt Structure Integrity:")
    for name, p_text in prompts.items():
        # Verify important text bits are present
        assert "CSE de mama izquierda" in p_text
        assert "Calcificaciones benignas" in p_text
        assert "Distorsión de la arquitectura" in p_text
        assert "HALL_presente" in p_text
        assert "DENS" in p_text
        print(f"  - {name}: verified integrity constraints successfully.")

    print("\nPrompt stability formatting tests completed successfully. All order variations maintain structural parsing contracts.")

if __name__ == '__main__':
    main()
