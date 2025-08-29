# scripts/evaluation/postprocess_rag_predictions.py
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).parent.parent.parent))

def find_nearest_match(substring, text, start_hint):
    """
    Finds the best match for a substring in a text, starting near a hint index.

    Args:
        substring (str): The entity text to search for.
        text (str): The source text to search within.
        start_hint (int): The original, possibly inaccurate, start offset.

    Returns:
        tuple: A tuple containing the corrected (start_offset, end_offset),
               or (None, None) if no match is found.
    """
    try:
        # Search forward from the hint
        forward_pos = text.find(substring, start_hint)
        # Search backward from the hint
        backward_pos = text.rfind(substring, 0, start_hint + len(substring))

        positions = []
        if forward_pos != -1:
            positions.append(forward_pos)
        if backward_pos != -1:
            positions.append(backward_pos)

        if not positions:
            return None, None

        # Find the position closest to the original hint
        best_pos = min(positions, key=lambda p: abs(p - start_hint))
        return best_pos, best_pos + len(substring)

    except Exception:
        return None, None


def postprocess_predictions(input_path: str, output_path: str):
    """
    Reads a RAG prediction file, corrects the entity offsets, and saves the result.

    Args:
        input_path (str): Path to the .jsonl file with raw RAG predictions.
        output_path (str): Path to save the post-processed .jsonl file.
    """
    logging.info(f"Reading predictions from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    corrected_records = []
    total_entities = 0
    corrections_made = 0
    failed_corrections = 0

    progress_bar = tqdm(records, desc="Post-processing predictions")
    for record in progress_bar:
        source_text = record.get("source_text", "")
        predicted_entities = record.get("predicted_entities", [])
        corrected_entities = []
        total_entities += len(predicted_entities)

        for entity in predicted_entities:
            entity_text = entity.get("text")
            original_start = entity.get("start_offset")

            if not all([isinstance(entity_text, str), isinstance(original_start, int)]):
                failed_corrections += 1
                continue

            new_start, new_end = find_nearest_match(entity_text, source_text, original_start)

            if new_start is not None:
                corrected_entity = entity.copy()
                corrected_entity["start_offset"] = new_start
                corrected_entity["end_offset"] = new_end
                corrected_entities.append(corrected_entity)

                if new_start != original_start:
                    corrections_made += 1
            else:
                # If no match is found, we could either drop the entity or keep the original.
                # For this implementation, we will drop it as it indicates a hallucination.
                failed_corrections += 1
                logging.warning(
                    f"Could not find a match for entity text '{entity_text}' "
                    f"in source text. Dropping entity."
                )

        new_record = record.copy()
        new_record["predicted_entities"] = corrected_entities
        corrected_records.append(new_record)

    # --- Save the corrected predictions ---
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        for record in corrected_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logging.info(f"\nPost-processing complete.")
    logging.info(f"  - Total entities processed: {total_entities}")
    logging.info(f"  - Offsets corrected: {corrections_made}")
    logging.info(f"  - Entities dropped (no match found): {failed_corrections}")
    logging.info(f"Corrected predictions saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Post-process RAG predictions to correct character offsets."
    )
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help="Path to the .jsonl file containing the raw RAG predictions."
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help="Path to save the post-processed .jsonl file with corrected offsets."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        stream=sys.stdout
    )

    postprocess_predictions(args.input_path, args.output_path)