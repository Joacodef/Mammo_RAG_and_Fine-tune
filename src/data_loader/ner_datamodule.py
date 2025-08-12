import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import warnings

class NERDataset(Dataset):
    """
    A PyTorch Dataset for Named Entity Recognition tasks.
    It handles tokenization and alignment of labels with tokens.
    """

    def __init__(self, file_path, tokenizer, label_map, warned_entities_set=None):
        """
        Args:
            file_path (str): Path to the .jsonl data file.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding text.
            label_map (dict): A mapping from entity labels (str) to integer IDs.
            warned_entities_set (set, optional): A set to track entities that have already triggered a warning.
        """
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.data = self._load_data(file_path)
        self.warned_entities = warned_entities_set if warned_entities_set is not None else set()

    def _load_data(self, file_path):
        """Loads data from a .jsonl file."""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def __len__(self):
        """Returns the number of records in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single data record.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        record = self.data[idx]
        text = record["text"]
        entities = record.get("entities", []) # Use .get for safety if entities are missing

        tokenized_inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        input_ids = tokenized_inputs["input_ids"].squeeze()
        attention_mask = tokenized_inputs["attention_mask"].squeeze()
        labels = self._align_labels(input_ids, entities)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _align_labels(self, input_ids, entities):
        """
        Aligns entity labels with the tokenized input_ids.
        If an entity label is not in the configured label_map, it is ignored and a warning is issued once.
        """
        labels = torch.full(input_ids.shape, fill_value=-100, dtype=torch.long) # Use -100 to ignore loss on non-entity tokens
        word_ids = self.tokenizer(self.tokenizer.decode(input_ids, skip_special_tokens=True), add_special_tokens=False).word_ids()

        for entity in entities:
            # Check for the entity itself
            if not isinstance(entity, dict):
                raise TypeError(f"Entity must be a dictionary, but got: {type(entity)}")

            entity_label = entity["label"]
            if f"B-{entity_label}" not in self.label_map:
                if entity_label not in self.warned_entities:
                    warnings.warn(f"Entity label '{entity_label}' not found in config and will be ignored.")
                    self.warned_entities.add(entity_label)
                continue

            start_char, end_char = entity["start_offset"], entity["end_offset"]

            # Validate offset types
            if not isinstance(start_char, int) or not isinstance(end_char, int):
                raise TypeError(
                    f"Entity offsets must be integers. "
                    f"Got start_offset: {start_char} (type {type(start_char)}) and "
                    f"end_offset: {end_char} (type {type(end_char)})."
                )

            # Check for inverted offsets
            if start_char >= end_char:
                continue
                
            start_token_idx, end_token_idx = -1, -1

            # This part of the logic can be optimized, but for now, we'll keep it as is.
            # A single call to the tokenizer would be more efficient.
            token_offsets = self.tokenizer(
                self.tokenizer.decode(input_ids, skip_special_tokens=True)
            ).encodings[0].offsets

            for i, (start, end) in enumerate(token_offsets):
                if start <= start_char < end:
                    start_token_idx = i
                if start < end_char <= end:
                    end_token_idx = i
                    break # Exit after finding the end token
            
            if start_token_idx != -1 and end_token_idx != -1:
                # Assign B-tag to the first token of the entity
                labels[start_token_idx + 1] = self.label_map[f"B-{entity_label}"]
                # Assign I-tags to subsequent tokens of the same entity
                for i in range(start_token_idx + 2, end_token_idx + 2):
                    if labels[i] == -100: # Ensure we don't overwrite existing labels
                        labels[i] = self.label_map[f"I-{entity_label}"]
        
        # Set all other tokens to 'O'
        labels[labels == -100] = 0
        return labels


class NERDataModule:
    """
    A data module to handle loading and preparing data for NER models.
    """

    def __init__(self, config, train_file=None, test_file=None):
        """
        Args:
            config (dict): The training or evaluation configuration dictionary.
            train_file (str, optional): Path to the training data file.
            test_file (str, optional): Path to the test data file.
        """
        self.config = config
        self.train_file = train_file
        self.test_file = test_file
        
        model_path = config.get('model_path') or config.get('model', {}).get('base_model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.label_map = self._create_label_map()
        self.warned_entities = set()

    def _create_label_map(self):
        """
        Creates a mapping from entity labels to integer IDs using the config file.
        """
        # For evaluation, entity_labels might be in the loaded model's config
        if 'entity_labels' in self.config.get('model', {}):
            entity_labels = self.config['model']['entity_labels']
        else: # Fallback for evaluation config
             entity_labels = ["FIND", "REG", "OBS", "GANGLIOS"] # Provide a default or load from model
        
        label_map = {"O": 0}
        for label in entity_labels:
            label_map[f"B-{label}"] = len(label_map)
            label_map[f"I-{label}"] = len(label_map)
        return label_map

    def setup(self, stage=None):
        """Creates the training and/or test datasets."""
        if self.train_file:
            self.train_dataset = NERDataset(
                file_path=self.train_file,
                tokenizer=self.tokenizer,
                label_map=self.label_map,
                warned_entities_set=self.warned_entities
            )
        if self.test_file:
            self.test_dataset = NERDataset(
                file_path=self.test_file,
                tokenizer=self.tokenizer,
                label_map=self.label_map
            )

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get('trainer', {}).get('batch_size', 8),
            shuffle=True
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        if not hasattr(self, 'test_dataset'):
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.get('batch_size', 16)
        )