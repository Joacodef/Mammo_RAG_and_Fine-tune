import torch
from tqdm import tqdm
import numpy as np

from src.models.bert_ner import BertNerModel
from src.data_loader.ner_datamodule import NERDataModule

class Predictor:
    """
    Handles loading a trained model and running inference on a test dataset.
    """

    def __init__(self, model_path, config):
        """
        Initializes the Predictor.

        Args:
            model_path (str): The path to the directory containing the saved model.
            config (dict): The evaluation configuration dictionary.
        """
        self.model_path = model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The number of labels needs to be known before loading the model.
        # This information is typically stored in the model's config.json.
        # For simplicity, we can reconstruct it from our evaluation config.
        # A more robust solution might load config.json from the model_path.
        n_labels = len(self._get_label_map_from_config())
        
        self.model = BertNerModel(base_model=model_path, n_labels=n_labels)
        self.model.to(self.device)
        self.model.eval()

    def _get_label_map_from_config(self):
        """Helper to reconstruct the label map."""
        # This is a simplified stand-in for loading the model's actual config
        # or having a more sophisticated config management.
        entity_labels = self.config.get('entity_labels', ["FIND", "REG", "OBS", "GANGLIOS"]) # Fallback
        label_map = {"O": 0}
        for label in entity_labels:
            label_map[f"B-{label}"] = len(label_map)
            label_map[f"I-{label}"] = len(label_map)
        return label_map

    def predict(self, test_dataloader):
        """
        Runs inference on the provided dataloader.

        Args:
            test_dataloader (DataLoader): The DataLoader for the test set.

        Returns:
            tuple: A tuple containing:
                - all_predictions (list): A list of predicted label sequences.
                - all_true_labels (list): A list of ground-truth label sequences.
        """
        all_predictions = []
        all_true_labels = []

        progress_bar = tqdm(test_dataloader, desc="Evaluating")
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Get the most likely token class prediction
                predictions = torch.argmax(outputs.logits, dim=2)

                # Move data back to CPU for evaluation
                predictions = predictions.detach().cpu().numpy()
                true_labels = labels.detach().cpu().numpy()

                # Align predictions and labels, ignoring padding
                for i in range(len(true_labels)):
                    pred_labels_i = []
                    true_labels_i = []
                    for j in range(len(true_labels[i])):
                        if true_labels[i][j] != -100: # -100 is often used to ignore tokens in loss
                            pred_labels_i.append(predictions[i][j])
                            true_labels_i.append(true_labels[i][j])
                    
                    all_predictions.append(pred_labels_i)
                    all_true_labels.append(true_labels_i)

        return all_predictions, all_true_labels