# src/models/re_model.py
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class REModel(nn.Module):
    """
    A wrapper around a Hugging Face AutoModelForSequenceClassification model,
    adapted for Relation Extraction (RE).
    """

    def __init__(self, base_model, n_labels, tokenizer):
        """
        Initializes the RE model.

        Args:
            base_model (str): The name or path of the pre-trained transformer model.
            n_labels (int): The total number of unique relation labels.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer, which includes
                                                         the special entity markers.
        """
        super().__init__()
        self.base_model_name = base_model
        self.n_labels = n_labels

        # Load the pre-trained model with a sequence classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=self.n_labels
        )

        # Resize the model's token embeddings to accommodate the new special tokens
        # for entity markers (e.g., [E1_START], [E1_END]).
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs the forward pass through the sequence classification model.

        Args:
            input_ids (torch.Tensor): A batch of token IDs, including markers.
                                      Shape: (batch_size, sequence_length).
            attention_mask (torch.Tensor): The attention mask for the batch.
                                           Shape: (batch_size, sequence_length).
            labels (torch.Tensor, optional): The ground-truth relation label IDs.
                                             Shape: (batch_size,).

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput:
                An object containing the loss (if labels are provided) and logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs