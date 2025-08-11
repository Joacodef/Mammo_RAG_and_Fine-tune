import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

class BertNerModel(nn.Module):
    """
    A wrapper around a Hugging Face AutoModelForTokenClassification model.
    This class simplifies the model initialization and forward pass for NER tasks.
    """

    def __init__(self, base_model, n_labels):
        """
        Initializes the NER model.

        Args:
            base_model (str): The name or path of the pre-trained transformer model
                              from the Hugging Face Hub (e.g., 'bert-base-cased').
            n_labels (int): The total number of unique labels for the token classification head.
                            This should include all B- and I- tags plus the 'O' tag.
        """
        super().__init__()
        self.base_model_name = base_model
        self.n_labels = n_labels

        # Load model configuration and update the number of labels
        config = AutoConfig.from_pretrained(
            self.base_model_name,
            num_labels=self.n_labels
        )
        
        # Load the pre-trained model with a token classification head
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.base_model_name,
            config=config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs the forward pass through the model.

        When `labels` are provided, the model returns a dictionary containing the loss
        and the logits. Otherwise, it returns only the logits.

        Args:
            input_ids (torch.Tensor): A batch of token IDs.
                                      Shape: (batch_size, sequence_length).
            attention_mask (torch.Tensor): The attention mask for the batch.
                                           Shape: (batch_size, sequence_length).
            labels (torch.Tensor, optional): The ground-truth label IDs for each token.
                                             Shape: (batch_size, sequence_length).

        Returns:
            transformers.modeling_outputs.TokenClassifierOutput:
                An object containing the loss (if labels are provided) and logits.
        """
        # The underlying Hugging Face model handles the loss calculation
        # when labels are provided.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs