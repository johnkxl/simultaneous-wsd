from pathlib import Path

import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from gloss_bert.config import BERT_MODEL, get_device


class CrossEncoderWSD(nn.Module):
    name = "bert_cross_encoder"

    def __init__(self, model_name_or_path: str = BERT_MODEL):
        super().__init__()
        self.device = get_device()
        self.model = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2  # binary classification (0=Wrong Sense, 1=Correct Sense)
        ).to(self.device)

    def train(self):
        """Sets the model to training mode, enabling dropout and gradients."""
        self.model.train()

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()
    
    def save_pretrained(self, save_directory: Path) -> None:
        self.model.save_pretrained(save_directory)
    
    def __call__(self, **kwargs) -> SequenceClassifierOutput:
        return self.model(**kwargs)