from pathlib import Path

import torch

BERT_MODEL = "google-bert/bert-base-cased"
MODELS_DIR = Path(__file__).parent.parent / "models"
ENCODER_PATH = MODELS_DIR / "bert_cross_encoder_epoch_3"


def get_device() -> torch.device:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    return torch.device(device)
