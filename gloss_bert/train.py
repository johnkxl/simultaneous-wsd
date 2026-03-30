from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from tqdm import tqdm

from gloss_bert.encoder import CrossEncoderWSD
from gloss_bert.dataset import WSDCrossEncoderDataset
from gloss_bert.config import BERT_MODEL, MODELS_DIR


def train_cross_encoder(encoder: CrossEncoderWSD,
                        train_dataset: WSDCrossEncoderDataset,
                        epochs: int = 3,
                        batch_size: int = 32) -> None:

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # AdamW is the standard optimizer for transformer models
    optimizer = AdamW(encoder.model.parameters(), lr=2e-5)
    encoder.train()

    # Ensure the base models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to available device
            input_ids = batch['input_ids'].to(encoder.device)
            attention_mask = batch['attention_mask'].to(encoder.device)
            token_type_ids = batch['token_type_ids'].to(encoder.device)
            labels = batch['labels'].to(encoder.device)

            # Forward pass
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            # Calculate loss and backpropagate
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")

        # Save a model checkpoint for this epoch
        epoch_save_path = MODELS_DIR / f"{encoder.name}_epoch_{epoch+1}"
        encoder.save_pretrained(epoch_save_path)
        print(f"Model checkpoint saved to {epoch_save_path}")


def main():
    from gloss_bert.prepare_data import build_training_data, SEMCOR_DATA, SEMCOR_KEYS

    encoder = CrossEncoderWSD(BERT_MODEL)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    train_dataset = build_training_data(tokenizer, SEMCOR_DATA, SEMCOR_KEYS)

    # Devlin et al. (2018) explicitly recommend 2-4 epochs for fine-tuning BERT
    # to prevent catastrophic forgetting and overfitting on downstream tasks.
    train_cross_encoder(encoder, train_dataset, epochs=4, batch_size=32)


if __name__ == "__main__":
    main()