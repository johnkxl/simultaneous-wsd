from pathlib import Path

import torch

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from utils import (
    TargetWord,
    Word,
    build_sentence_string,
    get_synsets,
    load_dataset,
    save_results,
)

from gloss_bert.config import BERT_MODEL, ENCODER_PATH, get_device
from gloss_bert.dataset import WSDCrossEncoderDataset
from gloss_bert.encoder import CrossEncoderWSD

TEST_DATASET = Path(__file__).parent.parent / "semeval2007.data.xml"


class CrossEncoderPredictor:
    """
    Handles inference for the fine-tuned Cross-Encoder WSD model.
    """
    def __init__(self, model_path: str = ENCODER_PATH):
        self.device = get_device()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        
        # Load the fine-tuned weights
        self.model = CrossEncoderWSD(model_path).to(self.device)
        
        # Set to evaluation mode to disable Dropout layers, ensuring deterministic outputs.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
        self.model.eval()

    def predict(self, targets: list[TargetWord], documents: dict[str, dict[str, list[Word]]]) -> list[tuple[str, str, str]]:
        # Flatten the entire dataset into global lists
        all_contexts = []
        all_lemmas = []
        all_glosses = []
        
        # Rmember which rows in massive lists belong to which target
        # Format: (TargetWord, list_of_lemma_keys, start_index, end_index)
        target_mappings: list[tuple[TargetWord, list[str], int, int]] = []

        print("Building global evaluation dataset...")
        for target in targets:
            candidate_synsets = get_synsets(target.word.lemma, target.word.pos)
            if not candidate_synsets:
                continue
                
            doc_id = target.document_id
            sent_id = target.sentence_id
            _, sentence_text = build_sentence_string(documents[doc_id][sent_id])
            
            start_idx = len(all_contexts)
            candidate_keys = []
            
            for synset in candidate_synsets:
                all_contexts.append(sentence_text)
                all_lemmas.append(target.word.lemma)
                all_glosses.append(synset.definition())
                
                # Extract the correct key
                key = next(
                    (l.key() for l in synset.lemmas()
                     if l.name().lower() == target.word.lemma.lower()), 
                None)
                candidate_keys.append(key or synset.lemmas()[0].key())

            end_idx = len(all_contexts)
            target_mappings.append((target, candidate_keys, start_idx, end_idx))

        train_dataset = WSDCrossEncoderDataset(
            contexts=all_contexts,
            target_lemmas=all_lemmas,
            glosses=all_glosses,
            tokenizer=self.tokenizer,
            labels=None
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        all_probabilities: list[float] = []

        print("Running batch inference...")
        # Inference Block
        # Disable gradient tracking, reducing memory usage and speeding up computation.
        # Reference: https://pytorch.org/docs/stable/generated/torch.no_grad.html
        with torch.no_grad():
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids
                )

                # Apply Softmax to convert logits to probabilities that sum to 1.0
                # dim=-1 applies it across the last dimsension—the class logits
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Extract the Class 1 (Correct) probabilities and add to our master list
                all_probabilities.extend(probs[:, 1].tolist())

        # Regroup and extract the argmax for each target
        preds = []
        for target, candidate_keys, start_idx, end_idx in target_mappings:
            # Slice out the specific probabilities for this word's candidates
            target_probs = all_probabilities[start_idx:end_idx]
            
            # Find the index of the highest score
            best_index = target_probs.index(max(target_probs))
            best_key = candidate_keys[best_index]
            
            preds.append((target.id, target.word.pos, best_key))

        return preds


def main():
    from utils import RESULTS_DIR

    print(f"Loading dataset from {TEST_DATASET}...")
    targets, documents = load_dataset(TEST_DATASET)
    
    print("Initializing CrossEncoderPredictor...")
    # This will automatically load the model from ENCODER_PATH defined in config.py
    predictor = CrossEncoderPredictor()
    
    preds = predictor.predict(targets, documents)
    
    save_results(preds, "CrossEncoderWSD", save_dir=RESULTS_DIR)
    print("Cross-Encoder predictions saved to CrossEncoderWSD.out")

if __name__ == "__main__":
    main()