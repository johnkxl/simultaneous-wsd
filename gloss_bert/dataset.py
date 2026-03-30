import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class WSDCrossEncoderDataset(Dataset):
    """
    A custom PyTorch Dataset for Word Sense Disambiguation using a Cross-Encoder architecture.
    
    Reference for Custom Datasets: 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    
    Reference for Sentence-Pair Classification (Cross-Encoder):
    https://huggingface.co/learn/nlp-course/chapter3/2
    """
    def __init__(self, 
                 contexts: list[str], 
                 target_lemmas: list[str], 
                 glosses: list[str], 
                 tokenizer: PreTrainedTokenizer, 
                 labels: list[int] = None, 
                 max_length: int = 256):
        """
        Args:
            contexts: The contexts for each target.
            target_lemmas: The base word being disambiguated.
            glosses: The dictionary definition or template gloss.
            tokenizer: Pretrained tokenizer.
            labels: 1 if the gloss is correct, 0 if it is wrong. None for inference/prediction.
        """
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.target_lemmas = target_lemmas
        self.glosses = glosses
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        """Required by PyTorch: Returns the total number of samples."""
        return len(self.target_lemmas)

    def _build_hypothesis(self, lemma: str, gloss: str) -> str:
        """
        Constructs the hypothesis string. 
        Abstracted to a separate method so it can be easily swapped for the 
        template defined by Wang and Wang (2020) in the milestone specifications.
        """
        return f"{lemma} : {gloss}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Required by PyTorch: Fetches and processes a single sample by its index.
        The DataLoader will call this multiple times to build a batch.
        """
        context = self.contexts[idx]
        hypothesis = self._build_hypothesis(self.target_lemmas[idx], self.glosses[idx])

        # Tokenizer natively handles the [CLS] Context [SEP] Hypothesis [SEP] structure
        # Reference: https://huggingface.co/docs/transformers/main_classes/tokenizer
        encoding: dict[str, torch.Tensor] = self.tokenizer(
            context,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Flatten the tensors to remove the batch dimension (DataLoader will re-add it)
        output_dict = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # token_type_ids tells BERT which tokens belong to the context vs. hypothesis
            'token_type_ids': encoding['token_type_ids'].flatten(), 
        }
        
        # Only attach labels if they were provided (Training mode)
        if self.labels is not None:
            output_dict['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return output_dict