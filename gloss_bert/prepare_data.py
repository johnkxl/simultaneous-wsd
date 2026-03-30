from pathlib import Path
from tqdm import tqdm

from utils import load_dataset, load_gold_keys, get_synsets, lemma_key_from_synset
from gloss_bert.dataset import WSDCrossEncoderDataset

# Paths to the SemCor files from the WSD Evaluation Framework
SEMCOR_DIR = Path(__file__).parent.parent / "WSD_Training_Corpora/SemCor"
SEMCOR_DATA = SEMCOR_DIR / "semcor.data.xml"
SEMCOR_KEYS = SEMCOR_DIR / "semcor.gold.key.txt"


def build_training_data(tokenizer, dataset_path: Path, gold_keys_path: Path):
    """
    Parses the SemCor XML and Gold Keys to generate positive and negative 
    training pairs for the Cross-Encoder.
    """
    print("Loading SemCor dataset...")
    targets, documents = load_dataset(dataset_path)
    gold_keys = load_gold_keys(gold_keys_path)

    contexts = []
    lemmas = []
    glosses = []
    labels = []

    print("Generating positive and negative training samples...")
    for target in tqdm(targets, desc="Parsing Targets"):
        # Get the ground-truth valid sense keys for this specific instance
        true_senses = gold_keys.get(target.id, set())
        if not true_senses:
            continue
            
        # Rebuild the context sentence
        doc_id = target.document_id
        sent_id = target.sentence_id

        sentence_words = documents[doc_id][sent_id]
        sentence_text = " ".join([w.word for w in sentence_words])

        # Gather all candidate synsets from WordNet
        candidate_synsets = get_synsets(target.word.lemma, target.word.pos)
        
        # Create the Positive and Negative pairs
        for synset in candidate_synsets:
            # Get the specific lemma key for this synset
            lemma_key = lemma_key_from_synset(synset, target.word.lemma)
            if not lemma_key:
                lemma_key = synset.lemmas()[0].key()

            is_correct = 1 if lemma_key in true_senses else 0

            contexts.append(sentence_text)
            lemmas.append(target.word.lemma)
            glosses.append(synset.definition())
            labels.append(is_correct)

    print(f"Generated {len(labels)} total training pairs.")
    print(f"Positive samples: {labels.count(1)}")
    print(f"Negative samples: {labels.count(0)}")

    return WSDCrossEncoderDataset(
        contexts=contexts,
        target_lemmas=lemmas,
        glosses=glosses,
        tokenizer=tokenizer,
        labels=labels
    )

if __name__ == "__main__":
    from transformers import BertTokenizer
    from gloss_bert.config import BERT_MODEL
    
    # Test to make sure it works
    tok = BertTokenizer.from_pretrained(BERT_MODEL)
    train_dataset = build_training_data(tok, SEMCOR_DATA, SEMCOR_KEYS)