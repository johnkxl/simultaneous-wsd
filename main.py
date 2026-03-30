from pathlib import Path

from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from base_bert.models import (
    BERT,
    BaseBERT,
    NaturalGlossWSD,
    ViterbiWSD,
    GameTheoryWSD,
    BERT_VERSION,
)
from utils import PredictionContext, Sentences, load_dataset, save_results, RESULTS_DIR

TEST_DATASET = Path(__file__).parent / "semeval2007.data.xml"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bert_model = BertModel.from_pretrained(BERT_VERSION)
    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION)
    model_name = BERT_VERSION.split("/")[1].split('-')[0]

    models: list[type[BERT]] = [
        BaseBERT,
        NaturalGlossWSD,
        ViterbiWSD,
        GameTheoryWSD,
    ]

    targets, documents = load_dataset(TEST_DATASET)

    sentences = Sentences(documents, targets)

    for i, model_cls in enumerate(models, 1):
        model = model_cls(bert_model, tokenizer, model_name)

        print(f"({i}/{len(models)}) Predicting with model: {model.name}...")

        prog = tqdm(total=len(targets), desc=f"[{model.name}]")
        preds = []

        for sentence in sentences:
            prediction_args = PredictionContext(
                words=sentence.words,
                targets=sentence.targets,
                sentence_text=sentence.text,
                sent_offset=sentence.offset,
                alpha=0.5,
            )

            predictions_dict = model.predict(prediction_args)

            for target in sentence.targets:
                predicted_sense = predictions_dict.get(target.id)
                preds.append((target.id, target.word.pos, predicted_sense))
                prog.update(1)
        
        prog.close()
        save_results(preds, model.name, save_dir=RESULTS_DIR)
        print()


if __name__ == "__main__":
    main()