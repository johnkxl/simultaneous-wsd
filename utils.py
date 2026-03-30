from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree as ET
import sys

import numpy as np
from scipy.spatial.distance import cosine

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset


RESULTS_DIR = Path(__file__).parent / "out"


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return 1 - cosine(u, v)


class Word(NamedTuple):
    lemma: str
    pos: str
    word: str
    doc_char_offset: int
    sent_char_offset: int


class TargetWord(NamedTuple):
    word: Word
    id: str
    document_id: str
    sentence_id: str


class PredictionContext(NamedTuple):
    """Encapsulates all necessary data for a model to make a prediction on a sentence."""
    words: list[Word]
    targets: list[TargetWord]
    sentence_text: str
    sent_offset: int
    alpha: float = 0.5


def load_dataset(pathname: Path) -> tuple[list[TargetWord], dict[str, dict[str, list[Word]]]]:
    """
    Load dataset from XML file and return a tuple `(targets, documents)`.

    Returns
    -------
    targets: list of TargetWords
        List of all words labelled as `instance` in the dataset.
    
    documents: dict of documents
        Dictionary mapping the `document_id` to a dictionary mapping
        `sentence_id` the the sentence as a list of `Word` objects.
        Each context is defined by the XML tag `text`.
    """
    tree = ET.parse(pathname)
    root = tree.getroot()

    targets = []
    documents = defaultdict(lambda: defaultdict(list))

    for document in root:
        if document.tag != "text":
            raise ValueError("Documents are expected to have the XML tag of 'text'.")
        
        # Document-level context
        document_id = document.attrib["id"]
        processed_document = documents[document_id]

        document_pos = 0

        for sentence in document:

            sentence_pos = 0
            
            # Sentence-level context
            sentence_id = sentence.attrib["id"]
            processed_sentence = processed_document[sentence_id]

            for word in sentence:
                pos: str = word.attrib['pos']

                # Add a space before a word
                if pos.isalpha():
                    if document_pos > 0:
                        document_pos += 1
                    
                    if sentence_pos > 0:
                        sentence_pos += 1

                word_obj = Word(word.attrib['lemma'], pos, word.text, document_pos, sentence_pos)
                processed_sentence.append(word_obj)

                document_pos += len(word.text)
                sentence_pos += len(word.text)

                if word.tag != "instance": continue
                
                targets.append(TargetWord(word_obj,
                                          word.attrib['id'],
                                          document_id,
                                          sentence_id))
    
    return targets, documents


def get_wordnet_pos(tag: str) -> Enum:
    try:
        return getattr(wn, tag)

    except AttributeError:
        
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('V'):
            return wn.VERB
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN


def get_synsets(lemma: str, pos: str) -> list[Synset]:
    return wn.synsets(lemma, get_wordnet_pos(pos))


def fill_synset_template(lemma: str, synset: Synset):
    lemmas = ", ".join([lemma_obj.name().lower().replace("_", " ") for lemma_obj in synset.lemmas()])
    definition = synset.definition()
    examples = " ".join(synset.examples())
    template = f"{lemma} - {lemmas} - {definition} {examples}"
    return template


def lemma_key_from_synset(synset: Synset, lemma_name: str) -> str:
    """
    Returns the first lemma key in the synset that matches the given lemma name.
    """
    return next(
        (
            lemma_obj.key() for lemma_obj in synset.lemmas()
            if lemma_obj.name().lower() == lemma_name.lower()
        ),
        None
    )


def save_results(results: list[tuple[str, str]], model_name: str, logger = sys.stdout, save_dir: Path = None) -> None:
    if not save_dir:
        save_dir = Path.cwd()
    
    out = save_dir / f"{model_name}.out"
    with open(out, 'w') as f:
        for word_id, pos, predicted_sense in results:
            f.write(f"{word_id} {pos} {predicted_sense}\n")
    print(f"Results saved to {out}")


def build_sentence_string(sentence: list[Word]) -> tuple[int, str]:
    sentence_text = ""
    offset = sentence[0].doc_char_offset

    for word in sentence:
        if len(sentence_text) != word.sent_char_offset:
            sentence_text += " "
        
        sentence_text += word.word
    
    return offset, sentence_text


def load_gold_keys(gold_filepath: Path) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    with open(gold_filepath, 'r') as f:
        for line in f.readlines():
            items = line.split()
            gold_id = items[0]
            gold_senses = items[1:]
            gold[gold_id] = set(gold_senses)
    return gold


def load_predictions(pred_filepath: Path) -> list[tuple[str, str, str]]:
    preds = []
    with open(pred_filepath, 'r') as f:
        while line := f.readline():
            preds.append(tuple(line.split()))
    
    return preds


def calculate_accuracy(counts: dict[bool, int]) -> float:
    total = sum(counts.values())
    return counts[True] / total


class Sentence(NamedTuple):
    sentence_id: str
    offset: int
    words: list[Word]
    targets: list[TargetWord]
    text: str

class Sentences:
    def __init__(self, documents: dict[str, dict[str, list[Word]]], targets: list[TargetWord]):
        self.sentences: list[Sentence] = []
        
        targets_by_sentence: dict[str, list[TargetWord]] = defaultdict(list)
        for target in targets:
            targets_by_sentence[target.sentence_id].append(target)

        for sentence_id, sentence_targets in targets_by_sentence.items():
            doc_id = sentence_targets[0].document_id
            sentence_words = documents[doc_id][sentence_id]

            sent_offset, sentence_text = build_sentence_string(sentence_words)

            sentence = Sentence(
                sentence_id,
                sent_offset,
                sentence_words,
                sentence_targets,
                sentence_text
            )
            self.sentences.append(sentence)

    def __iter__(self):
        return iter(self.sentences)

