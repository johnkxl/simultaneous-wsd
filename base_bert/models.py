from typing import NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from nltk.corpus.reader.wordnet import Synset

from transformers import logging as hf_logging

from utils import TargetWord, Word, PredictionContext, get_synsets, lemma_key_from_synset, fill_synset_template, cosine_similarity

BERT_VERSION = "google-bert/bert-base-cased"
DISTILBERT_VERSION = "distilbert/distilbert-base-cased"

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


class BERT(ABC):
    """
    Interface to interact with the BERT model.

    Basic methods for interaction with BERT are implemented.

    Methods for processing and formatting inputs, and for processing BERT's
    outputs are left to be implemented by subclasses.
    """
    # name = "bert"

    # Pass the model and tokenizer as parameters so that multiple instances of
    # subclasses amy use the same object in memory
    def __init__(self, model: BertModel, tokenizer: BertTokenizer, model_name: str = "bert"):
        torch.set_grad_enabled(False)
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.name = model_name

    def get_hidden_layers(self, text: str, layers: int = 4) -> torch.Tensor:
        """Returns the sum of the last N hidden layers for a given text string."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad(): # Disables gradient calculation, saving memory and time
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Stack the last 4 layers: shape becomes (4, 1, seq_len, embedding_dim)
        last_four_layers = torch.stack(outputs.hidden_states[-layers:])
        
        # Sum along the layer dimension (dim=0). [0] removes the batch dimension.
        # Final shape: (seq_len, embedding_dim)
        summed_layers = torch.sum(last_four_layers, dim=0)[0] 
        return summed_layers
    
    @staticmethod
    def _get_subword_indices(offset_map: list[tuple[int, int]], target: Word, context_offset: int) -> tuple[int, int]:
        start = target.doc_char_offset - context_offset
        end = start + len(target.word)

        first_subword, last_subword = 0, 0

        for subword_start, subword_end in offset_map:
            if subword_end <= start:
                first_subword += 1
            if subword_start > end:
                break
            last_subword += 1
        
        return first_subword, last_subword
    
    @abstractmethod
    def predict(self, ctx: PredictionContext) -> dict[str, str]:
        raise NotImplementedError
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Gets the single averaged embedding for an entire string (used for synsets)."""
        sentence_embeddings = self.get_hidden_layers(text)
        
        # Average across ALL tokens in the string (dim=0)
        return torch.mean(sentence_embeddings, dim=0).numpy()

    def embed_sentence(self, text: str) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        Runs the expensive BERT forward pass ONCE per sentence.
        Returns the summed hidden layers and the offset mapping.
        """
        # Get tokens and offsets
        inputs: dict[str, torch.Tensor] = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
        
        # Must remove 'offset_mapping' from the dictionary before passing it to the model,
        # otherwise PyTorch will throw an 'unexpected keyword argument' error.
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Stack and sum the last 4 layers
        last_four_layers = torch.stack(outputs.hidden_states[-4:])
        summed_layers = torch.sum(last_four_layers, dim=0)[0] 
        
        return summed_layers, offset_mapping

    def extract_word_embedding(self, sentence_embeddings: torch.Tensor, offset_map: list[tuple[int, int]], target: Word, context_offset: int) -> np.ndarray:
        """
        Cheaply slices the pre-computed sentence embeddings for a specific target word.
        """
        first_subword, last_subword = self._get_subword_indices(offset_map, target, context_offset)
        
        # Slice out just the subwords belonging to our target word
        subword_embeddings = sentence_embeddings[first_subword : last_subword]

        # Average the subwords and convert to NumPy array
        return torch.mean(subword_embeddings, dim=0).numpy()


class BaseBERT(BERT):
    def __init__(self, model: BertModel, tokenizer: BertTokenizer, model_name: str = "bert"):
        super().__init__(model, tokenizer, model_name)

    def predict(self, ctx: PredictionContext) -> dict[str, str]:
        return self.forward(
            targets=ctx.targets,
            sentence_text=ctx.sentence_text,
            sent_offset=ctx.sent_offset,
        )
    
    def get_sense_embedding(self, lemma: str, synset: Synset) -> np.ndarray:
        synset_string = fill_synset_template(lemma, synset)
        return self.get_embeddings(synset_string)

    def forward(
            self,
            targets: list[TargetWord],
            sentence_text: str,
            sent_offset: int,
    ) -> dict[str, str]:
        sentence_embeddings, offset_map = self.embed_sentence(sentence_text)

        target_predictions = {}

        for target in targets:
            context_embedding = self.extract_word_embedding(sentence_embeddings, offset_map, target.word, sent_offset)

            sense_scores = {}

            synsets = get_synsets(target.word.lemma, target.word.pos)
            if not synsets: continue

            for synset in synsets:
                lemma_key = lemma_key_from_synset(synset, target.word.lemma)
                if not lemma_key: continue

                synset_embedding = self.get_sense_embedding(target.word.lemma, synset)

                similarity = cosine_similarity(context_embedding, synset_embedding)

                sense_scores[lemma_key] = similarity
            
            # respects insertion order
            predicted_sense = max(sense_scores, key=sense_scores.get)
            target_predictions[target.id] = predicted_sense
            
        return target_predictions
    

class ViterbiWSD(BERT):
    def __init__(self, model: BertModel, tokenizer: BertTokenizer, model_name: str = "bert"):
        super().__init__(model, tokenizer, model_name)
        self.name = self.name + "-viterberi"
    
    def predict(self, ctx: PredictionContext) -> dict[str, str]:
        return self.viterbi_decode(
            words=ctx.words,
            targets=ctx.targets,
            sentence_text=ctx.sentence_text,
            sent_offset=ctx.sent_offset,
            alpha=ctx.alpha
        )
    
    def viterbi_decode(self, words: list[Word], targets: list[TargetWord], sentence_text: str, sent_offset: int, alpha: 0.5) -> dict[str, str]:
        """
        Runs Viterbi over ALL valid words in a sentence, returning a dictionary mapping target IDs to their predicted senses.
        """
        sentence_embeddings, offset_map = self.embed_sentence(sentence_text)

        target_map = {t.word.sent_char_offset: t for t in targets}

        states = []
        target_indices = {}

        synset_embeddings = {}

        for word in words:
            is_target = word.sent_char_offset in target_map

            synsets = get_synsets(word.lemma, word.pos)

            if not synsets and not is_target: continue

            word_states = []
            context_embedding = self.extract_word_embedding(sentence_embeddings, offset_map, word, sent_offset)

            for synset in synsets:
                lemma_key = lemma_key_from_synset(synset, word.lemma)
                if not lemma_key:
                    lemma_key = synset.lemmas()[0].key()
                
                synset_string = fill_synset_template(word.lemma, synset)
                synset_embedding = self.get_embeddings(synset_string)

                emission = cosine_similarity(context_embedding, synset_embedding)
                word_states.append((lemma_key, synset, emission))

                synset_embeddings[lemma_key] = synset_embedding

            if word_states:
                states.append(word_states)
                if is_target:
                    target_indices[len(states) - 1] = target_map[word.sent_char_offset].id
            
        if not states:
            return {}
        
        V = [{}]
        B = [{}]
        
        # Initialisation
        for lemma_key, synset, emission in states[0]:
            V[0][lemma_key] = emission
            B[0][lemma_key] = None

        for i in range(1, len(states)):
            V.append({})
            B.append({})

            for curr_key, curr_syn, curr_emission in states[i]:
                max_score = float('-inf')
                best_prev_key = None

                for prev_key, prev_syn, _ in states[i - 1]:
                    transition = cosine_similarity(synset_embeddings[prev_key], synset_embeddings[curr_key])
                    score = V[i - 1][prev_key] + (alpha * transition)

                    if score > max_score:
                        max_score = score
                        best_prev_key = prev_key
                    
                V[i][curr_key] = curr_emission + max_score
                B[i][curr_key] = best_prev_key
            
        optimal_path = []
        best_final_key = max(V[-1], key=V[-1].get)
        optimal_path.append(best_final_key)

        curr_key = best_final_key
        for i in range(len(states) - 1, 0, -1):
            curr_key = B[i][curr_key]
            optimal_path.append(curr_key)
        
        optimal_path = optimal_path[::-1]

        target_predictions = {}
        for seq_idx, target_id in target_indices.items():
            target_predictions[target_id] = optimal_path[seq_idx]
        
        return target_predictions

    def viterbi_decode_targets(self, targets: list[TargetWord], sentence_text: str, sent_offset: int, alpha: float = 0.5) -> list:

        # Embed the entire context with BERT
        sentence_embeddings, offset_map = self.embed_sentence(sentence_text)

        # Compute the emission scores for each target's synsets.
        states = []
        synset_embeddings = {}
        for target in targets:
            target_states = []
            context_embedding = self.extract_word_embedding(sentence_embeddings, offset_map, target.word, sent_offset)

            for synset in get_synsets(target.word.lemma, target.word.pos):
                lemma_key = lemma_key_from_synset(synset, target.word.lemma)
                if not lemma_key: continue

                synset_string = fill_synset_template(target.word.lemma, synset)
                synset_embedding = self.get_embeddings(synset_string)

                emission = cosine_similarity(context_embedding, synset_embedding)
                target_states.append((lemma_key, synset, emission))

                synset_embeddings[lemma_key] = synset_embedding

            states.append(target_states)
        
        V = [{}]  # Store max path scores
        B = [{}]  # Backpointers to store the path

        # Initialisation
        for lemma_key, synset, emission in states[0]:
            V[0][lemma_key] = emission
            B[0][lemma_key] = None
        
        for i in range(1, len(states)):
            V.append({})
            B.append({})

            for curr_key, curr_syn, curr_emission in states[i]:
                max_score = float("-inf")
                best_prev_key = None

                # Check against every sense of previous word
                for prev_key, prev_syn, _ in states[i - 1]:
                    transition = cosine_similarity(synset_embeddings[prev_key], synset_embeddings[curr_key])

                    # Viterbi equation
                    score = V[i - 1][prev_key] + (alpha * transition)

                    if score > max_score:
                        max_score = score
                        best_prev_key = prev_key

                V[i][curr_key] = curr_emission + max_score
                B[i][curr_key] = best_prev_key

        predicted_senses = []

        best_final_key = max(V[-1], key=V[-1].get)
        predicted_senses.append(best_final_key)

        curr_key = best_final_key
        for i in range(len(states) - 1, 0, -1):
            curr_key = B[i][curr_key]
            predicted_senses.append(curr_key)
        
        return predicted_senses[::-1]


# class Player(NamedTuple):
#     player: TargetWord
#     actions: list[str]
#     action_embeddings: np.ndarray
#     probabilities: np.ndarray


class Strategy(NamedTuple):
    key: str
    embedding: np.ndarray
    emission: float


class GameTheoryWSD(BERT):
    def __init__(self, model: BertModel, tokenizer: BertTokenizer, model_name: str = "bert"):
        super().__init__(model, tokenizer, model_name)
        self.name = self.name + "-nash"
    
    def predict(self, ctx: PredictionContext) -> dict[str, str]:
        return self.nash_equilibrium_decode(
            words=ctx.words,
            targets=ctx.targets,
            sentence_text=ctx.sentence_text,
            sent_offset=ctx.sent_offset,
            alpha=ctx.alpha
        )

    def nash_equilibrium_decode(
            self,
            words: list[Word],
            targets: list[TargetWord],
            sentence_text: str,
            sent_offset: int,
            alpha: 0.5
        ) -> dict[str, str]:
        """
        Finds a nash equilibrium of senses over ALL valid words in a sentence,
        returning a dictionary mapping target IDs to their predicted senses.
        """
        sentence_embeddings, offset_map = self.embed_sentence(sentence_text)

        target_map = {t.word.sent_char_offset: t for t in targets}

        target_indices: dict[int, int] = {}
        
        players: list[list[Strategy]] = []

        # Set up the strategies for each player with the initial utility as the
        # similarity between the sense embedding and the context embedding
        for word in words:
            is_target = word.sent_char_offset in target_map

            if not is_target: continue

            synsets = get_synsets(word.lemma, word.pos)

            if not synsets: continue

            strategies = []
            context_embedding = self.extract_word_embedding(sentence_embeddings, offset_map, word, sent_offset)

            for synset in synsets:
                lemma_key = lemma_key_from_synset(synset, word.lemma)
                if not lemma_key:
                    lemma_key = synset.lemmas()[0].key()
                
                synset_string = fill_synset_template(word.lemma, synset)
                synset_embedding = self.get_embeddings(synset_string)

                emission = cosine_similarity(context_embedding, synset_embedding)

                strategies.append(Strategy(lemma_key, synset_embedding, emission))

            players.append(strategies)
            if is_target:
                target_indices[len(players) - 1] = target_map[word.sent_char_offset].id

        # Initialise the strategy profile of all the players based on the highest emission scores
        current_profile: list[int] = []
        for strategies in players:
            best_idx = max(range(len(strategies)), key=lambda i: strategies[i].emission)
            current_profile.append(best_idx)

        for iteration in range(100):
            has_changed = False

            # Each player i evaluates their position
            for i, strategies in enumerate(players):
                best_strategy_idx = current_profile[i]
                max_utility = float('-inf')

                for strat_idx, strategy in enumerate(strategies):
                    cooperate_payoff = 0.0

                    # Calculate the payoff of choosing this strategy given the profile -i
                    for j, other_strat in enumerate(current_profile):
                        if i == j: continue

                        cooperate_payoff += cosine_similarity(
                            players[i][strat_idx].embedding,
                            players[j][other_strat].embedding
                        )
                    
                    utility = strategy.emission + (alpha * cooperate_payoff)

                    if utility > max_utility:
                        best_strategy_idx = strat_idx
                        max_utility = utility
                
                # Switch to best response
                if best_strategy_idx != current_profile[i]:
                    current_profile[i] = best_strategy_idx
                    has_changed = True
            
            if not has_changed:
                break
        
        target_predictions = {}
        for player_idx, target_id in target_indices.items():
            strategy_idx = current_profile[player_idx]
            target_predictions[target_id] = players[player_idx][strategy_idx].key
        
        return target_predictions
       

class NaturalGlossWSD(BaseBERT):
    def __init__(self, model: BertModel, tokenizer: BertTokenizer, model_name: str = "bert"):
        super().__init__(model, tokenizer, model_name)
        self.name = self.name + "-natural-gloss"
        self.sense_cache = {}

    def get_contextualized_sense_embedding(self, lemma: str, synset: Synset) -> np.ndarray:
        """
        Builds a sense embedding by averaging how the word is actually 
        used in WordNet's example sentences, rather than just reading the dictionary definition.
        """
        synset_id = synset.name()
        if synset_id in self.sense_cache:
            return self.sense_cache[synset_id]

        examples = synset.examples()

        example_embeddings = []

        # Run each example sentence through BERT naturally
        for example in examples:
            # Check lemma is actually in this example sentence
            if lemma.lower() not in example.lower():
                continue
                
            # Create a dummy Word object optimized subword extractor
            # Estimate the char offset by finding the lemma in the string
            char_offset = example.lower().find(lemma.lower())
            dummy_word = Word(lemma, "", lemma, char_offset, char_offset)
            
            sentence_embeddings, offset_map = self.embed_sentence(example)
            
            try:
                # Extract the vector for the word exactly as it appears in the example
                ex_embedding = self.extract_word_embedding(sentence_embeddings, offset_map, dummy_word, 0)
                example_embeddings.append(ex_embedding)
            except Exception:
                # If subword alignment fails for some reason
                continue

        # If successfully extracted examples, average them to create a master "Usage" vector
        if example_embeddings:
            final_embedding = np.mean(example_embeddings, axis=0)
        else:
            # Fallback if no examples of the lemma wasn't found cleanly in the examples
            template = fill_synset_template(lemma, synset)
            final_embedding = self.get_embeddings(template)

        self.sense_cache[synset_id] = final_embedding
        return final_embedding

    def get_sense_embedding(self, lemma: str, synset: Synset):
        return self.get_contextualized_sense_embedding(lemma, synset)

