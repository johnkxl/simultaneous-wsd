# Exploring Word Sense Disambiguation from Continuous to Simultaneous Comprehension

This repository contains the codebase and evaluation framework for exploring global semantic conditioning in Word Sense Disambiguation (WSD). 

Traditionally, WSD models predict the sense of a target word in isolation. This project evaluates methods for moving beyond independent classification by retroactively conditioning WSD predictions on the assigned senses of neighboring words. We evaluate baseline Bi-Encoders against dynamic programming (Viterbi), game-theoretic (Nash Equilibrium), and early-interaction (Cross-Encoder) architectures.

## Models Implemented
* **BaseBERT:** Baseline Bi-Encoder using isolated static gloss templates.
* **NaturalGlossWSD:** Bi-Encoder utilizing Contextualized Sense Embeddings from WordNet examples.
* **ViterbiWSD:** Dynamic programming model evaluating sequence transitions via the Bellman equation.
* **GameTheoryWSD:** A simultaneous, non-cooperative game model resolving semantic coherence via Nash Equilibria.
* **CrossEncoderWSD:** A fine-tuned Cross-Encoder leveraging deep, joint-attention mechanisms.

## Requirements

This codebase was tested on **Python 3.12.5** and **Python 3.13.3**. 

Install the required dependencies:
```bash
pip install torch transformers nltk scipy
```
*Note: You will also need to download the WordNet corpus via NLTK. Open a python shell and run: `import nltk; nltk.download('wordnet')`*

## Dataset Setup

Ensure the datasets are placed in the correct directories before running the pipeline:
1. Place the **SemEval-2007** evaluation dataset (`semeval2007.data.xml` and `semeval2007.gold.key.txt`) in the root directory.
2. [Download SemCor and WSD Evaluation Frameword](http://lcl.uniroma1.it/wsdeval/training-data) and ensure (`semcor.data.xml` and `semcor.gold.key.txt`) are in the `/WSD_Training_Corpora/SemCor` directory.

## Execution Pipeline

Follow these steps to reproduce the evaluation metrics.

### 1. Train the Cross-Encoder
Fine-tune the Cross-Encoder model on the SemCor dataset. *(Note: On an RTX 4070 Ti Super, this takes ~3 hours per epoch).*
```bash
python -m gloss_bert.train
```

### 2. Generate Cross-Encoder Predictions
Run inference using the fine-tuned weights to generate the prediction `.out` file.
```bash
python -m gloss_bert.predict
```

### 3. Generate Bi-Encoder Predictions
Run the standard models (`BaseBERT`, `NaturalGlossWSD`, `ViterbiWSD`, `GameTheoryWSD`) to generate their respective `.out` files.
```bash
python main.py
```

### 4. Standard Evaluation
Evaluate the overall and Part-of-Speech (POS) accuracy of all generated `.out` files against the gold keys. This script outputs a formatted LaTeX table.
```bash
python eval.py semeval2007.gold.key.txt out/*
```

### 5. Polysemy-Binned Evaluation
Evaluate the robustness of the models by grouping accuracy based on the number of candidate WordNet senses (Low, Medium, High). This script also outputs a formatted LaTeX table.
```bash
python eval_polysemy.py semeval2007.data.xml semeval2007.gold.key.txt out/*
```

## Hardware and Performance
All development and evaluation were conducted on Ubuntu (WSL2 via Windows 11) utilizing an Intel Core i9-14900K CPU and an MSI RTX 4070 Ti Super GPU (16GB VRAM). 
