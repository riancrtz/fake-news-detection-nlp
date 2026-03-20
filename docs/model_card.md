# Model Card

**Project:** A Multi-Component NLP System for Fake News and Misinformation Detection
**Team:** Rian Cortez, Ryence Cortez, Ranz Cuarto, Mark Darius David
**Course:** 6INTELSY — Intelligent Systems, Holy Angel University
**Status:** Week 2 Checkpoint (preliminary results)

---

## What this system does

This system takes a short political statement as input and classifies it into one of six veracity categories: pants-fire, false, barely-true, half-true, mostly-true, or true. It is built as a research prototype to support human fact-checkers, not to replace them.

The pipeline has four components working together. The core classifier is a fine-tuned DistilBERT model. A Text-CNN trained from scratch serves as a baseline for comparison. A Named Entity Recognition module using spaCy tags persons, organizations, and locations mentioned in each claim. A contextual bandit agent learns an optimal confidence threshold for deciding when to flag a claim versus abstaining.

---

## Training data

The model is trained on the LIAR dataset, which contains 12,836 short political statements sourced from PolitiFact.com. Each statement is labeled by PolitiFact editors across six veracity levels. The dataset comes with fixed train, validation, and test splits which we use as-is.

A few things worth knowing about this dataset. It covers US political content almost exclusively, so the system should not be expected to generalize to other countries or contexts. The training set has more Republican-affiliated statements (4,497) than Democrat-affiliated ones (3,336), which is a known imbalance we track carefully in our bias analysis. Barack Obama is the most frequently mentioned speaker in the test set with 44 appearances, which may cause the model to over-index on statements about him.

---

## Results so far (Week 2)

These are preliminary numbers. DistilBERT fine-tuning results will be added in Week 3.

| Model | Test Accuracy | Test Macro-F1 |
|-------|-------------|---------------|
| Logistic Regression (TF-IDF) | 0.2478 | 0.2214 |
| Text-CNN (GloVe 300d) | 0.2478 | 0.2384 |
| DistilBERT (fine-tuned) | TBD | TBD |

The low numbers are expected for a 6-class problem on short text. The Text-CNN also shows signs of overfitting after epoch 3, which we plan to address with early stopping in Week 3.

---

## NER slice analysis

We ran the Logistic Regression predictions through a slice analysis using spaCy entity tags to see if performance varies by the type of content in a statement.

| Entity type present | Macro-F1 | Count |
|--------------------|----------|-------|
| PERSON | 0.2172 | 493 |
| ORG | 0.2068 | 302 |
| GPE (location) | 0.2011 | 391 |
| No named entities | ~0.2242 | varies |

Statements that reference specific locations and organizations are consistently harder to classify. We will investigate this further in Week 3.

---

## RL component

The bandit agent uses epsilon-greedy exploration over nine discretized threshold values (0.1 to 0.9). After 500 episodes it converged on a threshold of 0.8, selecting it 378 out of 500 times. The reward function penalizes missed misinformation (false negatives) at -2.0, which is four times heavier than the false positive penalty of -0.5. Note that the agent currently runs on simulated softmax outputs — it will be re-evaluated with real DistilBERT probabilities in Week 3.

---

## Known limitations

- All models are trained on US political statements only
- Current Macro-F1 scores (~0.22 to 0.24) are low — improvement expected after DistilBERT fine-tuning
- The bandit agent has not yet seen real model outputs
- The system has not been tested on data outside the LIAR benchmark

---

## How to use this responsibly

This system is a research prototype built for a university course. It should not be used to make decisions about real content without human review. Every output comes with a confidence score — treat low-confidence predictions with extra skepticism. Do not use this system for anything beyond research and educational purposes.
