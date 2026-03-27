# Model Card
**Project:** A Multi-Component NLP System for Fake News and Misinformation Detection
**Team:** Rian Cortez, Ryence Cortez, Ranz Cuarto, Mark Darius David
**Course:** 6INTELSY — Intelligent Systems, Holy Angel University
**Status:** Final Submission — March 2026

---

## What this system does

This system takes a short political statement as input and classifies it into one of six veracity categories: pants-fire, false, barely-true, half-true, mostly-true, or true. It is built as a research prototype to support human fact-checkers, not to replace them.

The pipeline has four components working together. The core classifier is a fine-tuned DistilBERT model. A Text-CNN trained from scratch serves as a baseline for comparison. A Named Entity Recognition module using spaCy tags persons, organizations, and locations mentioned in each claim. A contextual bandit agent learns an optimal confidence threshold for deciding when to flag a claim versus abstaining.

---

## Training data

The model is trained on the LIAR dataset, which contains 12,836 short political statements sourced from PolitiFact.com. Each statement is labeled by PolitiFact editors across six veracity levels. The dataset comes with fixed train, validation, and test splits which we use as-is.

A few things worth knowing about this dataset. It covers US political content almost exclusively, so the system should not be expected to generalize to other countries or contexts. The training set has more Republican-affiliated statements (4,497) than Democrat-affiliated ones (3,336), which is a known imbalance we track carefully in our bias analysis. Barack Obama is the most frequently mentioned speaker in the test set with 44 appearances, which may cause the model to over-index on statements about him.

---

## Final Results

| Model | Test Accuracy | Test Macro-F1 |
|-------|-------------|---------------|
| Logistic Regression (TF-IDF) | 0.2478 | 0.2214 |
| Text-CNN (GloVe 300d) | 0.2478 | 0.2384 |
| DistilBERT (fine-tuned) | 0.2762 | 0.2602 |
| DistilBERT + Adam (Ablation 1) | 0.2897 | 0.2743 |
| DistilBERT + Early Stopping (Ablation 2) | 0.2762 | 0.2654 |

The low numbers are expected for a 6-class problem on short text. DistilBERT outperforms both baselines on all metrics. The false class achieved the highest per-class F1 at 0.34 while pants-fire performed worst at 0.20, consistent with its underrepresentation in training data.

---

## Ablation Studies

**Ablation 1 — Adam vs AdamW:** Adam achieves a test Macro-F1 of 0.2743 compared to AdamW's 0.2690. AdamW's weight decay regularization is slightly too aggressive for fine-tuning on a small dataset like LIAR.

**Ablation 2 — Early Stopping vs No Early Stopping:** Early stopping with patience of 2 achieves 0.2654 versus 0.2602 without. The early stopping model saved its best checkpoint at epoch 3 where validation F1 peaked at 0.2746.

---

## Calibration Analysis

Calibration curves show the model follows the diagonal reasonably well at low predicted probabilities but becomes erratic at probabilities above 0.5, indicating overconfidence in high-certainty predictions. This supports the bandit agent's learned threshold of 0.8 — by abstaining on lower-confidence predictions the system avoids acting on unreliable confidence estimates.

---

## NER Slice Analysis

We ran DistilBERT predictions through a slice analysis using spaCy entity tags to see if performance varies by the type of content in a statement.

| Entity type present | Macro-F1 | Count |
|--------------------|----------|-------|
| PERSON | 0.2172 | 493 |
| ORG | 0.2068 | 302 |
| GPE (location) | 0.2011 | 391 |
| No named entities | ~0.2242 | varies |

Statements that reference specific locations and organizations are consistently harder to classify, suggesting the model struggles with claims that require external institutional or geographic knowledge to verify.

---

## Party-level Slice Analysis

| Party | n | Macro-F1 |
|-------|---|----------|
| Republican | 571 | 0.2231 |
| Democrat | 406 | 0.2715 |
| None | 214 | 0.2641 |

Republican-affiliated statements achieve a Macro-F1 of 0.2231, substantially below the overall test Macro-F1 of 0.2602. Democrat-affiliated statements achieve 0.2715, above the overall average. This disparity likely reflects the imbalance in the training set and is documented here without softening.

---

## RL Component

The bandit agent uses epsilon-greedy exploration over nine discretized threshold values (0.1 to 0.9). After 500 episodes it converged on a threshold of 0.8, selecting it 378 out of 500 times. The reward function penalizes missed misinformation (false negatives) at -2.0, which is four times heavier than the false positive penalty of -0.5. The agent was evaluated with real DistilBERT softmax outputs and outperforms a fixed threshold baseline of 0.5 in cumulative reward.

---

## Error Analysis

The confusion matrix shows the model most commonly confuses adjacent veracity classes. Pants-fire statements are frequently predicted as false (37 out of 92), and barely-true statements are frequently predicted as false (61 out of 212). This pattern of predicting toward the middle of the veracity spectrum is consistent with the model learning a conservative classification strategy.

---

## Known Limitations

- All models are trained on US political statements only — does not generalize to other domains or languages
- Current best Macro-F1 of 0.2602 means a significant portion of predictions remain incorrect
- Calibration is unreliable above 0.5 probability — conservative threshold is necessary
- The system has not been tested on data outside the LIAR benchmark

---

## How to use this responsibly

This system is a research prototype built for a university course. It should not be used to make decisions about real content without human review. Every output comes with a confidence score — treat low-confidence predictions with extra skepticism. Do not use this system for anything beyond research and educational purposes.
