# Model Card — A Multi-Component NLP System for Fake News and Misinformation Detection

## Model Details
- **Developed by:** Rian Cortez, Ryence Cortez, Ranz Cuarto, Mark Darius David
- **Institution:** Holy Angel University — School of Computing
- **Date:** March 2026
- **Version:** 0.9 (Week 2 Checkpoint)
- **Model type:** Multi-component NLP pipeline (DistilBERT + Text-CNN + NER + Contextual Bandit)
- **Language:** English
- **License:** For academic research use only

---

## Intended Use
- **Primary use:** Assisting human fact-checkers in flagging potentially false political statements for review
- **Intended users:** Researchers, educators, and fact-checking organizations
- **Out-of-scope uses:** Automated content moderation, censorship, clinical or legal decision-making, non-English content

---

## Training Data
- **Dataset:** LIAR benchmark (Wang, 2017)
- **Source:** PolitiFact.com
- **Size:** 10,240 training / 1,284 validation / 1,267 test samples
- **Labels:** pants-fire, false, barely-true, half-true, mostly-true, true (6 classes)
- **Known biases:** Dataset skews toward US political content; Republican speakers (4,497) outnumber Democrat speakers (3,336) in training set

---

## Evaluation Results (Week 2 — Preliminary)

| Model | Test Accuracy | Test Macro-F1 |
|-------|-------------|---------------|
| Majority baseline | 0.2090 | — |
| Logistic Regression (TF-IDF) | 0.2478 | 0.2214 |
| Text-CNN (GloVe 300d) | 0.2478 | 0.2384 |
| DistilBERT (fine-tuned) | TBD | TBD |

---

## NER Slice Analysis (Preliminary)

| Entity Type | With Entity Macro-F1 | Without Entity Macro-F1 |
|-------------|---------------------|------------------------|
| PERSON | 0.2172 (n=493) | 0.2112 (n=774) |
| ORG | 0.2068 (n=302) | 0.2242 (n=965) |
| GPE | 0.2011 (n=391) | 0.2268 (n=876) |

Statements referencing specific locations and organizations are harder to classify correctly.

---

## RL Component
- **Type:** Contextual bandit with epsilon-greedy exploration
- **Action space:** Decision thresholds from 0.1 to 0.9
- **Best learned threshold:** 0.8
- **Status:** Preliminary — currently using simulated softmax outputs; will update with real DistilBERT outputs in Week 3

---

## Limitations
- Performance is low across all current models (Macro-F1 ~0.22—0.24) — expected for 6-class veracity classification on short text
- System is trained only on US political statements and may not generalize to other domains or languages
- Bandit agent currently uses simulated probabilities — final evaluation pending DistilBERT integration
- Text-CNN shows signs of overfitting after epoch 3

---

## Ethical Considerations
- System outputs must not be used as sole basis for content moderation decisions
- Known class imbalance by political party — see bias documentation in ethics_statement.md
- All outputs should be reviewed by a human fact-checker before any action is taken

---

## Caveats and Recommendations
- Always report calibration curves alongside F1 scores
- Do not deploy in any production environment without further validation
- Consult ethics_statement.md for full risk register and mitigations
