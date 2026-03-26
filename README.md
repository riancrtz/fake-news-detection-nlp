# A Multi-Component NLP System for Fake News and Misinformation Detection

**6INTELSY Final Project — AY 2025-2026, 2nd Semester**
Holy Angel University — School of Computing

## Team
| Role | Member |
|------|--------|
| Project Lead / Integration | Rian Cortez |
| Data & Ethics Lead | Ryence Cortez |
| Modeling Lead | Ranz Cuarto |
| Evaluation & MLOps Lead | Mark Darius David |

## Project Overview
This system is an NLP-based misinformation detection pipeline combining:
- **Core Model:** Fine-tuned DistilBERT for veracity classification
- **CNN Component:** Text-CNN (Kim, 2014) as baseline and ablation
- **NLP Component:** NER-based provenance tagging (spaCy)
- **RL Component:** Contextual bandit for adaptive threshold tuning

## Dataset
[LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) — 12,836 labeled political statements from PolitiFact.

## Quick Start
```bash
# 1. Setup environment
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Download data
python data/get_data.py

# 3. Run full pipeline (train + eval, ~60-75 min on Colab T4)
bash run.sh
```

## Results
| Model | Test Accuracy | Test Macro-F1 |
|-------|-------------|---------------|
| Logistic Regression (TF-IDF) | 0.2478 | 0.2214 |
| Text-CNN (GloVe 300d) | 0.2478 | 0.2384 |
| DistilBERT (fine-tuned) | 0.2762 | 0.2602 |
| DistilBERT + Adam | 0.2897 | 0.2743 |
| DistilBERT + Early Stopping | 0.2762 | 0.2654 |

## Party-level Slice Analysis
| Party | n | Macro-F1 |
|-------|---|----------|
| Republican | 571 | 0.2231 |
| Democrat | 406 | 0.2715 |
| None | 214 | 0.2641 |

## Repo Structure
```
project-root/
  README.md
  requirements.txt
  run.sh
  data/
    get_data.py
    README.md
  src/
    data_pipeline.py
    train.py
    eval.py
    rl_agent.py
    models/
      text_cnn.py
      ner_module.py
  notebooks/
    01_eda.ipynb
  experiments/
    results/
  docs/
    proposal.pdf
    checkpoint.pdf
    final_report.pdf
    slides.pdf
    model_card.md
    ethics_statement.md
```

## Release Tags
- `v0.1` — Proposal & Setup (Week 1)
- `v0.5` — Week 2 Checkpoint
- `v0.9` — Release Candidate (Week 3)
- `v1.0` — Final Submission
