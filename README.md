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

# 2. Download data
python data/get_data.py

# 3. Run full pipeline
bash run.sh
```

## Results (to be updated)
| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| Majority baseline | - | - |
| Text-CNN | - | - |
| DistilBERT (fine-tuned) | - | - |

## Repo Structure
See `docs/` for proposal, checkpoint, and final report.

## Release Tags
- `v0.1` — Proposal & Setup
- `v0.9` — Release Candidate
- `v1.0` — Final Submission
