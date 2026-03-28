# A Multi-Component NLP System for Fake News and Misinformation Detection

**6INTELSY — Intelligent Systems | 6DANCS — Data Analytics for Computer Science**
**AY 2025-2026, 2nd Semester**
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

# 3. Download model weights
# Get distilbert_best.pt from the v1.0 GitHub Release
# Place at: experiments/results/distilbert_best.pt

# 4. Run demo only (recommended)
bash run.sh --demo

# 5. Run full pipeline (retrain everything, ~60-75 min on GPU)
bash run.sh
```

> **Note:** Full pipeline requires a CUDA-capable GPU.
> Recommended: Google Colab with T4 GPU runtime (~60-75 min).

## Results

| Model | Test Accuracy | Test Macro-F1 |
|-------|-------------|---------------|
| Logistic Regression (TF-IDF) | 0.2478 | 0.2214 |
| Text-CNN (GloVe 300d) | 0.2478 | 0.2384 |
| DistilBERT (fine-tuned) | 0.2762 | 0.2602 |
| DistilBERT + Adam (Ablation 1) | 0.2897 | 0.2743 |
| DistilBERT + Early Stopping (Ablation 2) | 0.2762 | 0.2654 |

### Party-level Slice Analysis

| Party | n | Macro-F1 |
|-------|---|----------|
| Republican | 571 | 0.2231 |
| Democrat | 406 | 0.2715 |
| None | 214 | 0.2641 |

## Model Weights

The fine-tuned DistilBERT model exceeds GitHub's 100MB file size limit and is not stored in the repository.

**Download:** Available as an asset on the v1.0 GitHub Release page.

Place the file at:
```
experiments/results/distilbert_best.pt
```

## Repo Structure
```
project-root/
  README.md
  requirements.txt
  run.sh
  data/
    get_data.py
    README.md
    raw/
  src/
    data_pipeline.py
    train.py
    eval.py
    demo.py
    rl_agent.py
    models/
      __init__.py
      text_cnn.py
      ner_module.py
    utils/
  notebooks/
    01_eda.ipynb
    02_eda.ipynb
  experiments/
    configs/
    logs/
    results/
      week2/
        class_distribution.png
        text_length.png
        textcnn_learning_curves.png
        bandit_learning_curves.png
        baseline_lr_results.json
        textcnn_results.json
        bandit_results.json
      week3/
        distilbert_learning_curves.png
        distilbert_confusion_matrix.png
        distilbert_calibration_curves.png
        ablation1_adam_vs_adamw.png
        ablation2_early_stopping.png
        party_slice_analysis.png
        ablation1_results.json
        ablation2_results.json
        eval_results.json
  docs/
    proposal (Week 1).pdf
    checkpoint (Week 2).pdf
    final_report (Week 3).pdf
    slides.pdf
    model_card.md
    ethics_statement.md
```

## Release Tags

- `v0.1` — Proposal & Setup (Week 1)
- `v0.5` — Week 2 Checkpoint
- `v0.9` — Release Candidate (Week 3)
- `v1.0` — Final Submission
