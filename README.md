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

> **Which option should you choose?** If you have a GPU, use Option 1 or 3. If you don't have a GPU or want the easiest setup, use Option 2 (Google Colab) — it's free and requires no local installation.

### Option 1 — Local (Linux/Mac)
```bash
# 1. Clone the repo
git clone https://github.com/riancrtz/fake-news-detection-nlp.git
cd fake-news-detection-nlp

# 2. Setup environment
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Download LIAR dataset (required for "bash run.sh --demo" only, full pipeline "bash run.sh" downloads it automatically)
python data/get_data.py

# 4. Download distilbert_best.pt from the v1.0 GitHub Release
#    Place at: experiments/results/distilbert_best.pt
#    Note: Running the full pipeline (step 6) will overwrite this file
#    with a newly trained model. Download again if needed.

# 5. Run this command if you want to run the demo only
bash run.sh --demo

# 6. Run these commands if you want to run the full pipeline (retrain everything)
bash run.sh                   
```
> **Note:** All results including plots and JSON metrics are saved to `experiments/results/`.

### Option 2 — Google Colab (Recommended)
Run these cells in a new Colab notebook with T4 GPU runtime enabled:
```python
# Cell 1 — Clone and setup
import os

if not os.path.exists('fake-news-detection-nlp'):
    !git clone https://github.com/riancrtz/fake-news-detection-nlp.git

%cd /content/fake-news-detection-nlp
!pip install -r requirements.txt -q
!python -m spacy download en_core_web_sm -q

import spacy.cli
spacy.cli.download("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()
print("Setup complete!")

# Cell 2 — Download LIAR dataset
!python data/get_data.py    

# Cell 3 — Load model weights from Google Drive
# First, download distilbert_best.pt from the v1.0 GitHub Release
# Then upload it to your own Google Drive root folder (My Drive)
# Note: Running the full pipeline (step 6) will overwrite this file
#       with a newly trained model. Download again if needed.
from google.colab import drive
drive.mount('/content/drive')
import shutil, os
os.makedirs('experiments/results', exist_ok=True)
shutil.copy('/content/drive/MyDrive/distilbert_best.pt', 'experiments/results/distilbert_best.pt')
print("Model weights loaded!")

# Cell 4 — Run this command if you want to run the demo only
!python src/demo.py

# Cell 5 — Run these commands if you want to run the full pipeline (retrain everything)
!python src/train.py            # trains Logistic Regression, Text-CNN, and DistilBERT
!python src/eval.py             # evaluates DistilBERT, generates plots and metrics
!python src/rl_agent.py         # trains contextual bandit agent
!python src/demo.py             # runs live demo using DistilBERT + NER + bandit
```
> **Note:** All results including plots and JSON metrics are saved to `experiments/results/` within the Colab session. Download them manually or copy to Google Drive before the session ends.

### Option 3 — Local (Windows PowerShell)
```
# 1. Download the repo zip from GitHub and extract it
# Navigate into the inner folder:
cd fake-news-detection-nlp-main
cd fake-news-detection-nlp-main

# 2. Setup environment
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Download data
python data/get_data.py        # downloads LIAR dataset

# 4. Download distilbert_best.pt from the v1.0 GitHub Release
#    Place at: experiments/results/distilbert_best.pt
#    Note: Running the full pipeline (step 6) will overwrite this file
#    with a newly trained model. Download again if needed.

# 5. Run this command if you want to run the demo only 
python src/demo.py

# 6. Run these commands if you want to run the full pipeline (retrain everything)
python src/train.py            # trains Logistic Regression, Text-CNN, and DistilBERT
python src/eval.py             # evaluates DistilBERT, generates plots and metrics
python src/rl_agent.py         # trains contextual bandit agent
python src/demo.py             # runs live demo using DistilBERT + NER + bandit
```
> **Note:** All results including plots and JSON metrics are saved to `experiments/results/`.

> **Note:** Full pipeline requires a CUDA-capable GPU. Recommended: Google Colab with T4 GPU runtime (~60-75 min). Running on CPU is supported for demo only.

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

> **Note on model components:** The four AI components are organized as follows — Text-CNN and NER module are in `src/models/` because they are standalone reusable classes. The contextual bandit RL agent is in `src/rl_agent.py` because it operates as a decision-making agent on top of the classifier outputs rather than as a model itself. DistilBERT fine-tuning is handled directly in `src/train.py` and `src/eval.py` via the HuggingFace transformers library because it does not require a custom architecture definition.

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
