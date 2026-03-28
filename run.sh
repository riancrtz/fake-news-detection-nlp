#!/bin/bash
# run.sh
# Full pipeline for the fake news detection system.
# Usage:
#   bash run.sh          — full pipeline (download, train, eval, demo)
#   bash run.sh --demo   — demo only using pre-trained model

set -e

echo "============================================"
echo " Fake News Detection NLP Pipeline"
echo "============================================"

if [ "$1" == "--demo" ]; then
    echo ""
    echo "Demo mode — using pre-trained model"
    echo "Make sure distilbert_best.pt is at:"
    echo "experiments/results/distilbert_best.pt"
    echo ""
    echo "[1/1] Running demo..."
    python src/demo.py
    echo ""
    echo "============================================"
    echo " Demo complete!"
    echo "============================================"
    exit 0
fi

# Step 1 — Download data
echo ""
echo "[1/4] Downloading LIAR dataset..."
python data/get_data.py

# Step 2 — Train all models (Logistic Regression, Text-CNN, DistilBERT)
echo ""
echo "[2/4] Training all models (~60-75 min on GPU)..."
python src/train.py

# Step 3 — Evaluate
echo ""
echo "[3/4] Running evaluation..."
python src/eval.py

# Step 4 — Demo
echo ""
echo "[4/4] Running demo..."
python src/demo.py

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Results saved to experiments/results/"
echo "============================================"
