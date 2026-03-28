#!/bin/bash
# run.sh
# Full pipeline for the fake news detection system.
# Runs data download, training, evaluation, and demo.
# Usage: bash run.sh

set -e

echo "============================================"
echo " Fake News Detection NLP Pipeline"
echo "============================================"

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
