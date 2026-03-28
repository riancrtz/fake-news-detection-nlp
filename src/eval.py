"""
eval.py
Full evaluation script for the fake news detection pipeline.
Generates confusion matrix, calibration curves, party-level slice analysis,
and saves all results to experiments/results/.

Run: python src/eval.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, classification_report
)
from sklearn.calibration import calibration_curve
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from data_pipeline import load_all_splits
from models.text_cnn import TextCNN

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ["pants-fire", "false", "barely-true",
               "half-true", "mostly-true", "true"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
splits  = load_all_splits()
test_df = splits['test']
X_test  = test_df['statement'].tolist()
y_test  = test_df['label_id'].tolist()

# ── 2. Load DistilBERT ─────────────────────────────────────────────────────────
print("Loading DistilBERT model...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class LiarBertDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {
            'input_ids':      self.encodings['input_ids'][i],
            'attention_mask': self.encodings['attention_mask'][i],
            'labels':         self.labels[i]
        }

test_ds     = LiarBertDataset(X_test, y_test, tokenizer)
test_loader = DataLoader(test_ds, batch_size=32)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=6
).to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(RESULTS_DIR, 'distilbert_best.pt'),
    map_location=DEVICE
), strict=False)
model.eval()
print("Model loaded!")

# ── 3. Get predictions ─────────────────────────────────────────────────────────
print("Running inference on test set...")
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=1).cpu().numpy()
        preds   = probs.argmax(axis=1)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].numpy())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── 4. Overall metrics ─────────────────────────────────────────────────────────
test_acc = accuracy_score(all_labels, all_preds)
test_f1  = f1_score(all_labels, all_preds, average='macro')
print(f"\n── DistilBERT Test Results ──")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Test Macro-F1  : {test_f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ── 5. Confusion matrix ────────────────────────────────────────────────────────
cm   = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('DistilBERT Confusion Matrix (Test Set)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'distilbert_confusion_matrix.png'))
plt.close()
print("Confusion matrix saved!")

# ── 6. Calibration curves ──────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
for i, class_name in enumerate(CLASS_NAMES):
    binary_labels = (all_labels == i).astype(int)
    class_probs   = all_probs[:, i]
    try:
        fraction_pos, mean_pred = calibration_curve(
            binary_labels, class_probs, n_bins=10)
        plt.plot(mean_pred, fraction_pos, marker='o',
                 label=class_name, alpha=0.7)
    except:
        pass
plt.title('DistilBERT Calibration Curves (per class)')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'distilbert_calibration_curves.png'))
plt.close()
print("Calibration curves saved!")

# ── 7. Party-level slice analysis ─────────────────────────────────────────────
print("\n── Party-level Slice Analysis ──")
test_df = test_df.copy()
test_df['pred'] = all_preds

top_parties  = ['republican', 'democrat', 'none']
party_results = {}

for party in top_parties:
    subset = test_df[test_df['party'] == party]
    if len(subset) > 0:
        f1  = f1_score(subset['label_id'], subset['pred'], average='macro')
        acc = accuracy_score(subset['label_id'], subset['pred'])
        party_results[party] = {
            "n": len(subset),
            "macro_f1": round(f1, 4),
            "accuracy": round(acc, 4)
        }
        print(f"{party:<15} n={len(subset):>4} | "
              f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

parties  = list(party_results.keys())
f1_scores = [party_results[p]['macro_f1'] for p in parties]
counts    = [party_results[p]['n'] for p in parties]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(parties, f1_scores,
              color=['steelblue', 'darkorange', 'gray'])
ax.axhline(y=test_f1, color='red', linestyle='--',
           label=f'Overall Macro-F1 ({test_f1:.4f})')
ax.set_title('Party-level Slice Analysis — DistilBERT Macro-F1')
ax.set_xlabel('Party Affiliation')
ax.set_ylabel('Macro-F1')
ax.legend()
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f'n={count}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'party_slice_analysis.png'))
plt.close()
print("Party slice analysis saved!")

# ── 8. Save all results ────────────────────────────────────────────────────────
eval_results = {
    "distilbert_test": {
        "accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1, 4)
    },
    "party_slice_analysis": party_results,
    "ablation_summary": {
        "ablation1_adam_vs_adamw": {
            "adam_macro_f1":  0.2743,
            "adamw_macro_f1": 0.2690,
            "winner": "Adam"
        },
        "ablation2_early_stopping": {
            "with_early_stopping_macro_f1":    0.2654,
            "without_early_stopping_macro_f1": 0.2602,
            "winner": "With Early Stopping"
        }
    }
}

with open(os.path.join(RESULTS_DIR, 'eval_results.json'), 'w') as f:
    json.dump(eval_results, f, indent=2)
print("\nAll evaluation results saved to experiments/results/eval_results.json")
print("\nEvaluation complete!")
