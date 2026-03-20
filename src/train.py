"""
train.py
Training script for Logistic Regression baseline and Text-CNN.
Run: python src/train.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.append(os.path.dirname(__file__))
from data_pipeline import load_all_splits
from models.text_cnn import TextCNN

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
splits = load_all_splits()
X_train = splits['train']['statement'].tolist()
y_train = splits['train']['label_id'].tolist()
X_valid = splits['valid']['statement'].tolist()
y_valid = splits['valid']['label_id'].tolist()
X_test  = splits['test']['statement'].tolist()
y_test  = splits['test']['label_id'].tolist()

# ── 1. Logistic Regression baseline ───────────────────────────────────────────
print("\n=== Logistic Regression Baseline ===")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_valid_tfidf = tfidf.transform(X_valid)
X_test_tfidf  = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_train_tfidf, y_train)

y_pred_valid = lr.predict(X_valid_tfidf)
y_pred_test  = lr.predict(X_test_tfidf)

lr_results = {
    "model": "Logistic Regression (TF-IDF)",
    "validation": {
        "accuracy": round(accuracy_score(y_valid, y_pred_valid), 4),
        "macro_f1": round(f1_score(y_valid, y_pred_valid, average='macro'), 4)
    },
    "test": {
        "accuracy": round(accuracy_score(y_test, y_pred_test), 4),
        "macro_f1": round(f1_score(y_test, y_pred_test, average='macro'), 4)
    }
}
print(f"Val  Accuracy: {lr_results['validation']['accuracy']}")
print(f"Val  Macro-F1: {lr_results['validation']['macro_f1']}")
print(f"Test Accuracy: {lr_results['test']['accuracy']}")
print(f"Test Macro-F1: {lr_results['test']['macro_f1']}")

with open(os.path.join(RESULTS_DIR, 'baseline_lr_results.json'), 'w') as f:
    json.dump(lr_results, f, indent=2)
print("LR results saved!")

# ── 2. Text-CNN baseline ───────────────────────────────────────────────────────
print("\n=== Text-CNN Baseline ===")

MAX_LEN   = 64
EMBED_DIM = 300
EPOCHS    = 10
BATCH     = 64

def build_vocab(texts, max_vocab=20000):
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=MAX_LEN):
    tokens = text.lower().split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0] * (max_len - len(ids))
    return ids

class LiarDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.X = [encode(t, vocab) for t in texts]
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

vocab = build_vocab(X_train + X_valid + X_test)

train_loader = DataLoader(LiarDataset(X_train, y_train, vocab),
                          batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(LiarDataset(X_valid, y_valid, vocab), batch_size=BATCH)
test_loader  = DataLoader(LiarDataset(X_test,  y_test,  vocab), batch_size=BATCH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TextCNN(len(vocab), EMBED_DIM, num_classes=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            p = model(X_batch).argmax(dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y_batch.numpy())
    return (accuracy_score(labels, preds),
            f1_score(labels, preds, average='macro'))

best_val_f1 = 0
cnn_results = {"model": "Text-CNN (GloVe 300d)", "epochs": []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc, val_f1 = evaluate(valid_loader)
    avg_loss = total_loss / len(train_loader)
    cnn_results["epochs"].append({
        "epoch": epoch + 1,
        "loss": round(avg_loss, 4),
        "val_acc": round(val_acc, 4),
        "val_macro_f1": round(val_f1, 4)
    })
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} "
          f"| Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 
                   os.path.join(RESULTS_DIR, 'textcnn_best.pt'))

test_acc, test_f1 = evaluate(test_loader)
cnn_results["test"] = {
    "accuracy": round(test_acc, 4),
    "macro_f1": round(test_f1, 4)
}
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")

with open(os.path.join(RESULTS_DIR, 'textcnn_results.json'), 'w') as f:
    json.dump(cnn_results, f, indent=2)
print("CNN results saved!")
