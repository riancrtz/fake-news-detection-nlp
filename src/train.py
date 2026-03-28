"""
train.py
Training script for Logistic Regression baseline, Text-CNN, and DistilBERT.
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
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

cnn_model  = TextCNN(len(vocab), EMBED_DIM, num_classes=6).to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def evaluate_cnn(loader):
    cnn_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            p = cnn_model(X_batch).argmax(dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y_batch.numpy())
    return (accuracy_score(labels, preds),
            f1_score(labels, preds, average='macro'))

best_val_f1 = 0
cnn_results = {"model": "Text-CNN (GloVe 300d)", "epochs": []}

for epoch in range(EPOCHS):
    cnn_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(cnn_model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc, val_f1 = evaluate_cnn(valid_loader)
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
        torch.save(cnn_model.state_dict(),
                   os.path.join(RESULTS_DIR, 'textcnn_best.pt'))

test_acc, test_f1 = evaluate_cnn(test_loader)
cnn_results["test"] = {
    "accuracy": round(test_acc, 4),
    "macro_f1": round(test_f1, 4)
}
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")

with open(os.path.join(RESULTS_DIR, 'textcnn_results.json'), 'w') as f:
    json.dump(cnn_results, f, indent=2)
print("CNN results saved!")

# ── 3. DistilBERT fine-tuning ──────────────────────────────────────────────────
print("\n=== DistilBERT Fine-tuning ===")

BERT_EPOCHS = 5
BERT_BATCH  = 16
LR          = 2e-5

class LiarBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {key: val[i] for key, val in self.encodings.items()}
        item['labels'] = self.labels[i]
        return item

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=6
).to(device)

bert_train_loader = DataLoader(
    LiarBertDataset(X_train, y_train, tokenizer),
    batch_size=BERT_BATCH, shuffle=True
)
bert_valid_loader = DataLoader(
    LiarBertDataset(X_valid, y_valid, tokenizer),
    batch_size=BERT_BATCH
)
bert_test_loader = DataLoader(
    LiarBertDataset(X_test, y_test, tokenizer),
    batch_size=BERT_BATCH
)

bert_optimizer = optim.Adam(bert_model.parameters(), lr=LR)
total_steps = len(bert_train_loader) * BERT_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    bert_optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

def evaluate_bert(loader):
    bert_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)
            outputs   = bert_model(input_ids=input_ids, attention_mask=mask)
            preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch['labels'].numpy())
    return (accuracy_score(labels, preds),
            f1_score(labels, preds, average='macro'))

best_bert_val_f1 = 0
bert_results = {"model": "DistilBERT (fine-tuned)", "epochs": []}

for epoch in range(BERT_EPOCHS):
    bert_model.train()
    total_loss = 0
    for batch in bert_train_loader:
        input_ids = batch['input_ids'].to(device)
        mask      = batch['attention_mask'].to(device)
        labels_b  = batch['labels'].to(device)
        bert_optimizer.zero_grad()
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=mask,
            labels=labels_b
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        bert_optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(bert_train_loader)
    val_acc, val_f1 = evaluate_bert(bert_valid_loader)
    bert_results["epochs"].append({
        "epoch": epoch + 1,
        "loss": round(avg_loss, 4),
        "val_acc": round(val_acc, 4),
        "val_macro_f1": round(val_f1, 4)
    })
    print(f"Epoch {epoch+1}/{BERT_EPOCHS} | Loss: {avg_loss:.4f} "
          f"| Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

    if val_f1 > best_bert_val_f1:
        best_bert_val_f1 = val_f1
        torch.save(
            bert_model.state_dict(),
            os.path.join(RESULTS_DIR, 'distilbert_best.pt')
        )
        print(f"  New best model saved! (Val F1: {best_bert_val_f1:.4f})")

test_acc, test_f1 = evaluate_bert(bert_test_loader)
bert_results["test"] = {
    "accuracy": round(test_acc, 4),
    "macro_f1": round(test_f1, 4)
}
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")

with open(os.path.join(RESULTS_DIR, 'distilbert_results.json'), 'w') as f:
    json.dump(bert_results, f, indent=2)
print("DistilBERT results saved!")
print(f"\nAll training complete! Results saved to {RESULTS_DIR}")
