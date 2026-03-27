"""
demo.py
Live demonstration of the fake news detection system.
Runs DistilBERT predictions on example political statements.

Run: python src/demo.py
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "results")

LABEL_NAMES = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=6
).to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(RESULTS_DIR, 'distilbert_best.pt'),
    map_location=DEVICE
))
model.eval()
print("Model loaded!\n")

# ── NER demo ───────────────────────────────────────────────────────────────────
import spacy
nlp = spacy.load("en_core_web_sm")

def show_entities(text):
    """Extract and display named entities from a statement."""
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    print(f"Statement : '{text}'")
    print(f"PERSON    : {entities['PERSON'] or 'none'}")
    print(f"ORG       : {entities['ORG'] or 'none'}")
    print(f"GPE       : {entities['GPE'] or 'none'}")

# ── Prediction function ────────────────────────────────────────────────────────
def predict(statement, threshold=0.4, production_threshold=0.8):
    """
    Predict veracity of a political statement.
    Uses lower threshold for demo display purposes.
    Production threshold is 0.8 as learned by the bandit agent.
    """
    inputs = tokenizer(
        statement,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = F.softmax(outputs.logits, dim=1)[0]
        pred    = probs.argmax().item()
        conf    = probs[pred].item()

    print(f"Statement : '{statement}'")
    print(f"Prediction: {LABEL_NAMES[pred].upper()}")
    print(f"Confidence: {conf:.2%}")
    print(f"Production threshold (bandit): {production_threshold} — ", end="")
    if conf < production_threshold:
        print("would ABSTAIN and flag for human review")
    else:
        print("would ACT on this prediction")

    print(f"\nAll class probabilities:")
    for label, prob in zip(LABEL_NAMES.values(), probs):
        bar = '█' * int(prob.item() * 20)
        print(f"  {label:<15} {prob.item():.2%}  {bar}")

# ── Run demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("NER DEMO — Named Entity Recognition")
    print("=" * 60)
    show_entities(
        "Barack Obama said that Medicare spending has increased "
        "under his administration in Texas."
    )

    print("\n" + "=" * 60)
    print("PREDICTION DEMO 1 — Likely FALSE statement")
    print("=" * 60)
    predict(
        "The unemployment rate under Obama was the highest in history."
    )

    print("\n" + "=" * 60)
    print("PREDICTION DEMO 2 — Likely TRUE statement")
    print("=" * 60)
    predict(
        "The United States has the largest military budget in the world."
    )

    print("\n" + "=" * 60)
    print("PREDICTION DEMO 3 — Ambiguous statement")
    print("=" * 60)
    predict(
        "Texas has more wind energy capacity than any other state."
    )

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
