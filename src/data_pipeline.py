"""
data_pipeline.py
Loads, cleans, and preprocesses the LIAR dataset for both
the Text-CNN and DistilBERT branches.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Column names from the LIAR dataset TSV files ──────────────────────────────
COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party", "barely_true_count",
    "false_count", "half_true_count", "mostly_true_count",
    "pants_fire_count", "context"
]

# ── Label mapping (6 classes) ──────────────────────────────────────────────────
LABEL_MAP = {
    "pants-fire": 0,
    "false":      1,
    "barely-true": 2,
    "half-true":  3,
    "mostly-true": 4,
    "true":       5
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_split(split="train"):
    """
    Load a single split (train, valid, test) from the LIAR dataset.
    Returns a cleaned pandas DataFrame.
    """
    filename = f"{split}.tsv"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"{filepath} not found. Run data/get_data.py first."
        )

    df = pd.read_csv(filepath, sep="\t", header=None, names=COLUMNS)

    # Drop rows with missing statements or labels
    df = df.dropna(subset=["statement", "label"])

    # Normalize labels
    df = df[df["label"].isin(LABEL_MAP.keys())]
    df["label_id"] = df["label"].map(LABEL_MAP)

    # Clean text
    df["statement"] = df["statement"].str.strip()
    df["statement"] = df["statement"].str.replace(r"\s+", " ", regex=True)

    return df.reset_index(drop=True)


def load_all_splits():
    """
    Load train, validation, and test splits.
    Returns a dict with keys: train, valid, test.
    """
    splits = {}
    for split in ["train", "valid", "test"]:
        splits[split] = load_split(split)
        print(f"[{split}] loaded: {len(splits[split])} samples")
    return splits


def get_class_distribution(df, split_name=""):
    """
    Print class distribution for a given split.
    """
    print(f"\nClass distribution — {split_name}")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label:<15} {count:>5}  ({pct:.1f}%)")


if __name__ == "__main__":
    splits = load_all_splits()
    for name, df in splits.items():
        get_class_distribution(df, split_name=name)
    print("\nData pipeline ready!")
