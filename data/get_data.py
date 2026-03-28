"""
get_data.py
Downloads and extracts the LIAR dataset from UCSB.
Run: python data/get_data.py
"""

import os
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
ZIP_PATH = os.path.join(DATA_DIR, "liar_dataset.zip")
URL      = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(os.path.join(DATA_DIR, "train.tsv")):
        print("Data already downloaded — skipping.")
        return

    print("Downloading LIAR dataset...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Download complete!")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete!")

    print(f"Files: {os.listdir(DATA_DIR)}")

if __name__ == "__main__":
    download_data()
