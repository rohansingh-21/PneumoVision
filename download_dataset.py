"""
Download the Chest X-Ray Pneumonia dataset from Kaggle.

Prerequisites:
  1. Install kaggle: pip install kaggle
  2. Place your kaggle.json API key in ~/.kaggle/kaggle.json
     (Download from https://www.kaggle.com/settings > API > Create New Token)

Usage:
  python download_dataset.py
"""
import os
import subprocess
import zipfile

DATASET = "paultimothymooney/chest-xray-pneumonia"
ZIP_FILE = "chest-xray-pneumonia.zip"
OUTPUT_DIR = "."

def main():
    print("Downloading dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET],
        check=True
    )

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(OUTPUT_DIR)

    print("Removing zip file...")
    os.remove(ZIP_FILE)

    print("Dataset ready at ./chest_xray/")
    print("  train/  (NORMAL + PNEUMONIA)")
    print("  test/   (NORMAL + PNEUMONIA)")
    print("  val/    (NORMAL + PNEUMONIA)")


if __name__ == "__main__":
    main()
