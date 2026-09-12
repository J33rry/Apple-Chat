"""
Download the TWCS (Customer Support on Twitter) dataset if not already present.

Usage:
    python setup_data.py

This script attempts to download the dataset using kagglehub.
If kagglehub is not installed, it falls back to the kaggle CLI.
If neither is available, it prints manual download instructions.

The dataset is sourced from:
    https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
"""

import os
import shutil

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "twcs.csv")
KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"


def download_with_kagglehub():
    """Download using kagglehub (pip install kagglehub)."""
    import kagglehub
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    # kagglehub downloads to a cache dir — find the CSV and copy it
    for root, _, files in os.walk(path):
        for f in files:
            if f == "twcs.csv":
                src = os.path.join(root, f)
                shutil.copy2(src, CSV_PATH)
                print(f"Copied {src} -> {CSV_PATH}")
                return True
    print(f"Warning: twcs.csv not found in downloaded path: {path}")
    return False


def download_with_kaggle_cli():
    """Download using the kaggle CLI (pip install kaggle)."""
    import subprocess
    print("Downloading dataset via kaggle CLI...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", DATA_DIR, "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode == 0 and os.path.exists(CSV_PATH):
        print(f"Downloaded to {CSV_PATH}")
        return True
    print(f"kaggle CLI failed: {result.stderr.strip()}")
    return False


def print_manual_instructions():
    """Print manual download instructions as a last resort."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print(f"""
Could not auto-download the dataset.

Steps:
  1. Go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
  2. Click 'Download' (requires a free Kaggle account)
  3. Extract the ZIP file
  4. Place 'twcs.csv' into the '{DATA_DIR}/' directory

Alternatively, install kagglehub and re-run this script:
  pip install kagglehub
  python setup_data.py
""")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        size_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
        print(f"Dataset already exists: {CSV_PATH} ({size_mb:.1f} MB)")
        return

    print(f"Dataset not found at {CSV_PATH}. Attempting download...\n")

    # Strategy 1: kagglehub
    try:
        if download_with_kagglehub():
            return
    except ImportError:
        print("kagglehub not installed, trying kaggle CLI...")
    except Exception as e:
        print(f"kagglehub failed: {e}")

    # Strategy 2: kaggle CLI
    try:
        if download_with_kaggle_cli():
            return
    except FileNotFoundError:
        print("kaggle CLI not found.")
    except Exception as e:
        print(f"kaggle CLI failed: {e}")

    # Strategy 3: Manual instructions
    print_manual_instructions()


if __name__ == "__main__":
    main()
