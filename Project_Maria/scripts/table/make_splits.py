"""
make_splits.py — Data Splits + Versioning for Maria
=====================================================
Reads maria_sft_dataset.jsonl (the dataset Maria already builds internally)
and splits it into train / valid / test sets using an 80/10/10 ratio.

It also prints a version hash for every split so you know exactly which
data went into each training run.

Usage:
    python make_splits.py
    python make_splits.py --input my_data.jsonl --ratio 80 10 10

Output files (in the same directory as this script):
    train.jsonl   — 80% of data, used for model training
    valid.jsonl   — 10% of data, used during training to check for overfitting
    test.jsonl    — 10% of data, NEVER seen during training; used for final eval

Why three splits?
    train  → the model learns from this
    valid  → we watch this to stop training before the model memorises train data
    test   → a clean, untouched benchmark; if test score = valid score, we're good
"""

import os
import sys
import json
import random
import hashlib
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_hash(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def load_jsonl(path: str) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list, path: str):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_splits(
    input_path: str,
    train_pct: float = 0.80,
    valid_pct: float = 0.10,
    seed: int = 42,
):
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        print("   Run Maria and collect some conversations first,")
        print("   then use the 'Harvest Data Now' button in the SFT menu.")
        sys.exit(1)

    data = load_jsonl(input_path)
    random.seed(seed)
    random.shuffle(data)

    n        = len(data)
    n_train  = int(n * train_pct)
    n_valid  = int(n * valid_pct)
    n_test   = n - n_train - n_valid

    train = data[:n_train]
    valid = data[n_train : n_train + n_valid]
    test  = data[n_train + n_valid :]

    train_path = os.path.join(BASE_DIR, "train.jsonl")
    valid_path = os.path.join(BASE_DIR, "valid.jsonl")
    test_path  = os.path.join(BASE_DIR, "test.jsonl")

    save_jsonl(train, train_path)
    save_jsonl(valid, valid_path)
    save_jsonl(test,  test_path)

    # ── Version record ─────────────────────────────────────────────────────────
    version_record = {
        "created_at":    datetime.now().isoformat(),
        "source_file":   input_path,
        "source_hash":   compute_hash(input_path),
        "seed":          seed,
        "split_ratio":   {"train": train_pct, "valid": valid_pct, "test": round(1 - train_pct - valid_pct, 4)},
        "counts":        {"total": n, "train": n_train, "valid": n_valid, "test": n_test},
        "split_hashes":  {
            "train": compute_hash(train_path),
            "valid": compute_hash(valid_path),
            "test":  compute_hash(test_path),
        }
    }

    version_path = os.path.join(BASE_DIR, "dataset_version.json")
    with open(version_path, "w") as f:
        json.dump(version_record, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Maria Data Splits")
    print(f"{'='*55}")
    print(f"  Source : {os.path.basename(input_path)}")
    print(f"  Total  : {n} examples")
    print(f"  Seed   : {seed}  (change this to get different splits)")
    print()
    print(f"  train.jsonl  → {n_train:>5} examples  hash={version_record['split_hashes']['train']}")
    print(f"  valid.jsonl  → {n_valid:>5} examples  hash={version_record['split_hashes']['valid']}")
    print(f"  test.jsonl   → {n_test:>5} examples  hash={version_record['split_hashes']['test']}")
    print()
    print(f"  Version saved → dataset_version.json")
    print(f"{'='*55}\n")

    print("  ℹ️  What to do next:")
    print("     1. Point your train_sft.py at  train.jsonl")
    print("     2. Point your trainer at        valid.jsonl  (for early stopping)")
    print("     3. Run  python run_eval.py  to score on test.jsonl logic")
    print()

    return version_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/valid/test splits for Maria")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(BASE_DIR, "maria_sft_dataset.jsonl"),
        help="Path to the source JSONL file (default: maria_sft_dataset.jsonl)"
    )
    parser.add_argument(
        "--ratio", "-r",
        nargs=3, type=int, default=[80, 10, 10],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Split ratio as integers summing to 100 (default: 80 10 10)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    total = sum(args.ratio)
    if total != 100:
        print(f"❌ Ratio must sum to 100, got {total}")
        sys.exit(1)

    train_pct = args.ratio[0] / 100
    valid_pct = args.ratio[1] / 100

    make_splits(args.input, train_pct, valid_pct, args.seed)