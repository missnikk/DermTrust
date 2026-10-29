"""
ml/prepare_balanced_manifest.py
Prepares training manifest with verified SCIN Melasma cases added to training set.
Ensures 0 collisions across train, val, and locked test sets.
"""

from pathlib import Path
import hashlib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"

BASE_TRAIN = DATA_DIR / "train_6class_manifest.csv"
BASE_VAL = DATA_DIR / "val_6class_manifest.csv"
LOCKED_TEST = DATA_DIR / "test_5class_manifest.csv"
SCIN_DIR = ROOT_DIR / "ml" / "external_datasets" / "scin_melasma"

OUT_TRAIN = DATA_DIR / "train_6class_manifest_balanced.csv"

def compute_md5(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def main():
    train_df = pd.read_csv(BASE_TRAIN)
    val_df = pd.read_csv(BASE_VAL)
    test_df = pd.read_csv(LOCKED_TEST)

    val_hashes = set(val_df["md5hash"])
    test_hashes = set(test_df["md5hash"])
    train_hashes = set(train_df["md5hash"])

    print(f"Base train rows: {len(train_df)}")

    # Add SCIN melasma cases
    new_rows = []
    scin_files = sorted(SCIN_DIR.glob("*.png"))
    for f in scin_files:
        h = compute_md5(f)
        if h in val_hashes or h in test_hashes:
            print(f"Skipping SCIN {f.name} (found in val or test!)")
            continue
        if h in train_hashes:
            print(f"SCIN {f.name} already in train.")
            continue
        new_rows.append({
            "image_id": f"scin_mel_{f.stem[:12]}",
            "source_dataset": "SCIN_Melasma",
            "original_filename": f.name,
            "local_path": str(f.resolve()).replace("\\", "/"),
            "verified_label": "melasma",
            "category": "melasma",
            "fitzpatrick_scale": 4,
            "skin_tone_group": "darker",
            "is_augmented": False,
            "parent_image_id": f.stem,
            "license": "CC-BY 4.0",
            "attribution": "Google SCIN Dataset",
            "md5hash": h,
            "sha256": "uncomputed",
            "phash": "uncomputed",
            "leakage_group": f"scin_{f.stem[:8]}"
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([train_df, new_df], ignore_index=True)
        print(f"Added {len(new_rows)} SCIN Melasma cases to training manifest.")
    else:
        combined_df = train_df.copy()

    # Verify no collisions with val or test
    comb_hashes = set(combined_df["md5hash"])
    assert len(comb_hashes.intersection(val_hashes)) == 0, "Train vs Val collision!"
    assert len(comb_hashes.intersection(test_hashes)) == 0, "Train vs Test collision!"

    combined_df.to_csv(OUT_TRAIN, index=False)
    print(f"Balanced training manifest written: {OUT_TRAIN} (N={len(combined_df)})")
    print("\nClass distribution in balanced manifest:")
    print(combined_df["verified_label"].value_counts())

if __name__ == "__main__":
    main()
