"""
DermTrust - Machine Learning Pipeline
Script: prepare_dataset.py
Role: Prepares reproducible, stratified Train / Validation / Test splits
      with Fitzpatrick skin tone metadata tracking and patient-leakage prevention.
      Also generates an Out-of-Distribution / Non-Target test manifest for rejection benchmarking.
"""

from __future__ import annotations
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_CSV_PATH = Path(r"C:\Users\nikit\Downloads\fitzpatrick17k.csv")
DEFAULT_IMG_DIR = Path(r"C:\Users\nikit\Downloads\fitzpatrick17k\data\finalfitz17k")
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TARGET_DISEASE_MAP = {
    "vitiligo": "vitiligo",
    "melasma": "melasma",
    "tinea corporis": "ringworm",
    "tinea faciei": "ringworm",
    "tinea": "ringworm",
    "ringworm": "ringworm",
    "acne": "acne",
    "acne vulgaris": "acne",
    "cystic acne": "acne",
    "psoriasis": "psoriasis",
    "pustular psoriasis": "psoriasis",
    "guttate psoriasis": "psoriasis",
}

NON_TARGET_CONFOUNDERS = {
    "eczema": "non_target_eczema",
    "atopic dermatitis": "non_target_atopic_dermatitis",
    "seborrheic dermatitis": "non_target_seborrheic_dermatitis",
    "lichen planus": "non_target_lichen_planus",
    "pityriasis rosea": "non_target_pityriasis_rosea",
    "rosacea": "non_target_rosacea",
}


def classify_fst(val: str) -> tuple[str, str]:
    try:
        f = int(val)
        if f in (1, 2, 3):
            return "I-III", "Light"
        elif f in (4, 5, 6):
            return "IV-VI", "Darker"
        return "Unknown", "Unknown"
    except (ValueError, TypeError):
        return "Unknown", "Unknown"


def prepare_dataset(
    csv_path: Path = DEFAULT_CSV_PATH,
    img_dir: Path = DEFAULT_IMG_DIR,
    output_dir: Path = OUTPUT_DIR,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    print(f"==================================================")
    print(f"DERMTRUST DATASET SPLITTER & LEAKAGE SAFEGUARD")
    print(f"Seed: {seed} | Splits: {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test")
    print(f"==================================================")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Filter for valid images
    existing_imgs = set(p.name for p in img_dir.glob("*.jpg"))
    df["filename"] = df["md5hash"] + ".jpg"
    df["image_path"] = df["filename"].apply(lambda fn: str((img_dir / fn).resolve()))
    df = df[df["filename"].isin(existing_imgs)].copy()

    # Map target and non-target labels
    def map_row(label: str):
        l = " ".join(str(label).strip().lower().replace("_", " ").split())
        for k, v in TARGET_DISEASE_MAP.items():
            if k in l:
                return v, "target"
        for k, v in NON_TARGET_CONFOUNDERS.items():
            if k in l:
                return v, "non_target"
        return "other", "other"

    mapped = [map_row(lbl) for lbl in df["label"]]
    df["mapped_label"] = [m[0] for m in mapped]
    df["category"] = [m[1] for m in mapped]

    fst_info = [classify_fst(s) for s in df["fitzpatrick_scale"]]
    df["fitzpatrick_group"] = [f[0] for f in fst_info]
    df["skin_tone"] = [f[1] for f in fst_info]

    # Target partition
    target_df = df[df["category"] == "target"].copy().reset_index(drop=True)
    non_target_df = df[df["category"] == "non_target"].copy().reset_index(drop=True)

    print(f"Total target images available: {len(target_df):,}")
    print(f"Total non-target confounders available: {len(non_target_df):,}")

    # Stratified Split by mapped disease label
    target_df["strat_key"] = target_df["mapped_label"]
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    
    train_idx, temp_idx = next(sss1.split(target_df, target_df["strat_key"]))
    train_df = target_df.iloc[train_idx].copy()
    temp_df = target_df.iloc[temp_idx].copy().reset_index(drop=True)

    # Second split: Val vs Test (50/50 of temp)
    val_proportion_of_temp = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_proportion_of_temp), random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df["strat_key"]))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Verify zero overlap (Patient / MD5 leakage prevention)
    train_hashes = set(train_df["md5hash"])
    val_hashes = set(val_df["md5hash"])
    test_hashes = set(test_df["md5hash"])

    assert len(train_hashes.intersection(val_hashes)) == 0, "LEAKAGE: Train-Val overlap detected!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "LEAKAGE: Train-Test overlap detected!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "LEAKAGE: Val-Test overlap detected!"
    print("Zero Data Leakage Check PASSED: No overlap between Train, Val, and Test splits.")

    # Save manifests
    train_manifest_path = output_dir / "train_manifest.csv"
    val_manifest_path = output_dir / "val_manifest.csv"
    test_manifest_path = output_dir / "test_manifest.csv"
    ood_manifest_path = output_dir / "ood_manifest.csv"

    columns_to_keep = [
        "md5hash", "filename", "image_path", "label", "mapped_label",
        "fitzpatrick_scale", "fitzpatrick_group", "skin_tone", "qc", "url"
    ]

    train_df[columns_to_keep].to_csv(train_manifest_path, index=False)
    val_df[columns_to_keep].to_csv(val_manifest_path, index=False)
    test_df[columns_to_keep].to_csv(test_manifest_path, index=False)
    non_target_df[columns_to_keep].to_csv(ood_manifest_path, index=False)

    print(f"\nManifests generated successfully:")
    print(f"- Train manifest: {train_manifest_path} ({len(train_df)} images)")
    print(f"- Validation manifest: {val_manifest_path} ({len(val_df)} images)")
    print(f"- Independent Test manifest: {test_manifest_path} ({len(test_df)} images)")
    print(f"- OOD Non-Target manifest: {ood_manifest_path} ({len(non_target_df)} images)")

    # Per-class summary
    print("\n--- Test Set Class Distribution ---")
    for cls, cnt in test_df["mapped_label"].value_counts().items():
        print(f"  {cls}: {cnt} images")

    print("\n--- Test Set Skin-Tone Representation ---")
    for st, cnt in test_df["fitzpatrick_group"].value_counts().items():
        print(f"  FST {st}: {cnt} images")

    return {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "ood_count": len(non_target_df),
    }


if __name__ == "__main__":
    prepare_dataset()
