"""
DermTrust - Machine Learning Pipeline
Script: prepare_5class_dataset.py
Role: Generates reproducible, cluster-grouped, stratified Train / Validation / Test splits
      for the full 5-class target scope (Acne, Melasma, Psoriasis, Ringworm, Vitiligo)
      and Out-of-Distribution (OOD) confounders.
      Prevents data leakage by ensuring all related images/variants share the same partition.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "ml" / "data" / "unified_ingested_manifest.csv"
DATA_DIR = ROOT_DIR / "ml" / "data"


def assign_leakage_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns grouping keys so that:
    1. Melasma crops from the same volunteer donor (user_mel_2, user_mel_3, etc.) stay in the same split.
    2. Augmented variants (flips/rotations) stay in the same split as their parent.
    3. SkinDiseaseBD variants sharing similar pHash patterns stay in the same split.
    4. Fitzpatrick 17K images from the same MD5 or source cluster stay together.
    """
    groups = []
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        parent_id = str(row["parent_image_id"])
        label = str(row["verified_label"])
        source = str(row["source_dataset"])

        if label == "melasma":
            if "user_mel_2" in img_id:
                grp = "melasma_donor_2"
            elif "user_mel_3" in img_id:
                grp = "melasma_donor_3"
            elif "user_cheek_melasma" in img_id or "user_facial_melasma" in img_id:
                grp = "melasma_donor_1"
            elif "atlas_melasma" in img_id:
                grp = f"melasma_atlas_{parent_id}"
            else:
                grp = f"melasma_wiki_{parent_id[:12]}"
        elif "SkinDiseaseBD" in source:
            # Group by 4-hex prefix of perceptual hash to cluster visual near-duplicates
            ph = str(row.get("phash", ""))
            cluster_prefix = ph[:3] if len(ph) >= 3 else "skindbd"
            grp = f"skindbd_{label}_{cluster_prefix}"
        else:
            # Default: use parent_image_id
            grp = f"{label}_{parent_id}"
        
        groups.append(grp)

    df["leakage_group"] = groups
    return df


def prepare_5class_splits(seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    print("==================================================")
    print("DERMTRUST 5-CLASS DATASET SPLITTER & LEAKAGE GUARD")
    print(f"Seed: {seed} | Splits: {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test")
    print("==================================================")

    df = pd.read_csv(MANIFEST_PATH)
    target_df = df[df["category"] == "target"].copy().reset_index(drop=True)
    ood_df = df[df["category"] == "non_target"].copy().reset_index(drop=True)

    target_df = assign_leakage_groups(target_df)

    print(f"Total Target Images: {len(target_df)}")
    print(f"Total Leakage Groups: {target_df['leakage_group'].nunique()}")

    # Stratified Group Split using StratifiedGroupKFold
    # 7 folds = ~14.3% per fold. Fold 0 -> Test (14.3%), Fold 1 -> Val (14.3%), Folds 2-6 -> Train (71.4%)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    
    splits = list(sgkf.split(target_df, target_df["verified_label"], target_df["leakage_group"]))
    test_idx = splits[0][1]
    val_idx = splits[1][1]
    
    # Train is the remainder
    train_idx = np.setdiff1d(np.arange(len(target_df)), np.concatenate([test_idx, val_idx]))

    train_df = target_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = target_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = target_df.iloc[test_idx].copy().reset_index(drop=True)

    # Verify zero leakage across groups
    train_groups = set(train_df["leakage_group"])
    val_groups = set(val_df["leakage_group"])
    test_groups = set(test_df["leakage_group"])

    assert len(train_groups.intersection(val_groups)) == 0, "LEAKAGE: Train-Val group collision!"
    assert len(train_groups.intersection(test_groups)) == 0, "LEAKAGE: Train-Test group collision!"
    assert len(val_groups.intersection(test_groups)) == 0, "LEAKAGE: Val-Test group collision!"

    # Verify zero MD5 collisions
    train_md5 = set(train_df["md5hash"])
    val_md5 = set(val_df["md5hash"])
    test_md5 = set(test_df["md5hash"])

    assert len(train_md5.intersection(val_md5)) == 0, "LEAKAGE: Train-Val MD5 collision!"
    assert len(train_md5.intersection(test_md5)) == 0, "LEAKAGE: Train-Test MD5 collision!"
    assert len(val_md5.intersection(test_md5)) == 0, "LEAKAGE: Val-Test MD5 collision!"

    print("\nZero Data Leakage Verification: PASSED (Zero group or MD5 overlap).")

    # Save 5-class manifests
    train_path = DATA_DIR / "train_5class_manifest.csv"
    val_path = DATA_DIR / "val_5class_manifest.csv"
    test_path = DATA_DIR / "test_5class_manifest.csv"
    ood_path = DATA_DIR / "ood_5class_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    ood_df.to_csv(ood_path, index=False)

    print(f"\nManifests exported:")
    print(f"  Train: {train_path} ({len(train_df)} images)")
    print(f"  Val:   {val_path} ({len(val_df)} images)")
    print(f"  Test:  {test_path} ({len(test_df)} images)")
    print(f"  OOD:   {ood_path} ({len(ood_df)} images)")

    print("\n--- Per-Class Distribution Across Splits ---")
    summary_data = []
    for cls in sorted(target_df["verified_label"].unique()):
        n_train = sum(train_df["verified_label"] == cls)
        n_val = sum(val_df["verified_label"] == cls)
        n_test = sum(test_df["verified_label"] == cls)
        total = n_train + n_val + n_test
        summary_data.append({"Class": cls, "Train": n_train, "Val": n_val, "Test": n_test, "Total": total})
        print(f"  {cls.capitalize():<12}: Train={n_train:3d} | Val={n_val:3d} | Test={n_test:3d} | Total={total:4d}")

    return {
        "train_len": len(train_df),
        "val_len": len(val_df),
        "test_len": len(test_df),
        "summary": summary_data,
    }


if __name__ == "__main__":
    prepare_5class_splits()
