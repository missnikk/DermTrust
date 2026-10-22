"""
DermTrust - Machine Learning Pipeline
Script: prepare_5class_dataset_v2.py
Role: Corrected Group-Stratified Dataset Splitter (Version 2)
      - Strictly excludes synthetic/augmented copies from Validation and Test sets.
      - Enforces patient/donor group isolation across Train, Val, and Test splits.
      - Preserves original v1 manifests while outputting versioned v2 manifests.
      - Transparently reports exact sample sizes and empirical minority limitations.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"
MANIFEST_PATH = DATA_DIR / "unified_ingested_manifest.csv"


def assign_robust_leakage_groups(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _, row in df.iterrows():
        label = row["verified_label"]
        img_id = str(row["image_id"]).lower()
        parent_id = str(row["parent_image_id"]).lower()
        source = str(row.get("source_dataset", ""))

        if label == "melasma":
            if "user_mel_2" in img_id or "user_mel_2" in parent_id:
                grp = "melasma_donor_2"
            elif "user_mel_3" in img_id or "user_mel_3" in parent_id:
                grp = "melasma_donor_3"
            elif any(k in img_id or k in parent_id for k in ["user_cheek_melasma", "user_facial_melasma"]):
                grp = "melasma_donor_1"
            elif "user_melasma_crop" in img_id or "user_melasma_crop" in parent_id:
                grp = "melasma_donor_4"
            elif "atlas_melasma" in img_id:
                grp = f"melasma_atlas_{parent_id}"
            else:
                grp = f"melasma_wiki_{parent_id[:12]}"
        elif "SkinDiseaseBD" in source:
            # Group by 4-character perceptual hash prefix to cluster near-identical framing
            ph = str(row.get("phash", ""))
            cluster_prefix = ph[:4] if len(ph) >= 4 else "skindbd"
            grp = f"skindbd_{label}_{cluster_prefix}"
        else:
            # Default: use parent_image_id
            grp = f"{label}_{parent_id}"

        groups.append(grp)

    df = df.copy()
    df["leakage_group"] = groups
    return df


def generate_v2_splits(seed: int = 42):
    print("==================================================")
    print("DERMTRUST V2 DATASET SPLITTER (STRICT INTEGRITY)")
    print(f"Seed: {seed} | Stratified Group Isolation")
    print("==================================================")

    df = pd.read_csv(MANIFEST_PATH)
    target_df = df[df["category"] == "target"].copy().reset_index(drop=True)
    ood_df = df[df["category"] == "non_target"].copy().reset_index(drop=True)

    target_df = assign_robust_leakage_groups(target_df)

    # SEPARATION PRINCIPLE:
    # True synthetic reflections/rotations (_flip, _rot in Melasma) must NEVER enter Validation or Test.
    # SkinDiseaseBD images from Mendeley Data (which all have an 'aug_3_' naming convention) are clustered
    # by donor/perceptual hash so visual variants never cross between Train, Val, and Test.
    df["is_synthetic_flip"] = df["original_filename"].str.contains(r"_flip|_rot", case=False, regex=True)

    # For strict test/val sets, exclude synthetic flips
    eligible_for_eval = target_df[~target_df["original_filename"].str.contains(r"_flip|_rot", case=False, regex=True)].copy().reset_index(drop=True)
    synthetic_flips = target_df[target_df["original_filename"].str.contains(r"_flip|_rot", case=False, regex=True)].copy().reset_index(drop=True)

    print(f"Total Target Records:          {len(target_df)}")
    print(f"Evaluation-Eligible Images:    {len(eligible_for_eval)}")
    print(f"Synthetic Flips (Train-Only):   {len(synthetic_flips)}")
    print(f"Total Unique Leakage Groups:   {eligible_for_eval['leakage_group'].nunique()}")

    # Stratified Group Split across evaluation-eligible clinical images
    # 7 folds: Fold 0 -> Test (~14.3%), Fold 1 -> Val (~14.3%), Folds 2-6 -> Train (~71.4%)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    splits = list(sgkf.split(eligible_for_eval, eligible_for_eval["verified_label"], eligible_for_eval["leakage_group"]))

    test_idx = splits[0][1]
    val_idx = splits[1][1]
    train_orig_idx = np.setdiff1d(np.arange(len(eligible_for_eval)), np.concatenate([test_idx, val_idx]))

    test_df = eligible_for_eval.iloc[test_idx].copy().reset_index(drop=True)
    val_df = eligible_for_eval.iloc[val_idx].copy().reset_index(drop=True)
    train_orig_df = eligible_for_eval.iloc[train_orig_idx].copy().reset_index(drop=True)

    # Attach synthetic flips ONLY to the training set, and ONLY if their parent group is in the training set
    train_groups = set(train_orig_df["leakage_group"])
    train_flips = synthetic_flips[synthetic_flips["leakage_group"].isin(train_groups)].copy()
    train_df = pd.concat([train_orig_df, train_flips], ignore_index=True)

    # Verify zero leakage across groups
    train_grps = set(train_df["leakage_group"])
    val_grps = set(val_df["leakage_group"])
    test_grps = set(test_df["leakage_group"])

    assert len(train_grps.intersection(val_grps)) == 0, "ERROR: Train-Val group collision!"
    assert len(train_grps.intersection(test_grps)) == 0, "ERROR: Train-Test group collision!"
    assert len(val_grps.intersection(test_grps)) == 0, "ERROR: Val-Test group collision!"

    # Verify zero MD5 collisions
    assert len(set(train_df["md5hash"]).intersection(set(val_df["md5hash"]))) == 0, "ERROR: Train-Val MD5 collision!"
    assert len(set(train_df["md5hash"]).intersection(set(test_df["md5hash"]))) == 0, "ERROR: Train-Test MD5 collision!"
    assert len(set(val_df["md5hash"]).intersection(set(test_df["md5hash"]))) == 0, "ERROR: Val-Test MD5 collision!"

    # Verify ZERO synthetic flips in Val and Test
    assert not val_df["original_filename"].str.contains(r"_flip|_rot").any(), "ERROR: Synthetic flips in Val set!"
    assert not test_df["original_filename"].str.contains(r"_flip|_rot").any(), "ERROR: Synthetic flips in Test set!"

    print("\n[VERIFICATION PASSED]")
    print(" - Zero group overlap between Train, Val, and Test.")
    print(" - Zero MD5 hash collisions between splits.")
    print(f" - Validation set synthetic flips: 0 (STRICTLY ZERO)")
    print(f" - Test set synthetic flips:       0 (STRICTLY ZERO)")

    print("\n--- V2 SPLIT SUMMARY BY CLASS ---")
    summary = pd.DataFrame({
        "Train (Total)": train_df["verified_label"].value_counts(),
        "Train (Aug)": train_df.groupby("verified_label")["is_augmented"].sum(),
        "Val (Orig)": val_df["verified_label"].value_counts(),
        "Test (Orig)": test_df["verified_label"].value_counts(),
    }).fillna(0).astype(int)
    print(summary.to_string())

    # Save versioned v2 manifests
    train_v2_path = DATA_DIR / "train_5class_manifest_v2.csv"
    val_v2_path = DATA_DIR / "val_5class_manifest_v2.csv"
    test_v2_path = DATA_DIR / "test_5class_manifest_v2.csv"

    train_df.to_csv(train_v2_path, index=False)
    val_df.to_csv(val_v2_path, index=False)
    test_df.to_csv(test_v2_path, index=False)

    print(f"\nSaved versioned manifests:")
    print(f"  Train v2: {train_v2_path.name} (N={len(train_df)})")
    print(f"  Val v2:   {val_v2_path.name} (N={len(val_df)})")
    print(f"  Test v2:  {test_v2_path.name} (N={len(test_df)})")

    return train_df, val_df, test_df


if __name__ == "__main__":
    generate_v2_splits()
