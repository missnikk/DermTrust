"""
DermTrust — V3.2 Manifest Preparation Script
Role: Incorporates verified external clinical dataset MEMI-DS (DOI: 10.6084/m9.figshare.29209229)
      into V3.1 manifests to resolve the extreme Melasma data deficit and facial shortcut.
Rules:
1. Strict patient-level grouping: all multi-angle views of a patient stay in either Train or Val.
2. Zero pre-augmented images in Validation (Val remains 100% natural clinical photography).
3. Zero cross-split patient/case collisions.
4. Zero collision with quarantined regression images.
5. Preserves all existing V3.1 historical manifests and locked test sets unchanged.
"""

from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"
EXTERNAL_DIR = ROOT_DIR / "ml" / "external_datasets"
MEMI_DIR = EXTERNAL_DIR / "memi_ds"
REGRESSION_DIR = DATA_DIR / "regression_input_images"

TRAIN_V3_1 = DATA_DIR / "train_5class_manifest_v3_1.csv"
VAL_V3_1 = DATA_DIR / "val_5class_manifest_v3_1.csv"
TEST_LOCKED = DATA_DIR / "test_5class_manifest.csv"

TRAIN_V3_2 = DATA_DIR / "train_5class_manifest_v3_2.csv"
VAL_V3_2 = DATA_DIR / "val_5class_manifest_v3_2.csv"


def compute_hashes(path: Path) -> tuple[str, str]:
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()


def prepare_v3_2():
    print("=" * 65)
    print("DERMTRUST V3.2 DATASET MANIFEST GENERATOR")
    print("=" * 65)

    assert TRAIN_V3_1.exists(), f"Missing {TRAIN_V3_1}"
    assert VAL_V3_1.exists(), f"Missing {VAL_V3_1}"
    assert MEMI_DIR.exists(), f"Missing MEMI-DS directory at {MEMI_DIR}"

    df_train_base = pd.read_csv(TRAIN_V3_1)
    df_val_base = pd.read_csv(VAL_V3_1)
    df_test_locked = pd.read_csv(TEST_LOCKED) if TEST_LOCKED.exists() else None

    print(f"Loaded base V3.1 Train: {len(df_train_base)} rows")
    print(f"Loaded base V3.1 Val:   {len(df_val_base)} rows")

    # Step 1: Collect and hash all MEMI-DS images
    memi_files = sorted(MEMI_DIR.glob("*.jpg"))
    print(f"\nProcessing MEMI-DS clinical images from {MEMI_DIR}...")
    print(f"Total MEMI-DS files found: {len(memi_files)}")
    assert len(memi_files) == 131, f"Expected 131 MEMI-DS files, found {len(memi_files)}"

    pt_files = defaultdict(list)
    file_records = {}

    for f in memi_files:
        stem = f.stem
        parts = stem.split("_")
        pid = parts[0]
        view = parts[1] if len(parts) > 1 else "unknown"

        md5, sha = compute_hashes(f)
        record = {
            "image_id": f"memi_ds_{stem}",
            "dataset_source": "MEMI-DS (Zhai et al. 2025, DOI: 10.6084/m9.figshare.29209229)",
            "source_image_id": f.name,
            "local_path": str(f.resolve()),
            "verified_label": "melasma",
            "target_class": "melasma",
            "license": "CC BY 4.0",
            "attribution": "Zhenwei Zhai, Chen Li, Zihao Wang, Tao Jiang (Figshare DOI: 10.6084/m9.figshare.29209229)",
            "md5hash": md5,
            "sha256": sha,
            "is_pre_augmented": False,
            "leakage_group": f"melasma_memi_patient_{pid}",
            "skin_tone_group": "Intermediate (FST III-IV, Asian cohort)",
        }
        pt_files[pid].append(record)
        file_records[f.name] = record

    print(f"Total unique MEMI-DS patients: {len(pt_files)}")
    assert len(pt_files) == 63, f"Expected 63 patients, found {len(pt_files)}"

    # Step 2: Stratified patient split (reproducible seed=42)
    by_views = defaultdict(list)
    for pid, records in pt_files.items():
        by_views[len(records)].append(pid)

    rng = random.Random(42)
    train_pts = []
    val_pts = []

    for n_views, pids in sorted(by_views.items()):
        pids_sorted = sorted(pids, key=lambda x: int(x))
        rng.shuffle(pids_sorted)
        # 20% to val
        n_val = max(1, round(len(pids_sorted) * 0.20))
        val_pts.extend(pids_sorted[:n_val])
        train_pts.extend(pids_sorted[n_val:])

    assert len(set(train_pts).intersection(set(val_pts))) == 0, "Patient leakage detected!"
    print(f"Patient partition: {len(train_pts)} Train patients, {len(val_pts)} Val patients")

    memi_train_records = [rec for pid in train_pts for rec in pt_files[pid]]
    memi_val_records = [rec for pid in val_pts for rec in pt_files[pid]]

    print(f"MEMI-DS Image partition: {len(memi_train_records)} Train images, {len(memi_val_records)} Val images")
    assert len(memi_train_records) == 104, f"Expected 104 train images, got {len(memi_train_records)}"
    assert len(memi_val_records) == 27, f"Expected 27 val images, got {len(memi_val_records)}"

    # Step 3: Combine with base V3.1 and populate physical SHA-256
    df_memi_train = pd.DataFrame(memi_train_records)
    df_memi_val = pd.DataFrame(memi_val_records)

    df_train_v3_2 = pd.concat([df_train_base, df_memi_train], ignore_index=True)
    df_val_v3_2 = pd.concat([df_val_base, df_memi_val], ignore_index=True)

    print("\nComputing / verifying physical SHA-256 hashes for all records...")
    def get_real_sha(row):
        existing = str(row.get("sha256", ""))
        if existing and existing != "pre-indexed" and len(existing) == 64:
            return existing
        with open(row["local_path"], "rb") as f:
            return hashlib.sha256(f.read()).hexdigest().lower()

    df_train_v3_2["sha256"] = [get_real_sha(r) for _, r in df_train_v3_2.iterrows()]
    df_val_v3_2["sha256"] = [get_real_sha(r) for _, r in df_val_v3_2.iterrows()]

    print("\n--- Summary of V3.2 Datasets ---")
    print(f"Train V3.2 Total Rows: {len(df_train_v3_2)} (Base: {len(df_train_base)} + MEMI-DS: {len(df_memi_train)})")
    print(f"Val V3.2 Total Rows:   {len(df_val_v3_2)} (Base: {len(df_val_base)} + MEMI-DS: {len(df_memi_val)})")

    # Step 4: Strict Collision & Leakage Verification
    print("\nExecuting comprehensive data integrity checks...")

    # 4a: Check file existence
    for split_name, df_s in [("Train V3.2", df_train_v3_2), ("Val V3.2", df_val_v3_2)]:
        missing = [p for p in df_s["local_path"] if not Path(p).exists()]
        assert len(missing) == 0, f"Missing files in {split_name}: {missing[:5]}"
        print(f"  [OK] All {len(df_s)} files in {split_name} exist on disk.")

    # 4b: Check pre-augmented in val
    val_preaug = df_val_v3_2["is_pre_augmented"].sum()
    assert val_preaug == 0, f"Pre-augmented files found in validation: {val_preaug}"
    print("  [OK] 0 pre-augmented images in Validation (100% natural clinical photos).")

    # 4c: Check physical SHA-256 collision between Train and Val
    tr_sha = set(df_train_v3_2["sha256"])
    va_sha = set(df_val_v3_2["sha256"])
    sha_col = tr_sha.intersection(va_sha)
    assert len(sha_col) == 0, f"Cross-split SHA-256 collision: {sha_col}"
    print(f"  [OK] 0 physical SHA-256 collisions between Train and Val.")

    # 4c-2: Check MD5 collision between Train and Val
    tr_md5 = set(df_train_v3_2["md5hash"])
    va_md5 = set(df_val_v3_2["md5hash"])
    md5_col = tr_md5.intersection(va_md5)
    assert len(md5_col) == 0, f"Cross-split MD5 collision: {md5_col}"
    print(f"  [OK] 0 MD5 collisions between Train and Val.")

    # 4d: Check patient/case leakage_group collision between Train and Val
    tr_grp = set(df_train_v3_2["leakage_group"].dropna())
    va_grp = set(df_val_v3_2["leakage_group"].dropna())
    grp_col = tr_grp.intersection(va_grp)
    assert len(grp_col) == 0, f"Cross-split patient/group collision: {grp_col}"
    print(f"  [OK] 0 patient/case leakage group collisions between Train and Val.")

    # 4e: Check quarantined regression images
    if REGRESSION_DIR.exists():
        reg_files = list(REGRESSION_DIR.glob("*.*"))
        reg_hashes = set()
        for rf in reg_files:
            with open(rf, "rb") as f:
                reg_hashes.add(hashlib.sha256(f.read()).hexdigest().lower())
        
        tr_reg_match = tr_sha.intersection(reg_hashes)
        va_reg_match = va_sha.intersection(reg_hashes)
        assert len(tr_reg_match) == 0, f"Regression images leaked into Train: {tr_reg_match}"
        assert len(va_reg_match) == 0, f"Regression images leaked into Val: {va_reg_match}"
        print(f"  [OK] 0 regression images in Train or Val (quarantined regression set intact).")

    # 4f: Check locked test set leakage
    if df_test_locked is not None:
        te_sha = set(df_test_locked["sha256"])
        te_tr_sha = te_sha.intersection(tr_sha)
        te_va_sha = te_sha.intersection(va_sha)
        print(f"  [Info] Locked test SHA collisions with Train: {len(te_tr_sha)}, Val: {len(te_va_sha)}")

    # Step 5: Save V3.2 manifests
    df_train_v3_2.to_csv(TRAIN_V3_2, index=False)
    df_val_v3_2.to_csv(VAL_V3_2, index=False)
    print(f"\nSaved proposed V3.2 manifests:")
    print(f"  - {TRAIN_V3_2}")
    print(f"  - {VAL_V3_2}")

    print("\nTrain V3.2 Class Counts:")
    for c, cnt in df_train_v3_2["verified_label"].value_counts().items():
        print(f"  {c:<12}: {cnt}")

    print("\nVal V3.2 Class Counts:")
    for c, cnt in df_val_v3_2["verified_label"].value_counts().items():
        print(f"  {c:<12}: {cnt}")


if __name__ == "__main__":
    prepare_v3_2()
