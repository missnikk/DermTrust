"""
ml/prepare_6class_manifest.py
Role: Prepares 6-Class Manifests (5 Target Diseases + Class 6: Not Recognised / Look-Alikes)
Derived strictly from clean baseline manifests with 100% collision filtering against all splits.
"""

from collections import defaultdict
import hashlib
import os
from pathlib import Path
import random
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"
EXTERNAL_DIR = ROOT_DIR / "ml" / "external_datasets"

TRAIN_BASE = DATA_DIR / "train_5class_manifest.csv"
VAL_BASE = DATA_DIR / "val_5class_manifest.csv"
TEST_LOCKED = DATA_DIR / "test_5class_manifest.csv"
OOD_BASE = DATA_DIR / "ood_5class_manifest.csv"
MEMI_DIR = EXTERNAL_DIR / "memi_ds"
FIGSHARE_DIR = EXTERNAL_DIR / "figshare_ringworm" / "extracted" / "SkinDiseaseDataset_2100"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def compute_hashes(path: Path) -> tuple[str, str]:
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()


def main():
    print("=" * 65)
    print("DERMTRUST 6-CLASS CLEAN DATASET MANIFEST GENERATOR")
    print("=" * 65)

    df_train = pd.read_csv(TRAIN_BASE)
    df_val = pd.read_csv(VAL_BASE)
    df_test_locked = pd.read_csv(TEST_LOCKED)
    df_ood = pd.read_csv(OOD_BASE)

    base_train_hashes = set(df_train["md5hash"])
    base_val_hashes = set(df_val["md5hash"])
    locked_test_hashes = set(df_test_locked["md5hash"])
    all_base_hashes = base_train_hashes.union(base_val_hashes).union(locked_test_hashes)

    print(f"Base Train:   {len(df_train)} rows")
    print(f"Base Val:     {len(df_val)} rows")
    print(f"Locked Test:  {len(df_test_locked)} rows")
    print(f"Base OOD:     {len(df_ood)} rows")

    # 1. Process MEMI-DS Melasma (131 images from 63 patients)
    pt_files = defaultdict(list)
    if MEMI_DIR.exists():
        for f in sorted(MEMI_DIR.glob("*.jpg")):
            stem = f.stem
            pid = stem.split("_")[0]
            md5, sha = compute_hashes(f)
            assert md5 not in all_base_hashes, f"Collision with base: {f.name}"
            rec = {
                "image_id": f"memi_ds_{stem}",
                "source_dataset": "MEMI-DS (Zhai et al. 2025, DOI: 10.6084/m9.figshare.29209229)",
                "original_filename": f.name,
                "local_path": str(f.resolve()),
                "verified_label": "melasma",
                "category": "Acquired Hyperpigmentation Disorder",
                "fitzpatrick_scale": "Unknown",
                "skin_tone_group": "Darker",
                "is_augmented": False,
                "parent_image_id": f"melasma_memi_patient_{pid}",
                "license": "CC BY 4.0",
                "attribution": "Zhai et al. (Figshare DOI: 10.6084/m9.figshare.29209229)",
                "md5hash": md5,
                "sha256": sha,
                "phash": "",
                "leakage_group": f"melasma_memi_patient_{pid}"
            }
            pt_files[pid].append(rec)

        pids = sorted(list(pt_files.keys()))
        random.seed(SEED)
        random.shuffle(pids)
        # 50 patients train, 13 patients val
        train_pids = set(pids[:50])
        val_pids = set(pids[50:])

        memi_train = [r for pid in train_pids for r in pt_files[pid]]
        memi_val = [r for pid in val_pids for r in pt_files[pid]]
        print(f"\nMEMI-DS Melasma: {len(memi_train)} Train ({len(train_pids)} pts), {len(memi_val)} Val ({len(val_pids)} pts)")
    else:
        memi_train, memi_val = [], []

    # 2. Process Figshare Natural Ringworm images (strictly excluding any hash already in base datasets)
    figshare_train_records = []
    figshare_val_records = []
    seen_figshare = set()

    if FIGSHARE_DIR.exists():
        rw_train_dir = FIGSHARE_DIR / "train" / "ringworm"
        rw_test_dir = FIGSHARE_DIR / "test" / "ringworm"

        # Figshare Train images -> add to Train only
        for f in sorted(rw_train_dir.glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                md5, sha = compute_hashes(f)
                if md5 in all_base_hashes or md5 in seen_figshare:
                    continue
                seen_figshare.add(md5)
                figshare_train_records.append({
                    "image_id": f"figshare_rw_{f.stem}",
                    "source_dataset": "Figshare Ringworm Dataset (DOI: 10.6084/m9.figshare.33084632)",
                    "original_filename": f.name,
                    "local_path": str(f.resolve()),
                    "verified_label": "ringworm",
                    "category": "Fungal Infection",
                    "fitzpatrick_scale": "Unknown",
                    "skin_tone_group": "Darker",
                    "is_augmented": False,
                    "parent_image_id": f"figshare_rw_{f.stem[:12]}",
                    "license": "CC BY 4.0",
                    "attribution": "Apurbo Biswas (Figshare DOI: 10.6084/m9.figshare.33084632)",
                    "md5hash": md5,
                    "sha256": sha,
                    "phash": "",
                    "leakage_group": f"figshare_rw_{f.stem[:12]}"
                })

        # Figshare Test images -> add to Val only (so no cross-split leakage with Train)
        for f in sorted(rw_test_dir.glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                md5, sha = compute_hashes(f)
                if md5 in all_base_hashes or md5 in seen_figshare:
                    continue
                seen_figshare.add(md5)
                figshare_val_records.append({
                    "image_id": f"figshare_rw_{f.stem}",
                    "source_dataset": "Figshare Ringworm Dataset (DOI: 10.6084/m9.figshare.33084632)",
                    "original_filename": f.name,
                    "local_path": str(f.resolve()),
                    "verified_label": "ringworm",
                    "category": "Fungal Infection",
                    "fitzpatrick_scale": "Unknown",
                    "skin_tone_group": "Darker",
                    "is_augmented": False,
                    "parent_image_id": f"figshare_rw_{f.stem[:12]}",
                    "license": "CC BY 4.0",
                    "attribution": "Apurbo Biswas (Figshare DOI: 10.6084/m9.figshare.33084632)",
                    "md5hash": md5,
                    "sha256": sha,
                    "phash": "",
                    "leakage_group": f"figshare_rw_{f.stem[:12]}"
                })
        print(f"Figshare Ringworm added: {len(figshare_train_records)} Train, {len(figshare_val_records)} Val")

    # 3. Class 6: "not_recognised" (Look-alikes: Eczema, Lichen Planus, Scabies, Rosacea)
    df_ood_shuffled = df_ood.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    parents = list(df_ood_shuffled["parent_image_id"].unique())
    random.seed(SEED)
    random.shuffle(parents)

    train_p = set(parents[:500])
    val_p = set(parents[500:650])
    eval_p = set(parents[650:])

    ood_train = df_ood_shuffled[df_ood_shuffled["parent_image_id"].isin(train_p)].copy()
    ood_val = df_ood_shuffled[df_ood_shuffled["parent_image_id"].isin(val_p)].copy()
    ood_unseen = df_ood_shuffled[df_ood_shuffled["parent_image_id"].isin(eval_p)].copy()

    for d in [ood_train, ood_val, ood_unseen]:
        d["verified_label"] = "not_recognised"
        if "leakage_group" not in d.columns:
            d["leakage_group"] = d["parent_image_id"]

    print(f"\nClass 6 (not_recognised look-alikes):")
    print(f"  Train:     {len(ood_train)} images")
    print(f"  Val:       {len(ood_val)} images")
    print(f"  OOD Eval:  {len(ood_unseen)} images (unseen test holdout)")

    # 4. Assemble final manifests
    train_components = [df_train]
    if memi_train: train_components.append(pd.DataFrame(memi_train))
    if figshare_train_records: train_components.append(pd.DataFrame(figshare_train_records))
    train_components.append(ood_train)
    df_train_final = pd.concat(train_components, ignore_index=True)

    val_components = [df_val]
    if memi_val: val_components.append(pd.DataFrame(memi_val))
    if figshare_val_records: val_components.append(pd.DataFrame(figshare_val_records))
    val_components.append(ood_val)
    df_val_final = pd.concat(val_components, ignore_index=True)

    # Clean local paths
    df_train_final["local_path"] = df_train_final["local_path"].apply(lambda p: str(Path(p).resolve()))
    df_val_final["local_path"] = df_val_final["local_path"].apply(lambda p: str(Path(p).resolve()))

    # Verify zero collisions across all splits
    tr_hashes = set(df_train_final["md5hash"])
    vl_hashes = set(df_val_final["md5hash"])
    te_hashes = set(df_test_locked["md5hash"])
    ood_hashes = set(ood_unseen["md5hash"])

    coll_tr_vl = tr_hashes.intersection(vl_hashes)
    coll_tr_te = tr_hashes.intersection(te_hashes)
    coll_vl_te = vl_hashes.intersection(te_hashes)
    coll_tr_ood = tr_hashes.intersection(ood_hashes)
    coll_vl_ood = vl_hashes.intersection(ood_hashes)
    coll_te_ood = te_hashes.intersection(ood_hashes)

    assert len(coll_tr_vl) == 0, f"Train vs Val collision: {len(coll_tr_vl)}"
    assert len(coll_tr_te) == 0, f"Train vs Locked Test collision: {len(coll_tr_te)}"
    assert len(coll_vl_te) == 0, f"Val vs Locked Test collision: {len(coll_vl_te)}"
    assert len(coll_tr_ood) == 0, f"Train vs OOD Unseen collision: {len(coll_tr_ood)}"
    assert len(coll_vl_ood) == 0, f"Val vs OOD Unseen collision: {len(coll_vl_ood)}"
    assert len(coll_te_ood) == 0, f"Test vs OOD Unseen collision: {len(coll_te_ood)}"

    train_out = DATA_DIR / "train_6class_manifest.csv"
    val_out = DATA_DIR / "val_6class_manifest.csv"
    ood_eval_out = DATA_DIR / "ood_unseen_evaluation_manifest.csv"

    df_train_final.to_csv(train_out, index=False)
    df_val_final.to_csv(val_out, index=False)
    ood_unseen.to_csv(ood_eval_out, index=False)

    print("\n" + "=" * 65)
    print(f"SUCCESS: 6-Class Manifests Built with ZERO Cross-Split Collisions!")
    print(f"  Train:     {train_out} ({len(df_train_final)} rows)")
    print(df_train_final["verified_label"].value_counts())
    print(f"\n  Val:       {val_out} ({len(df_val_final)} rows)")
    print(df_val_final["verified_label"].value_counts())
    print(f"\n  Locked Test Set: {TEST_LOCKED} ({len(df_test_locked)} rows)")
    print(f"  Unseen OOD Holdout: {ood_eval_out} ({len(ood_unseen)} rows)")
    print("=" * 65)

if __name__ == "__main__":
    main()
