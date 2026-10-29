"""
DermTrust — Comprehensive V3.2 Manifest Audit & Integrity Verification
Generates machine-readable audit JSON: ml/data/v3_2_manifest_audit.json
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"

TRAIN_V3_2 = DATA_DIR / "train_5class_manifest_v3_2.csv"
VAL_V3_2 = DATA_DIR / "val_5class_manifest_v3_2.csv"
TEST_LOCKED = DATA_DIR / "test_5class_manifest.csv"
REGRESSION_DIR = DATA_DIR / "regression_input_images"
AUDIT_JSON = DATA_DIR / "v3_2_manifest_audit.json"

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]

PROTECTED_FILES = {
    "Baseline Keras Model": (ROOT_DIR / "ml" / "models" / "dermtrust_mobilenetv2_5class.keras", "c3f03239647e3a6667b1a5ad614a84085ba00ab03bfbcd1c121dd9c59d9ad4ff"),
    "Baseline Classifier TFLite": (ROOT_DIR / "flutter_app" / "assets" / "models" / "dermtrust_classifier.tflite", "76d6fb199843879cf3e87060d596ea84d1df9d0cbb251daa3b13492ecceb6a58"),
    "Baseline CAM TFLite": (ROOT_DIR / "flutter_app" / "assets" / "models" / "dermtrust_cam.tflite", "1c28705ac3122a17b3ca58bd10fd91e7c9b162ea67a0c33b45c6f3b002fb5d3d"),
    "Baseline CAM Weights JSON": (ROOT_DIR / "flutter_app" / "assets" / "models" / "cam_weights.json", "338f5365980edbe7a60b3582f02bd3a8f8155dd77c50e5b9c53d578a6b5cd104"),
    "Release APK": (ROOT_DIR / "flutter_app" / "build" / "app" / "outputs" / "flutter-apk" / "app-arm64-v8a-release.apk", "9a258d3e1ba9660f504ef936fd5095e56f83aee12b6b1368efd517b9c84647f4"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def audit():
    print("=" * 65)
    print("DERMTRUST V3.2 COMPREHENSIVE DATA INTEGRITY AUDIT")
    print("=" * 65)

    df_train = pd.read_csv(TRAIN_V3_2)
    df_val = pd.read_csv(VAL_V3_2)
    df_test = pd.read_csv(TEST_LOCKED) if TEST_LOCKED.exists() else None

    # 1. Manifest file hashes
    train_hash = sha256_file(TRAIN_V3_2)
    val_hash = sha256_file(VAL_V3_2)
    test_hash = sha256_file(TEST_LOCKED) if df_test is not None else "N/A"

    print(f"\nManifest SHA-256 Hashes:")
    print(f"  Train V3.2 ({len(df_train)} rows): {train_hash}")
    print(f"  Val V3.2   ({len(df_val)} rows):   {val_hash}")
    print(f"  Test Locked ({len(df_test) if df_test is not None else 0} rows): {test_hash}")

    # 2. Check protected production files
    print(f"\nProtected Production File Verification:")
    protected_status = {}
    for name, (path, expected_hash) in PROTECTED_FILES.items():
        if not path.exists():
            print(f"  [MISSING] {name}: {path}")
            protected_status[name] = {"exists": False, "intact": False}
        else:
            actual = sha256_file(path)
            intact = (actual == expected_hash)
            print(f"  [{'INTACT' if intact else 'MODIFIED'}] {name} ({actual[:16]}...)")
            protected_status[name] = {"exists": True, "intact": intact, "hash": actual}

    # 3. Class by split breakdown
    train_counts = df_train["verified_label"].value_counts().to_dict()
    val_counts = df_val["verified_label"].value_counts().to_dict()

    train_preaug_by_class = df_train[df_train["is_pre_augmented"]]["verified_label"].value_counts().to_dict()
    val_preaug_by_class = df_val[df_val["is_pre_augmented"]]["verified_label"].value_counts().to_dict()

    # 4. Source by class breakdown
    train_sources = df_train.groupby(["verified_label", "dataset_source"]).size().to_dict()
    val_sources = df_val.groupby(["verified_label", "dataset_source"]).size().to_dict()

    # Format sources nicely
    train_src_formatted = {f"{k[0]} | {k[1]}": v for k, v in train_sources.items()}
    val_src_formatted = {f"{k[0]} | {k[1]}": v for k, v in val_sources.items()}

    # 5. Leakage checks
    tr_groups = set(df_train["leakage_group"].dropna())
    va_groups = set(df_val["leakage_group"].dropna())
    group_collision = sorted(list(tr_groups.intersection(va_groups)))

    tr_md5 = set(df_train["md5hash"])
    va_md5 = set(df_val["md5hash"])
    md5_collision = sorted(list(tr_md5.intersection(va_md5)))

    tr_sha = set(df_train["sha256"])
    va_sha = set(df_val["sha256"])
    sha_collision = sorted(list(tr_sha.intersection(va_sha)))

    # 6. Regression image check
    reg_hashes = {}
    if REGRESSION_DIR.exists():
        for rf in REGRESSION_DIR.glob("*.*"):
            reg_hashes[rf.name] = sha256_file(rf)
    
    reg_hash_set = set(reg_hashes.values())
    tr_reg_leak = sorted(list(tr_sha.intersection(reg_hash_set)))
    va_reg_leak = sorted(list(va_sha.intersection(reg_hash_set)))

    # 7. Skin tone distribution
    train_skin = df_train.groupby(["verified_label", "skin_tone_group"]).size().to_dict()
    val_skin = df_val.groupby(["verified_label", "skin_tone_group"]).size().to_dict()
    train_skin_fmt = {f"{k[0]} | {k[1]}": v for k, v in train_skin.items()}
    val_skin_fmt = {f"{k[0]} | {k[1]}": v for k, v in val_skin.items()}

    audit_data = {
        "manifest_hashes": {
            "train_v3_2": train_hash,
            "val_v3_2": val_hash,
            "test_locked": test_hash,
        },
        "protected_production_files": protected_status,
        "partition_totals": {
            "train_total": len(df_train),
            "train_natural": len(df_train) - int(df_train["is_pre_augmented"].sum()),
            "train_pre_augmented": int(df_train["is_pre_augmented"].sum()),
            "val_total": len(df_val),
            "val_natural": len(df_val) - int(df_val["is_pre_augmented"].sum()),
            "val_pre_augmented": int(df_val["is_pre_augmented"].sum()),
        },
        "class_counts": {
            "train": train_counts,
            "val": val_counts,
            "train_pre_augmented_by_class": train_preaug_by_class,
            "val_pre_augmented_by_class": val_preaug_by_class,
        },
        "sources": {
            "train": train_src_formatted,
            "val": val_src_formatted,
        },
        "skin_tone_distributions": {
            "train": train_skin_fmt,
            "val": val_skin_fmt,
        },
        "integrity_checks": {
            "cross_split_group_collisions": len(group_collision),
            "cross_split_group_collision_details": group_collision,
            "cross_split_md5_collisions": len(md5_collision),
            "cross_split_sha256_collisions": len(sha_collision),
            "regression_images_in_train": len(tr_reg_leak),
            "regression_images_in_val": len(va_reg_leak),
            "all_train_files_exist": all(Path(p).exists() for p in df_train["local_path"]),
            "all_val_files_exist": all(Path(p).exists() for p in df_val["local_path"]),
            "val_pre_augmented_is_zero": int(df_val["is_pre_augmented"].sum()) == 0,
        }
    }

    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"\nGenerated Audit JSON: {AUDIT_JSON}")
    print(f"Summary:")
    print(f"  Train Rows: {len(df_train)} | Val Rows: {len(df_val)}")
    print(f"  Cross-Split Collisions (Group/MD5/SHA): {len(group_collision)} / {len(md5_collision)} / {len(sha_collision)}")
    print(f"  Regression Leaks: Train={len(tr_reg_leak)}, Val={len(va_reg_leak)}")
    print(f"  Val Pre-Augmented Count: {int(df_val['is_pre_augmented'].sum())} (Must be 0)")


if __name__ == "__main__":
    audit()
