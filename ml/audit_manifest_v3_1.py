"""
DermTrust — Comprehensive V3.1 Manifest Audit & Collision Verification
Generates machine-readable audit JSON: ml/data/exp8_collision_audit.json
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"

TRAIN_V3_1 = DATA_DIR / "train_5class_manifest_v3_1.csv"
VAL_V3_1 = DATA_DIR / "val_5class_manifest_v3_1.csv"
TEST_LOCKED = DATA_DIR / "test_5class_manifest.csv"
AUDIT_JSON = DATA_DIR / "exp8_collision_audit.json"

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
    print("=" * 60)
    print("DERMTRUST V3.1 COMPREHENSIVE DATA & COLLISION AUDIT")
    print("=" * 60)

    # 1. Load Manifests
    df_train = pd.read_csv(TRAIN_V3_1)
    df_val = pd.read_csv(VAL_V3_1)
    df_test = pd.read_csv(TEST_LOCKED) if TEST_LOCKED.exists() else None

    # Check manifest hashes
    train_hash = sha256_file(TRAIN_V3_1)
    val_hash = sha256_file(VAL_V3_1)
    test_hash = sha256_file(TEST_LOCKED) if df_test is not None else "N/A"

    print(f"\nManifest SHA-256 Hashes:")
    print(f"  Train V3.1 ({len(df_train)} rows): {train_hash}")
    print(f"  Val V3.1   ({len(df_val)} rows):   {val_hash}")
    print(f"  Locked Test ({len(df_test) if df_test is not None else 0} rows): {test_hash}")

    # 2. Per-class row counts and pre-augmented counts
    train_class_counts = df_train["verified_label"].value_counts().to_dict()
    val_class_counts = df_val["verified_label"].value_counts().to_dict()
    
    train_preaug = int(df_train["is_pre_augmented"].sum())
    train_natural = len(df_train) - train_preaug
    val_preaug = int(df_val.get("is_pre_augmented", pd.Series([False]*len(df_val))).sum())
    val_natural = len(df_val) - val_preaug

    print(f"\nPartition Composition:")
    print(f"  Train V3.1: Total={len(df_train)}, Natural={train_natural}, Pre-Augmented={train_preaug}")
    print(f"  Val V3.1:   Total={len(df_val)}, Natural={val_natural}, Pre-Augmented={val_preaug} (Must be 0)")

    # 3. Missing files on disk
    missing_train = [p for p in df_train["local_path"] if not Path(p).exists()]
    missing_val = [p for p in df_val["local_path"] if not Path(p).exists()]
    print(f"  Missing files on disk (Train): {len(missing_train)}")
    print(f"  Missing files on disk (Val):   {len(missing_val)}")

    # 4. Content Unique Hashes on disk
    print("\nComputing physical file hashes...")
    train_shas = [sha256_file(Path(p)) for p in df_train["local_path"]]
    val_shas = [sha256_file(Path(p)) for p in df_val["local_path"]]

    unique_train_sha = len(set(train_shas))
    unique_val_sha = len(set(val_shas))
    cross_sha_overlap = set(train_shas).intersection(set(val_shas))
    print(f"  Unique physical content hashes in Train: {unique_train_sha} / {len(df_train)}")
    print(f"  Unique physical content hashes in Val:   {unique_val_sha} / {len(df_val)}")
    print(f"  Cross-Split Content Hash Collisions:     {len(cross_sha_overlap)}")

    # 5. Patient / Case Group Overlap Check
    tr_groups = set(df_train["leakage_group"].dropna())
    va_groups = set(df_val["leakage_group"].dropna())
    group_overlap = tr_groups.intersection(va_groups)
    print(f"  Cross-Split Patient/Case Group Collisions: {len(group_overlap)}")
    if group_overlap:
        print(f"  COLLISIONS DETECTED: {group_overlap}")

    # Specific SCIN case check
    scin_train_cases = set(df_train[df_train["dataset_source"].str.contains("SCIN", na=False)]["leakage_group"])
    scin_val_cases = set(df_val[df_val["dataset_source"].str.contains("SCIN", na=False)]["leakage_group"])
    scin_overlap = scin_train_cases.intersection(scin_val_cases)
    print(f"  SCIN Case Collisions: {len(scin_overlap)}")

    # Specific Figshare stem check
    def clean_stem(p):
        fn = Path(p).stem.lower()
        fn = re.sub(r"^aug_\d+_", "", fn)
        fn = re.sub(r"_(jpg|jpeg|png)\.rf\.[a-f0-9]+", "", fn)
        fn = re.sub(r"_(flip|rot\d*|crop\d*)", "", fn)
        return fn

    tr_stems = set(df_train["local_path"].apply(clean_stem))
    va_stems = set(df_val["local_path"].apply(clean_stem))
    stem_overlap = tr_stems.intersection(va_stems)
    print(f"  Exact Clean Stem Collisions: {len(stem_overlap)}")

    # 6. Test Set Overlap Disclosure
    test_overlap_train = len(set(df_train["local_path"]).intersection(set(df_test["local_path"]))) if df_test is not None else 0
    test_overlap_val = len(set(df_val["local_path"]).intersection(set(df_test["local_path"]))) if df_test is not None else 0

    # 7. Protected Production Artifacts Integrity
    print("\nVerifying Protected Production Artifacts...")
    all_protected_pass = True
    for name, (path, expected_hash) in PROTECTED_FILES.items():
        actual = sha256_file(path)
        match = (actual == expected_hash)
        print(f"  {name:<28}: {'INTACT' if match else 'CORRUPTED'}")
        if not match:
            all_protected_pass = False

    audit_pass = (
        len(missing_train) == 0 and
        len(missing_val) == 0 and
        val_preaug == 0 and
        len(cross_sha_overlap) == 0 and
        len(group_overlap) == 0 and
        len(scin_overlap) == 0 and
        len(stem_overlap) == 0 and
        all_protected_pass
    )

    print("\n" + "=" * 60)
    print(f"AUDIT VERDICT: {'ALL INTEGRITY CRITERIA PASSED' if audit_pass else 'AUDIT FAILED'}")
    print("=" * 60)

    # Machine-readable output
    audit_data = {
        "audit_version": "3.1.0",
        "audit_status": "PASSED" if audit_pass else "FAILED",
        "remediation_summary": {
            "scin_case_3986995536806137754_leakage": "RESOLVED (Image -5195598708424329883.png removed from Train V3.1)",
            "figshare_afp20080515p1415_derivative_leakage": "RESOLVED (Image aug_0_afp20080515p1415-f2.jpg removed from Train V3.1)",
            "leakage_group_hash_suffix": "RESOLVED (Stripped arbitrary md5 suffixes, restored true case grouping)",
        },
        "manifest_hashes": {
            "train_5class_manifest_v3_1.csv": train_hash,
            "val_5class_manifest_v3_1.csv": val_hash,
            "test_5class_manifest.csv": test_hash,
        },
        "row_counts": {
            "train_v3_1": {
                "total": len(df_train),
                "natural": train_natural,
                "pre_augmented": train_preaug,
                "per_class": train_class_counts,
            },
            "val_v3_1": {
                "total": len(df_val),
                "natural": val_natural,
                "pre_augmented": val_preaug,
                "per_class": val_class_counts,
            },
        },
        "collision_audit": {
            "cross_split_content_sha_collisions": len(cross_sha_overlap),
            "cross_split_patient_group_collisions": len(group_overlap),
            "scin_patient_overlap": len(scin_overlap),
            "figshare_derivative_overlap": len(stem_overlap),
            "test_overlap_with_train_v3_1": test_overlap_train,
            "test_overlap_with_val_v3_1": test_overlap_val,
        },
        "protected_artifacts_intact": all_protected_pass,
        "provenance_limitations": {
            "fitzpatrick17k": "Lacks patient identifiers in upstream metadata. Grouping is discrete image-level hash.",
            "skindiseasebd": "Authors supplied only pre-augmented variations without source patient mapping. Confined 100% to training.",
            "figshare": "Natural images lack explicit clinical case metadata. Grouped by file stem.",
        }
    }

    with open(AUDIT_JSON, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Machine-readable audit saved to: {AUDIT_JSON}")


if __name__ == "__main__":
    audit()
