"""
DermTrust — V3.1 Split Remediation Script
Role: Repairs the patient/case grouping logic at source to resolve:
1. Google SCIN Case ID -3986995536806137754 cross-split patient overlap
   (removes image -5195598708424329883.png from training so validation case remains 100% patient-isolated).
2. Figshare afp20080515p1415-f2.jpg pre-augmented derivative overlap
   (removes aug_0_afp20080515p1415-f2.jpg from training so validation image has zero train derivatives).
3. Strips erroneous _{md5hash[:8]} suffixes from leakage_group, restoring true patient/case group integrity.
4. Preserves historical train_5class_manifest_v3.csv, val_5class_manifest_v3.csv, and test manifests.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import re
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"

TRAIN_V3 = DATA_DIR / "train_5class_manifest_v3.csv"
VAL_V3 = DATA_DIR / "val_5class_manifest_v3.csv"
EXP_MANIFEST = DATA_DIR / "candidate_expansion_manifest.csv"

TRAIN_V3_1 = DATA_DIR / "train_5class_manifest_v3_1.csv"
VAL_V3_1 = DATA_DIR / "val_5class_manifest_v3_1.csv"


def extract_authoritative_case_group(row: pd.Series, exp_lookup: dict) -> str:
    """
    Computes a true clinical case/patient grouping string without appending individual image hashes.
    """
    lbl = str(row["verified_label"]).lower()
    local_path_str = str(Path(row["local_path"]).resolve()).lower()
    fn = Path(row["local_path"]).name.lower()
    source = str(row.get("dataset_source", "")).lower()

    # 1. Authoritative candidate_expansion_manifest patient_group_id if available
    if local_path_str in exp_lookup and pd.notna(exp_lookup[local_path_str]):
        pg = exp_lookup[local_path_str]
        if pg != "figshare_unspecified":
            return f"{lbl}_{pg}"

    # 2. Google SCIN images
    if "scin" in source or "-5195598708" in fn or "-5764637704" in fn:
        # Check known SCIN case IDs
        if "3904209653088168276" in local_path_str or fn in ["8752048456455067248.png", "-7948351738605132490.png"]:
            return f"{lbl}_scin_case_3904209653088168276"
        elif "8058850286709929094" in local_path_str or fn in ["-8049100538036745495.png", "-57638369826405659.png"]:
            return f"{lbl}_scin_case_8058850286709929094"
        elif "-71846644280357894" in local_path_str or fn in ["3617083469771402105.png", "-7237119304068840452.png"]:
            return f"{lbl}_scin_case_-71846644280357894"
        elif "-7962583725800017306" in local_path_str or fn in ["8598865896046559060.png"]:
            return f"{lbl}_scin_case_-7962583725800017306"
        elif "-3986995536806137754" in local_path_str or fn in ["-5195598708424329883.png", "-5764637704090046026.png"]:
            return f"{lbl}_scin_case_-3986995536806137754"

    # 3. Figshare images (group by root filename stem before roboflow hash or aug prefix)
    if "figshare" in source or "figshare" in local_path_str:
        clean = Path(row["local_path"]).name
        clean = re.sub(r"_(jpg|jpeg|png)\.rf\.[a-f0-9]+", "", clean, flags=re.I)
        clean = re.sub(r"^aug_\d+_", "", clean, flags=re.I)
        return f"{lbl}_figshare_{clean}"

    # 4. Atlas Dermatologico (Freire da Silva)
    if "atlas" in source or "atlas" in fn:
        clean = fn.replace(".jpg", "").replace(".png", "")
        return f"{lbl}_atlas_{clean}"

    # 5. Wikimedia Commons
    if "wikimedia" in source or "wiki" in fn or "chloasma" in fn:
        clean = fn.replace(".jpg", "").replace(".png", "")
        return f"{lbl}_wiki_{clean}"

    # 6. SkinDiseaseBD (all files are pre-augmented, no patient metadata available)
    if "skindiseasebd" in source or "skindiseasebd" in local_path_str:
        # Documented limitation: SkinDiseaseBD does not provide patient IDs
        return f"{lbl}_skindiseasebd_{fn.split('.')[0]}"

    # 7. Fitzpatrick 17k (NIH / Groh et al. does not supply patient grouping)
    md5 = str(row.get("md5hash", ""))
    return f"{lbl}_fitz17k_{md5}"


def remediate_manifests():
    print("=" * 60)
    print("DERMTRUST V3.1 MANIFEST SPLIT REMEDIATION")
    print("=" * 60)

    assert TRAIN_V3.exists(), f"Missing {TRAIN_V3}"
    assert VAL_V3.exists(), f"Missing {VAL_V3}"

    df_train = pd.read_csv(TRAIN_V3)
    df_val = pd.read_csv(VAL_V3)

    # Load expansion manifest lookup for patient_group_id
    exp_lookup = {}
    if EXP_MANIFEST.exists():
        df_exp = pd.read_csv(EXP_MANIFEST)
        for _, r in df_exp.iterrows():
            lp = str(Path(r["local_path"]).resolve()).lower()
            exp_lookup[lp] = str(r["patient_group_id"]) if pd.notna(r.get("patient_group_id")) else None

    # Step 1: Identify collisions to remove from training
    # Case 1: SCIN case -3986995536806137754 in train
    scin_leak_mask = df_train["local_path"].str.contains("-5195598708424329883", na=False)
    # Case 2: Figshare afp20080515p1415-f2 pre-augmented in train
    fig_leak_mask = df_train["local_path"].str.contains("aug_0_afp20080515p1415-f2", na=False)

    removal_mask = scin_leak_mask | fig_leak_mask
    removed_rows = df_train[removal_mask]
    print(f"\nIdentified {len(removed_rows)} training entries to remove:")
    for _, r in removed_rows.iterrows():
        print(f"  - [{r['verified_label']}] {Path(r['local_path']).name} (Reason: Cross-split collision with validation)")

    # Filter training manifest
    df_train_v3_1 = df_train[~removal_mask].copy().reset_index(drop=True)
    df_val_v3_1 = df_val.copy().reset_index(drop=True)

    # Step 2: Assign corrected authoritative leakage groups
    print("\nRecomputing authoritative patient/case leakage groups...")
    df_train_v3_1["leakage_group"] = [extract_authoritative_case_group(r, exp_lookup) for _, r in df_train_v3_1.iterrows()]
    df_val_v3_1["leakage_group"] = [extract_authoritative_case_group(r, exp_lookup) for _, r in df_val_v3_1.iterrows()]

    # Verify zero group collisions between train and val
    tr_groups = set(df_train_v3_1["leakage_group"])
    va_groups = set(df_val_v3_1["leakage_group"])
    collision_set = tr_groups.intersection(va_groups)
    print(f"\nCross-Split Patient/Case Group Collisions: {len(collision_set)}")
    assert len(collision_set) == 0, f"Error: Remaining collisions detected: {collision_set}"

    # Verify zero file hash collisions
    tr_md5 = set(df_train_v3_1["md5hash"])
    va_md5 = set(df_val_v3_1["md5hash"])
    hash_collision_set = tr_md5.intersection(va_md5)
    print(f"Cross-Split MD5 Hash Collisions: {len(hash_collision_set)}")
    assert len(hash_collision_set) == 0, f"Error: Hash collision detected: {hash_collision_set}"

    # Export V3.1 manifests
    df_train_v3_1.to_csv(TRAIN_V3_1, index=False)
    df_val_v3_1.to_csv(VAL_V3_1, index=False)
    print(f"\nSuccessfully created revised manifests:")
    print(f"  - {TRAIN_V3_1} ({len(df_train_v3_1)} rows)")
    print(f"  - {VAL_V3_1} ({len(df_val_v3_1)} rows)")

    print("\nTrain V3.1 Class Breakdown:")
    for c, cnt in df_train_v3_1["verified_label"].value_counts().items():
        print(f"  {c:<12}: {cnt}")

    print("\nVal V3.1 Class Breakdown:")
    for c, cnt in df_val_v3_1["verified_label"].value_counts().items():
        print(f"  {c:<12}: {cnt}")


if __name__ == "__main__":
    remediate_manifests()
