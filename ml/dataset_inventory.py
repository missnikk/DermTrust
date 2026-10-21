"""
DermTrust - Machine Learning Pipeline
Script: dataset_inventory.py
Role: Inspects, validates, and profiles the Fitzpatrick 17K dataset
      specifically for DermTrust's 5 target disease classes:
      1. Vitiligo
      2. Melasma
      3. Ringworm (Tinea corporis / tinea)
      4. Acne (Acne vulgaris)
      5. Psoriasis
      Also profiles non-target / visually similar conditions for rejection/uncertainty benchmarking.
"""

from __future__ import annotations
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Paths
DEFAULT_CSV_PATH = Path(r"C:\Users\nikit\Downloads\fitzpatrick17k.csv")
DEFAULT_IMG_DIR = Path(r"C:\Users\nikit\Downloads\fitzpatrick17k\data\finalfitz17k")
OUTPUT_REPORT_DIR = Path(__file__).resolve().parent.parent / "docs"

# Target class mapping
TARGET_DISEASE_MAP = {
    # 1. Vitiligo
    "vitiligo": "vitiligo",
    
    # 2. Melasma
    "melasma": "melasma",
    
    # 3. Ringworm / Tinea
    "tinea corporis": "ringworm",
    "tinea faciei": "ringworm",
    "tinea": "ringworm",
    "ringworm": "ringworm",
    "tinea cruris": "ringworm",
    
    # 4. Acne
    "acne": "acne",
    "acne vulgaris": "acne",
    "cystic acne": "acne",
    
    # 5. Psoriasis
    "psoriasis": "psoriasis",
    "pustular psoriasis": "psoriasis",
    "guttate psoriasis": "psoriasis",
    "erythrodermic psoriasis": "psoriasis",
    "plaque psoriasis": "psoriasis",
}

# Relevant non-target confounders for rejection / uncertainty benchmark
CONFUSE_NON_TARGET_MAP = {
    "eczema": "non_target_eczema",
    "atopic dermatitis": "non_target_atopic_dermatitis",
    "seborrheic dermatitis": "non_target_seborrheic_dermatitis",
    "lichen planus": "non_target_lichen_planus",
    "pityriasis rosea": "non_target_pityriasis_rosea",
    "post inflammatory hyperpigmentation": "non_target_post_inflammatory_hyperpigmentation",
    "rosacea": "non_target_rosacea",
}


def normalize_label(raw_label: str) -> tuple[str | None, str]:
    """
    Normalizes a clinical diagnosis label to one of the 5 DermTrust target classes,
    or a non-target confounder, or other.
    """
    clean = " ".join(raw_label.strip().lower().replace("_", " ").split())
    if clean in TARGET_DISEASE_MAP:
        return TARGET_DISEASE_MAP[clean], "target"
    for k, v in TARGET_DISEASE_MAP.items():
        if k in clean:
            return v, "target"
            
    if clean in CONFUSE_NON_TARGET_MAP:
        return CONFUSE_NON_TARGET_MAP[clean], "non_target_confounder"
    for k, v in CONFUSE_NON_TARGET_MAP.items():
        if k in clean:
            return v, "non_target_confounder"
            
    return None, "other"


def classify_fitzpatrick(scale_str: str) -> str:
    """
    Maps numeric Fitzpatrick scale 1-6 into operational groups:
    Light: Types I-III
    Darker: Types IV-VI
    Unknown: -1 or invalid
    """
    try:
        val = int(scale_str)
        if val in (1, 2, 3):
            return "Light (Types I-III)"
        elif val in (4, 5, 6):
            return "Darker (Types IV-VI)"
        else:
            return "Unknown / Unclassified"
    except (ValueError, TypeError):
        return "Unknown / Unclassified"


def run_inventory(csv_path: Path, img_dir: Path) -> dict:
    print(f"==================================================")
    print(f"DERMTRUST DATASET INVENTORY & AUDIT")
    print(f"CSV Metadata: {csv_path}")
    print(f"Image Directory: {img_dir}")
    print(f"==================================================")

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found at {img_dir}")

    # Index available image files
    print("\nScanning image directory...")
    image_files = set(f.name for f in img_dir.iterdir() if f.is_file())
    print(f"Total physical image files found in disk: {len(image_files):,}")

    total_rows = 0
    missing_image_count = 0
    corrupt_or_unreadable = 0
    
    target_counts = Counter()
    non_target_counts = Counter()
    other_counts = Counter()
    
    # Matrix: target_class x fitzpatrick_group
    disease_by_skin_tone = defaultdict(lambda: Counter())
    # Matrix: target_class x exact_fitzpatrick (1..6)
    disease_by_exact_fitz = defaultdict(lambda: Counter())
    
    # Store records for target diseases
    valid_target_records = []
    valid_confounder_records = []

    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            md5 = (row.get("md5hash") or "").strip()
            raw_label = (row.get("label") or "").strip()
            fitz_val = (row.get("fitzpatrick_scale") or "-1").strip()
            
            # File resolution
            img_filename = f"{md5}.jpg"
            has_image = img_filename in image_files
            if not has_image:
                # Also check without extension if applicable
                if md5 in image_files:
                    img_filename = md5
                    has_image = True
                else:
                    missing_image_count += 1
                    continue
            
            mapped_class, category = normalize_label(raw_label)
            skin_group = classify_fitzpatrick(fitz_val)
            
            record = {
                "md5": md5,
                "filename": img_filename,
                "raw_label": raw_label,
                "mapped_class": mapped_class,
                "fitzpatrick_scale": fitz_val,
                "skin_group": skin_group,
                "qc": row.get("qc", ""),
                "url": row.get("url", ""),
            }
            
            if category == "target":
                target_counts[mapped_class] += 1
                disease_by_skin_tone[mapped_class][skin_group] += 1
                disease_by_exact_fitz[mapped_class][fitz_val] += 1
                valid_target_records.append(record)
            elif category == "non_target_confounder":
                non_target_counts[mapped_class] += 1
                valid_confounder_records.append(record)
            else:
                other_counts[raw_label] += 1

    print(f"\nMetadata Processing Complete:")
    print(f"- Total rows in metadata: {total_rows:,}")
    print(f"- Missing physical image files: {missing_image_count:,}")
    print(f"- Total target images available: {len(valid_target_records):,}")
    print(f"- Total non-target confounders available: {len(valid_confounder_records):,}")
    print(f"- Other condition images available (out-of-distribution pool): {sum(other_counts.values()):,}")

    print(f"\n--- 5 Target Disease Representation ---")
    for cls_name in ["vitiligo", "melasma", "ringworm", "acne", "psoriasis"]:
        cnt = target_counts[cls_name]
        print(f"  {cls_name.capitalize():<12}: {cnt:>5} images")

    print(f"\n--- Disease-by-Skin-Tone Count Matrix (Fitzpatrick I-III vs IV-VI) ---")
    header = f"{'Disease':<12} | {'Light (I-III)':<14} | {'Darker (IV-VI)':<14} | {'Unknown':<10} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    for cls_name in ["vitiligo", "melasma", "ringworm", "acne", "psoriasis"]:
        lg = disease_by_skin_tone[cls_name]["Light (Types I-III)"]
        dk = disease_by_skin_tone[cls_name]["Darker (Types IV-VI)"]
        uk = disease_by_skin_tone[cls_name]["Unknown / Unclassified"]
        tot = target_counts[cls_name]
        print(f"{cls_name.capitalize():<12} | {lg:<14} | {dk:<14} | {uk:<10} | {tot:<8}")

    print(f"\n--- Non-Target Confounder Representation (for open-set rejection evaluation) ---")
    for k, v in non_target_counts.most_common():
        print(f"  {k:<35}: {v:>4} images")

    summary_data = {
        "total_rows": total_rows,
        "physical_images_found": len(image_files),
        "target_counts": dict(target_counts),
        "disease_by_skin_tone": {k: dict(v) for k, v in disease_by_skin_tone.items()},
        "disease_by_exact_fitz": {k: dict(v) for k, v in disease_by_exact_fitz.items()},
        "non_target_counts": dict(non_target_counts),
        "valid_target_total": len(valid_target_records),
        "valid_confounder_total": len(valid_confounder_records),
    }

    # Save summary json to ml/
    summary_path = Path(__file__).resolve().parent / "dataset_inventory_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nInventory summary exported to: {summary_path}")
    
    return summary_data


if __name__ == "__main__":
    csv_p = DEFAULT_CSV_PATH if len(sys.argv) < 2 else Path(sys.argv[1])
    img_p = DEFAULT_IMG_DIR if len(sys.argv) < 3 else Path(sys.argv[2])
    run_inventory(csv_p, img_p)
