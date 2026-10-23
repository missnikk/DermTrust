"""
DermTrust - Candidate V3 vs Production Baseline Comparative Evaluation
Script: ml/compare_candidate_v3.py
Role: Rigorously compares Candidate V3 against the Production Baseline model
      on the 100% natural clinical validation split (val_5class_manifest_v3.csv, N=273).
      Enforces strict acceptance gates:
        1. Macro-F1 candidate >= baseline
        2. Overall accuracy >= baseline - 0.02
        3. Zero class collapse (all class recalls > 0)
        4. Melasma recall >= baseline
      If and only if ALL gates pass, exports and deploys candidate assets to production.
"""

from __future__ import annotations
import os
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"
MODELS_DIR = ROOT_DIR / "ml" / "models"
EXP_MODELS_DIR = ROOT_DIR / "ml" / "models" / "experiments"
FLUTTER_ASSETS_DIR = ROOT_DIR / "flutter_app" / "assets" / "models"
DOCS_DIR = ROOT_DIR / "docs"

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_dataset(manifest_path: Path):
    df = pd.read_csv(manifest_path)
    imgs = []
    y_true = []
    y_idx = []
    valid_rows = []

    for _, row in df.iterrows():
        p = row["local_path"]
        lbl = str(row["verified_label"]).lower()
        if lbl in CLASS_TO_IDX and os.path.exists(p):
            try:
                with Image.open(p) as img:
                    rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    arr = np.array(rgb, dtype=np.float32) / 255.0
                    imgs.append(arr)
                    y_true.append(lbl)
                    y_idx.append(CLASS_TO_IDX[lbl])
                    valid_rows.append(row)
            except Exception as e:
                print(f"Error loading {p}: {e}")

    return np.array(imgs, dtype=np.float32), y_true, np.array(y_idx, dtype=np.int32), pd.DataFrame(valid_rows)


def evaluate_model(model, X, y_true, y_idx):
    probs = model.predict(X, batch_size=32, verbose=0)
    preds_idx = np.argmax(probs, axis=1)
    preds = [CLASSES[i] for i in preds_idx]

    acc = float(accuracy_score(y_true, preds))
    report = classification_report(y_true, preds, target_names=CLASSES, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, preds, labels=CLASSES).tolist()
    mean_conf = float(np.mean(np.max(probs, axis=1)))

    return {
        "accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "mean_confidence": mean_conf,
        "classification_report": report,
        "confusion_matrix": cm,
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
        "probs": probs,
    }


def export_and_promote(candidate_model_path: Path, candidate_cam_path: Path, metrics: dict):
    print("\n" + "=" * 70)
    print("PROMOTING CANDIDATE V3 TO PRODUCTION ASSETS")
    print("=" * 70)

    # 1. Update Keras model
    prod_keras = MODELS_DIR / "dermtrust_mobilenetv2_5class.keras"
    shutil.copyfile(candidate_model_path, prod_keras)
    print(f"  [PROMOTED] Keras Model: {prod_keras}")

    # 2. Export Classifier TFLite (Float32 for full compatibility)
    model = tf.keras.models.load_model(str(candidate_model_path), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = []
    tflite_model = converter.convert()

    tflite_dest = FLUTTER_ASSETS_DIR / "dermtrust_classifier.tflite"
    with open(tflite_dest, "wb") as f:
        f.write(tflite_model)
    print(f"  [PROMOTED] Classifier TFLite: {tflite_dest} ({len(tflite_model)/1024/1024:.2f} MB)")

    # 3. Export CAM TFLite
    cam_model = tf.keras.models.load_model(str(candidate_cam_path), compile=False)
    cam_converter = tf.lite.TFLiteConverter.from_keras_model(cam_model)
    cam_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    cam_converter.optimizations = []
    cam_tflite = cam_converter.convert()

    cam_dest = FLUTTER_ASSETS_DIR / "dermtrust_cam.tflite"
    with open(cam_dest, "wb") as f:
        f.write(cam_tflite)
    print(f"  [PROMOTED] CAM TFLite: {cam_dest} ({len(cam_tflite)/1024/1024:.2f} MB)")

    # 4. Extract CAM dense weights
    dense_layer = model.get_layer("predictions")
    weights = dense_layer.get_weights()[0]  # [1280, 5]
    weights_transposed = weights.T.tolist()

    cam_dict = {
        "description": "Grad-CAM alpha weights (dense layer weights for GAP model)",
        "num_features": int(weights.shape[0]),
        "num_classes": int(weights.shape[1]),
        "feature_map_size": 7,
        "classes": CLASSES,
        "weights": weights_transposed,
    }
    for idx, cls_name in enumerate(CLASSES):
        cam_dict[cls_name] = weights_transposed[idx]

    cam_weights_dest = FLUTTER_ASSETS_DIR / "cam_weights.json"
    with open(cam_weights_dest, "w") as f:
        json.dump(cam_dict, f, indent=2)
    print(f"  [PROMOTED] CAM Weights JSON: {cam_weights_dest}")

    # 5. Update model metadata
    metadata = {
        "model_name": "DermTrust-MobileNetV2-5Class-XAI-v3",
        "version": "3.0.0",
        "evaluated_classes": CLASSES,
        "classes": CLASSES,
        "target_disease_scope": CLASSES,
        "input_size": 224,
        "input_tensor_shape": [1, 224, 224, 3],
        "cam_feature_shape": [1, 7, 7, 1280],
        "input_mean": 0.0,
        "input_std": 255.0,
        "architecture": "MobileNetV2 (alpha=0.75) + Spatial Attention + GAP + Dense",
        "grad_cam_property": "Exact alpha_k^c == w_k^c equivalence (no intermediate BatchNorm)",
        "rejection_threshold": 0.60,
        "max_entropy_threshold": 1.40,
        "min_blur_laplacian_variance": 30.0,
        "metrics": {
            "evaluation_date": "2026-10-23",
            "overall_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mean_confidence": metrics["mean_confidence"],
            "per_class": metrics["per_class"],
        },
    }
    metadata_dest = FLUTTER_ASSETS_DIR / "model_metadata.json"
    with open(metadata_dest, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [PROMOTED] Metadata JSON: {metadata_dest}")


def main():
    val_csv = DATA_DIR / "val_5class_manifest_v3.csv"
    baseline_path = MODELS_DIR / "dermtrust_mobilenetv2_5class.keras"
    candidate_path = EXP_MODELS_DIR / "candidate_v3_best.keras"
    candidate_cam_path = EXP_MODELS_DIR / "candidate_v3_cam.keras"

    if not candidate_path.exists():
        print(f"Error: Candidate model not found at {candidate_path}")
        return

    print("Loading V3 pure natural validation dataset...")
    X_val, y_true, y_idx, val_rows = load_dataset(val_csv)
    print(f"Loaded {len(X_val)} validation samples.")

    print("\nEvaluating Production Baseline Model...")
    baseline_model = tf.keras.models.load_model(str(baseline_path), compile=False)
    base_res = evaluate_model(baseline_model, X_val, y_true, y_idx)

    print("\nEvaluating Candidate V3 Model...")
    candidate_model = tf.keras.models.load_model(str(candidate_path), compile=False)
    cand_res = evaluate_model(candidate_model, X_val, y_true, y_idx)

    # Print comparison
    print("\n" + "=" * 78)
    print(f"{'METRIC':<20} | {'BASELINE':<15} | {'CANDIDATE V3':<15} | {'DIFFERENCE':<15}")
    print("-" * 78)
    print(f"{'Overall Accuracy':<20} | {base_res['accuracy']*100:13.2f}% | {cand_res['accuracy']*100:13.2f}% | {(cand_res['accuracy'] - base_res['accuracy'])*100:+13.2f}%")
    print(f"{'Macro-F1 Score':<20} | {base_res['macro_f1']:14.4f} | {cand_res['macro_f1']:14.4f} | {(cand_res['macro_f1'] - base_res['macro_f1']):+14.4f}")
    print(f"{'Weighted-F1':<20} | {base_res['weighted_f1']:14.4f} | {cand_res['weighted_f1']:14.4f} | {(cand_res['weighted_f1'] - base_res['weighted_f1']):+14.4f}")
    print(f"{'Mean Confidence':<20} | {base_res['mean_confidence']:14.4f} | {cand_res['mean_confidence']:14.4f} | {(cand_res['mean_confidence'] - base_res['mean_confidence']):+14.4f}")
    print("-" * 78)

    print("\nPER-CLASS RECALL & F1 COMPARISON:")
    print(f"{'CLASS':<12} | {'BASE RECALL':<12} | {'CAND RECALL':<12} | {'BASE F1':<10} | {'CAND F1':<10} | {'SUPPORT':<8}")
    print("-" * 78)
    for c in CLASSES:
        b_rec = base_res["per_class"][c]["recall"]
        c_rec = cand_res["per_class"][c]["recall"]
        b_f1 = base_res["per_class"][c]["f1"]
        c_f1 = cand_res["per_class"][c]["f1"]
        supp = base_res["per_class"][c]["support"]
        print(f"{c:<12} | {b_rec:11.4f}  | {c_rec:11.4f}  | {b_f1:9.4f}  | {c_f1:9.4f}  | {supp:<8}")
    print("=" * 78)

    # Acceptance Gates
    gate_macro_f1 = cand_res["macro_f1"] >= base_res["macro_f1"]
    gate_acc = cand_res["accuracy"] >= (base_res["accuracy"] - 0.02)
    gate_no_collapse = all(cand_res["per_class"][c]["recall"] > 0 for c in CLASSES)
    gate_melasma = cand_res["per_class"]["melasma"]["recall"] >= base_res["per_class"]["melasma"]["recall"]

    print("\nACCEPTANCE GATES STATUS:")
    print(f"  Gate 1 - Macro-F1 Non-inferiority (Cand >= Base):   {'PASSED' if gate_macro_f1 else 'FAILED'} ({cand_res['macro_f1']:.4f} vs {base_res['macro_f1']:.4f})")
    print(f"  Gate 2 - Overall Accuracy Safety (>= Base - 2%):      {'PASSED' if gate_acc else 'FAILED'} ({cand_res['accuracy']*100:.2f}% vs {base_res['accuracy']*100:.2f}%)")
    print(f"  Gate 3 - Zero-Class Collapse (All Recalls > 0):       {'PASSED' if gate_no_collapse else 'FAILED'}")
    print(f"  Gate 4 - Melasma Recall Non-inferiority:              {'PASSED' if gate_melasma else 'FAILED'} ({cand_res['per_class']['melasma']['recall']:.4f} vs {base_res['per_class']['melasma']['recall']:.4f})")

    all_passed = gate_macro_f1 and gate_acc and gate_no_collapse and gate_melasma

    # Save comparative audit report
    comparison_summary = {
        "evaluation_dataset": "val_5class_manifest_v3.csv (N=273, 100% natural clinical)",
        "baseline_model": str(baseline_path),
        "candidate_model": str(candidate_path),
        "baseline_metrics": {
            "accuracy": base_res["accuracy"],
            "macro_f1": base_res["macro_f1"],
            "weighted_f1": base_res["weighted_f1"],
            "per_class": base_res["per_class"],
            "confusion_matrix": base_res["confusion_matrix"],
        },
        "candidate_metrics": {
            "accuracy": cand_res["accuracy"],
            "macro_f1": cand_res["macro_f1"],
            "weighted_f1": cand_res["weighted_f1"],
            "per_class": cand_res["per_class"],
            "confusion_matrix": cand_res["confusion_matrix"],
        },
        "gates": {
            "gate_macro_f1": bool(gate_macro_f1),
            "gate_accuracy": bool(gate_acc),
            "gate_no_collapse": bool(gate_no_collapse),
            "gate_melasma": bool(gate_melasma),
            "all_gates_passed": bool(all_passed),
        },
        "action_taken": "PROMOTED_TO_PRODUCTION" if all_passed else "RETAINED_BASELINE_PRODUCTION_ASSETS",
    }

    report_path = DOCS_DIR / "v3_candidate_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\nComparison report saved to: {report_path}")

    if all_passed:
        export_and_promote(candidate_path, candidate_cam_path, cand_res)
    else:
        print("\n" + "=" * 70)
        print("CANDIDATE DID NOT PASS ALL ACCEPTANCE GATES.")
        print("PRESERVING EXISTING PRODUCTION MODEL ASSETS UNTOUCHED.")
        print("=" * 70)


if __name__ == "__main__":
    main()
