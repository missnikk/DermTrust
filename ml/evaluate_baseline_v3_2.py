"""
DermTrust — Comprehensive Baseline Evaluation Script
Evaluates the existing production model (dermtrust_mobilenetv2_5class.keras)
on the exact V3.2 validation manifest (N=300) and V3.1 validation manifest (N=273).
Saves all per-image predictions, confusion matrices, and metrics to:
ml/experiments/exp9_v3_2_candidate/baseline_evaluation_v3_2.json
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "ml" / "models"
DATA_DIR = ROOT_DIR / "ml" / "data"
EXP_DIR = ROOT_DIR / "ml" / "experiments" / "exp9_v3_2_candidate"

PROD_KERAS = MODELS_DIR / "dermtrust_mobilenetv2_5class.keras"
VAL_V3_2 = DATA_DIR / "val_5class_manifest_v3_2.csv"
VAL_V3_1 = DATA_DIR / "val_5class_manifest_v3_1.csv"
OOD_CSV = DATA_DIR / "ood_5class_manifest.csv"
OUT_JSON = EXP_DIR / "baseline_evaluation_v3_2.json"

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / total) + ((z**2) / (4 * (total**2))))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def compute_entropy_bits(probs: np.ndarray) -> float:
    eps = 1e-9
    clipped = np.clip(probs, eps, 1.0)
    return float(-np.sum(clipped * np.log2(clipped)))


def load_and_preprocess_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        return np.array(rgb, dtype=np.float32) / 255.0


def evaluate_manifest(model: keras.Model, manifest_path: Path, split_name: str) -> dict:
    df = pd.read_csv(manifest_path)
    y_true_indices = []
    y_pred_indices = []
    y_pred_probs = []
    per_image_results = []

    for _, row in df.iterrows():
        p = row["local_path"]
        lbl = row["verified_label"]
        true_idx = CLASS_TO_IDX[lbl]
        y_true_indices.append(true_idx)

        arr = load_and_preprocess_image(p)
        probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        entropy = compute_entropy_bits(probs)

        y_pred_indices.append(pred_idx)
        y_pred_probs.append(probs)

        per_image_results.append({
            "image_id": row.get("image_id", Path(p).name),
            "local_path": p,
            "dataset_source": row.get("dataset_source", "unknown"),
            "true_label": lbl,
            "pred_label": CLASSES[pred_idx],
            "correct": bool(true_idx == pred_idx),
            "max_confidence": float(probs[pred_idx]),
            "entropy_bits": entropy,
            "probabilities": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
        })

    y_true = np.array(y_true_indices)
    y_pred = np.array(y_pred_indices)
    y_probs = np.array(y_pred_probs)

    overall_acc = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    loss_val = float(log_loss(y_true, y_probs, labels=list(range(len(CLASSES)))))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist()

    precisions = precision_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)

    per_class = {}
    for i, c in enumerate(CLASSES):
        support = int(np.sum(y_true == i))
        correct = int(np.sum((y_true == i) & (y_pred == i)))
        rec = float(recalls[i])
        low_ci, up_ci = wilson_score_interval(correct, support)
        per_class[c] = {
            "precision": float(precisions[i]),
            "recall": rec,
            "f1_score": float(f1s[i]),
            "support": support,
            "correct": correct,
            "recall_95ci": [low_ci, up_ci],
        }

    return {
        "split_name": split_name,
        "total_samples": len(df),
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "log_loss": loss_val,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "per_image_results": per_image_results,
    }


def evaluate_ood(model: keras.Model) -> dict:
    if not OOD_CSV.exists():
        return {"status": "skipped", "reason": "ood manifest missing"}
    
    df = pd.read_csv(OOD_CSV)
    conditions = sorted(df["verified_label"].unique())
    
    total_evaluated = 0
    total_rejected = 0
    by_condition = {}

    for cond in conditions:
        sub = df[df["verified_label"] == cond].head(100)
        cond_total = 0
        cond_rejected = 0
        for _, row in sub.iterrows():
            p = row["local_path"]
            if not Path(p).exists():
                continue
            arr = load_and_preprocess_image(p)
            probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
            max_conf = float(np.max(probs))
            ent = compute_entropy_bits(probs)

            is_rejected = (max_conf < 0.60) or (ent > 1.40)
            cond_total += 1
            if is_rejected:
                cond_rejected += 1

        total_evaluated += cond_total
        total_rejected += cond_rejected
        by_condition[cond] = {
            "evaluated": cond_total,
            "rejected": cond_rejected,
            "rejection_rate": (cond_rejected / cond_total) if cond_total > 0 else 0.0,
        }

    rej_rate = (total_rejected / total_evaluated) if total_evaluated > 0 else 0.0
    return {
        "total_evaluated": total_evaluated,
        "total_rejected": total_rejected,
        "rejection_rate": rej_rate,
        "false_accept_rate": 1.0 - rej_rate,
        "by_condition": by_condition,
    }


def main():
    print("=" * 65)
    print("EVALUATING PRODUCTION BASELINE MODEL")
    print(f"Model Path: {PROD_KERAS}")
    print("=" * 65)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(PROD_KERAS, compile=False)

    print("\n--- 1. Evaluating on V3.2 Validation Set (N=300) ---")
    res_v3_2 = evaluate_manifest(model, VAL_V3_2, "V3.2 Validation")
    print(f"Overall Accuracy:  {res_v3_2['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy: {res_v3_2['balanced_accuracy']:.4f}")
    print(f"Macro-F1:          {res_v3_2['macro_f1']:.4f}")
    print(f"Log Loss:          {res_v3_2['log_loss']:.4f}")
    print("Per-class Metrics (V3.2):")
    for c, m in res_v3_2["per_class_metrics"].items():
        ci = m["recall_95ci"]
        print(f"  {c:<10}: Prec={m['precision']:.3f}, Rec={m['recall']:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]), F1={m['f1_score']:.3f}, Supp={m['support']}")

    print("\nConfusion Matrix (V3.2):")
    print("Pred ->", CLASSES)
    for c, row in zip(CLASSES, res_v3_2["confusion_matrix"]):
        print(f"{c:<10}", row)

    print("\n--- 2. Evaluating on V3.1 Validation Set (N=273, Historical Baseline) ---")
    res_v3_1 = evaluate_manifest(model, VAL_V3_1, "V3.1 Validation")
    print(f"Overall Accuracy:  {res_v3_1['overall_accuracy']:.4f}")
    print(f"Macro-F1:          {res_v3_1['macro_f1']:.4f}")
    mel_m = res_v3_1["per_class_metrics"]["melasma"]
    print(f"Melasma Rec (V3.1): {mel_m['recall']:.3f} (Correct: {mel_m['correct']} / {mel_m['support']})")

    print("\n--- 3. Evaluating on OOD Confounder Dataset (Rejection Behavior) ---")
    ood_res = evaluate_ood(model)
    print(f"OOD Evaluated: {ood_res.get('total_evaluated')}")
    print(f"OOD Rejected:  {ood_res.get('total_rejected')} ({ood_res.get('rejection_rate', 0)*100:.1f}%)")
    print(f"OOD False-Accept Rate: {ood_res.get('false_accept_rate', 0)*100:.1f}%")

    output_payload = {
        "model_file": str(PROD_KERAS),
        "v3_2_validation": res_v3_2,
        "v3_1_validation": res_v3_1,
        "ood_evaluation": ood_res,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved full baseline evaluation results to: {OUT_JSON}")


if __name__ == "__main__":
    main()
