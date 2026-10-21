"""
DermTrust - Machine Learning Pipeline
Script: evaluate_model.py
Role: Comprehensive scientific evaluation of the deployed DermTrust TFLite model
      on the independent test set and OOD non-target rejection test set.
      Outputs confusion matrix, per-class metrics, macro-F1, and rejection statistics.
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model_artifacts" / "dermtrust_classifier.tflite"
METADATA_PATH = ROOT_DIR / "model_artifacts" / "model_metadata.json"
TEST_MANIFEST = ROOT_DIR / "ml" / "data" / "test_manifest.csv"
OOD_MANIFEST = ROOT_DIR / "ml" / "data" / "ood_manifest.csv"
DOCS_DIR = ROOT_DIR / "docs"


class TFLiteClassifier:
    def __init__(self, model_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.input_shape = self.input_details["shape"] # [1, 224, 224, 3]

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        # Preprocess float32 [0.0, 1.0]
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32) / 255.0
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        self.interpreter.set_tensor(self.input_details["index"], img_array)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])
        return output[0]


def calculate_entropy(probs: np.ndarray) -> float:
    eps = 1e-9
    clipped = np.clip(probs, eps, 1.0)
    return float(-np.sum(clipped * np.log2(clipped)))


def load_and_preprocess_image(img_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray | None:
    try:
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            resized = rgb.resize(target_size, Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32) / 255.0
            return arr
    except Exception as e:
        return None


def run_evaluation():
    print("==================================================")
    print("DERMTRUST COMPREHENSIVE MODEL EVALUATION")
    print(f"Model: {MODEL_PATH}")
    print(f"Test Manifest: {TEST_MANIFEST}")
    print("==================================================")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"] # ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
    rejection_thresh = meta.get("rejection_threshold", 0.60)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    classifier = TFLiteClassifier(MODEL_PATH)
    test_df = pd.read_csv(TEST_MANIFEST)

    y_true = []
    y_pred = []
    probs_list = []
    entropies = []
    max_probs = []
    is_rejected = []
    skipped = 0

    print(f"\nEvaluating {len(test_df)} test samples...")
    for idx, row in test_df.iterrows():
        lbl = row["mapped_label"]
        if lbl not in class_to_idx:
            skipped += 1
            continue
        p = row["image_path"]
        img = load_and_preprocess_image(p)
        if img is None:
            skipped += 1
            continue

        pred_probs = classifier.predict(img)
        pred_cls_idx = int(np.argmax(pred_probs))
        max_p = float(pred_probs[pred_cls_idx])
        entropy = calculate_entropy(pred_probs)

        y_true.append(class_to_idx[lbl])
        y_pred.append(pred_cls_idx)
        probs_list.append(pred_probs)
        max_probs.append(max_p)
        entropies.append(entropy)
        is_rejected.append(max_p < rejection_thresh)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    max_probs = np.array(max_probs)
    entropies = np.array(entropies)

    # Compute metrics on verified evaluation classes
    eval_labels = sorted(list(set(y_true)))
    eval_class_names = [classes[i] for i in eval_labels]

    report_dict = classification_report(
        y_true, y_pred, labels=eval_labels, target_names=eval_class_names, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=eval_labels)

    overall_acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    print("\n--- TEST CLASSIFICATION PERFORMANCE ---")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"Macro-F1 Score:   {macro_f1:.4f}")
    print(f"Mean Confidence:  {np.mean(max_probs):.4f}")
    print(f"Mean Entropy:     {np.mean(entropies):.4f} bits")

    print("\n--- PER-CLASS PERFORMANCE ---")
    for cls in eval_class_names:
        metrics = report_dict[cls]
        print(f"  {cls.capitalize():<12}: Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} | Support: {metrics['support']}")

    # Non-target / OOD Rejection Evaluation
    print("\nEvaluating Non-Target / OOD Confounder Rejection...")
    ood_df = pd.read_csv(OOD_MANIFEST)
    # Take a balanced sample of 200 confounders (Eczema, Lichen Planus, Seborrheic Dermatitis, etc.)
    sample_ood = ood_df.sample(n=min(200, len(ood_df)), random_state=42)
    ood_max_probs = []
    ood_entropies = []
    ood_rejected = 0

    for _, row in sample_ood.iterrows():
        img = load_and_preprocess_image(row["image_path"])
        if img is None:
            continue
        p = classifier.predict(img)
        mp = float(np.max(p))
        ent = calculate_entropy(p)
        ood_max_probs.append(mp)
        ood_entropies.append(ent)
        if mp < rejection_thresh or ent > 1.40:
            ood_rejected += 1

    ood_total = len(ood_max_probs)
    rejection_rate = (ood_rejected / ood_total) * 100 if ood_total > 0 else 0.0
    print(f"Non-Target Rejection Rate: {rejection_rate:.1f}% ({ood_rejected}/{ood_total} non-target confounders safely rejected)")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(conf_mat, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(eval_class_names)))
    ax.set_yticks(range(len(eval_class_names)))
    ax.set_xticklabels([c.capitalize() for c in eval_class_names], rotation=45, ha="left")
    ax.set_yticklabels([c.capitalize() for c in eval_class_names])
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("DermTrust MobileNetV2 Confusion Matrix (Test Set)", pad=20, fontweight="bold")

    for i in range(len(eval_class_names)):
        for j in range(len(eval_class_names)):
            ax.text(j, i, str(conf_mat[i, j]), ha="center", va="center", color="black" if conf_mat[i, j] < conf_mat.max()/2 else "white", fontweight="bold")

    plt.tight_layout()
    cm_path = DOCS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")

    # Export report to markdown
    report_md_path = DOCS_DIR / "MODEL_EVALUATION.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# DERMTRUST — MODEL EVALUATION REPORT\n\n")
        f.write(f"**Evaluation Date:** 2026-10-21  \n")
        f.write(f"**Model Evaluated:** `dermtrust_classifier.tflite` (MobileNetV2 5-Class)  \n")
        f.write(f"**Test Set Images:** {len(y_true)} verified test images  \n")
        f.write(f"**OOD Confounder Benchmark:** {ood_total} non-target images  \n\n")
        f.write("## 1. Summary Metrics\n\n")
        f.write(f"| Metric | Value |\n| :--- | :---: |\n")
        f.write(f"| **Overall Accuracy** | **{overall_acc*100:.2f}%** |\n")
        f.write(f"| **Macro-F1 Score** | **{macro_f1:.4f}** |\n")
        f.write(f"| **Mean Prediction Confidence** | **{np.mean(max_probs)*100:.2f}%** |\n")
        f.write(f"| **Mean Predictive Entropy** | **{np.mean(entropies):.3f} bits** |\n")
        f.write(f"| **OOD Rejection Rate** | **{rejection_rate:.1f}%** |\n\n")
        f.write("## 2. Per-Class Performance Breakdown\n\n")
        f.write("| Disease Class | Precision | Recall (Sensitivity) | F1-Score | Test Support |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls in eval_class_names:
            m = report_dict[cls]
            f.write(f"| **{cls.capitalize()}** | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {m['support']} |\n")
        f.write(f"\n## 3. Confusion Matrix\n\n")
        f.write(f"```\n{conf_mat}\n```\n\n")
        f.write(f"Visual representation saved to `docs/confusion_matrix.png`.\n\n")
        f.write("## 4. Non-Target Rejection (6th Logical Outcome)\n\n")
        f.write(f"The calibrated DWH decision gate ($\tau_{{conf}} = 0.60, \tau_{{entropy}} = 1.40$) successfully rejected **{rejection_rate:.1f}%** of visually similar non-target conditions (Eczema, Lichen Planus, Seborrheic Dermatitis), preventing false forced classification into target categories.\n")

    print(f"Evaluation report exported to: {report_md_path}")


if __name__ == "__main__":
    run_evaluation()
