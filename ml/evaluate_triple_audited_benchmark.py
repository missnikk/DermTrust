"""
ml/evaluate_triple_audited_benchmark.py
Evaluates the DermTrust 6-Class TFLite model on the Triple-Audited 60-Image Benchmark:
- 10 images per disease (5 Light, 5 Dark) for 6 classes.
- Target: >= 85% accuracy per disease.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_CSV = ROOT_DIR / "ml" / "data" / "triple_audited_benchmark_60.csv"
import sys
TFLITE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT_DIR / "flutter_app" / "assets" / "models" / "dermtrust_classifier.tflite")

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "not_recognised"]

def load_preprocessed_image(path: str) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(rgb, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)
    except Exception:
        return None

def calculate_entropy(probs: np.ndarray) -> float:
    eps = 1e-9
    clipped = np.clip(probs, eps, 1.0)
    return float(-np.sum(clipped * np.log2(clipped)))

def run_evaluation():
    print("=" * 70)
    print("DERMTRUST TRIPLE-AUDITED 60-IMAGE BENCHMARK EVALUATION")
    print(f"Model: {TFLITE_PATH}")
    print(f"Benchmark: {BENCHMARK_CSV}")
    print("=" * 70)

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    df = pd.read_csv(BENCHMARK_CSV)
    results = []

    for idx, row in df.iterrows():
        p = row["image_path"]
        inp = load_preprocessed_image(p)
        if inp is None:
            print(f"Failed to load: {p}")
            continue

        interpreter.set_tensor(input_details["index"], inp)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details["index"])[0]

        pred_idx = int(np.argmax(probs))
        pred_label = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        entropy = calculate_entropy(probs)

        # Sorted top 2
        sorted_indices = np.argsort(probs)[::-1]
        top1 = CLASSES[sorted_indices[0]]
        top2 = CLASSES[sorted_indices[1]]
        p1 = float(probs[sorted_indices[0]])
        p2 = float(probs[sorted_indices[1]])

        is_correct = (pred_label == row["disease"])
        
        results.append({
            "index": idx + 1,
            "disease": row["disease"],
            "skin_tone": row["skin_tone"],
            "ita_angle": row["ita_angle"],
            "image_path": Path(p).name,
            "pred_label": pred_label,
            "confidence": round(conf, 4),
            "entropy": round(entropy, 3),
            "top1": f"{top1} ({p1*100:.1f}%)",
            "top2": f"{top2} ({p2*100:.1f}%)",
            "is_correct": is_correct
        })

    res_df = pd.DataFrame(results)
    
    print("\n--- PER-DISEASE ACCURACY BREAKDOWN ---")
    disease_summary = []
    for d in CLASSES:
        sub = res_df[res_df["disease"] == d]
        total = len(sub)
        corr = sub["is_correct"].sum()
        acc = (corr / total * 100.0) if total > 0 else 0.0
        
        # Skin tone sub-breakdown
        dark_sub = sub[sub["skin_tone"] == "Darker"]
        light_sub = sub[sub["skin_tone"] == "Light"]
        dark_corr = dark_sub["is_correct"].sum() if len(dark_sub) > 0 else 0
        light_corr = light_sub["is_correct"].sum() if len(light_sub) > 0 else 0
        
        status = "PASS (>=85%)" if acc >= 85.0 else "FAIL (<85%)"
        disease_summary.append({
            "disease": d.upper(),
            "correct": f"{corr}/{total}",
            "accuracy": f"{acc:.1f}%",
            "darker_acc": f"{dark_corr}/{len(dark_sub)} ({dark_corr/len(dark_sub)*100:.1f}%)" if len(dark_sub) > 0 else "N/A",
            "light_acc": f"{light_corr}/{len(light_sub)} ({light_corr/len(light_sub)*100:.1f}%)" if len(light_sub) > 0 else "N/A",
            "status": status
        })
    
    summary_df = pd.DataFrame(disease_summary)
    print(summary_df.to_string(index=False))

    total_correct = res_df["is_correct"].sum()
    total_imgs = len(res_df)
    print(f"\nOVERALL BENCHMARK ACCURACY: {total_correct}/{total_imgs} ({total_correct/total_imgs*100:.1f}%)")
    
    # Fairness disparity
    dark_all = res_df[res_df["skin_tone"] == "Darker"]
    light_all = res_df[res_df["skin_tone"] == "Light"]
    dark_acc = dark_all["is_correct"].sum() / len(dark_all) * 100.0
    light_acc = light_all["is_correct"].sum() / len(light_all) * 100.0
    print(f"Darker Skin Accuracy: {dark_all['is_correct'].sum()}/{len(dark_all)} ({dark_acc:.1f}%)")
    print(f"Light Skin Accuracy:  {light_all['is_correct'].sum()}/{len(light_all)} ({light_acc:.1f}%)")
    print(f"Skin Tone Fairness Disparity: |{dark_acc:.1f}% - {light_acc:.1f}%| = {abs(dark_acc - light_acc):.1f}%")

    # Misclassified details
    failures = res_df[~res_df["is_correct"]]
    print(f"\nMISCLASSIFIED CASES ({len(failures)} total):")
    if len(failures) > 0:
        print(failures[["disease", "skin_tone", "image_path", "pred_label", "top1", "top2", "confidence"]].to_string(index=False))

    out_json = ROOT_DIR / "ml" / "experiments" / "triple_audited_benchmark_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "overall_accuracy": total_correct / total_imgs,
            "per_disease": disease_summary,
            "disparity": abs(dark_acc - light_acc),
            "failures": failures.to_dict(orient="records")
        }, f, indent=2)
    print(f"\nResults saved to {out_json}")

if __name__ == "__main__":
    run_evaluation()
