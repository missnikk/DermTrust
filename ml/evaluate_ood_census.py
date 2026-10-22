"""
DermTrust - Full OOD Census Evaluation Script
Evaluates ALL 2,183 Out-of-Distribution non-target confounders
to eliminate sampling discrepancies and produce exact condition-by-condition metrics.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model_artifacts" / "dermtrust_classifier.tflite"
OOD_CSV = ROOT / "ml" / "data" / "ood_5class_manifest.csv"
OUTPUT_JSON = ROOT / "docs" / "ood_rejection_census.json"

clf = tf.lite.Interpreter(model_path=str(MODEL_PATH))
clf.allocate_tensors()
inp = clf.get_input_details()[0]
out = clf.get_output_details()[0]

def pred(p):
    try:
        with Image.open(p) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.expand_dims(np.array(rgb, dtype=np.float32) / 255.0, axis=0)
            clf.set_tensor(inp["index"], arr)
            clf.invoke()
            return clf.get_tensor(out["index"])[0]
    except Exception:
        return None

def entropy(probs):
    eps = 1e-9
    clipped = np.clip(probs, eps, 1.0)
    return float(-np.sum(clipped * np.log2(clipped)))

def run():
    df = pd.read_csv(OOD_CSV)
    total = len(df)
    print(f"Loaded {total} OOD confounder images from {OOD_CSV}", flush=True)

    results_by_cond = {}
    total_evaluated = 0
    total_rejected = 0
    all_confs = []
    all_ents = []

    for idx, r in df.iterrows():
        p = pred(r["local_path"])
        if p is None:
            continue
        total_evaluated += 1
        cond = r["verified_label"]
        if cond not in results_by_cond:
            results_by_cond[cond] = {"total": 0, "rejected": 0, "accepted": 0, "mean_conf": [], "mean_ent": []}
        
        mp = float(np.max(p))
        ent = entropy(p)
        all_confs.append(mp)
        all_ents.append(ent)
        
        results_by_cond[cond]["total"] += 1
        results_by_cond[cond]["mean_conf"].append(mp)
        results_by_cond[cond]["mean_ent"].append(ent)
        
        # Calibrated DWH Rejection Gate
        if mp < 0.60 or ent > 1.40:
            total_rejected += 1
            results_by_cond[cond]["rejected"] += 1
        else:
            results_by_cond[cond]["accepted"] += 1
            
        if total_evaluated % 500 == 0:
            print(f"Evaluated {total_evaluated}/{total} images... (Current rejection: {total_rejected}/{total_evaluated} = {total_rejected/total_evaluated*100:.1f}%)", flush=True)

    overall_rate = total_rejected / total_evaluated * 100 if total_evaluated > 0 else 0.0
    print("\n" + "="*60, flush=True)
    print(f"FINAL FULL POPULATION OOD CENSUS (N={total_evaluated}):", flush=True)
    print(f"Overall Rejection Rate: {total_rejected}/{total_evaluated} ({overall_rate:.2f}%)", flush=True)
    print(f"Mean Confidence on Confounders: {np.mean(all_confs)*100:.2f}%", flush=True)
    print(f"Mean Predictive Entropy on Confounders: {np.mean(all_ents):.3f} bits", flush=True)
    print("="*60, flush=True)

    summary = {
        "evaluation_scope": "Full Population Census (Zero Subsampling)",
        "total_images_in_manifest": total,
        "total_evaluated_successfully": total_evaluated,
        "total_rejected": total_rejected,
        "total_accepted_as_target": total_evaluated - total_rejected,
        "overall_rejection_rate": float(overall_rate / 100.0),
        "mean_confidence": float(np.mean(all_confs)),
        "mean_entropy": float(np.mean(all_ents)),
        "per_condition_breakdown": {}
    }

    print("\nBreakdown by Non-Target Clinical Condition:", flush=True)
    for cond, d in sorted(results_by_cond.items(), key=lambda x: x[1]["total"], reverse=True):
        t = d["total"]
        rej = d["rejected"]
        rate = rej / t * 100
        m_conf = float(np.mean(d["mean_conf"]))
        m_ent = float(np.mean(d["mean_ent"]))
        print(f"  {cond:<35}: {rej:>4}/{t:>4} rejected ({rate:>5.1f}%) | Conf: {m_conf*100:>5.1f}% | Ent: {m_ent:.2f} bits", flush=True)
        summary["per_condition_breakdown"][cond] = {
            "total": t,
            "rejected": rej,
            "rejection_rate": float(rate / 100.0),
            "mean_confidence": m_conf,
            "mean_entropy": m_ent
        }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCensus results exported to: {OUTPUT_JSON}", flush=True)

if __name__ == "__main__":
    run()
