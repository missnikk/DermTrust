"""
DermTrust - Threshold Tradeoff Analysis (Validation Target vs OOD Non-Target)
Evaluates False Rejection Rate on Target Validation vs False Acceptance Rate on OOD Non-Target.
"""

import os
import math
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

keras_path = 'ml/models/dermtrust_mobilenetv2_5class.keras'
model = tf.keras.models.load_model(keras_path, compile=False)
classes = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo']

def compute_entropy(probs):
    eps = 1e-9
    return -sum(max(eps, p) * math.log2(max(eps, p)) for p in probs)

# Load validation target
val_df = pd.read_csv('ml/data/val_5class_manifest.csv')
val_preds = []
for idx, row in val_df.iterrows():
    p = row['local_path']
    if not os.path.exists(p): 
        continue
    try:
        with Image.open(p) as img:
            rgb = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.expand_dims(np.array(rgb, dtype=np.float32) / 255.0, axis=0)
        probs = model(arr, training=False).numpy()[0]
        max_idx = int(np.argmax(probs))
        max_conf = float(probs[max_idx])
        ent = compute_entropy(probs)
        val_preds.append({
            'verified_label': row['verified_label'],
            'max_conf': max_conf,
            'entropy': ent,
            'pred_class': classes[max_idx],
            'is_correct': (classes[max_idx] == row['verified_label'])
        })
    except Exception:
        pass
val_res = pd.DataFrame(val_preds)

# Load OOD samples (100 per condition)
ood_df = pd.read_csv('ml/data/ood_5class_manifest.csv')
ood_preds = []
for cond in ood_df['verified_label'].unique():
    sub = ood_df[ood_df['verified_label'] == cond].head(100)
    for idx, row in sub.iterrows():
        p = row['local_path']
        if not os.path.exists(p): 
            continue
        try:
            with Image.open(p) as img:
                rgb = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
                arr = np.expand_dims(np.array(rgb, dtype=np.float32) / 255.0, axis=0)
            probs = model(arr, training=False).numpy()[0]
            max_idx = int(np.argmax(probs))
            ood_preds.append({
                'condition': cond,
                'max_conf': float(probs[max_idx]),
                'entropy': compute_entropy(probs),
                'pred_class': classes[max_idx]
            })
        except Exception:
            pass
ood_res = pd.DataFrame(ood_preds)

print(f"Loaded {len(val_res)} validation target samples and {len(ood_res)} OOD samples.")

sweep_results = []
for c_th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    for e_th in [1.20, 1.30, 1.40, 1.50]:
        t_rej = float(((val_res['max_conf'] < c_th) | (val_res['entropy'] > e_th)).mean())
        ood_acc = float(((ood_res['max_conf'] >= c_th) & (ood_res['entropy'] <= e_th)).mean())
        sweep_results.append({
            'conf_threshold': c_th,
            'entropy_threshold': e_th,
            'target_false_rejection_pct': t_rej * 100.0,
            'ood_false_acceptance_pct': ood_acc * 100.0
        })

print("\n--- CONFIDENCE & ENTROPY COMPARISON (Current Entropy = 1.40 bits) ---")
print(f"{'Conf Threshold':<15} | {'Target False Rejection':<24} | {'OOD False Acceptance':<22}")
for r in sweep_results:
    if r['entropy_threshold'] == 1.40:
        print(f"{r['conf_threshold']:<15.2f} | {r['target_false_rejection_pct']:<23.1f}% | {r['ood_false_acceptance_pct']:<21.1f}%")

# Per-class target false rejection at current baseline threshold (0.60 / 1.40)
per_class_rejections = {}
print("\n--- PER-CLASS FALSE REJECTION AT CURRENT BASELINE (Conf < 0.60 or Ent > 1.40) ---")
for c in classes:
    sub = val_res[val_res['verified_label'] == c]
    rej_rate = float(((sub['max_conf'] < 0.60) | (sub['entropy'] > 1.40)).mean())
    per_class_rejections[c] = {
        'total': len(sub),
        'rejected': int(round(rej_rate * len(sub))),
        'false_rejection_pct': rej_rate * 100.0
    }
    print(f"Class {c:<10} | N={len(sub):3d} | Falsely Rejected: {rej_rate*100:5.1f}% ({per_class_rejections[c]['rejected']}/{len(sub)})")

# Save outputs
output_data = {
    'validation_target_stats': {
        'count': len(val_res),
        'mean_conf': float(val_res['max_conf'].mean()),
        'median_conf': float(val_res['max_conf'].median()),
        'mean_entropy': float(val_res['entropy'].mean()),
        'median_entropy': float(val_res['entropy'].median())
    },
    'ood_stats': {
        'count': len(ood_res),
        'mean_conf': float(ood_res['max_conf'].mean()),
        'median_conf': float(ood_res['max_conf'].median()),
        'mean_entropy': float(ood_res['entropy'].mean()),
        'median_entropy': float(ood_res['entropy'].median())
    },
    'current_threshold_per_class': per_class_rejections,
    'sweep': sweep_results
}

with open('ml/threshold_tradeoff_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\nSaved tradeoff analysis to ml/threshold_tradeoff_results.json")
