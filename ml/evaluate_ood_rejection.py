import os
import math
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

ood_df = pd.read_csv('ml/data/ood_5class_manifest.csv')
print(f"Total OOD manifest entries: {len(ood_df)}")

# Evaluate each non-target condition
conditions = ood_df['verified_label'].unique()

all_results = []

for cond in conditions:
    sub_df = ood_df[ood_df['verified_label'] == cond].head(100) # sample 100 per condition
    results = []
    for idx, row in sub_df.iterrows():
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
            pred_cls = classes[max_idx]
            rejected = (max_conf < 0.60) or (ent > 1.40)
            results.append({
                'condition': cond,
                'pred_class': pred_cls,
                'max_conf': max_conf,
                'entropy': ent,
                'rejected': rejected
            })
        except Exception:
            pass
    if results:
        res_df = pd.DataFrame(results)
        rej_rate = res_df['rejected'].mean() * 100.0
        accepted_df = res_df[~res_df['rejected']]
        top_false_pred = accepted_df['pred_class'].mode().iloc[0] if len(accepted_df) > 0 else "None"
        mean_conf_acc = accepted_df['max_conf'].mean() * 100.0 if len(accepted_df) > 0 else 0.0
        print(f"Condition: {cond:<32} | N={len(res_df):3d} | Rejected: {rej_rate:5.1f}% | False Accept: {100-rej_rate:5.1f}% | Top Misclass: {top_false_pred:<10} (Avg Conf: {mean_conf_acc:4.1f}%)")
        all_results.extend(results)

overall_df = pd.DataFrame(all_results)
print("\n=== OVERALL OOD REJECTION SUMMARY ===")
print(f"Total evaluated OOD samples: {len(overall_df)}")
print(f"Total rejected: {overall_df['rejected'].sum()} ({overall_df['rejected'].mean()*100:.1f}%)")
print(f"Total falsely accepted: {(~overall_df['rejected']).sum()} ({(1-overall_df['rejected'].mean())*100:.1f}%)")
print("\nDistribution of false predictions across falsely accepted OOD samples:")
print(overall_df[~overall_df['rejected']]['pred_class'].value_counts(normalize=True)*100)
