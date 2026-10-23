"""
DermTrust - Comprehensive Unseen Generalization & Image-Quality Stress Evaluation
Evaluates:
1. Independent Target Test Set (test_5class_manifest.csv, N=294)
2. Unseen Non-Target Clinical Lesions from Fitzpatrick 17k (Allergic Contact Dermatitis, Folliculitis, Melanoma)
3. Unseen OOD Non-Target Manifest (700 samples across 7 conditions)
4. DIP Image-Quality Rejection Suite (Blurry, Low-Res, Uniform Blank images)
"""

import os
import math
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import tensorflow as tf

CLASSES = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MODEL_PATH = 'ml/models/dermtrust_mobilenetv2_5class.keras'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def compute_entropy(probs):
    eps = 1e-9
    return -sum(max(eps, p) * math.log2(max(eps, p)) for p in probs)

def compute_laplacian_variance(pil_img):
    # Grayscale thumbnail 128x128
    thumb = pil_img.convert('L').resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.array(thumb, dtype=np.float32)
    lap = (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
        np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4 * arr
    )[1:-1, 1:-1]
    return float(np.var(lap))

# -------------------------------------------------------------
# 1. EVALUATE TARGET LOCKED TEST SET (N=294)
# -------------------------------------------------------------
print("="*60)
print("1. EVALUATING TARGET LOCKED TEST SET (N=294)")
print("="*60)
test_df = pd.read_csv('ml/data/test_5class_manifest.csv')
target_records = []

for idx, row in test_df.iterrows():
    p = row['local_path']
    if not os.path.exists(p): continue
    with Image.open(p) as img:
        rgb = img.convert('RGB')
        blur_var = compute_laplacian_variance(rgb)
        low_res = (rgb.width < 100 or rgb.height < 100)
        dip_valid = (not low_res) and (blur_var >= 30.0)
        
        # Preprocess
        resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)
        
    probs = model(arr, training=False).numpy()[0]
    max_idx = int(np.argmax(probs))
    max_conf = float(probs[max_idx])
    ent = compute_entropy(probs)
    pred_cls = CLASSES[max_idx]
    true_cls = row['verified_label']
    
    # Gate rejection
    dwh_rejected = (max_conf < 0.60) or (ent > 1.40)
    
    target_records.append({
        'image_id': row['image_id'],
        'true_class': true_cls,
        'pred_class': pred_cls,
        'max_conf': max_conf,
        'entropy': ent,
        'blur_var': blur_var,
        'dip_valid': dip_valid,
        'dwh_rejected': dwh_rejected,
        'is_correct': (pred_cls == true_cls),
        'skin_tone': str(row.get('skin_tone_group', 'Unknown'))
    })

df_tgt = pd.DataFrame(target_records)
print(f"Target Samples Evaluated: {len(df_tgt)}")
print(f"Raw Accuracy: {df_tgt['is_correct'].mean()*100:.2f}% ({df_tgt['is_correct'].sum()}/{len(df_tgt)})")
print(f"DIP Quality Pass Rate: {df_tgt['dip_valid'].mean()*100:.2f}%")
print(f"DWH Target Rejection Rate (Outcome 6): {df_tgt['dwh_rejected'].mean()*100:.2f}% ({df_tgt['dwh_rejected'].sum()}/{len(df_tgt)})")

# -------------------------------------------------------------
# 2. EVALUATE UNSEEN NON-TARGET CONDITIONS (Fitzpatrick 17k Unseen)
# -------------------------------------------------------------
print("\n" + "="*60)
print("2. EVALUATING UNSEEN NON-TARGET CLINICAL LESIONS (Fitzpatrick 17k)")
print("="*60)
fitz_csv = r'C:\Users\nikit\Downloads\fitzpatrick17k.csv'
fitz_img_dir = r'C:\Users\nikit\Downloads\fitzpatrick17k\data\finalfitz17k'
unified_df = pd.read_csv('ml/data/unified_ingested_manifest.csv')
used_md5 = set(unified_df['md5hash'])

df_fitz = pd.read_csv(fitz_csv)
df_unseen_fitz = df_fitz[~df_fitz['md5hash'].isin(used_md5)]

unseen_categories = ['allergic contact dermatitis', 'folliculitis', 'melanoma']
unseen_records = []

for u_cat in unseen_categories:
    sub = df_unseen_fitz[df_unseen_fitz['label'] == u_cat].head(100)
    for idx, row in sub.iterrows():
        md5 = row['md5hash']
        p = os.path.join(fitz_img_dir, f"{md5}.jpg")
        if not os.path.exists(p): continue
        try:
            with Image.open(p) as img:
                rgb = img.convert('RGB')
                blur_var = compute_laplacian_variance(rgb)
                resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
                arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)
            probs = model(arr, training=False).numpy()[0]
            max_idx = int(np.argmax(probs))
            max_conf = float(probs[max_idx])
            ent = compute_entropy(probs)
            pred_cls = CLASSES[max_idx]
            rejected = (max_conf < 0.60) or (ent > 1.40)
            unseen_records.append({
                'dataset': 'Fitzpatrick17k_Unseen',
                'condition': u_cat,
                'pred_class': pred_cls,
                'max_conf': max_conf,
                'entropy': ent,
                'blur_var': blur_var,
                'rejected': rejected
            })
        except Exception:
            pass

df_unseen_fitz_res = pd.DataFrame(unseen_records)
print(f"Evaluated {len(df_unseen_fitz_res)} unseen clinical lesions across {unseen_categories}:")
for u_cat in unseen_categories:
    sub = df_unseen_fitz_res[df_unseen_fitz_res['condition'] == u_cat]
    rej = sub['rejected'].mean() * 100
    top_pred = sub['pred_class'].mode().iloc[0] if len(sub) > 0 else 'None'
    print(f"  {u_cat:<30}: N={len(sub):3d} | Rejected: {rej:5.1f}% | False Accept: {100-rej:5.1f}% | Top False Pred: {top_pred}")

# -------------------------------------------------------------
# 3. EVALUATE OOD MANIFEST (N=700)
# -------------------------------------------------------------
print("\n" + "="*60)
print("3. EVALUATING OOD MANIFEST NON-TARGETS (N=700)")
print("="*60)
ood_df = pd.read_csv('ml/data/ood_5class_manifest.csv')
ood_records = []
for cond in ood_df['verified_label'].unique():
    sub = ood_df[ood_df['verified_label'] == cond].head(100)
    for idx, row in sub.iterrows():
        p = row['local_path']
        if not os.path.exists(p): continue
        try:
            with Image.open(p) as img:
                rgb = img.convert('RGB')
                resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
                arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)
            probs = model(arr, training=False).numpy()[0]
            max_idx = int(np.argmax(probs))
            max_conf = float(probs[max_idx])
            ent = compute_entropy(probs)
            pred_cls = CLASSES[max_idx]
            rejected = (max_conf < 0.60) or (ent > 1.40)
            ood_records.append({
                'dataset': 'OOD_Manifest',
                'condition': cond,
                'pred_class': pred_cls,
                'max_conf': max_conf,
                'entropy': ent,
                'rejected': rejected
            })
        except Exception:
            pass

df_ood_res = pd.DataFrame(ood_records)
print(f"Evaluated {len(df_ood_res)} OOD manifest samples:")
print(f"Total Rejection Rate: {df_ood_res['rejected'].mean()*100:.1f}%")
print(f"Total False Acceptance Rate: {(1-df_ood_res['rejected'].mean())*100:.1f}%")

# -------------------------------------------------------------
# 4. EVALUATE IMAGE-QUALITY STRESS TESTS (DIP REJECTION TIER 1)
# -------------------------------------------------------------
print("\n" + "="*60)
print("4. EVALUATING IMAGE-QUALITY REJECTION TIER (BLUR, LOW-RES, BLANK)")
print("="*60)
stress_records = []

# A. 100 Blurry images (Gaussian blur radius=8 applied to test images)
for idx, row in test_df.head(100).iterrows():
    p = row['local_path']
    if not os.path.exists(p): continue
    with Image.open(p) as img:
        blurred = img.convert('RGB').filter(ImageFilter.GaussianBlur(radius=8))
        blur_var = compute_laplacian_variance(blurred)
        low_res = (blurred.width < 100 or blurred.height < 100)
        rejected = (blur_var < 30.0) or low_res
        stress_records.append({
            'test_type': 'Blurry_Images (Gaussian radius=8)',
            'blur_var': blur_var,
            'is_rejected_by_dip': rejected
        })

# B. 50 Low-Resolution images (downscaled to 64x64)
for idx, row in test_df.head(50).iterrows():
    p = row['local_path']
    if not os.path.exists(p): continue
    with Image.open(p) as img:
        low_res_img = img.convert('RGB').resize((64, 64), Image.Resampling.BILINEAR)
        is_low_res = (low_res_img.width < 100 or low_res_img.height < 100)
        stress_records.append({
            'test_type': 'Low_Resolution_Images (<100px)',
            'blur_var': 0.0,
            'is_rejected_by_dip': is_low_res
        })

# C. 50 Uniform Blank / Flat Images (gray, white, black, solid color)
for val in [0, 64, 128, 192, 255]:
    for _ in range(10):
        blank = Image.new('RGB', (224, 224), color=(val, val, val))
        blur_var = compute_laplacian_variance(blank)
        stress_records.append({
            'test_type': 'Uniform_Blank_Images (Flat Patch)',
            'blur_var': blur_var,
            'is_rejected_by_dip': (blur_var < 30.0)
        })

df_stress = pd.DataFrame(stress_records)
print(f"Evaluated {len(df_stress)} Image-Quality Stress Samples:")
for t_type in df_stress['test_type'].unique():
    sub = df_stress[df_stress['test_type'] == t_type]
    rej_rate = sub['is_rejected_by_dip'].mean() * 100.0
    print(f"  {t_type:<36}: N={len(sub):3d} | DIP Rejection Rate: {rej_rate:5.1f}% (Expected: 100.0%)")

# Combine and save all results
out_data = {
    'target_locked_test': {
        'total': len(df_tgt),
        'raw_accuracy': float(df_tgt['is_correct'].mean()),
        'dwh_rejected_pct': float(df_tgt['dwh_rejected'].mean() * 100),
        'per_class': {}
    },
    'unseen_fitzpatrick17k_nontarget': {
        'total': len(df_unseen_fitz_res),
        'rejection_pct': float(df_unseen_fitz_res['rejected'].mean() * 100),
        'by_condition': {}
    },
    'ood_manifest_nontarget': {
        'total': len(df_ood_res),
        'rejection_pct': float(df_ood_res['rejected'].mean() * 100),
        'false_acceptance_pct': float((1 - df_ood_res['rejected'].mean()) * 100)
    },
    'image_quality_stress_tests': {
        'total': len(df_stress),
        'rejection_by_type': {}
    }
}

for c in CLASSES:
    sub = df_tgt[df_tgt['true_class'] == c]
    out_data['target_locked_test']['per_class'][c] = {
        'count': len(sub),
        'raw_recall': float(sub['is_correct'].mean()),
        'gate_rejected_pct': float(sub['dwh_rejected'].mean() * 100)
    }

for u_cat in unseen_categories:
    sub = df_unseen_fitz_res[df_unseen_fitz_res['condition'] == u_cat]
    out_data['unseen_fitzpatrick17k_nontarget']['by_condition'][u_cat] = {
        'count': len(sub),
        'rejected_pct': float(sub['rejected'].mean() * 100),
        'false_acceptance_pct': float((1 - sub['rejected'].mean()) * 100)
    }

for t_type in df_stress['test_type'].unique():
    sub = df_stress[df_stress['test_type'] == t_type]
    out_data['image_quality_stress_tests']['rejection_by_type'][t_type] = {
        'count': len(sub),
        'rejection_pct': float(sub['is_rejected_by_dip'].mean() * 100)
    }

with open('ml/unseen_generalization_results.json', 'w') as f:
    json.dump(out_data, f, indent=2)

print("\nSaved comprehensive results to ml/unseen_generalization_results.json")
