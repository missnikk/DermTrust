"""
DermTrust - Evaluation of Selected Candidate Model (Exp 3: Balanced Sampling)
on the Strictly Held-Out Locked Test Set (test_5class_manifest.csv).
Also converts candidate to Float32 TFLite and verifies parity.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf

CLASSES = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CANDIDATE_KERAS = 'ml/models/experiments/model_exp3_balanced_sampling.keras'
TEST_MANIFEST = 'ml/data/test_5class_manifest.csv'

print(f"Loading candidate model: {CANDIDATE_KERAS}")
model = tf.keras.models.load_model(CANDIDATE_KERAS, compile=False)

test_df = pd.read_csv(TEST_MANIFEST)
print(f"Loaded locked test manifest: {len(test_df)} samples.")

imgs = []
y_true = []
tones = []
valid_rows = []

for idx, row in test_df.iterrows():
    p = row['local_path']
    lbl = row['verified_label']
    st = str(row.get('skin_tone_group', 'Unknown'))
    if lbl in CLASS_TO_IDX and os.path.exists(p):
        with Image.open(p) as img:
            rgb = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(rgb, dtype=np.float32) / 255.0
            imgs.append(arr)
            y_true.append(lbl)
            tones.append(st)
            valid_rows.append(row)

X_test = np.array(imgs, dtype=np.float32)
print(f"Processed {len(X_test)} test images.")

# Keras prediction
probs_keras = model.predict(X_test, batch_size=32, verbose=0)
preds_keras_idx = np.argmax(probs_keras, axis=1)
preds_keras = [CLASSES[i] for i in preds_keras_idx]

acc_keras = float(accuracy_score(y_true, preds_keras))
report_keras = classification_report(y_true, preds_keras, target_names=CLASSES, digits=4, output_dict=True)
cm_keras = confusion_matrix(y_true, preds_keras, labels=CLASSES).tolist()

print("\n" + "="*60)
print(f"CANDIDATE MODEL KERAS TEST ACCURACY: {acc_keras*100:.2f}%")
print("="*60)
print(classification_report(y_true, preds_keras, target_names=CLASSES, digits=4))

# Convert to Float32 TFLite
print("\nExporting candidate to Float32 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = []
tflite_bytes = converter.convert()

candidate_tflite_path = 'ml/models/experiments/candidate_classifier.tflite'
with open(candidate_tflite_path, 'wb') as f:
    f.write(tflite_bytes)
print(f"Candidate Float32 TFLite saved: {candidate_tflite_path} ({len(tflite_bytes)/(1024*1024):.2f} MB)")

# Verify Parity on Float32 TFLite
interp = tf.lite.Interpreter(model_path=candidate_tflite_path)
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']

probs_tflite_list = []
for i in range(len(X_test)):
    tensor = np.expand_dims(X_test[i], axis=0)
    interp.set_tensor(in_idx, tensor)
    interp.invoke()
    probs_tflite_list.append(interp.get_tensor(out_idx)[0])

probs_tflite = np.array(probs_tflite_list, dtype=np.float32)
preds_tflite_idx = np.argmax(probs_tflite, axis=1)
preds_tflite = [CLASSES[i] for i in preds_tflite_idx]

acc_tflite = float(accuracy_score(y_true, preds_tflite))
disagreements = sum(k != t for k, t in zip(preds_keras, preds_tflite))
max_diff = float(np.max(np.abs(probs_keras - probs_tflite)))
mean_diff = float(np.mean(np.abs(probs_keras - probs_tflite)))

print(f"\nParity (Keras vs Float32 TFLite): Agreement = {len(y_true)-disagreements}/{len(y_true)} ({((len(y_true)-disagreements)/len(y_true))*100:.2f}%)")
print(f"Max abs diff: {max_diff:.6e}, Mean abs diff: {mean_diff:.6e}")
print(f"TFLite Accuracy: {acc_tflite*100:.2f}%")

# Subgroup metrics
subgroup_results = {}
for st in ['Light', 'Darker', 'Unknown']:
    mask = [t == st for t in tones]
    if sum(mask) > 0:
        sub_y = [y_true[i] for i in range(len(y_true)) if mask[i]]
        sub_pred = [preds_tflite[i] for i in range(len(y_true)) if mask[i]]
        sub_acc = float(accuracy_score(sub_y, sub_pred))
        subgroup_results[st] = {
            'count': int(sum(mask)),
            'accuracy': sub_acc,
            'report': classification_report(sub_y, sub_pred, target_names=CLASSES, labels=CLASSES, output_dict=True, zero_division=0)
        }
        print(f"\n--- Subgroup {st} (N={sum(mask)}) ---")
        print(f"Accuracy: {sub_acc*100:.2f}%")
        for c in CLASSES:
            c_mask = [sub_y[i] == c for i in range(len(sub_y))]
            if sum(c_mask) > 0:
                c_pred_correct = sum(sub_pred[i] == c for i in range(len(sub_y)) if c_mask[i])
                print(f"  {c:<10}: {c_pred_correct}/{sum(c_mask)} ({c_pred_correct/sum(c_mask)*100:.1f}%)")

# Inspect Melasma test samples specifically
print("\n" + "="*60)
print("MELASMA TEST SAMPLES DETAIL (N=15):")
print("="*60)
mel_results = []
for i in range(len(y_true)):
    if y_true[i] == 'melasma':
        r = valid_rows[i]
        p_cls = preds_tflite[i]
        conf = float(probs_tflite[i][preds_tflite_idx[i]])
        mel_results.append({
            'image_id': r['image_id'],
            'pred_class': p_cls,
            'conf': conf,
            'is_augmented': bool(r['is_augmented']),
            'source': r['source_dataset']
        })
        print(f"{str(r['image_id'])[:38]:<40} | Pred: {p_cls:<10} ({conf*100:5.1f}%) | is_aug: {r['is_augmented']}")

output = {
    'keras_accuracy': acc_keras,
    'tflite_accuracy': acc_tflite,
    'parity_agreement': int(len(y_true) - disagreements),
    'max_diff': max_diff,
    'mean_diff': mean_diff,
    'classification_report': report_keras,
    'confusion_matrix': cm_keras,
    'subgroups': subgroup_results,
    'melasma_samples': mel_results
}

with open('ml/candidate_test_evaluation.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nResults saved to ml/candidate_test_evaluation.json")
