"""
DermTrust - Machine Learning Pipeline
Script: ml/reproduce_baseline_audit.py
Role: Reproduce and verify baseline metrics for both Keras and TFLite models
      on the validation split (N=293) and locked test split (N=294).
"""

import os
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf

CLASSES = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

KERAS_PATH = 'ml/models/dermtrust_mobilenetv2_5class.keras'
TFLITE_PATH = 'flutter_app/assets/models/dermtrust_classifier.tflite'
VAL_MANIFEST = 'ml/data/val_5class_manifest.csv'
TEST_MANIFEST = 'ml/data/test_5class_manifest.csv'

def compute_entropy_bits(probs):
    eps = 1e-9
    return -sum(max(eps, float(p)) * math.log2(max(eps, float(p))) for p in probs)

def calculate_ece(probs, labels_idx, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels_idx

    ece = 0.0
    total_samples = len(labels_idx)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)

def load_data(manifest_path):
    df = pd.read_csv(manifest_path)
    imgs = []
    y_true = []
    y_idx = []
    tones = []
    valid_rows = []
    
    for idx, row in df.iterrows():
        p = row['local_path']
        lbl = row['verified_label']
        st = str(row.get('skin_tone_group', 'Unknown'))
        if lbl in CLASS_TO_IDX and os.path.exists(p):
            with Image.open(p) as img:
                rgb = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
                arr = np.array(rgb, dtype=np.float32) / 255.0
                imgs.append(arr)
                y_true.append(lbl)
                y_idx.append(CLASS_TO_IDX[lbl])
                tones.append(st)
                valid_rows.append(row)
    return np.array(imgs, dtype=np.float32), y_true, np.array(y_idx, dtype=np.int32), tones, pd.DataFrame(valid_rows)

print(f"Loading Keras model: {KERAS_PATH}")
keras_model = tf.keras.models.load_model(KERAS_PATH, compile=False)

print(f"Loading TFLite model: {TFLITE_PATH}")
tflite_interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
tflite_interp.allocate_tensors()
input_idx = tflite_interp.get_input_details()[0]['index']
output_idx = tflite_interp.get_output_details()[0]['index']

def run_eval(split_name, manifest_path):
    print(f"\n==================== EVALUATING {split_name} ({manifest_path}) ====================")
    X, y_true, y_idx, tones, df = load_data(manifest_path)
    print(f"Total valid samples: {len(X)}")

    # 1. Keras inference
    keras_probs = keras_model.predict(X, batch_size=32, verbose=0)
    keras_preds_idx = np.argmax(keras_probs, axis=1)
    keras_preds = [CLASSES[i] for i in keras_preds_idx]
    keras_acc = accuracy_score(y_true, keras_preds)
    keras_rep = classification_report(y_true, keras_preds, target_names=CLASSES, digits=4, output_dict=True)
    keras_cm = confusion_matrix(y_true, keras_preds, labels=CLASSES)
    keras_ece = calculate_ece(keras_probs, y_idx)

    # 2. TFLite inference
    tflite_probs = []
    for i in range(len(X)):
        inp = np.expand_dims(X[i], axis=0)
        tflite_interp.set_tensor(input_idx, inp)
        tflite_interp.invoke()
        out = tflite_interp.get_tensor(output_idx)[0]
        tflite_probs.append(out)
    tflite_probs = np.array(tflite_probs, dtype=np.float32)
    tflite_preds_idx = np.argmax(tflite_probs, axis=1)
    tflite_preds = [CLASSES[i] for i in tflite_preds_idx]
    tflite_acc = accuracy_score(y_true, tflite_preds)
    tflite_rep = classification_report(y_true, tflite_preds, target_names=CLASSES, digits=4, output_dict=True)
    tflite_cm = confusion_matrix(y_true, tflite_preds, labels=CLASSES)
    tflite_ece = calculate_ece(tflite_probs, y_idx)

    max_delta = float(np.max(np.abs(keras_probs - tflite_probs)))

    # Rejection analysis on TFLite probabilities
    # tau_conf = 0.60, tau_entropy = 1.40
    rejected_count = 0
    accepted_correct = 0
    accepted_incorrect = 0
    for i in range(len(X)):
        conf = float(np.max(tflite_probs[i]))
        ent = compute_entropy_bits(tflite_probs[i])
        is_correct = (tflite_preds[i] == y_true[i])
        if conf < 0.60 or ent > 1.40:
            rejected_count += 1
        else:
            if is_correct:
                accepted_correct += 1
            else:
                accepted_incorrect += 1

    print(f"\n--- RESULTS FOR {split_name} ---")
    print(f"Keras Accuracy:     {keras_acc*100:.2f}% ({np.sum(keras_preds_idx == y_idx)}/{len(X)})")
    print(f"TFLite Accuracy:    {tflite_acc*100:.2f}% ({np.sum(tflite_preds_idx == y_idx)}/{len(X)})")
    print(f"Max |Keras - TFLite| Probability Diff: {max_delta:.8e}")
    print(f"TFLite Macro F1:    {tflite_rep['macro avg']['f1-score']:.4f}")
    print(f"TFLite Macro Rec:   {tflite_rep['macro avg']['recall']:.4f}")
    print(f"TFLite Macro Prec:  {tflite_rep['macro avg']['precision']:.4f}")
    print(f"TFLite ECE:         {tflite_ece*100:.2f}%")
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(f"{'':12s} " + " ".join([f"{c[:8]:>8s}" for c in CLASSES]))
    for idx_c, c in enumerate(CLASSES):
        print(f"{c:12s} " + " ".join([f"{tflite_cm[idx_c, j]:8d}" for j in range(len(CLASSES))]))
    
    print("\nPer-class Metrics:")
    for c in CLASSES:
        prec = tflite_rep[c]['precision']
        rec = tflite_rep[c]['recall']
        f1 = tflite_rep[c]['f1-score']
        supp = tflite_rep[c]['support']
        print(f"  {c:12s} | Prec: {prec*100:6.2f}% | Rec: {rec*100:6.2f}% | F1: {f1:.4f} | Supp: {supp}")

    print(f"\nRejection Gate (tau_conf=0.60, tau_ent=1.40):")
    print(f"  Total Samples:        {len(X)}")
    print(f"  Rejected (Outcome 6): {rejected_count} ({rejected_count/len(X)*100:.2f}%)")
    print(f"  Accepted:             {len(X) - rejected_count} ({(len(X)-rejected_count)/len(X)*100:.2f}%)")
    print(f"    - Accepted Correct:   {accepted_correct}")
    print(f"    - Accepted Incorrect: {accepted_incorrect}")
    if (len(X) - rejected_count) > 0:
        print(f"  Accepted Accuracy:    {accepted_correct / (len(X) - rejected_count) * 100:.2f}%")

    return {
        'split': split_name,
        'count': len(X),
        'keras_acc': keras_acc,
        'tflite_acc': tflite_acc,
        'macro_f1': tflite_rep['macro avg']['f1-score'],
        'macro_recall': tflite_rep['macro avg']['recall'],
        'macro_precision': tflite_rep['macro avg']['precision'],
        'ece': tflite_ece,
        'max_delta': max_delta,
        'confusion_matrix': tflite_cm.tolist(),
        'per_class': {c: tflite_rep[c] for c in CLASSES},
        'rejection': {
            'rejected': rejected_count,
            'accepted_correct': accepted_correct,
            'accepted_incorrect': accepted_incorrect
        }
    }

val_summary = run_eval('VALIDATION_SPLIT', VAL_MANIFEST)
test_summary = run_eval('LOCKED_TEST_SPLIT', TEST_MANIFEST)

with open('ml/models/baseline_audit_results.json', 'w') as f:
    json.dump({'val': val_summary, 'test': test_summary}, f, indent=2)
print("\nSaved baseline audit to ml/models/baseline_audit_results.json")
