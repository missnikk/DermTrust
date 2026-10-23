import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

classes = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo']
class_to_idx = {c: i for i, c in enumerate(classes)}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

keras_path = 'ml/models/dermtrust_mobilenetv2_5class.keras'
tflite_path = 'flutter_app/assets/models/dermtrust_classifier.tflite'
backup_tflite_path = 'model_artifacts/backup_v1_quantized/dermtrust_classifier.tflite'

log(f"Loading Keras model: {keras_path}")
keras_model = tf.keras.models.load_model(keras_path, compile=False)

log(f"Loading Float32 TFLite model: {tflite_path}")
tflite_interp = tf.lite.Interpreter(model_path=tflite_path)
tflite_interp.allocate_tensors()
tflite_in = tflite_interp.get_input_details()[0]['index']
tflite_out = tflite_interp.get_output_details()[0]['index']

has_backup = os.path.exists(backup_tflite_path)
if has_backup:
    log(f"Loading Backup Quantized TFLite model: {backup_tflite_path}")
    backup_interp = tf.lite.Interpreter(model_path=backup_tflite_path)
    backup_interp.allocate_tensors()
    backup_in = backup_interp.get_input_details()[0]['index']
    backup_out = backup_interp.get_output_details()[0]['index']

all_manifest_results = {}

for manifest_name in ['ml/data/test_5class_manifest.csv', 'ml/data/test_5class_manifest_v2.csv']:
    log(f"=== EVALUATING MANIFEST: {manifest_name} ===")
    df = pd.read_csv(manifest_name)
    
    y_true = []
    valid_paths = []
    
    for _, row in df.iterrows():
        p = row['local_path']
        lbl = row['verified_label']
        if lbl in class_to_idx and os.path.exists(p):
            y_true.append(lbl)
            valid_paths.append(p)
            
    log(f"Loaded manifest with {len(y_true)} verified test images on disk.")
    counts = pd.Series(y_true).value_counts()
    for cls in classes:
        log(f"  {cls:<10}: {counts.get(cls, 0)}")
        
    log("Loading and preprocessing images into memory...")
    imgs = []
    for p in valid_paths:
        with Image.open(p) as img:
            rgb = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(rgb, dtype=np.float32) / 255.0
            imgs.append(arr)
    
    log(f"Loaded {len(imgs)} images. Running Keras predictions in batches...")
    batch_size = 32
    keras_probs_list = []
    for b_start in range(0, len(imgs), batch_size):
        b_imgs = np.array(imgs[b_start:b_start+batch_size], dtype=np.float32)
        probs = keras_model(b_imgs, training=False).numpy()
        keras_probs_list.append(probs)
    keras_probs = np.vstack(keras_probs_list)
    keras_preds = [classes[i] for i in np.argmax(keras_probs, axis=1)]
    log("Keras predictions complete.")
    
    log("Running Float32 TFLite predictions...")
    tflite_probs_list = []
    for i, img_arr in enumerate(imgs):
        tensor = np.expand_dims(img_arr, axis=0)
        tflite_interp.set_tensor(tflite_in, tensor)
        tflite_interp.invoke()
        tp = tflite_interp.get_tensor(tflite_out)[0]
        tflite_probs_list.append(tp)
        if (i + 1) % 100 == 0 or i == len(imgs) - 1:
            log(f"  Float32 TFLite progress: {i+1}/{len(imgs)}")
    tflite_probs = np.array(tflite_probs_list, dtype=np.float32)
    tflite_preds = [classes[i] for i in np.argmax(tflite_probs, axis=1)]
    log("Float32 TFLite predictions complete.")
    
    backup_preds = []
    backup_probs = None
    if has_backup:
        log("Running Backup Quantized TFLite predictions...")
        b_list = []
        for i, img_arr in enumerate(imgs):
            tensor = np.expand_dims(img_arr, axis=0)
            backup_interp.set_tensor(backup_in, tensor)
            backup_interp.invoke()
            bp = backup_interp.get_tensor(backup_out)[0]
            b_list.append(bp)
            if (i + 1) % 100 == 0 or i == len(imgs) - 1:
                log(f"  Backup TFLite progress: {i+1}/{len(imgs)}")
        backup_probs = np.array(b_list, dtype=np.float32)
        backup_preds = [classes[i] for i in np.argmax(backup_probs, axis=1)]
        log("Backup Quantized TFLite predictions complete.")
        
    k_acc = accuracy_score(y_true, keras_preds)
    t_acc = accuracy_score(y_true, tflite_preds)
    
    log(f"KERAS ACCURACY: {k_acc*100:.2f}%")
    log(f"TFLITE ACCURACY: {t_acc*100:.2f}%")
    
    k_report = classification_report(y_true, keras_preds, target_names=classes, digits=4, output_dict=True)
    t_report = classification_report(y_true, tflite_preds, target_names=classes, digits=4, output_dict=True)
    
    cm_k = confusion_matrix(y_true, keras_preds, labels=classes)
    cm_t = confusion_matrix(y_true, tflite_preds, labels=classes)
    
    log("\n--- FLOAT32 TFLITE CLASSIFICATION REPORT ---")
    log(classification_report(y_true, tflite_preds, target_names=classes, digits=4))
    log("\n--- CONFUSION MATRIX (rows=true, cols=pred) ---")
    log("\n" + str(pd.DataFrame(cm_t, index=classes, columns=classes)))
    
    # Parity
    disagreements = sum(k != t for k, t in zip(keras_preds, tflite_preds))
    mean_prob_diff = float(np.mean(np.abs(keras_probs - tflite_probs)))
    max_prob_diff = float(np.max(np.abs(keras_probs - tflite_probs)))
    log(f"Parity (Keras vs Float32 TFLite): Class agreement = {len(y_true) - disagreements}/{len(y_true)} ({((len(y_true)-disagreements)/len(y_true))*100:.2f}%)")
    log(f"Mean prob diff: {mean_prob_diff:.6e}, Max prob diff: {max_prob_diff:.6e}")
    
    if has_backup:
        disagreements_b = sum(k != b for k, b in zip(keras_preds, backup_preds))
        mean_b_diff = float(np.mean(np.abs(keras_probs - backup_probs)))
        max_b_diff = float(np.max(np.abs(keras_probs - backup_probs)))
        log(f"Parity (Keras vs Backup Quantized TFLite): Class agreement = {len(y_true) - disagreements_b}/{len(y_true)} ({((len(y_true)-disagreements_b)/len(y_true))*100:.2f}%)")
        log(f"Mean prob diff: {mean_b_diff:.6e}, Max prob diff: {max_b_diff:.6e}")
        
    # Distribution of predictions:
    pred_counts = pd.Series(tflite_preds).value_counts().to_dict()
    log(f"Predicted Class Counts: {pred_counts}")
    
    # Confidence distribution:
    correct_confs = [float(tflite_probs[i, np.argmax(tflite_probs[i])]) for i in range(len(y_true)) if tflite_preds[i] == y_true[i]]
    incorrect_confs = [float(tflite_probs[i, np.argmax(tflite_probs[i])]) for i in range(len(y_true)) if tflite_preds[i] != y_true[i]]
    log(f"Confidence (Correct N={len(correct_confs)}): Mean={np.mean(correct_confs)*100:.2f}%, Median={np.median(correct_confs)*100:.2f}%")
    if incorrect_confs:
        log(f"Confidence (Incorrect N={len(incorrect_confs)}): Mean={np.mean(incorrect_confs)*100:.2f}%, Median={np.median(incorrect_confs)*100:.2f}%")

    manifest_base = os.path.basename(manifest_name)
    all_manifest_results[manifest_base] = {
        'sample_count': len(y_true),
        'class_distribution': counts.to_dict(),
        'keras_accuracy': float(k_acc),
        'tflite_accuracy': float(t_acc),
        'keras_report': k_report,
        'tflite_report': t_report,
        'confusion_matrix': cm_t.tolist(),
        'parity_agreement': int(len(y_true) - disagreements),
        'mean_prob_diff': mean_prob_diff,
        'max_prob_diff': max_prob_diff,
        'predicted_counts': pred_counts,
    }

with open('ml/evaluation_results.json', 'w') as f:
    json.dump(all_manifest_results, f, indent=2)

log("Evaluation complete. Results saved to ml/evaluation_results.json")
