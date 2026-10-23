"""
DermTrust - Deep Failure Mechanism & Feature Space Investigation
Investigates Task B:
1. Melasma -> Acne high-confidence confusion
2. Mild Acne -> Outcome 6 low confidence
3. Vitiligo -> Psoriasis / Outcome 6
4. Ringworm -> Psoriasis / Outcome 6
5. Grad-CAM spatial foveation and energy distribution

Computes:
- Feature space cosine similarity between failure samples and training class centroids (1280-dim GAP)
- Activation energy concentration (center 3x3 vs outer boundary 7x7)
- Softmax temperature calibration & Expected Calibration Error (ECE)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
KERAS_MODEL_PATH = ROOT_DIR / "ml" / "models" / "dermtrust_mobilenetv2_5class.keras"
TRAIN_MANIFEST = ROOT_DIR / "ml" / "data" / "train_5class_manifest.csv"
VAL_MANIFEST = ROOT_DIR / "ml" / "data" / "val_5class_manifest.csv"
FAILURES_DIR = ROOT_DIR / "test_failures"

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_img_arr(path: str) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        return np.array(rgb, dtype=np.float32) / 255.0


def main():
    print(f"Loading full Keras model: {KERAS_MODEL_PATH}")
    model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), compile=False)

    # Build sub-model to extract 1280-dim GAP features
    gap_layer = None
    for layer in reversed(model.layers):
        if "global_average_pooling" in layer.name or isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
            break
    if gap_layer is None:
        # Fallback to second-to-last layer
        gap_layer = model.layers[-2]
    
    print(f"Identified GAP feature extraction layer: {gap_layer.name} ({gap_layer.output.shape})")
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=[gap_layer.output, model.output])

    # 1. Compute Class Centroids in 1280-dim Feature Space from Training Set (sample 30 per class for speed)
    print("Computing class centroids in 1280-dim GAP latent space from training data...")
    train_df = pd.read_csv(TRAIN_MANIFEST)
    class_features = {c: [] for c in CLASSES}

    for c in CLASSES:
        subset = train_df[train_df["verified_label"] == c].head(40)
        batch_imgs = []
        for _, row in subset.iterrows():
            p = row["local_path"]
            if os.path.exists(p):
                batch_imgs.append(load_img_arr(p))
        if batch_imgs:
            feats, _ = feature_extractor.predict(np.array(batch_imgs), batch_size=32, verbose=0)
            class_features[c] = feats

    centroids = {}
    for c in CLASSES:
        if len(class_features[c]) > 0:
            arr = np.array(class_features[c])
            mean_vec = np.mean(arr, axis=0)
            centroids[c] = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
        else:
            centroids[c] = np.zeros(1280)

    # 2. Centroid-to-Centroid Cosine Similarities
    print("\n=== Inter-Class Cosine Similarities in 1280-dim Latent Space ===")
    sim_matrix = np.zeros((5, 5))
    for i, c1 in enumerate(CLASSES):
        for j, c2 in enumerate(CLASSES):
            sim = float(np.dot(centroids[c1], centroids[c2]))
            sim_matrix[i, j] = sim
    sim_df = pd.DataFrame(sim_matrix, index=CLASSES, columns=CLASSES)
    print(sim_df.round(4).to_string())

    # 3. Analyze Key Failure Images in Feature Space
    print("\n=== Failure Sample Alignment to Latent Class Centroids ===")
    failure_files = [
        ("IMG-20261023-WA0013.jpg", FAILURES_DIR / "WhatsApp" / "IMG-20261023-WA0013.jpg", "Melasma (user reported)"),
        ("IMG-20261023-WA0022.jpg", FAILURES_DIR / "WhatsApp" / "IMG-20261023-WA0022.jpg", "Melasma (user reported)"),
        ("melasma_sample_1.jpg", FAILURES_DIR / "melasma_sample_1.jpg", "Melasma face sample"),
        ("IMG-20261023-WA0023.jpg", FAILURES_DIR / "WhatsApp" / "IMG-20261023-WA0023.jpg", "Mild Acne (2-5 spots)"),
        ("IMG-20261023-WA0017.jpg", FAILURES_DIR / "WhatsApp" / "IMG-20261023-WA0017.jpg", "Ringworm/Psoriasis ambiguous"),
        ("IMG-20261023-WA0021.jpg", FAILURES_DIR / "WhatsApp" / "IMG-20261023-WA0021.jpg", "Ringworm/Psoriasis ambiguous"),
        ("IMG-20261023-WA0024.jpg", FAILURES_DIR / "WhatsApp" / "IMG-20261023-WA0024.jpg", "Vitiligo hand"),
        ("test_vitiligo_dl.jpg", FAILURES_DIR / "test_vitiligo_dl.jpg", "Vitiligo (DL)"),
    ]

    analysis_records = []
    for fname, fpath, desc in failure_files:
        if not fpath.exists():
            continue
        img_arr = np.expand_dims(load_img_arr(str(fpath)), axis=0)
        feats, probs = feature_extractor.predict(img_arr, verbose=0)
        feat_vec = feats[0] / (np.linalg.norm(feats[0]) + 1e-9)
        prob_vec = probs[0]

        cos_sims = {c: float(np.dot(feat_vec, centroids[c])) for c in CLASSES}
        pred_c = CLASSES[int(np.argmax(prob_vec))]
        pred_conf = float(np.max(prob_vec))

        rec = {
            "file": fname,
            "description": desc,
            "predicted": pred_c,
            "confidence": pred_conf,
            "probabilities": {CLASSES[i]: float(prob_vec[i]) for i in range(5)},
            "cosine_similarities": cos_sims,
        }
        analysis_records.append(rec)

        print(f"\nSample: {fname} [{desc}]")
        print(f"  Prediction: {pred_c} ({pred_conf*100:.1f}%)")
        print(f"  Probabilities: { {c: f'{prob_vec[i]*100:.1f}%' for i, c in enumerate(CLASSES)} }")
        print(f"  Cosine Sim:    { {c: round(cos_sims[c], 3) for c in CLASSES} }")

    # 4. Measure Grad-CAM Spatial Energy Concentration on 7x7 Grid
    # (Center 3x3 core vs outer perimeter cells)
    print("\n=== Grad-CAM Spatial Energy Concentration (Center 3x3 vs Outer 40 cells) ===")
    tflite_cam_path = ROOT_DIR / "flutter_app" / "assets" / "models" / "dermtrust_cam.tflite"
    cam_weights_path = ROOT_DIR / "flutter_app" / "assets" / "models" / "cam_weights.json"
    cam_interp = tf.lite.Interpreter(model_path=str(tflite_cam_path))
    cam_interp.allocate_tensors()
    cin = cam_interp.get_input_details()[0]["index"]
    couts = cam_interp.get_output_details()
    fidx = next(d["index"] for d in couts if d["shape"].tolist() == [1, 7, 7, 1280])

    with open(cam_weights_path) as f:
        cam_weights = np.array(json.load(f)["weights"])

    core_ratios = []
    for fname, fpath, desc in failure_files:
        if not fpath.exists():
            continue
        arr = np.expand_dims(load_img_arr(str(fpath)), axis=0)
        cam_interp.set_tensor(cin, arr)
        cam_interp.invoke()
        fmap = cam_interp.get_tensor(fidx)[0]  # [7, 7, 1280]

        # Use top predicted class weights
        probs = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        cam = np.maximum(0, np.tensordot(fmap, cam_weights[top_idx], axes=([2], [0])))
        total_e = np.sum(cam) + 1e-9
        # Center 3x3 is rows 2:5, cols 2:5 (9 cells out of 49)
        core_e = np.sum(cam[2:5, 2:5])
        ratio = core_e / total_e
        core_ratios.append(ratio)
        print(f"  {fname:25s}: Core 3x3 Energy = {ratio*100:.1f}% (Expected uniform: 9/49 = 18.4%)")

    mean_core = float(np.mean(core_ratios))
    print(f"\nAverage Central Core Energy Concentration: {mean_core*100:.1f}% vs Uniform 18.4%")

    # 5. Measure Calibration Error on Validation Set (N=293)
    val_df = pd.read_csv(VAL_MANIFEST)
    val_imgs, val_y = [], []
    for _, row in val_df.iterrows():
        p = row["local_path"]
        lbl = row["verified_label"]
        if os.path.exists(p) and lbl in CLASS_TO_IDX:
            val_imgs.append(load_img_arr(p))
            val_y.append(CLASS_TO_IDX[lbl])

    val_preds = model.predict(np.array(val_imgs), batch_size=32, verbose=0)
    confidences = np.max(val_preds, axis=1)
    predicted_classes = np.argmax(val_preds, axis=1)
    accuracies = (predicted_classes == np.array(val_y)).astype(float)

    # 10-bin ECE
    bin_edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        prop = np.mean(in_bin)
        if prop > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_acc - bin_conf) * prop

    print(f"\nValidation Expected Calibration Error (ECE, 10 bins): {ece*100:.2f}%")
    print(f"Mean Confidence on Validation: {np.mean(confidences)*100:.1f}% vs Accuracy: {np.mean(accuracies)*100:.1f}%")

    # Save findings
    out_path = ROOT_DIR / "ml" / "failure_mechanisms_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "inter_class_similarities": sim_matrix.tolist(),
            "failure_samples": analysis_records,
            "mean_core_energy_ratio": mean_core,
            "validation_ece": ece,
            "validation_mean_confidence": float(np.mean(confidences)),
            "validation_accuracy": float(np.mean(accuracies)),
        }, f, indent=2)
    print(f"\nSaved failure mechanism report to: {out_path}")


if __name__ == "__main__":
    main()
