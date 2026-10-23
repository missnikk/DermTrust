"""
DermTrust - Failure Image Diagnostic Harness
Audits the 14 user real images and other test images:
1. Runs both Keras and packaged TFLite models to verify numerical parity.
2. Evaluates DIP Laplacian variance blur score.
3. Computes exact probability vectors, top-2 class margin, Shannon entropy.
4. Checks DWH rejection gate outcome (conf >= 0.60, entropy <= 1.40).
5. Computes Grad-CAM 7x7 heatmap, peak activation (y_max, x_max), activation centroid (cy, cx),
   and distance from center to quantify foveation/center-bias.
6. Saves detailed audit report and visual overlay comparisons.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
KERAS_MODEL_PATH = ROOT_DIR / "ml" / "models" / "dermtrust_mobilenetv2_5class.keras"
TFLITE_CLF_PATH = ROOT_DIR / "flutter_app" / "assets" / "models" / "dermtrust_classifier.tflite"
TFLITE_CAM_PATH = ROOT_DIR / "flutter_app" / "assets" / "models" / "dermtrust_cam.tflite"
CAM_WEIGHTS_PATH = ROOT_DIR / "flutter_app" / "assets" / "models" / "cam_weights.json"
FAILURES_DIR = ROOT_DIR / "test_failures"
OUTPUT_DIR = ROOT_DIR / "docs" / "failure_diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
CONF_THRESHOLD = 0.60
ENTROPY_THRESHOLD = 1.40
BLUR_THRESHOLD = 30.0


def compute_laplacian_variance(img_pil: Image.Image) -> float:
    # Resize to 128x128 grayscale to match Flutter ImageProcessor
    thumb = img_pil.resize((128, 128), Image.Resampling.BILINEAR).convert("L")
    arr = np.array(thumb, dtype=np.float32)
    lap = (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
        np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4.0 * arr
    )[1:-1, 1:-1]
    return float(np.var(lap))


def compute_entropy(probs: np.ndarray) -> float:
    eps = 1e-9
    clipped = np.clip(probs, eps, 1.0)
    return float(-np.sum(clipped * np.log2(clipped)))


def main():
    print(f"Loading Keras model: {KERAS_MODEL_PATH}")
    keras_model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), compile=False)

    print(f"Loading TFLite Classifier: {TFLITE_CLF_PATH}")
    clf_interp = tf.lite.Interpreter(model_path=str(TFLITE_CLF_PATH))
    clf_interp.allocate_tensors()
    clf_in = clf_interp.get_input_details()[0]
    clf_out = clf_interp.get_output_details()[0]

    print(f"Loading TFLite CAM model: {TFLITE_CAM_PATH}")
    cam_interp = tf.lite.Interpreter(model_path=str(TFLITE_CAM_PATH))
    cam_interp.allocate_tensors()
    cam_in = cam_interp.get_input_details()[0]
    cam_outs = cam_interp.get_output_details()
    feat_idx = next(d["index"] for d in cam_outs if d["shape"].tolist() == [1, 7, 7, 1280])
    prob_idx = next(d["index"] for d in cam_outs if d["shape"].tolist() == [1, 5])

    with open(CAM_WEIGHTS_PATH, "r") as f:
        cam_weights_data = json.load(f)
    cam_weights = np.array(cam_weights_data["weights"])  # [5, 1280]

    # Collect images
    target_files = []
    wa_dir = FAILURES_DIR / "WhatsApp"
    if wa_dir.exists():
        for p in sorted(wa_dir.glob("IMG-20261023-WA*.jpg")):
            target_files.append(p)
    for p in sorted(FAILURES_DIR.glob("*.*")):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            target_files.append(p)

    print(f"Found {len(target_files)} failure/test images to audit.")

    audit_records = []

    for img_path in target_files:
        try:
            with Image.open(img_path) as raw_img:
                raw_rgb = raw_img.convert("RGB")
                orig_w, orig_h = raw_rgb.size
                blur_score = compute_laplacian_variance(raw_rgb)

                resized = raw_rgb.resize((224, 224), Image.Resampling.BILINEAR)
                input_arr = np.array(resized, dtype=np.float32) / 255.0
                input_tensor = np.expand_dims(input_arr, axis=0)

            # Keras inference
            k_probs = keras_model.predict(input_tensor, verbose=0)[0]

            # TFLite inference
            clf_interp.set_tensor(clf_in["index"], input_tensor)
            clf_interp.invoke()
            t_probs = clf_interp.get_tensor(clf_out["index"])[0]

            parity_max_diff = float(np.max(np.abs(k_probs - t_probs)))

            # CAM inference
            cam_interp.set_tensor(cam_in["index"], input_tensor)
            cam_interp.invoke()
            feats = cam_interp.get_tensor(feat_idx)[0]  # [7, 7, 1280]

            top_idx = int(np.argmax(t_probs))
            top_class = CLASSES[top_idx]
            top_conf = float(t_probs[top_idx])
            entropy = compute_entropy(t_probs)

            # Sorted probabilities
            sorted_indices = np.argsort(t_probs)[::-1]
            top2_idx = sorted_indices[1]
            top2_class = CLASSES[top2_idx]
            top2_conf = float(t_probs[top2_idx])
            margin = top_conf - top2_conf

            # Rejection Gate
            if blur_score < BLUR_THRESHOLD:
                gate_outcome = "REJECTED_BLUR"
                gate_reason = f"DIP Blur rejection: Laplacian variance {blur_score:.1f} < {BLUR_THRESHOLD}"
            elif top_conf < CONF_THRESHOLD:
                gate_outcome = "REJECTED_LOW_CONFIDENCE"
                gate_reason = f"Low confidence: {top_conf*100:.1f}% < {CONF_THRESHOLD*100:.0f}%"
            elif entropy > ENTROPY_THRESHOLD:
                gate_outcome = "REJECTED_HIGH_ENTROPY"
                gate_reason = f"High predictive entropy: {entropy:.2f} > {ENTROPY_THRESHOLD:.2f}"
            else:
                gate_outcome = "ACCEPTED"
                gate_reason = f"Accepted: {top_class} ({top_conf*100:.1f}%)"

            # Grad-CAM heatmap for top predicted class
            weights_c = cam_weights[top_idx]
            cam_raw = np.maximum(0, np.tensordot(feats, weights_c, axes=([2], [0])))  # [7, 7]
            cam_min, cam_max = float(cam_raw.min()), float(cam_raw.max())
            cam_range = (cam_max - cam_min) if (cam_max - cam_min) > 1e-6 else 1.0
            cam_norm = (cam_raw - cam_min) / cam_range

            # Quantitative spatial metrics
            y_max, x_max = np.unravel_index(np.argmax(cam_norm), cam_norm.shape)
            # Centroid
            total_mass = float(np.sum(cam_norm))
            if total_mass > 1e-6:
                y_coords, x_coords = np.indices(cam_norm.shape)
                cy = float(np.sum(y_coords * cam_norm) / total_mass)
                cx = float(np.sum(x_coords * cam_norm) / total_mass)
            else:
                cy, cx = 3.0, 3.0
            # Center of 7x7 grid is (3.0, 3.0)
            dist_from_center = float(np.sqrt((cy - 3.0)**2 + (cx - 3.0)**2))

            prob_dict = {CLASSES[i]: float(t_probs[i]) for i in range(len(CLASSES))}

            record = {
                "filename": img_path.name,
                "relative_path": str(img_path.relative_to(ROOT_DIR)),
                "dimensions": [orig_w, orig_h],
                "blur_score": blur_score,
                "keras_tflite_max_diff": parity_max_diff,
                "top_class": top_class,
                "top_confidence": top_conf,
                "top2_class": top2_class,
                "top2_confidence": top2_conf,
                "margin": margin,
                "entropy": entropy,
                "gate_outcome": gate_outcome,
                "gate_reason": gate_reason,
                "probabilities": prob_dict,
                "cam_peak": [int(y_max), int(x_max)],
                "cam_centroid": [round(cy, 2), round(cx, 2)],
                "cam_dist_from_center": round(dist_from_center, 2),
            }
            audit_records.append(record)

            # Generate visualization overlay
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(resized)
            axes[0].set_title(f"Input: {img_path.name[:20]}\n({orig_w}x{orig_h}, blur={blur_score:.1f})")
            axes[0].axis("off")

            im_cam = axes[1].imshow(cam_norm, cmap="jet", interpolation="bilinear")
            axes[1].plot(cx, cy, "w+", markersize=14, markeredgewidth=2, label=f"Centroid ({cx:.1f}, {cy:.1f})")
            axes[1].plot(x_max, y_max, "ro", markersize=8, label=f"Peak ({x_max}, {y_max})")
            axes[1].plot(3.0, 3.0, "bx", markersize=10, label="Center (3, 3)")
            axes[1].set_title(f"Grad-CAM 7x7 ({top_class})\nDist={dist_from_center:.2f}")
            axes[1].legend(loc="lower right", fontsize=7)
            plt.colorbar(im_cam, ax=axes[1], fraction=0.046, pad=0.04)

            # Overlay
            axes[2].imshow(resized)
            axes[2].imshow(cam_norm, cmap="jet", alpha=0.45, interpolation="bilinear")
            axes[2].set_title(f"{gate_outcome}\n{top_class}: {top_conf*100:.1f}% (H={entropy:.2f})")
            axes[2].axis("off")

            out_fig_path = OUTPUT_DIR / f"{img_path.stem}_audit.png"
            plt.tight_layout()
            plt.savefig(out_fig_path, dpi=120)
            plt.close()

            print(f"[{gate_outcome:22s}] {img_path.name[:25]:25s} -> {top_class:9s} ({top_conf*100:5.1f}%) | 2nd: {top2_class:9s} ({top2_conf*100:5.1f}%) | H={entropy:4.2f} | Blur={blur_score:5.1f} | CAM_Dist={dist_from_center:.2f}")

        except Exception as e:
            print(f"[ERROR] Failed {img_path.name}: {e}")

    # Save complete JSON
    out_json = ROOT_DIR / "ml" / "failure_audit_results.json"
    with open(out_json, "w") as f:
        json.dump(audit_records, f, indent=2)
    print(f"\nAudit complete! Saved detailed records to: {out_json}")
    print(f"Saved visual audit plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
