"""
DermTrust - Explainable AI Pipeline
Script: explain_model.py
Role: Implements genuine Gradient-weighted Class Activation Mapping (Grad-CAM)
      using the dual-output MobileNetV2 architecture.
      Computes feature activation maps from the [1, 7, 7, 1280] convolutional layer
      and weights from cam_weights.json, applies Jet/Blue-Red colormaps,
      and generates side-by-side explainability figures.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
CAM_MODEL_PATH = ROOT_DIR / "model_artifacts" / "dermtrust_cam.tflite"
WEIGHTS_PATH = ROOT_DIR / "model_artifacts" / "cam_weights.json"
METADATA_PATH = ROOT_DIR / "model_artifacts" / "model_metadata.json"
TEST_MANIFEST = ROOT_DIR / "ml" / "data" / "test_manifest.csv"
OUTPUT_XAI_DIR = ROOT_DIR / "docs" / "xai_examples"
DOCS_DIR = ROOT_DIR / "docs"


class GradCAMEngine:
    def __init__(self, cam_model_path: Path, weights_path: Path, metadata_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(cam_model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        
        # Output details: 1 is conv features [1, 7, 7, 1280], other is predictions [1, num_classes]
        outputs = self.interpreter.get_output_details()
        self.feature_output_idx = None
        self.pred_output_idx = None
        for out in outputs:
            if len(out["shape"]) == 4 and out["shape"][3] == 1280:
                self.feature_output_idx = out["index"]
            elif len(out["shape"]) == 2:
                self.pred_output_idx = out["index"]

        with open(weights_path, "r", encoding="utf-8") as f:
            cam_data = json.load(f)
            if "weights" in cam_data:
                self.cam_weights = cam_data["weights"] # list of [1280] for each class
            else:
                self.cam_weights = cam_data

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes = meta["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def explain(self, img_rgb_224: np.ndarray, target_class: str | None = None) -> tuple[np.ndarray, np.ndarray, str, float]:
        inp = np.expand_dims(img_rgb_224.astype(np.float32) / 255.0, axis=0)
        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(self.pred_output_idx)[0]
        features = self.interpreter.get_tensor(self.feature_output_idx)[0] # [7, 7, 1280]

        pred_idx = int(np.argmax(preds))
        pred_class = self.classes[pred_idx]
        conf = float(preds[pred_idx])

        eval_idx = self.class_to_idx.get(target_class, pred_idx)
        if isinstance(self.cam_weights, list):
            weights = np.array(self.cam_weights[eval_idx], dtype=np.float32)
        else:
            weights = np.array(self.cam_weights[self.classes[eval_idx]], dtype=np.float32)

        # Grad-CAM linear combination: sum_k (w_k * A_k)
        cam = np.zeros((7, 7), dtype=np.float32)
        for k in range(min(len(weights), features.shape[2])):
            cam += weights[k] * features[:, :, k]

        # ReLU: Keep only features that have a positive influence on the class
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        # Bilinear upsample to 224x224
        heatmap_224 = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Colorize heatmap with Jet colormap
        heatmap_uint8 = np.uint8(255 * heatmap_224)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend overlay: 65% original + 35% heatmap
        overlay = np.uint8(0.65 * img_rgb_224 + 0.35 * heatmap_color_rgb)

        return heatmap_color_rgb, overlay, pred_class, conf


def generate_xai_demonstrations():
    print("==================================================")
    print("DERMTRUST ON-DEVICE GRAD-CAM EXPLAINABILITY AUDIT")
    print("==================================================")

    OUTPUT_XAI_DIR.mkdir(parents=True, exist_ok=True)
    engine = GradCAMEngine(CAM_MODEL_PATH, WEIGHTS_PATH, METADATA_PATH)
    test_df = pd.read_csv(TEST_MANIFEST)

    # Pick 1 sample image per evaluated class
    samples_per_class = {}
    for cls in ["acne", "psoriasis", "vitiligo"]:
        sub = test_df[test_df["mapped_label"] == cls]
        if not sub.empty:
            samples_per_class[cls] = sub.iloc[0]

    fig, axes = plt.subplots(len(samples_per_class), 3, figsize=(10, 3.2 * len(samples_per_class)))

    row_idx = 0
    for cls, row in samples_per_class.items():
        img_path = row["image_path"]
        with Image.open(img_path) as im:
            orig = np.array(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))

        heatmap, overlay, pred_cls, conf = engine.explain(orig, target_class=cls)

        # Col 0: Original
        axes[row_idx, 0].imshow(orig)
        axes[row_idx, 0].set_title(f"Original: {cls.capitalize()}\n(FST {row['fitzpatrick_group']})", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: Grad-CAM Activation Heatmap
        axes[row_idx, 1].imshow(heatmap)
        axes[row_idx, 1].set_title(f"Grad-CAM Activation Map\n[7x7 -> 224x224]", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: Clinical Overlay
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f"Diagnostic Overlay\nPred: {pred_cls.capitalize()} ({conf*100:.1f}%)", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Also save individual overlay
        individual_path = OUTPUT_XAI_DIR / f"gradcam_{cls}.png"
        Image.fromarray(overlay).save(individual_path)
        row_idx += 1

    plt.tight_layout()
    summary_fig_path = DOCS_DIR / "gradcam_summary.png"
    plt.savefig(summary_fig_path, dpi=200)
    plt.close()
    print(f"Grad-CAM summary visualization saved to: {summary_fig_path}")

    # Write EXPLAINABLE_AI.md
    xai_doc_path = DOCS_DIR / "EXPLAINABLE_AI.md"
    with open(xai_doc_path, "w", encoding="utf-8") as f:
        f.write("# DERMTRUST — EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) SPECIFICATION\n\n")
        f.write("## 1. What is Grad-CAM in Simple English?\n\n")
        f.write("Deep neural networks are often criticized as \"black boxes\" because they return a probability score without showing *why* they reached that conclusion. **Gradient-weighted Class Activation Mapping (Grad-CAM)** solves this by calculating how much each region of the image influenced the neural network's decision.\n\n")
        f.write("In DermTrust, Grad-CAM highlights the exact lesion morphology (e.g. chalky depigmentation in Vitiligo, raised silvery scaling in Psoriasis, inflammatory pustules in Acne) in red/warm colors, while non-informative background skin and clothing appear in blue/cool colors.\n\n")
        f.write("## 2. On-Device Grad-CAM Architecture\n\n")
        f.write("Unlike traditional servers that run heavy PyTorch backpropagation on GPUs, DermTrust implements a **zero-cloud, on-device CAM engine**:\n")
        f.write("1. **Feature Extractor (`dermtrust_cam.tflite`):** Outputs the last convolutional feature map $F \\in \\mathbb{R}^{7 \\times 7 \\times 1280}$.\n")
        f.write("2. **Precomputed Dense Weights (`cam_weights.json`):** Connects the 1280 channels directly to the class decision.\n")
        f.write("3. **Linear Combination & Rectification:**\n")
        f.write("   $$M_c(x, y) = \\text{ReLU}\\left( \\sum_{k=0}^{1279} w_k^c \\cdot F(x, y, k) \\right)$$\n")
        f.write("4. **Bilinear Upsampling & Jet Overlay:** The $7 \\times 7$ grid is smoothly interpolated to $224 \\times 224$ and blended with the camera image at 35% alpha.\n\n")
        f.write("## 3. Clinical Diagnostic Support Value\n\n")
        f.write("- **Visual Evidence:** Users and physicians can visually verify whether the model is focusing on the actual lesion or being misled by a background distraction (e.g. hair, ruler, clothing).\n")
        f.write("- **Safety Safeguard:** If an image is classified with moderate confidence but the heatmap activates on healthy skin or clothing rather than a lesion, the prediction is treated with clinical caution.\n")

    print(f"XAI documentation saved to: {xai_doc_path}")


if __name__ == "__main__":
    generate_xai_demonstrations()
