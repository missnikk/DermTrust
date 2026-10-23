"""
Generate Grad-CAM visual verification for Acne to complete all 5 classes.
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

CAM_MODEL_PATH = "flutter_app/assets/models/dermtrust_cam.tflite"
WEIGHTS_PATH = "flutter_app/assets/models/cam_weights.json"
TEST_MANIFEST = "ml/data/test_5class_manifest.csv"
OUT_PATH = "docs/gradcam_comparison/test_acne.jpg_gradcam_before_after.png"

with open(WEIGHTS_PATH, "r") as f:
    cam_data = json.load(f)
classes = cam_data["classes"]
weights = np.array(cam_data["weights"])

cam_interp = tf.lite.Interpreter(model_path=CAM_MODEL_PATH)
cam_interp.allocate_tensors()
inp_details = cam_interp.get_input_details()[0]
out_details = cam_interp.get_output_details()

feat_idx = None
prob_idx = None
for d in out_details:
    shape = list(d["shape"])
    if shape == [1, 7, 7, 1280]:
        feat_idx = d["index"]
    elif shape == [1, 5]:
        prob_idx = d["index"]

import pandas as pd
df = pd.read_csv(TEST_MANIFEST)
acne_row = df[df["verified_label"] == "acne"].iloc[0]
img_path = acne_row["local_path"]

orig_img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
arr = np.expand_dims(np.array(orig_img, dtype=np.float32) / 255.0, axis=0)

cam_interp.set_tensor(inp_details["index"], arr)
cam_interp.invoke()

feats = cam_interp.get_tensor(feat_idx)[0]
probs = cam_interp.get_tensor(prob_idx)[0]
pred_c = int(np.argmax(probs))

cls_weights = weights[pred_c]
cam_proper = np.maximum(0, np.tensordot(feats, cls_weights, axes=([2], [0])))
norm_proper = (cam_proper - cam_proper.min()) / (cam_proper.max() - cam_proper.min() + 1e-6)

cam_aliased = np.full((7, 7), cam_proper[6, 6])
norm_aliased = np.zeros((7, 7))

def make_heatmap_overlay(orig, norm7x7):
    h_img = Image.fromarray((norm7x7 * 255).astype(np.uint8), mode='L').resize((224, 224), Image.Resampling.BILINEAR)
    h_arr = np.array(h_img, dtype=np.float32) / 255.0
    r = np.clip(1.5 - np.abs(h_arr * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(h_arr * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(h_arr * 4 - 1), 0, 1)
    rgb_heat = np.stack([r, g, b], axis=-1) * 255.0
    orig_arr = np.array(orig, dtype=np.float32)
    blend = 0.5 * orig_arr + 0.5 * rgb_heat
    return Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))

overlay_aliased = make_heatmap_overlay(orig_img, norm_aliased)
overlay_proper = make_heatmap_overlay(orig_img, norm_proper)

combined = Image.new("RGB", (672, 224))
combined.paste(orig_img, (0, 0))
combined.paste(overlay_aliased, (224, 0))
combined.paste(overlay_proper, (448, 0))
combined.save(OUT_PATH)

print(f"Saved Acne Grad-CAM comparison: {OUT_PATH}")
print(f"Proper CAM range: [{cam_proper.min():.2f}, {cam_proper.max():.2f}], Delta={cam_proper.max()-cam_proper.min():.2f}")
