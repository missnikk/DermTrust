"""
DermTrust - Machine Learning Pipeline
Script: export_5class_model.py
Role: Exports trained 5-class MobileNetV2 and CAM models to TFLite format,
      generates cam_weights.json and model_metadata.json, and syncs
      them to both model_artifacts/ and flutter_app/assets/models/.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "ml" / "models"
EXPORT_DIR = ROOT_DIR / "model_artifacts"
FLUTTER_ASSETS_DIR = ROOT_DIR / "flutter_app" / "assets" / "models"

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]


def export_classifier_tflite(keras_model_path: Path, output_tflite_path: Path):
    print(f"Converting classifier to TFLite from: {keras_model_path}")
    model = tf.keras.models.load_model(str(keras_model_path), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Default optimization (Float32 / hybrid dynamic quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  [OK] Exported Classifier TFLite: {output_tflite_path} ({size_mb:.2f} MB)")
    return tflite_model


def export_cam_tflite(cam_model_path: Path, output_tflite_path: Path):
    print(f"Converting CAM model to TFLite from: {cam_model_path}")
    cam_model = tf.keras.models.load_model(str(cam_model_path), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(cam_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  [OK] Exported CAM TFLite: {output_tflite_path} ({size_mb:.2f} MB)")
    return tflite_model


def export_cam_weights(keras_model_path: Path, output_json_path: Path):
    print(f"Extracting Dense classification weights for CAM from: {keras_model_path}")
    model = tf.keras.models.load_model(str(keras_model_path), compile=False)
    dense_layer = model.get_layer("predictions")
    weights = dense_layer.get_weights()[0]  # Shape: [1280, 5]
    
    print(f"  Dense weights shape: {weights.shape} (1280 features x {len(CLASSES)} classes)")
    weights_transposed = weights.T.tolist()  # Shape: [5, 1280]

    cam_dict = {
        "description": "Grad-CAM alpha weights (dense layer weights for GAP model)",
        "num_features": int(weights.shape[0]),
        "num_classes": int(weights.shape[1]),
        "feature_map_size": 7,
        "classes": CLASSES,
        "weights": weights_transposed,
    }
    # Also attach per-class dictionary key for flexible lookup
    for idx, cls_name in enumerate(CLASSES):
        cam_dict[cls_name] = weights_transposed[idx]

    with open(output_json_path, "w") as f:
        json.dump(cam_dict, f, indent=2)
    print(f"  [OK] Exported CAM weights: {output_json_path}")
    return cam_dict


def export_metadata(output_json_path: Path, test_metrics: dict | None = None):
    metadata = {
        "model_name": "DermTrust-MobileNetV2-5Class-XAI",
        "version": "2.0.0",
        "evaluated_classes": CLASSES,
        "classes": CLASSES,
        "target_disease_scope": CLASSES,
        "input_size": 224,
        "input_tensor_shape": [1, 224, 224, 3],
        "cam_feature_shape": [1, 7, 7, 1280],
        "input_mean": 0.0,
        "input_std": 255.0,
        "architecture": "MobileNetV2 (alpha=0.75) + Spatial Attention + GAP + Dense",
        "grad_cam_property": "Exact alpha_k^c == w_k^c equivalence (no intermediate BatchNorm)",
        "rejection_threshold": 0.60,
        "max_entropy_threshold": 1.40,
        "min_blur_laplacian_variance": 30.0,
        "metrics": test_metrics or {
            "status": "pending_evaluation"
        }
    }
    with open(output_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Exported model metadata: {output_json_path}")
    return metadata


def sync_to_flutter():
    print(f"\nSyncing exported artifacts to Flutter assets: {FLUTTER_ASSETS_DIR}")
    FLUTTER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ["dermtrust_classifier.tflite", "dermtrust_cam.tflite", "cam_weights.json", "model_metadata.json"]:
        src = EXPORT_DIR / filename
        dst = FLUTTER_ASSETS_DIR / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {filename} -> {dst}")
        else:
            print(f"  [WARN] Source file missing: {src}")


def run_export():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    clf_keras = MODELS_DIR / "dermtrust_mobilenetv2_5class.keras"
    cam_keras = MODELS_DIR / "dermtrust_cam_5class.keras"

    if not clf_keras.exists() or not cam_keras.exists():
        raise FileNotFoundError(f"Trained Keras models not found in {MODELS_DIR}. Run train_5class_model.py first.")

    export_classifier_tflite(clf_keras, EXPORT_DIR / "dermtrust_classifier.tflite")
    export_cam_tflite(cam_keras, EXPORT_DIR / "dermtrust_cam.tflite")
    export_cam_weights(clf_keras, EXPORT_DIR / "cam_weights.json")
    export_metadata(EXPORT_DIR / "model_metadata.json")
    sync_to_flutter()
    print("\nAll 5-class model artifacts successfully exported and synchronized!")


if __name__ == "__main__":
    run_export()
