"""
DermTrust - Machine Learning Pipeline
Script: export_model.py
Role: Converts trained Keras MobileNetV2 and CAM models to optimized TFLite formats
      with INT8 and Float32 quantization options and exports cam_weights.json.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "ml" / "models"
EXPORT_DIR = ROOT_DIR / "model_artifacts"


def export_tflite(keras_model_path: Path, output_tflite_path: Path, quantize_int8: bool = False):
    model = tf.keras.models.load_model(str(keras_model_path), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Exported TFLite: {output_tflite_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")


def export_cam_weights(keras_model_path: Path, output_json_path: Path):
    model = tf.keras.models.load_model(str(keras_model_path), compile=False)
    dense_layer = model.get_layer("predictions")
    weights = dense_layer.get_weights()[0] # [1280, num_classes]
    
    with open(EXPORT_DIR / "model_metadata.json", "r") as f:
        meta = json.load(f)
    classes = meta["classes"]
    
    cam_dict = {
        "description": "Grad-CAM alpha weights (dense layer weights for GAP model)",
        "num_features": int(weights.shape[0]),
        "num_classes": int(weights.shape[1]),
        "feature_map_size": 7,
        "weights": weights.T.tolist()
    }
    with open(output_json_path, "w") as f:
        json.dump(cam_dict, f, indent=2)
    print(f"Exported CAM weights: {output_json_path}")


if __name__ == "__main__":
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    m_path = MODELS_DIR / "dermtrust_mobilenetv2.keras"
    if m_path.is_file():
        export_tflite(m_path, EXPORT_DIR / "dermtrust_classifier.tflite", quantize_int8=True)
        export_cam_weights(m_path, EXPORT_DIR / "cam_weights.json")
    else:
        print("Model checkpoint not found. Using deployed production models in model_artifacts/.")
