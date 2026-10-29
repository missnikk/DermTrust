"""
ml/export_compatible_tflite.py
Exports the high-accuracy model to standard Float32 TFLite without experimental op versioning,
ensuring 100% compatibility with tflite_flutter runtime on Android.
"""

from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "ml" / "experiments" / "clinical_engine_100" / "final_100pct_candidate_model.keras"
OUT_DIR = ROOT_DIR / "flutter_app" / "assets" / "models"

print("Loading final trained model...")
model = keras.models.load_model(MODEL_PATH, compile=False)

print("Exporting classifier TFLite (Float32, optimizations=[])...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []  # Standard Float32 - avoids FULLY_CONNECTED v12 opcode
tflite_classifier = converter.convert()

out_cls = OUT_DIR / "dermtrust_classifier.tflite"
with open(out_cls, "wb") as f:
    f.write(tflite_classifier)
print(f"Wrote {out_cls} ({len(tflite_classifier)/(1024*1024):.2f} MB)")

print("Exporting CAM TFLite (Float32, optimizations=[])...")
conv_feat = model.get_layer("attended_features").output
outputs = model.get_layer("predictions").output
cam_model = keras.models.Model(inputs=model.input, outputs=[outputs, conv_feat])

cam_converter = tf.lite.TFLiteConverter.from_keras_model(cam_model)
cam_converter.optimizations = []  # Standard Float32
tflite_cam = cam_converter.convert()

out_cam = OUT_DIR / "dermtrust_cam.tflite"
with open(out_cam, "wb") as f:
    f.write(tflite_cam)
print(f"Wrote {out_cam} ({len(tflite_cam)/(1024*1024):.2f} MB)")

dense_layer = model.get_layer("predictions")
weights, _ = dense_layer.get_weights()
out_weights = OUT_DIR / "cam_weights.json"
with open(out_weights, "w") as f:
    json.dump(weights.T.tolist(), f)
print(f"Wrote {out_weights} (Shape: [6, 1280])")

print("\nTFLite Export Complete!")
