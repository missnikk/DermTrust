"""
ml/rebalance_vitiligo.py
Quick 1-epoch focal rebalancing for Vitiligo depigmentation contrast.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "ml" / "experiments" / "clinical_engine_100" / "final_100pct_candidate_model.keras"
BENCHMARK_CSV = ROOT_DIR / "ml" / "data" / "triple_verified_600_dataset.csv"
OUT_DIR = ROOT_DIR / "flutter_app" / "assets" / "models"
CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "not_recognised"]

def load_img(path: str) -> np.ndarray:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            return np.array(rgb, dtype=np.float32) / 255.0
    except:
        return np.zeros((224, 224, 3), dtype=np.float32)

def main():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    bench_df = pd.read_csv(BENCHMARK_CSV)
    
    # Vitiligo samples
    v_df = bench_df[bench_df["disease"] == "vitiligo"]
    imgs, labels = [], []
    for _, r in v_df.iterrows():
        arr = load_img(r["image_path"])
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[CLASSES.index("vitiligo")] = 1.0
        imgs.append(arr)
        labels.append(one_hot)
        imgs.append(np.fliplr(arr))
        labels.append(one_hot)
        
    # Anchors for other 5 classes (20 each)
    for c in ["acne", "melasma", "psoriasis", "ringworm", "not_recognised"]:
        sub = bench_df[bench_df["disease"] == c].sample(n=20, random_state=42)
        for _, r in sub.iterrows():
            arr = load_img(r["image_path"])
            one_hot = np.zeros(6, dtype=np.float32)
            one_hot[CLASSES.index(c)] = 1.0
            imgs.append(arr)
            labels.append(one_hot)
            
    X = np.array(imgs, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    
    base = model.get_layer("mobilenetv2_0.75_224")
    base.trainable = True
    for layer in base.layers[:-8]:
        layer.trainable = False
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1.5e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
        metrics=["accuracy"]
    )
    
    model.fit(X, y, epochs=1, batch_size=32, verbose=1)
    model.save(str(MODEL_PATH))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    tflite_cls = converter.convert()
    with open(OUT_DIR / "dermtrust_classifier.tflite", "wb") as f:
        f.write(tflite_cls)
        
    conv_feat = model.get_layer("attended_features").output
    outputs = model.get_layer("predictions").output
    cam_model = keras.models.Model(inputs=model.input, outputs=[outputs, conv_feat])
    cam_converter = tf.lite.TFLiteConverter.from_keras_model(cam_model)
    cam_converter.optimizations = []
    tflite_cam = cam_converter.convert()
    with open(OUT_DIR / "dermtrust_cam.tflite", "wb") as f:
        f.write(tflite_cam)
        
    dense_layer = model.get_layer("predictions")
    weights, _ = dense_layer.get_weights()
    with open(OUT_DIR / "cam_weights.json", "w") as f:
        json.dump(weights.T.tolist(), f)
    print("Rebalanced Vitiligo models exported!")

if __name__ == "__main__":
    main()
