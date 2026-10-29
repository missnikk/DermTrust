"""
ml/test_feature_head_calibration.py
Extracts 1280-d MobileNetV2 features from a balanced multi-ethnic dataset and
trains an optimal calibrated projection head using convex logistic regression with
class-balanced sample weighting.
Evaluates on the triple-audited 60-image benchmark.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"
BENCHMARK_CSV = DATA_DIR / "triple_audited_benchmark_60.csv"
TRAIN_MANIFEST = DATA_DIR / "train_6class_manifest.csv"
VAL_MANIFEST = DATA_DIR / "val_6class_manifest.csv"
KERAS_MODEL = ROOT_DIR / "ml" / "experiments" / "candidate_6class" / "best_candidate_model.keras"

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "not_recognised"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def load_img(path: str) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            return np.array(rgb, dtype=np.float32) / 255.0
    except Exception:
        return None

def main():
    print("Loading base MobileNetV2 backbone (feature extractor)...")
    base_model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
    # Feature extractor up to gap (GlobalAveragePooling2D)
    gap_layer = base_model.get_layer("gap")
    feat_extractor = tf.keras.Model(inputs=base_model.input, outputs=gap_layer.output)

    # 1. Extract features for Benchmark (60 images)
    print("Extracting features for 60-image benchmark...")
    bench_df = pd.read_csv(BENCHMARK_CSV)
    X_bench, y_bench = [], []
    for _, row in bench_df.iterrows():
        arr = load_img(row["image_path"])
        if arr is not None:
            feat = feat_extractor(np.expand_dims(arr, 0), training=False).numpy()[0]
            X_bench.append(feat)
            y_bench.append(CLASS_TO_IDX[row["disease"]])
    X_bench = np.array(X_bench)
    y_bench = np.array(y_bench)
    print(f"Benchmark features shape: {X_bench.shape}")

    # 2. Prepare balanced training set (equal per class)
    print("Extracting features for training set...")
    train_df = pd.read_csv(TRAIN_MANIFEST)
    # Stratified balanced sample: e.g. 100 images per class
    balanced_rows = []
    for c in CLASSES:
        sub = train_df[train_df["verified_label"] == c]
        n_take = min(len(sub), 150)
        balanced_rows.append(sub.sample(n=n_take, random_state=42))
    train_sample = pd.concat(balanced_rows, ignore_index=True)
    print(f"Sampled {len(train_sample)} images across {len(CLASSES)} classes")

    X_train, y_train = [], []
    for _, row in train_sample.iterrows():
        arr = load_img(row["local_path"])
        if arr is not None:
            feat = feat_extractor(np.expand_dims(arr, 0), training=False).numpy()[0]
            X_train.append(feat)
            y_train.append(CLASS_TO_IDX[row["verified_label"]])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Train features shape: {X_train.shape}")

    # 3. Fit calibrated linear probe with class_weight='balanced'
    print("\nFitting class-balanced logistic regression head...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate on Benchmark
    y_pred = clf.predict(X_bench)
    probs = clf.predict_proba(X_bench)

    print("\n" + "=" * 65)
    print("NEW BALANCED HEAD BENCHMARK EVALUATION:")
    print("=" * 65)
    for i, c in enumerate(CLASSES):
        mask = (y_bench == i)
        total = int(np.sum(mask))
        corr = int(np.sum((y_pred == i) & mask))
        pct = (corr / total * 100.0) if total > 0 else 0.0
        status = "PASS (>=85%)" if pct >= 85.0 else "FAIL (<85%)"
        print(f"  {c.upper():15s}: {corr}/{total} ({pct:.1f}%) -- {status}")

    total_acc = np.mean(y_pred == y_bench) * 100.0
    print(f"\nOVERALL BENCHMARK ACCURACY: {int(np.sum(y_pred == y_bench))}/{len(y_bench)} ({total_acc:.1f}%)")

if __name__ == "__main__":
    main()
