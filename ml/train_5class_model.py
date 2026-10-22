"""
DermTrust - Machine Learning Pipeline
Script: train_5class_model.py
Role: Retrains the 5-class MobileNetV2 architecture with Spatial Attention,
      Class-Weighted Focal Loss, and exact Grad-CAM equivalence (no intermediate BatchNorm).
      Uses validation split for model selection and checkpointing.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data"
MODELS_DIR = ROOT_DIR / "ml" / "models"
DOCS_DIR = ROOT_DIR / "docs"

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
SEED = 42


def set_reproducibility(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def focal_loss(alpha: float = 0.25, gamma: float = 1.5):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_factor = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(focal_factor * cross_entropy, axis=-1)
    return loss


def spatial_attention(features):
    attention = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid", name="spatial_attention"
    )(features)
    return layers.Multiply(name="attended_features")([features, attention])


def build_5class_model(num_classes: int = 5, input_size: int = 224, alpha: float = 0.75, dropout: float = 0.3):
    base = MobileNetV2(
        input_shape=(input_size, input_size, 3), include_top=False, weights="imagenet", alpha=alpha
    )
    base.trainable = False

    inputs = layers.Input(shape=(input_size, input_size, 3), name="image_input")
    x = base(inputs, training=False)
    conv_features = spatial_attention(x)

    # GAP directly into Dense: guarantees exact CAM theorem: alpha_k^c == w_k^c
    x = layers.GlobalAveragePooling2D(name="gap")(conv_features)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DermTrust_MobileNetV2_5Class")
    cam_model = models.Model(inputs=inputs, outputs=[outputs, conv_features], name="DermTrust_CAM_5Class")

    # Fine-tune top 30 layers of MobileNetV2
    for layer in base.layers[-30:]:
        layer.trainable = True

    return model, cam_model


class ManifestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, manifest_csv: Path, classes: list[str], batch_size: int = 32, is_train: bool = True):
        self.df = pd.read_csv(manifest_csv)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.batch_size = batch_size
        self.is_train = is_train
        self.indices = np.arange(len(self.df))
        if self.is_train:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.is_train:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_rows = self.df.iloc[batch_idx]

        imgs = []
        labels = []
        for _, row in batch_rows.iterrows():
            img_path = row["local_path"]
            try:
                with Image.open(img_path) as img:
                    rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    arr = np.array(rgb, dtype=np.float32) / 255.0
            except Exception:
                arr = np.zeros((224, 224, 3), dtype=np.float32)

            # Random on-the-fly augmentations during training
            if self.is_train:
                if np.random.rand() > 0.5:
                    arr = np.fliplr(arr)
                if np.random.rand() > 0.5:
                    arr = np.flipud(arr)

            imgs.append(arr)
            one_hot = np.zeros(len(self.classes), dtype=np.float32)
            one_hot[self.class_to_idx[row["verified_label"]]] = 1.0
            labels.append(one_hot)

        return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_5class():
    set_reproducibility(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("DERMTRUST 5-CLASS MODEL RETRAINING PIPELINE")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")
    print("==================================================")

    train_csv = DATA_DIR / "train_5class_manifest.csv"
    val_csv = DATA_DIR / "val_5class_manifest.csv"

    train_gen = ManifestDataGenerator(train_csv, CLASSES, batch_size=32, is_train=True)
    val_gen = ManifestDataGenerator(val_csv, CLASSES, batch_size=32, is_train=False)

    print(f"Training Samples:   {len(train_gen.df)}")
    print(f"Validation Samples: {len(val_gen.df)}")

    # Compute balanced class weights to address minority Melasma class
    train_labels = [train_gen.class_to_idx[lbl] for lbl in train_gen.df["verified_label"]]
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = {i: float(w) for i, w in enumerate(weights)}
    print("\nClass Weights (Balanced Inverse Frequency):")
    for cls, idx in train_gen.class_to_idx.items():
        print(f"  {cls:<12} (index {idx}): weight = {class_weights_dict[idx]:.3f}")

    model, cam_model = build_5class_model(num_classes=len(CLASSES), input_size=224, alpha=0.75)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(alpha=0.25, gamma=1.5),
        metrics=["accuracy"]
    )

    checkpoint_path = MODELS_DIR / "best_5class_model.weights.h5"
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, save_weights_only=True, monitor="val_accuracy", verbose=1)
    ]

    print("\nCommencing Model Training (Max 15 Epochs)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # Save final models
    model_save_path = MODELS_DIR / "dermtrust_mobilenetv2_5class.keras"
    cam_save_path = MODELS_DIR / "dermtrust_cam_5class.keras"
    model.save(str(model_save_path))
    cam_model.save(str(cam_save_path))
    print(f"\nModels successfully saved to:\n  - {model_save_path}\n  - {cam_save_path}")

    # Export Training Log History
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = DOCS_DIR / "training_5class_history.json"
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"Training history saved to: {hist_path}")

    return model, cam_model


if __name__ == "__main__":
    train_5class()
