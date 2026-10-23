"""
DermTrust - Candidate Model V3 Training Pipeline
Script: ml/train_candidate_v3.py
Role: Trains Candidate Model V3 using the expanded clean V3 dataset (train_5class_manifest_v3.csv),
      featuring balanced class weights (giving higher weight to Melasma), MobileNetV2 (alpha=0.75),
      spatial attention, and exact Grad-CAM head structure.
      Evaluated strictly against val_5class_manifest_v3.csv.
"""

from __future__ import annotations
import os
import json
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
EXP_MODELS_DIR = ROOT_DIR / "ml" / "models" / "experiments"
EXP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 42


def set_seed(seed: int = 42):
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


def build_v3_model(num_classes: int = 5, input_size: int = 224, alpha: float = 0.75, dropout: float = 0.3):
    base = MobileNetV2(
        input_shape=(input_size, input_size, 3), include_top=False, weights="imagenet", alpha=alpha
    )
    base.trainable = False

    inputs = layers.Input(shape=(input_size, input_size, 3), name="image_input")
    x = base(inputs, training=False)
    conv_features = spatial_attention(x)

    # GAP directly into Dense: exact Grad-CAM equivalence
    x = layers.GlobalAveragePooling2D(name="gap")(conv_features)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DermTrust_Candidate_V3")
    cam_model = models.Model(inputs=inputs, outputs=[outputs, conv_features], name="DermTrust_Candidate_V3_CAM")

    # Fine-tune top 30 layers
    for layer in base.layers[-30:]:
        layer.trainable = True

    return model, cam_model


class ManifestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, manifest_csv: Path, classes: list[str], batch_size: int = 32, is_train: bool = True):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
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

            # Clinical safe augmentation: horizontal flip only (no vertical flip of lesions)
            if self.is_train and np.random.rand() > 0.5:
                arr = np.fliplr(arr)

            imgs.append(arr)
            one_hot = np.zeros(len(self.classes), dtype=np.float32)
            lbl = str(row["verified_label"]).lower()
            if lbl in self.class_to_idx:
                one_hot[self.class_to_idx[lbl]] = 1.0
            labels.append(one_hot)

        return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)


def main():
    set_seed(SEED)

    train_csv = DATA_DIR / "train_5class_manifest_v3.csv"
    val_csv = DATA_DIR / "val_5class_manifest_v3.csv"

    print("==================================================")
    print("DERMTRUST CANDIDATE MODEL V3 TRAINING")
    print(f"Classes: {CLASSES}")
    print("==================================================")

    train_gen = ManifestDataGenerator(train_csv, CLASSES, batch_size=32, is_train=True)
    val_gen = ManifestDataGenerator(val_csv, CLASSES, batch_size=32, is_train=False)

    print(f"Train samples: {len(train_gen.df)}")
    print(f"Val samples:   {len(val_gen.df)}")

    # Compute balanced class weights
    train_labels = [train_gen.class_to_idx[lbl.lower()] for lbl in train_gen.df["verified_label"]]
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = {i: float(w) for i, w in enumerate(weights)}
    print("\nClass Weights (Balanced):")
    for cls, idx in train_gen.class_to_idx.items():
        print(f"  {cls:<12} (index {idx}): weight = {class_weights_dict[idx]:.4f}")

    model, cam_model = build_v3_model(num_classes=len(CLASSES), input_size=224, alpha=0.75)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(alpha=0.25, gamma=1.5),
        metrics=["accuracy"]
    )

    checkpoint_path = EXP_MODELS_DIR / "candidate_v3_checkpoint.weights.h5"
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, save_weights_only=True, monitor="val_accuracy", verbose=1)
    ]

    print("\nStarting Training (15 Epochs)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # Save candidate models
    candidate_save_path = EXP_MODELS_DIR / "candidate_v3_best.keras"
    cam_save_path = EXP_MODELS_DIR / "candidate_v3_cam.keras"
    model.save(str(candidate_save_path))
    cam_model.save(str(cam_save_path))
    print(f"\nCandidate models successfully saved:\n  {candidate_save_path}\n  {cam_save_path}")

    # Export history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = EXP_MODELS_DIR / "candidate_v3_history.json"
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"Training history saved: {hist_path}")


if __name__ == "__main__":
    main()
