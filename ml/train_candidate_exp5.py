"""
DermTrust - Machine Learning Pipeline
Script: ml/train_candidate_exp5.py
Role: Train Candidate 5 (Focal Loss gamma=1.0 with Mild Multi-Scale Lesion Augmentation)
      Strictly using train_5class_manifest.csv.
      Evaluated strictly on val_5class_manifest.csv for model selection.
      Locked test split remains strictly untouched.
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

CLASSES = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DATA_DIR = Path("ml/data")
EXP_MODELS_DIR = Path("ml/models/experiments")
EXP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

train_csv = DATA_DIR / "train_5class_manifest.csv"
val_csv = DATA_DIR / "val_5class_manifest.csv"

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

train_labels = [CLASS_TO_IDX[lbl] for lbl in train_df["verified_label"]]
weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {i: float(w) for i, w in enumerate(weights)}
print("Balanced Class Weights:")
for c, i in CLASS_TO_IDX.items():
    print(f"  {c:12s}: {class_weights_dict[i]:.4f}")

def focal_loss(alpha=0.25, gamma=1.0):
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

def build_model(alpha=0.75, dropout=0.3):
    base = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", alpha=alpha
    )
    base.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="image_input")
    x = base(inputs, training=False)
    conv_features = spatial_attention(x)
    x = layers.GlobalAveragePooling2D(name="gap")(conv_features)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(5, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DermTrust_Candidate5")
    cam_model = models.Model(inputs=inputs, outputs=[outputs, conv_features], name="DermTrust_Candidate5_CAM")

    for layer in base.layers[-30:]:
        layer.trainable = True

    return model, cam_model

class AugManifestGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, is_train=True):
        self.df = df.copy().reset_index(drop=True)
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
                    if self.is_train:
                        if random.random() > 0.5:
                            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                        if random.random() > 0.5:
                            rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
                        if random.random() > 0.7:
                            factor = random.uniform(0.9, 1.1)
                            rgb = ImageEnhance.Contrast(rgb).enhance(factor)
                    arr = np.array(rgb, dtype=np.float32) / 255.0
            except Exception:
                arr = np.zeros((224, 224, 3), dtype=np.float32)

            imgs.append(arr)
            one_hot = np.zeros(5, dtype=np.float32)
            one_hot[CLASS_TO_IDX[row["verified_label"]]] = 1.0
            labels.append(one_hot)

        return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)

train_gen = AugManifestGenerator(train_df, batch_size=32, is_train=True)
val_gen = AugManifestGenerator(val_df, batch_size=32, is_train=False)

model, cam_model = build_model(alpha=0.75, dropout=0.3)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss(alpha=0.25, gamma=1.0),
    metrics=["accuracy"]
)

checkpoint_path = EXP_MODELS_DIR / "candidate5_best.weights.h5"
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, save_weights_only=True, monitor="val_accuracy", verbose=1)
]

print("\nStarting Candidate 5 Training (Max 15 Epochs)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

cand5_keras_path = EXP_MODELS_DIR / "model_exp5_focal_contrast.keras"
model.save(str(cand5_keras_path))
print(f"Saved Candidate 5 model to: {cand5_keras_path}")

# Evaluate strictly on validation split
val_imgs = []
val_y_true = []
for _, row in val_df.iterrows():
    p = row["local_path"]
    lbl = row["verified_label"]
    if lbl in CLASS_TO_IDX and os.path.exists(p):
        with Image.open(p) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            val_imgs.append(np.array(rgb, dtype=np.float32) / 255.0)
            val_y_true.append(lbl)

val_X = np.array(val_imgs, dtype=np.float32)
val_probs = model.predict(val_X, batch_size=32, verbose=0)
val_preds_idx = np.argmax(val_probs, axis=1)
val_preds = [CLASSES[i] for i in val_preds_idx]

val_acc = float(accuracy_score(val_y_true, val_preds))
rep = classification_report(val_y_true, val_preds, target_names=CLASSES, digits=4, output_dict=True)
cm = confusion_matrix(val_y_true, val_preds, labels=CLASSES)

print("\n" + "="*50)
print(f"CANDIDATE 5 VALIDATION ACCURACY: {val_acc*100:.2f}% ({np.sum(np.array(val_preds)==np.array(val_y_true))}/{len(val_y_true)})")
print(f"Macro F1:        {rep['macro avg']['f1-score']:.4f}")
print(f"Macro Recall:    {rep['macro avg']['recall']:.4f}")
print(f"Macro Precision: {rep['macro avg']['precision']:.4f}")
print("Per-class Metrics:")
for c in CLASSES:
    print(f"  {c:12s} | Prec: {rep[c]['precision']*100:6.2f}% | Rec: {rep[c]['recall']*100:6.2f}% | F1: {rep[c]['f1-score']:.4f} | Supp: {rep[c]['support']}")

results = {
    'val_accuracy': val_acc,
    'macro_f1': rep['macro avg']['f1-score'],
    'macro_recall': rep['macro avg']['recall'],
    'macro_precision': rep['macro avg']['precision'],
    'per_class': {c: rep[c] for c in CLASSES},
    'confusion_matrix': cm.tolist()
}
with open(EXP_MODELS_DIR / "candidate5_val_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved validation results to candidate5_val_results.json")
