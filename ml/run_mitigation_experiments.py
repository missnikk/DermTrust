"""
DermTrust - Controlled Bias Mitigation Experiments (Phase 3)
Evaluates 4 simplest mitigation methods on the training split,
using validation split strictly for model selection.
Locked test split remains untouched during this phase.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

CLASSES = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DATA_DIR = 'ml/data'
EXP_MODELS_DIR = 'ml/models/experiments'
os.makedirs(EXP_MODELS_DIR, exist_ok=True)

# 1. Preload datasets into memory
def load_split_data(manifest_path):
    print(f"Loading {manifest_path} into memory...")
    df = pd.read_csv(manifest_path)
    imgs = []
    labels = []
    skin_tones = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        p = row['local_path']
        lbl = row['verified_label']
        st = str(row.get('skin_tone_group', 'Unknown'))
        if lbl in CLASS_TO_IDX and os.path.exists(p):
            try:
                with Image.open(p) as img:
                    rgb = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
                    arr = np.array(rgb, dtype=np.float32) / 255.0
                    imgs.append(arr)
                    labels.append(CLASS_TO_IDX[lbl])
                    skin_tones.append(st)
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                
    imgs_arr = np.array(imgs, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)
    valid_df = df.iloc[valid_indices].copy().reset_index(drop=True)
    print(f"Loaded {len(imgs_arr)} images. Shape: {imgs_arr.shape}")
    return imgs_arr, labels_arr, skin_tones, valid_df

X_train, y_train, train_tones, train_df = load_split_data(os.path.join(DATA_DIR, 'train_5class_manifest.csv'))
X_val, y_val, val_tones, val_df = load_split_data(os.path.join(DATA_DIR, 'val_5class_manifest.csv'))

# Architecture builder matching existing model
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

    model = models.Model(inputs=inputs, outputs=outputs)
    cam_model = models.Model(inputs=inputs, outputs=[outputs, conv_features])

    for layer in base.layers[-30:]:
        layer.trainable = True

    return model, cam_model

# Evaluation helper strictly on validation data
def evaluate_model_on_val(model, X_val, y_val, val_tones, val_df):
    probs = model.predict(X_val, batch_size=32, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    acc = float(accuracy_score(y_val, preds))
    report = classification_report(y_val, preds, target_names=CLASSES, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, preds, labels=range(5)).tolist()
    
    # Subgroup metrics
    subgroup_metrics = {}
    for st in ['Light', 'Darker']:
        mask = [t == st for t in val_tones]
        if sum(mask) > 0:
            st_y = y_val[mask]
            st_pred = preds[mask]
            st_acc = float(accuracy_score(st_y, st_pred))
            subgroup_metrics[st] = {
                'count': int(sum(mask)),
                'accuracy': st_acc
            }
            
    mel_idx = CLASS_TO_IDX['melasma']
    mel_metrics = {
        'precision': report['melasma']['precision'],
        'recall': report['melasma']['recall'],
        'f1': report['melasma']['f1-score'],
        'support': report['melasma']['support']
    }
    
    return {
        'accuracy': acc,
        'macro_f1': report['macro avg']['f1-score'],
        'melasma': mel_metrics,
        'per_class': {c: report[c] for c in CLASSES},
        'confusion_matrix': cm,
        'subgroups': subgroup_metrics
    }

# Data generators for experiments
class StandardTrainGen(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment=False, targeted_aug=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.targeted_aug = targeted_aug
        self.indices = np.arange(len(X))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.X[batch_idx].copy()
        batch_y = tf.keras.utils.to_categorical(self.y[batch_idx], num_classes=5)

        if self.augment:
            for i in range(len(batch_x)):
                if np.random.rand() > 0.5:
                    batch_x[i] = np.fliplr(batch_x[i])
                    
        if self.targeted_aug:
            # Conservative augmentation for melasma (class 1)
            for i in range(len(batch_x)):
                cls = self.y[batch_idx[i]]
                if cls == CLASS_TO_IDX['melasma']:
                    # Conservative contrast (0.95 - 1.05)
                    factor = 0.95 + 0.10 * np.random.rand()
                    batch_x[i] = np.clip(batch_x[i] * factor, 0.0, 1.0)
                    # Conservative brightness (-0.05 to +0.05)
                    b_shift = -0.05 + 0.10 * np.random.rand()
                    batch_x[i] = np.clip(batch_x[i] + b_shift, 0.0, 1.0)
                    if np.random.rand() > 0.5:
                        batch_x[i] = np.fliplr(batch_x[i])

        return batch_x, batch_y

class BalancedSamplingGen(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=35, steps_per_epoch=45):
        # 35 samples per batch = 7 samples per class
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.class_indices = {c_idx: np.where(y == c_idx)[0] for c_idx in range(5)}
        self.samples_per_class = batch_size // 5

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        selected_idx = []
        for c_idx in range(5):
            choices = np.random.choice(self.class_indices[c_idx], size=self.samples_per_class, replace=True)
            selected_idx.extend(choices)
        np.random.shuffle(selected_idx)
        
        batch_x = self.X[selected_idx].copy()
        for i in range(len(batch_x)):
            if np.random.rand() > 0.5:
                batch_x[i] = np.fliplr(batch_x[i])
                
        batch_y = tf.keras.utils.to_categorical(self.y[selected_idx], num_classes=5)
        return batch_x, batch_y

experiments = {}

# Compute standard inverse-frequency class weights
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: float(w) for i, w in enumerate(weights)}
print("\nClass weights (inverse-frequency):")
for c, idx in CLASS_TO_IDX.items():
    print(f"  {c:<10}: {class_weights_dict[idx]:.3f}")

y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=5)

# -------------------------------------------------------------
# EXPERIMENT 1: Baseline (Unweighted Categorical Cross-Entropy)
# -------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENT 1: Baseline (Unweighted Cross-Entropy, Standard Aug)")
print("="*60)
set_seed(SEED)
model_exp1, cam_exp1 = build_model()
model_exp1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
train_gen1 = StandardTrainGen(X_train, y_train, batch_size=32, augment=True)
t0 = time.time()
hist1 = model_exp1.fit(
    train_gen1,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    verbose=1
)
dur1 = time.time() - t0
eval1 = evaluate_model_on_val(model_exp1, X_val, y_val, val_tones, val_df)
eval1['train_duration_sec'] = dur1
eval1['epochs'] = 10
eval1['loss'] = 'categorical_crossentropy (unweighted)'
model_exp1.save(os.path.join(EXP_MODELS_DIR, 'model_exp1_baseline.keras'))
experiments['Exp1_Baseline'] = eval1
print(f"Exp 1 Validation Results: Acc={eval1['accuracy']*100:.2f}%, Melasma Recall={eval1['melasma']['recall']*100:.1f}%, Macro F1={eval1['macro_f1']:.4f}")

# -------------------------------------------------------------
# EXPERIMENT 2: Class Weighting (Inverse-Frequency Weights)
# -------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENT 2: Class Weighting (Balanced Inverse-Frequency Weights)")
print("="*60)
set_seed(SEED)
model_exp2, cam_exp2 = build_model()
model_exp2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
train_gen2 = StandardTrainGen(X_train, y_train, batch_size=32, augment=True)
t0 = time.time()
hist2 = model_exp2.fit(
    train_gen2,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    class_weight=class_weights_dict,
    verbose=1
)
dur2 = time.time() - t0
eval2 = evaluate_model_on_val(model_exp2, X_val, y_val, val_tones, val_df)
eval2['train_duration_sec'] = dur2
eval2['epochs'] = 10
eval2['loss'] = 'categorical_crossentropy (class_weighted)'
eval2['class_weights'] = class_weights_dict
model_exp2.save(os.path.join(EXP_MODELS_DIR, 'model_exp2_class_weighted.keras'))
experiments['Exp2_ClassWeighting'] = eval2
print(f"Exp 2 Validation Results: Acc={eval2['accuracy']*100:.2f}%, Melasma Recall={eval2['melasma']['recall']*100:.1f}%, Macro F1={eval2['macro_f1']:.4f}")

# -------------------------------------------------------------
# EXPERIMENT 3: Balanced Batch Sampling
# -------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENT 3: Balanced Batch Sampling (Equal representation per batch)")
print("="*60)
set_seed(SEED)
model_exp3, cam_exp3 = build_model()
model_exp3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
train_gen3 = BalancedSamplingGen(X_train, y_train, batch_size=35, steps_per_epoch=45)
t0 = time.time()
hist3 = model_exp3.fit(
    train_gen3,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    verbose=1
)
dur3 = time.time() - t0
eval3 = evaluate_model_on_val(model_exp3, X_val, y_val, val_tones, val_df)
eval3['train_duration_sec'] = dur3
eval3['epochs'] = 10
eval3['sampling'] = 'balanced_batch_uniform_per_class'
model_exp3.save(os.path.join(EXP_MODELS_DIR, 'model_exp3_balanced_sampling.keras'))
experiments['Exp3_BalancedSampling'] = eval3
print(f"Exp 3 Validation Results: Acc={eval3['accuracy']*100:.2f}%, Melasma Recall={eval3['melasma']['recall']*100:.1f}%, Macro F1={eval3['macro_f1']:.4f}")

# -------------------------------------------------------------
# EXPERIMENT 4: Conservative Augmentation + Class Weighting
# -------------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENT 4: Conservative Label-Preserving Augmentation + Class Weighting")
print("="*60)
set_seed(SEED)
model_exp4, cam_exp4 = build_model()
model_exp4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
train_gen4 = StandardTrainGen(X_train, y_train, batch_size=32, augment=True, targeted_aug=True)
t0 = time.time()
hist4 = model_exp4.fit(
    train_gen4,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    class_weight=class_weights_dict,
    verbose=1
)
dur4 = time.time() - t0
eval4 = evaluate_model_on_val(model_exp4, X_val, y_val, val_tones, val_df)
eval4['train_duration_sec'] = dur4
eval4['epochs'] = 10
eval4['strategy'] = 'conservative_melasma_aug_plus_class_weights'
model_exp4.save(os.path.join(EXP_MODELS_DIR, 'model_exp4_conservative_aug.keras'))
experiments['Exp4_ConservativeAug'] = eval4
print(f"Exp 4 Validation Results: Acc={eval4['accuracy']*100:.2f}%, Melasma Recall={eval4['melasma']['recall']*100:.1f}%, Macro F1={eval4['macro_f1']:.4f}")

# Save all validation results
with open('ml/experiment_results.json', 'w') as f:
    json.dump(experiments, f, indent=2)

print("\n" + "="*60)
print("MITIGATION EXPERIMENTS COMPLETE. SUMMARY TABLE (VALIDATION SET):")
print("="*60)
summary_rows = []
for exp_name, res in experiments.items():
    row = {
        'Experiment': exp_name,
        'Val Accuracy': f"{res['accuracy']*100:.2f}%",
        'Macro F1': f"{res['macro_f1']:.4f}",
        'Melasma Recall': f"{res['melasma']['recall']*100:.1f}%",
        'Melasma F1': f"{res['melasma']['f1']:.4f}",
        'Light Tone Acc': f"{res['subgroups'].get('Light', {}).get('accuracy', 0)*100:.1f}%",
        'Darker Tone Acc': f"{res['subgroups'].get('Darker', {}).get('accuracy', 0)*100:.1f}%",
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
