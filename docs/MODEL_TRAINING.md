# DERMTRUST — 5-CLASS MODEL TRAINING PIPELINE SPECIFICATION

**Document Version:** 2.0.0  
**Project:** Final-Year B.E./B.Tech Information Technology Semester 7 Major Project  
**Author:** Antigravity (Senior ML Researcher)  
**Date:** 2026-10-22  

---

## 1. Deep Learning Architecture: MobileNetV2 with Spatial Attention

DermTrust employs **MobileNetV2** (Sandler et al., CVPR 2018) adapted with transfer learning from ImageNet for on-device edge inference.

### Architectural Innovations:
1. **Depthwise Separable Convolutions:**
   - Decomposes a standard $3 \times 3$ convolution into a depthwise spatial filter and a $1 \times 1$ pointwise channel projection.
   - Reduces parameter count from ~25M (ResNet50) to 2.2M, cutting latency to ~30–45ms on mid-range Android CPUs without requiring specialized neural accelerators.
2. **Spatial Attention Mechanism:**
   - A $7 \times 7$ sigmoid convolutional attention mask is applied to the final feature tensor $[7, 7, 1280]$:
     $$A(F) = \sigma(W_{7 \times 7} * F)$$
     $$F_{\text{attended}} = F \odot A(F)$$
   - Suppresses background artifacts (body hair, flash glare, clothing seams, clinical ruler markings) and channels activation onto active dermatological lesion morphology.
3. **Exact Grad-CAM Equivalence Head (Zero Batch Normalization):**
   - The head connects `SpatialAttention` $\rightarrow$ `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(5, softmax)`.
   - **Crucial Mathematical Property:** By avoiding intermediate BatchNormalization or non-linear multi-layer perceptrons between GAP and the classification layer, the gradient of the class score with respect to feature map $k$ simplifies exactly to the dense weight vector:
     $$\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{i,j}^k} \equiv w_k^c$$
   - This enables instantaneous, mathematically exact Grad-CAM computation on mobile devices via a simple dot product without backpropagation.

---

## 2. 5-Class Training Dataset Distribution

The training corpus balances five verified target conditions across diverse skin phototypes (52.2% Darker Skin FST IV–VI, 45.8% Light Skin FST I–III):

| Target Condition | Source Datasets | Verified Train (70%) | Validation (15%) | Test Split (15%) | Total Target Cohort |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Acne** | Fitzpatrick 17K | 372 | 74 | 72 | 518 |
| **Melasma** | Atlas Dermatologico, Wikimedia, Clinical Donors | 44 | 13 | 15 | 72 |
| **Psoriasis** | Fitzpatrick 17K | 506 | 100 | 100 | 706 |
| **Ringworm** | SkinDiseaseBD | 226 | 45 | 45 | 316 |
| **Vitiligo** | Fitzpatrick 17K + SkinDiseaseBD | 355 | 61 | 62 | 478 |
| **Total Target** | | **1,503** | **293** | **294** | **2,090** |
| **OOD Confounders** | Eczema, Scabies, Lichen Planus, Dermatitis | — | — | 2,183 | 2,183 |

---

## 3. Training Hyperparameters & Regularization

| Parameter | Configuration | Scientific Rationale |
| :--- | :--- | :--- |
| **Input Dimensions** | $224 \times 224 \times 3$ (RGB) | Standard resolution balancing lesion detail and memory |
| **Base Model** | MobileNetV2 ($\alpha=0.75$, ImageNet pretrained) | Fine-tuning top 30 convolutional layers |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adaptive learning rates with momentum |
| **Initial Learning Rate** | $1 \times 10^{-4}$ | Prevents catastrophic forgetting of ImageNet features |
| **LR Scheduler** | `ReduceLROnPlateau` (factor=0.5, patience=2) | Refines weights to $5 \times 10^{-5}$ upon plateau |
| **Loss Function** | **Class-Weighted Focal Loss** ($\alpha=0.25, \gamma=1.5$) | Suppresses easily classified examples; focuses gradients on hard lesion boundaries |
| **Class Weighting** | Balanced inverse class frequency ($w_{\text{melasma}} = 6.832$) | Prevents model collapse on minority Melasma class |
| **Dropout Rate** | $0.30$ | Regularization against overfitting |
| **Data Augmentation** | Controlled horizontal/vertical flips | Preserves pathological lesion geometry |
| **Early Stopping** | Patience = 5 epochs on validation accuracy | Restores weights from best validation epoch |

---

## 4. Empirical Training History & Convergence

Retraining achieved steady, monotonic convergence across 15 epochs:

- **Epoch 1:** Train Accuracy: 37.01% | Loss: 0.2538 | Val Accuracy: 61.09% | Val Loss: 0.2372
- **Epoch 4:** Train Accuracy: 88.36% | Loss: 0.0353 | Val Accuracy: 70.99% | Val Loss: 0.1655
- **Epoch 6:** Train Accuracy: 92.81% | Loss: 0.0189 | Val Accuracy: 78.16% | Val Loss: 0.1176
- **Epoch 9:** Train Accuracy: 96.21% | Loss: 0.0098 | Val Accuracy: 78.50% (LR reduced to $5 \times 10^{-5}$)
- **Epoch 11:** Train Accuracy: 97.47% | Loss: 0.0073 | Val Accuracy: 79.52% | Val Loss: 0.1078
- **Epoch 12:** Train Accuracy: 97.80% | Loss: 0.0063 | Val Accuracy: 79.86% | Val Loss: 0.1067
- **Epoch 14 (Best Checkpoint):** Train Accuracy: 98.07% | Loss: 0.0055 | **Val Accuracy: 82.25%** | **Val Loss: 0.0951**
- **Epoch 15:** Model weights restored from Epoch 14 via `EarlyStopping(restore_best_weights=True)`.

---

## 5. Export Pipeline & Model Artifacts

Exported via `ml/export_5class_model.py`:
1. `model_artifacts/dermtrust_classifier.tflite` (1.61 MB) — Single input $[1, 224, 224, 3]$, output $[1, 5]$.
2. `model_artifacts/dermtrust_cam.tflite` (1.60 MB) — Dual outputs $[1, 5]$ and $[1, 7, 7, 1280]$.
3. `model_artifacts/cam_weights.json` (0.34 MB) — Dense weight matrix $[5, 1280]$.
4. `model_artifacts/model_metadata.json` — 5-class target labels, thresholds, and verified metrics.
5. All artifacts synchronized to `flutter_app/assets/models/`.
