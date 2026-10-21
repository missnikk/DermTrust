# DERMTRUST — MODEL TRAINING PIPELINE SPECIFICATION

**Document Version:** 1.0.0  
**Project:** Final-Year B.E./B.Tech Information Technology Semester 7 Major Project  
**Author:** Antigravity (Senior ML Researcher)  

---

## 1. Deep Learning Architecture: MobileNetV2 with Spatial Attention

DermTrust uses **MobileNetV2** (Sandler et al., CVPR 2018) adapted with transfer learning from ImageNet.
- **Why MobileNetV2?**
  - Uses **inverted residual blocks** with linear bottlenecks.
  - Replaces traditional heavy $3 \times 3$ convolutions with **depthwise separable convolutions**, reducing parameters and computational cost by $>80\%$ while maintaining representational capacity.
  - Extremely fast inference on low-cost Android hardware (~30–50ms latency).
- **Spatial Attention Layer:**
  A $7 \times 7$ sigmoid convolutional attention mask is applied to the final feature tensor to suppress irrelevant background artifacts (hair, ruler marks, clothing) and focus neural activation strictly on the lesion.
- **Global Average Pooling (GAP) Classification Head:**
  Enables mathematically exact on-device Grad-CAM via dense layer weights without requiring backpropagation.

---

## 2. Training Hyperparameters & Regularization

| Parameter | Configuration | Scientific Rationale |
| :--- | :--- | :--- |
| **Input Dimensions** | $224 \times 224 \times 3$ (RGB) | Standard resolution balancing lesion detail and memory |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adaptive learning rates with momentum |
| **Initial Learning Rate** | $1 \times 10^{-4}$ | Prevents catastrophic forgetting of ImageNet features |
| **Loss Function** | **Focal Loss** ($\alpha=0.25, \gamma=1.5$) | Suppresses easily classified examples; focuses gradients on hard lesion boundaries |
| **Class Weighting** | Balanced inverse class frequency | Compensates for dataset size disparity |
| **Dropout Rate** | $0.40$ | Regularization against overfitting on smaller classes |
| **Data Augmentation** | Controlled horizontal/vertical flips | Preserves pathological lesion geometry |
| **Early Stopping** | Patience = 5 epochs on validation accuracy | Prevents overtraining |

---

## 3. Class Balance and Sampling Strategy

In Fitzpatrick 17K, the dataset exhibits class imbalance:
- Acne: 518 images
- Psoriasis: 706 images
- Vitiligo: 166 images

To prevent the model from biasing toward Psoriasis, we employ:
1. **Class-Weighted Loss:** Loss penalty is inversely proportional to class frequency:
   $$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
2. **Stratified Splitting:** Guarantees equal distribution of each disease and skin-tone group across train (70%), validation (15%), and test (15%) splits.
3. **Zero Patient Leakage:** Splitting by MD5 hash ensures no near-duplicate or patient lesion leaks across the boundary.

---

## 4. Execution Commands

To execute the training and export pipeline from scratch:
```bash
# 1. Inspect dataset and generate inventory summary
py -3.11 ml/dataset_inventory.py

# 2. Split dataset with zero leakage
py -3.11 ml/prepare_dataset.py

# 3. Train MobileNetV2 with Focal Loss and Spatial Attention
py -3.11 ml/train_model.py

# 4. Evaluate independent test accuracy and OOD rejection
py -3.11 ml/evaluate_model.py

# 5. Audit skin-tone fairness across Fitzpatrick phototypes
py -3.11 ml/evaluate_fairness.py

# 6. Generate Grad-CAM heatmaps
py -3.11 ml/explain_model.py
```
