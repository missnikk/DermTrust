# DERMTRUST — ARCHITECTURAL SIMPLICITY & TECHNICAL JUSTIFICATION

**Document Version:** 1.0.0  
**Project:** Final-Year B.Tech Major Project — DermTrust  
**Engineering Principle:** "The simplest suitable design that satisfies all functional and clinical requirements without unnecessary complexity."

---

## 1. Architectural Philosophy & Context

DermTrust is designed as a practical, offline-first mobile clinical decision-support system targeted for deployment on standard Android smartphones in resource-constrained or rural healthcare environments. 

In medical and edge AI, excess architectural complexity is an engineering anti-pattern: it introduces unexplainable failure modes, causes thermal throttling and battery drain on mobile hardware, inflates application binary size, and complicates safety auditing.

This document details the core algorithms selected for DermTrust, their exact code locations, and the technical justification for why simpler, robust techniques were chosen over more complicated alternatives.

---

## 2. Core Algorithmic Decisions & Comparative Justifications

### 2.1 Deep Learning Backbone: MobileNetV2 with Spatial Attention
- **Implemented Location:** `ml/models/mobilenet_attention.py`, `flutter_app/assets/models/dermtrust_classifier.tflite`
- **Selected Technique:** MobileNetV2 (width multiplier $\alpha = 0.75$) augmented with a lightweight 2D Spatial Attention mechanism and Global Average Pooling.
- **Why Selected:**
  - Employs inverted residual blocks and depthwise separable convolutions, reducing parameters to ~2.2M and model file size to **6.31 MB**.
  - On-device inference latency is only **~18 ms** on modern ARM64 processors without requiring specialized NPU hardware.
  - Spatial attention explicitly forces the network to weight spatial lesion features (erythema borders, scales, follicular plugs) over surrounding healthy skin.
- **Why More Complicated Alternatives Were Unsuitable:**
  - *Vision Transformers (ViT / Swin-T):* Require massive datasets to avoid severe overfitting, exhibit quadratic attention computational complexity, and take >350 ms per inference on mobile CPUs.
  - *Large Multi-Model Ensembles (e.g., EfficientNet-B4 + ResNet-50 + ConvNeXt):* Inflate APK download size by over 200 MB, drain battery rapidly, and create debugging nightmares without yielding statistically significant diagnostic gains on our 5-target screening problem.

---

### 2.2 Explainable AI: Offline Gradient-Weighted Channel Attribution for Mobile Grad-CAM
- **Implemented Location:** `flutter_app/lib/services/gradcam_service.dart`, `flutter_app/assets/models/cam_weights.json`
- **Selected Technique:** Pre-computed class-specific backward feature weights combined with the on-device $7 \times 7 \times 1280$ forward feature tensor.
- **Why Selected:**
  - Standard Grad-CAM requires runtime reverse-mode automatic differentiation ($\frac{\partial y^c}{\partial A^k}$ via `tf.GradientTape`), which is unsupported by the lightweight mobile TensorFlow Lite C++ runtime.
  - By computing the global average gradients of the target class score with respect to feature map activations across the verified validation cohort offline, we obtain a calibrated $1280$-dimensional weight vector $\mathbf{w}^c$ for each class.
  - On device, generating a high-fidelity $224 \times 224$ heatmap reduces to a simple matrix-vector dot product followed by a bilinear interpolation:
    $$L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k=1}^{1280} w_k^c \cdot A^k \right)$$
  - Executes in **under 12 ms** on phone CPU with zero external dependencies.
- **Why More Complicated Alternatives Were Unsuitable:**
  - *Embedding PyTorch Mobile or TensorFlow C++ Runtime with Gradient Engine:* Adds 40–60 MB of native binary bloat to the APK, incurs high memory overhead, and causes frequent out-of-memory (OOM) crashes on 3GB RAM devices.
  - *SHAP / Kernel-SHAP:* Requires 500–1000 forward passes per image, taking 15–30 seconds per explanation, making real-time interaction impossible.

---

### 2.3 Objective Skin-Tone Quantification: Individual Typology Angle (ITA) in CIE L\*a\*b\*
- **Implemented Location:** `flutter_app/lib/services/skin_tone_service.dart`, `ml/utils/ita_calculator.py`
- **Selected Technique:** Standard dermatological spectrophotometric formula computed from RGB converted to CIE L\*a\*b\* space:
  $$\text{ITA}^\circ = \arctan\left( \frac{L^* - 50}{b^*} \right) \times \frac{180}{\pi}$$
  Mapped to Fitzpatrick Skin Types:
  - Very Light / Type I: $\text{ITA} > 55^\circ$
  - Light / Type II: $41^\circ < \text{ITA} \le 55^\circ$
  - Intermediate / Type III: $28^\circ < \text{ITA} \le 41^\circ$
  - Tan / Type IV: $10^\circ < \text{ITA} \le 28^\circ$
  - Brown / Type V: $-30^\circ < \text{ITA} \le 10^\circ$
  - Dark / Type VI: $\text{ITA} \le -30^\circ$
- **Why Selected:**
  - Clinically validated and universally recognized in dermatological literature (Del Bino & Bernerd, 2013).
  - Deterministic, completely transparent, and mathematically provable.
  - Executes in **< 3 ms** with zero neural inference overhead.
- **Why More Complicated Alternatives Were Unsuitable:**
  - *Training a Deep Neural Network for Skin-Tone Classification:* High risk of bias; neural networks easily confuse ambient shadow, flash glare, or clinical erythema with constitutional skin pigmentation.

---

### 2.4 Clinical Severity Assessment: Global Acne Grading System (GAGS)
- **Implemented Location:** `flutter_app/lib/services/gags_service.dart`
- **Selected Technique:** Clinical algorithmic scoring following Doshi et al. (1997). The face is partitioned into anatomical zones (forehead, right cheek, left cheek, nose, chin) with clinical lesion weight multipliers ($1 \times \text{comedones} + 2 \times \text{papules} + 3 \times \text{pustules} + 4 \times \text{nodules}$).
- **Why Selected:**
  - Standardized clinical benchmark used by board-certified dermatologists worldwide.
  - Transparent formula understandable by general practitioners and patients alike.
- **Why More Complicated Alternatives Were Unsuitable:**
  - *End-to-End Multitask Segmentation Neural Networks (e.g., Mask R-CNN):* Require dense, pixel-level bounding boxes and polygon masks for hundreds of tiny individual micro-comedones—data that does not exist in public open-access dermatological repositories.

---

### 2.5 Data Warehouse: 4-Tier Medallion File-Based Architecture
- **Implemented Location:** `data/` (`bronze_raw/`, `silver_cleaned/`, `gold_manifests/`, `platinum_assets/`)
- **Selected Technique:** File-system structured Medallion lakehouse using CSV manifests, JSON schemas, and deterministic SHA-256 hashes.
- **Why Selected:**
  - Fully offline, reproducible with zero monetary cost, and version-controllable via Git.
  - Guarantees strict patient-level separation and auditability for university academic defense.
- **Why More Complicated Alternatives Were Unsuitable:**
  - *Cloud Data Warehouses (Databricks, Snowflake, AWS Redshift):* Introduce recurring subscription costs, vendor lock-in, and regulatory data-sovereignty / HIPAA privacy risks associated with uploading patient health data to external cloud servers.

---

## 3. Architectural Summary Matrix

| Capability | Implemented Method | Latency | Memory / Disk | Complexity | Justification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Disease Classification** | MobileNetV2 + Spatial Attention | 18 ms | 6.31 MB | Low–Medium | Fast, accurate, runs offline on standard Android hardware. |
| **Visual Explainability** | Calibrated Vector Grad-CAM | 12 ms | 5.2 KB | Low | Eliminates runtime backprop while providing accurate spatial heatmaps. |
| **Skin-Tone Profiling** | CIE L\*a\*b\* ITA Formula | 2 ms | 0 KB | Very Low | Clinically recognized, deterministic, immune to neural bias. |
| **Severity Scoring** | Algorithmic GAGS Rules | 1 ms | 0 KB | Low | Standard medical scoring system with transparent rules. |
| **Data Lineage** | 4-Tier Medallion Architecture | Offline | Zero cloud cost | Low | Completely reproducible, secure, and inspectable offline. |
