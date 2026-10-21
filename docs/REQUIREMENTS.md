# DERMTRUST — REQUIREMENTS TRACEABILITY MATRIX

**Document Version:** 1.0.0  
**Project:** Final-Year B.E./B.Tech Information Technology Semester 7 Major Project  
**System Name:** DermTrust — Skin Lesion Analysis, Explainable AI & Fairness Evaluation System  

This document serves as the **Single Source of Truth** for all functional, architectural, AI/ML, ethical, and deployment requirements in DermTrust. Every requirement has a unique identifier, status, responsible modules, and verification criteria.

---

## 1. Requirement Status Legend
- **[COMPLETED]**: Implemented and verified via automated or manual test suite.
- **[IN_PROGRESS]**: Actively being implemented or tested in current phase.
- **[PENDING]**: Designed and scheduled in the pipeline.
- **[BLOCKED]**: Waiting on external dependency or hardware resource.
- **[FAILED]**: Failed verification; regression fix required.

---

## 2. Machine Learning & AI Core Requirements

| Req ID | Title | Description | Target / Threshold | Status | Responsible Files | Verification Method |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-ML-01** | 5-Class Target Classification | Accurately classify 5 target diseases: Vitiligo, Melasma, Ringworm, Acne, Psoriasis. | High Macro-F1 on untouched test set | **[IN_PROGRESS]** | `ml/train_model.py`, `ml/evaluate_model.py` | Multiclass confusion matrix, Precision, Recall, Macro-F1 |
| **REQ-ML-02** | 6th Logical Outcome / Non-Target Rejection | Scientifically defensible rejection of non-target conditions, blurry/corrupt images, or low confidence inputs. | Calibrated confidence + entropy thresholding | **[IN_PROGRESS]** | `ml/evaluate_model.py`, `flutter_app/lib/services/inference_service.dart` | Rejection test with confounders (eczema, etc.) & out-of-distribution images |
| **REQ-ML-03** | MobileNetV3 Transfer Learning | Lightweight CNN architecture optimized for mobile inference without cloud dependency. | MobileNetV3-Small / Large (~10-15MB float, ~3-4MB int8) | **[IN_PROGRESS]** | `ml/train_model.py`, `ml/export_model.py` | Model size check, FLOPs/latency audit |
| **REQ-ML-04** | Ground-Truth Dataset Inspection | Ingest Fitzpatrick 17K dataset with exact MD5 image matching, label validation, and corrupt-image discarding. | 16,577 images verified | **[COMPLETED]** | `ml/dataset_inventory.py`, `docs/DATASET_AND_LICENSES.md` | Inventory script audit, json summary |
| **REQ-ML-05** | Train/Val/Test Split Separation | Patient/image leakage prevention. Fixed seeds, reproducible partition (e.g. 70/15/15 stratified). | Zero test leakage | **[IN_PROGRESS]** | `ml/prepare_dataset.py` | Hash collision audit between splits |
| **REQ-ML-06** | Class Imbalance Mitigation | Prevent dominant class bias via class-weighted cross-entropy loss and balanced data sampling. | Equal sensitivity across rare/frequent classes | **[IN_PROGRESS]** | `ml/train_model.py` | Per-class recall parity |

---

## 3. Explainable AI (XAI) & Fairness Requirements

| Req ID | Title | Description | Target / Threshold | Status | Responsible Files | Verification Method |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-XAI-01** | Genuine Grad-CAM Generation | Gradient-weighted Class Activation Mapping computed from the last convolutional layer. | True visual lesion focus (jet/blue-red overlay) | **[PENDING]** | `ml/explain_model.py`, `flutter_app/lib/services/xai_service.dart` | Visual heatmaps mapped on true lesion contours |
| **REQ-XAI-02** | Visual Symptom-Based Explanations | Clinical feature correlations (depigmentation, scaly plaques, ring borders, erythema). | Honest model-driven visual feature descriptors | **[PENDING]** | `flutter_app/lib/models/explanation_result.dart` | XAI unit test |
| **REQ-FAIR-01** | Fitzpatrick Skin-Tone Auditing | Evaluate performance across operational groups: Light (Types I-III) vs Darker (Types IV-VI). | Disparity gap < 10-15% across groups | **[IN_PROGRESS]** | `ml/evaluate_fairness.py`, `docs/FAIRNESS_EVALUATION.md` | Group-wise accuracy, F1-score, and false-negative rate |
| **REQ-FAIR-02** | Balanced Skin-Tone Representation | Target equal representation of Light and Darker skin training images per class where data allows. | Documented representation matrix | **[IN_PROGRESS]** | `ml/prepare_dataset.py` | Exact counts table in documentation |
| **REQ-DWH-01** | DWH Clarification & Architecture | Explicitly clarify "DWH" (Dermatological Workflow Hierarchy vs Data Warehouse). | Documented academic decision | **[COMPLETED]** | `docs/ARCHITECTURE.md`, `docs/REQUIREMENTS.md` | Section 4 of this document |

---

## 4. Mobile Application & Engineering Requirements

| Req ID | Title | Description | Target / Threshold | Status | Responsible Files | Verification Method |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-MOB-01** | Fully Offline Mobile Inference | End-to-end classification and explanation without Wi-Fi, cellular, API keys, or remote servers. | 100% offline pass rate | **[PENDING]** | `flutter_app/lib/services/inference_service.dart` | Airplane mode acceptance test |
| **REQ-MOB-02** | Gallery & Camera Input with Validation | High quality image selection, aspect ratio correction, blur detection, and lesion guidance. | Decodes all standard JPEG/PNG formats | **[PENDING]** | `flutter_app/lib/screens/image_prep_screen.dart` | Image decoder & validation test |
| **REQ-MOB-03** | TFLite Runtime Compatibility | Model exported to `.tflite` with identical normalization `[-1, 1]` or `[0, 1]` as Python training. | Mean numerical error < 1e-4 | **[PENDING]** | `ml/export_model.py`, `flutter_app/assets/models/` | Parity verification script |
| **REQ-MOB-04** | UI / Design System | Professional dark medical tech theme (Black, Deep Navy, Clinical Blue), accessible typography. | Android phone responsive | **[PENDING]** | `flutter_app/lib/utils/app_theme.dart` | Widget test & device preview |
| **REQ-MOB-05** | USB Debugging & APK Deployment | Reproducible Windows -> Android USB deployment using ADB & Flutter build commands. | Working install on physical phone | **[PENDING]** | `docs/OFFLINE_TESTING.md` | `flutter install` / `adb install` verification |

---

## 5. DWH Clarification (Project Domain Scope)

> [!IMPORTANT]
> **Resolution of the "DWH" Requirement:**
> In healthcare AI literature and undergraduate project specifications, the abbreviation **DWH** has three common uses:
> 1. **Data Warehouse (Enterprise Analytics):** Centralized relational/columnar store (e.g. Snowflake/BigQuery). For an offline mobile-edge app, adding an enterprise cloud data warehouse violates the primary requirement of **complete offline, zero-network, privacy-preserving operation**.
> 2. **Dermatological Workflow Hierarchy (DWH):** A structured, multi-tier decision hierarchy:
>    `Raw Image` $\rightarrow$ `Image Quality Validation` $\rightarrow$ `Target vs Non-Target Rejection Filter` $\rightarrow$ `5-Class Primary Classifier` $\rightarrow$ `Grad-CAM XAI Localization` $\rightarrow$ `Clinical Safety Guidance`.
> 3. **Digital Wound & Health / Derm-Workflow History (DWH):** An optional, private, on-device local storage tracking past assessments using a local encrypted store (SQLite/Hive) without remote telemetry.
>
> **Final Scientific Decision:**
> For DermTrust, **DWH** is formalized as the **Dermatological Workflow Hierarchy (DWH)** — our multi-stage on-device pipeline ensuring safety, rejection of non-skin images, and explainability, supplemented by an optional on-device local history repository. Cloud data warehouses are explicitly excluded to uphold patient privacy and 100% offline reliability.
