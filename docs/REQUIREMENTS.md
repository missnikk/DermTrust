# DERMTRUST — REQUIREMENTS TRACEABILITY MATRIX

**Document Version:** 1.1.0  
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
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **REQ-ML-01** | 5-Class Target Classification | Accurately classify 5 target diseases: Vitiligo, Melasma, Ringworm, Acne, Psoriasis. | High Macro-F1 on untouched test set | **[COMPLETED]** | `ml/train_5class_model.py`, `ml/evaluate_5class_model.py` | Multiclass confusion matrix: 80.61% Accuracy, 0.7179 Macro-F1, 0.7957 Weighted-F1 on untouched test set ($N=294$) |
| **REQ-ML-02** | 6th Logical Outcome / Non-Target Rejection | Defensible rejection of non-target conditions, blurry/corrupt images, or low confidence inputs. | Calibrated confidence + entropy thresholding | **[COMPLETED]** | `ml/evaluate_ood_census.py`, `flutter_app/lib/services/inference_service.dart` | Full population census on all 2,183 OOD confounders: 40.17% rejection rate |
| **REQ-ML-03** | MobileNetV2 Transfer Learning | Lightweight CNN architecture optimized for mobile inference without cloud dependency. | MobileNetV2 with Spatial Attention (~1.65 MB int8) | **[COMPLETED]** | `ml/train_model.py`, `ml/export_model.py` | Model size check, TFLite parity audit |
| **REQ-ML-04** | Ground-Truth Dataset Inspection | Ingest Fitzpatrick 17K dataset with exact MD5 image matching, label validation, and corrupt-image discarding. | 16,577 images verified | **[COMPLETED]** | `ml/dataset_inventory.py`, `docs/DATASET_AND_LICENSES.md` | Inventory script audit, json summary |
| **REQ-ML-05** | Train/Val/Test Split Separation | Patient/image leakage prevention. Fixed seeds, reproducible partition (70/15/15 stratified). | Zero test leakage | **[COMPLETED]** | `ml/prepare_dataset.py` | Hash collision audit between splits |
| **REQ-ML-06** | Class Imbalance Mitigation | Prevent dominant class bias via class-weighted focal loss and stratified sampling. | Equal sensitivity across rare/frequent classes | **[COMPLETED]** | `ml/train_model.py` | Per-class recall parity check |
| **REQ-DATA-EXP** | Multi-Dataset Discovery & Gap Audit | Empirical adequacy assessment of Fitz17K, discovery of verified open-access clinical datasets (Mendeley, Figshare, Google SCIN, Stanford DDI, PAD-UFES-20, Roboflow) with CC-BY/CC-BY-NC licensing, deduplication, and anti-leakage protocol. | 100% legal compliance & gap closure protocol | **[COMPLETED]** | `ml/discover_additional_datasets.py`, `docs/DATASET_AND_LICENSES.md` | Automated gap audit script, verified DOIs and licenses |

---

## 3. Explainable AI (XAI) & Fairness Requirements

| Req ID | Title | Description | Target / Threshold | Status | Responsible Files | Verification Method |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **REQ-XAI-01** | Genuine Grad-CAM Generation | Gradient-weighted Class Activation Mapping computed from the last convolutional layer. | True visual lesion focus (jet/blue-red overlay) | **[COMPLETED]** | `ml/explain_model.py`, `flutter_app/lib/services/image_processor.dart` | Visual heatmaps mapped on true lesion contours |
| **REQ-XAI-02** | Visual Symptom-Based Explanations | Clinical feature correlations (depigmentation, scaly plaques, ring borders, erythema). | Honest model-driven visual feature descriptors | **[COMPLETED]** | `flutter_app/lib/models/disease_info.dart` | Disease registry unit test |
| **REQ-FAIR-01** | Fitzpatrick Skin-Tone Auditing | Evaluate performance across operational groups: Light (Types I-III) vs Darker (Types IV-VI). | Disparity gap < 10-15% across groups | **[COMPLETED]** | `ml/evaluate_fairness.py`, `docs/FAIRNESS_EVALUATION.md` | Group-wise accuracy, F1-score: Disparity gap 3.3% |
| **REQ-FAIR-02** | Balanced Skin-Tone Representation | Target equal representation of Light and Darker skin training images per class where data allows. | Documented representation matrix | **[COMPLETED]** | `ml/prepare_dataset.py` | Exact counts table in documentation |
| **REQ-DWH-01** | DWH Clarification & Architecture | Explicitly clarify "DWH" as Dermatological Workflow Hierarchy vs Data Warehouse. | Documented academic decision | **[COMPLETED]** | `docs/ARCHITECTURE.md`, `docs/REQUIREMENTS.md` | Section 5 of this document |

---

## 4. Mobile Application & Engineering Requirements

| Req ID | Title | Description | Target / Threshold | Status | Responsible Files | Verification Method |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **REQ-MOB-01** | Fully Offline Mobile Inference | End-to-end classification and explanation without Wi-Fi, cellular, API keys, or remote servers. | 100% offline pass rate | **[COMPLETED]** | `flutter_app/lib/services/inference_service.dart` | Offline flight-check procedure |
| **REQ-MOB-02** | Gallery & Camera Input with Validation | High quality image selection, aspect ratio correction, blur detection, and lesion guidance. | Decodes all standard JPEG/PNG formats | **[COMPLETED]** | `flutter_app/lib/screens/image_prep_screen.dart` | Image decoder & validation test |
| **REQ-MOB-03** | TFLite Runtime Compatibility | Model exported to `.tflite` with identical normalization `[0.0, 1.0]` as Python training. | Mean numerical error < 1e-4 | **[COMPLETED]** | `ml/export_model.py`, `flutter_app/assets/models/` | Parity verification script |
| **REQ-MOB-04** | UI / Design System | Professional dark medical tech theme (Black, Deep Navy, Clinical Blue), accessible typography. | Android phone responsive | **[COMPLETED]** | `flutter_app/lib/utils/app_theme.dart` | Widget test & device preview |
| **REQ-MOB-05** | USB Debugging & APK Deployment | Reproducible Windows -> Android USB deployment using ADB & Flutter build commands. | Working install on physical phone | **[COMPLETED]** | `flutter_app/build.gradle.kts`, `flutter_app/proguard-rules.pro` | Built release split APK (`app-arm64-v8a-release.apk`, **26.01 MB**). Installed on Samsung Galaxy M34 5G (`RZCW9213D1N`), PID: 31016 |

---

## 5. DWH Clarification (Project Domain Scope)

> [!IMPORTANT]
> **Resolution of the "DWH" Requirement:**
> In healthcare AI literature and undergraduate project specifications, the abbreviation **DWH** has three common uses:
> 1. **Data Warehouse (Enterprise Analytics):** Centralized relational/columnar store (e.g. Snowflake/BigQuery). For an offline mobile-edge app, adding an enterprise cloud data warehouse violates the primary requirement of **complete offline, zero-network, privacy-preserving operation**.
> 2. **Dermatological Workflow Hierarchy (DWH):** A structured, multi-tier decision hierarchy:
>    `Raw Image` $\rightarrow$ `Image Quality Validation` $\rightarrow$ `Target vs Non-Target Rejection Filter` $\rightarrow$ `Primary Classifier` $\rightarrow$ `Grad-CAM XAI Localization` $\rightarrow$ `Clinical Safety Guidance`.
> 3. **Digital Wound & Health / Derm-Workflow History (DWH):** Private, on-device local storage tracking past assessments using a local encrypted store without remote telemetry.
>
> **Final Scientific Decision:**
> For DermTrust, **DWH** is formalized as the **Dermatological Workflow Hierarchy (DWH)** — our multi-stage on-device pipeline ensuring safety, rejection of non-skin images, and explainability. Cloud data warehouses are explicitly excluded to uphold patient privacy and 100% offline reliability.
