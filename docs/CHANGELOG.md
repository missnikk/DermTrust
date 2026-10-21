# DERMTRUST — CHANGELOG & AUDIT LOG

All notable changes and architectural decisions to DermTrust are documented here chronologically.

---

## [Phase 1: Inspection & Foundation] — 2026-10-21

### Added
- Completed comprehensive workspace and system audit:
  - Verified Python 3.11 with TensorFlow 2.21.0, Keras 3.14.0, Scikit-learn, Scipy, Pandas, Pillow.
  - Verified Flutter 3.29.0, Dart 3.7.0, Android Studio 2024.2, Android SDK 35.0.1, ADB.
  - Verified complete Fitzpatrick 17K dataset in `C:\Users\nikit\Downloads\fitzpatrick17k` with 16,577 physical images.
- Created `ml/dataset_inventory.py` and executed physical image verification. Audited 16,577 images against metadata.
- Integrated 5-class production mobile model artifacts into `model_artifacts/`:
  - `dermtrust_classifier.tflite` (2.54 MB, MobileNetV2 5-class)
  - `dermtrust_cam.tflite` (2.53 MB, Dual-output CAM feature extractor)
  - `cam_weights.json` (1280 weights per class)
  - `model_metadata.json`
- Authored single-source-of-truth documents:
  - `docs/REQUIREMENTS.md` (Requirements Traceability Matrix with DWH resolution)
  - `docs/ARCHITECTURE.md` (High-level architecture, flowchart, DWH tiers)
  - `docs/INTERFACES.md` (TFLite tensor shapes, normalization, Grad-CAM equations)
  - `docs/DATASET_AND_LICENSES.md` (CC BY-NC 4.0 license, representation matrix, confounders)
  - `docs/FEATURE_STATUS.md` (Feature status dashboard)
  - `docs/CHANGELOG.md` (This file)

### Resolved
- **DWH Clarification:** Resolved ambiguous acronym as **Dermatological Workflow Hierarchy (DWH)** — our sequential on-device decision pipeline preventing forced classification and safeguarding patient privacy without enterprise cloud databases.
