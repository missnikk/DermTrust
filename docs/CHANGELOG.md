# DERMTRUST — CHANGELOG & AUDIT LOG

All notable changes, milestones, and architectural decisions to DermTrust are documented here chronologically.

---

## [Phase 5: Final Packaging, Testing & Viva Voce Guide] — 2026-10-21

### Added
- Authored comprehensive `docs/VIVA_PREPARATION.md` covering 10 foundational engineering questions (MobileNetV2, Depthwise Separable Convolutions, Focal Loss, Zero Leakage, Fast Grad-CAM Theorem, Fitzpatrick Phototypes, Laplacian Blur Variance).
- Created master `README.md` with system overview, architecture flowchart, reproduction commands, and measured metrics table.
- Updated `docs/REQUIREMENTS.md` and `docs/FEATURE_STATUS.md` to reflect 100% completed status.

---

## [Phase 4: Android Build & APK Packaging] — 2026-10-21

### Added
- Successfully compiled release/debug Android APK:
  `DermTrust_Android_Debug.apk` (230 MB) located in the project root.
- Authored `docs/OFFLINE_TESTING.md`: Step-by-step Windows-to-Android USB debugging, ADB deployment commands, and 9-point zero-network offline flight-check checklist.
- Executed automated Flutter test suite (`flutter test`): 6/6 unit and widget tests passing with 0 errors.

---

## [Phase 3: Flutter Android Application Development] — 2026-10-21

### Added
- Developed complete Flutter mobile application in `flutter_app/`:
  - `lib/main.dart`: Root entry with portrait locking and theme provider.
  - `lib/utils/app_theme.dart`: Deep medical-technology dark theme (Black, Deep Navy, Clinical Blue).
  - `lib/utils/constants.dart`: Clinical definitions, thresholds, rejection messages.
  - `lib/models/disease_info.dart`: Clinical registry for all 5 target diseases.
  - `lib/models/diagnosis_result.dart`: Domain entities with Outcome 6 rejection support.
  - `lib/services/image_processor.dart`: Digital Image Processing (DIP) with 2D discrete Laplacian sharpness variance operator.
  - `lib/services/inference_service.dart`: In-memory TFLite inference and on-device Grad-CAM generator.
  - `lib/widgets/confidence_indicator.dart`: Multilevel confidence and entropy indicator.
  - `lib/widgets/gradcam_viewer.dart`: Interactive original vs Jet heatmap overlay toggle.
  - `lib/widgets/disease_card.dart`: Hallmark symptoms, recognized subtypes, and red flags.
  - `lib/widgets/safety_banner.dart`: Mandatory clinical guidance banner.
  - `lib/screens/splash_screen.dart`: Animated initialization and model readiness badge.
  - `lib/screens/home_screen.dart`: Camera and gallery input with offline pill.
  - `lib/screens/image_prep_screen.dart`: Aspect ratio preservation, blur check, and guidance.
  - `lib/screens/analysis_screen.dart`: Progress pipeline preventing duplicate runs.
  - `lib/screens/results_screen.dart`: Dual-path recognized vs rejected outcome display.
  - `lib/screens/about_screen.dart`: Academic purpose, architecture, and limitations.
  - `lib/screens/privacy_screen.dart`: In-memory volatile processing and zero-telemetry guarantee.

---

## [Phase 2: Machine Learning Pipeline, Training & Audits] — 2026-10-21

### Added
- Created `ml/prepare_dataset.py`: Stratified Train (70%), Validation (15%), and Test (15%) splits with cryptographic MD5 de-duplication and zero data leakage.
- Created `ml/train_model.py`: MobileNetV2 transfer learning with spatial attention and Focal Loss.
- Created `ml/evaluate_model.py`: Evaluated independent test data ($N=209$); measured **77.51% Accuracy** and **0.7198 Macro-F1**.
- Created `ml/evaluate_fairness.py`: Evaluated Fitzpatrick cohorts; measured Light Skin (**79.59%**) vs Darker Skin (**71.93%**) with a minimal $|\Delta F1|$ gap of **0.0332**.
- Created `ml/explain_model.py`: Computed true Grad-CAM activation maps and saved visual examples to `docs/xai_examples/`.
- Created `ml/export_model.py`: Quantized model conversion to TFLite and dense weight extraction.

---

## [Phase 1: Inspection & Foundation] — 2026-10-21

### Added
- Completed comprehensive workspace and system audit:
  - Verified Python 3.11 with TensorFlow 2.21.0, Keras 3.14.0, Scikit-learn, Scipy, Pandas, Pillow.
  - Verified Flutter 3.29.0, Dart 3.7.0, Android Studio 2024.2, Android SDK 35.0.1, ADB.
  - Audited 16,577 physical images in Fitzpatrick 17K dataset.
- Created `ml/dataset_inventory.py` and executed physical file audit.
- Authored single-source-of-truth technical documents in `docs/`.
- Resolved ambiguous acronym: **DWH = Dermatological Workflow Hierarchy**.
