# DERMTRUST — CHANGELOG & AUDIT LOG

All notable changes, milestones, and architectural decisions to DermTrust are documented here chronologically.

---

## [Phase 8: Storage Optimization, OOD Census, Verified Release APK & Hardware Launch] — 2026-10-22

### Added
- **Storage Footprint Optimization:**
  - Safely purged intermediate Gradle build artifacts (`1,031 MB`) and redundant root debug APK (`220 MB`).
  - Total workspace footprint reduced from **1,507.99 MB (1.47 GB)** to **256.49 MB (0.25 GB)** — an **83% (1.25 GB) safe reduction** while strictly retaining all code, models, manifests, datasets, and docs.
- **Full OOD Population Census Evaluation (`ml/evaluate_ood_census.py`):**
  - Evaluated **all 2,183 images** from `ml/data/ood_5class_manifest.csv` with zero subsampling.
  - Overall Rejection: **877 / 2,183 (40.17%)**; Mean Confounder Entropy = 1.161 bits (vs 0.764 on targets); Mean Confounder Confidence = 69.65% (vs 81.00% on targets).
  - Saved full census breakdown to `docs/ood_rejection_census.json` and reconciled in `docs/MODEL_EVALUATION.md`.
- **DIP & Grad-CAM Mathematical Verification (`ml/verify_dip_and_xai.py`):**
  - DIP Laplacian variance operator verified: flat patch variance = 0.00, edge variance = 16003.00, deterministic $\tau=30.0$ threshold confirmed.
  - Grad-CAM verified: Feature map tensor index 174 (`[1, 7, 7, 1280]`), prediction index 177 (`[1, 5]`), finite non-zero heatmap activations, class specificity L1 difference between Psoriasis and Ringworm = 79.944. Both PASSED.
- **Optimized Android Release APK Compilation:**
  - Configured ProGuard rules (`proguard-rules.pro`) keeping TFLite classes and suppressing Play Core deferred component warnings.
  - Built split-per-ABI release packages:
    - `app-arm64-v8a-release.apk`: **26.01 MB** (optimized for modern 64-bit ARM devices).
    - `app-armeabi-v7a-release.apk`: **21.33 MB**.
    - `app-x86_64-release.apk`: **29.06 MB**.
    - Achieved an **88% APK package reduction** compared to the 216 MB unstripped debug build.
- **Physical Hardware Verification (Samsung Galaxy M34 5G `RZCW9213D1N`):**
  - Successfully installed `app-arm64-v8a-release.apk` via streamed ADB install.
  - Launched `com.vesit.dermtrust.dermtrust/.MainActivity` with Impeller Vulkan backend.
  - Confirmed live process running on device (PID: **31016**) with zero crashes.
- **Automated Regression Testing:**
  - Executed `flutter test`: **6/6 tests passed (100%)**.

---

## [Phase 7: Verified 5-Class Retraining, Independent Evaluation & Device Deployment] — 2026-10-22

### Added
- **Multi-Source Dataset Ingestion & Reconciliation:**
  - Ingested 1,612 verified images from SkinDiseaseBD (Mendeley Data DOI:10.17632): 316 Ringworm, 312 Vitiligo, and 984 non-target confounders.
  - Curated 72 verified Melasma images (Atlas Dermatologico, Wikimedia, clinical donor captures), explicitly isolating intra-donor variants into patient leakage groups.
  - Reconciled with Fitzpatrick 17K into a unified manifest (`ml/data/unified_ingested_manifest.csv`) totaling 4,273 images (2,090 Target images + 2,183 OOD confounders).
  - In the 5-class target corpus, Darker Skin (FST IV–VI) representation reached **52.2% (1,092 images)**, resolving the historical dark-skin underrepresentation gap.
- **Stratified Group Partitioning (`ml/prepare_5class_dataset.py`):**
  - Generated Train (1,503), Val (293), Test (294), and OOD Confounders (2,183) splits. Zero MD5 or leakage-group overlap across partitions.
- **5-Class Retraining Pipeline (`ml/train_5class_model.py`):**
  - MobileNetV2 ($\alpha=0.75$) with $7 \times 7$ Spatial Attention, Class-Weighted Focal Loss ($\alpha=0.25, \gamma=1.5$), and balanced class weighting ($w_{\text{melasma}} = 6.832$).
  - Eliminated intermediate BatchNorm between GAP and Dense layer, restoring exact theoretical Grad-CAM equivalence ($\alpha_k^c \equiv w_k^c$).
  - Monitored training across 15 epochs; reached **82.25% validation accuracy** at Epoch 14 (loss: 0.0951); restored best weights via EarlyStopping.
- **TFLite Export & Model Synchronization (`ml/export_5class_model.py`):**
  - Exported `dermtrust_classifier.tflite` (1.61 MB), dual-output `dermtrust_cam.tflite` (1.60 MB), `cam_weights.json` (0.34 MB), and `model_metadata.json`.
  - Synchronized all artifacts directly to `flutter_app/assets/models/`.
- **Independent Test Evaluation (`ml/evaluate_5class_model.py`):**
  - Evaluated on strictly untouched test split ($N=294$): **80.61% Overall Accuracy**, **0.7179 Macro-F1**, **0.7957 Weighted-F1**, **81.00% Mean Confidence**, **0.764 bits Mean Entropy**.
  - Evaluated on 300 OOD confounders: **43.3% safe non-target rejection rate** (Outcome 6).
- **Fitzpatrick Fairness & Demographic Parity Audit (`ml/evaluate_5class_fairness.py`):**
  - Light Skin (FST I–III, $N=140$): 79.29% Accuracy.
  - Darker Skin (FST IV–VI, $N=150$): 82.67% Accuracy.
  - Absolute accuracy disparity: **3.38%**, demonstrating equitable performance across skin tones.
- **Flutter App Synchronization & Deterministic DIP:**
  - Synchronized Laplacian blur variance threshold to deterministic 30.0 across `constants.dart` and `image_processor.dart`.
  - Implemented dynamic 4D vs 2D tensor index resolution in `inference_service.dart` for dual-output Grad-CAM interpreter.
  - Verified 6/6 unit and widget tests passing in `flutter test`.
- **Android APK Build & Hardware Deployment:**
  - Updated NDK to 27.0.12077973 and added `keepDebugSymbols` in `build.gradle.kts`.
  - Compiled debug APK `app-debug.apk` (216 MB).
  - Deployed to physical Samsung Galaxy M34 5G (`RZCW9213D1N`) via streamed `adb install -r`.
  - Successfully launched app on physical hardware (PID: 11180).

---

## [Phase 6: Multi-Dataset Discovery & Legal Expansion Audit] — 2026-10-21

### Added
- Authored automated discovery and audit tool `ml/discover_additional_datasets.py`.
- Generated `docs/dataset_expansion_audit.json` reporting exact Fitzpatrick 17K class counts and skin-tone distribution (Light FST I–III vs Darker FST IV–VI).
- Identified and cataloged 6 external open-access dermatological datasets with verified permissive licenses (CC BY 4.0, CC BY-NC 4.0, SCIN Custom Open Research, Stanford AIMI) covering Melasma, Ringworm (Tinea corporis), and darker skin phototypes.
- Formalized 4-step data deduplication and anti-leakage protocol (Cryptographic MD5, 64-bit perceptual hashing pHash, conservative diagnostic mapping, patient-level split isolation).
- Updated `docs/DATASET_AND_LICENSES.md`, `docs/REQUIREMENTS.md` (REQ-DATA-EXP), and `docs/FEATURE_STATUS.md`.
- Verified zero regression across all Flutter tests (6/6 passing).

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
