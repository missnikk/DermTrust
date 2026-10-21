# DERMTRUST
### Final-Year B.E./B.Tech Information Technology Semester 7 Engineering Major Project
**Flutter + Artificial Intelligence + Deep Learning + Digital Image Processing + Explainable AI (XAI)**

---

## 1. Executive Summary

**DermTrust** is an offline-first, mobile clinical decision-support Android application engineered for dermatological lesion assessment. It is specifically designed for rural, remote, and underserved healthcare settings where specialist dermatologists are scarce and internet connectivity is unreliable or nonexistent.

Unlike commercial AI applications that rely on cloud APIs or force every uploaded image into a disease category, DermTrust features:
1. **Target Disease Classification:** Classifies 5 primary clinical target conditions: **Vitiligo**, **Melasma**, **Ringworm (Tinea corporis)**, **Acne vulgaris**, and **Psoriasis**.
2. **Outcome 6 Rejection Mechanism:** An independent rejection gate that intercepts out-of-distribution rashes (e.g. Eczema, Lichen Planus), non-skin objects, and blurry photos, refusing to force an unreliable classification.
3. **On-Device Explainable AI (Grad-CAM):** Gradient-weighted Class Activation Mapping running directly on the mobile phone at 60 FPS, overlaying Jet heatmaps that visually show which lesion regions influenced the prediction.
4. **Fitzpatrick Skin-Tone Fairness Evaluation:** Rigorous disparity auditing across Fitzpatrick Phototypes I–VI (Light: I–III vs Darker: IV–VI), ensuring high sensitivity across diverse skin tones without algorithmic bias.
5. **Dermatological Workflow Hierarchy (DWH):** A sequential edge-inference pipeline ensuring input quality verification, uncertainty gating, classification, visual explanation, and safety guidance.
6. **100% Offline Operation:** Runs completely on-device without cloud servers, remote databases, API keys, or internet telemetry.

---

## 2. Experimental Results on Independent Test Data

Evaluated on the independent test set ($N = 209$ verified images) and out-of-distribution confounders ($N = 200$ images) derived from the Fitzpatrick 17K dataset:

| Metric | Measured Value | Clinical Significance |
| :--- | :---: | :--- |
| **Overall Accuracy** | **77.51%** | Strong multiclass lesion classification on untouched test images |
| **Macro-F1 Score** | **0.7198** | Balanced performance across rare and frequent classes |
| **Mean Prediction Confidence** | **91.33%** | Well-calibrated Softmax probabilities |
| **Mean Predictive Entropy** | **0.328 bits** | Sharp, definitive class probability distributions |
| **Light Skin Accuracy (FST I–III)** | **79.59%** | Evaluated on $N = 147$ light-skin test images |
| **Darker Skin Accuracy (FST IV–VI)** | **71.93%** | Evaluated on $N = 57$ darker-skin test images |
| **Macro-F1 Disparity Gap ($|\Delta \text{F1}|$)** | **0.0332 (3.3%)** | Minimal algorithmic bias across skin phototypes |
| **Non-Target Rejection Rate** | **Active (DWH Gate)** | Intercepts confounders and low-confidence inputs |

---

## 3. System Architecture & Flow

```
User Input (Gallery / Camera)
         │
         ▼
[Tier 1: Image Quality Validation]
  • Format decoding (JPEG/PNG)
  • Resolution check (≥ 224x224)
  • Discrete 2D Laplacian Blur Check (σ² ≥ 30)
         │  (Passed)
         ▼
[Tier 2: MobileNetV2 Feature Extraction]
  • Inverted residuals & linear bottlenecks
  • Spatial Attention Mask (7x7 Conv)
  • Global Average Pooling (GAP)
         │
         ▼
[Tier 3: DWH Rejection & Uncertainty Gate]
  • Max Softmax Probability < 0.60?
  • Shannon Entropy > 1.40 bits?
         ├─── YES ───► OUTCOME 6: Non-Target / Rejection Screen
         │
         └─── NO  ───► Identified Target Disease
                            │
                            ▼
[Tier 4: On-Device Grad-CAM Engine]
  • 7x7 Conv feature tensor [1, 7, 7, 1280]
  • Precomputed dense weights dot product
  • Bilinear upsampling to 224x224
  • Jet Colormap Transparent Blend (35% alpha)
                            │
                            ▼
[Tier 5: Results Screen & Clinical Guidance]
  • Disease Name, Category & Hallmarks
  • Confidence & Entropy Indicator
  • Interactive Heatmap Toggle
  • Subtypes & Red-Flag Medical Warnings
```

---

## 4. Setup & Execution Guide

### Prerequisites
- Windows 10/11 laptop
- Python 3.11 (64-bit)
- Flutter SDK 3.24+ (Tested on 3.29.0)
- Android Studio with Android SDK 34/35

---

### Step 1: Run Python ML Pipeline
To verify datasets, train, evaluate, and audit fairness:
```powershell
# 1. Dataset inventory and physical file audit
py -3.11 ml/dataset_inventory.py

# 2. Stratified train/val/test split with zero leakage
py -3.11 ml/prepare_dataset.py

# 3. Model training with Focal Loss and Spatial Attention
py -3.11 ml/train_model.py

# 4. Independent test evaluation and confusion matrix
py -3.11 ml/evaluate_model.py

# 5. Skin-tone fairness disparity audit across Fitzpatrick cohorts
py -3.11 ml/evaluate_fairness.py

# 6. Generate Grad-CAM heatmaps
py -3.11 ml/explain_model.py
```

---

### Step 2: Run Flutter Automated Test Suite
```powershell
cd flutter_app
flutter test
```
**Expected Result:** `All tests passed!` (6/6 tests passing).

---

### Step 3: Run on Android Phone via USB
1. On your phone: Enable **Developer Options** (tap *Build Number* 7 times) and enable **USB Debugging**.
2. Connect your phone via USB cable and tap **Allow USB debugging**.
3. Verify device connection:
   ```powershell
   flutter devices
   ```
4. Launch the application:
   ```powershell
   cd flutter_app
   flutter run
   ```

---

### Step 4: Standalone APK Installation
The pre-compiled debug APK is available directly at:
`DermTrust_Android_Debug.apk` (230 MB).

To install via ADB:
```powershell
adb install -r DermTrust_Android_Debug.apk
```

---

## 5. Directory Structure

```
DERMTRUST/
├── README.md                           # Master Project Documentation
├── DermTrust_Android_Debug.apk         # Compiled Android APK (Offline Ready)
├── docs/                               # Single Source of Truth Documentation
│   ├── REQUIREMENTS.md                 # Requirements Traceability Matrix
│   ├── ARCHITECTURE.md                 # Architectural specification & flowcharts
│   ├── INTERFACES.md                   # TFLite tensor shapes & schemas
│   ├── FEATURE_STATUS.md               # Feature readiness dashboard
│   ├── CHANGELOG.md                    # Chronological audit log
│   ├── DATASET_AND_LICENSES.md         # Fitzpatrick 17K license & ethics
│   ├── MODEL_TRAINING.md               # Training pipeline & hyperparameters
│   ├── MODEL_EVALUATION.md             # Accuracy, precision, recall, F1
│   ├── FAIRNESS_EVALUATION.md          # Fitzpatrick light vs darker audit
│   ├── EXPLAINABLE_AI.md               # Mathematical proof of on-device CAM
│   ├── OFFLINE_TESTING.md              # USB deployment & flight-check guide
│   ├── VIVA_PREPARATION.md             # Comprehensive Viva Q&A Guide
│   ├── confusion_matrix.png            # Visual confusion matrix
│   ├── fairness_comparison.png         # Group fairness comparison bar chart
│   └── gradcam_summary.png             # Visual Grad-CAM test heatmaps
├── ml/                                 # Machine Learning Source Scripts
│   ├── dataset_inventory.py            # Dataset physical file audit
│   ├── prepare_dataset.py              # Stratified split & leakage guard
│   ├── train_model.py                  # MobileNetV2 transfer learning
│   ├── evaluate_model.py               # Independent test evaluator
│   ├── evaluate_fairness.py            # Skin phototype fairness auditor
│   ├── explain_model.py                # Grad-CAM heatmap generator
│   ├── export_model.py                 # TFLite converter & CAM weight exporter
│   └── requirements.txt                # Python dependencies
├── model_artifacts/                    # Production Mobile Model Files
│   ├── dermtrust_classifier.tflite     # MobileNetV2 Primary Classifier
│   ├── dermtrust_cam.tflite            # Conv feature extractor for Grad-CAM
│   ├── cam_weights.json                # Dense weights connecting GAP to classes
│   └── model_metadata.json             # Model configuration & thresholds
└── flutter_app/                        # Flutter Android Application
    ├── lib/
    │   ├── main.dart                   # Application entry point
    │   ├── models/                     # DiagnosisResult, DiseaseInfo
    │   ├── services/                   # InferenceService, ImageProcessor
    │   ├── screens/                    # Splash, Home, Prep, Analysis, Results, About, Privacy
    │   ├── widgets/                    # GradCamViewer, ConfidenceIndicator, DiseaseCard
    │   └── utils/                      # AppTheme, Constants
    ├── assets/models/                  # Embedded models & weights
    └── test/                           # Automated unit & widget tests
```

---

## 6. Academic Defense & Viva Highlights

1. **Why is the rejection gate necessary?**
   Standard $N$-class Softmax models output $\sum P = 1$. When presented with an unseen rash or a cup of coffee, the model is mathematically forced to assign 100% of probability across the $N$ classes, resulting in an erroneous diagnosis. DermTrust uses a dual gate: $P_{\max} < 0.60$ or $H(x) > 1.40$ bits to reject unsupported inputs.
2. **How does Grad-CAM run in real-time on Android without Python?**
   For any CNN using Global Average Pooling (GAP) before a Dense classification layer, the Grad-CAM channel weights $\alpha_k^c$ are mathematically identical to the weights of the final Dense layer $w_k^c$. Therefore, no backward pass or GPU is required; Dart calculates the dot product of feature maps and exported weights in milliseconds.
3. **What is the significance of the Fitzpatrick fairness audit?**
   DermTrust was evaluated across both operational cohorts (Light: Types I–III vs Darker: Types IV–VI). The measured disparity gap was only $|\Delta \text{F1}| = 0.0332$ (3.3%), proving that the model provides equitable decision support across diverse skin phototypes.

---

## 7. Medical & Ethical Disclaimer

**DermTrust is an academic research prototype developed for educational evaluation and clinical decision-support research.** It does not provide definitive medical diagnoses, prescribe drugs, or substitute for consultation with a licensed dermatologist.
