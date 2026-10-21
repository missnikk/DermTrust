# DERMTRUST — SYSTEM ARCHITECTURE & TECHNICAL SPECIFICATION

**Document Version:** 1.0.0  
**Project:** Final-Year B.E./B.Tech Information Technology Semester 7 Major Project  
**Author:** Antigravity (Senior AI / Flutter Systems Architect)  

---

## 1. System Overview & Core Philosophy

**DermTrust** is an offline-first, mobile clinical decision-support and fairness evaluation application for dermatological lesion assessment. It is engineered specifically for rural, remote, and underserved environments where internet connectivity is intermittent or absent.

The system combines:
1. **Lightweight Convolutional Neural Networks (MobileNetV2 / MobileNetV3)** adapted via transfer learning.
2. **Dermatological Workflow Hierarchy (DWH):** A sequential pipeline preventing false-positive forced diagnoses on out-of-distribution, corrupt, or non-skin inputs.
3. **Genuine Explainable AI (XAI):** On-device Gradient-weighted Class Activation Mapping (Grad-CAM) that renders visual heatmaps indicating morphological lesion cues.
4. **Algorithmic Fairness & Bias Evaluation:** Auditing cross-entropy loss, sensitivity, and macro-F1 parity across Fitzpatrick Skin Phototype operational groups: Light (Types I–III) vs Darker (Types IV–VI).
5. **Completely Offline Mobile Runtime:** Zero cloud dependencies, zero external API keys, zero network telemetry.

---

## 2. High-Level Architectural Flow (Mermaid Diagram)

```mermaid
flowchart TD
    subgraph Client ["Client Device: Offline Android Flutter App"]
        A["User Input: Gallery / Camera"] --> B["Image Quality & Preprocessing Engine"]
        B -->|Check Resolution, Blur, Non-Skin| C{"Passes Validation?"}
        C -->|No| R1["Rejection: Unsuitable / Blurry Image"]
        C -->|Yes: Normalized [1, 224, 224, 3]| D["TFLite Mobile Classifier Runtime"]
        
        subgraph DWH ["Dermatological Workflow Hierarchy (DWH)"]
            D --> E["Logits / Softmax Class Probabilities"]
            E --> F{"Rejection Gate:<br/>Max Prob < Threshold OR<br/>Entropy > Max Entropy"}
            F -->|True| R2["Outcome 6: Non-Target Condition / Uncertain"]
            F -->|False| G["Target Disease Identified:<br/>1 of 5 Target Classes"]
        end
        
        G --> H["On-Device Grad-CAM Engine"]
        H -->|Extract Conv Tensor [1, 7, 7, 1280] + CAM Weights| I["Jet/BWR Heatmap Matrix [224, 224]"]
        I --> J["Transparent Overlay on Original Image"]
        
        G --> K["Visual Feature & Safety Guidance Module"]
        J --> L["Clinical Decision-Support Presentation Screen"]
        K --> L
    end
```

---

## 3. The 5 Target Disease Classes + 6th Rejection Outcome

| Class Index | Disease Name | Clinical Category | Visual Morphological Hallmark |
| :---: | :--- | :--- | :--- |
| **0** | **Acne (Acne vulgaris)** | Inflammatory Pilosebaceous | Comedones, erythematous papules, pustules, inflammatory nodules |
| **1** | **Melasma** | Hyperpigmentary Disorder | Irregular brown-to-macular patches with centrofacial / malar distribution |
| **2** | **Psoriasis** | Autoimmune / Hyperkeratotic | Well-demarcated erythematous plaques topped with silvery micaceous scale |
| **3** | **Ringworm (Tinea corporis)** | Superficial Fungal Dermatophytosis | Annular circular lesions with active, scaly borders and central clearing |
| **4** | **Vitiligo** | Autoimmune Depigmentation | Well-circumscribed chalky-white macules/patches lacking melanin |
| **5 (Virtual)** | **Non-Target / Uncertain** | Rejection / Open-Set Fallback | Input is out-of-distribution, low confidence, or resembles non-target condition |

---

## 4. Dermatological Workflow Hierarchy (DWH) Specifications

To avoid forcing every skin lesion into one of the 5 classes:
1. **Tier 1 (Input Validation):** Ensures resolution $\ge 224 \times 224$, standard aspect ratio, luminance check (rejects pitch-black or bleached images), and variance of Laplacian blur check.
2. **Tier 2 (Open-Set Uncertainty Assessment):**
   - Maximum Softmax Probability: $P_{max} = \max_c P(y = c \mid x)$
   - Shannon Entropy: $H(x) = -\sum_{c=1}^5 P(c) \log_2 P(c)$
   - If $P_{max} < \tau_{conf}$ (calibrated default = $0.60$) OR $H(x) > \tau_{entropy}$ (calibrated default = $1.40$ bits), input is routed to **Outcome 6: Rejection / Suspected Non-Target Condition**.
3. **Tier 3 (Primary Classification):** Identifies the target disease from high-confidence activations.
4. **Tier 4 (XAI Verification):** Computes class-specific activation map. If heatmap focuses on uninformative background or clothing, uncertainty flag is raised.
5. **Tier 5 (Decision Support & Guidance):** Displays preliminary assessment, confidence, Grad-CAM visualization, and explicit advice to consult a board-certified dermatologist.

---

## 5. Directory Structure & Component Separation

```
DERMTRUST/
├── docs/                      # Single Source of Truth Technical Docs
│   ├── REQUIREMENTS.md        # Traceability Matrix
│   ├── ARCHITECTURE.md        # System architecture and design
│   ├── INTERFACES.md          # Input/output contracts and schemas
│   ├── FEATURE_STATUS.md      # Feature readiness tracker
│   ├── CHANGELOG.md           # Audit trail of modifications
│   ├── DATASET_AND_LICENSES.md# Dataset inventory, rights, and ethical notes
│   ├── FAIRNESS_EVALUATION.md # Fitzpatrick skin-tone disparity audit
│   ├── OFFLINE_TESTING.md     # Offline flight-check & USB deployment guide
│   └── VIVA_PREPARATION.md    # Viva exam guide, Q&A, and technical defense
├── ml/                        # Python Machine Learning Pipeline
│   ├── dataset_inventory.py   # Ground-truth dataset profile & integrity check
│   ├── prepare_dataset.py     # Stratified split & leakage prevention
│   ├── train_model.py         # MobileNet transfer learning with focal/weighted loss
│   ├── evaluate_model.py      # Independent test evaluation & confusion matrix
│   ├── evaluate_fairness.py   # Fitzpatrick group disparity metrics
│   ├── explain_model.py       # Grad-CAM heatmap generation & verification
│   ├── export_model.py        # TFLite conversion, metadata bundling
│   └── requirements.txt       # Frozen Python dependencies
├── model_artifacts/           # Deployed model artifacts
│   ├── dermtrust_classifier.tflite
│   ├── dermtrust_cam.tflite
│   ├── cam_weights.json
│   └── model_metadata.json
└── flutter_app/               # Cross-Platform Mobile Application
    ├── lib/
    │   ├── main.dart          # Entry point
    │   ├── core/              # Theme, constants, audio/haptic feedback
    │   ├── models/            # Dart domain entities
    │   ├── services/          # TFLite inference, Grad-CAM, image validation
    │   ├── screens/           # UI Screens (Splash, Home, Prep, Analysis, Results, About)
    │   └── widgets/           # Reusable UI widgets
    ├── assets/                # Bundled models, metadata, sample test images
    └── test/                  # Automated unit, widget, and regression tests
```
