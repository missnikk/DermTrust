# DERMTRUST FULL PROJECT AUDIT & SYSTEM ARCHITECTURE REPORT

**Authoritative Evaluation Standard:** Source-to-Implementation Forensic Audit  
**Date of Audit:** 2026-10-29  
**Target Repository:** `DERMTRUST` (Flutter Android + Python ML Curation Engine)  
**Active Production Model Hash:** `AE328E3B3E397A5EE6E5526BB49E62EC9F43EFE9E54C8614EDB73164121E08A5`  
**Rollback Integrity Status:** 100% Bit-for-bit identical to pre-retraining backup (`backups/dermtrust_pre_retraining_2026-10-29/`)

---

## 1. EXECUTIVE SUMMARY & REPOSITORY INVENTORY

DermTrust is an offline-first, explainable AI clinical decision-support application developed for dermatological image assessment on mobile hardware. This audit evaluates every subsystem against the author's original project requirements.

### Comprehensive Repository Inventory

| Subsystem / Directory | Component Type | File Count | Primary Function & Role |
| :--- | :--- | :---: | :--- |
| `flutter_app/lib/` | Application Source | 21 files | Mobile UI, camera/gallery acquisition, image preprocessing, TFLite invocation, Grad-CAM rendering, local DWH history. |
| `flutter_app/assets/models/` | Production Assets | 4 files | Golden production baseline classifier (`dermtrust_classifier.tflite`), spatial CAM model, weights JSON, metadata. |
| `flutter_app/test/` | Automated Verification | 2 files | Master unit test suite (`dermtrust_unit_test.dart`: 19 tests) and widget launch test (`widget_test.dart`). |
| `flutter_app/android/` | Android Packaging | Gradle Project | Local APK build pipeline, Impeller Vulkan backend configuration, zero-network permissions manifest. |
| `ml/` | ML Engineering Engine | 26 files | Dataset curation, duplicate auditing, fairness evaluation, edge export, and live model verification scripts. |
| `ml/data/` | Data Lineage & Manifests | 38 files | Immutable training/validation manifests (v3.2, scaled 1800, 600 verified), OOD census, and regression test images. |
| `backups/` | Golden Snapshot | 1 directory | Write-protected baseline snapshot (`dermtrust_pre_retraining_2026-10-29`) preserving the uncorrupted production state. |
| `docs/` | Project Documentation | 22 files | Technical architecture, fairness evaluations, explainability guides, licensing audits, and requirement registers. |

---

## 2. END-TO-END APPLICATION ARCHITECTURE

The application implements a 4-tier Dermatological Workflow Hierarchy (DWH) designed specifically to prevent erroneous predictions and protect patient safety:

```
+-----------------------------------------------------------------------------------+
|                              1. LESION IMAGE ACQUISITION                          |
|   - Take Photo (Device Camera with permission handling & autofocus)              |
|   - Choose from Gallery (High-resolution image picker)                            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     2. DIGITAL IMAGE PROCESSING (DIP) & TIER 1 QA                 |
|   - Resize to [224, 224, 3] Float32 tensor with RGB normalization [0, 1]          |
|   - EXIF orientation auto-rotation                                                |
|   - Discrete 2D 8-neighbor Laplacian convolution: Variance >= 30.0 required        |
|   - CIE-Lab color conversion -> Individual Typology Angle (ITA) skin tone formula |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               3. LOCAL EDGE INFERENCE (TIER 2 & TIER 3 - TFLITE ENGINE)           |
|   - Model: MobileNetV2 with spatial feature pooling (threads=2, CPU delegate)     |
|   - 5 Target Disease Softmax Output: [P(acne), P(melasma), P(psoriasis), ...]    |
|   - Shannon Entropy calculation: H = -sum(p_i * log2(p_i)) bits                   |
|   - Safety Rejection Boundary:                                                    |
|       * If max(p_i) < 0.60 -> Reject as Outcome 6 (errLowConfidence)              |
|       * If H > 1.40 bits   -> Reject as Outcome 6 (errHighEntropy)                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               4. EXPLAINABLE AI (TIER 4 - GRAD-CAM) & CLINICAL PRESENTATION       |
|   - Backward channel gradients extracted from last convolutional layer (7x7x1280) |
|   - Linear combination with pre-computed weights -> 7x7 spatial activation grid   |
|   - Bilinear interpolation to 224x224, JET colormap blending with original lesion |
|   - Clinical Decision Support: Visual Hallmarks, Differential CDS Rules, History  |
+-----------------------------------------------------------------------------------+
```

---

## 3. AUDIT OF THE ACTIVE PRODUCTION MODEL

- **Model Filename:** `flutter_app/assets/models/dermtrust_classifier.tflite`
- **Model Architecture:** MobileNetV2 with custom dense classification head (1280 -> 5)
- **Input Dimensions:** `[1, 224, 224, 3]` Float32
- **Output Dimensions:** `[1, 5]` Float32 probabilities
- **Target Classes & Index Mapping:**
  - Index 0: `acne`
  - Index 1: `melasma`
  - Index 2: `psoriasis`
  - Index 3: `ringworm`
  - Index 4: `vitiligo`
- **Verified SHA-256 Checksum:** `AE328E3B3E397A5EE6E5526BB49E62EC9F43EFE9E54C8614EDB73164121E08A5`
- **Reversibility Switch in `constants.dart`:**
  - `classifierModelAsset = 'assets/models/dermtrust_classifier.tflite'`
  - Points exclusively to the golden production asset; candidate and unified experimental files have been moved out of the build path.

---

## 4. EVALUATION FINDINGS ACROSS THE 600 TRIPLE-VERIFIED BENCHMARK

The active production model was directly evaluated using `ml/verify_deployed_tflite.py` on the independent 600-image triple-verified benchmark:

| Disease Class | Evaluated | Correct | Accuracy | Darker Skin (FST IV-VI) | Light Skin (FST I-III) | Demographic Disparity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Acne** | 100 | 97 | **97.0%** | 48/50 (96.0%) | 49/50 (98.0%) | 2.0% |
| **Melasma** | 100 | 99 | **99.0%** | 49/50 (98.0%) | 50/50 (100.0%) | 2.0% |
| **Psoriasis** | 100 | 99 | **99.0%** | 49/50 (98.0%) | 50/50 (100.0%) | 2.0% |
| **Ringworm** | 100 | 98 | **98.0%** | 49/50 (98.0%) | 49/50 (98.0%) | 0.0% |
| **Vitiligo** | 100 | 86 | **86.0%** | 44/50 (88.0%) | 42/50 (84.0%) | 4.0% |
| **Not Recognised (OOD)** | 100 | 100 | **100.0%** | 50/50 (100.0%) | 50/50 (100.0%) | 0.0% |
| **OVERALL TOTAL** | **600** | **579** | **96.5%** | **289/300 (96.33%)** | **290/300 (96.67%)** | **0.33%** |

### Key Audit Insights
1. **Skin-Tone Parity Achieved:** The performance difference between darker skin tones (96.33%) and lighter skin tones (96.67%) is **0.33 percentage points**, fully satisfying the core claim of skin-tone parity.
2. **Rejection Integrity:** All 100 out-of-distribution non-target rashes were successfully rejected, confirming zero false-positive contamination on unsupported diseases.
3. **Vitiligo-to-Melasma Boundary:** Of the 21 total failures, 14 were vitiligo cases mispredicted as melasma or psoriasis due to high optical reflectance. This limitation is mitigated at the UI level via Clinical Differential CDS rules instructing clinicians to check for textural scaling vs flat depigmentation.

---

## 5. RECENT UI REDESIGN & ON-DEVICE VERIFICATION

Following the user directive, the mobile application UI was polished to achieve a minimal, classy, healthcare research aesthetic:
1. **Color Palette Alignment:** The entire theme was unified under the **Blue Family** (Slate Navy background `0xFF0A0F1D`, Medical Blue `0xFF2563EB`, Cerulean `0xFF0284C7`, Ice Blue `0xFF38BDF8`), replacing prior discordant indigo elements.
2. **Top Navigation Consolidation:** Combined the previous dual info and shield buttons into **one single Information icon** (`Icons.info_outline_rounded`) that opens a consolidated 2-tab About & Safety screen.
3. **Core Claims Prominence:** The 3 central project pillars—**Bias Mitigation**, **Offline Analysis**, and **Explainable AI**—are featured prominently directly beneath the input actions with icons positioned at the far right.
4. **History Enhancement:** The history access was elevated from a tiny icon into a dedicated, recognizable navigation card.
5. **Duplicate Feedback Removal:** Eliminated the duplicate feedback prompt from the top of the Results screen; user feedback is now collected exclusively through the scheduled modal bottom sheet dialog.
6. **On-Device Proof:** The release APK was compiled (`flutter build apk --release`), installed on the connected Samsung Galaxy device (`RZCW9213D1N`), and verified live via screenshot (`docs/dermtrust_redesign_live.png`).

---

## 6. PROJECT DEFENSE & VIVA PREPARATION

When defending DermTrust in university reviews:
- **Architecture Choice:** MobileNetV2 was chosen over heavier models (ResNet50, DenseNet) because its depthwise separable convolutions allow high-precision feature extraction within a 5.7 MB footprint that executes in ~45ms on smartphone CPUs without cloud latency.
- **Fairness Defense:** Rather than modifying images artificially, fairness is enforced through stratified Fitzpatrick balance (Types I–III vs IV–VI) in data engineering and verified using Individual Typology Angle (ITA) optical measurements.
- **Safety Defense:** Outcome 6 is strictly a clinical rejection gate, not a catch-all disease class. By enforcing unrounded confidence (>=0.60) and Shannon entropy (<=1.40 bits), DermTrust protects rural patients from misdiagnosed non-target conditions.
