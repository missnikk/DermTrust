# DERMTRUST — INTERFACE SPECIFICATIONS & DATA SCHEMAS

**Document Version:** 1.0.0  
**Project:** Final-Year B.E./B.Tech Information Technology Semester 7 Major Project  
**Author:** Antigravity (Senior Systems Architect)  

---

## 1. Mobile Inference TFLite Interface Contracts

### A. Primary Classifier Model: `dermtrust_classifier.tflite`

- **Model Format:** TensorFlow Lite v3 (Float32 weights-optimized)
- **File Size:** ~2.54 MB
- **Input Tensor Details:**
  - Name: `serving_default_image_input:0`
  - Shape: `[1, 224, 224, 3]`
  - Data Type: `Float32`
  - Color Channel Order: **RGB**
  - Normalization Formula:
    $$x_{\text{norm}} = \frac{x_{\text{uint8}}}{255.0} \quad \in [0.0, 1.0]$$
- **Output Tensor Details:**
  - Name: `StatefulPartitionedCall_1:0`
  - Shape: `[1, 5]`
  - Data Type: `Float32`
  - Activation: **Softmax** ($\sum_{i=0}^4 P_i = 1.0$)
- **Canonical Class Index Mapping:**
  ```json
  {
    "0": "acne",
    "1": "melasma",
    "2": "psoriasis",
    "3": "ringworm",
    "4": "vitiligo"
  }
  ```

---

### B. On-Device Grad-CAM Extractor Model: `dermtrust_cam.tflite`

- **Model Format:** Dual-output TFLite graph
- **File Size:** ~2.53 MB
- **Input Tensor Details:** `[1, 224, 224, 3]` (same as classifier)
- **Output Tensor 0 (Predictions):**
  - Shape: `[1, 5]` (Float32 softmax logits)
- **Output Tensor 1 (Attended Conv Feature Map):**
  - Shape: `[1, 7, 7, 1280]` (Float32 convolutional feature tensor from spatial attention layer)
- **Associated Weight Matrix:** `model_artifacts/cam_weights.json`
  - Contains $5 \times 1280$ float weights connecting each feature channel to the output classes.
- **On-Device Heatmap Computation Algorithm (Grad-CAM):**
  For predicted class $c$:
  $$M_c(x, y) = \text{ReLU}\left( \sum_{k=0}^{1279} w_k^c \cdot F(x, y, k) \right)$$
  Where $F(x, y, k)$ is the $7 \times 7 \times 1280$ feature tensor, and $w_k^c$ is the weight for channel $k$ in class $c$.
  The resulting $7 \times 7$ matrix is normalized to $[0.0, 1.0]$ and bilinearly upsampled to $224 \times 224$ pixels, then colormapped with a Jet/Blue-Red colormap.

---

## 2. Rejection & Uncertainty Decision Schema (DWH Tier 2)

```json
{
  "rejection_rules": {
    "min_confidence_threshold": 0.60,
    "max_shannon_entropy_bits": 1.40,
    "blur_laplacian_var_min": 100.0,
    "min_skin_fraction_pct": 25.0
  },
  "rejection_codes": {
    "ERR_LOW_CONFIDENCE": "Prediction confidence is below clinical decision support threshold (60%). Lesion may be a non-target condition.",
    "ERR_HIGH_ENTROPY": "Model output shows high uncertainty across multiple classes.",
    "ERR_BLURRY_IMAGE": "Input image failed focus/sharpness check. Please retake under stable lighting.",
    "ERR_NON_SKIN_OR_OCCLUDED": "No clear skin lesion detected in frame."
  }
}
```

---

## 3. Flutter Domain Entities & Results Contract

```dart
class DiagnosisResult {
  final bool isRecognized;
  final String? predictedDisease; // "Acne", "Melasma", "Psoriasis", "Ringworm", "Vitiligo"
  final double confidence;        // e.g. 0.88 (88%)
  final double entropy;           // Shannon entropy in bits
  final Map<String, double> classProbabilities; // All 5 class probabilities
  final String? rejectionReason;  // Populated when isRecognized == false
  final Uint8List? gradCamOverlayBytes; // Encoded PNG overlay
  final List<String> clinicalFeatures;  // Visual hallmarks (e.g. "Annular border", "Silvery scale")
  final String safetyDisclaimer; // Mandatory medical guidance
}
```
