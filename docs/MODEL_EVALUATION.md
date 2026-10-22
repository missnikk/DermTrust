# DERMTRUST — 5-CLASS MODEL SCIENTIFIC EVALUATION REPORT

**Evaluation Date:** 2026-10-22  
**Model Architecture:** MobileNetV2 ($\alpha=0.75$) + Spatial Attention + GlobalAveragePooling2D + Dense(5, softmax)  
**Model Artifact:** `model_artifacts/dermtrust_classifier.tflite`  
**Test Set Images (Strictly Partitioned, N=294):** Acne (72), Melasma (15), Psoriasis (100), Ringworm (45), Vitiligo (62)  
**OOD Confounder Benchmark:** 300 non-target images (Eczema, Scabies, Contact Dermatitis, Lichen Planus)  

## 1. Primary Diagnostic Metrics

| Metric | Value |
| :--- | :---: |
| **Overall Accuracy** | **80.61%** |
| **Macro-F1 Score** | **0.7179** |
| **Weighted-F1 Score** | **0.7957** |
| **Mean Target Confidence** | **81.00%** |
| **Mean Target Predictive Entropy** | **0.764 bits** |
| **OOD Confounder Rejection Rate (Outcome 6)** | **43.3%** |

## 2. Per-Class Diagnostic Performance

| Disease Class | Precision | Recall (Sensitivity) | F1-Score | Test Cohort Size |
| :--- | :---: | :---: | :---: | :---: |
| **Acne** | 0.663 | 0.903 | 0.765 | 72 |
| **Melasma** | 0.667 | 0.133 | 0.222 | 15 |
| **Psoriasis** | 0.823 | 0.790 | 0.806 | 100 |
| **Ringworm** | 0.938 | 1.000 | 0.968 | 45 |
| **Vitiligo** | 0.939 | 0.742 | 0.829 | 62 |

## 3. Multiclass Confusion Matrix

```
[[65  0  7  0  0]
 [11  2  1  0  1]
 [16  1 79  2  2]
 [ 0  0  0 45  0]
 [ 6  0  9  1 46]]
```

Visualized in `docs/confusion_matrix.png`.

## 4. Rejection of Visually Similar Confounders (Outcome 6)

The multi-tier DWH decision gate (calibrated $\tau_{conf} = 0.60, \tau_{entropy} = 1.40$) successfully rejected **43.3%** of out-of-distribution dermatology images without forcing classification into target conditions.

## 5. Academic & Clinical Disclaimers

> [!IMPORTANT]
> DermTrust is designed as an educational, explainable decision-support tool. It is **not** a certified medical device and does not claim clinical diagnostic autonomy. Melasma test cohorts are small (N=15) and Ringworm cases are derived from outpatient photography in Bangladesh; clinical validation across diverse global populations is required before healthcare adoption.
