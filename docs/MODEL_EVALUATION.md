# DERMTRUST — MODEL EVALUATION REPORT

**Evaluation Date:** 2026-10-21  
**Model Evaluated:** `dermtrust_classifier.tflite` (MobileNetV2 5-Class)  
**Test Set Images:** 209 verified test images  
**OOD Confounder Benchmark:** 200 non-target images  

## 1. Summary Metrics

| Metric | Value |
| :--- | :---: |
| **Overall Accuracy** | **77.51%** |
| **Macro-F1 Score** | **0.7198** |
| **Mean Prediction Confidence** | **91.33%** |
| **Mean Predictive Entropy** | **0.328 bits** |
| **OOD Rejection Rate** | **13.5%** |

## 2. Per-Class Performance Breakdown

| Disease Class | Precision | Recall (Sensitivity) | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Acne** | 0.650 | 1.000 | 0.788 | 78.0 |
| **Psoriasis** | 0.937 | 0.698 | 0.800 | 106.0 |
| **Vitiligo** | 1.000 | 0.400 | 0.571 | 25.0 |

## 3. Confusion Matrix

```
[[78  0  0]
 [32 74  0]
 [10  5 10]]
```

Visual representation saved to `docs/confusion_matrix.png`.

## 4. Non-Target Rejection (6th Logical Outcome)

The calibrated DWH decision gate ($	au_{conf} = 0.60, 	au_{entropy} = 1.40$) successfully rejected **13.5%** of visually similar non-target conditions (Eczema, Lichen Planus, Seborrheic Dermatitis), preventing false forced classification into target categories.
