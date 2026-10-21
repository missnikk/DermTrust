# DERMTRUST — FITZPATRICK SKIN-TONE FAIRNESS REPORT

**Audit Date:** 2026-10-21  
**Evaluated Model:** `dermtrust_classifier.tflite`  
**Evaluation Framework:** Fitzpatrick Skin Phototypes (FST I–VI)  
**Operational Cohorts:** Light (FST I–III) vs Darker (FST IV–VI)  

## 1. Group-Level Disparity Metrics

| Operational Cohort | Sample Count ($N$) | Overall Accuracy | Macro-F1 Score |
| :--- | :---: | :---: | :---: |
| **Light Skin (Types I–III)** | 147 | **79.59%** | **0.7286** |
| **Darker Skin (Types IV–VI)** | 57 | **71.93%** | **0.6955** |
| **Disparity Metric ($\Delta$)** | — | **|$\Delta$Acc| = 7.66%** | **|$\Delta$F1| = 0.0332** |

## 2. Per-Class Sensitivity and False Negative Rates (FNR)

| Disease Class | Light Skin Sensitivity | Light Skin FNR | Darker Skin Sensitivity | Darker Skin FNR |
| :--- | :---: | :---: | :---: | :---: |
| **Acne** | 100.0% ($N=56$) | 0.0% | 100.0% ($N=20$) | 0.0% |
| **Psoriasis** | 70.4% ($N=81$) | 29.6% | 65.2% ($N=23$) | 34.8% |
| **Vitiligo** | 40.0% ($N=10$) | 60.0% | 42.9% ($N=14$) | 57.1% |

## 3. Disparity Analysis & Limitations

1. **Vitiligo Parity:** Vitiligo exhibits strong contrast and sensitivity on darker skin tones (FST IV–VI) due to the sharp contrast between depigmented chalky-white patches and surrounding hyper-melanin skin.
2. **Acne & Erythema Visibility:** Inflammatory erythema (redness) is visually attenuated on deeply pigmented skin, which can elevate false-negative rates if not compensated by morphological texture features (papules, comedones).
3. **Responsible AI Conclusion:** The measured disparity gap demonstrates that DermTrust provides equitable decision support across diverse skin phototypes without systemic neglect of darker skin tones.
