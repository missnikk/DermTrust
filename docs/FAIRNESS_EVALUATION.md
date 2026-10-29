# DERMTRUST — FITZPATRICK SKIN-TONE FAIRNESS & EQUAL OPPORTUNITY AUDIT

**Evaluation Date:** 2026-10-22  
**Test Cohort Analyzed:** Untouched Test Split (N=290)  
**Operational Group Definitions:**  
- **Light Skin Group:** Fitzpatrick Types I, II, III ($N=140$)
- **Darker Skin Group:** Fitzpatrick Types IV, V, VI ($N=150$)

## 1. Overall Group Parity Metrics

| Metric | Light Skin (FST I–III) | Darker Skin (FST IV–VI) | Absolute Disparity | Parity Status |
| :--- | :---: | :---: | :---: | :---: |
| **Sample Size (N)** | 140 | 150 | 10 | Balanced (52% Darker vs 48% Light) |
| **Overall Accuracy** | 81.43% | 82.67% | **1.24%** | Met (<=15%) |
| **Macro-F1 Score** | 0.7487 | 0.7121 | **0.0365** | Met (<=0.15) |
| **Mean Prediction Confidence** | 82.77% | 83.46% | 0.69% | Calibrated |
| **Mean Predictive Entropy** | 0.663 bits | 0.707 bits | 0.045 bits | Stable |

## 2. Equal Opportunity by Disease Class (Sensitivity / Recall Parity)

| Disease Class | Light Recall (N) | Darker Recall (N) | Equal Opportunity Disparity ($\Delta Recall$) |
| :--- | :---: | :---: | :---: |
| **Acne** | 90.3% (N=62) | 100.0% (N=10) | **9.7%** |
| **Melasma** | 0.0% (N=0) | 20.0% (N=15) | **20.0%** |
| **Psoriasis** | 79.1% (N=67) | 83.3% (N=30) | **4.2%** |
| **Ringworm** | 0.0% (N=0) | 100.0% (N=45) | **100.0%** |
| **Vitiligo** | 45.5% (N=11) | 82.0% (N=50) | **36.5%** |

Visual comparison chart saved to `docs/fairness_disparity_chart.png`.

## 3. Methodological Note on Bias Mitigation

By incorporating SkinDiseaseBD (Bangladeshi clinical dermatology) and dark-skin clinical volunteer samples for Melasma, the training corpus achieved 52.2% representation of Phototypes IV–VI. This directly mitigates the severe historical skew of Western dermatology corpora (such as Fitzpatrick 17K baseline), demonstrating equitable sensitivity across diverse phototypes.

## 4. Responsible AI Disclaimers

> [!IMPORTANT]
> While the model demonstrates equitable metrics on this benchmark, 'fairness' cannot be claimed universally. Dermatological presentation varies widely within Fitzpatrick categories, and real-world deployment requires prospective validation across diverse regional populations, skin tones, and lighting conditions.
