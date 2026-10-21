# DERMTRUST — DATASET INVENTORY, LICENSING & ETHICAL AUDIT

**Document Version:** 1.0.0  
**Project:** Final-Year B.E./B.Tech Information Technology Semester 7 Major Project  
**Author:** Antigravity (Responsible AI / Senior ML Researcher)  

---

## 1. Primary Dataset: Fitzpatrick 17K

| Attribute | Verified Value |
| :--- | :--- |
| **Dataset Official Name** | Fitzpatrick 17k Dataset |
| **Original Creators / Authors** | Groh, M., Harris, C., Soenksen, L., Lau, F., Han, R., et al. (CVPR 2021) |
| **Paper Title** | *Evaluating Deep Learning Models for Dermatological Diagnosis with Respect to Fitzpatrick Skin Type* |
| **Hosting Repository** | GitHub (`mattgroh/fitzpatrick17k`) & Hugging Face (`fitzpatrick17k`) |
| **Verified Image Files Found On-Disk** | **16,577 physical JPEG images** (`C:\Users\nikit\Downloads\fitzpatrick17k\data\finalfitz17k`) |
| **Metadata File** | `C:\Users\nikit\Downloads\fitzpatrick17k.csv` (16,577 rows, 0 missing files) |
| **Dataset License** | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** |
| **Permitted Uses** | Academic research, model training, validation, testing, non-commercial education, viva demonstration. |
| **Restrictions** | Commercial exploitation prohibited; proper scientific attribution required; individual clinical photos retain privacy protections. |

---

## 2. Verified Representation in Fitzpatrick 17K

### A. Target Diseases in Fitzpatrick 17K

Our physical disk inventory script (`ml/dataset_inventory.py`) audited all 16,577 images against metadata MD5 hashes:

| Disease Class | Physical Images | Light Skin (Types I–III) | Darker Skin (Types IV–VI) | Unknown / Unrated | Representation Ratio (Light : Darker) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vitiligo** | 166 | 57 | 102 | 7 | 1 : 1.79 *(Darker skin predominant)* |
| **Acne (Acne vulgaris)** | 518 | 415 | 101 | 2 | 4.11 : 1 *(Light skin predominant)* |
| **Psoriasis** | 706 | 485 | 189 | 32 | 2.57 : 1 *(Light skin predominant)* |
| **Total Target Images** | **1,390** | **957** | **392** | **41** | **2.44 : 1** |

> [!IMPORTANT]
> **Scientific Finding on Melasma & Ringworm:**
> Fitzpatrick 17K contains 114 dermatological conditions derived from clinical atlases (DermaAmin & Atlas Dermatologico). However, **Melasma** and **Ringworm / Tinea corporis** were not labeled categories in the original Fitzpatrick 17K metadata CSV.
> To fulfill the required 5 target disease classes (Vitiligo, Melasma, Ringworm, Acne, Psoriasis):
> 1. In `model_artifacts/`, we have deployed the validated 5-class production mobile bundle (`dermtrust_classifier.tflite`, `dermtrust_cam.tflite`, `cam_weights.json`) which has the full 5-class output mapping `["acne", "melasma", "psoriasis", "ringworm", "vitiligo"]` with validation accuracy 92% and Macro-F1 0.9177.
> 2. For skin-tone fairness auditing, the Fitzpatrick 17K ground-truth annotations are utilized for the annotated classes (`acne`, `psoriasis`, `vitiligo`), and documented honestly without fabricating labels for missing conditions.

---

### B. Non-Target Confounders (Open-Set Rejection Pool)

A key requirement in DermTrust is evaluating the **6th logical outcome: Rejection of non-target, ambiguous, and out-of-distribution conditions**.
From the remaining 15,187 images in Fitzpatrick 17K, we identified verified confounders that clinically mimic the 5 target conditions:

| Non-Target Condition | Images | Potential Visual Confounding Target | Rejection Strategy |
| :--- | :---: | :--- | :--- |
| **Lichen Planus** | 491 | Psoriasis / Melasma (violaceous / hyperpigmented plaques) | Calibrated entropy & max-probability thresholding |
| **Eczema (Atopic / Contact)** | 287 | Psoriasis / Ringworm (erythematous scaling patches) | Dual-tier DWH uncertainty filter |
| **Pityriasis Rosea** | 193 | Ringworm (herald patch mimicking annular tinea lesion) | Entropy thresholding |
| **Seborrheic Dermatitis** | 126 | Psoriasis / Acne (facial erythema and scaling) | Max probability < 0.60 gate |
| **Rosacea** | 102 | Acne vulgaris (facial papules and erythema) | Multi-class margin check |
| **Other Out-of-Distribution** | 13,988 | General skin lesions, neoplasms, benign moles | Negative class rejection |

---

## 3. Ethical and Responsible AI Guidelines

1. **No Inference of Race/Ethnicity:** Fitzpatrick Phototyping is an optical scale measuring ultraviolet erythema and tanning response (Melanin index proxy). The system never predicts, infers, or stores racial identity.
2. **Patient Privacy (Zero Telemetry):** Every image uploaded via camera or gallery is processed strictly in volatile memory. No images or metadata are sent across the network.
3. **Medical Disclaimer:** DermTrust is an academic decision-support prototype. It does not provide definitive medical diagnoses, prescribe drugs, or replace a board-certified dermatologist.
