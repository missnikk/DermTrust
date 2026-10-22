# DERMTRUST — CORRECTED & VERIFIED DATA AUDIT REPORT

**Document Version:** 3.0.0 (Pre-Training Phase 1 Baseline)  
**Verification Date:** 2026-10-22  
**Audit Scope:** Reconciled file-level audit across Fitzpatrick 17K, SkinDiseaseBD, and Clinical Melasma archives.

---

## 1. Executive Reconciled Counts

All counts below are verified against on-disk files and [`ml/data/unified_ingested_manifest.csv`](file:///C:/Users/nikit/OneDrive/Desktop/DERMTRUST/ml/data/unified_ingested_manifest.csv):

| Metric | Verified Value | Evidence / Physical Verification |
| :--- | :---: | :--- |
| **Total Ingested Images on Disk** | **4,273** | 100% physically readable files; 0 missing files. |
| **Target Disease Images (5 Classes)** | **2,090** | Acne (518), Melasma (72), Psoriasis (706), Ringworm (316), Vitiligo (478). |
| **Non-Target Confounders (OOD)** | **2,183** | Fitzpatrick 17K (1,199) + SkinDiseaseBD (984). |
| **Original Clinical Captures** | **1,440** | Unaugmented medical photography. |
| **Augmented Variants** | **650** | Tracked via `is_augmented = True` and bound to parent ID. |
| **Exact Cryptographic Duplicates Purged** | **16** | Identical MD5 hashes across merged archives purged. |
| **Clinically Excluded Files** | **4** | Non-melasma images (PIH, torso) strictly excluded. |

---

## 2. Granular Target Disease Breakdown

```
+---------------+-------+----------------------+--------------------+-----------------------------------------------+
| Disease Class | Total | Original Captures    | Augmented Variants | Primary Contributing Repositories             |
+---------------+-------+----------------------+--------------------+-----------------------------------------------+
| Psoriasis     |   706 | 706 (100.0%)         |   0 (  0.0%)       | Fitzpatrick 17k Dataset (CVPR 2021)           |
| Acne          |   518 | 518 (100.0%)         |   0 (  0.0%)       | Fitzpatrick 17k Dataset (CVPR 2021)           |
| Vitiligo      |   478 | 166 ( 34.7%)         | 312 ( 65.3%)       | Fitzpatrick 17k (166) + SkinDiseaseBD (312)   |
| Ringworm      |   316 |   0 (  0.0%)         | 316 (100.0%)       | SkinDiseaseBD (Mendeley Data DOI 10.17632/v3) |
| Melasma       |    72 |  50 ( 69.4%)         |  22 ( 30.6%)       | Atlas Dermatologico (4), Wiki (20), Local (48)|
+---------------+-------+----------------------+--------------------+-----------------------------------------------+
| TOTAL TARGET  | 2,090 | 1,440 (68.9%)        | 650 ( 31.1%)       | 100% Verified Open / Research Permissible     |
+---------------+-------+----------------------+--------------------+-----------------------------------------------+
```

---

## 3. Provenance & Clinical Label Rigor

### A. SkinDiseaseBD (Ringworm & Vitiligo)
- **Source:** Mendeley Data DOI: `10.17632/9ggd3shdr7.3` (Faridpur Medical College, Bangladesh).
- **Clinical Reality:** In Mendeley Data Version 2/3, the authors collected 197 raw baseline clinical cases and pre-generated 1,612 augmented images of size $512 \times 512$.
- **Rigor Rule:** All 316 Ringworm images are acknowledged as **augmented clinical representations** derived from ~35 distinct patient presentations. They are clustered via perceptual hashing (`pHash`) to ensure related variants never cross train/test splits.

### B. Melasma Cohort (72 Images)
- **Atlas Dermatologico (4 images):** Board-certified dermatologist verified (Dr. Samuel Freire da Silva, Brazil, CC BY-NC 3.0).
- **Wikimedia Commons Medical Archive (20 images):** Sourced from peer-reviewed dermatology atlases and clinical archives published under CC0/CC-BY.
- **Clinical Photography Donor Cohort (48 images):** 26 original captures + 22 flips from 3 clinical volunteer donors (`melasma_donor_1`, `donor_2`, `donor_3`).
- **Rigor Rule:** Donor images are acknowledged as clinical volunteer captures, not board-certified trial cohorts. All images from the same donor are strictly locked to the same split.

---

## 4. Skin-Tone Phototype Auditing

- **Darker Skin Phototypes (Fitzpatrick IV–VI):** **1,092 images (52.2%)**
  - SkinDiseaseBD contributes South Asian phototypes (FST IV–V).
  - Melasma cohort contributes FST IV–V.
  - Fitzpatrick 17K contributes documented darker skin cohorts.
- **Light Skin Phototypes (Fitzpatrick I–III):** **957 images (45.8%)**
  - Fitzpatrick 17K acne, psoriasis, and vitiligo.
- **Unknown / Unspecified:** **41 images (2.0%)**
  - Excluded from group disparity calculations. Never guessed or fabricated.

---

## 5. 5-Class Split Allocation (Zero Data Leakage)

Partitioned via Stratified Group K-Fold (`seed = 42`):

```
+------------+-------------+-------------+-------------+-------------+
| Disease    | Train (70%) | Val (15%)   | Test (15%)  | Total       |
+------------+-------------+-------------+-------------+-------------+
| Acne       |         372 |          74 |          72 |         518 |
| Melasma    |          44 |          13 |          15 |          72 |
| Psoriasis  |         506 |         100 |         100 |         706 |
| Ringworm   |         226 |          45 |          45 |         316 |
| Vitiligo   |         355 |          61 |          62 |         478 |
+------------+-------------+-------------+-------------+-------------+
| TOTAL      |       1,503 |         293 |         294 |       2,090 |
+------------+-------------+-------------+-------------+-------------+
```

*Hash & Group Collision Check:* **0 train-val collisions, 0 train-test collisions, 0 val-test collisions.**
*Manifest Paths:*
- Train: [`ml/data/train_5class_manifest.csv`](file:///C:/Users/nikit/OneDrive/Desktop/DERMTRUST/ml/data/train_5class_manifest.csv)
- Val: [`ml/data/val_5class_manifest.csv`](file:///C:/Users/nikit/OneDrive/Desktop/DERMTRUST/ml/data/val_5class_manifest.csv)
- Test: [`ml/data/test_5class_manifest.csv`](file:///C:/Users/nikit/OneDrive/Desktop/DERMTRUST/ml/data/test_5class_manifest.csv)
- OOD: [`ml/data/ood_5class_manifest.csv`](file:///C:/Users/nikit/OneDrive/Desktop/DERMTRUST/ml/data/ood_5class_manifest.csv)