# DERMTRUST — FEATURE IMPLEMENTATION STATUS

**Last Updated:** Phase 1 (Inspection & Foundation Completed)  
**Status Audit Source:** Automated checks + workspace file inspection  

---

## 1. Feature Dashboard

| Module | Feature | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Data Engine** | Fitzpatrick 17K Ingestion | **[DONE]** | 16,577 physical images audited, 0 missing. |
| **Data Engine** | Target Class Extraction & Counts | **[DONE]** | Acne (518), Psoriasis (706), Vitiligo (166). |
| **Data Engine** | Non-Target Confounders Indexed | **[DONE]** | 1,199 confounder images (Eczema, Lichen Planus, etc.). |
| **Data Engine** | Train/Val/Test Splitter | **[IN_PROGRESS]** | Stratified split script `prepare_dataset.py`. |
| **AI Models** | 5-Class TFLite Classifier | **[DONE]** | MobileNetV2 5-class model in `model_artifacts/`. |
| **AI Models** | On-Device Grad-CAM TFLite | **[DONE]** | `dermtrust_cam.tflite` + `cam_weights.json`. |
| **AI Models** | Rejection / 6th Logical Outcome | **[DONE]** | Dual gate: Confidence $\ge 0.60$, Entropy $\le 1.40$. |
| **Fairness** | Fitzpatrick Light vs Darker Audit | **[IN_PROGRESS]** | Auditing disparity across Groups I–III vs IV–VI. |
| **Flutter App** | Project Initialization & Dependencies | **[IN_PROGRESS]** | Flutter 3.29.0, TFLite runtime, camera, gallery. |
| **Flutter App** | Offline Clinical UI Design System | **[PENDING]** | Dark medical tech theme, 6 responsive screens. |
| **Flutter App** | TFLite Inference & Grad-CAM Service | **[PENDING]** | Pure offline local computation in Dart. |
| **DevOps** | Android USB Debugging & APK Build | **[PENDING]** | Verified SDK 35.0.1, ADB configured. |
| **Academic** | Viva Voce Notes & Report Outline | **[IN_PROGRESS]** | `docs/VIVA_PREPARATION.md`. |
