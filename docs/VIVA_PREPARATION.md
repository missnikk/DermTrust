# DERMTRUST — VIVA VOCE EXAMINATION & TECHNICAL INTERVIEW GUIDE
## Final-Year B.E./B.Tech Information Technology Semester 7 Major Project

This document is engineered to prepare you for your **Final-Year Project Viva Examination**, placement technical interviews, and project defense. Every major concept, formula, architectural choice, and implementation detail is explained in clear, beginner-friendly English with deep engineering rigor.

---

## SECTION 1: HIGH-LEVEL PROJECT FOUNDATIONS

### Q1: What is DermTrust, and what real-world problem does it solve?
**Answer:**
**DermTrust** is an offline-first, mobile clinical decision-support application for dermatological lesion assessment.
In rural, remote, and underserved regions, access to board-certified dermatologists is critically scarce, and internet connectivity is often intermittent or absent. Many commercial AI dermatology tools rely on expensive cloud APIs, transmit sensitive patient photos across the internet, or force every image into one of a few disease categories—even if the image is a blurry photograph or an unrelated skin condition.
DermTrust solves this by running an optimized deep learning model entirely on-device (zero cloud dependencies), providing **Explainable AI (Grad-CAM)** to visually prove why a prediction was made, auditing **algorithmic fairness** across diverse skin phototypes (Fitzpatrick I–VI), and implementing a **Dermatological Workflow Hierarchy (DWH)** that safely rejects non-target and poor-quality images.

---

### Q2: What are the 5 target disease classes and the 6th logical outcome?
**Answer:**
The 5 target clinical conditions are:
1. **Acne (Acne vulgaris):** Inflammatory disorder of the pilosebaceous unit (comedones, papules, pustules).
2. **Melasma:** Acquired hyperpigmentation causing irregular brown patches in sun-exposed facial zones.
3. **Psoriasis:** Immune-mediated chronic disorder marked by well-demarcated erythematous plaques with silvery micaceous scale.
4. **Ringworm (Tinea corporis):** Superficial fungal infection presenting as annular (circular) lesions with active scaly borders and central clearing.
5. **Vitiligo:** Autoimmune depigmentation disorder resulting from selective melanocyte loss, causing chalky-white macules.

**The 6th Logical Outcome:**
**Not Recognized / Non-Target Condition / Rejection.**
Standard classifiers with $N$ output neurons are forced to assign every input image to one of those $N$ classes, resulting in dangerous diagnostic false positives when presented with an unseen rash (e.g. Eczema or Lichen Planus) or a non-skin object. In DermTrust, the 6th outcome is an independent rejection gate that intercepts low-confidence or high-entropy predictions and explicitly declines to force a false diagnosis.

---

### Q3: What does "DWH" mean in DermTrust? Why didn't you add an enterprise cloud Data Warehouse like Snowflake or BigQuery?
**Answer:**
In academic and clinical literature, the acronym DWH has multiple meanings:
1. Enterprise Data Warehouse (cloud database).
2. **Dermatological Workflow Hierarchy (DWH):** A multi-stage clinical decision and triage pipeline.

We resolved this requirement by formalizing DWH as the **Dermatological Workflow Hierarchy**. Adding an enterprise cloud Data Warehouse would violate our primary non-functional requirement: **100% offline, privacy-preserving edge execution without network dependency**.
Our Dermatological Workflow Hierarchy operates in sequential tiers:
- **Tier 1 (Input Quality Gate):** Discrete 2D Laplacian blur check and resolution validation.
- **Tier 2 (Open-Set Uncertainty Assessment):** Dual-gate rejection evaluating calibrated maximum probability ($P_{\max} < 0.60$) and Shannon entropy ($H(x) > 1.40$ bits).
- **Tier 3 (Primary Classification):** MobileNetV2 with spatial attention.
- **Tier 4 (XAI Verification):** On-device Grad-CAM heatmap overlay.
- **Tier 5 (Clinical Triage Guidance):** Displaying triage information, hallmarks, and red flags.

---

## SECTION 2: MACHINE LEARNING & DEEP LEARNING CONCEPTS

### Q4: Why did you choose MobileNetV2 instead of heavier models like ResNet50 or VGG16?
**Answer:**
Standard architectures like ResNet50 (~100 MB, 25M parameters) and VGG16 (~500 MB, 138M parameters) are designed for cloud GPUs. On a mobile phone, they cause thermal throttling, high battery drain, and sluggish latency.
**MobileNetV2** was specifically designed for edge devices:
1. **Depthwise Separable Convolutions:** Splitting standard $3 \times 3$ convolution into a depthwise convolution (filtering one channel at a time) followed by a $1 \times 1$ pointwise convolution (combining channels). This reduces parameters and computational complexity by $>80\%$ with negligible loss of accuracy.
2. **Inverted Residuals & Linear Bottlenecks:** Unlike standard ResNet blocks that compress $\rightarrow$ expand, MobileNetV2 expands to high dimensions in the middle $\rightarrow$ applies depthwise filtering $\rightarrow$ projects back to a low-dimensional linear bottleneck, preserving manifold structure.
3. **Model Footprint:** Our deployed model is only **~1.65 MB (quantized)**, running inference in **~35 milliseconds** on a standard smartphone CPU.

---

### Q5: What is Transfer Learning? Why is it essential for medical imaging?
**Answer:**
Medical skin lesion datasets are relatively small (thousands of images) compared to general computer vision datasets like ImageNet (1.4 million images). If we trained a deep neural network with millions of weights from scratch, it would quickly overfit and memorize background artifacts.
**Transfer learning** begins with a model pretrained on ImageNet that has already learned universal low-level visual features: edge detectors, color gradients, textures, and corner primitives.
We freeze the pretrained feature extractor and adapt only the classification head and top residual layers to skin lesion morphology using **Focal Loss** and **Class Weighting**.

---

### Q6: What is Focal Loss, and why is it superior to standard Categorical Cross-Entropy here?
**Answer:**
Standard Cross-Entropy Loss:
$$\mathcal{L}_{\text{CE}} = -\log(p_t)$$
Where $p_t$ is the model's estimated probability for the true class.
In skin image datasets, many examples are visually distinct and easy to classify ($p_t \approx 0.9$). Even though the individual loss is small, the sheer number of easy examples dominates the gradient updates, drowning out rare or subtle lesion boundaries.
**Focal Loss** (Lin et al., ICCV 2017) introduces a modulating factor $(1 - p_t)^\gamma$:
$$\mathcal{L}_{\text{FL}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- When an example is well-classified ($p_t \to 1$), $(1 - p_t)^\gamma \to 0$, dynamically down-weighting the loss.
- When an example is difficult or misclassified ($p_t \to 0$), $(1 - p_t)^\gamma \approx 1$, preserving full gradient signal.
Setting $\gamma = 1.5$ and $\alpha = 0.25$ forces the network to concentrate learning capacity on ambiguous lesion borders and underrepresented classes.

---

### Q7: How did you prevent Data Leakage between your training, validation, and test sets?
**Answer:**
In clinical dermatology datasets, multiple photographs are often taken of the same patient or lesion from slightly different angles. If images are split randomly, photos of the exact same lesion appear in both the training set and the test set, creating artificial, fabricated test accuracy.
In `ml/prepare_dataset.py`, we implemented:
1. **MD5 Hash De-duplication:** All images are verified and tracked by their cryptographic MD5 hash.
2. **Stratified Splitting:** Partitioning into 70% Train, 15% Validation, and 15% Test while preserving identical class distributions.
3. **Mathematical Set Intersection Assertion:**
   $$\text{Train}_{\text{hashes}} \cap \text{Test}_{\text{hashes}} = \emptyset$$
   Our automated check verifies zero overlap between training and test sets.

---

## SECTION 3: EXPLAINABLE AI (XAI) & GRAD-CAM

### Q8: How does Grad-CAM work mathematically, and how can it run on Android without a Python server?
**Answer:**
**Gradient-weighted Class Activation Mapping (Grad-CAM)** uses the gradient of the class score $y^c$ with respect to feature activation map $A^k$ of the convolutional layer to compute importance weights:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k(i, j)}$$
Then the heatmap is the rectified linear combination:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

**The On-Device Fast Grad-CAM Theorem:**
In our architecture, the feature map $A \in \mathbb{R}^{7 \times 7 \times 1280}$ passes through a **Global Average Pooling (GAP)** layer directly into a Dense layer with weights $w_k^c$:
$$y^c = \sum_{k=1}^{1280} w_k^c \left( \frac{1}{Z} \sum_{i,j} A^k(i, j) \right)$$
Taking the partial derivative:
$$\frac{\partial y^c}{\partial A^k(i, j)} = \frac{1}{Z} w_k^c$$
Substituting back into the Grad-CAM formula:
$$\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{1}{Z} w_k^c = w_k^c$$
**For a GAP + Dense CNN, the Grad-CAM channel weights $\alpha_k^c$ are identically the weights of the final Dense layer $w_k^c$!**
Therefore, on Android:
1. `dermtrust_cam.tflite` extracts the $7 \times 7 \times 1280$ feature map in one forward pass.
2. The app loads precomputed dense weights from `cam_weights.json`.
3. Dart computes the dot-product $\sum_k w_k^c A^k(x, y)$, applies $\text{ReLU}$, and smoothly upsamples to $224 \times 224$ via bilinear interpolation.
This requires **zero backpropagation**, executing in real-time on Android at 60 FPS!

---

## SECTION 4: FAIRNESS AUDITING ACROSS FITZPATRICK SKIN TONES

### Q9: What is the Fitzpatrick Skin Phototype framework, and what were your measured fairness results?
**Answer:**
The **Fitzpatrick Skin Phototype (FST)** scale classifies human skin from Type I to Type VI based on optical response to ultraviolet exposure:
- **Light Skin Operational Cohort:** Types I–III (pale/white to fair skin, burns easily, tans minimally).
- **Darker Skin Operational Cohort:** Types IV–VI (moderate brown to deeply pigmented skin, rarely burns, tans deeply).

**Measured Test Performance from `ml/evaluate_fairness.py`:**
- **Light Skin (FST I–III):** $N = 147$ images, Accuracy = **79.59%**, Macro-F1 = **0.7286**
- **Darker Skin (FST IV–VI):** $N = 57$ images, Accuracy = **71.93%**, Macro-F1 = **0.6955**
- **Accuracy Disparity Gap ($|\Delta \text{Acc}|$):** **7.66%**
- **Macro-F1 Disparity Gap ($|\Delta \text{F1}|$):** **0.0332 (only 3.3%)**

**Per-Class Clinical Insight:**
- **Vitiligo** achieved **42.9% sensitivity on darker skin** versus **40.0% on light skin** because chalky-white depigmentation creates stronger optical contrast against melanin-rich skin.
- **Acne** achieved **100% sensitivity** across both light and dark skin cohorts.

---

## SECTION 5: DIGITAL IMAGE PROCESSING (DIP)

### Q10: How does your discrete 2D Laplacian blur detector work?
**Answer:**
Blurry images destroy high-frequency edge information essential for classifying scaly plaques or pustules.
In `ImageProcessor.computeLaplacianVariance()`, we apply the discrete 2D Laplacian convolution kernel:
$$K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
The Laplacian computes the second spatial derivative of image luminance, highlighting rapid intensity transitions (edges).
We then calculate the statistical variance of the response map:
$$\sigma_L^2 = \frac{1}{N} \sum_{i=1}^N (L_i - \bar{L})^2$$
- A sharp, focused lesion with distinct borders generates large positive and negative Laplacian spikes, yielding a high variance ($\sigma_L^2 > 80$).
- A blurry, out-of-focus image smooths out transitions, causing the variance to drop ($\sigma_L^2 < 30$).
Images with $\sigma_L^2 < 30$ are rejected before inference, preventing garbage-in-garbage-out classifications.
