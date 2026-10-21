# DERMTRUST — EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) SPECIFICATION

## 1. What is Grad-CAM in Simple English?

Deep neural networks are often criticized as "black boxes" because they return a probability score without showing *why* they reached that conclusion. **Gradient-weighted Class Activation Mapping (Grad-CAM)** solves this by calculating how much each region of the image influenced the neural network's decision.

In DermTrust, Grad-CAM highlights the exact lesion morphology (e.g. chalky depigmentation in Vitiligo, raised silvery scaling in Psoriasis, inflammatory pustules in Acne) in red/warm colors, while non-informative background skin and clothing appear in blue/cool colors.

## 2. On-Device Grad-CAM Architecture

Unlike traditional servers that run heavy PyTorch backpropagation on GPUs, DermTrust implements a **zero-cloud, on-device CAM engine**:
1. **Feature Extractor (`dermtrust_cam.tflite`):** Outputs the last convolutional feature map $F \in \mathbb{R}^{7 \times 7 \times 1280}$.
2. **Precomputed Dense Weights (`cam_weights.json`):** Connects the 1280 channels directly to the class decision.
3. **Linear Combination & Rectification:**
   $$M_c(x, y) = \text{ReLU}\left( \sum_{k=0}^{1279} w_k^c \cdot F(x, y, k) \right)$$
4. **Bilinear Upsampling & Jet Overlay:** The $7 \times 7$ grid is smoothly interpolated to $224 \times 224$ and blended with the camera image at 35% alpha.

## 3. Clinical Diagnostic Support Value

- **Visual Evidence:** Users and physicians can visually verify whether the model is focusing on the actual lesion or being misled by a background distraction (e.g. hair, ruler, clothing).
- **Safety Safeguard:** If an image is classified with moderate confidence but the heatmap activates on healthy skin or clothing rather than a lesion, the prediction is treated with clinical caution.
