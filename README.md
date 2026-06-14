WEMamba-MDFL: Wavelet-Enhanced Mamba with Multi-Domain Feature Learning for Image Inpainting

This repository contains the official PyTorch implementation of the paper:

> WEMamba-MDFL: Wavelet-Enhanced Mamba with Multi-Domain Feature Learning for Image Inpainting  
 



📌 Overview

Image inpainting aims to fill missing or corrupted regions in images with semantically coherent and visually realistic content. Existing methods often struggle to balance global structural coherence and local texture fidelity, while also suffering from quadratic computational complexity.

WEMamba-MDFL introduces a novel framework that synergistically integrates:
- Discrete Wavelet Transform (DWT) to decompose images into low-frequency (global structure) and high-frequency (local texture) components.
- Low-Frequency Spatial-Channel Mamba (LFSC-Mamba) – a Mamba-based module with spatial & channel attention to reconstruct global structures with **linear complexity**.
- High-Frequency Dynamic Perception (HFDP) – a Fourier-domain dynamic convolution module to synthesize fine-grained, direction‑aware textures.

The proposed method achieves state-of-the-art performance on five benchmark datasets (Places2, Paris Street View, CelebA, CelebA‑HQ, MuralDH) with significantly fewer parameters and FLOPs compared to Transformer‑based approaches.

---

 ✨ Key Features

- **Multi‑Domain Decomposition** – DWT separates structural and textural information for tailored processing.
- **Linear Complexity** – Mamba‑based global modeling avoids the quadratic cost of self‑attention.
- **Dynamic High‑Frequency Enhancement** – Fourier‑domain kernel generation adapts to local texture patterns.
- **Lightweight Design** – Only **3.54M** parameters, yet outperforms heavier models.
- **Comprehensive Evaluation** – Quantitative (PSNR, SSIM, MAE) and qualitative results on five datasets.

---



The model follows a U‑Net encoder‑decoder with stacked Multi‑Domain Feature Learning (MDFL) modules. Each MDFL module:
1. Applies DWT to split features into `LL`, `LH`, `HL`, `HH` subbands.
2. Processes `LL` with LFSC-Mamba .
3. Processes `LH`, `HL`, `HH` with HFDP .
4. Reconstructs via inverse DWT.

---

## 📦 Requirements

- Python 3.8+
- PyTorch 1.12+ (with CUDA)




