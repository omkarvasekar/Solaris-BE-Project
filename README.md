# 🌞 Solaris – Rooftop Solar Power Estimation Using Deep Learning

**Solaris** is a deep learning-based solution that estimates the solar power potential of rooftops using high-resolution aerial imagery. By leveraging a U-Net image segmentation model, it identifies suitable rooftop areas for photovoltaic (PV) panel installation and calculates the expected energy yield based on global solar irradiance data.

---

## 📌 Project Overview

- **Goal**: Automate the identification of usable rooftop areas for solar panels and estimate their solar power yield.
- **Approach**: 
  - Segment rooftops using a U-Net convolutional neural network.
  - Calculate area based on image resolution.
  - Estimate solar yield using GHI (Global Horizontal Irradiance) and panel efficiency.
- **Dataset**: 519 high-resolution images (512×512) of flat rooftops sourced from Sofia Municipality, manually labeled with binary masks.

---

## 🧠 Model Highlights

- **Architecture**: U-Net with encoder-decoder and skip connections.
- **Training**: 
  - Optimizer: Adam
  - Loss: Binary Cross Entropy
  - Epochs: 75
  - Early stopping enabled
- **Performance**:
  - **IoU**: 0.7627 (Test)
  - **Dice Score**: 0.8534 (Test)
  - **Accuracy**: 97.78% (Test)

---

