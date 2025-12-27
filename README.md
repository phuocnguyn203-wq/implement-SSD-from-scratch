# Object Detection from Scratch (SSD)

A Python implementation of the Single Shot MultiBox Detector (SSD) built from scratch. This project demonstrates the core concepts of computer vision, including anchor boxes, loss functions, and dataset handling using Pascal VOC.

![Demo Result](assets/group.png)
*(Result visualization)*

## 📂 Project Structure

```text
├── assets/             # Example images for testing
├── data/               # Dataset (ignored by Git)
├── notebooks/          # Jupyter notebooks for training & experiments
│   ├── training.ipynb  # Main training loop
│   └── hnm.ipynb       # Hard Negative Mining experiments
├── src/                # Source code
│   ├── model.py        # SSD Network Architecture
│   ├── loss.py         # MultiBox Loss implementation
│   └── utils.py        # Encoder/Decoder utilities
└── weights/            # Trained model weights
