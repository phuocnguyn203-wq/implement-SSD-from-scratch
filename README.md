# Object Detection from Scratch (SSD)

A Python implementation of the Single Shot MultiBox Detector (SSD) built from scratch. This project demonstrates the core concepts of computer vision, including anchor boxes, loss functions, and dataset handling using Pascal VOC.

# 📂 Project Structure

```text
└── 📁assets
│   └── 📁images
│       ├── cat.png
│       ├── cat2.png
│       ├── cat3.png
│       ├── group.png
│       ├── no_hnm.png
└── 📁notebooks
│   ├── handle_voc.ipynb # preprocessing data for format correction
│   ├── loc_loss.ipynb # implementing location loss of SSF
│   ├── match_loss.ipynb # matching prior boxes with ground truth boxes
│   ├── training.ipynb # Run this file for training
└── 📁src            # Source code
│   ├── loss_fn.py   # SSD Network Architecture
│   ├── model.py     # MultiBox Loss implementation
│   ├── prior_box.py # Encoder/Decoder utilities
│   ├── utils.py     # Trained model weights
```
# Installation

To run this project, clone the repo and install dependencies:

```bash
git clone https://github.com/phuocnguyn203-wq/implement-SSD-from-scratch
cd object-detection-scratch
pip install -r requirements.txt
```
# Data setup
1. Download `VOCdevkit` from Kaggle: [VOCdevkit](https://www.kaggle.com/datasets/wangyuhang3303/vocdevkit)
2. After downloading, extract file into `data/` folder so structure looks like this:
   ```text
    data/
    └── VOCdevkit/
        └── VOC2012/
            ├── JPEGImages/
            ├── Annotations/
            └── ...
   ```
# Training
To train model, run training.ipynb file
   
