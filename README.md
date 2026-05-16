
---
title: PneumoVision
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Pneumonia Detector

Detects pneumonia from chest X-rays using EfficientNetB0 with Grad-CAM visualization.

## Model Performance
- Test Accuracy: 87.18%
- Test AUC: 95.78%

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## How it works
1. Upload a chest X-ray image
2. EfficientNetB0 classifies it as NORMAL or PNEUMONIA
3. Grad-CAM highlights the regions that influenced the prediction

## Project Structure
```
pneumonia-detector/
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── download_dataset.py     # Script to download Kaggle dataset
├── README.md               # This file
├── model/
│   └── model.keras         # Trained model weights
└── src/
    ├── data/
    │   └── data_loader.py  # Data generators and class weights
    ├── model/
    │   └── build_model.py  # EfficientNetB0 architecture
    ├── training/
    │   └── train.py        # Training and fine-tuning logic
    ├── evaluation/
    │   └── evaluate.py     # Metrics and confusion matrix
    └── gradcam/
        ├── gradcam.py      # Grad-CAM heatmap generation
        └── display.py      # Heatmap overlay and visualization
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
- Model: EfficientNetB0 (transfer learning)
- Framework: TensorFlow / Keras
- Frontend: Streamlit
- Interpretability: Grad-CAM
