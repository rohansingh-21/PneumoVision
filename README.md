---
title: PneumoVision
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# PneumoVision — Pneumonia Detection from Chest X-Rays

A deep learning app that detects pneumonia from chest X-rays and explains its predictions using **Grad-CAM** heatmaps — highlighting exactly which lung regions influenced the diagnosis.

---

## Live Demo

Upload a chest X-ray → get a prediction + visual explanation in seconds.

> Built with EfficientNetB0 (transfer learning) + Grad-CAM interpretability

---

## Model Performance

### Final Model: EfficientNetB0 + Fine-Tuning

| Metric | Score |
|---|---|
| Test Accuracy | **89.58%** |
| Test AUC | **95.96%** |
| Pneumonia Recall | **93%** |
| Pneumonia F1 | **0.92** |
| Normal Recall | 82% |
| Normal F1 | 0.85 |

> Threshold tuned to **0.4** (instead of default 0.5) to prioritize pneumonia recall — in medical diagnosis, missing a real case is far more costly than a false alarm.

---

## Model Evolution

This project went through three iterations. Here's how each performed:

### Iteration 1 — Custom CNN (4 Conv layers, trained from scratch)

| Metric | Score |
|---|---|
| Test Accuracy | ~74% |
| Test AUC | ~55% (val AUC collapsed to 0.5 in early epochs) |
| Pneumonia Recall | Poor |
| Training | Stopped early due to unstable validation loss |

**Architecture:** 4× Conv2D → BatchNorm → MaxPool → GlobalAveragePooling → Dense(128) → Dense(1)

**Why it failed:** Training from scratch on ~5,000 images is not enough for a 4-layer CNN to learn robust chest X-ray features. Val AUC dropped to 0.5 in epochs 2–3 (random chance level), meaning the model completely failed to generalize. The training was interrupted manually due to diverging validation loss.

---

### Iteration 2 — EfficientNetB0 (frozen base, no fine-tuning)

| Metric | Score |
|---|---|
| Test Accuracy | **87.18%** |
| Test AUC | **95.27%** |
| Pneumonia Recall | 78% |
| Normal Recall | 94% |

**Why it was better but not great:** Transfer learning from ImageNet weights gave a massive boost. However, the frozen base couldn't adapt its features to X-ray domain, and the model was over-predicting Normal — leading to poor pneumonia recall (78%).

---

### Iteration 3 — EfficientNetB0 + Fine-Tuning + Threshold Tuning ✅

| Metric | Score |
|---|---|
| Test Accuracy | **89.58%** |
| Test AUC | **95.96%** |
| Pneumonia Recall | **93%** |
| Pneumonia F1 | **0.92** |

**What changed:**
- Unfroze last 50 layers of EfficientNetB0 and retrained at `lr=5e-6`
- Added `BatchNormalization` after GlobalAveragePooling
- Increased Dense head from 128→256 neurons, reduced regularization from `l2(0.001)` → `l2(0.0005)`
- Monitored `val_auc` instead of `val_loss` for early stopping
- Lowered prediction threshold from 0.5 → 0.4
- Used separate fresh callbacks for fine-tuning phase

---

## Comparison Summary

| | Custom CNN | EfficientNetB0 v1 | EfficientNetB0 v2 (Final) |
|---|---|---|---|
| Accuracy | ~74% | 87.18% | **89.58%** |
| AUC | ~55% | 95.27% | **95.96%** |
| Pneumonia Recall | Poor | 78% | **93%** |
| Parameters | ~3M | ~5.3M | ~5.3M |
| Training time | Fast | Medium | Medium + Fine-tune |
| Interpretability | ❌ | ✅ Grad-CAM | ✅ Grad-CAM |

---

## How It Works

1. Upload a chest X-ray image
2. EfficientNetB0 classifies it as **NORMAL** or **PNEUMONIA**
3. Grad-CAM generates a heatmap over the last convolutional layer (`top_conv`)
4. The heatmap is overlaid on the X-ray — red/yellow regions are what the model focused on

---

## Dataset

[Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Normal | Pneumonia | Total |
|---|---|---|---|
| Train (80%) | ~1,072 | ~3,101 | 4,173 |
| Val (20%) | ~268 | ~775 | 1,043 |
| Test | 234 | 390 | 624 |

Class imbalance handled via **balanced class weights** during training.

---

## Project Structure

```
pneumonia-detector/
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── model/
│   └── model.keras         # Trained EfficientNetB0 weights
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

---

## Run Locally

```bash
git clone https://huggingface.co/spaces/<your-username>/pneumovision
cd pneumovision
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | EfficientNetB0 (ImageNet pretrained) |
| Framework | TensorFlow / Keras |
| Interpretability | Grad-CAM |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces |
