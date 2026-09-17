# AI-Based Tuberculosis (TB) Detection Using Chest X-Ray Images

An AI-assisted screening system that classifies chest X-ray images as **TB-positive** or **Normal/Non-TB**, using transfer learning with a pretrained CNN and Grad-CAM for explainability.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [Known Limitations](#known-limitations)
- [Future Scope](#future-scope)
- [References](#references)
- [Team](#team)

---

## Overview

Tuberculosis (TB) remains one of the leading infectious causes of death worldwide. Chest X-rays are a common first step in TB screening, but manual interpretation requires trained radiologists and can be time-consuming, especially in resource-limited settings.

This project builds a deep-learning pipeline that:
1. Classifies a chest X-ray as **TB** or **Normal**
2. Provides a confidence score for the prediction
3. Generates a **Grad-CAM heatmap** showing which regions of the X-ray influenced the model's decision
4. Includes a basic input-validation check to flag images that are not likely to be chest X-rays

## Problem Statement

To develop a machine-learning-based system that can analyze chest X-ray images and assist in screening for Tuberculosis by classifying images into TB-positive and non-TB/normal categories, while providing visual explainability for its predictions.

## Features

- ✅ Binary classification: TB vs Normal
- ✅ Transfer learning using a pretrained CNN backbone (DenseNet121)
- ✅ Grad-CAM–based visual explanation of predictions
- ✅ Batch prediction support (test multiple images at once, results exported to CSV)
- ✅ Input validation to reject non-X-ray images (color photos, low-contrast/corrupted images)
- ✅ Evaluation using multiple metrics (not accuracy alone)

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Environment | Google Colab (GPU-enabled) |
| Deep Learning | TensorFlow / Keras |
| Pretrained Model | DenseNet121 (ImageNet weights) |
| Explainability | Grad-CAM |
| Image Processing | OpenCV, NumPy |
| Data Handling | Pandas |
| Visualization | Matplotlib |

## Dataset

This project uses publicly available chest X-ray datasets containing TB-positive and Normal images:

- Tuberculosis (TB) Chest X-ray Database A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Malaysia in collaboration with medical doctors from Hamad Medical Corporation and Bangladesh have created a database of chest X-ray images for Tuberculosis (TB) positive cases along with Normal images. In our current release, there are 700 TB images publicly accessible and 2800 TB images can be downloaded from NIAID TB portal[3] by signing an agreement, and 720 normal images.

## Methodology


Chest X-ray Dataset
        ↓
  Data Collection
        ↓
Data Cleaning & Analysis
        ↓
  Image Preprocessing (resize, normalize)
        ↓
 Train / Validation / Test Split
        ↓
 CNN / Transfer Learning (DenseNet121)
        ↓
    Model Training
        ↓
   Model Evaluation
        ↓
  ┌──────────┴──────────┐
  ↓                      ↓
TB Classification     Grad-CAM
  ↓                      ↓
  └──────────┬───────────┘
        ↓
   Result Display
        ↓
 AI-Assisted Screening Output
```

## Project Structure

```
├── data/
│   ├── TB/                     # TB-positive images
│   └── Normal/                 # Normal images
├── new_test_images/            # Unseen images for external testing
├── notebooks/
│   └── tb_detection.ipynb       # Main Colab notebook (training + inference)
├── outputs/
│   └── new_image_predictions.csv
├── README.md
└── requirements.txt


**`requirements.txt`**
```
tensorflow
numpy
opencv-python
matplotlib
pandas
scikit-learn
```

## Usage

### 1. Train the model
Run the training cells in the notebook to load the dataset, preprocess images, and fine-tune DenseNet121 on the TB/Normal classification task.

### 2. Predict on a single new image
```python
predict_new_image_checked("test_xray.jpg")
```
Output includes the predicted label (TB/Normal), confidence score, and the displayed X-ray image.



### 4. Input validation
Before prediction, `is_likely_xray()` checks whether the uploaded image is plausibly a grayscale chest X-ray (based on color-channel similarity and contrast) and rejects unsuitable images (e.g., regular photographs) with a clear message.

## Model Evaluation

The model is evaluated using multiple metrics rather than accuracy alone, given the class-imbalance and high-stakes nature of medical screening:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-Score
- ROC-AUC
- Confusion Matrix

|              | Predicted TB | Predicted Normal |
|--------------|:------------:|:-----------------:|
| **Actual TB**     | TP | FN | 99 | 9
| **Actual Normal** | FP | TN | 2  | 103


## Explainable AI (Grad-CAM)

To make the model's predictions interpretable, Grad-CAM is used to generate a heatmap over the original X-ray, highlighting the regions that most influenced the TB/Normal decision.

**Grad-CAM workflow:**
```
Chest X-ray → Trained CNN → Prediction → Gradients (final conv layer)
→ Feature Map Weights → Heatmap → Overlay on original X-ray
```

Planned/possible improvements:
- Grad-CAM++ for sharper, multi-region localization
- Comparing heatmaps across multiple convolutional layers
- Quantitative validation (IoU against annotated lesion regions, where available)
- Comparing heatmaps for correct vs. incorrect predictions to diagnose model errors

## Known Limitations

- Trained on a relatively small, balanced dataset — performance on truly external data may vary.
- Only accepts chest X-ray images; non-X-ray inputs (e.g., regular photos) are rejected by the input-validation check, not classified.
- Not validated against radiologist-confirmed ground truth beyond dataset labels.
- Intended strictly as a **screening aid / academic prototype**, not a diagnostic system.

## Future Scope

- Compare multiple deep-learning architectures (ResNet, EfficientNet, Vision Transformers)
- Increase dataset size and diversity across imaging sources/populations
- Improve generalization via external validation datasets
- Add Grad-CAM++ or other explainability techniques
- Deploy as a simple web app (Streamlit/Gradio) for demonstration
- Extend to multi-disease chest X-ray screening  

## References/Research papers

1. Lakhani, P., & Sundaram, B. (2017). Deep Learning at Chest Radiography: Automated Classification of Pulmonary Tuberculosis by Using Convolutional Neural Networks. *Radiology*, 284(2), 574–582.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV*.
3. Comparative study of deep learning models for binary classification on the combined Montgomery + Shenzhen chest X-ray dataset. *arXiv:2309.10829*.
4. Deep learning models for tuberculosis detection and infected region visualization in chest X-ray images. *Intelligent Medicine*.

## Team

**Group No. 14 — B.Tech CSE Core, Fall Semester 2026-27**
Supervisor: Dr. Rajneesh Kumar Patel

| Name | Registration No. |
|---|---|
| Nishant Mishra | 25BCE10488 |
| Jayesh Sule | 25BCE10731 |
| Samay Jindal | 25BCE10491 |
| Aditya Namdev | 25BCE10297 |
| Ritvik Mahajan | 25BCE10509 |
