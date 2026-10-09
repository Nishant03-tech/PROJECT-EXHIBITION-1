# 🫁 RespiraAI: AI-Based Tuberculosis Detection from Chest X-Rays

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Test Accuracy](https://img.shields.io/badge/Accuracy-99.07%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9971-blueviolet.svg)]()
[![Specificity](https://img.shields.io/badge/Specificity-100%25-success.svg)]()

> **Academic Research Prototype • Project Exhibition 1**
>
> RespiraAI is a student research project that explores how deep learning can help identify signs of tuberculosis (TB) in chest X-ray images. It uses DenseNet121 with CBAM attention, Binary Focal Loss, and a lung-focused Grad-CAM view to make the model's predictions easier to inspect.

---

## 📌 Table of Contents

- [Why We Built RespiraAI](#-why-we-built-respiraai)
- [How the Model Works](#-how-the-model-works)
- [Test Results](#-test-results)
- [Understanding the Model's Predictions](#-understanding-the-models-predictions)
- [Chest X-Ray Input Checks](#-chest-x-ray-input-checks)
- [The RespiraAI Web App](#-the-respiraai-web-app)
- [Project Folder Structure](#-project-folder-structure)
- [Installation and Quickstart](#-installation-and-quickstart)
- [Deployment](#-deployment)
- [Important Disclaimer](#-important-disclaimer)
- [References](#-references)

---

## 🔬 Why We Built RespiraAI

Tuberculosis is a serious infectious disease, and finding it early can help people get the care they need sooner. Chest X-rays are often used during TB screening, but reviewing scans takes time and access to trained specialists can be limited in some areas.

For this project, we wanted to explore whether an AI model could help classify chest X-rays as **Normal** or **Tuberculosis**. We also wanted to make its output easier to inspect, rather than showing only a prediction.

A couple of challenges stood out while designing the system:

1. **Learning from the wrong details:** A model may rely on image borders, labels, or scanner text instead of useful patterns in the lungs.
2. **Handling subtle signs:** Some abnormal patterns can be difficult to distinguish from normal lung tissue. A model needs to learn from these harder examples, not just the easiest ones.

RespiraAI combines attention mechanisms, focal loss, and lung-focused visual explanations to explore these challenges. The results in this README are from our project dataset and should not be treated as evidence of clinical performance.

---

## 🚀 How the Model Works

The model processes an input chest X-ray and produces a prediction. Its main components are:

```text
Input Chest X-ray (224 × 224 × 3)
             │
             ▼
     DenseNet121 Backbone
             │
             ▼
       CBAM Attention
     ├── Channel Attention
     └── Spatial Attention
             │
             ▼
     Hybrid Pooling (GAP + GMP)
             │
             ▼
   Classification Head
   (BatchNorm + Dropout 0.35)
             │
             ▼
      Sigmoid Prediction
             │
             ▼
       Binary Focal Loss
```

### Main components

1. **DenseNet121:** We use a pretrained DenseNet121 model to extract features from chest X-rays. Its dense connections allow layers to reuse features learned by earlier layers.
2. **CBAM attention:** The Convolutional Block Attention Module applies channel and spatial attention. In simple terms, it helps the model learn which feature channels and image regions may be useful for the classification task.
3. **Hybrid pooling (GAP + GMP):** Global Average Pooling summarizes features across the image, while Global Max Pooling keeps strong local responses. Both are combined before classification.
4. **Binary Focal Loss:** This loss function gives more weight to difficult examples and reduces the contribution of examples the model already classifies confidently.
5. **Two-stage fine-tuning:** Training starts with the pretrained backbone frozen while the classification head is trained. The top 35 backbone layers are then fine-tuned using a low learning rate of `1e-5`.

The focal-loss settings used in the project are `alpha = 0.25` and `gamma = 2.0`.

---

## 📊 Test Results

The model was evaluated on a held-out test set of **215 chest X-ray images**: 109 labelled Normal and 106 labelled Tuberculosis. The project notes that the split used random seed `42` and exact SHA-256 deduplication.

The results below are the values reported by our evaluation:

| Metric | Original Baseline | RespiraAI | Reported Change |
|---|---:|---:|---:|
| Accuracy | 94.84% | **99.07%** | +4.23 percentage points |
| Precision | 91.96% | **100.00%** | +8.04 percentage points |
| Sensitivity / Recall | 98.10% | **98.11%** | Similar |
| Specificity | 91.67% | **100.00%** | +8.33 percentage points |
| F1-score | 94.93% | **99.05%** | +4.12 percentage points |
| ROC-AUC | 0.9957 | **0.9971** | Higher score |

At the reported classification threshold, the confusion matrix was:

```text
                   Predicted Normal    Predicted TB
Actual Normal             109                0
Actual TB                   2              104
```

In this test set, the model correctly classified all 109 Normal images and identified 104 of the 106 TB images. It missed 2 TB-labelled images.

These numbers describe performance on this particular test set only. The test set is relatively small, and results can change when the model is evaluated on images from different hospitals, equipment, or patient groups. Independent validation would be needed before drawing conclusions about real-world use.

---

## 👁️ Understanding the Model's Predictions

A prediction alone does not show which parts of an image influenced the model. RespiraAI therefore includes a Grad-CAM visualization to help users inspect the regions associated with a prediction.

The implementation includes:

1. **Pre-sigmoid gradients:** Gradients are calculated from the model's output before the sigmoid activation.
2. **Lung-region masking:** Contrast enhancement, adaptive Otsu thresholding, and morphological filtering are used to estimate the left and right lung regions.
3. **A focused heatmap view:** The visualization dims areas outside the estimated lung regions so the viewer can more easily inspect the highlighted areas.

The app can show the original X-ray alongside the lung-masked Grad-CAM result. This is intended to support visual inspection; a heatmap does not prove that the model is focusing on clinically meaningful evidence.

---

## 🛡️ Chest X-Ray Input Checks

RespiraAI includes a set of image checks intended to reject files that are unlikely to be valid chest X-rays. The checks are designed to catch common input mistakes before the model runs.

1. **Non-chest images:** The app checks for features associated with lung regions and rejects images that appear to be scans of hands, feet, skulls, knees, or other body parts.
2. **Screenshots and colour photos:** It checks colour variation that may indicate a desktop screenshot, wallpaper, or regular photograph.
3. **Documents and artificial white blocks:** It checks for large, flat white regions that may occur in documents, receipts, or application windows.
4. **Unusual aspect ratios:** Images with width-to-height ratios outside `0.60` to `1.65` are flagged.

Depending on the check, the app may display a message such as:

- `Input rejected: No anatomical lung fields detected.`
- `Input rejected: Multicoloured elements detected.`
- `Input rejected: Synthetic white block detected.`
- `Input rejected: Unnatural aspect ratio.`

These are heuristic checks, not a guarantee that every invalid image will be rejected or every valid chest X-ray will pass. They should not be considered a substitute for proper clinical image validation.

---

## 💻 The RespiraAI Web App

The project uses Streamlit to provide a simple web interface with five tabs:

- **Home:** An overview of the project, its main components, and the reported metrics.
- **Analysis & Grad-CAM:** Upload an image, use available test examples, view validation results, and compare the original image with its Grad-CAM visualization.
- **Performance Dashboard:** View the ROC curve, confusion matrix, and evaluation metrics.
- **Scan History:** Review the images analysed during the current app session.
- **About & Methodology:** Read about the project, dataset, model approach, and limitations.

---

## 📁 Project Folder Structure

```text
PROEJECT-EXHIBITION---1/
│
├── streamlit_app.py               # Main Streamlit web app
├── model.py                       # DenseNet121, CBAM, and focal loss
├── gradcam.py                     # Lung-masked Grad-CAM visualizations
├── split_data.py                  # Deduplication and stratified data split
├── train_model.py                 # Model training and fine-tuning
├── evaluate_model.py              # Evaluation metrics and plots
├── predict_new_image.py           # Command-line prediction and image checks
├── run_pipeline.py                # Runs the end-to-end pipeline
│
├── best_tb_densenet_model.keras   # Saved trained model (~42 MB)
├── requirements.txt               # Python dependencies
├── .python-version                # Python version for Streamlit Cloud
│
├── new_test_images/               # Example images for the demo
│   ├── normal1.jpeg
│   ├── normal2.jpeg
│   ├── normal3.jpeg
│   ├── tb1.png
│   ├── tb2.png
│   └── tb3.png
│
├── data/                          # Dataset images (720 Normal, 700 TB)
└── TB_Detection_Complete_Colab.ipynb
                                   # Google Colab training notebook
```

---

## ⚙️ Installation and Quickstart

### Run it locally

**1. Clone the repository**

Replace the example repository URL with your actual GitHub repository URL.

```bash
git clone https://github.com/YOUR_USERNAME/tb-detection-ai.git
cd tb-detection-ai
```

**2. Install the dependencies**

Python 3.11 is recommended for this project.

```bash
pip install -r requirements.txt
```

**3. Start the Streamlit app**

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🌐 Deployment

### Deploy with Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Select **New app**.
4. Choose your repository and the `main` branch.
5. Set the main file path to `streamlit_app.py`.
6. Click **Deploy** and wait for the build to finish.

Once deployment succeeds, Streamlit will provide a URL for the app. Make sure the model file and all required dependencies are available in the deployed repository or configured environment.

---

## ⚠️ Important Disclaimer

RespiraAI is an **academic research prototype** developed for a project exhibition.

- Its reported performance is based on the chest X-ray dataset used for this project.
- It has not been established as a clinically validated diagnostic system.
- It does not replace assessment by a qualified healthcare professional, radiologist review, sputum culture, or GeneXpert testing.
- Do not use its predictions to make medical decisions or to delay seeking professional care.

---

## 📚 References

1. Huang et al., **“Densely Connected Convolutional Networks,”** CVPR 2017. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
2. Woo et al., **“CBAM: Convolutional Block Attention Module,”** ECCV 2018. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)
3. Lin et al., **“Focal Loss for Dense Object Detection,”** ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
4. Selvaraju et al., **“Grad-CAM: Visual Explanations from Deep Networks,”** ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
