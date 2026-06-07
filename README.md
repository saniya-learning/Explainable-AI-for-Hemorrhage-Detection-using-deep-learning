#  Explainable AI for Brain Hemorrhage Detection Using Deep Learning

> Multilabel intracranial hemorrhage detection using ResNet-50 with GradCAM-based explainability on the RSNA 2019 Brain CT dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-RSNA%202019-orange.svg)

---

## 📌 Overview

Intracranial hemorrhage (ICH) is a life-threatening neurological emergency where delayed diagnosis can lead to permanent disability or death. This project presents an **Explainable AI system** for reliable ICH detection that:

- Simultaneously detects **all 5 hemorrhage subtypes** from a single CT scan slice
- Provides **class-specific GradCAM heatmaps** for every prediction
- Is deployed as an **interactive web application** using Gradio
- Runs on **CPU** — no GPU required for inference

---

##  Key Results

| Metric | Value |
|---|---|
| **Validation Macro F1** | **0.729** |
| **Validation ROC-AUC** | **0.929** |
| Best Epoch | 27 / 30 |
| Dataset Size (balanced) | 53,911 images |
| Architecture | ResNet-50 (ImageNet pretrained) |

### Per-Class Performance (Test Set — Multilabel ResNet-50)

| Subtype | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Epidural | 0.526 | 0.740 | 0.615 | 0.951 |
| Intraparenchymal | 0.843 | 0.788 | 0.815 | 0.944 |
| Intraventricular | 0.761 | 0.903 | 0.826 | 0.970 |
| Subarachnoid | 0.690 | 0.703 | 0.697 | 0.889 |
| Subdural | 0.657 | 0.748 | 0.700 | 0.894 |

---

##  Dataset

**RSNA 2019 Intracranial Hemorrhage Detection** ([Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection))

- 752,803 DICOM CT slices with 5 subtype labels per image
- Balanced to **53,911 images** via random downsampling of normal cases
- Split: **70% train / 15% validation / 15% test** (scan-level GroupShuffleSplit — no patient leakage)
- Severe class imbalance: Epidural accounts for < 0.4% of all cases

---

## 🏗️ System Architecture

```
RSNA 2019 DICOMs
      ↓
HU Conversion (pixel × slope + intercept)
      ↓
Triple-Window Stacking [3 × 224 × 224]
  ├─ Brain Window    (C=40,  W=80)
  ├─ Subdural Window (C=75,  W=215)
  └─ Bone Window     (C=600, W=2800)
      ↓
ResNet-50 Backbone (ImageNet1K V2)
  └─ Custom Head: 2048 → 512 → 128 → 5 (Sigmoid)
      ↓
5 Independent Sigmoid Outputs (threshold = 0.5)
      ↓
GradCAM @ layer4[-1].conv3 (per active label)
      ↓
Gradio Web Interface
```

---

## 🔬 Methodology

### Preprocessing — Triple-Window Stacking
Rather than a single grayscale image, three Hounsfield Unit windows are stacked as separate channels — directly replicating the multi-window reading approach used by radiologists:

| Channel | Window | Centre | Width | Purpose |
|---|---|---|---|---|
| R | Brain | 40 HU | 80 HU | Normal tissue contrast |
| G | Subdural | 75 HU | 215 HU | Surface blood collections |
| B | Bone | 600 HU | 2800 HU | Skull structure & acute bleeds |

### Model — Two-Stage Transfer Learning
- **Stage 1 (Warmup):** Backbone frozen, head trained for 6 epochs (LR = 1e-4)
- **Stage 2 (Fine-tuning):** Full model unfrozen for 24 epochs (Backbone LR = 1e-5, Head LR = 1e-4)
- **Loss:** BCEWithLogitsLoss with per-class positive weights to handle class imbalance
- **Optimizer:** AdamW with weight decay

### Explainability — GradCAM
GradCAM is applied independently for each active hemorrhage subtype at `layer4[-1].conv3`, generating separate class-specific heatmaps per prediction.

---

## 📊 Model Comparison

### Multilabel Approach (Final)

| Metric | ResNet-50 | EfficientNet-B0 |
|---|---|---|
| Macro F1 (Val) | **0.7290** | 0.7254 |
| Macro ROC-AUC (Val) | 0.9290 | **0.9306** |
| Macro F1 (Test) | **0.7290** | 0.7188 |
| Macro ROC-AUC (Test) | **0.9290** | 0.9270 |

### Multiclass Approach (Initial — Abandoned)

| Metric | ResNet-50 | ViT-B/16 |
|---|---|---|
| Accuracy (Val) | 77.57% | 69.33% |
| Macro F1 (Val) | 0.78 | 0.68 |

> ⚠️ Multiclass was abandoned because it cannot detect co-occurring hemorrhage subtypes. Over 32,000 scans in RSNA 2019 contain multiple simultaneous subtypes.

---

## 🖥️ Web Interface (Gradio)

The system is deployed as an interactive web app that accepts DICOM or PNG input and produces:
- Brain-window CT scan display
- Per-label probability scores with detection status
- GradCAM heatmap overlay for each detected subtype
- Clinical descriptions of detected conditions

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/brain-hemorrhage-detection-xai.git
cd brain-hemorrhage-detection-xai
pip install -r requirements.txt
```

### Run the Web App
```bash
python app.py
```

### Run Inference on a Single Image
```python
from inference import predict
result = predict("path/to/ct_scan.dcm")
print(result)
```

---

## 📦 Requirements

```
torch>=2.0
torchvision>=0.15
gradio>=3.0
pydicom
opencv-python
numpy
pandas
scikit-learn
matplotlib
```

---

## 📁 Project Structure

```
brain-hemorrhage-detection-xai/
│
├── README.md
├── requirements.txt
├── app.py                  # Gradio web interface
├── inference.py            # Inference pipeline
├── train.py                # Training script
│
├── notebooks/
│   └── hemorrhage_detection.ipynb
│
├── models/
│   └── resnet50_multilabel.pth
│
├── report/
│   └── project_report.pdf
│
└── images/
    ├── architecture.png
    ├── gradcam_samples/
    └── results/
```

---



## 📚 Key References

1. Chilamkurthy et al. (2018) — Deep learning for critical findings in head CT scans, *The Lancet*
2. Selvaraju et al. (2017) — Grad-CAM: Visual explanations from deep networks, *ICCV*
3. He et al. (2016) — Deep residual learning for image recognition, *CVPR*
4. Nishio et al. (2020) — Multi-window technique for CT classification, *Scientific Reports*
5. Lee et al. (2019) — Explainable deep learning for intracranial haemorrhage, *Nature Biomedical Engineering*

---

## ⚠️ Disclaimer

This system is designed for **research and demonstration purposes only**. It is not intended for direct clinical deployment without further clinical validation.
