#  Explainable AI for Reliable ICH Detection
## ResNet-50 Multilabel · RSNA 2019 · GradCAM

---

## Your Results (Are They Good?)

| Metric | Your Score | Context |
|--------|-----------|---------|
| **Val Macro-F1** | **0.729** |  Strong — above 0.70 is considered good for this task |
| **Val ROC-AUC** | **0.929** |  Excellent — >0.90 is clinical-grade |
| Best Epoch | 27 / 30 |  Converged well, no early stopping |
| Train F1 @ epoch 27 | 0.773 | Small train/val gap = healthy, not overfit |

**Verdict: Your results are GOOD.** AUC of 0.929 is genuinely strong for multilabel ICH — published papers in this space often cite 0.90–0.95. The macro-F1 of 0.729 reflects the class imbalance challenge (epidural is very rare at 0.4%).

---

## Project Structure

```
ich_xai/
│
├── models/                          ← PUT YOUR .pth FILE HERE
│   └── best_rsna_multilabel_resnet50_1_.pth
│
├── src/                             ← Core source code
│   ├── __init__.py
│   ├── config.py                    ← All constants (matches Kaggle training)
│   ├── preprocessing.py             ← DICOM → triple-window tensor
│   ├── model.py                     ← ResNet50Multilabel class + loader
│   ├── gradcam.py                   ← GradCAM class + multilabel runner
│   └── inference.py                 ← Predict pipeline (with/without GradCAM)
│
├── static/
│   └── sample_dicoms/               ← Put test DICOM files here
│
├── outputs/                         ← GradCAM images saved here
│
├── app.py                           ←  Gradio web UI (run this for demo)
├── predict.py                       ← CLI tool (terminal inference)
└── requirements.txt
```

---

## Setup in VS Code (Step by Step)

### Step 1 — Open in VS Code
```bash
unzip ich_xai.zip          # if you downloaded the zip
cd ich_xai
code .                     # open in VS Code
```

### Step 2 — Place your model weights
Download from Kaggle Output tab and place in `models/`:
```
models/best_rsna_multilabel_resnet50_1_.pth
```

### Step 3 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```
>  For GPU support, first install PyTorch from https://pytorch.org/get-started/locally/

### Step 5 — Test CLI (optional, verify everything works)
```bash
python predict.py --file static/sample_dicoms/your_file.dcm --gradcam
```

### Step 6 — Launch the demo 
```bash
python app.py
```
Then open **http://localhost:7860** in your browser.

---

## How GradCAM Works in Your Model

```
Input DICOM
    ↓
Triple-Window Preprocessing (Brain + Subdural + Bone channels)
    ↓
ResNet-50 Backbone (layer1 → layer2 → layer3 → layer4)
    ↓                                              ↑
    └──── GradCAM hook on layer4[-1].conv3 ←──────┘
    ↓
Custom Head (2048 → 512 → 128 → 5 logits)
    ↓
Sigmoid → Probabilities [0,1] per subtype
    ↓
Threshold 0.5 → Active labels
    ↓
GradCAM runs ONCE PER ACTIVE LABEL
(each label gets its own heatmap)
```

---

## Demo Script for Presentation

1. **Show architecture slide** — ResNet-50 with 2-stage training
2. **Show training curve** — smooth improvement over 27 epochs
3. **Upload a normal CT** → model shows green, all probs low
4. **Upload a hemorrhage CT** → model shows red, GradCAM highlights bleed area
5. **Explain GradCAM** — "This is WHERE the model looks, not just what it predicts"
6. **Show per-label heatmaps** — different subtypes activate different brain regions

---

## CLI Reference

```bash
# Basic prediction
python predict.py --file path/to/scan.dcm

# Prediction + GradCAM + save images
python predict.py --file path/to/scan.dcm --gradcam --save outputs/

# Custom threshold
python predict.py --file path/to/scan.dcm --threshold 0.4
```
