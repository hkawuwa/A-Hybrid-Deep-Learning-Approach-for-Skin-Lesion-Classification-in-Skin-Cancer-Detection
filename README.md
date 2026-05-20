# A Hybrid EfficientNet and Support Vector Machine Approach for Skin Lesion Classification

This repository contains a hybrid machine learning pipeline designed for multi-class skin lesion classification using the ISIC 2019 Dataset. The project implements custom artifact pre-processing, data re-balancing, deep feature extraction and fusion from dual Convolutional Neural Networks (EfficientNet B4 and B5), and a downstream Support Vector Machine (SVM) classifier to optimize final diagnostics.

---

## Key Features

* **Data Balancing and Structuring:** Addresses class imbalance via structured sequential under-sampling and over-sampling using the imbalanced-learn library, resulting in a balanced training dataset with 625 images per class.
* **DullRazor Pre-processing:** Automated hair artifact isolation and Telea-based image inpainting using OpenCV morphology to remove distracting visual structures from dermatological images.
* **Deep Feature Fusion:** Dual-model feature extraction utilizing fine-tuned EfficientNet-B4 and EfficientNet-B5 models. Top classification heads are removed to allow multi-model deep feature concatenation.
* **Downstream SVM Classifier:** Employs a Radial Basis Function (RBF) kernel SVM optimized with class weighting to map high-dimensional feature vectors into 8 clinical target spaces.

---

## Dataset and Classes

The model classifies skin lesions across 8 target classes from the ISIC 2019 challenge:
1. AK - Actinic keratosis
2. BCC - Basal cell carcinoma
3. BKL - Benign keratosis (solar lentigines / seborrheic keratoses)
4. DF - Dermatofibroma
5. MEL - Melanoma
6. NV - Melanocytic nevus
7. SCC - Squamous cell carcinoma
8. VASC - Vascular lesions

---

## Pipeline Architecture

1. **Setup and Verification:** Mounts Google Drive, sets recursive directory parsing, and installs dependencies including timm, medpy, and torchmetrics.
2. **Phase I (Data Overhaul):** Filters out unknown data classes, handles balanced random resampling, and executes a stratified 80/10/10 split (Train: 3999, Val: 501, Test: 500).
3. **Phase II (Pre-processing):** Converts inputs to grayscale, applies a Black-Hat morphological kernel operation to isolate hair shafts, builds binary masks, and inpaints the original RGB array.
4. **Phase III (Deep Training and Extraction):** Trains PyTorch backbones using Mixed Precision Autocasting and plateau-driven learning rate schedulers. Extracts raw feature arrays via nn.Identity().
5. **Phase IV (Evaluation):** Trains the RBF SVM and generates performance summaries, confusion matrices, and multi-class One-vs-Rest (OvR) ROC-AUC curves.

---

## Experimental Results

The hybrid pipeline delivers an evaluation performance jump over standalone neural architectures:


| Model Config | Training Accuracy | Testing Accuracy |
| :--- | :---: | :---: |
| EfficientNet B4 Baseline | 91.80% | 67.20% |
| EfficientNet B5 Baseline | 93.15% | 70.00% |
| EfficientNet + SVM (Hybrid Pipeline) | 100.00% | 77.00% |

### Performance Analysis
* **Feature Synergy:** Combining EfficientNet feature variants into a single normalized array provided the SVM with robust, multi-scale visual details, resulting in a +7.00% testing boost over baseline networks.
* **Class Performance:** The system displays high diagnostic reliability on VASC (1.00 recall) and DF classes (0.98 precision). Highly similar visual phenotypes like Melanoma (MEL) and Melanocytic Nevi (NV) remain primary candidates for ongoing boundary optimization.

---

## Requirements and Installation

Install the required environment frameworks via pip:

```bash
pip install torch torchvision timm torchmetrics medpy pandas numpy scikit-learn seaborn matplotlib opencv-python tqdm imbalanced-learn
```

---

## Code Usage

Run the cells inside the notebook file to execute the pipeline end-to-end:

```python
# Configure components or train backbones manually
from models import timm
model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=8)

# Strip classification head for structural hybridization
import torch.nn as nn
model.classifier = nn.Identity()
model.eval()
```
