# A Hybrid EfficientNet and Support Vector Machine Approach for Skin Lesion Classification

Code for the manuscript *"A Hybrid EfficientNet and Support Vector Machine Approach for Skin Lesion Classification in Skin Cancer Detection"* (Satpute, Sani, Kawuwa and Ibrahim), submitted to *Discover Computing*.

The pipeline classifies dermoscopic images from the ISIC 2019 dataset into eight lesion classes. Hair artifacts are removed with DullRazor, EfficientNet-B4 and EfficientNet-B5 are fine-tuned, and their pooled features are concatenated and classified by a Support Vector Machine (SVM) with an RBF kernel.

## Repository contents

| File | Description |
| --- | --- |
| `skin_lesion_classification_revised.ipynb` | **Revised pipeline used for the revised manuscript (current version).** Leakage-free lesion-level evaluation, ablations, baselines and statistical tests. |
| `skin_lesion_classification.ipynb` | Original notebook from the first submission, kept unchanged for transparency. Its results are superseded (see below). |

## What changed in the revised version

The revision follows the reviewer's comments. The main corrections are:

- **No data leakage.** The original notebook balanced the data *before* splitting it, so over-sampled copies of minority-class images (DF, VASC) appeared in both the training and the test set. The revised notebook splits the data first, at lesion level, and balances the training split only.
- **Realistic test set.** The validation and test sets keep the natural ISIC 2019 class distribution instead of an artificially balanced 500-image test set.
- **Corrected metrics.** In the original notebook, the ROC curves of EfficientNet-B4 and B5 were computed after the classification heads had been replaced for feature extraction, so they did not reflect class probabilities. In the revised notebook, features are extracted from a separate copy of each model, and all metrics are computed from class probabilities.
- **Additional experiments.** DullRazor on/off ablation, three class-balancing strategies, five classifiers on the same features, four additional baselines, bootstrap confidence intervals, McNemar tests, calibration, inference cost and Grad-CAM.

## Dataset

[ISIC 2019 Skin Lesion Images for Classification](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification) (25,331 dermoscopic images, 8 classes; the UNK class is excluded):

AK (actinic keratosis), BCC (basal cell carcinoma), BKL (benign keratosis), DF (dermatofibroma), MEL (melanoma), NV (melanocytic nevus), SCC (squamous cell carcinoma), VASC (vascular lesion).

## Revised pipeline

1. **Lesion-level split.** A stratified group split (`StratifiedGroupKFold`, seed 42) uses the `lesion_id` field of the ISIC 2019 metadata, so all images of a lesion fall into one subset. Images without a `lesion_id` form their own group (13,931 groups in total). Result: 20,197 training, 2,629 validation and 2,505 test images; validation and test keep the natural class distribution.
2. **DullRazor hair removal** on the original 8-bit image at native resolution (black-hat filter with a 17 × 17 kernel, Otsu threshold, Telea inpainting), followed by resizing to a 456 × 456 cache. A second cache without hair removal is used for the ablation.
3. **Training-set balancing only.** Main strategy: random under/over-sampling of the training split to 625 images per class. Alternatives: all training images with class-weighted cross-entropy, or with class-weighted focal loss (γ = 2). Every strategy uses 5,000 training images per epoch.
4. **Backbone fine-tuning** with `timm` (PyTorch) and ImageNet weights at native input size: EfficientNet-B4 (380 × 380), EfficientNet-B5 (456 × 456); baselines ResNet50, DenseNet121, ConvNeXt-T and Swin-T (224 × 224). AdamW (learning rate 5 × 10⁻⁵, weight decay 10⁻⁴), ReduceLROnPlateau, mixed precision, at most 30 epochs with early stopping on validation balanced accuracy. Augmentation (horizontal flip, rotation up to 30°) is applied to training images only.
5. **Two-stage hybrid classifier.** The fine-tuned B4 and B5 backbones are frozen; their global-average-pooled features (1,792 + 2,048 = 3,840 dimensions) are concatenated, standardised with statistics of the training features, and classified by an RBF-kernel SVM with Platt-scaled probabilities. The SVM and the comparison classifiers (linear SVM, logistic regression, random forest, XGBoost) are tuned on validation balanced accuracy.
6. **Evaluation** on the untouched test set: accuracy, balanced accuracy, macro-F1, macro one-vs-rest ROC-AUC, per-class sensitivity and specificity, expected calibration error, bootstrap 95% confidence intervals (1,000 resamples) and exact McNemar tests.

## Results (revised notebook, test set of 2,505 images)

| Model | Training data / loss | Accuracy | Balanced accuracy | Macro-F1 | Macro AUC |
| --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | Resampled | 0.6391 | 0.5145 | 0.4527 | 0.8962 |
| EfficientNet-B5 | Resampled | 0.6770 | 0.5542 | 0.5029 | 0.8991 |
| **Hybrid B4+B5 + SVM-RBF** | Resampled | **0.7066** | **0.5699** | **0.5422** | **0.9135** |
| EfficientNet-B5 | All images, weighted CE | 0.7238 | 0.6143 | 0.5629 | 0.9339 |
| EfficientNet-B5 | All images, focal loss | 0.7062 | 0.6330 | 0.5206 | 0.9241 |
| Swin-T | Resampled | 0.6387 | 0.5925 | 0.5008 | 0.9112 |

The full table (DullRazor ablation, all classifiers and baselines) is produced in section 8 of the notebook and reported as Table 6 of the revised manuscript.

**Main findings:**

- The hybrid model is significantly more accurate than its EfficientNet-B5 backbone (McNemar p = 3.3 × 10⁻⁵); its balanced accuracy is 0.5699 (95% CI 0.5244–0.6170).
- Other classifiers on the same fused features perform almost identically, so the gain comes mainly from feature fusion rather than from the RBF-SVM itself.
- Training on all images with a class-weighted or focal loss is more effective than random resampling; EfficientNet-B5 with focal loss reaches the highest balanced accuracy (0.6330).
- DullRazor hair removal gives no measurable benefit.
- Melanoma sensitivity of the hybrid model is 0.6688; melanoma and nevus remain the most confused pair of classes.

**Limitations:** results come from a single training seed (42) and a single lesion-level split, and no external dataset was used for validation.

> The original notebook reported 77% test accuracy for the hybrid model. That value was obtained with the data leakage described above and is not valid; please use the revised notebook.

## Reproducing the results

The revised notebook is written for **Kaggle**:

1. Create a Kaggle notebook and import `skin_lesion_classification_revised.ipynb`.
2. Settings: **Accelerator = GPU T4**, **Internet = On** (timm downloads the ImageNet weights).
3. Add the dataset `salviohexia/isic-2019-skin-lesion-images-for-classification` as input.
4. Run all cells. Training runs, features and predictions are cached in `/kaggle/working`, so an interrupted session can be resumed by re-running the cells.

Main dependencies (pre-installed on Kaggle): `torch`, `torchvision`, `timm`, `scikit-learn`, `xgboost`, `opencv-python`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `pillow`, `tqdm`, `joblib`.

## Splits, seeds and settings

Section 12 of the notebook writes the following files to `/kaggle/working/splits/`:

| File | Content |
| --- | --- |
| `train_natural.csv`, `val.csv`, `test.csv` | Image IDs, labels and lesion groups of the lesion-level split |
| `train_resampled_seed42.csv` | Resampled training rows (625 per class) |
| `settings.json` | Split seed, training seeds, sampling settings and the selected classifier hyperparameters |

These files are included in the `splits/` folder of this repository.

## Citation

If you use this code, please cite the manuscript (citation details will be added after publication).
