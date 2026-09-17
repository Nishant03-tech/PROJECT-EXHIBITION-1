# Tuberculosis Chest X-Ray Classification Report

## Conclusion

A DenseNet121 transfer-learning prototype was trained on the supplied replacement dataset consisting of 700 tuberculosis-positive images and 720 normal images. After exact duplicate removal for splitting, the data were divided with a fixed seed of 42 into training, validation, and test partitions. On the unseen test partition, the model achieved **94.84% accuracy**, **98.10% sensitivity**, **91.67% specificity**, **94.93% F1-score**, and **0.9957 ROC-AUC**. These values describe this experiment only. They do not establish clinical reliability or diagnostic capability.

## Dataset analysis

| Property | Finding |
|---|---:|
| Total supplied images | 1,420 |
| Tuberculosis class | 700 |
| Normal class | 720 |
| File format | PNG |
| Image dimensions | 512 × 512 pixels |
| Corrupted files | 0 |
| TB color modes | 392 RGB; 308 grayscale |
| Normal color mode | 720 RGB |
| Exact duplicate groups | 3 groups, 6 files |

The class distribution is nearly balanced. Three exact duplicate pairs were detected in the tuberculosis folder. One copy from each duplicate group was excluded before splitting, preventing identical files from appearing in different partitions. No patient identifiers or patient-level metadata were supplied, so patient-level splitting could not be performed.

## Method

Images were resized to 224 × 224 pixels and converted to three-channel RGB. DenseNet-specific ImageNet preprocessing was applied. Training-only augmentation used small rotations, translations, and zooms. Validation and test images were not augmented.

The model used ImageNet-pretrained DenseNet121 without its classification head, followed by global average pooling, dropout, a 128-unit ReLU dense layer, additional dropout, and a sigmoid output representing TB probability. The backbone was initially frozen. The final 40 backbone layers were then fine-tuned with a lower learning rate. Model checkpointing selected the best validation ROC-AUC. Early stopping and learning-rate reduction were enabled.

DenseNet121 was selected because its dense feature connectivity supports feature reuse and stable gradient flow. This is a reasonable transfer-learning baseline for image classification, but it does not make the architecture medically validated.

## Test results

The test partition contained 108 Normal images and 105 TB images. At the default probability threshold of 0.5, the confusion matrix was:

| Actual \ Predicted | Normal | TB |
|---|---:|---:|
| Normal | 99 | 9 |
| TB | 2 | 103 |

| Metric | Result |
|---|---:|
| Accuracy | 0.9484 |
| Precision | 0.9196 |
| Recall / sensitivity | 0.9810 |
| Specificity | 0.9167 |
| F1-score | 0.9493 |
| ROC-AUC | 0.9957 |

Sensitivity is important for a screening-oriented prototype because false negatives may be consequential. The two TB false negatives in this test split should not be generalized beyond this dataset.

## Explainability and prediction

The project includes Grad-CAM generation for selected test images and a `predict_xray(image_path)` function. Grad-CAM highlights regions that influenced the model output. It does not prove the presence of tuberculosis and should not be interpreted as a medical diagnosis.

## Limitations

The dataset may contain selection bias, acquisition artifacts, label noise, and unmeasured correlations between images. The absence of patient metadata means that patient-level independence cannot be confirmed. Exact duplicate checking does not detect all near-duplicates or images from the same patient. The test set is relatively small, so metrics have uncertainty and may change under another split. External validation on an independent dataset, calibration assessment, subgroup analysis, and clinical review would be required before any clinical use. Privacy, licensing, and usage conditions for the dataset must be respected.

## Reproduction

Install dependencies with `pip install -r requirements.txt`. Run dataset analysis with `python src/dataset_analysis.py --root raw --out results/dataset_analysis.json`, then run `python src/tb_pipeline.py`. The best model is saved at `models/best_densenet121.keras`; plots, reports, metrics, and Grad-CAM images are saved under `results/`.

## References

[1]: https://keras.io/api/applications/densenet/ "Keras DenseNet application documentation"

[2]: https://arxiv.org/abs/1608.06993 "Densely Connected Convolutional Networks"

[3]: https://arxiv.org/abs/1610.02391 "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
