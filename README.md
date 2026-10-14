# Plant Disease Classification with CNNs

## Objective
Plant diseases can affect the health and yield of crops, so identifying diseases from their visual appearance can help with early detection. This project designs a convolutional neural network (CNN) image classification model that determines whether a plant is healthy or diseased, and identifies the disease category. CNNs are well suited to this task because they can learn important visual features such as color, texture, and leaf patterns.

Many existing models perform very well on images taken in controlled lab conditions but struggle on real-world photos. Our main challenge is **generalization to unseen, real-world images**.

## Approach
1. **Train and test** a CNN on **PlantWild**, a dataset of in-the-wild plant images.
2. **Analyze** what the model focuses on using **PlantSeg**, comparing Grad-CAM heatmaps with PlantSeg's disease masks.
3. **Test generalization** by evaluating the trained (frozen) model on **FieldPlant**, a dataset of real field photos from plantations.

We also apply techniques to reduce overfitting and compare performance on training data versus real-world images.

## Project Specifications

| Component | Specification |
|---|---|
| Project objective | Classify healthy plants and plant disease categories from images |
| Model | Convolutional Neural Network (CNN) |
| Training dataset | PlantWild |
| Feature analysis dataset | PlantSeg |
| Independent testing dataset | FieldPlant |
| Preprocessing | Duplicate removal, resizing, normalization |
| Evaluation metrics | Accuracy, precision, recall, F1-score |
| Additional evaluation | Confusion matrix, training/validation loss and accuracy |
| Expected output | Predicted plant health or disease category |
| Main challenge | Generalization to unseen, real-world images |

## Datasets

| Dataset | Role | Size | Download | Paper |
|---|---|---|---|---|
| PlantWild | Training + main test | 18,542 images, 89 classes | [Hugging Face](https://huggingface.co/datasets/uqtwei2/PlantWild) | [arXiv 2408.03120](https://arxiv.org/abs/2408.03120) |
| PlantSeg | Feature analysis | 7,774 images, 115 classes, with segmentation masks | [Zenodo](https://doi.org/10.5281/zenodo.17719108) | [Scientific Data 2026](https://www.nature.com/articles/s41597-025-06513-4) |
| FieldPlant | Independent field test | 5,156 images | [Roboflow](https://universe.roboflow.com/plant-disease-detection/fieldplant) | [IEEE Access 2023](https://doi.org/10.1109/ACCESS.2023.3263042) |

> **Note:** Dataset images are not stored in this repository due to their size and licensing. See `data/README.md` for download instructions.

## Repository Structure
```
data/        dataset setup, splits, and class mappings (images downloaded locally)
scripts/     download and preprocessing scripts
```

## Team
- Member 1 – Data pipeline
- Member 2 – Model training and experiments
- Member 3 – Analysis and demo
