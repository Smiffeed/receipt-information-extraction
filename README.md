# 👁️ YOLO Object Detection Pipeline

An end-to-end computer vision pipeline designed for training, ensembling, and evaluating YOLO (You Only Look Once) models. This repository provides a robust framework for converting custom datasets, applying advanced augmentations, and performing rigorous K-Fold ensemble evaluations using custom Non-Maximum Suppression (NMS).

> [!NOTE]
> This project heavily extends the standard Ultralytics framework by implementing custom cross-validation ensembling, precise dataset coordinate normalization, and deep statistical evaluation plotting.

## Key Features

- **Automated Label Processing**: Includes scripts (`convert_json.py`, `json_to_txt.py`) to automatically convert JSON annotations into normalized YOLO `.txt` format, ensuring bounding boxes are correctly constrained to `[0, 1]` ranges.
- **Advanced Training Augmentations**: The `train_yolo.py` pipeline utilizes highly optimized hyperparameters including Mosaic, Mixup, Copy-Paste, and HSV shifts, paired with cosine learning rate scheduling.
- **Model Ensembling & Custom NMS**: The `evaluate.py` script automatically scans for K-Fold checkpoints and aggregates predictions using a custom Ensemble Non-Maximum Suppression algorithm based on precise IoU (Intersection over Union) calculations.
- **Detailed Statistical Visualization**: Automatically generates confusion matrices, per-class Precision/Recall/F1 bar plots, and heatmap metrics using Seaborn and Pandas.

## Pipeline Architecture

1. **Dataset Normalization**: `train_yolo.py` cleans label coordinates and removes out-of-bound class IDs.
2. **Model Training**: Leverages `yolo11s.pt` (or `yolo11n.pt`) as the backbone for multi-epoch training with advanced mixed-precision (AMP).
3. **Inference & Ensembling**: `evaluate.py` loads the best checkpoints from multiple folds, runs inference, and merges overlapping bounding boxes using custom IoU thresholds.
4. **Metrics Generation**: Predicts against the test set and outputs `.csv` reports and `.png` heatmaps to the `evaluation_results/` directory.

## Getting Started

### Prerequisites

Ensure you have Python installed, along with PyTorch and the required data science libraries:

```bash
pip install -r requirements.txt
```

### 1. Data Preparation

Place your dataset inside the `yolo_dataset/` directory. If you are starting with JSON annotations, convert them first:

```bash
python convert_json.py
python json_to_txt.py
```

### 2. Training the Model

To begin training the YOLO model with the advanced augmentation pipeline, run:

```bash
python train_yolo.py
```
*This will automatically normalize the labels, train the model, and save the weights and validation metrics in `yolo_experiments/`.*

### 3. Ensemble Evaluation

Once training is complete (or if you have K-Fold checkpoints in `runs/`), evaluate the ensemble's performance on the test set:

```bash
python evaluate.py
```
*The script will output comprehensive heatmaps, confusion matrices, and precision/recall statistics to the `evaluation_results/` folder.*