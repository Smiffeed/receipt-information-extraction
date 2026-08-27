from ultralytics import YOLO
import os
from datetime import datetime
import torch
import numpy as np
from pathlib import Path
import glob
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import yaml
import json

def normalize_labels(dataset_path):
    """
    Normalize YOLO format labels to ensure all coordinates are between 0 and 1
    
    Args:
        dataset_path: Path to the dataset root directory
    """
    print("Normalizing dataset labels...")
    
    # Create backup of original labels
    labels_dir = os.path.join(dataset_path, 'labels')
    backup_dir = os.path.join(dataset_path, 'labels_backup')
    if not os.path.exists(backup_dir):
        shutil.copytree(labels_dir, backup_dir)
        print(f"Created backup of original labels in {backup_dir}")
    
    # Process all label files
    label_files = glob.glob(os.path.join(dataset_path, 'labels', '**', '*.txt'), recursive=True)
    normalized_count = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            normalized_lines = []
            needs_normalization = False
            
            for line in lines:
                values = line.strip().split()
                if len(values) >= 5:  # class_id + 4 coordinates
                    class_id = values[0]
                    coords = [float(x) for x in values[1:]]
                    
                    # Check if normalization is needed
                    if any(x > 1.0 for x in coords):
                        needs_normalization = True
                        # Normalize coordinates to [0, 1]
                        coords = [min(max(x/6.0, 0.0), 1.0) for x in coords]  # Divide by 6.0 as a scaling factor
                    
                    # Reconstruct the line
                    normalized_line = f"{class_id} {' '.join(f'{x:.6f}' for x in coords)}\n"
                    normalized_lines.append(normalized_line)
                else:
                    normalized_lines.append(line)  # Keep original line if format is unexpected
            
            if needs_normalization:
                with open(label_file, 'w') as f:
                    f.writelines(normalized_lines)
                normalized_count += 1
                
        except Exception as e:
            print(f"Warning: Error processing {label_file}: {str(e)}")
    
    print(f"Normalized {normalized_count} label files")
    print("Label normalization completed")

def clean_labels(dataset_path, max_class_id=6):
    """
    Clean label files by removing annotations with class IDs greater than max_class_id
    
    Args:
        dataset_path: Path to the dataset root directory
        max_class_id: Maximum allowed class ID (inclusive)
    """
    print(f"Cleaning dataset labels (removing classes > {max_class_id})...")
    
    # Create backup of original labels if not already done
    labels_dir = os.path.join(dataset_path, 'labels')
    backup_dir = os.path.join(dataset_path, 'labels_backup')
    if not os.path.exists(backup_dir):
        shutil.copytree(labels_dir, backup_dir)
        print(f"Created backup of original labels in {backup_dir}")
    
    # Process all label files
    label_files = glob.glob(os.path.join(dataset_path, 'labels', '**', '*.txt'), recursive=True)
    cleaned_count = 0
    removed_labels_count = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            file_modified = False
            
            for line in lines:
                values = line.strip().split()
                if len(values) >= 5:  # class_id + 4 coordinates
                    class_id = int(values[0])
                    if class_id <= max_class_id:
                        cleaned_lines.append(line)
                    else:
                        removed_labels_count += 1
                        file_modified = True
                else:
                    cleaned_lines.append(line)
            
            if file_modified:
                with open(label_file, 'w') as f:
                    f.writelines(cleaned_lines)
                cleaned_count += 1
                
        except Exception as e:
            print(f"Warning: Error processing {label_file}: {str(e)}")
    
    print(f"Cleaned {cleaned_count} label files")
    print(f"Removed {removed_labels_count} invalid class labels")
    print("Label cleaning completed")

def create_model_ensemble(model_paths):
    """
    Create an ensemble of models for better predictions
    
    Args:
        model_paths: List of paths to trained model weights
    """
    models = [YOLO(path) for path in model_paths]
    return models

def ensemble_predict(models, img):
    """
    Make predictions using model ensemble
    """
    predictions = []
    for model in models:
        pred = model(img)
        predictions.append(pred)
    # Average predictions (you might want to implement more sophisticated ensemble methods)
    return predictions

def train_yolo(data_yaml_path, epochs=100, imgsz=640, batch_size=16):
    """
    Train YOLO model on the converted dataset with enhanced augmentation and hyperparameters
    
    Args:
        data_yaml_path: Path to the data.yaml file
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
    """
    # Create unique experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"yolo_custom_{timestamp}"

    # Load a YOLOv8 model
    model = YOLO('yolo11s.pt')  # load pretrained
    
    # Multi-GPU settings if available
    num_gpus = torch.cuda.device_count()
    device = '0' if num_gpus == 1 else ','.join([str(i) for i in range(num_gpus)])
    
    # Train the model with enhanced parameters
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=exp_name,
        device=device,
        patience=50,
        save=True,
        plots=True,
        
        # Optimizer settings
        lr0=0.01,  # initial learning rate
        lrf=0.01,  # final learning rate factor
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay
        warmup_epochs=3.0,  # warmup epochs
        warmup_momentum=0.8,  # warmup initial momentum
        warmup_bias_lr=0.1,  # warmup initial bias lr
        
        # Data augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=10.0,  # rotation (+/- deg)
        translate=0.1,  # translation (+/- fraction)
        scale=0.5,  # scale (+/- gain)
        shear=2.0,  # shear (+/- deg)
        perspective=0.0,  # perspective (+/- fraction)
        flipud=0.5,  # probability of flip up-down
        fliplr=0.5,  # probability of flip left-right
        mosaic=1.0,  # mosaic augmentation probability
        mixup=0.1,  # mixup augmentation probability
        copy_paste=0.1,  # segment copy-paste probability
        
        # Additional training configurations
        cos_lr=True,  # use cosine learning rate scheduler
        close_mosaic=10,  # disable mosaic augmentation for final epochs
        
        # Advanced training settings
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,  # dataset fraction to train on
        cache=True,  # cache images for faster training
        workers=8,  # number of worker threads
        project='yolo_experiments',  # project name for logging
        exist_ok=True,  # whether to overwrite existing experiment
        pretrained=True,  # whether to use pretrained backbone
        optimizer='auto',  # optimizer to use (auto/SGD/Adam/AdamW)
        verbose=True,  # whether to print verbose output
        seed=42,  # random seed for reproducibility
        deterministic=True,  # whether to enable deterministic mode
        single_cls=False,  # train as single-class dataset
        rect=False,  # rectangular training
        resume=False,  # resume training from last checkpoint
        nbs=64,  # nominal batch size
        overlap_mask=True,  # masks should overlap during training
        mask_ratio=4,  # mask downsample ratio
    )
    
    # Validate and calculate metrics
    val_results = model.val()
    
    # Get class names from data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', [f'class_{i}' for i in range(data_config.get('nc', 1))])
    
    # Create metrics report using validation results directly
    metrics_report = {
        'Overall Metrics': {
            'mAP50-95': float(val_results.box.map),    # mean AP for IoU from 0.5 to 0.95
            'mAP50': float(val_results.box.map50),     # mean AP at IoU=0.50
            'mAP75': float(val_results.box.map75),     # mean AP at IoU=0.75
            'Precision': float(val_results.box.mp),     # mean precision
            'Recall': float(val_results.box.mr)        # mean recall
        },
        'Per-class Metrics': {}
    }
    
    # Add per-class metrics if available
    for i, class_name in enumerate(class_names):
        metrics_report['Per-class Metrics'][class_name] = {
            'Precision': float(val_results.box.ap50[i]) if len(val_results.box.ap50) > i else 0.0,
            'mAP50': float(val_results.box.ap[i]) if len(val_results.box.ap) > i else 0.0,
            'Recall': float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0.0
        }
    
    # Create metrics directory
    metrics_dir = Path(f'yolo_experiments/{exp_name}/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics report as JSON
    with open(metrics_dir / 'metrics_report.json', 'w') as f:
        json.dump(metrics_report, f, indent=4)
    
    # Print summary
    print("\nTraining and Validation Results:")
    print("-" * 50)
    print(f"Model saved in: yolo_experiments/{exp_name}")
    print(f"Metrics saved in: {metrics_dir}")
    print("\nOverall Metrics:")
    for metric, value in metrics_report['Overall Metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPer-class Metrics Summary:")
    for class_name, class_metrics in metrics_report['Per-class Metrics'].items():
        print(f"\n{class_name}:")
        for metric, value in class_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return model, metrics_report

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot confusion matrix using seaborn
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the confusion matrix plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(model, val_loader, class_names):
    """
    Calculate comprehensive metrics including confusion matrix, precision, recall, and F1 score
    
    Args:
        model: Trained YOLO model
        val_loader: Validation data loader
        class_names: List of class names
    """
    all_preds = []
    all_targets = []
    
    # Get predictions
    for batch in val_loader:
        preds = model(batch['images'])
        targets = batch['labels']
        
        # Convert predictions and targets to class indices
        pred_classes = [pred.boxes.cls.cpu().numpy() for pred in preds]
        target_classes = [target.boxes.cls.cpu().numpy() for target in targets]
        
        all_preds.extend(pred_classes)
        all_targets.extend(target_classes)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    
    # Calculate overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    return {
        'confusion_matrix': cm,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1
    }

if __name__ == '__main__':
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'yolo_dataset', 'data.yaml')
    
    # Normalize and clean dataset labels
    dataset_path = os.path.join(current_dir, 'yolo_dataset')
    normalize_labels(dataset_path)
    clean_labels(dataset_path, max_class_id=6)  # Clean labels to keep only classes 0-6
    
    # Training parameters
    config = {
        'epochs': 100,
        'imgsz': 640,
        'batch_size': 16
    }
    
    # Create experiment directory
    exp_dir = Path('yolo_experiments')
    exp_dir.mkdir(exist_ok=True)
    
    # Perform single training run
    print("Starting training...")
    model, metrics = train_yolo(
        data_yaml_path,
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch_size=config['batch_size']
    )
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = exp_dir / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nTraining completed successfully!")
    print(f"Model and metrics saved in {exp_dir}")