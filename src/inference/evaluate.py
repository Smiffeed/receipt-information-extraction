import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns
from ultralytics import YOLO

class YOLOEvaluator:
    def __init__(self, test_dir: str, model_weights: List[str] = None):
        self.test_dir = Path(test_dir)
        self.images_dir = self.test_dir / 'images'
        self.labels_dir = self.test_dir / 'labels'
        
        # Load class names from data.yaml
        with open(self.test_dir / 'data.yaml', 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
            self.class_names = self.data_config['names']
        
        # Initialize ensemble models if weights are provided
        self.models = []
        if model_weights:
            print("Loading model ensemble...")
            for weight_path in model_weights:
                try:
                    model = YOLO(weight_path)
                    self.models.append(model)
                    print(f"Loaded model from {weight_path}")
                except Exception as e:
                    print(f"Error loading model from {weight_path}: {e}")
    
    def load_ground_truth(self, label_file: Path) -> List[Dict]:
        """Load ground truth annotations from a label file."""
        annotations = []
        if not label_file.exists():
            return annotations
        
        with open(label_file, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                annotations.append({
                    'class_id': int(class_id),
                    'class_name': self.class_names[int(class_id)],
                    'bbox': [x_center, y_center, width, height]
                })
        return annotations
    
    def get_ensemble_predictions(self, image_path: str) -> List[Dict]:
        """Get predictions from ensemble of models."""
        if not self.models:
            raise ValueError("No models loaded for prediction")
        
        all_predictions = []
        for model in self.models:
            results = model(image_path)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates (normalized)
                    x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                    # Convert to center format
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width/2
                    y_center = y1 + height/2
                    
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    all_predictions.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'bbox': [x_center, y_center, width, height],
                        'confidence': confidence
                    })
        
        # Perform non-maximum suppression on ensemble predictions
        return self.ensemble_nms(all_predictions)
    
    def ensemble_nms(self, predictions: List[Dict], iou_threshold: float = 0.5, 
                    conf_threshold: float = 0.25) -> List[Dict]:
        """Apply Non-Maximum Suppression to ensemble predictions."""
        # Filter by confidence
        predictions = [p for p in predictions if p['confidence'] >= conf_threshold]
        if not predictions:
            return []
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply NMS
        kept_predictions = []
        while predictions:
            best_pred = predictions.pop(0)
            kept_predictions.append(best_pred)
            
            # Filter out overlapping predictions
            filtered_predictions = []
            for pred in predictions:
                if (pred['class_id'] != best_pred['class_id'] or
                    self.calculate_iou(pred['bbox'], best_pred['bbox']) < iou_threshold):
                    filtered_predictions.append(pred)
            predictions = filtered_predictions
        
        return kept_predictions
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        # Convert from center format to corner format
        box1_x1 = box1[0] - box1[2]/2
        box1_y1 = box1[1] - box1[3]/2
        box1_x2 = box1[0] + box1[2]/2
        box1_y2 = box1[1] + box1[3]/2
        
        box2_x1 = box2[0] - box2[2]/2
        box2_y1 = box2[1] - box2[3]/2
        box2_x2 = box2[0] + box2[2]/2
        box2_y2 = box2[1] + box2[3]/2
        
        # Calculate intersection
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict], 
                           iou_threshold: float = 0.5) -> Dict:
        """Evaluate predictions against ground truth for a single image."""
        results = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'class_metrics': {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(len(self.class_names))}
        }
        
        # Mark ground truth boxes as unmatched initially
        gt_matched = [False] * len(ground_truth)
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            # Find the best matching ground truth box
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx] or pred['class_id'] != gt['class_id']:
                    continue
                
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if we found a match above the threshold
            if best_iou >= iou_threshold:
                results['true_positives'] += 1
                results['class_metrics'][pred['class_id']]['tp'] += 1
                gt_matched[best_gt_idx] = True
            else:
                results['false_positives'] += 1
                results['class_metrics'][pred['class_id']]['fp'] += 1
        
        # Count unmatched ground truth boxes as false negatives
        for gt_idx, gt in enumerate(ground_truth):
            if not gt_matched[gt_idx]:
                results['false_negatives'] += 1
                results['class_metrics'][gt['class_id']]['fn'] += 1
        
        return results
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate precision, recall, and F1-score from evaluation results."""
        metrics = {}
        
        # Overall metrics
        tp = results['true_positives']
        fp = results['false_positives']
        fn = results['false_negatives']
        
        metrics['overall'] = {
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        }
        metrics['overall']['f1'] = (2 * metrics['overall']['precision'] * metrics['overall']['recall'] /
                                  (metrics['overall']['precision'] + metrics['overall']['recall'])
                                  if (metrics['overall']['precision'] + metrics['overall']['recall']) > 0 else 0)
        
        # Per-class metrics
        metrics['per_class'] = {}
        for class_id, class_name in self.class_names.items():
            class_results = results['class_metrics'][class_id]
            class_tp = class_results['tp']
            class_fp = class_results['fp']
            class_fn = class_results['fn']
            
            precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0
            recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0)
            
            metrics['per_class'][class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics
    
    def visualize_results(self, metrics: Dict, output_dir: str = 'evaluation_results'):
        """Create visualizations of the evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for per-class metrics visualization
        class_names = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_name, class_metrics in metrics['per_class'].items():
            class_names.append(class_name)
            precisions.append(class_metrics['precision'])
            recalls.append(class_metrics['recall'])
            f1_scores.append(class_metrics['f1'])
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1_scores, width, label='F1-score')
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Class')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_metrics.png'))
        plt.close()
        
        # Create heatmap of confusion matrix
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precisions,
            'Recall': recalls,
            'F1-score': f1_scores
        })
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_df[['Precision', 'Recall', 'F1-score']].values.T,
                   xticklabels=class_names,
                   yticklabels=['Precision', 'Recall', 'F1-score'],
                   annot=True,
                   fmt='.3f',
                   cmap='YlOrRd')
        plt.title('Performance Metrics Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
        plt.close()
        
        # Save numerical results to CSV
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Save overall metrics
        with open(os.path.join(output_dir, 'overall_metrics.txt'), 'w') as f:
            f.write(f"Overall Metrics:\n")
            f.write(f"Precision: {metrics['overall']['precision']:.3f}\n")
            f.write(f"Recall: {metrics['overall']['recall']:.3f}\n")
            f.write(f"F1-score: {metrics['overall']['f1']:.3f}\n")

def find_best_checkpoints(exp_dir: Path, k_folds: int) -> List[str]:
    """Find the best checkpoint weights for each fold."""
    best_checkpoints = []
    for fold in range(k_folds):
        # Find the folder for this fold
        fold_dirs = list(exp_dir.glob(f"yolo_custom_*_fold{fold}"))
        if fold_dirs:
            # Get the most recent folder if multiple exist
            fold_dir = sorted(fold_dirs)[-1]
            checkpoint_path = fold_dir / 'weights' / 'best.pt'
            if checkpoint_path.exists():
                best_checkpoints.append(str(checkpoint_path))
            else:
                print(f"Warning: Could not find checkpoint for fold {fold} at {checkpoint_path}")
    return best_checkpoints

def main():
    # Configuration
    exp_dir = Path('runs/detect/lone')
    k_folds = 5  # Number of folds used in training
    
    # Find best checkpoints from training
    print("Finding best model checkpoints...")
    best_checkpoints = find_best_checkpoints(exp_dir, k_folds)
    
    if not best_checkpoints:
        print("Error: No model checkpoints found. Please ensure models are trained first.")
        return
    
    print(f"Found {len(best_checkpoints)} model checkpoints")
    
    # Initialize evaluator with ensemble models
    evaluator = YOLOEvaluator('test', best_checkpoints)
    
    # Collect all image files
    image_files = list(evaluator.images_dir.glob('*.jpg')) + list(evaluator.images_dir.glob('*.JPG'))
    
    # Initialize overall results
    overall_results = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'class_metrics': {i: {'tp': 0, 'fp': 0, 'fn': 0} 
                         for i in range(len(evaluator.class_names))}
    }
    
    print("Starting evaluation...")
    for image_file in tqdm(image_files):
        # Load ground truth
        label_file = evaluator.labels_dir / f"{image_file.stem}.txt"
        ground_truth = evaluator.load_ground_truth(label_file)
        
        # Get ensemble predictions
        predictions = evaluator.get_ensemble_predictions(str(image_file))
        
        # Evaluate predictions for this image
        results = evaluator.evaluate_predictions(predictions, ground_truth)
        
        # Accumulate results
        overall_results['true_positives'] += results['true_positives']
        overall_results['false_positives'] += results['false_positives']
        overall_results['false_negatives'] += results['false_negatives']
        for class_id in evaluator.class_names:
            for metric in ['tp', 'fp', 'fn']:
                overall_results['class_metrics'][class_id][metric] += \
                    results['class_metrics'][class_id][metric]
    
    # Calculate final metrics
    metrics = evaluator.calculate_metrics(overall_results)
    
    # Visualize results
    evaluator.visualize_results(metrics)
    print("Evaluation complete! Results saved in 'evaluation_results' directory.")

if __name__ == "__main__":
    main() 