from ultralytics import YOLO

# Load the trained YOLOv11 model (replace with the actual path)
model = YOLO("runs/detect/lone/weights/best.pt")

# Evaluate the model on a validation dataset (replace 'val_data.yaml' with your dataset)
metrics = model.val(data="yolo_dataset/data.yaml")

# Print key evaluation metrics
print("Evaluation Results:")
print(f"mAP@0.5: {metrics.box.map50:.4f}")  # mAP at IoU 0.5
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")  # mAP across multiple IoU thresholds
print(f"Precision: {metrics.box.precision:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")

# Optional: Save evaluation results to a file
with open("evaluation_results.txt", "w") as f:
    f.write("Evaluation Results:\n")
    f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
    f.write(f"Precision: {metrics.box.precision:.4f}\n")
    f.write(f"Recall: {metrics.box.recall:.4f}\n")
