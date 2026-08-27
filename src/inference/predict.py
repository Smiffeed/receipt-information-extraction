from ultralytics import YOLO
 
# Load a model
# model = YOLO("last.pt")  # pretrained YOLO11n model
model = YOLO("yolo_experiments/yolo_custom_20250301_194711/weights/last.pt")  # pretrained YOLO11n model
 
# Run batched inference on a list of images
# results = model(["dataset/1.jpg", "dataset/2.jpg"])  # return a list of Results objects
results = model(["test/101.jpg"])  # return a list of Results objects
 
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk