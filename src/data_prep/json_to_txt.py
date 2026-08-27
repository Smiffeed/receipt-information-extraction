import json
import os
import glob
import cv2

# Define class names (ensure order matches your dataset)
class_names = ['ชื่อผู้ซื้อภาษาไทย (buyer_name_thai)', 
                'ชื่อผู้ซื้อภาษาอังกฤษ (buyer_name_eng)',
                'ชื่อผู้ขายภาษาไทย (seller_name_thai)',
                'ชื่อผู้ขายภาษาอังกฤษ (seller_name_eng)',
                'หมายเลขภาษีผู้ซื้อ (buyer_vat_number)',
                'หมายเลขภาษีผู้ขาย (seller_vat_number)',
                'ยอดรวมสุทธิ (total_due_amount)']

# Function to convert LabelMe JSON to YOLO format
def convert_labelme_to_yolo(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    for json_file in json_files:
        with open(json_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        img_path = os.path.join(json_dir, data["imagePath"])
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        label_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".txt")

        with open(label_file, "w", encoding='utf-8') as f:
            for shape in data["shapes"]:
                points = shape["points"]
                label = shape["label"]

                if label not in class_names:
                    continue

                class_id = class_names.index(label)

                x_min = min(points[0][0], points[1][0]) / w
                y_min = min(points[0][1], points[1][1]) / h
                x_max = max(points[0][0], points[1][0]) / w
                y_max = max(points[0][1], points[1][1]) / h

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Convert LabelMe JSON annotations
json_dir = "test/"
output_dir = "test"
convert_labelme_to_yolo(json_dir, output_dir)
