import json
import base64
from pathlib import Path
import glob
import codecs

def convert_azure_to_labelme(json_path, output_path):
    # Get the corresponding image number
    json_number = json_path.stem  # Gets filename without extension
    image_path = Path('datasets/input_images') / f"{json_number}.jpg"
    
    if not image_path.exists():
        print(f"Warning: Image file not found for {json_path}")
        return False
    
    try:
        # Read Azure JSON with explicit UTF-8 encoding
        with codecs.open(json_path, 'r', encoding='utf-8-sig') as f:
            azure_data = json.load(f)
        
        # Create LabelMe format
        labelme_data = {
            "version": "5.6.1",
            "flags": {},
            "shapes": [],
            "imagePath": f"../input_images/{json_number}.jpg"
        }

        # Define label mapping from Azure to LabelMe format
        label_mapping = {
            "VendorAddressRecipient": "ชื่อผู้ขายภาษาไทย (seller_name_thai)",
            "VendorTaxId": "หมายเลขภาษีผู้ขาย (seller_vat_number)",
            "CustomerAddressRecipient": "ชื่อผู้ซื้อภาษาไทย (buyer_name_thai)",
            "CustomerTaxId": "หมายเลขภาษีผู้ซื้อ (buyer_vat_number)",
            "InvoiceTotal": "ยอดรวมสุทธิ (total_due_amount)"
        }

        # Convert fields from Azure format
        if "analyzeResult" in azure_data:
            analyze_result = azure_data["analyzeResult"]
            if "documents" in analyze_result:
                for doc in analyze_result["documents"]:
                    for field_name, field_data in doc["fields"].items():
                        if field_name in label_mapping and "boundingRegions" in field_data:
                            for region in field_data["boundingRegions"]:
                                points = [[region["polygon"][i], region["polygon"][i+1]] 
                                        for i in range(0, len(region["polygon"]), 2)]
                                # Convert 4 points to 2 points (top-left and bottom-right)
                                if len(points) == 4:
                                    x_coords = [p[0] for p in points]
                                    y_coords = [p[1] for p in points]
                                    points = [[min(x_coords), min(y_coords)], 
                                            [max(x_coords), max(y_coords)]]
                                
                                shape = {
                                    "label": label_mapping[field_name],
                                    "points": points,
                                    "group_id": None,
                                    "description": "",
                                    "shape_type": "rectangle",
                                    "flags": {},
                                    "mask": None
                                }
                                labelme_data["shapes"].append(shape)
        
        # Read and encode image
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Add imageData
        labelme_data['imageData'] = img_data
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write LabelMe JSON with explicit UTF-8 encoding and no BOM
        with codecs.open(output_path, 'w', encoding='utf-8') as f:
            # Use ensure_ascii=False to properly handle Thai characters
            json_str = json.dumps(labelme_data, ensure_ascii=False, indent=2)
            f.write(json_str)
        
        print(f"Successfully processed {json_path.name}")
        return True
    
    except Exception as e:
        print(f"Error processing {json_path.name}: {str(e)}")
        return False

def main():
    # Get all JSON files in the unpreprocessed directory
    json_files = Path('datasets/unpreprocessed_labels').glob('*.json')
    output_dir = Path('datasets/labels')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    for json_path in json_files:
        total_count += 1
        output_path = output_dir / json_path.name
        if convert_azure_to_labelme(json_path, output_path):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count} out of {total_count} files")

if __name__ == "__main__":
    main()