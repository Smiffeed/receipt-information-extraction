import json
import os

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_valid_labels(labelme_json):
    """Extract unique labels from LabelMe JSON to use as a filter"""
    valid_labels = set()
    for shape in labelme_json['shapes']:
        valid_labels.add(shape['label'])
    return valid_labels

def convert_azure_to_labelme(azure_json, valid_labels):
    """Convert Azure Document Intelligence format to LabelMe format"""
    # Initialize LabelMe format
    page = azure_json['analyzeResult']['pages'][0]
    
    # Use the input filename as the image path since Azure format doesn't contain it
    image_filename = "97.jpg"  # Assuming the image file has the same base name as the JSON
    
    labelme_format = {
        "version": "5.1.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_filename,
        "imageData": None,  # We don't copy image data
        "imageHeight": page['height'],
        "imageWidth": page['width']
    }
    
    # Process each word from Azure format
    for word in page['words']:
        if word['content'] in valid_labels:
            # Convert polygon coordinates to points format
            points = [
                [word['polygon'][0], word['polygon'][1]],  # top-left
                [word['polygon'][2], word['polygon'][3]],  # top-right
                [word['polygon'][4], word['polygon'][5]],  # bottom-right
                [word['polygon'][6], word['polygon'][7]]   # bottom-left
            ]
            
            shape = {
                "label": word['content'],
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_format['shapes'].append(shape)
    
    return labelme_format

def main():
    # Load both JSON files
    azure_path = "datasets/labels/97.json"
    labelme_path = "datasets/labels/100.json"
    output_path = "datasets/labels/97_converted.json"
    
    try:
        azure_json = load_json_file(azure_path)
        labelme_json = load_json_file(labelme_path)
        
        # Get valid labels from LabelMe format
        valid_labels = get_valid_labels(labelme_json)
        
        # Convert Azure format to LabelMe format
        converted_json = convert_azure_to_labelme(azure_json, valid_labels)
        
        # Save the converted JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_json, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully converted {azure_path} to LabelMe format")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main() 