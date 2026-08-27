import json
import codecs
from pathlib import Path

def fix_image_path(json_path):
    try:
        # Read JSON with explicit UTF-8 encoding
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Replace forward slashes with backslashes in imagePath
        if 'imagePath' in data:
            data['imagePath'] = data['imagePath'].replace('/', '\\')
        
        # Write back the JSON with explicit UTF-8 encoding
        with codecs.open(json_path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            f.write(json_str)
        
        print(f"Successfully processed {json_path.name}")
        return True
    
    except Exception as e:
        print(f"Error processing {json_path.name}: {str(e)}")
        return False

def main():
    # Get all JSON files in the labels directory
    json_files = Path('datasets/labels').glob('*.json')
    
    success_count = 0
    total_count = 0
    
    for json_path in json_files:
        total_count += 1
        if fix_image_path(json_path):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count} out of {total_count} files")

if __name__ == "__main__":
    main() 