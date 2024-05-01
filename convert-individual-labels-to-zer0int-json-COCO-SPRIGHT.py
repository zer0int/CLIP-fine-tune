import json
import os

def build_unified_json(base_path, subfolder_names, output_file):
    unified_data = {}

    for subfolder in subfolder_names:
        folder_path = os.path.join(base_path, subfolder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(folder_path, json_filename)
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        img_path = os.path.join(f"data/{subfolder}", filename)
                        unified_data[img_path] = [data["spatial_caption"], data["original_caption"]]

    with open(output_file, 'w') as f:
        json.dump(unified_data, f, indent=4)

base_path = "data"  # Make sure to update this path
subfolder_names = [str(i) for i in range(10)]  # Subfolders "0" to "10"
output_file = "labels-coco_spright-unified_data.json"  # The file to save the unified JSON data

build_unified_json(base_path, subfolder_names, output_file)
