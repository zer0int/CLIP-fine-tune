import json
import os

# Define the path to your original and corrected metadata files
original_metadata_path = 'coco-spright-train-0_9.json'
corrected_metadata_path = 'coco-spright-train-0_9-kohya.json'

# Load the original metadata
with open(original_metadata_path, 'r') as f:
    original_metadata = json.load(f)

corrected_metadata = {}

for image_path, caption_file_path in original_metadata.items():
    # Read the caption from the .caption file
    try:
        with open(caption_file_path, 'r', encoding='utf-8') as cf:
            caption_text = cf.readline().strip()  # Reads the first line; assumes there's only one line per file
    except FileNotFoundError:
        print(f"Caption file not found: {caption_file_path}")
        caption_text = ""  # Use an empty string if the file is missing

    # Assign the read caption and static tags to the corrected metadata
    corrected_metadata[image_path] = {
        "caption": caption_text,
    }

# Save the corrected metadata back to file
with open(corrected_metadata_path, 'w', encoding='utf-8') as f:
    json.dump(corrected_metadata, f, indent=4, ensure_ascii=False)

print(f"Corrected metadata saved to {corrected_metadata_path}")
