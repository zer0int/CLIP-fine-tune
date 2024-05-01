import json
import os

# Load the JSON data from the file
json_file_path = 'coco-spright-train-0_9.json'  # Update this to the path of your JSON file
output_directory = 'individual_labels'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Process each entry and save the first caption into individual text files
for image_name, captions in data.items():
    # Generate the output file name based on the image name
    output_file_name = os.path.join(output_directory, f"{image_name.split('.')[0]}.txt")
    
    # Write the first caption to the file
    with open(output_file_name, 'w') as output_file:
        output_file.write(captions[0])

print("Conversion completed successfully.")
