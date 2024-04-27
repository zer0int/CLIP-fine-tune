import csv
import json
import os

# This script is for CLIP interrogator desc.csv output https://github.com/pharmapsychotic/clip-interrogator
# Check the example json file for the expected structure as used by the fine-tuning script
# If your dataset labels are in a different format, use the examples to get an AI like GPT-4 or Claude 3 to help you :-)

def process_clip_opinions(opinion_str, max_length=140):
    opinions = opinion_str.split(", ")
    truncated_opinions = []
    current_length = 0

    for opinion in opinions:
        if current_length + len(opinion) + 2 > max_length:  # +2 for comma and space
            break
        truncated_opinions.append(opinion)
        current_length += len(opinion) + 2

    return truncated_opinions

def convert_csv_to_json(csv_filename, json_filename):
    data = {}

    with open(csv_filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            image_name = row[0]
            clip_opinions = process_clip_opinions(row[1])
            data[image_name] = clip_opinions

    with open(json_filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# Specify your file names here
csv_filename = 'desc.csv'
json_filename = 'my-dataset-labels.json'

# Convert CSV to JSON
convert_csv_to_json(csv_filename, json_filename)

print("Conversion complete. JSON file created.")
