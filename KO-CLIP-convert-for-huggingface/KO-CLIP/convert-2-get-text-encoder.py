from safetensors.torch import load_file, save_file

def filter_and_save_model(input_file, output_file):
    model = load_file(input_file)
    filtered_model = {key: value for key, value in model.items() if "vision_" not in key and "visual_" not in key}

    save_file(filtered_model, output_file)

    print(f"Filtered model saved to {output_file}")

# Input and output file paths
input_file = "converted_model/model.safetensors"
output_file = "converted_model/model_text_encoder.safetensors"

# Run the filtering function
filter_and_save_model(input_file, output_file)