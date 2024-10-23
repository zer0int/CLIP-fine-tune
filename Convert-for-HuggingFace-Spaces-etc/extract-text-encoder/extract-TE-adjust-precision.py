import torch
from safetensors.torch import load_file, save_file

"""
This is the ORIGINAL CLIP-L text encoder only model. Get it from HuggingFace, for example for Flux.1:
https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/text_encoder
This is important as we are comparing the names of the keys in the model, discarding any that
should not be present in the text-encoder-only model, as expected by [here: Flux.1].
"""
original_state_dict = load_file("model.safetensors")

# Fine-tune after conversion with HuggingFace pytorch-to-HF script:
finetuned_state_dict = load_file("my-finetune.safetensors")

# Create a new dictionary for the text encoder with matching keys AND! precision
# If you have any issues with this, please use the standard "extract-TE.py" script
filtered_state_dict = {}
for key, tensor in finetuned_state_dict.items():
    if key in original_state_dict:
        # Match precision (dtype) of the original tensor
        target_dtype = original_state_dict[key].dtype
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        filtered_state_dict[key] = tensor

# Save the filtered state dictionary with matched precision
save_file(filtered_state_dict, "my-finetune_TE-only_dtype.safetensors")

# Load the saved text encoder model
filtered_loaded_state_dict = load_file("my-finetune_TE-only_dtype.safetensors")

# Function to compare the model structures
def compare_models(model1, model2):
    """Compare two model state dictionaries by key, shape, and dtype."""
    print(f"{'Key':<50} {'Model 1 Shape':<30} {'Model 2 Shape':<30} {'Match'}")
    print("-" * 130)
    for key in model1.keys() | model2.keys():
        shape1 = model1.get(key, None)
        shape2 = model2.get(key, None)
        if shape1 is not None and shape2 is not None:
            match = shape1.shape == shape2.shape and shape1.dtype == shape2.dtype
            print(f"{key:<50} {str(shape1.shape):<30} {str(shape2.shape):<30} {match}")
        else:
            print(f"{key:<50} {'N/A' if shape1 is None else str(shape1.shape):<30} "
                  f"{'N/A' if shape2 is None else str(shape2.shape):<30} {'No'}")

# Perform comparison
compare_models(original_state_dict, filtered_loaded_state_dict)
