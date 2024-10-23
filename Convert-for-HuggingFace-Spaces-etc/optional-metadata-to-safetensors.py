from safetensors import safe_open
from safetensors.torch import save_file
import torch

# IF you get some error mentioning something about 'pt', use this script:

# Path to already pytorch-to-hf converted CLIP finetune .safetensors
safetensors_file = "my-finetune.safetensors"
new_safetensors_file = "my-finetune_metadata.safetensors"

with safe_open(safetensors_file, framework="pt") as f:
    state_dict = {key: f.get_tensor(key) for key in f.keys()}
    metadata = f.metadata()  # Get current metadata

print("Current Metadata:", metadata)

new_metadata = {"format": "pt"}  # Add the 'pt' format to the metadata

save_file(state_dict, new_safetensors_file, metadata=new_metadata)

print(f"Model saved with metadata: {new_metadata}")
