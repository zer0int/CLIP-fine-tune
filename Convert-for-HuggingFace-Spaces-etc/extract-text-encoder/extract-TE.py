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

# Create a new dictionary for the text encoder: Only contains what also exists in original model.safetensors.
filtered_state_dict = {k: v for k, v in finetuned_state_dict.items() if k in original_state_dict}

# Save the filtered state dictionary
save_file(filtered_state_dict, "my-finetune_TE-only.safetensors")
