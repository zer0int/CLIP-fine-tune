import torch

# Load the original ViT-L/14 model state_dict made in step 0
vit_state_dict = torch.load("ViT-L-14-state-dict.pt")

# Load the converted text encoder state_dict
text_state_dict = torch.load("step1-state-dict-clip-like.pt")

# Initialize a new state dictionary for the combined model
combined_state_dict = {}

# Copy all items from the ViT-L/14 state_dict to the combined state_dict
combined_state_dict.update(vit_state_dict)

# Copy relevant items from the converted text encoder state_dict to the combined state_dict
keys_to_copy = [
    "logit_scale",
    "positional_embedding",
    "token_embedding.weight",
    "text_projection",
    "ln_final.weight",
    "ln_final.bias",
]

for key in keys_to_copy:
    combined_state_dict[key] = text_state_dict[key]

# Copy all resblocks from the text transformer to the combined state_dict
for key, value in text_state_dict.items():
    if key.startswith("transformer.resblocks"):
        combined_state_dict[key] = value

# Save the combined state dictionary to a new file
torch.save(combined_state_dict, "step2-clip_combined_state_dict.pt")
