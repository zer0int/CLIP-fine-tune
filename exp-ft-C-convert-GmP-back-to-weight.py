import torch
import torch.nn.functional as F
from orgclip.modeloriginal import CLIP  # Make sure this imports the original CLIP class definition

def convert_back_to_original(state_dict):
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.endswith(".theta"):
            base_key = key.replace(".theta", "")
            r_key = base_key + ".r"
            new_weight = state_dict[r_key] * F.normalize(value, p=2, dim=1)
            new_state_dict[base_key + ".weight"] = new_weight
        elif key.endswith(".r") or key.endswith(".theta"):
            continue  # Skip the .r and .theta keys
        else:
            new_state_dict[key] = value

    return new_state_dict

# Example usage
# Load the fine-tuned model object
modelft = torch.load("ft-checkpoints/clip_ft_20.pt")

# Extract model parameters from the fine-tuned model
embed_dim = modelft.text_projection.shape[1]
image_resolution = modelft.visual.input_resolution
vision_layers = modelft.visual.transformer.layers
vision_width = modelft.visual.conv1.out_channels
vision_patch_size = modelft.visual.conv1.kernel_size[0]
context_length = modelft.context_length
vocab_size = modelft.vocab_size
transformer_width = modelft.transformer.width
transformer_heads = modelft.transformer.resblocks[0].attn.num_heads
transformer_layers = modelft.transformer.layers

# Convert the fine-tuned model to a state_dict
fine_tuned_state_dict = modelft.state_dict()

# Convert back to original weights
original_state_dict = convert_back_to_original(fine_tuned_state_dict)

# Rebuild the original model using the converted state_dict
original_model = CLIP(
    embed_dim=embed_dim,
    image_resolution=image_resolution,
    vision_layers=vision_layers,
    vision_width=vision_width,
    vision_patch_size=vision_patch_size,
    context_length=context_length,
    vocab_size=vocab_size,
    transformer_width=transformer_width,
    transformer_heads=transformer_heads,
    transformer_layers=transformer_layers
)

# Load the converted state_dict into the original model
original_model.load_state_dict(original_state_dict)

# Save the original model object
torch.save(original_model, "ft-checkpoints/full_model_converted_model.pth")

print("Model has been successfully converted back to the original format and saved.")
