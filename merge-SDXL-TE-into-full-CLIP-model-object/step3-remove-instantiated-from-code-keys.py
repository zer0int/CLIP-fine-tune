import torch
import clip

# Load the combined state dictionary
state_dict = torch.load("step2-clip_combined_state_dict.pt")

# Remove the unwanted keys
keys_to_remove = ["input_resolution", "context_length", "vocab_size"]
for key in keys_to_remove:
    if key in state_dict:
        del state_dict[key]

# Load the original CLIP model
model, preprocess = clip.load("ViT-L/14", device='cpu')

# Replace the model's state dictionary with the combined state dictionary
model.load_state_dict(state_dict)

# Save the full model object to a .pt file
torch.save(model.state_dict(), "step3-removed-error-keys.pt")

print("Model saved successfully.")
