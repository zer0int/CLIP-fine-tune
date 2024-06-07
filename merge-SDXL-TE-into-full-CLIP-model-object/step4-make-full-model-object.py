import torch
import clip

# Load the combined state dictionary
state_dict = torch.load("step3-removed-error-keys.pt")

# Define the additional attributes
input_resolution = 224  # Example value, set according to your model requirements
context_length = 77     # Example value, set according to your model requirements
vocab_size = 49408      # Example value, set according to your model requirements

# Load the original CLIP model
model, preprocess = clip.load("ViT-L/14", device='cpu')

# Remove unexpected keys from the state dictionary
keys_to_remove = ["input_resolution", "context_length", "vocab_size"]
for key in keys_to_remove:
    state_dict.pop(key, None)

# Replace the model's state dictionary with the combined state dictionary
model.load_state_dict(state_dict)

# Set the additional attributes
model.input_resolution = input_resolution
model.context_length = context_length
model.vocab_size = vocab_size

# Save the full model object to a .pt file
torch.save(model, "combined-ViT-L-14-full-model-object.pt")

print("Model saved successfully.")
