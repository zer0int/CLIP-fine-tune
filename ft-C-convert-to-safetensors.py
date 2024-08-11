import torch
import clip
from safetensors.torch import save_file

# Load the model you used for the fine-tune, the fine-tune .weight after conversion with 'ft-C-convert-back-to-weight.py':
model, preprocess = clip.load("ViT-L/14") # Model used for fine-tune
model_path = "ft-checkpoints/clip_ft_20-backtoweight.pt" # Fine-tuned model
model = torch.load(model_path)

# Extract the state dictionary
state_dict = model.state_dict()

# Save the state dictionary as a .safetensors file
safetensors_path = 'ft-checkpoints/clip_ft_20_state_dict.safetensors'
save_file(state_dict, safetensors_path)