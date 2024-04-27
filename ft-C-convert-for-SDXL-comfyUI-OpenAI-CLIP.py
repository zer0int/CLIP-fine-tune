import torch
import os

# Load the fine-tuned model checkpoint of your choice
device = 'cuda'
THE_FINETUNED_MODEL = torch.load('ft-checkpoints/clip_ft_10.pt', map_location=device)

# Save only the state dictionary of the fine-tuned model
# This can be used e.g. with SDXL ComfyUI
torch.save(THE_FINETUNED_MODEL.state_dict(), 'ft-checkpoints/clip_ft_10_STATE_DICT.pt')
