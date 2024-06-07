import torch
import os

# Load the original OpenAI CLIP model
device = 'cuda'
THE_FINETUNED_MODEL = torch.load("path/to/.cache/clip/ViT-L-14.pt", map_location=device)

# Save only the state dictionary
torch.save(THE_FINETUNED_MODEL.state_dict(), "path/to/.cache/clip/ViT-L-14-state-dict.pt")
