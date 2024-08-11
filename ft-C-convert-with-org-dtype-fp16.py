import torch
from clip.model import convert_weights  # Import the convert_weights function from original openai/clip package

# Step 1: Load your fine-tuned model after converting it back to .weight with 'ft-C-convert-back-to-weight.py':
fine_tuned_model = torch.load("ft-checkpoints/clip_ft_20-backtoweight.pt", map_location="cpu")

# Step 2: Apply the convert_weights function for dtype
convert_weights(fine_tuned_model)  # This will convert applicable weights to fp16

# Step 3: Save the updated model
torch.save(fine_tuned_model, "ft-checkpoints/clip_ft_20-backtoweight_dtype.pt")
