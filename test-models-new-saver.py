"""
Example script on how to use different types of saved checkpoints.
To use, make sure you specify ft_checkpoints_folder AND have a photo of a cat.jpg available!
"""

save_is_gmp = False # if saved with 'save_as_gmp = True', set 'True'. If 'False', set 'False'.

if save_is_gmp:    
    import gmpclip as clip
    suffix = 'gmp'
if not save_is_gmp:
    import clip
    suffix = 'weight'

import torch
from PIL import Image
import os
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # disbale torch nag about pickle spam

device = "cuda:0" if torch.cuda.is_available() else "cpu"
clipmodel='ViT-L/14'

ft_checkpoints_folder = 'ft-checkpoints'
epoch = 1 # epoch of saved file; e.g. if filename is clip_ft_1_*.pt => enter 1

# Define paths
full_model_path = os.path.join(ft_checkpoints_folder, f'clip_ft_{epoch}_full_as-{suffix}.pt')
state_dict_path = os.path.join(ft_checkpoints_folder, f'clip_ft_{epoch}_dict_as-{suffix}.pt')
jit_model_path = os.path.join(ft_checkpoints_folder, f'clip_ft_{epoch}_jit_as-{suffix}.pt')

# Make sure you have a cat image available!
image_path = 'cat.jpg'

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = clip.load(clipmodel)[1]
    return preprocess(image).unsqueeze(0)

image_input = preprocess_image(image_path).to(device)

# Define a text prompt
text_inputs = clip.tokenize(["a photo of a cat"]).to(device)

# Function to calculate and print cosine similarity
def print_cosine_similarity(image_features, text_features, model_name):
    cosine_sim = F.cosine_similarity(image_features, text_features)
    print(f"{model_name} Cosine Similarity:", cosine_sim.item())

# 0. Load Original CLIP Model
original_clip = clip.load(clipmodel)[0].to(device).float()
original_clip.eval()
with torch.no_grad():
    image_features = original_clip.encode_image(image_input)
    text_features = original_clip.encode_text(text_inputs)
    logits_per_image, logits_per_text = original_clip(image_input, text_inputs)
    print("Original CLIP Results:")
    print("Logits per Image:", logits_per_image)
    print("Logits per Text:", logits_per_text)
    print_cosine_similarity(image_features, text_features, "Original CLIP")


# 1. Load the Full Model Object
print("\nLoading Full Model Object...")
full_model = torch.load(full_model_path).to(device)
full_model.eval().float()
with torch.no_grad():
    image_features = full_model.encode_image(image_input)
    text_features = full_model.encode_text(text_inputs)
    logits_per_image, logits_per_text = full_model(image_input, text_inputs)
    print("Full Model Object Results:")
    print("Logits per Image:", logits_per_image)
    print("Logits per Text:", logits_per_text)
    print_cosine_similarity(image_features, text_features, "Full Model Object")


# 2. Load the Model from State Dictionary
print("\nLoading Model from State Dictionary...")
state_dict_model = clip.load(clipmodel)[0]  # Create an empty model instance of the correct architecture
state_dict = torch.load(state_dict_path, map_location=device)
state_dict_model.load_state_dict(state_dict)
state_dict_model = state_dict_model.to(device).float()

state_dict_model.eval()
with torch.no_grad():
    image_features = state_dict_model.encode_image(image_input)
    text_features = state_dict_model.encode_text(text_inputs)
    logits_per_image, logits_per_text = state_dict_model(image_input, text_inputs)
    print("State Dictionary Model Results:")
    print("Logits per Image:", logits_per_image)
    print("Logits per Text:", logits_per_text)
    print_cosine_similarity(image_features, text_features, "State Dictionary Model")

# 3. Load the JiT-Traced Model   
print("\nLoading JIT-Traced Model...")
jit_model = torch.jit.load(jit_model_path).to(device).float()
jit_model.eval()
with torch.no_grad():
    # Directly pass both inputs through the model, as tracing only captures the forward pass
    logits_per_image, logits_per_text = jit_model(image_input, text_inputs)
    print("JIT Model Results:")
    print("Logits per Image:", logits_per_image)
    print("Logits per Text:", logits_per_text)
    # Create a new CLIP model instance to hold the structure
    jit_injected_model = clip.load(clipmodel, jit=True)[0].to(device).float()
    jit_injected_model.eval()

    # Inject the openai/clip JIT model's forward function for use with the torch.jit loaded fine-tune
    # RecursiveScriptModule does not have encode_image, encode_text, and so on.
    jit_injected_model.forward = lambda image_input, text_inputs: jit_model(image_input, text_inputs)
    print_cosine_similarity(image_features, text_features, "Injected JIT Model")
    
print("\nDone. Enjoy scratching your head about the diff in floating-point numerical precision!")