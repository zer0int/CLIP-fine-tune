import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import load_file
import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import clip
from clip.model import CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------
# This code plots the vision patch L2 norms 
# for each layer of the model. See also:
# ko-tools-visualize-patch-norms-ablation.py
# ------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize CLIP ViT Patch Norms (all layers)')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--image_folder', default="image_sets/special_attn_img", help="Folder with images to get norms for")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

if model_name_or_path.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    
    # Load ViT-L/14 explicitly
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # Load the safetensors state_dict and apply
    state_dict = load_file(model_name_or_path)
    model.load_state_dict(state_dict)

else:
    print("Detected non-.safetensors file or name. Attempting to load model...")
    
    # Load normally as per the existing logic
    model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float()

out_dir = "results_oai/patch-norm"
os.makedirs(out_dir, exist_ok=True)

img_dir = args.image_folder

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '_', text)

# --- Collect per-layer outputs (forward hook) ---
def get_all_layer_outputs(transformer, x):
    all_outputs = []

    def hook_fn(module, input, output):
        all_outputs.append(output.detach().clone())

    hooks = []
    for i, block in enumerate(transformer.resblocks):
        hooks.append(block.register_forward_hook(hook_fn))
    with torch.no_grad():
        _ = transformer(x)
    for h in hooks:
        h.remove()
    # Return [layer0, layer1, ..., layerN] (shape: seq, batch, dim)
    return all_outputs

# --- Process all layers ---
def clip_encode_image_all_layers(modelorg, image_input):
    with torch.no_grad():
        x = modelorg.visual.conv1(image_input)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = modelorg.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + modelorg.visual.positional_embedding.to(x.dtype)
        x = modelorg.visual.ln_pre(x)
        x = x.permute(1, 0, 2) # (seq, batch, dim)
        # --- Get all transformer outputs per layer ---
        all_layer_outputs = get_all_layer_outputs(modelorg.visual.transformer, x)
        # Convert to (batch, seq, dim)
        all_layer_outputs = [layer_out.permute(1, 0, 2) for layer_out in all_layer_outputs]
        # Optionally add ln_post to final layer's CLS token (for consistency with normal CLIP encode)
        return all_layer_outputs

def save_heatmap(patch_norms, cls_norm, image_name, text_token, layer_idx):
    patch_grid = patch_norms.reshape(16, 16)
    patch_max = patch_norms.max()
    vmin = min(patch_norms.min(), cls_norm)
    vmax = max(patch_norms.max(), cls_norm)
    fig = plt.figure(figsize=(7, 7))

    # Patch token heatmap (top)
    ax_patch = plt.axes([0.1, 0.35, 0.8, 0.6])
    im = ax_patch.imshow(patch_grid, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax_patch.set_title(f"Layer {layer_idx}: Patch Norm Heatmap: {image_name}")
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])
    cbar = plt.colorbar(im, ax=ax_patch, fraction=0.046, pad=0.04)
    cbar.set_label("L2 Norm")

    # Bottom info bar
    ax_bar = plt.axes([0.1, 0.15, 0.8, 0.11])
    ax_bar.axis("off")

    cmap = plt.get_cmap("viridis")
    normed_cls = (cls_norm - vmin) / (vmax - vmin + 1e-8)
    normed_patch = (patch_max - vmin) / (vmax - vmin + 1e-8)
    cls_color = cmap(normed_cls)
    patch_color = cmap(normed_patch)

    # Define fixed locations for left (CLS) and right (Patch Max)
    left_x = 0.02
    center_x = 0.41
    right_x = 0.68
    bar_y = 0.45
    bar_w = 0.07
    bar_h = 0.35

    # CLS
    ax_bar.text(left_x, bar_y + 0.13, "CLS:", ha="left", va="center", fontsize=13, fontweight="bold", transform=ax_bar.transAxes)
    ax_bar.text(left_x + 0.10, bar_y + 0.13, f"{cls_norm:.1f}", ha="left", va="center", fontsize=13, transform=ax_bar.transAxes)
    ax_bar.add_patch(plt.Rectangle((left_x + 0.19, bar_y), bar_w, bar_h, color=cls_color, transform=ax_bar.transAxes, clip_on=False))

    # Patch Max
    ax_bar.text(center_x, bar_y + 0.13, "Patch Max:", ha="left", va="center", fontsize=13, fontweight="bold", transform=ax_bar.transAxes)
    ax_bar.text(right_x, bar_y + 0.13, f"{patch_max:.1f}", ha="left", va="center", fontsize=13, transform=ax_bar.transAxes)
    ax_bar.add_patch(plt.Rectangle((right_x + 0.13, bar_y), bar_w, bar_h, color=patch_color, transform=ax_bar.transAxes, clip_on=False))

    ax_bar.set_title("CLS Token and Patch Max Norm", fontsize=12, pad=10)

    sanitized_token = sanitize_filename(text_token)
    filename = f"{out_dir}/layer{layer_idx:02d}_heatmap_{image_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()



def save_barplot(norms, image_name, text_token, layer_idx):
    sanitized_token = sanitize_filename(text_token)
    filename = f"{out_dir}/layer{layer_idx:02d}_l2norm_{image_name}.png"
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(norms)), norms)
    plt.xlabel("Token ID (0=CLS, 1-256=Patch)")
    plt.ylabel("L2 Norm")
    plt.title(f"Layer {layer_idx}: L2 Norms of ViT Tokens: {image_name}")
    plt.savefig(filename)
    plt.close()

# --- MAIN LOOP ---
csv_records = []

for img_file in tqdm(os.listdir(img_dir)):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_name = os.path.splitext(img_file)[0]
        image_path = os.path.join(img_dir, img_file)
        image = load_image(image_path)
        all_layer_outputs = clip_encode_image_all_layers(model, image)
        token = "dummy"
        # --- Per-layer visualization ---
        for layer_idx, layer_output in enumerate(all_layer_outputs):
            # Layer_output: shape (batch=1, seq=257, dim) -- [0]=CLS, [1:]=patches
            x = layer_output[0] # (seq, dim)
            cls_token = x[0:1]      # (1, dim)
            patch_tokens = x[1:257] # (256, dim)
            # Compute L2 norm for each token
            cls_norm = torch.norm(cls_token, dim=-1).item()
            patch_norms = torch.norm(patch_tokens, dim=-1).cpu().detach().numpy()
            # [CLS] + patch tokens
            norms = np.concatenate(([cls_norm], patch_norms), axis=0)
            save_heatmap(patch_norms, cls_norm, image_name, token, layer_idx)
            save_barplot(norms, image_name, token, layer_idx)
            for idx, norm in enumerate(norms):
                flag = "CLS" if idx == 0 else ""
                csv_records.append([layer_idx, idx, norm, flag, image_name, token])

df = pd.DataFrame(csv_records, columns=["Layer", "TokenID", "Norm", "Flag", "Image", "TextToken"])
df.to_csv(f"{out_dir}/patch_norms_layers.csv", index=False)

print(f"Processing complete. Check '{out_dir}' for visualizations and CSV output.")
