import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from safetensors.torch import load_file
from glob import glob
import cv2

# attnclip only returns attention as QKV
# attnclipindiv also exposes attention weights
import attnclipindiv as clip
from attnclipindiv.model import convert_state_dict_inproj_to_qkv
from cliptools import fix_random_seed
fix_random_seed() # deterministic backends, fixed seed

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------
# Visualizes Attention Per-Head
#
# Only useful for initial layers,
# as attention heads later 'mix'
# (register neuron emergence)
# and are not interpretable on their own.
#
# Early layers: 'striped' position heads,
# edge-attendings heads, and so on.
# ---------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize CLIP Vision Transformer Attention Heatmaps, per head, per layer')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--image_folder', default="image_sets/images_for_attn", help="Folder with images, matching for .txt files: 'image.png' -> 'tokens_image.txt'")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

if model_name_or_path.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    state_dict = load_file(model_name_or_path)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        msg = str(e)
        if (
            "Missing key(s) in state_dict" in msg or
            "Unexpected key(s) in state_dict" in msg
        ):
            print("State dict format mismatch, attempting QKV conversion...")
            state_dict = convert_state_dict_inproj_to_qkv(state_dict)
            try:
                model.load_state_dict(state_dict)
                print("OK!")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load state_dict after QKV conversion: {e2}\n"
                    f"Original error: {e}"
                )
        else:
            raise
else:
    print("Detected non-.safetensors file or name. Attempting to load model...")
    model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float()

# Folders
image_folder = args.image_folder
output_folder = "results_oai/attn-heatmaps-perhead"
os.makedirs(output_folder, exist_ok=True)

# ---- Settings ----
HEAD_TOKEN_IDX = 0  # 0 = CLS token; change to view attention for another patch
patch_size = model.visual.conv1.weight.shape[-1]
grid_size = round((model.visual.positional_embedding.shape[0] - 1) ** 0.5)
MAX_PATCHES = grid_size  # e.g. 16 for ViT-L/14


def save_attention_map(attn, orig_image, layer, head, out_path):
    # attn: [seq_len,] attention distribution for this head
    # orig_image: [3, H, W] tensor (0-1 range), already resized
    # out_path: file to save the heatmap

    # Ignore CLS token, visualize only patch-patch (if desired)
    patch_attn = attn[1:]  # [seq_len-1]
    num_patches = patch_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    assert grid_size * grid_size == num_patches

    # Normalize and reshape
    patch_attn = patch_attn.cpu().detach().numpy()
    patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.ptp() + 1e-9)
    patch_attn = patch_attn.reshape(grid_size, grid_size)
    patch_attn = cv2.resize(patch_attn, orig_image.size[::-1], interpolation=cv2.INTER_CUBIC)

    # Convert PIL image to array
    img = np.array(orig_image).astype(np.float32) / 255.0

    # Overlay heatmap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(patch_attn)[:, :, :3]
    overlay = 0.4 * heatmap + 0.6 * img
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(overlay)
    plt.title(f'Layer {layer}, Head {head}')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def summarize_attention(attn, layer, head, image_name):
    # attn: [seq_len] (one head, one token)
    # Print summary stats
    patch_attn = attn[1:]  # skip CLS
    max_idx = patch_attn.argmax()
    max_value = patch_attn.max().item()
    mean = patch_attn.mean().item()
    std = patch_attn.std().item()
    grid_size = round((model.visual.positional_embedding.shape[0] - 1) ** 0.5)
    y, x = divmod(max_idx.item(), grid_size)
    print(f"[{image_name}] Layer {layer} Head {head}: Max={max_value:.3f} at (y={y}, x={x}), Mean={mean:.3f}, Std={std:.3f}")


image_files = sorted(glob(os.path.join(image_folder, "*.png")))

for img_idx, img_file in enumerate(image_files):
    image_name = os.path.splitext(os.path.basename(img_file))[0]
    orig_img = Image.open(img_file).convert("RGB")
    img = preprocess(orig_img).unsqueeze(0).to(device)

    # Forward pass to populate attn_probs
    with torch.no_grad():
        _ = model.encode_image(img)

    # Extract blocks
    blocks = list(model.visual.transformer.resblocks)
    n_layers = len(blocks)
    n_heads = blocks[0].attn.num_heads

    # For each layer and each head
    for layer, block in enumerate(blocks):
        attn_probs = block.attn_probs  # [batch, heads, seq, seq]
        print("attention probs shape:", attn_probs.shape)
        attn_map = attn_probs[0]  # [heads, seq, seq], batch 0

        for head in range(n_heads):
            # Attention from CLS (token 0) to all others (can swap for patch-patch)
            attn = attn_map[head, HEAD_TOKEN_IDX, :]  # [seq_len]
            out_dir = os.path.join(output_folder, f"{image_name}")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"layer{layer:02d}_head{head:02d}.png")

            # Visualize and save
            save_attention_map(attn, orig_img, layer, head, out_file)
            summarize_attention(attn, layer, head, image_name)

print("Done. See attn-heatmaps-perhead/ for results.")

