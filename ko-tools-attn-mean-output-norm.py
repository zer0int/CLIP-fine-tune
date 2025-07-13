import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import random
from cliptools import fix_random_seed
from safetensors.torch import load_file
import argparse
fix_random_seed() # deterministic backends, fixed seed

import attnclip as clip
from attnclip.model import convert_state_dict_inproj_to_qkv

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
# >>>  ADVANCED TOOL  <<<
# Plots mean L2 Output Norm for individual Attention Heads
#
# Optional, with ablated register neurons: --ablate_neurons
# For information about them, see: 
# https://github.com/zer0int/CLIP-test-time-registers
# -----------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plots mean L2 Attn Output Norm for all Attention Heads')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument("--ablate_neurons", action='store_true', help="Ablate the 13 known Register Neurons in Pre-Trained CLIP")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

ablate_neurons = False
if args.ablate_neurons:
    ablate_neurons = True

local_path = "results_oai/attention_heads_output_norm"
os.makedirs(local_path, exist_ok=True)
append_to_filename = "normal"

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

# ---- DATASET ----
base_img_path = "image_sets/conceptfusion"
img_files = [
    "cathalf1.png", "cathalf2.png", "cathalf3.png",
    "carhalf1.png", "carhalf2.png", "carhalf3.png",
    "doghalf1.png", "doghalf2.png", "doghalf3.png",
    "catblankright.png", "dogblankright.png", "carblankright.png",
    "blankleft.png", "blankright.png", "dogcat.png", "catdog.png"
]
words = ["a photo of a cat", "a photo of a car", "a photo of a dog", "a photo of a text"] # Not used in norm calculation

# Load all images and stack into a batch tensor
img_tensors = []
for fname in img_files:
    path = os.path.join(base_img_path, fname)
    img = Image.open(path).convert('RGB')
    img_tensors.append(preprocess(img))
images = torch.stack(img_tensors).to(device)


# ---- HOOK: Neuron Ablation ----
class FeatureScalerHook:
    def __init__(self, model, layer_idx, feature_idx, scale_factor, transformer_type='visual'):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_idx = feature_idx
        self.scale_factor = scale_factor
        self.transformer_type = transformer_type
        self.handle = None
        self.register_hook()

    def register_hook(self):
        def hook(module, input, output):
            output[:, :, self.feature_idx] *= self.scale_factor
            return output

        if self.transformer_type == 'visual':
            layer = self.model.visual.transformer.resblocks[self.layer_idx].mlp.c_fc
        else:
            layer = self.model.transformer.resblocks[self.layer_idx].mlp.c_fc
        self.handle = layer.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()

if ablate_neurons:
    append_to_filename = "ablated"
    print("---------------------------------------")
    print("WARNING!! Ablating Register Neurons!")
    print("Ensure this is what you want!")
    print("---------------------------------------")
    # ---------- HOOKS: ABLATE Register Neurons ---------
    # Register Neurons. For information about them + how to find them, see: https://github.com/zer0int/CLIP-test-time-registers
    top_activations_layer_11 = [9, 987, 1967, 2555, 3661, 3784] 
    top_activations_layer_12 = [42, 983, 1571, 2687, 3002, 3008, 3868]  

    # Scale factor 0 ablates them. 1 Does nothing. >1 amplifies them.
    hooks_layer_11 = []
    for feature_idx in top_activations_layer_11:
        hook = FeatureScalerHook(model, layer_idx=11, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
        hooks_layer_11.append(hook)

    hooks_layer_12 = []
    for feature_idx in top_activations_layer_12:
        hook = FeatureScalerHook(model, layer_idx=12, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
        hooks_layer_12.append(hook)
    # --------------------------------------------------

# ---- HOOK: Collect per-head L2s ----
vision_transformer = model.visual.transformer

num_layers = len(vision_transformer.resblocks)
num_heads = vision_transformer.resblocks[0].attn.num_heads

per_head_data = {l: [] for l in range(num_layers)}

def make_hook(layer_idx):
    def hook(module, input, output):
        seq_len, batch_size, embed_dim = output.shape
        num_heads = module.attn.num_heads
        head_dim = embed_dim // num_heads
        x = output.permute(1, 0, 2).contiguous()
        x = x.view(batch_size, seq_len, num_heads, head_dim)
        head_norms = x.norm(dim=-1).mean(dim=(0, 1)).detach().cpu().numpy()
        per_head_data[layer_idx].append(head_norms)
    return hook

hooks = []
for l, block in enumerate(vision_transformer.resblocks):
    hooks.append(block.register_forward_hook(make_hook(l)))

with torch.no_grad():
    _ = model.visual(images)

for h in hooks:
    h.remove()

# ---- Aggregate results ----
per_head_mean = np.zeros((num_layers, num_heads))
for l in range(num_layers):
    arrs = per_head_data[l]
    arrs = np.stack(arrs, axis=0)
    per_head_mean[l] = arrs.mean(axis=0)

# ---- Save to CSV ----
csv_path = f"{local_path}/per_head_norms-{append_to_filename}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "head", "mean_l2"])
    for l in range(num_layers):
        for h in range(num_heads):
            writer.writerow([l, h, per_head_mean[l, h]])
print(f"Wrote per-head mean L2 norms to {csv_path}")

# ---- Summary Print ----
print("\nPer-head mean L2 norm (layer major, then head):")
for l in range(num_layers):
    vals = " ".join([f"{per_head_mean[l, h]:.3f}" for h in range(num_heads)])
    print(f"Layer {l:2d}: {vals}")

# ---- Cumulative sum per head ----
cumsum_per_head = per_head_mean.cumsum(axis=0)
final_cumsum = cumsum_per_head[-1]

print("\nCumulative sum of mean L2 norms (by head):")
for h in range(num_heads):
    print(f"Head {h:2d}: {final_cumsum[h]:.3f}")

# ---- Save cumulative sum CSV ----
cum_csv_path = f"{local_path}/per_head_cumsum-{append_to_filename}.csv"
with open(cum_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["head", "cumsum"])
    for h in range(num_heads):
        writer.writerow([h, final_cumsum[h]])
print(f"Wrote cumulative sum per head to {cum_csv_path}")

# ---- Plot heatmap ----
plt.figure(figsize=(18, 6))
plt.imshow(per_head_mean.T, aspect="auto", cmap="viridis")
plt.colorbar(label="Mean L2 norm")
plt.xlabel("Block (Layer)")
plt.ylabel("Head")
plt.title("Vision Transformer: Per-head mean L2 norm (output tokens)")
plt.xticks(np.arange(num_layers))
plt.yticks(np.arange(num_heads))
plt.tight_layout()
plt.savefig(f"{local_path}/per_head_l2norm_heatmap-{append_to_filename}.png")
plt.close()
print("Saved per-head heatmap.")

# ---- Lineplot per head ----
plt.figure(figsize=(18, 5))
for h in range(num_heads):
    plt.plot(range(num_layers), per_head_mean[:, h], label=f"Head {h}")
plt.xlabel("Block (Layer)")
plt.ylabel("Mean L2 norm")
plt.title("Vision Transformer: Per-head mean L2 norm per block")
plt.legend(loc='upper right', ncol=4)
plt.tight_layout()
plt.savefig(f"{local_path}/per_head_l2norm_lineplot-{append_to_filename}.png")
plt.close()
print("Saved per-head lineplot.")

# ---- Plot cumulative sum per head ----
plt.figure(figsize=(12, 5))
plt.bar(range(num_heads), final_cumsum)
plt.xlabel("Head")
plt.ylabel("Cumulative sum of mean L2 norm")
plt.title("Vision Transformer: Cumulative sum per head (all blocks)")
plt.xticks(range(num_heads))
plt.tight_layout()
plt.savefig(f"{local_path}/per_head_l2norm_cumsum-{append_to_filename}.png")
plt.close()
print("Saved cumulative sum per head plot.")
