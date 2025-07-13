import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import argparse

import attnclip as clip
from attnclip.model import convert_state_dict_inproj_to_qkv

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# >>>  ADVANCED TOOL  <<<
# Plots Layer-Wise K-Norms (Attn Head Key Norms) vs. Features
# (Features, as in: MLP expanded dimension Neurons)
# ------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description='Measure Influence of Attention Heads')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

output_folder = "results_oai/attn_heads_k_norms"
os.makedirs(output_folder, exist_ok=True)

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

model = model.float() # full precision

def get_expanded_k_norms(block, device='cuda'):
    d_model = block.mlp[0].in_features     # 1024 for ViT-L/14
    expanded_dim = block.mlp[0].out_features  # 4096 for ViT-L/14
    num_heads = block.attn.num_heads
    head_dim = d_model // num_heads

    # Prepare for [head, feature] K-norm matrix
    k_norms = np.zeros((num_heads, expanded_dim), dtype=np.float32)

    # For each expanded neuron, input is all zeros except that neuron
    for f in range(expanded_dim):
        x = torch.zeros(1, 1, d_model, device=device)
        # Pass through the MLP (c_fc, gelu, c_proj)
        z = torch.zeros(1, 1, expanded_dim, device=device)
        z[0, 0, f] = 1.0  # Activate only expanded feature f
        x_proj = block.mlp[2](block.mlp[1](z))  # c_proj(gelu(z)) -> [1,1,1024]
        # Add norm (LayerNorm) if present (block.ln_2)
        x_proj = block.ln_2(x_proj)
        # Compute keys for all heads
        k_all = block.attn.k_proj(x_proj)  # [1,1,1024]
        k_heads = k_all.view(1, 1, num_heads, head_dim)
        for h in range(num_heads):
            k = k_heads[0, 0, h, :]
            k_norms[h, f] = k.norm().item()
    return k_norms  # shape: [num_heads, expanded_dim]

def find_collaborators(k_norms, threshold_quantile=0.98):
    """
    Returns:
        collaborator_dict: {feature: [head list]} (features used by >1 head)
        specialist_dict: {feature: [head]} (used by only one head)
        per_head_dict: {head: [feature list]}
    """
    num_heads, num_feats = k_norms.shape
    # Find threshold (high norm) for each head, e.g. 98th percentile
    thresholds = np.quantile(k_norms, threshold_quantile, axis=1)
    # Binary mask: [head, feature] is True if above threshold for this head
    mask = k_norms >= thresholds[:, None]
    collaborator_dict = {}
    specialist_dict = {}
    per_head_dict = {h: [] for h in range(num_heads)}

    for f in range(num_feats):
        heads = np.where(mask[:, f])[0].tolist()
        if len(heads) > 1:
            collaborator_dict[f] = heads
        elif len(heads) == 1:
            specialist_dict[f] = heads
        for h in heads:
            per_head_dict[h].append(f)
    return collaborator_dict, specialist_dict, per_head_dict

# === MAIN PIPELINE ===
num_layers = len(model.visual.transformer.resblocks)
num_heads = model.visual.transformer.resblocks[0].attn.num_heads

for layer in range(num_layers):
    print(f"Scanning layer {layer}...")
    block = model.visual.transformer.resblocks[layer]
    k_norms = get_expanded_k_norms(block, device=device)  # [num_heads, 4096]
    
    # Save heatmap plot
    plt.figure(figsize=(16, 5))
    plt.imshow(k_norms, aspect='auto', interpolation='nearest')
    plt.xlabel('Expanded feature (MLP neuron)')
    plt.ylabel('Head')
    plt.title(f'K-norm: head × feature, Layer {layer}')
    plt.colorbar(label='K-norm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'Layer{layer}_Knorm.png'))
    plt.close()

    # Save as .txt: collaborator, specialist, per-head features
    colab, spec, per_head = find_collaborators(k_norms, threshold_quantile=0.98)
    with open(os.path.join(output_folder, f'Layer{layer}_colab.txt'), "w") as f:
        f.write(f"=== Collaborator features (used by >1 head): ===\n")
        for feat, heads in sorted(colab.items()):
            f.write(f"Feature {feat}: heads {heads}\n")
        f.write(f"\n=== Specialist features (unique to one head): ===\n")
        for feat, heads in sorted(spec.items()):
            f.write(f"Feature {feat}: head {heads}\n")
        f.write(f"\n=== Features per head: ===\n")
        for h, feats in per_head.items():
            f.write(f"Head {h}: {feats}\n")

print(f"Collaboration scan complete. Results in: {output_folder}")
