import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
from safetensors.torch import load_file
import argparse

# attnclip only returns attention as QKV
# attnclipindiv also exposes attention weights
import attnclipindiv as clip
from attnclipindiv.model import convert_state_dict_inproj_to_qkv

# Suppress warnings spam from torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# >>>  ADVANCED TOOL  <<<
# Compares attention head behavior (pretrained and fine-tuned).
# For each layer, computes:  
# per-head attention entropy (diversity), head-to-head correlation, 
# mean pairwise saliency alignment (IoU of most-attended tokens)
# Reveals attention specialization, redundancy, and collapse.
# ------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Corr Entropy')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model


MODEL_CONFIGS = [
    {
        "name": model_name_or_path,
        "alias": "fine-tuned"
    },
    {
        "name": "ViT-L/14",
        "alias": "pre-trained"
    },
    # Add more models as needed...
]

RESULTS_DIR = "results_oai/head_corr_entropy_sal"
IMG_PATH = "image_sets/conceptfusion"
IMG_FILES = [
    "cathalf1.png", "cathalf2.png", "cathalf3.png",
    "carhalf1.png", "carhalf2.png", "carhalf3.png",
    "doghalf1.png", "doghalf2.png", "doghalf3.png",
    "catblankright.png", "dogblankright.png", "carblankright.png",
    "blankleft.png", "blankright.png", "dogcat.png", "catdog.png"
]
os.makedirs(RESULTS_DIR, exist_ok=True)


def per_head_entropy(attn_probs):
    """Returns entropy for each head [num_heads], mean, std."""
    num_heads, seq, _ = attn_probs.shape
    entropies = []
    for h in range(num_heads):
        attn = attn_probs[h]
        # Average across queries (axis=0) to get marginal distribution over keys
        marginals = attn.mean(axis=0)
        marginals = marginals / (marginals.sum() + 1e-12)
        ent = scipy.stats.entropy(marginals + 1e-12, base=np.e)
        entropies.append(ent)
    entropies = np.array(entropies)
    return entropies, float(entropies.mean()), float(entropies.std())

def per_head_correlation(attn_probs):
    """Returns (num_heads x num_heads) correlation matrix for attention patterns across heads."""
    num_heads, seq, _ = attn_probs.shape
    # Flatten per-head attention maps: [num_heads, seq*seq]
    flat = attn_probs.reshape(num_heads, -1)
    corr = np.corrcoef(flat)
    return corr  # [num_heads, num_heads]

def saliency_alignment(attn_probs):
    """Returns average pairwise overlap (IoU) of top attended locations across heads."""
    num_heads, seq, _ = attn_probs.shape
    ious = []
    # For each pair of heads, compute IoU of most attended positions (argmax along key axis)
    top_indices = attn_probs.argmax(axis=2)  # [num_heads, seq] (per query, top key idx)
    for h1 in range(num_heads):
        for h2 in range(h1+1, num_heads):
            overlap = (top_indices[h1] == top_indices[h2]).sum()
            union = seq  # since for each query, there is 1 top index per head
            ious.append(overlap / union)
    return np.mean(ious) if ious else 0.0


def main():
    for config in MODEL_CONFIGS:
        MODEL_NAME = config["name"]
        MODEL_ALIAS = config["alias"]

        print(f"\n=== Analyzing Model: {MODEL_ALIAS} ===")

        if MODEL_NAME.endswith(".safetensors"):
            print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
            model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
            state_dict = load_file(MODEL_NAME)
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
            model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)

        #model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)
        model.eval().float()

        num_layers = len(model.visual.transformer.resblocks)
        num_heads = model.visual.transformer.resblocks[0].attn.num_heads

        # Store per-layer metrics
        layer_entropy_means, layer_entropy_stds = [], []
        layer_iou_means = []
        layer_corr_means = []

        for layer_idx in range(num_layers):
            head_entropies = []
            ious = []
            corrs = []

            for img_file in tqdm(IMG_FILES, desc=f"{MODEL_ALIAS} Layer {layer_idx:02d}"):
                img_path = os.path.join(IMG_PATH, img_file)
                img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

                with torch.no_grad():
                    _ = model.encode_image(img)

                block = model.visual.transformer.resblocks[layer_idx]
                attn_probs = block.attn_probs.squeeze(0).cpu().numpy()  # [num_heads, seq, seq]

                # Per-head entropy
                ent, ent_mean, ent_std = per_head_entropy(attn_probs)
                head_entropies.append(ent)
                # Per-head saliency alignment
                ious.append(saliency_alignment(attn_probs))
                # Per-head correlation
                corr = per_head_correlation(attn_probs)
                # To summarize: mean off-diagonal correlation
                mean_corr = (corr.sum() - np.trace(corr)) / (num_heads**2 - num_heads)
                corrs.append(mean_corr)

            # Layer summary
            head_entropies = np.stack(head_entropies)  # [num_images, num_heads]
            layer_entropy_means.append(head_entropies.mean())
            layer_entropy_stds.append(head_entropies.std())
            layer_iou_means.append(np.mean(ious))
            layer_corr_means.append(np.mean(corrs))

        layers = list(range(num_layers))
        # Plotting: Entropy
        plt.figure(figsize=(8,4))
        plt.errorbar(layers, layer_entropy_means, yerr=layer_entropy_stds, marker='o', capsize=3)
        plt.title(f"Per-head Attention Entropy per Layer, {MODEL_ALIAS}")
        plt.xlabel("Layer")
        plt.ylabel("Entropy")
        plt.ylim(2, 6)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"entropy_per_layer_{MODEL_ALIAS}.png"))
        plt.close()

        # Plotting: Mean pairwise head saliency alignment (IoU)
        plt.figure(figsize=(8,4))
        plt.plot(layers, layer_iou_means, marker='^')
        plt.title(f"Mean Pairwise Head Saliency Alignment (IoU) per Layer, {MODEL_ALIAS}")
        plt.xlabel("Layer")
        plt.ylabel("Mean IoU (Headwise Saliency)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"saliency_alignment_per_layer_{MODEL_ALIAS}.png"))
        plt.close()

        # Plotting: Mean pairwise head correlation
        plt.figure(figsize=(8,4))
        plt.plot(layers, layer_corr_means, marker='s')
        plt.title(f"Mean Pairwise Head Attention Correlation per Layer, {MODEL_ALIAS}")
        plt.xlabel("Layer")
        plt.ylabel("Mean Off-diagonal Corr")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"head_corr_per_layer_{MODEL_ALIAS}.png"))
        plt.close()

        # Print summary table
        print(f"\n===== {MODEL_ALIAS}: Entropy / Saliency IoU / Head Corr (per-layer) =====")
        for l in range(num_layers):
            print(f"Layer {l:02d}: Entropy={layer_entropy_means[l]:.3f}  IoU={layer_iou_means[l]:.3f}  Corr={layer_corr_means[l]:.3f}")
        print(f"Results saved in: {RESULTS_DIR}/\n")

if __name__ == "__main__":
    main()
