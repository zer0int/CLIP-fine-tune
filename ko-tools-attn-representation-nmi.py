import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from safetensors.torch import load_file
import argparse

# attnclip only returns attention as QKV
# attnclipindiv also exposes attention weights
import attnclipindiv as clip
from attnclipindiv.model import convert_state_dict_inproj_to_qkv

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
# >>>  ADVANCED TOOL  <<<
# Analyzes per-layer / per-head attn representation stats:
# **diversity** (NMI, Normalized Mutual Information) 
# **representation collapse** (cos sim, L2 norm)
# -----------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Attention Head NMI and Cos Sim')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

MODEL_NAME = args.use_model

RESULTS_DIR = "results_oai/attention_head_nmi"
IMG_PATH = "image_sets/conceptfusion"
IMG_FILES = [
    "cathalf1.png", "cathalf2.png", "cathalf3.png",
    "carhalf1.png", "carhalf2.png", "carhalf3.png",
    "doghalf1.png", "doghalf2.png", "doghalf3.png",
    "catblankright.png", "dogblankright.png", "carblankright.png",
    "blankleft.png", "blankright.png", "dogcat.png", "catdog.png"
]

os.makedirs(RESULTS_DIR, exist_ok=True)

def attention_nmi_headwise(attn_probs):
    """
    attn_probs: [num_heads, seq_len, seq_len]
    Returns: headwise NMI array [num_heads], mean, std
    """
    num_heads, seq_len, _ = attn_probs.shape
    nmis = []
    for h in range(num_heads):
        attn = attn_probs[h] + 1e-12
        attn = attn / attn.sum()
        p_q = np.ones(seq_len) / seq_len
        p_k = attn.sum(axis=0) / attn.sum()
        joint = attn / attn.sum()
        Hq = -np.sum(p_q * np.log(p_q + 1e-12))
        Hk = -np.sum(p_k * np.log(p_k + 1e-12))
        pqk = joint / (joint.sum() + 1e-12)
        pqk_flat = pqk.flatten()
        MI = np.nansum(pqk_flat * np.log(pqk_flat / (np.outer(p_q, p_k).flatten() + 1e-12)))
        nmi = MI / (np.sqrt(Hq * Hk) + 1e-12)
        nmis.append(nmi)
    nmis = np.array(nmis)
    return nmis, float(np.mean(nmis)), float(np.std(nmis))

def avg_token_cosine_similarity(tokens):
    tokens = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
    sims = cosine_similarity(tokens)
    seq_len = tokens.shape[0]
    mask = np.triu_indices(seq_len, k=1)
    return float(sims[mask].mean()), float(sims[mask].std())

def main():
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

    model.eval().float()

    num_layers = len(model.visual.transformer.resblocks)
    num_heads = model.visual.transformer.resblocks[0].attn.num_heads

    l2_norm_cls_layers = []
    l2_norm_patch_layers = []
    l2_norm_all_layers = []

    layer_nmi_means = []
    layer_nmi_stds = []
    layer_cos_means = []
    layer_cos_stds = []
    all_headwise_nmis = []  # list of [num_layers][num_images][num_heads]

    for layer_idx in range(num_layers):
        nmi_means = []
        cos_vals = []
        headwise_nmis = []

        for img_file in tqdm(IMG_FILES, desc=f"Layer {layer_idx:02d}"):
            img_path = os.path.join(IMG_PATH, img_file)
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                _ = model.encode_image(img)

            block = model.visual.transformer.resblocks[layer_idx]
            attn_probs = block.attn_probs.squeeze(0).cpu().numpy()  # [num_heads, seq, seq]

            head_nmis, nmi_mean, nmi_std = attention_nmi_headwise(attn_probs)
            nmi_means.append(nmi_mean)
            headwise_nmis.append(head_nmis)

            # Representation collapse: tokens after this block
            with torch.no_grad():
                x = model.visual.conv1(img.type(model.visual.conv1.weight.dtype))
                x = x.reshape(1, x.shape[1], -1).permute(0, 2, 1)
                class_embedding = model.visual.class_embedding.to(x.dtype)
                x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, class_embedding.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
                x = x + model.visual.positional_embedding.to(x.dtype)
                x = model.visual.ln_pre(x)
                for i in range(layer_idx + 1):
                    x = model.visual.transformer.resblocks[i](x)
                tokens = x.squeeze(0).cpu().numpy()  # [seq, emb_dim]

                # L2 norms
                l2_norms = np.linalg.norm(tokens, axis=-1)  # [seq_len]
                l2_norms_cls = l2_norms[0]
                l2_norms_patch = l2_norms[1:]  # Standard CLIP: [CLS, PATCH...]
                l2_norms_all = l2_norms.mean()

                # Store mean values
                if layer_idx == 0:
                    if 'l2_norms_cls_list' not in locals():
                        l2_norms_cls_list, l2_norms_patch_list, l2_norms_all_list = [], [], []
                l2_norms_cls_list.append(l2_norms_cls)
                l2_norms_patch_list.append(l2_norms_patch.mean() if len(l2_norms_patch) > 0 else float("nan"))
                l2_norms_all_list.append(l2_norms.mean())


                cos_mean, cos_std = avg_token_cosine_similarity(tokens)
                cos_vals.append(cos_mean)

        l2_norm_cls_layers.append(np.mean(l2_norms_cls_list))
        l2_norm_patch_layers.append(np.mean(l2_norms_patch_list))
        l2_norm_all_layers.append(np.mean(l2_norms_all_list))


        # Save CSV for this layer (now includes headwise NMI)
        nmi_head_cols = {f"nmi_head{h:02d}": [hn[h] for hn in headwise_nmis] for h in range(num_heads)}
        df = pd.DataFrame({
            "image": IMG_FILES,
            "nmi_mean": nmi_means,
            "cosine": cos_vals,
            **nmi_head_cols
        })
        csv_path = os.path.join(RESULTS_DIR, f"layer_{layer_idx:02d}_results.csv")
        df.to_csv(csv_path, index=False)

        layer_nmi_means.append(np.mean(nmi_means))
        layer_nmi_stds.append(np.std(nmi_means))
        layer_cos_means.append(np.mean(cos_vals))
        layer_cos_stds.append(np.std(cos_vals))

        all_headwise_nmis.append(headwise_nmis)  # shape: [num_images][num_heads]

        print(f"Layer {layer_idx:02d}: NMI mean={np.mean(nmi_means):.4f} ± {np.std(nmi_means):.4f}, CosSim mean={np.mean(cos_vals):.4f} ± {np.std(cos_vals):.4f}")

    # ==== Summary Plots ====
    layers = list(range(num_layers))

    # NMI plot (mean ± std)
    plt.figure(figsize=(8,4))
    plt.errorbar(layers, layer_nmi_means, yerr=layer_nmi_stds, marker='o', capsize=3)
    plt.title(f"Attention NMI per Layer")
    plt.xlabel("Layer")
    plt.ylabel("NMI (higher=diverse, lower=collapse)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"nmi_per_layer.png"))
    plt.close()

    # Cosine sim plot
    plt.figure(figsize=(8,4))
    plt.errorbar(layers, layer_cos_means, yerr=layer_cos_stds, marker='s', capsize=3)
    plt.title(f"Avg. Cosine Similarity of Tokens per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean Cosine Similarity (higher=collapse)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"cosine_per_layer.png"))
    plt.close()

    # Headwise NMI boxplot per layer
    plt.figure(figsize=(14,6))
    data = []
    for layer_idx, headwise_nmis in enumerate(all_headwise_nmis):
        # headwise_nmis: [num_images][num_heads]
        arr = np.array(headwise_nmis)  # [num_images, num_heads]
        for h in range(num_heads):
            data.append(arr[:, h])
    
    # Boxplot: x = layer*num_heads + head, y = nmi value
    positions = [l + h/(num_heads+1) for l in layers for h in range(num_heads)]
    plt.boxplot(data, positions=positions, widths=0.6/num_heads, showfliers=False, patch_artist=True)
    for l in layers:
        plt.axvline(l - 0.5, color='gray', linestyle='--', linewidth=0.6, zorder=0)
    plt.xticks(layers, [f"L{l}" for l in layers])
    plt.title(f"Headwise Attention NMI per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Headwise NMI")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"nmi_headwise_boxplot.png"))
    plt.close()

    # Token L2 norm plot per layer
    plt.figure(figsize=(10, 5))
    plt.plot(layers, l2_norm_cls_layers, label='CLS', marker='o')
    plt.plot(layers, l2_norm_patch_layers, label='Patch (mean)', marker='s')
    plt.plot(layers, l2_norm_all_layers, label='All tokens (mean)', marker='x', linestyle='--', alpha=0.7)
    plt.title(f"Mean Token L2 Norm per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean L2 Norm")
    plt.legend()
    plt.ylim(6, 22)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"token_l2norm_per_layer.png"))
    plt.close()


    print(f"\n===== Summary (per-layer means over all images) =====")
    for l in range(num_layers):
        print(
            f"Layer {l:02d}: "
            f"NMI={layer_nmi_means[l]:.4f}  "
            f"CosSim={layer_cos_means[l]:.4f}  "
            f"L2_CLS={l2_norm_cls_layers[l]:.2f}  "
            f"L2_Patch={l2_norm_patch_layers[l]:.2f}  "
            f"L2_All={l2_norm_all_layers[l]:.2f}"
        )
    print(f"\nResults saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
