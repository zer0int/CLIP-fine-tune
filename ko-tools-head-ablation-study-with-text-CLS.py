import os
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import pandas as pd
from safetensors.torch import load_file
import argparse
import matplotlib.pyplot as plt

import attnclip as clip
from attnclip.model import convert_state_dict_inproj_to_qkv

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
# >>>  ADVANCED TOOL  <<<
# Performs an ablation study for individual attention heads,
# plots the results (delta for change 'if head n lost')
# -----------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Head Ablation Study vs. CLS with Text Encoder')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model
output_dir = "results_oai/concept_headablation"
os.makedirs(output_dir, exist_ok=True)

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


def ablate_head_output_all_layers(model, head_idx):
    # Returns hook handles; pass head_idx=None to do nothing (baseline)
    handles = []
    if head_idx is None:
        return handles  # No ablation
    for block in model.visual.transformer.resblocks:
        def make_hook(head_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_out = output[0]
                    rest = output[1:]
                else:
                    attn_out = output
                    rest = ()
                B, N, D = attn_out.shape
                output_reshaped = attn_out.view(B, N, module.num_heads, module.head_dim)
                output_reshaped[:, :, head_idx, :] = 0.0
                attn_out_new = output_reshaped.view(B, N, D)
                if rest:
                    return (attn_out_new, *rest)
                else:
                    return attn_out_new
            return hook_fn
        handles.append(block.attn.register_forward_hook(make_hook(head_idx)))
    return handles

def remove_all_hooks(handles):
    for h in handles:
        h.remove()

def load_img(img_path):
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    return img

def get_patch_embeddings_per_layer(model, image_input):
    with torch.no_grad():
        x = model.visual.conv1(image_input)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_token = model.visual.class_embedding.to(x.dtype).to(device) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype).to(device)
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        patch_activations = []
        for block in model.visual.transformer.resblocks:
            x = block(x)
            patch_activations.append(x.permute(1, 0, 2).clone())
        return patch_activations

def get_visual_proj(x, model):
    with torch.no_grad():
        x = x.to(device)
        x = model.visual.ln_post(x)
        if hasattr(model.visual, 'proj'):
            x = x @ model.visual.proj
        x = F.normalize(x, p=2, dim=-1)
        return x

def get_text_features(model, text):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

img_files = [
    "cathalf1.png", "cathalf2.png", "cathalf3.png",
    "carhalf1.png", "carhalf2.png", "carhalf3.png",
    "doghalf1.png", "doghalf2.png", "doghalf3.png",
    "catblankright.png", "dogblankright.png", "carblankright.png",
    "blankleft.png", "blankright.png", "dogcat.png", "catdog.png"
]
words = ["cat", "car", "dog", "text", "a cat", "a car", "a dog", "a text", "a photo of a cat", "a photo of a car", "a photo of a dog", "a photo of a text"]

def run_head_ablation_and_save(head_idx, run_tag):
    # Ablate (or not), run analysis, save CSVs and npys
    if head_idx is not None:
        hook_handles = ablate_head_output_all_layers(model, head_idx)
    else:
        hook_handles = []

    results_cls = {}   # {(img, word): [cos_sim per layer]}
    results_mean = {}
    all_layer_matrix_cls = {}
    all_layer_matrix_mean = {}

    for img_name in img_files:
        img = load_img(os.path.join("image_sets/conceptfusion", img_name))
        patch_acts = get_patch_embeddings_per_layer(model, img)
        mat_cls = []
        mat_mean = []
        for word in words:
            text_feat = get_text_features(model, word)
            sims_cls = []
            sims_mean = []
            for act in patch_acts:
                act = act[0].to(device)
                cls = act[0:1, :]
                patches = act[1:, :]
                mean_patch = patches.mean(0, keepdim=True)
                vis_cls = get_visual_proj(cls, model)
                vis_mean = get_visual_proj(mean_patch, model)
                sim_cls = (vis_cls @ text_feat.T).item()
                sim_mean = (vis_mean @ text_feat.T).item()
                sims_cls.append(sim_cls)
                sims_mean.append(sim_mean)
            results_cls[(img_name, word)] = sims_cls
            results_mean[(img_name, word)] = sims_mean
            mat_cls.append(sims_cls)
            mat_mean.append(sims_mean)
        all_layer_matrix_cls[img_name] = np.stack(mat_cls, axis=1)
        all_layer_matrix_mean[img_name] = np.stack(mat_mean, axis=1)

    layer_count = len(next(iter(results_cls.values())))
    # Save as CSV (flatten columns as "img_word" for ease of later analysis)
    colnames = [f"{img}_{word}" for img in img_files for word in words]
    data_cls = np.array([results_cls[(img, word)] for img in img_files for word in words]).reshape(len(img_files)*len(words), layer_count).T
    data_mean = np.array([results_mean[(img, word)] for img in img_files for word in words]).reshape(len(img_files)*len(words), layer_count).T
    df_cls = pd.DataFrame(data_cls, columns=colnames, index=[f"layer_{i}" for i in range(layer_count)])
    df_mean = pd.DataFrame(data_mean, columns=colnames, index=[f"layer_{i}" for i in range(layer_count)])

    df_cls.to_csv(os.path.join(output_dir, f"headablation_{run_tag}_cls.csv"))
    df_mean.to_csv(os.path.join(output_dir, f"headablation_{run_tag}_patchmean.csv"))

    # Save all-layer matrices as .npy per image (optional, but useful for plotting deltas)
    for img_name in img_files:
        np.save(os.path.join(output_dir, f"headablation_{run_tag}_matrix_cls_{img_name}.npy"), all_layer_matrix_cls[img_name])
        np.save(os.path.join(output_dir, f"headablation_{run_tag}_matrix_patchmean_{img_name}.npy"), all_layer_matrix_mean[img_name])

    # Remove hooks if applied
    remove_all_hooks(hook_handles)

# --- Run baseline (no ablation) ---
print("Running baseline (no ablation)...")
run_head_ablation_and_save(None, "baseline")

# --- Run all head ablations ---
for head_idx in range(16):
    print(f"Running ablation for head {head_idx}...")
    run_head_ablation_and_save(head_idx, f"head{head_idx}")

print("All runs complete. Files saved to 'concept_headablation/'.\nRunning delta/difference analysis....")

# Setup paths, image and word lists
output_dir = "results_oai/concept_headablation"

img_files = [
    "cathalf1.png", "cathalf2.png", "cathalf3.png",
    "carhalf1.png", "carhalf2.png", "carhalf3.png",
    "doghalf1.png", "doghalf2.png", "doghalf3.png",
    "catblankright.png", "dogblankright.png", "carblankright.png",
    "blankleft.png", "blankright.png", "dogcat.png", "catdog.png"
]
words = ["cat", "car", "dog", "text", "a cat", "a car", "a dog", "a text", 
         "a photo of a cat", "a photo of a car", "a photo of a dog", "a photo of a text"]

layers = 24  # For ViT-L/14

# --- GROUP DEFS ---
group_map = {}
for img in img_files:
    if "half" in img:
        if "cat" in img: group_map[img] = "cat+word"
        if "dog" in img: group_map[img] = "dog+word"
        if "car" in img: group_map[img] = "car+word"
    elif "blankright" in img or "blankleft" in img:
        group_map[img] = "justcat" if "cat" in img else "justdog" if "dog" in img else "justcar" if "car" in img else "justblank"
    elif "blank" in img:
        group_map[img] = "justblank"
    elif "catblankright" in img:
        group_map[img] = "justword"
    elif "dogcat" in img or "catdog" in img:
        group_map[img] = "dogcat"
    else:
        group_map[img] = "other"

# For your groupings, tweak as needed:
group_map.update({
    "catblankright.png": "justword", "dogblankright.png": "justword", "carblankright.png": "justword",
    "blankleft.png": "justcat", "blankright.png": "justcat",
    "dogcat.png": "dogcat", "catdog.png": "dogcat"
})

# --- Helper for colname ---
def colname(img, word):
    return f"{img}_{word}"

# --- Load baseline (for layer indexing) ---
df_base_cls = pd.read_csv(os.path.join(output_dir, "headablation_baseline_cls.csv"), index_col=0)
# Get layers 1-21 indices (not including 0, not including 22/23)
layer_indices = [i for i in range(1, 22)]

# --- Load all head deltas ---
head_deltas = {h: {} for h in range(16)}  # {head: {(img, word): [delta_by_layer]}}

for head in range(16):
    df_cls = pd.read_csv(os.path.join(output_dir, f"headablation_head{head}_cls.csv"), index_col=0)
    # Layer index (ensure correct, not string)
    df_cls.index = df_cls.index.str.extract(r"(\d+)$").astype(int)[0]
    for img in img_files:
        for word in words:
            col = colname(img, word)
            if col in df_cls.columns and col in df_base_cls.columns:
                base = df_base_cls[col].astype(float).values
                ablated = df_cls[col].astype(float).values
                # delta only up to layer 21
                delta = ablated[layer_indices] - base[layer_indices]
                head_deltas[head][(img, word)] = delta

# --- Image groups ---
group_imgs = {}
for img in img_files:
    g = group_map.get(img, "other")
    group_imgs.setdefault(g, []).append(img)
group_names = list(group_imgs.keys())

# --- 1. SUMMARY OF ALL LAYERS (UP TO 21) BY GROUP ---
print("\n=== 1. MAX ABS DELTA PER GROUP, LAYERS 1-21 ===")
summary_rows = []
for head in range(16):
    for g in group_names:
        maxabs, mean = 0, 0
        n = 0
        vals = []
        for img in group_imgs[g]:
            for word in words:
                delta = head_deltas[head].get((img, word))
                if delta is not None:
                    vals.append(np.abs(delta))
        if vals:
            vals = np.concatenate(vals)
            maxabs = np.max(vals)
            mean = np.mean(vals)
        print(f"Head {head:2d}, Group '{g:10s}': max_abs={maxabs:.4f}, mean_abs={mean:.4f}")
        summary_rows.append(dict(head=head, group=g, max_abs=maxabs, mean_abs=mean))

# --- 2. BLOCKS OF THREE LAYERS, ALL IMAGES ---
print("\n=== 2. MAX DELTA PER 3-LAYER BLOCK, ALL IMAGES ===")
blocks = [
    (1,3), (4,6), (7,9), (10,12), (13,15), (16,18), (19,21)
]
block_indices = [list(range(b[0], b[1]+1)) for b in blocks]
block_labels = [f"{b[0]}-{b[1]}" for b in blocks]
for head in range(16):
    print(f"\nHead {head:2d}:")
    for block, label in zip(block_indices, block_labels):
        vals = []
        for img in img_files:
            for word in words:
                delta = head_deltas[head].get((img, word))
                if delta is not None:
                    # restrict to layers in block (layer_indices is 1-based, so offset for 0-based array)
                    block_vals = [delta[i-1] for i in block if (i-1) < len(delta)]
                    vals.extend(np.abs(block_vals))
        if vals:
            maxblock = np.max(vals)
            meanblock = np.mean(vals)
            print(f"  Block {label:7s}: max_abs={maxblock:.4f}, mean_abs={meanblock:.4f}")

# --- 3. BLOCKS OF THREE LAYERS BY GROUP ---
print("\n=== 3. BLOCKS OF THREE LAYERS, BY GROUP ===")
for head in range(16):
    print(f"\nHead {head:2d}:")
    for g in group_names:
        for block, label in zip(block_indices, block_labels):
            vals = []
            for img in group_imgs[g]:
                for word in words:
                    delta = head_deltas[head].get((img, word))
                    if delta is not None:
                        block_vals = [delta[i-1] for i in block if (i-1) < len(delta)]
                        vals.extend(np.abs(block_vals))
            if vals:
                maxblock = np.max(vals)
                meanblock = np.mean(vals)
                print(f"  Group '{g:10s}', Block {label:7s}: max_abs={maxblock:.4f}, mean_abs={meanblock:.4f}")


print("\n\n")
# --- GROUP DEFINITIONS ---
no_text_imgs = {"blankleft.png", "blankright.png", "dogcat.png", "catdog.png"}
only_text_imgs = {"carblankright.png", "catblankright.png", "dogblankright.png"}
both_imgs = set(img_files) - no_text_imgs - only_text_imgs

group_lookup = {}
for img in img_files:
    if img in no_text_imgs:
        group_lookup[img] = "No Text"
    elif img in only_text_imgs:
        group_lookup[img] = "Only Text"
    else:
        group_lookup[img] = "Both"

groupnames = ["No Text", "Only Text", "Both"]

# --- Threshold ---
THRESH = 0.01

# --- For each layer and group, find heads that matter ---
layers = 21  # layers 1..21
results = {layer: {g: [] for g in groupnames} for layer in range(1, 22)}

for layer in range(1, 22):
    for head in range(16):
        # For each group, collect all abs(delta) in this layer, for all group imgs, all words
        for group in groupnames:
            vals = []
            for img in img_files:
                if group_lookup[img] == group:
                    for word in words:
                        delta = head_deltas[head].get((img, word))
                        if delta is not None and (layer-1) < len(delta):
                            vals.append(abs(delta[layer-1]))
            # If any value exceeds threshold, "head matters"
            if any(v > THRESH for v in vals):
                results[layer][group].append(head)

# --- Print results ---
for layer in range(1, 22):
    for group in groupnames:
        heads = results[layer][group]
        head_str = ", ".join(str(h) for h in heads) if heads else "none"
        print(f"Layer {layer:2d}, Group {group:9s}: Head {head_str}")

# --- Load baseline ---
df_base_cls = pd.read_csv(os.path.join(output_dir, "headablation_baseline_cls.csv"), index_col=0)
df_base_patch = pd.read_csv(os.path.join(output_dir, "headablation_baseline_patchmean.csv"), index_col=0)

# --- Helper for column names ---
def colname(img, word):
    return f"{img}_{word}"

# --- Prepare per-head data ---
head_deltas = {h: {} for h in range(16)}  # {head: {(img, word): delta array}}
summary_rows = []

for head in range(16):
    df_cls = pd.read_csv(os.path.join(output_dir, f"headablation_head{head}_cls.csv"), index_col=0)
    # Make sure index is int for layers
    df_cls.index = df_cls.index.str.extract(r"(\d+)$").astype(int)[0]

    for img in img_files:
        for word in words:
            col = colname(img, word)
            if col not in df_cls.columns:
                print(f"Warning: missing {col} in head {head} file")
                continue
            base = df_base_cls[col].astype(float).values
            ablated = df_cls[col].astype(float).values
            delta = ablated - base
            head_deltas[head][(img, word)] = delta

            # --- Collect summary info ---
            max_idx = np.argmax(np.abs(delta))
            summary_rows.append({
                "head": head,
                "img": img,
                "word": word,
                "max_delta": float(delta[max_idx]),
                "layer_max_delta": int(df_cls.index[max_idx]),
                "delta_mean": float(delta.mean()),
                "delta_absmax": float(np.abs(delta).max())
            })

# --- Save summary table ---
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(output_dir, "ablation_summary_table.csv"), index=False)

# Use a high-contrast 8-color palette (e.g., Tableau, or any preferred)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]
STYLES = ["-", "--"]

for img in img_files:
    for word in words:
        plt.figure(figsize=(12, 7))
        for head in range(16):
            color = COLORS[head % 8]
            linestyle = STYLES[head // 8]  # 0 for heads 0–7, 1 for 8–15
            delta = head_deltas[head].get((img, word))
            if delta is not None:
                plt.plot(range(len(delta)), delta, label=f"head {head}",
                         color=color, linestyle=linestyle, linewidth=2 if head in (0,8) else 1)
        plt.axhline(0, color='k', linestyle=':', alpha=0.7, lw=1)
        plt.xlabel("ViT Layer")
        plt.ylabel("Delta (Ablated - Baseline) Cosine similarity")
        plt.title(f"CLS: Head ablation effect on '{img}' / '{word}'\n(Delta vs baseline)")
        plt.legend(fontsize="small", ncol=2, loc="upper right", title="Head (solid=0-7, dashed=8-15)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"plot_ablation_delta_CLS_{img}_{word}.png"), dpi=200)
        plt.close()


# --- Print summary ---
print("\nBIGGEST observed deltas per head, image, word (sorted by abs max):\n")
print(df_summary.sort_values("delta_absmax", ascending=False).head(40).to_string(index=False))

print("\nFor full statistics, see ablation_summary_table.csv and plots in 'concept_headablation/'.")