"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

CLIP ViT Visual-Stream Inspection on SCAM (Multi-Model, Per-Variant)
===================================================================

Quantify how CLIP’s *visual* transformer computes and routes information through:
(1) residual-stream decomposition (attention vs MLP), (2) per-head Q/K/V norms, and
(3) which MLP expanded features most strongly influence next-layer key vectors.
Core Measurements (per model × variant)
---------------------------------------
A) Per-layer residual-stream norm decomposition (patch tokens only; CLS excluded)
   For each visual block ℓ:
     x_in        = token stream entering block ℓ
     attn_delta  = Attention( LN1(x_in) )
     x_mid       = x_in + attn_delta
     mlp_delta   = MLP( LN2(x_mid) )
     x_out       = x_mid + mlp_delta
   Report mean L2 norms over (patch tokens, batch):
     E||x_in||, E||attn_delta||, E||mlp_delta||, E||x_out||

B) Per-head Q/K/V mean norms from LN1(x_in) (patch tokens only)
   For each block ℓ:
     q = q_proj(LN1(x_in)), k = k_proj(LN1(x_in)), v = v_proj(LN1(x_in))
   Split into heads and compute:
     For each head h:  E||Q_h||, E||K_h||, E||V_h||  over (patch tokens, batch)
   Also compute per-head cumulative key magnitude:
     K_cumsum[h] = Σ_ℓ  E||K_h||_ℓ

C) MLP feature activation statistics (expanded width = 4d, typically 4096)
   For each block ℓ:
     z = c_fc(LN2(x_mid))              # pre-gelu expanded features
     a = GELU(z)                       # post-gelu activations
   Compute featurewise mean absolute activation (patch tokens only):
     a_abs_mean[f] = E[ |a_f| ]

D) Gain maps: (block ℓ MLP feature) → (block ℓ+1 key head) sensitivity proxy
   Using weights only:
     Wproj = c_proj(ℓ) weight          # [d,4d]
     Wk    = k_proj(ℓ+1) weight        # [d,d], reshaped per head: Wk_h ∈ R^{Hd×d}
     M_h   = Wk_h @ Wproj              # [Hd,4d]
     weight_gain[h,f] = || M_h[:,f] ||_2
   Activation-weighted version:
     act_gain[h,f] = weight_gain[h,f] * a_abs_mean_ℓ[f]

E) Finite-difference perturbation on real residual stream (causal check)
   For each chosen layer ℓ and selected features f:
     Perturb x_out with feature-directed delta:
       x_out_pert = x_out + α * a_f * Wproj[:,f]
     Measure effect on next block’s key norms:
       ΔK_h = E||K_h||_pert - E||K_h||_base
   Aggregates ΔK_h per layer (heatmap: layers × heads) to validate gain-map heuristics.
"""

import os
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_dataset
import attnclipindiv as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything


# ============================================================
# Models: OpenAI / local path .pt .safetensors / HuggingFace Hub
# ============================================================

MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
    ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
    ("regr-norm", "zer0int/CLIP-Regression-ViT-L-14"),
    ("regr-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
]

OUT_ROOT: str = "out_eval_measure/attn_mlp_analysis_scam"
SEED: int = 42
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset: BLISS-e-V/SCAM
SCAM_DATASET_ID: str = "BLISS-e-V/SCAM"
SCAM_SPLIT: str = "train"
SCAM_VARIANTS: List[str] = ["NoSCAM", "SynthSCAM"]
SCAM_FRACTION: float = 0.1                          # Needs 40 GB VRAM if 1.0
MAX_SAMPLES_PER_VARIANT: Optional[int] = None       # set if you want a hard cap

# Plots / analysis
TOPM_GAIN_PLOT: int = 4096
TOPK_FINITE_DIFF: int = 4096        # how many features to use; ViT-L/14 has 4096
ALPHA: float = 0.25
FINITE_DIFF_LAYERS: str = "all"     # "all" or "10,11,12" or "0-22"


# Optional intervention: Ablate neurons [sets to 0]
ABLATE_REG_NEURONS: bool = False
REG_NEURONS: Dict[int, List[int]] = {
    11: [9, 987, 1967, 2555, 3661, 3784],
    12: [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868],
}


# Determinism
def fix_random_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def attach_reg_neuron_nuke_hooks(visual: torch.nn.Module):
    """
    Zero specified MLP expanded dims at c_fc output (pre-gelu), for blocks in REG_NEURONS.
    """
    if not REG_NEURONS:
        return
    print("[INFO] Attaching register-neuron nuke hooks...")

    for block_idx, block in enumerate(visual.transformer.resblocks):
        if block_idx not in REG_NEURONS:
            continue
        idx_tensor = torch.tensor(REG_NEURONS[block_idx], dtype=torch.long)

        c_fc = block.mlp.c_fc if hasattr(block.mlp, "c_fc") else block.mlp[0]

        def make_hook(idxs: torch.Tensor, blk_idx: int):
            def hook(module, input, output):
                out = output.clone()
                out[..., idxs] = 0.0
                return out
            hook.__name__ = f"reg_nuke_block_{blk_idx}"
            return hook

        c_fc.register_forward_hook(make_hook(idx_tensor, block_idx))
        print(f"[INFO] Hook attached on block {block_idx} c_fc for neurons {REG_NEURONS[block_idx]}")


# Dataset: SCAM (indices + shared sampling across models)
def prepare_scam_indices(
    dataset_id: str,
    split: str,
    variants: List[str],
) -> Tuple[object, Dict[str, List[int]]]:
    """
    Loads the HF dataset once and returns:
      - ds: datasets.Dataset
      - variant_to_indices: dict variant -> list of row indices
    """
    print(f"[INFO] Loading dataset: {dataset_id} [{split}]")
    ds = load_dataset(dataset_id, split=split)

    variant_to_indices: Dict[str, List[int]] = {v: [] for v in variants}

    for i, entry in enumerate(ds):
        sid = str(entry.get("id", ""))
        for v in variants:
            if sid.startswith(v):
                variant_to_indices[v].append(i)
                break

    for v in variants:
        print(f"[INFO] SCAM variant {v}: {len(variant_to_indices[v])} samples indexed")

    return ds, variant_to_indices


def sample_indices(
    indices: List[int],
    fraction: float,
    seed: int,
    max_samples: Optional[int] = None,
) -> List[int]:
    """
    Deterministically sample a subset of indices.
    """
    if len(indices) == 0:
        return []

    frac = float(fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"SCAM_FRACTION must be in (0, 1], got {fraction}")

    n = int(math.floor(len(indices) * frac))
    n = max(1, n)

    if max_samples is not None:
        n = min(n, int(max_samples))

    if n >= len(indices):
        return list(indices)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(indices), size=n, replace=False)
    return [indices[int(j)] for j in chosen]


def load_images_from_scam_indices(
    ds: object,
    chosen_indices: List[int],
    preprocess,
    device: str,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Returns:
      images: [N,3,H,W] tensor on device
      ids:    list of SCAM ids for bookkeeping
    """
    if len(chosen_indices) == 0:
        raise RuntimeError("No indices selected for this variant (empty bucket or fraction too small).")

    img_tensors: List[torch.Tensor] = []
    ids: List[str] = []

    for i in chosen_indices:
        entry = ds[int(i)]
        sid = str(entry.get("id", ""))
        img = entry["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensors.append(preprocess(img))
        ids.append(sid)

    images = torch.stack(img_tensors, dim=0).to(device)
    return images, ids

def split_heads(x_seq_batch_embed: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    x: [L, N, E] -> [N, L, H, Hd]
    """
    L, N, E = x_seq_batch_embed.shape
    head_dim = E // num_heads
    x = x_seq_batch_embed.permute(1, 0, 2).contiguous()  # [N,L,E]
    x = x.view(N, L, num_heads, head_dim)
    return x  # [N,L,H,Hd]

def mean_head_vector_norm(x_n_l_h_hd: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> np.ndarray:
    """
    x: [N, L, H, Hd]
    token_mask: [L] boolean mask over token positions (same for all N)
    returns: [H] mean ||.|| over (N, L)
    """
    x = x_n_l_h_hd
    if token_mask is not None:
        x = x[:, token_mask, :, :]
    norms = torch.linalg.norm(x, dim=-1)  # [N,L,H]
    return norms.mean(dim=(0, 1)).detach().cpu().numpy()

def mean_token_l2(x_seq_batch_embed: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> float:
    """
    x: [L,N,E], returns mean ||x|| over tokens+batch (optionally masked over L)
    """
    x = x_seq_batch_embed
    if token_mask is not None:
        x = x[token_mask, :, :]
    norms = torch.linalg.norm(x, dim=-1)  # [L,N]
    return float(norms.mean().item())

# Core: forward visual transformer manually to expose x_in/attn_delta/mlp_delta/x_out
@dataclass
class LayerStats:
    layer: int
    x_in_mean: float
    attn_delta_mean: float
    mlp_delta_mean: float
    x_out_mean: float

    q_head_mean: np.ndarray  # [H]
    k_head_mean: np.ndarray  # [H]
    v_head_mean: np.ndarray  # [H]

    # activation stats for MLP expanded features
    a_abs_mean: np.ndarray   # [4d]
    # for convenience: reg neuron abs means if layer in REG_NEURONS
    reg_abs_mean: Optional[Dict[int, float]]


@torch.no_grad()
def collect_stats_passA(
    visual: torch.nn.Module,
    images: torch.Tensor,
    device: str,
    out_dir: str,
    tag: str,
) -> Tuple[List[LayerStats], int, int]:
    """
    Pass A:
      - Computes per-layer block decomposition norms
      - Computes per-head mean norms of q/k/v from ln_1(x_in)
      - Computes E[|a_f|] for MLP expanded activations a=gelu(c_fc(ln_2(x_mid)))
    Returns:
      stats_per_layer, num_layers, num_heads
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build initial token stream like VisualTransformer.forward (but we keep tokens)
    x = visual.conv1(images)                # [N, width, grid, grid]
    N, width, gh, gw = x.shape
    x = x.reshape(N, width, gh * gw).permute(0, 2, 1)  # [N, HW, width]

    cls = visual.class_embedding.to(x.dtype)
    cls_tok = cls + torch.zeros(N, 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_tok, x], dim=1)      # [N, 1+HW, width]

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # NLD -> LND
    x = x.permute(1, 0, 2).contiguous()     # [L,N,width]
    L, N, d_model = x.shape

    # token masks
    tok_all = torch.ones(L, dtype=torch.bool, device=x.device)
    tok_patch = tok_all.clone()
    tok_patch[0] = False  # exclude CLS

    resblocks = visual.transformer.resblocks
    num_layers = len(resblocks)
    num_heads = resblocks[0].attn.num_heads

    stats: List[LayerStats] = []

    for layer_idx, block in enumerate(resblocks):
        # decomposition
        x_in = x

        ln1 = block.ln_1(x_in)                       # [L,N,d]
        # Compute q,k,v from *online* ln1
        q = block.attn.q_proj(ln1)
        k = block.attn.k_proj(ln1)
        v = block.attn.v_proj(ln1)

        q_h = split_heads(q, num_heads)              # [N,L,H,Hd]
        k_h = split_heads(k, num_heads)
        v_h = split_heads(v, num_heads)

        q_head_mean = mean_head_vector_norm(q_h, token_mask=tok_patch)
        k_head_mean = mean_head_vector_norm(k_h, token_mask=tok_patch)
        v_head_mean = mean_head_vector_norm(v_h, token_mask=tok_patch)

        attn_delta = block.attention(ln1)            # [L,N,d]
        x_mid = x_in + attn_delta

        ln2 = block.ln_2(x_mid)
        c_fc = block.mlp.c_fc if hasattr(block.mlp, "c_fc") else block.mlp[0]
        gelu = block.mlp.gelu if hasattr(block.mlp, "gelu") else block.mlp[1]
        c_proj = block.mlp.c_proj if hasattr(block.mlp, "c_proj") else block.mlp[2]

        z = c_fc(ln2)                                # [L,N,4d]
        a = gelu(z)                                  # [L,N,4d]
        mlp_delta = c_proj(a)                        # [L,N,d]

        x_out = x_mid + mlp_delta
        x = x_out  # advance stream

        # stats
        x_in_mean = mean_token_l2(x_in, token_mask=tok_patch)
        attn_mean = mean_token_l2(attn_delta, token_mask=tok_patch)
        mlp_mean = mean_token_l2(mlp_delta, token_mask=tok_patch)
        x_out_mean = mean_token_l2(x_out, token_mask=tok_patch)

        # E|a_f| over patches only
        a_abs = a[tok_patch, :, :].abs().mean(dim=(0, 1)).detach().cpu().numpy()  # [4d]

        reg_means = None
        if layer_idx in REG_NEURONS:
            reg_means = {int(f): float(a_abs[int(f)]) for f in REG_NEURONS[layer_idx]}

        stats.append(LayerStats(
            layer=layer_idx,
            x_in_mean=x_in_mean,
            attn_delta_mean=attn_mean,
            mlp_delta_mean=mlp_mean,
            x_out_mean=x_out_mean,
            q_head_mean=q_head_mean,
            k_head_mean=k_head_mean,
            v_head_mean=v_head_mean,
            a_abs_mean=a_abs,
            reg_abs_mean=reg_means,
        ))

    # write quick text summary for reg neurons
    reg_path = os.path.join(out_dir, f"{tag}_reg_neuron_a_abs_means.txt")
    with open(reg_path, "w", encoding="utf-8") as f:
        for s in stats:
            if s.reg_abs_mean is None:
                continue
            f.write(f"Layer {s.layer}:\n")
            for feat, val in s.reg_abs_mean.items():
                f.write(f"  a_abs_mean[{feat}] = {val:.6f}\n")
    print(f"[INFO] Wrote reg neuron activation stats: {reg_path}")

    return stats, num_layers, num_heads


# Gain maps: feature (layer l MLP) -> next layer (l+1) key head
def compute_gain_maps(
    visual: torch.nn.Module,
    stats: List[LayerStats],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each layer l in [0..L-2]:
      weight_gain[l] : [H, 4d]
      act_gain[l]    : [H, 4d] = weight_gain * E|a_f|   (from layer l)
    """
    resblocks = visual.transformer.resblocks
    L = len(resblocks)
    H = resblocks[0].attn.num_heads
    d_model = resblocks[0].attn.embed_dim if hasattr(resblocks[0].attn, "embed_dim") else resblocks[0].ln_1.normalized_shape[0]
    head_dim = d_model // H

    weight_gains: List[np.ndarray] = []
    act_gains: List[np.ndarray] = []

    for l in range(L - 1):
        blk = resblocks[l]
        nxt = resblocks[l + 1]

        # W_proj: [d, 4d]  (c_proj weight is [d, 4d])
        c_proj = blk.mlp.c_proj if hasattr(blk.mlp, "c_proj") else blk.mlp[2]
        Wproj = c_proj.weight.detach().float()  # [d, 4d]

        # Wk: [d, d] for next block, slice rows per head -> [H, head_dim, d]
        Wk = nxt.attn.k_proj.weight.detach().float()  # [d, d]
        Wk_h = Wk.view(H, head_dim, d_model)          # [H, Hd, d]

        # Compute M = Wk_h @ Wproj => [H, Hd, 4d]
        M = torch.einsum("hmd,df->hmf", Wk_h, Wproj)  # [H, Hd, 4d]
        G = torch.linalg.norm(M, dim=1)               # [H, 4d]

        G_np = G.detach().cpu().numpy().astype(np.float32)
        weight_gains.append(G_np)

        a_abs_np = stats[l].a_abs_mean  # [4d]
        a_abs = torch.as_tensor(a_abs_np, device=G.device, dtype=G.dtype)  # [4d]
        if a_abs.numel() != G.shape[1]:
            raise ValueError(f"stats[{l}].a_abs_mean has {a_abs.numel()} elems, expected {G.shape[1]} (4d).")
        act = (G * a_abs.view(1, -1)).detach().cpu().numpy().astype(np.float32)
        act_gains.append(act)

    return weight_gains, act_gains


# Finite-diff perturbation on real residual streams
@torch.no_grad()
def finite_diff_delta_k_norms(
    visual: torch.nn.Module,
    images: torch.Tensor,
    layer_l: int,
    feat_indices: List[int],
    alpha: float,
) -> np.ndarray:
    """
    For a given layer l and selected expanded features f:
      - run forward up to block l, obtain x_out and a_f (gelu(c_fc(ln2(x_mid))) for those f)
      - perturb: x_out_pert = x_out + alpha * a_f * Wproj[:,f]
      - compute next block (l+1) ln1 and k_proj, return delta of mean ||k_h|| over patches.

    Returns:
      delta_k_mean: [H] = mean over features of (k_norm_pert - k_norm_base)
    """
    resblocks = visual.transformer.resblocks
    assert 0 <= layer_l < len(resblocks) - 1
    blk = resblocks[layer_l]
    nxt = resblocks[layer_l + 1]

    # Prepare initial tokens (same as passA)
    x = visual.conv1(images)
    N, width, gh, gw = x.shape
    x = x.reshape(N, width, gh * gw).permute(0, 2, 1)  # [N,HW,d]

    cls = visual.class_embedding.to(x.dtype)
    cls_tok = cls + torch.zeros(N, 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_tok, x], dim=1)  # [N,1+HW,d]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2).contiguous()  # [L,N,d]
    Ltok = x.shape[0]

    tok_patch = torch.ones(Ltok, dtype=torch.bool, device=x.device)
    tok_patch[0] = False

    # forward through blocks up to l
    for j in range(layer_l):
        b = resblocks[j]
        x = x + b.attention(b.ln_1(x))
        x = x + b.mlp(b.ln_2(x))

    # Now at input to block l
    x_in = x
    ln1 = blk.ln_1(x_in)
    attn_delta = blk.attention(ln1)
    x_mid = x_in + attn_delta
    ln2 = blk.ln_2(x_mid)

    c_fc = blk.mlp.c_fc if hasattr(blk.mlp, "c_fc") else blk.mlp[0]
    gelu = blk.mlp.gelu if hasattr(blk.mlp, "gelu") else blk.mlp[1]
    c_proj = blk.mlp.c_proj if hasattr(blk.mlp, "c_proj") else blk.mlp[2]

    # Compute a_f only for selected features
    W_fc = c_fc.weight.detach()
    b_fc = c_fc.bias.detach() if c_fc.bias is not None else None
    W_fc_sel = W_fc[feat_indices, :]  # [K,d]
    b_fc_sel = b_fc[feat_indices] if b_fc is not None else None

    z_sel = F.linear(ln2, W_fc_sel, b_fc_sel)  # [L,N,K]
    a_sel = gelu(z_sel)                        # [L,N,K]

    # Baseline x_out
    z_full = c_fc(ln2)
    a_full = gelu(z_full)
    mlp_delta = c_proj(a_full)
    x_out = x_mid + mlp_delta  # [L,N,d]

    # Baseline next-block k norms
    ln1_next_base = nxt.ln_1(x_out)
    k_base = nxt.attn.k_proj(ln1_next_base)              # [L,N,d]
    H = nxt.attn.num_heads
    k_base_h = split_heads(k_base, H)                    # [N,L,H,Hd]
    k_base_mean = mean_head_vector_norm(k_base_h, token_mask=tok_patch)  # [H]

    Wproj = c_proj.weight.detach()  # [d,4d]
    delta_sum = np.zeros((H,), dtype=np.float64)

    for ii, f in enumerate(feat_indices):
        w_col = Wproj[:, int(f)].view(1, 1, -1)  # [1,1,d]
        delta_x = alpha * a_sel[:, :, ii].unsqueeze(-1) * w_col  # [L,N,d]
        x_pert = x_out + delta_x

        ln1_next = nxt.ln_1(x_pert)
        k_pert = nxt.attn.k_proj(ln1_next)
        k_pert_h = split_heads(k_pert, H)
        k_pert_mean = mean_head_vector_norm(k_pert_h, token_mask=tok_patch)

        delta = (k_pert_mean - k_base_mean).astype(np.float64)
        delta_sum += delta

    delta_mean = (delta_sum / max(1, len(feat_indices))).astype(np.float32)
    return delta_mean

# Plotting helpers
def plot_block_decomposition(stats: List[LayerStats], out_dir: str, tag: str):
    layers = [s.layer for s in stats]
    x_in = [s.x_in_mean for s in stats]
    attn = [s.attn_delta_mean for s in stats]
    mlp = [s.mlp_delta_mean for s in stats]
    x_out = [s.x_out_mean for s in stats]

    plt.figure(figsize=(10, 5))
    plt.plot(layers, x_in, marker="o", label="mean ||x_in|| (patch tokens)")
    plt.plot(layers, attn, marker="o", label="mean ||attn_delta||")
    plt.plot(layers, mlp, marker="o", label="mean ||mlp_delta||")
    plt.plot(layers, x_out, marker="o", label="mean ||x_out||")
    plt.xlabel("Layer (block index)")
    plt.ylabel("Mean L2 norm")
    plt.title(f"{tag}: Block decomposition norms (patch tokens only)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{tag}_block_decomposition_norms.png")
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved: {path}")


def _stats_to_block_arrays(stats: List[LayerStats]) -> Dict[str, np.ndarray]:
    """
    Returns per-layer arrays (float32) for the 4 decomposition norms.
    Keys: x_in, attn, mlp, x_out
    """
    x_in = np.array([s.x_in_mean for s in stats], dtype=np.float32)
    attn = np.array([s.attn_delta_mean for s in stats], dtype=np.float32)
    mlp = np.array([s.mlp_delta_mean for s in stats], dtype=np.float32)
    x_out = np.array([s.x_out_mean for s in stats], dtype=np.float32)
    return {"x_in": x_in, "attn": attn, "mlp": mlp, "x_out": x_out}


def plot_block_decomp_delta_canvas(
    delta_block: Dict[str, np.ndarray],
    out_path: str,
    title: str,
):
    """
    2x2 canvas of bar plots (per layer) for:
      Δx_in, Δattn, Δmlp, Δx_out
    """
    keys = [("x_in", "Δ mean ||x_in||"),
            ("attn", "Δ mean ||attn_delta||"),
            ("mlp", "Δ mean ||mlp_delta||"),
            ("x_out", "Δ mean ||x_out||")]

    L = len(delta_block["x_in"])
    layers = np.arange(L)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    axes = np.array(axes).reshape(2, 2)

    for i, (k, ylabel) in enumerate(keys):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        ax.bar(layers, delta_block[k])
        ax.set_title(f"{title} | {ylabel}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δ (vs baseline)")

    plt.savefig(out_path)
    plt.close(fig)


def plot_k_cumsum_delta_bar(
    delta_k: np.ndarray,
    out_path: str,
    title: str,
):
    """
    Bar plot over heads: Δ cumulative sum over layers of mean ||K_h||.
    """
    H = delta_k.shape[0]
    heads = np.arange(H)

    plt.figure(figsize=(14, 5))
    plt.bar(heads, delta_k)
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Δ cumulative Σ_layers mean ||K_h|| (patch tokens)")
    plt.xticks(heads)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_block_delta_csv(
    delta_block: Dict[str, np.ndarray],
    out_csv: str,
):
    """
    CSV: layer, dx_in, datt, dmlp, dx_out
    """
    L = len(delta_block["x_in"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "dx_in", "dattn", "dmlp", "dx_out"])
        for l in range(L):
            w.writerow([
                l,
                float(delta_block["x_in"][l]),
                float(delta_block["attn"][l]),
                float(delta_block["mlp"][l]),
                float(delta_block["x_out"][l]),
            ])


def save_k_delta_csv(
    delta_k: np.ndarray,
    out_csv: str,
):
    """
    CSV: head, delta_k_cumsum
    """
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["head", "delta_k_cumsum"])
        for h, v in enumerate(delta_k.tolist()):
            w.writerow([h, float(v)])


def print_block_delta_table(
    delta_block: Dict[str, np.ndarray],
    model_alias: str,
    baseline_alias: str,
):
    L = len(delta_block["x_in"])
    print(f"\n=== Δ Block decomposition (patch tokens): {model_alias} - {baseline_alias} ===")
    for l in range(L):
        print(
            f"Layer {l:2d}: "
            f"Δ||x_in||={delta_block['x_in'][l]:+7.3f}  "
            f"Δ||attn_delta||={delta_block['attn'][l]:+7.3f}  "
            f"Δ||mlp_delta||={delta_block['mlp'][l]:+7.3f}  "
            f"Δ||x_out||={delta_block['x_out'][l]:+7.3f}"
        )


def print_k_cumsum_delta(
    delta_k: np.ndarray,
    model_alias: str,
    baseline_alias: str,
    topn: int = 8,
):
    print(f"\n=== Δ cumulative Σ_layers mean ||K_h||: {model_alias} - {baseline_alias} ===")
    for h, v in enumerate(delta_k):
        print(f"Head {h:2d}: {v:+.3f}")

    # quick highlight: biggest movers
    idx = np.argsort(-np.abs(delta_k))[:topn]
    msg = ", ".join([f"H{h}:{delta_k[h]:+.3f}" for h in idx])
    print(f"[INFO] Top-{topn} |Δ| heads: {msg}")


def plot_k_norms_per_head(stats: List[LayerStats], out_dir: str, tag: str, which: str = "k"):
    layers = [s.layer for s in stats]
    H = len(stats[0].k_head_mean)

    plt.figure(figsize=(12, 6))
    for h in range(H):
        if which == "k":
            ys = [s.k_head_mean[h] for s in stats]
            title = "Per-head mean ||K_h|| from ln_1(x_in) (patch tokens)"
            fname = f"{tag}_per_head_k_norms.png"
            ylabel = "Mean ||K_h||"
        elif which == "q":
            ys = [s.q_head_mean[h] for s in stats]
            title = "Per-head mean ||Q_h|| from ln_1(x_in) (patch tokens)"
            fname = f"{tag}_per_head_q_norms.png"
            ylabel = "Mean ||Q_h||"
        else:
            ys = [s.v_head_mean[h] for s in stats]
            title = "Per-head mean ||V_h|| from ln_1(x_in) (patch tokens)"
            fname = f"{tag}_per_head_v_norms.png"
            ylabel = "Mean ||V_h||"

        plt.plot(layers, ys, label=f"H{h}")

    plt.xlabel("Layer (block index)")
    plt.ylabel(ylabel)
    plt.title(f"{tag}: {title}")
    plt.legend(ncol=4, fontsize=8, loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_gain_heatmap_top_features(
    gain: np.ndarray,
    out_path: str,
    top_m: int = 96,
    title: str = "",
):
    """
    gain: [H, 4d]

    Behavior:
    - If top_m < Fexp: pick top_m features by max over heads (as before), sorted by gain.
    - If top_m >= Fexp (i.e. "all features"): keep numerical order [0..Fexp-1] and
      render a single image containing a 2x2 grid of heatmaps, each showing 1024 features:
        [0:1024]   [1024:2048]
        [2048:3072] [3072:4096]
    """
    H, Fexp = gain.shape

    # "all features" special-case -> 2x2 canvas in numeric order
    if top_m >= Fexp:
        chunk = 1024
        n_chunks = int(math.ceil(Fexp / chunk))
        if n_chunks != 4:
            # support non-4096 cases gracefully
            n_rows = int(math.ceil(n_chunks / 2))
            n_cols = 2
        else:
            n_rows, n_cols = 2, 2

        vmin = float(np.nanmin(gain))
        vmax = float(np.nanmax(gain))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), constrained_layout=True)
        axes = np.array(axes).reshape(n_rows, n_cols)

        last_im = None
        for ci in range(n_chunks):
            r = ci // n_cols
            c = ci % n_cols
            ax = axes[r, c]

            start = ci * chunk
            end = min((ci + 1) * chunk, Fexp)
            sub = gain[:, start:end]  # numeric order

            last_im = ax.imshow(sub, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} | features [{start}:{end}]")
            ax.set_xlabel("Feature index")
            ax.set_ylabel("Head")

        # Hide any unused axes (if Fexp not exactly 4096)
        for ci in range(n_chunks, n_rows * n_cols):
            r = ci // n_cols
            c = ci % n_cols
            axes[r, c].axis("off")

        # One shared colorbar for the whole canvas
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9, label="gain")

        plt.savefig(out_path)
        plt.close(fig)
        return

    # Default behavior: top-m by max-head gain (keeps it readable)
    scores = gain.max(axis=0)
    idx = np.argsort(-scores)[:top_m]
    sub = gain[:, idx]  # [H, top_m]

    plt.figure(figsize=(12, 5))
    plt.imshow(sub, aspect="auto")
    plt.colorbar(label="gain")
    plt.xlabel("Feature (top subset, sorted by max-head gain)")
    plt.ylabel("Head")
    plt.title(title + f" (showing top {top_m}/{Fexp} features)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_layers(s: str, max_layer: int) -> List[int]:
    if s.strip().lower() == "all":
        return list(range(max_layer - 1))  # only layers with next-block exist
    items = [t.strip() for t in s.split(",") if t.strip()]
    out: List[int] = []
    for it in items:
        if "-" in it:
            a, b = it.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            if b < a:
                a, b = b, a
            out.extend(list(range(a, min(b, max_layer - 2) + 1)))
        else:
            out.append(int(it))
    out = sorted(list(set([x for x in out if 0 <= x < max_layer - 1])))
    return out


# Runner: one model, variant
def run_one(
    model_alias: str,
    model_id: str,
    variant: str,
    ds: object,
    chosen_indices: List[int],
):
    fix_random_seed(SEED)

    tag = f"{model_alias}__{variant}"
    out_dir = os.path.join(OUT_ROOT, model_alias, variant)
    os.makedirs(out_dir, exist_ok=True)

    device = DEVICE

    model, preprocess, _ = load_openai_clip_anything(
        clip,
        model_id,
        device=device,
        jit=False,
        strict=True,
    )
    model = model.float().eval()
    visual = model.visual

    if ABLATE_REG_NEURONS:
        attach_reg_neuron_nuke_hooks(visual)

    images, ids = load_images_from_scam_indices(
        ds=ds,
        chosen_indices=chosen_indices,
        preprocess=preprocess,
        device=device,
    )
    print(f"\n==================== {tag} ====================")
    print(f"[INFO] Using {len(ids)} images from {SCAM_DATASET_ID} ({variant})")
    print(f"[INFO] Output dir: {os.path.abspath(out_dir)}")

    # PASS A
    stats, num_layers, num_heads = collect_stats_passA(
        visual=visual,
        images=images,
        device=device,
        out_dir=out_dir,
        tag=tag,
    )

    # Print compact layer table
    print("\n=== Block decomposition norms (patch tokens) ===")
    for s in stats:
        print(
            f"Layer {s.layer:2d}: "
            f"||x_in||={s.x_in_mean:7.3f}  "
            f"||attn_delta||={s.attn_delta_mean:7.3f}  "
            f"||mlp_delta||={s.mlp_delta_mean:7.3f}  "
            f"||x_out||={s.x_out_mean:7.3f}"
        )

    # Head summaries
    k_cumsum = np.stack([s.k_head_mean for s in stats], axis=0).sum(axis=0)  # [H]
    print("\n=== Cumulative sum over layers: mean ||K_h|| (patch tokens) ===")
    for h in range(num_heads):
        print(f"Head {h:2d}: {k_cumsum[h]:.3f}")

    # Plots
    plot_block_decomposition(stats, out_dir, tag=tag)
    plot_k_norms_per_head(stats, out_dir, tag=tag, which="q")
    plot_k_norms_per_head(stats, out_dir, tag=tag, which="k")
    plot_k_norms_per_head(stats, out_dir, tag=tag, which="v")

    # Gain maps
    weight_gains, act_gains = compute_gain_maps(visual, stats)

    gain_npz = os.path.join(out_dir, f"{tag}_gain_maps.npz")
    np.savez(
        gain_npz,
        weight_gains=np.stack(weight_gains, axis=0),  # [L-1,H,4d]
        act_gains=np.stack(act_gains, axis=0),        # [L-1,H,4d]
    )
    print(f"\n[INFO] Saved gain maps to: {gain_npz}")

    gain_dir = os.path.join(out_dir, f"{tag}_gain_heatmaps")
    os.makedirs(gain_dir, exist_ok=True)

    for l in range(num_layers - 1):
        out1 = os.path.join(gain_dir, f"{tag}_layer{l:02d}_weight_gain_top.png")
        out2 = os.path.join(gain_dir, f"{tag}_layer{l:02d}_act_gain_top.png")
        plot_gain_heatmap_top_features(
            weight_gains[l],
            out1,
            top_m=TOPM_GAIN_PLOT,
            title=f"{tag}: Layer {l} -> {l+1}: weight-only gain (K heads × MLP features)",
        )
        plot_gain_heatmap_top_features(
            act_gains[l],
            out2,
            top_m=TOPM_GAIN_PLOT,
            title=f"{tag}: Layer {l} -> {l+1}: activation-weighted gain (gain × E|a_f|)",
        )

    print(f"[INFO] Saved per-layer gain heatmaps to: {gain_dir}")

    # Finite diff
    finite_layers = parse_layers(FINITE_DIFF_LAYERS, max_layer=num_layers)
    print(f"\n[INFO] Finite diff layers: {finite_layers} (alpha={ALPHA}, topK={TOPK_FINITE_DIFF})")

    fd_delta = np.zeros((num_layers - 1, num_heads), dtype=np.float32) * np.nan

    for l in finite_layers:
        agg = act_gains[l].sum(axis=0)  # [4d]
        top_idx = np.argsort(-agg)[:TOPK_FINITE_DIFF].astype(int).tolist()

        if l in REG_NEURONS:
            for f in REG_NEURONS[l]:
                if int(f) not in top_idx:
                    top_idx.append(int(f))

        # keep list bounded, preserve forced regs
        if len(top_idx) > max(TOPK_FINITE_DIFF, len(REG_NEURONS.get(l, []))):
            top_idx = top_idx[:max(TOPK_FINITE_DIFF, len(REG_NEURONS.get(l, [])))]

        dkh = finite_diff_delta_k_norms(
            visual=visual,
            images=images,
            layer_l=l,
            feat_indices=top_idx,
            alpha=float(ALPHA),
        )
        fd_delta[l, :] = dkh

        top_heads = np.argsort(-np.abs(dkh))[:5]
        msg = ", ".join([f"H{h}:{dkh[h]:+.4f}" for h in top_heads])
        print(f"Layer {l:2d} -> {l+1:2d} finite-diff Δmean||K_h|| (avg over {len(top_idx)} feats): {msg}")

    plt.figure(figsize=(14, 6))
    plt.imshow(fd_delta, aspect="auto")
    plt.colorbar(label="Δ mean ||K_h|| (patch tokens), averaged over selected features")
    plt.xlabel("Head")
    plt.ylabel("Layer (block index l, measuring effect on layer l+1 keys)")
    plt.title(f"{tag}: Finite-diff sensitivity heatmap")
    plt.xticks(np.arange(num_heads))
    plt.yticks(np.arange(num_layers - 1))
    plt.tight_layout()
    fd_path = os.path.join(out_dir, f"{tag}_finite_diff_delta_k_norms_heatmap.png")
    plt.savefig(fd_path)
    plt.close()
    print(f"[INFO] Saved: {fd_path}")

    np.save(os.path.join(out_dir, f"{tag}_finite_diff_delta_k_norms.npy"), fd_delta)

    # Save chosen ids for reproducibility/debug
    ids_path = os.path.join(out_dir, f"{tag}_scam_ids.txt")
    with open(ids_path, "w", encoding="utf-8") as f:
        for sid in ids:
            f.write(sid + "\n")
    print(f"[INFO] Wrote sample IDs: {ids_path}")

    print(f"[OK] Done: {tag}")

    return {
        "tag": tag,
        "out_dir": out_dir,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "block": _stats_to_block_arrays(stats),  # dict of 4 arrays [L]
        "k_cumsum": k_cumsum,                    # [H]
    }


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    fix_random_seed(SEED)

    # Build dataset index once, sample once per variant (shared across models)
    ds, variant_to_indices = prepare_scam_indices(
        dataset_id=SCAM_DATASET_ID,
        split=SCAM_SPLIT,
        variants=SCAM_VARIANTS,
    )

    chosen_by_variant: Dict[str, List[int]] = {}
    for v in SCAM_VARIANTS:
        v_seed = (SEED * 1_000_000 + (abs(hash(v)) % 1_000_000)) & 0xFFFFFFFF
        chosen = sample_indices(
            variant_to_indices[v],
            fraction=SCAM_FRACTION,
            seed=v_seed,
            max_samples=MAX_SAMPLES_PER_VARIANT,
        )
        chosen_by_variant[v] = chosen
        print(f"[INFO] Chosen {len(chosen)} / {len(variant_to_indices[v])} for {v} (fraction={SCAM_FRACTION})")


    # collect summaries for cross-model comparisons
    # results[variant][model_alias] = dict returned by run_one(...)
    results: Dict[str, Dict[str, dict]] = {v: {} for v in SCAM_VARIANTS}

    for model_alias, model_id in MODELS:
        for variant in SCAM_VARIANTS:
            out = run_one(
                model_alias=model_alias,
                model_id=model_id,
                variant=variant,
                ds=ds,
                chosen_indices=chosen_by_variant[variant],
            )
            results[variant][model_alias] = out

    # compare models vs baseline (first in MODELS) per variant
    if len(MODELS) > 1:
        baseline_alias = MODELS[0][0]
        compare_root = os.path.join(OUT_ROOT, "_compare_vs_baseline")
        os.makedirs(compare_root, exist_ok=True)

        for variant in SCAM_VARIANTS:
            if baseline_alias not in results[variant]:
                print(f"[WARN] Baseline alias '{baseline_alias}' missing for variant {variant}; skipping compare.")
                continue

            base = results[variant][baseline_alias]
            base_block = base["block"]
            base_k = base["k_cumsum"]

            variant_dir = os.path.join(compare_root, variant)
            os.makedirs(variant_dir, exist_ok=True)

            # One summary text file per variant
            summary_txt = os.path.join(variant_dir, f"compare_vs_{baseline_alias}__{variant}.txt")
            with open(summary_txt, "w", encoding="utf-8") as f:
                f.write(f"Baseline: {baseline_alias}\nVariant: {variant}\n\n")

                for model_alias, _model_id in MODELS[1:]:
                    if model_alias not in results[variant]:
                        continue
                    cur = results[variant][model_alias]
                    cur_block = cur["block"]
                    cur_k = cur["k_cumsum"]

                    delta_block = {
                        "x_in":  cur_block["x_in"]  - base_block["x_in"],
                        "attn":  cur_block["attn"]  - base_block["attn"],
                        "mlp":   cur_block["mlp"]   - base_block["mlp"],
                        "x_out": cur_block["x_out"] - base_block["x_out"],
                    }
                    delta_k = (cur_k - base_k).astype(np.float32)

                    print_block_delta_table(delta_block, model_alias, baseline_alias)
                    print_k_cumsum_delta(delta_k, model_alias, baseline_alias)

                    # text file
                    f.write(f"=== {model_alias} - {baseline_alias} ===\n")
                    for l in range(len(delta_block["x_in"])):
                        f.write(
                            f"Layer {l:2d}: "
                            f"Δ||x_in||={delta_block['x_in'][l]:+7.3f}  "
                            f"Δ||attn_delta||={delta_block['attn'][l]:+7.3f}  "
                            f"Δ||mlp_delta||={delta_block['mlp'][l]:+7.3f}  "
                            f"Δ||x_out||={delta_block['x_out'][l]:+7.3f}\n"
                        )
                    f.write("\nΔ cumulative Σ_layers mean ||K_h||:\n")
                    for h in range(len(delta_k)):
                        f.write(f"Head {h:2d}: {delta_k[h]:+.3f}\n")
                    f.write("\n")

                    # CSVs
                    block_csv = os.path.join(variant_dir, f"{model_alias}__{variant}__delta_block_vs_{baseline_alias}.csv")
                    k_csv = os.path.join(variant_dir, f"{model_alias}__{variant}__delta_kcumsum_vs_{baseline_alias}.csv")
                    save_block_delta_csv(delta_block, block_csv)
                    save_k_delta_csv(delta_k, k_csv)

                    # plots
                    block_plot = os.path.join(variant_dir, f"{model_alias}__{variant}__delta_block_vs_{baseline_alias}.png")
                    k_plot = os.path.join(variant_dir, f"{model_alias}__{variant}__delta_kcumsum_vs_{baseline_alias}.png")

                    plot_block_decomp_delta_canvas(
                        delta_block=delta_block,
                        out_path=block_plot,
                        title=f"{variant}: {model_alias} - {baseline_alias}",
                    )
                    plot_k_cumsum_delta_bar(
                        delta_k=delta_k,
                        out_path=k_plot,
                        title=f"{variant}: Δ cumulative Σ_layers mean ||K_h|| | {model_alias} - {baseline_alias}",
                    )

            print(f"[INFO] Wrote compare summary: {summary_txt}")

    print(f"\n[OK] All done. Outputs root: {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()