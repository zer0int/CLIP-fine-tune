"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

 A very **quick** Second Order Lens with Jaccard

 Download Flickr8k: Many places, e.g.:
 https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip

 Implements a practical "second-order lens" style analysis for CLIP ViT MLP neurons
 (blocks LAYER_START..LAYER_END) and adds Jaccard overlap tracking of the TOP-K
 neuron sets across checkpoints (vs pre and consecutive).

 Notes:
   - Second-order effect magnitude proxy per neuron: |s_{l,n}(img)| * ||r_{l,n}||
     where s_{l,n} aggregates post-GELU activations weighted by downstream CLS->token attention mass,
     and r_{l,n} is a direction in the CLIP embedding space computed via downstream (W_O W_V) and proj.
   - Jaccard is computed over TOPK_NEURONS_JACCARD selected by sum over dataset of |s_{l,n}|.
"""

import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

import attnclipdecouple as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything


SEED: int = 123
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# USER CONFIG
# ============================================================

# Note: This is primarily for measuring the trajectory during fine-tuning.
MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("ckpt00", "/path/to/your/clip_ft_e0_raw_full_as-weight.pt"),
    ("ckpt01", "/path/to/your/clip_ft_e1_raw_full_as-weight.pt"),
    ("ckpt02", "/path/to/your/clip_ft_e2_raw_full_as-weight.pt"),
    ("ckpt03", "/path/to/your/clip_ft_e3_raw_full_as-weight.pt"),
    ("ckpt05", "/path/to/your/clip_ft_e5_raw_full_as-weight.pt"),
    ("ckpt10", "/path/to/your/clip_ft_e10_raw_full_as-weight.pt"),
    ("ckpt19", "/path/to/your/clip_ft_e19_raw_full_as-weight.pt"),
]

OUT_DIR: str = "out_eval_measure/second_order_reg"
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_ROOT: str = "path/to/Flickr8k"  # contains Images/
SUBSET_N: int = 100
BATCH_SIZE: int = 10
NUM_WORKERS: int = 2

LAYER_START: int = 0
LAYER_END: int = 23

# Compute second-order directions/effects for TOPK_NEURONS_EFFECT per layer
TOPK_NEURONS_EFFECT: int = 4096

# Jaccard overlap is computed on TOPK_NEURONS_JACCARD neurons per layer
TOPK_NEURONS_JACCARD: int = 256

# Source tokens: whether CLS is allowed as a "source token" for w_total
INCLUDE_CLS_AS_SOURCE_TOKEN: bool = False

# Implicit registers = high-norm patch tokens (computed from final tokens)
REGISTER_THRESHOLD: float = 70.0
MAX_REGISTERS: Optional[int] = 8
MIN_REGISTERS: int = 1

# Effect sign vs magnitude
USE_ABS_EFFECT: bool = True


def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_epoch(alias: str) -> int:
    m = re.match(r"ft-(\d+)", alias)
    if m:
        return int(m.group(1))
    return -1


def jaccard(a: Set[int], b: Set[int]) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union)


def save_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def plot_heatmap_layers_x_epochs(
    rows: List[Dict[str, Any]],
    metric: str,
    out_png: str,
    layer_start: int,
    layer_end: int,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    epochs = sorted({int(r["epoch"]) for r in rows})
    layers = list(range(layer_start, layer_end + 1))

    mat = np.full((len(layers), len(epochs)), np.nan, dtype=np.float32)
    e2j = {e: j for j, e in enumerate(epochs)}
    l2i = {l: i for i, l in enumerate(layers)}

    for r in rows:
        l = int(r["layer"])
        e = int(r["epoch"])
        v = float(r[metric])
        mat[l2i[l], e2j[e]] = v

    plt.figure()
    plt.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks(range(len(epochs)), [str(e) for e in epochs], rotation=45)
    plt.yticks(range(len(layers)), [str(l) for l in layers])
    plt.xlabel("epoch")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_layer_trajectory(
    rows: List[Dict[str, Any]],
    metric: str,
    out_png: str,
    layer: int,
    title: str,
) -> None:
    rows_l = [r for r in rows if int(r["layer"]) == int(layer)]
    rows_l = sorted(rows_l, key=lambda r: int(r["epoch"]))

    xs = [int(r["epoch"]) for r in rows_l]
    ys = [float(r[metric]) for r in rows_l]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


class FlickrImages(Dataset):
    def __init__(self, images_root: str, preprocess, subset_n: int, seed: int):
        self.images_root = Path(images_root)
        self.preprocess = preprocess

        all_imgs = sorted([p for p in self.images_root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if len(all_imgs) == 0:
            raise FileNotFoundError(f"No images found under: {self.images_root}")

        rng = random.Random(seed)
        rng.shuffle(all_imgs)
        self.paths = all_imgs[:subset_n]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.preprocess(img)
        return x, str(p)


def make_implicit_register_mask(
    patch_token_norms: torch.Tensor,   # [B, n_patches]
    register_threshold: float,
    max_registers: Optional[int],
    min_registers: int,
) -> torch.Tensor:
    B, P = patch_token_norms.shape
    mask = patch_token_norms > register_threshold

    if max_registers is not None:
        out = torch.zeros_like(mask)
        for b in range(B):
            idx = torch.nonzero(mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                topk = torch.topk(patch_token_norms[b], k=min_registers, largest=True).indices
                out[b, topk] = True
            else:
                k = min(max_registers, idx.numel())
                topk = idx[torch.topk(patch_token_norms[b, idx], k=k, largest=True).indices]
                out[b, topk] = True
        return out

    out = mask.clone()
    for b in range(B):
        if out[b].sum().item() < min_registers:
            topk = torch.topk(patch_token_norms[b], k=min_registers, largest=True).indices
            out[b, topk] = True
    return out


def safe_clip_load(model_path: str, device: str):
    model, preprocess, _ = load_openai_clip_anything(clip, model_path, device=device, jit=False, strict=True)
    model.eval()
    model.float()
    return model, preprocess


def install_gelu_hooks(model, layer_start: int, layer_end: int) -> Tuple[Dict[int, torch.Tensor], List[Any]]:
    """
    Captures post-GELU activations for each block's MLP during forward:
      gelu_out[layer] = [T, B, 4C]
    """
    gelu_out: Dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx in range(layer_start, layer_end + 1):
        blk = model.visual.transformer.resblocks[layer_idx]
        gelu_mod = blk.mlp.gelu

        def _make_hook(li: int):
            def _hook(module, inputs, output):
                gelu_out[li] = output.detach()
            return _hook

        hooks.append(gelu_mod.register_forward_hook(_make_hook(layer_idx)))

    return gelu_out, hooks


def cleanup_capture_caches(model, capture_layers: Set[int], gelu_out: Dict[int, torch.Tensor]) -> None:
    """
    Prevent slow VRAM creep by nulling module-held caches after each batch.
    """
    for li in capture_layers:
        blk = model.visual.transformer.resblocks[li]
        blk.attn.last_probs = None
        blk.attn.last_logits = None
        blk.attn.last_q = None
        blk.attn.last_k = None
        blk.attn.last_v = None
    gelu_out.clear()


@torch.no_grad()
def compute_second_order_stats_for_batch(
    model,
    gelu_out: Dict[int, torch.Tensor],
    tokens_pre_ln_post: torch.Tensor,              # [B, T, C]
    attn_probs_cache: Dict[int, torch.Tensor],     # layer -> [B,H,T,S]
    layer_start: int,
    layer_end: int,
    topk_neurons_effect: int,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, torch.Tensor]]:
    """
    Returns:
      stats_by_layer: layer -> dict of scalars
      score_abs_sum_cpu_by_layer: layer -> CPU float64 [4C], sum over batch of |s_total|
    """
    B, T, C = tokens_pre_ln_post.shape

    # implicit registers from FINAL tokens (token norms)
    patch_norms = tokens_pre_ln_post[:, 1:, :].norm(dim=-1)  # [B, P]
    reg_mask_patch = make_implicit_register_mask(
        patch_token_norms=patch_norms,
        register_threshold=REGISTER_THRESHOLD,
        max_registers=MAX_REGISTERS,
        min_registers=MIN_REGISTERS,
    )  # [B, P] bool

    reg_mask_token = torch.zeros((B, T), dtype=torch.bool, device=tokens_pre_ln_post.device)
    reg_mask_token[:, 1:] = reg_mask_patch
    if not INCLUDE_CLS_AS_SOURCE_TOKEN:
        reg_mask_token[:, 0] = False

    stats_by_layer: Dict[int, Dict[str, float]] = {}
    score_abs_sum_cpu_by_layer: Dict[int, torch.Tensor] = {}

    for l in range(layer_start, layer_end + 1):
        # no downstream attentions => defined as zero
        if l >= layer_end:
            stats_by_layer[l] = {
                "effect_total_mean": 0.0,
                "effect_reg_mean": 0.0,
                "effect_patch_mean": 0.0,
                "reg_share_effect": 0.0,
                "reg_share_attn_mass": 0.0,
            }
            # still return a score vector for jaccard (all zeros)
            blk_l = model.visual.transformer.resblocks[l]
            n_neurons = blk_l.mlp.c_fc.weight.shape[0]  # 4C
            score_abs_sum_cpu_by_layer[l] = torch.zeros((n_neurons,), dtype=torch.float64)
            continue

        # w_total: [B, T] = sum_{m>l} sum_{h} attn_probs[m][b,h,CLS,:]
        w_total = torch.zeros((B, T), device=tokens_pre_ln_post.device, dtype=tokens_pre_ln_post.dtype)
        for m in range(l + 1, layer_end + 1):
            probs = attn_probs_cache[m]  # [B,H,T,S]
            w_total = w_total + probs[:, :, 0, :].sum(dim=1)

        if not INCLUDE_CLS_AS_SOURCE_TOKEN:
            w_total[:, 0] = 0.0

        w_reg = w_total * reg_mask_token.float()
        w_patch = w_total * (~reg_mask_token).float()

        attn_mass_total = w_total.sum(dim=1) + 1e-12
        attn_mass_reg = w_reg.sum(dim=1)
        reg_share_attn_mass = (attn_mass_reg / attn_mass_total).mean().item()

        # post-GELU activations for layer l: [T,B,4C] -> [B,T,4C]
        if l not in gelu_out:
            raise RuntimeError(f"Missing GELU activations for layer {l}. Hook did not fire?")

        A = gelu_out[l].permute(1, 0, 2).contiguous()  # [B,T,N]
        # scalar coefficients s_*: [B, N]
        s_total = torch.einsum("bt,btn->bn", w_total, A)
        s_reg = torch.einsum("bt,btn->bn", w_reg, A)
        s_patch = torch.einsum("bt,btn->bn", w_patch, A)

        # score vector for Jaccard ranking: sum over batch of |s_total|
        score_abs_sum_cpu_by_layer[l] = s_total.abs().sum(dim=0).double().cpu()

        # select neurons for effect computation
        mean_abs = s_total.abs().mean(dim=0)  # [N]
        k = min(topk_neurons_effect, mean_abs.numel())
        top_idx = torch.topk(mean_abs, k=k, largest=True).indices  # [K]

        # direction computation for top-K only:
        # W_out^{l,n}: c_proj.weight[:, n]
        blk_l = model.visual.transformer.resblocks[l]
        W_out = blk_l.mlp.c_proj.weight[:, top_idx]  # [C, K]

        # downstream mapping: sum_{m>l} (W_O^m W_V^m) applied to W_out
        U2 = torch.zeros((C, k), device=tokens_pre_ln_post.device, dtype=tokens_pre_ln_post.dtype)
        for m in range(l + 1, layer_end + 1):
            blk_m = model.visual.transformer.resblocks[m]
            Vw = blk_m.attn.v_proj.weight
            Ow = blk_m.attn.out_proj.weight
            U2 = U2 + (Ow @ (Vw @ W_out))  # [C,K]

        # project to CLIP embedding space: [K, D]
        P = model.visual.proj  # [C, D]
        r = (U2.T @ P)         # [K, D]
        r_norm = r.norm(dim=1) + 1e-12  # [K]

        s_total_k = s_total[:, top_idx]
        s_reg_k = s_reg[:, top_idx]
        s_patch_k = s_patch[:, top_idx]

        if USE_ABS_EFFECT:
            eff_total = s_total_k.abs() * r_norm.unsqueeze(0)
            eff_reg = s_reg_k.abs() * r_norm.unsqueeze(0)
            eff_patch = s_patch_k.abs() * r_norm.unsqueeze(0)
        else:
            eff_total = s_total_k * r_norm.unsqueeze(0)
            eff_reg = s_reg_k * r_norm.unsqueeze(0)
            eff_patch = s_patch_k * r_norm.unsqueeze(0)

        per_img_total = eff_total.sum(dim=1)
        per_img_reg = eff_reg.sum(dim=1)
        per_img_patch = eff_patch.sum(dim=1)

        effect_total_mean = per_img_total.mean().item()
        effect_reg_mean = per_img_reg.mean().item()
        effect_patch_mean = per_img_patch.mean().item()

        reg_share_effect = (per_img_reg / (per_img_reg + per_img_patch + 1e-12)).mean().item()

        stats_by_layer[l] = {
            "effect_total_mean": effect_total_mean,
            "effect_reg_mean": effect_reg_mean,
            "effect_patch_mean": effect_patch_mean,
            "reg_share_effect": reg_share_effect,
            "reg_share_attn_mass": reg_share_attn_mass,
        }

    return stats_by_layer, score_abs_sum_cpu_by_layer


def main() -> None:
    fix_random_seed(SEED)

    # Load preprocess from first model and reuse
    print(f"[INFO] Loading preprocess from first model: {MODELS[0][0]}")
    tmp_model, preprocess = safe_clip_load(MODELS[0][1], DEVICE)
    del tmp_model
    torch.cuda.empty_cache()

    images_root = os.path.join(DATASET_ROOT, "Images")
    ds = FlickrImages(images_root=images_root, preprocess=preprocess, subset_n=SUBSET_N, seed=SEED)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    capture_layers = set(range(LAYER_START, LAYER_END + 1))

    # Main summaries
    summary_rows: List[Dict[str, Any]] = []

    # For Jaccard: alias -> layer -> set(neuron_ids)
    top_neuron_sets: Dict[str, Dict[int, Set[int]]] = {}

    # ------------------------------------------------------------------
    # Run models
    # ------------------------------------------------------------------
    for alias, model_path in MODELS:
        epoch = parse_epoch(alias)
        print(f"\n[MODEL] {alias} (epoch={epoch})")

        model, _ = safe_clip_load(model_path, DEVICE)
        gelu_out, hooks = install_gelu_hooks(model, LAYER_START, LAYER_END)

        # accumulators for scalar stats
        acc: Dict[int, Dict[str, float]] = {
            l: {
                "effect_total_mean": 0.0,
                "effect_reg_mean": 0.0,
                "effect_patch_mean": 0.0,
                "reg_share_effect": 0.0,
                "reg_share_attn_mass": 0.0,
                "n_batches": 0.0,
            }
            for l in range(LAYER_START, LAYER_END + 1)
        }

        # accumulator for neuron ranking scores (CPU float64)
        # n_neurons = 4C, where C = model.visual.transformer.width
        n_neurons = model.visual.transformer.width * 4
        score_abs_sum_total: Dict[int, torch.Tensor] = {
            l: torch.zeros((n_neurons,), dtype=torch.float64)
            for l in range(LAYER_START, LAYER_END + 1)
        }

        for xb, _paths in dl:
            xb = xb.to(DEVICE, non_blocking=True)

            # Forward: require tokens + captured attention probs
            out = model.visual(
                xb.type(model.dtype),
                return_trace=False,
                return_tokens=True,
                capture_layers=capture_layers,
            )

            if not (isinstance(out, dict) and "tokens_pre_ln_post_full" in out):
                raise RuntimeError(
                    "visual(... return_tokens=True) did not return tokens. "
                    "Your model.py patch may not be active."
                )

            tokens = out["tokens_pre_ln_post_full"]  # [B,T,C]

            # attn probs cache for layers (must be populated by capture)
            attn_probs_cache: Dict[int, torch.Tensor] = {}
            for li in range(LAYER_START, LAYER_END + 1):
                blk = model.visual.transformer.resblocks[li]
                probs = blk.attn.last_probs
                if probs is None:
                    raise RuntimeError(
                        f"Missing attn probs for layer {li}. "
                        f"Ensure capture_layers includes it and capture=True is wired."
                    )
                attn_probs_cache[li] = probs  # [B,H,T,S]

            batch_stats, batch_score_abs_sum_cpu = compute_second_order_stats_for_batch(
                model=model,
                gelu_out=gelu_out,
                tokens_pre_ln_post=tokens,
                attn_probs_cache=attn_probs_cache,
                layer_start=LAYER_START,
                layer_end=LAYER_END,
                topk_neurons_effect=TOPK_NEURONS_EFFECT,
            )

            # accumulate scalar stats
            for l, d in batch_stats.items():
                acc[l]["effect_total_mean"] += d["effect_total_mean"]
                acc[l]["effect_reg_mean"] += d["effect_reg_mean"]
                acc[l]["effect_patch_mean"] += d["effect_patch_mean"]
                acc[l]["reg_share_effect"] += d["reg_share_effect"]
                acc[l]["reg_share_attn_mass"] += d["reg_share_attn_mass"]
                acc[l]["n_batches"] += 1.0

            # accumulate neuron ranking score (CPU)
            for l, v_cpu in batch_score_abs_sum_cpu.items():
                score_abs_sum_total[l] += v_cpu

            # IMPORTANT: cleanup module-held caches each batch (prevents VRAM creep)
            cleanup_capture_caches(model, capture_layers, gelu_out)

        # remove hooks
        for h in hooks:
            h.remove()

        # finalize scalar stats into rows
        for l in range(LAYER_START, LAYER_END + 1):
            nb = max(acc[l]["n_batches"], 1.0)
            row = {
                "alias": alias,
                "epoch": epoch,
                "layer": l,
                "effect_total_mean": acc[l]["effect_total_mean"] / nb,
                "effect_reg_mean": acc[l]["effect_reg_mean"] / nb,
                "effect_patch_mean": acc[l]["effect_patch_mean"] / nb,
                "reg_share_effect": acc[l]["reg_share_effect"] / nb,
                "reg_share_attn_mass": acc[l]["reg_share_attn_mass"] / nb,
                "subset_n": SUBSET_N,
                "topk_neurons_effect": TOPK_NEURONS_EFFECT,
                "register_threshold": REGISTER_THRESHOLD,
            }
            summary_rows.append(row)

            print(
                f"  [L{l:02d}] total={row['effect_total_mean']:.4g} "
                f"reg_share_eff={row['reg_share_effect']:.3f} "
                f"reg_share_attn={row['reg_share_attn_mass']:.3f}"
            )

        # finalize top-K neuron sets for Jaccard (per layer)
        top_neuron_sets[alias] = {}
        for l in range(LAYER_START, LAYER_END + 1):
            scores = score_abs_sum_total[l]  # CPU float64 [4C]
            kJ = min(TOPK_NEURONS_JACCARD, scores.numel())
            top_idx = torch.topk(scores, k=kJ, largest=True).indices.tolist()
            top_neuron_sets[alias][l] = set(top_idx)

        # free model
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save scalar summary + plots
    # ------------------------------------------------------------------
    summary_csv = os.path.join(OUT_DIR, "second_order_summary.csv")
    save_csv(summary_rows, summary_csv)
    print(f"\n[WRITE] {summary_csv}")

    plot_heatmap_layers_x_epochs(
        summary_rows, "reg_share_effect",
        os.path.join(OUT_DIR, "heatmap_reg_share_effect.png"),
        LAYER_START, LAYER_END,
        title="reg_share_effect (second-order magnitude share via regs)",
        vmin=0.0, vmax=1.0
    )
    plot_heatmap_layers_x_epochs(
        summary_rows, "reg_share_attn_mass",
        os.path.join(OUT_DIR, "heatmap_reg_share_attn_mass.png"),
        LAYER_START, LAYER_END,
        title="reg_share_attn_mass (CLS->token attention mass via regs)",
        vmin=0.0, vmax=1.0
    )
    plot_heatmap_layers_x_epochs(
        summary_rows, "effect_total_mean",
        os.path.join(OUT_DIR, "heatmap_effect_total_mean.png"),
        LAYER_START, LAYER_END,
        title="effect_total_mean (sum over top-K neurons of |s|*||r||)",
        vmin=None, vmax=None
    )

    for l in range(LAYER_START, LAYER_END + 1):
        plot_layer_trajectory(
            summary_rows, "reg_share_effect",
            os.path.join(OUT_DIR, f"traj_reg_share_effect_L{l:02d}.png"),
            layer=l,
            title=f"Layer {l}: reg_share_effect"
        )
        plot_layer_trajectory(
            summary_rows, "effect_total_mean",
            os.path.join(OUT_DIR, f"traj_effect_total_mean_L{l:02d}.png"),
            layer=l,
            title=f"Layer {l}: effect_total_mean"
        )

    # ------------------------------------------------------------------
    # Jaccard computations
    # ------------------------------------------------------------------
    # choose baseline alias "pre" if present else first model
    pre_alias = "pre" if any(a == "pre" for a, _ in MODELS) else MODELS[0][0]

    # Jaccard vs pre
    jacc_vs_pre_rows: List[Dict[str, Any]] = []
    for alias, _ in MODELS:
        if alias == pre_alias:
            continue
        epoch = parse_epoch(alias)
        for l in range(LAYER_START, LAYER_END + 1):
            j = jaccard(top_neuron_sets[pre_alias][l], top_neuron_sets[alias][l])
            jacc_vs_pre_rows.append({
                "alias": alias,
                "epoch": epoch,
                "layer": l,
                "jaccard": j,
                "mode": f"vs_{pre_alias}",
                "topk_neurons_jaccard": TOPK_NEURONS_JACCARD,
            })

    # Jaccard consecutive
    aliases_sorted = sorted([a for a, _ in MODELS], key=lambda a: parse_epoch(a))
    jacc_consecutive_rows: List[Dict[str, Any]] = []
    for i in range(1, len(aliases_sorted)):
        a0 = aliases_sorted[i - 1]
        a1 = aliases_sorted[i]
        e1 = parse_epoch(a1)
        for l in range(LAYER_START, LAYER_END + 1):
            j = jaccard(top_neuron_sets[a0][l], top_neuron_sets[a1][l])
            jacc_consecutive_rows.append({
                "alias": a1,
                "epoch": e1,
                "layer": l,
                "jaccard": j,
                "mode": "consecutive",
                "topk_neurons_jaccard": TOPK_NEURONS_JACCARD,
                "prev_alias": a0,
            })

    # print summaries
    print(f"\n[JACCARD] top-{TOPK_NEURONS_JACCARD} neuron-set overlap vs {pre_alias}")
    for l in range(LAYER_START, LAYER_END + 1):
        vals = [(r["epoch"], r["jaccard"]) for r in jacc_vs_pre_rows if int(r["layer"]) == l]
        vals = sorted(vals, key=lambda x: x[0])
        s = " ".join([f"e{e}:{v:.3f}" for e, v in vals])
        print(f"  [L{l:02d}] {s}")

    print(f"\n[JACCARD] top-{TOPK_NEURONS_JACCARD} neuron-set overlap consecutive")
    for l in range(LAYER_START, LAYER_END + 1):
        vals = [(r["epoch"], r["jaccard"], r["prev_alias"]) for r in jacc_consecutive_rows if int(r["layer"]) == l]
        vals = sorted(vals, key=lambda x: x[0])
        s = " ".join([f"{p}->e{e}:{v:.3f}" for e, v, p in vals])
        print(f"  [L{l:02d}] {s}")

    # save Jaccard CSVs
    jacc_vs_pre_csv = os.path.join(OUT_DIR, "jaccard_vs_pre.csv")
    jacc_consecutive_csv = os.path.join(OUT_DIR, "jaccard_consecutive.csv")
    save_csv(jacc_vs_pre_rows, jacc_vs_pre_csv)
    save_csv(jacc_consecutive_rows, jacc_consecutive_csv)
    print(f"\n[WRITE] {jacc_vs_pre_csv}")
    print(f"[WRITE] {jacc_consecutive_csv}")

    # plot Jaccard heatmaps + per-layer trajectories
    plot_heatmap_layers_x_epochs(
        jacc_vs_pre_rows, "jaccard",
        os.path.join(OUT_DIR, "heatmap_jaccard_vs_pre.png"),
        LAYER_START, LAYER_END,
        title=f"Jaccard(top{TOPK_NEURONS_JACCARD}) vs {pre_alias}",
        vmin=0.0, vmax=1.0
    )
    plot_heatmap_layers_x_epochs(
        jacc_consecutive_rows, "jaccard",
        os.path.join(OUT_DIR, "heatmap_jaccard_consecutive.png"),
        LAYER_START, LAYER_END,
        title=f"Jaccard(top{TOPK_NEURONS_JACCARD}) consecutive",
        vmin=0.0, vmax=1.0
    )

    for l in range(LAYER_START, LAYER_END + 1):
        plot_layer_trajectory(
            jacc_vs_pre_rows, "jaccard",
            os.path.join(OUT_DIR, f"traj_jaccard_vs_pre_L{l:02d}.png"),
            layer=l,
            title=f"Layer {l}: Jaccard(top{TOPK_NEURONS_JACCARD}) vs {pre_alias}"
        )
        plot_layer_trajectory(
            jacc_consecutive_rows, "jaccard",
            os.path.join(OUT_DIR, f"traj_jaccard_consecutive_L{l:02d}.png"),
            layer=l,
            title=f"Layer {l}: Jaccard(top{TOPK_NEURONS_JACCARD}) consecutive"
        )

    print(f"\n[DONE] Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()