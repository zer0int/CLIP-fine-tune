"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

This script instruments the *vision* transformer of CLIP-like models to study how MLP features
(neurons in the expanded dimension, typically 4096 for ViT-L/14) route into the *next* block’s
attention *key* projection, and how various ablations change per-head output magnitudes.

Lazily makes a bunch of synthetic images with text on them on-the-fly.

1) MLP(ℓ) -> K(ℓ+1) routing:
   - For each block ℓ, take Wc = c_proj.weight of the MLP in block ℓ and Wk = k_proj.weight of
     the attention in block ℓ+1.
   - Slice Wk by head (head_dim chunks) and compute per-head per-feature scores:
         S[h, f] = || (Wk_head[h] @ Wc[:, f]) ||_2
     where f indexes the MLP expanded features (0..4095).
   - From S, derive:
       * per-head thresholds (quantile),
       * "collaborators" (features high in >1 head),
       * "specialists" (features high in exactly 1 head),
       * per-head top-K lists,
       * Spearman + Jaccard@K comparisons vs the pretrained baseline.

2) 'Register' feature discovery (per layer):
   - See: "Vision Transformers Need Registers", https://arxiv.org/abs/2309.16588
   - Uses a small image set (real or synthetic) to identify "register-like" MLP features for a
     given layer ℓ by combining:
       * per-token patch norms in block output space (proxy for register tokens),
       * MLP pre-activation -> activation (QuickGELU fallback) -> feature magnitude,
       * column norms of Wc (c_proj) to weight feature impact.
   - Select regs via absolute threshold, top-N, or quantile; with fallbacks if no tokens/features pass.

3) Per-head output L2 norms (three flavors):
   - Heuristic block output slice:
       * hooks block outputs and measures per-head mean ||x||_2 by reshaping [B,T,E] -> [B,T,H,D].
   - True attention PRE/POST out_proj norms:
       * recompute attention: A = softmax(QK^T / sqrt(D)), out_pre = A @ V in head space,
       * out_post applies per-head slices of out_proj.weight into embed space,
       * reports means over all query tokens and over CLS-only.
   - Attention diagnostics:
       * entropy of attention distribution (all tokens vs CLS),
       * reg-mass (attention mass to "high-norm patch tokens" used as reg proxy; CLS excluded as key),
       * Q/K/V norms (all tokens vs CLS).

4) Controlled ablation variants (multi-run):
   - baseline (no hooks)
   - reg_nuke_only: zero selected MLP neurons at c_fc output for chosen blocks (REG_NEURONS)
   - zero_attn_only / zero_mlp_only: zero the residual contribution of attention and/or MLP for
     selected blocks (ablate_blocks)
   - zero_attn_mlp: both
   - reg_nuke_plus_zero_attn_mlp: combined
"""

from __future__ import annotations

import os
import argparse
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import hashlib
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any


import attnclipindiv as clip

from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

# ============================================================
# Models: OpenAI / local path .pt .safetensors / HuggingFace Hub
# ============================================================

# WARNING:
# Depending on settings,
# This will take a significant amount of time, 
# and will result in hundreds of MB of data (including plots).

MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
    ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
    ("regr-norm", "zer0int/CLIP-Regression-ViT-L-14"),
    ("regr-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
]

REG_NEURONS: Dict[int, List[int]] = {
    11: [9, 987, 1967, 2555, 3661, 3784],
    12: [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868],
}

DEFAULT_OUT_DIR = "out_eval_measure/attn_reg_delta_norms"

FONT_PATH_SYN_IMAGES = "C:/Windows/Fonts/arial.ttf" # for synthetic images, if applicable

def parse_arguments():
    p = argparse.ArgumentParser(description="Rage attending keys: MLP(ℓ)->K(ℓ+1) per head; multi-model; always NPZ+summary; reg discovery+verify; L2 output norms + deltas")
    
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reg_image_dir", default="image_sets/adv_dataset/typoattack", help="Folder of images to use (optional)")    
    p.add_argument("--font_path", default=FONT_PATH_SYN_IMAGES, help="Font path for synthetic text images")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output folder")
    p.add_argument("--dpi", type=int, default=200)
    
    p.add_argument("--layers", default="all", help="e.g. all or 0,1,2,10,11,12")
    p.add_argument("--threshold_quantile", type=float, default=0.98, help="Quantile threshold per head")
    p.add_argument("--topk_per_head", type=int, default=100, help="How many top features to list per head")

    p.add_argument("--attn_contrib_diag", action="store_true", help="Compute effective-contribution diagnostics: entropy(contrib), KL(attn||contrib), reg attn vs reg contrib masses + correlations.")
    p.set_defaults(attn_contrib_diag=True)
    p.add_argument("--attn_special_heads", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", help="Comma-separated head indices to overlay on CLS entropy/KL plots (e.g. '10,13').")
    p.add_argument("--reg_random_sample", action="store_true", help="When loading real images from --reg_image_dir, randomly sample --reg_num_images from ALL images under subfolders.")
    p.set_defaults(reg_random_sample=True)

    p.add_argument("--outcomes_csv", default="", help="Optional CSV mapping image->outcome scalar for correlations. Requires columns: path,outcome (customizable via --outcomes_path_col/--outcomes_value_col).")
    p.add_argument("--outcomes_path_col", default="path")
    p.add_argument("--outcomes_value_col", default="outcome")

    p.add_argument("--nuke_regs", action="store_true", help="If set, run TWO variants: baseline and (nuke regs + zero attn + zero mlp).")
    
    p.add_argument("--apply_ln1_gamma", action="store_true", help="Approx: scale K-proj columns by next block ln_1 gamma (ignores mean/std).")
    p.add_argument("--print_weight_sanity", action="store_true", help="Print Wc/Wk fingerprints + pairwise diffs for every scanned layer")    

    p.add_argument("--reg_abs_threshold", type=float, default=70.0, help="Abs threshold on impact_f (see gating below)")
    p.add_argument("--reg_topn", type=int, default=0, help="If >0, take top-N regs by impact instead of threshold")
    p.add_argument("--reg_quantile", type=float, default=0.0, help="If >0, take regs with impact >= quantile (e.g. 0.999). Ignored if reg_topn>0.")

    p.add_argument("--ablate_blocks", default="12", help="Comma-separated block indices for attach_zero_attn/attach_zero_mlp in nuke_regs mode (default: 12).")
    p.add_argument("--focus_layers", default="11,12,20,22", help="compare_special: comma-separated block indices to focus on (e.g. 11,12,20)")    
    p.add_argument("--delta_focus_layer", type=int, default=22, help="Layer to print per-head before/after/delta table for (default: 12).")
    p.add_argument("--delta_topk", type=int, default=16, help="How many heads to show in the focused delta table (default: 16).")

    p.add_argument("--summary_topk", type=int, default=30, help="Top-K indices to show in concise summary per head")
    p.add_argument("--summary_max_heads", type=int, default=16, help="Show only the top-N most 'reg-focused' heads in summary (per model, per layer)")
    p.add_argument("--compare_topk", type=int, default=30, help="Top-K for Jaccard comparisons per head (vs pretrained)")
    p.add_argument("--compare_max_heads", type=int, default=16, help="Compare only the top-N most reg-focused heads (ranked per model) per layer (vs pretrained)")
    
    p.add_argument("--reg_token_norm_threshold", type=float, default=70.0, help="Only consider patch tokens with ||x_patch||_2 > this when computing neuron impact.")
    p.add_argument("--reg_token_fallback_topk", type=int, default=16, help="If no patch tokens exceed reg_token_norm_threshold, use top-K patch tokens by norm instead.")
    p.add_argument("--reg_fallback_topn", type=int, default=32, help="If thresholding yields 0 regs and reg_topn/reg_quantile not set, fall back to top-N regs by impact.")

    p.add_argument("--attn_diag", action="store_true", help="Compute attention diagnostics (entropy/reg-mass/QKV norms) and save plots/NPZs. Default: enabled.")
    p.set_defaults(attn_diag=True)
    p.add_argument("--reg_num_images", type=int, default=236, help="How many images to use")
    p.add_argument("--reg_batch_size", type=int, default=64, help="Batch size for forward passes")
    p.add_argument("--reg_seed", type=int, default=0, help="Seed for synthetic reg images")
    p.add_argument("--reg_use_synth", action="store_true", help="If dir empty/insufficient, generate synthetic images")
    p.set_defaults(reg_use_synth=True)
    p.add_argument("--reg_image_size", type=int, default=224, help="Synthetic image size (square)")
    p.add_argument("--debug", action="store_true", help="Verbose logging (file paths, per-layer prints, etc.)")

    return p.parse_args()



def png_path(local_path: str, prefix: str, stem: str) -> str:
    """
    Build a standardized PNG filename:
        <local_path>/<prefix>__<stem>.png
    """
    ensure_dir(local_path)
    prefix = (prefix or "").strip()
    if not prefix:
        return os.path.join(local_path, f"{stem}.png")
    return os.path.join(local_path, f"{prefix}__{stem}.png")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _log(msg: str, debug: bool, always: bool = False):
    if always or debug:
        print(msg)

def parse_layers_arg(layers_arg: str, n_layers: int) -> List[int]:
    s = layers_arg.strip().lower()
    if s == "all":
        return list(range(0, max(n_layers - 1, 0)))
    out = []
    for x in layers_arg.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    out = [l for l in out if 0 <= l < n_layers - 1]
    out.sort()
    return out

def parse_int_list_arg(s: str) -> List[int]:
    out: List[int] = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except Exception:
            continue
    out = sorted(list(dict.fromkeys(out)))
    return out

def _zero_like(out: Any) -> Any:
    """
    Return a zeroed version of `out` while preserving structure.
    Handles:
      - Tensor
      - (Tensor, ...) tuples
      - lists/dicts (rare, but let's be defensive)
    """
    if torch.is_tensor(out):
        return out * 0.0
    if isinstance(out, tuple):
        if len(out) == 0:
            return out
        first = out[0]
        if torch.is_tensor(first):
            return (first * 0.0,) + out[1:]
        return out
    if isinstance(out, list):
        return [(_zero_like(x) if torch.is_tensor(x) else x) for x in out]
    if isinstance(out, dict):
        return {k: (_zero_like(v) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out


def attach_zero_attn(block: torch.nn.Module) -> Any:
    """
    Zero the attention *output* for a single ResBlock (so the residual adds ~0 from attn).
    Returns the hook handle (call .remove()).
    """
    if not hasattr(block, "attn"):
        raise AttributeError("Block has no .attn attribute; cannot attach_zero_attn().")

    def hook(_module, _inp, output):
        return _zero_like(output)

    h = block.attn.register_forward_hook(hook)
    return h


def attach_zero_mlp(block: torch.nn.Module) -> Any:
    """
    Zero the MLP *output* for a single ResBlock (so the residual adds ~0 from MLP).
    Returns the hook handle (call .remove()).
    """
    if not hasattr(block, "mlp"):
        raise AttributeError("Block has no .mlp attribute; cannot attach_zero_mlp().")

    def hook(_module, _inp, output):
        return _zero_like(output)

    h = block.mlp.register_forward_hook(hook)
    return h


def attach_zero_attn_for_blocks(visual: torch.nn.Module, block_idxs: List[int]) -> List[Any]:
    handles: List[Any] = []
    blocks = visual.transformer.resblocks
    for bi in block_idxs:
        if not (0 <= bi < len(blocks)):
            continue
        handles.append(attach_zero_attn(blocks[bi]))
    return handles

def attach_zero_mlp_for_blocks(visual: torch.nn.Module, block_idxs: List[int]) -> List[Any]:
    handles: List[Any] = []
    blocks = visual.transformer.resblocks
    for bi in block_idxs:
        if not (0 <= bi < len(blocks)):
            continue
        handles.append(attach_zero_mlp(blocks[bi]))
    return handles


@torch.no_grad()
def scan_register_mover_candidates(
    model: torch.nn.Module,
    images_cpu: torch.Tensor,
    device: str,
    batch_size: int,
    delta_threshold: float = 1.0,
    adaptive_mult: float = 2.5,
    patch_beg: int = 1,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    use_max: bool = True,
    debug: bool = False,
) -> Dict[int, Dict[str, Any]]:

    # Hooks the MLP activation (gelu/act) output per block, then:
    #  - select "high-norm patch tokens" in that activation space via adaptive threshold
    #  - compute per-dim delta vs per-dim median patch activation
    #  - mark dims where delta > delta_threshold

    model.eval()
    blocks = model.visual.transformer.resblocks
    L = len(blocks)
    if end_layer is None:
        end_layer = L

    out: Dict[int, Dict[str, Any]] = {}
    hooks: List[Any] = []

    def _get_act_module(blk):
        # try common patterns
        if hasattr(blk.mlp, "gelu"):
            return blk.mlp.gelu
        if hasattr(blk.mlp, "act"):
            return blk.mlp.act
        # sometimes Sequential: [Linear, GELU, Linear]
        if isinstance(blk.mlp, torch.nn.Sequential) and len(blk.mlp) >= 2:
            return blk.mlp[1]
        return None

    def make_hook(layer_idx: int):
        def hook(_m, _inp, output):
            if not torch.is_tensor(output):
                return output

            x = _maybe_bt(output)  # -> [B,T,C] if possible
            if x.dim() != 3:
                return output

            B, T, C = x.shape
            if T <= patch_beg:
                return output

            patches = x[:, patch_beg:, :]  # [B, Tp, C]
            norms = torch.linalg.vector_norm(patches, dim=-1)  # [B, Tp]
            flat = norms.flatten()
            if flat.numel() == 0:
                return output

            med = flat.median().item()
            thr = float(med * adaptive_mult)
            high_mask = norms > thr

            if not high_mask.any():
                return output

            # init accumulators for this layer if needed
            if layer_idx not in out:
                out[layer_idx] = {
                    "count": np.zeros((C,), dtype=np.int64),
                    "max_delta": np.zeros((C,), dtype=np.float32),
                    "seen_high_tokens": 0,
                    "adaptive_thr_last": thr,
                    "median_norm_last": med,
                }
            out[layer_idx]["adaptive_thr_last"] = thr
            out[layer_idx]["median_norm_last"] = med

            for b in range(B):
                mask_b = high_mask[b]  # [Tp]
                if not mask_b.any():
                    continue

                vis_b = patches[b, mask_b, :]     # [n_high, C]
                ref_b = patches[b, :, :]          # [Tp, C]
                median_vec = ref_b.median(dim=0).values  # [C]

                if use_max:
                    delta = (vis_b - median_vec).max(dim=0).values  # [C]
                else:
                    delta = (vis_b - median_vec).mean(dim=0)        # [C]

                idxs = (delta > float(delta_threshold)).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue

                # update counts and max_delta
                idxs_cpu = idxs.detach().cpu().numpy().astype(np.int64)
                out[layer_idx]["count"][idxs_cpu] += 1
                d_cpu = delta[idxs].detach().cpu().numpy().astype(np.float32)
                out[layer_idx]["max_delta"][idxs_cpu] = np.maximum(out[layer_idx]["max_delta"][idxs_cpu], d_cpu)
                out[layer_idx]["seen_high_tokens"] += int(mask_b.sum().item())

            return output
        return hook

    # register hooks
    for li in range(int(start_layer), int(end_layer)):
        if not (0 <= li < L):
            continue
        act_mod = _get_act_module(blocks[li])
        if act_mod is None:
            continue
        hooks.append(act_mod.register_forward_hook(make_hook(li)))

    try:
        N = int(images_cpu.shape[0])
        bs = max(1, int(batch_size))
        for i in range(0, N, bs):
            x = images_cpu[i:i + bs].to(device)
            _ = model.encode_image(x)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    # finalize: convert to candidate sets
    for li, d in out.items():
        count = d["count"]
        maxd = d["max_delta"]
        cand = set(np.where(count > 0)[0].tolist())
        # also include anything with big max delta even if count is weirdly zero
        cand |= set(np.where(maxd > float(delta_threshold))[0].tolist())
        d["candidates"] = cand
        if debug:
            top = np.argsort(-maxd)[:20]
            print(f"[mover-scan] L{li}: candidates={len(cand)} top20_by_maxΔ=" +
                  ", ".join([f"{int(j)}:{float(maxd[j]):.3f}" for j in top]))
    return out


def correlate_and_plot_late_qnorm_vs_heuristic(
    tag_delta_dir: str,
    n_layers: int,
    model_tag: str,
    late_k: int = 6,
    dpi: int = 200,
    debug: bool = False,    
):
    """
    Expects NPZs in tag_delta_dir:
      - delta__heuristic_block_output_slice.npz  (delta shape [L, Hslices])
      - delta__attn_qnorm_all.npz                (delta shape [L, Hheads])

    Correlates only late layers: layers in [n_layers-late_k, ..., n_layers-1].
    """
    p_heur = os.path.join(tag_delta_dir, "delta__heuristic_block_output_slice.npz")
    p_q = os.path.join(tag_delta_dir, "delta__attn_qnorm_all.npz")

    if not (os.path.exists(p_heur) and os.path.exists(p_q)):
        _log(f"[corr] missing NPZs for correlation in {tag_delta_dir} (need attn_qnorm_all + heuristic)", debug, always=False)
        return

    A = np.load(p_heur)["delta"]  # [L, Hs]
    Q = np.load(p_q)["delta"]     # [L, Hh]
    L1 = min(int(A.shape[0]), int(Q.shape[0]), int(n_layers))
    A = A[:L1]
    Q = Q[:L1]

    lo = max(0, L1 - int(late_k))
    A_late = A[lo:L1].reshape(-1)
    Q_late = Q[lo:L1].reshape(-1)

    # Pearson
    A0 = A_late.astype(np.float64)
    Q0 = Q_late.astype(np.float64)
    A0 -= A0.mean()
    Q0 -= Q0.mean()
    pearson = float((A0 @ Q0) / ((np.linalg.norm(A0) * np.linalg.norm(Q0)) + 1e-12))

    # Spearman via ranks
    ra = A_late.argsort().argsort().astype(np.float64)
    rq = Q_late.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rq -= rq.mean()
    spearman = float((ra @ rq) / ((np.linalg.norm(ra) * np.linalg.norm(rq)) + 1e-12))

    # Save txt + scatter
    with open(os.path.join(tag_delta_dir, "corr__late_qnorm_vs_heuristic.txt"), "w", encoding="utf-8") as f:
        f.write(f"late_layers=[{lo}..{L1-1}] late_k={late_k}\n")
        f.write(f"pearson={pearson:.6f}\n")
        f.write(f"spearman={spearman:.6f}\n")

    plt.figure(figsize=(7, 6))
    plt.scatter(A_late, Q_late, s=10)
    plt.xlabel("Δ heuristic_block_output_slice (late layers, flattened)")
    plt.ylabel("Δ attn_qnorm_all (late layers, flattened)")
    plt.title(f"Late-layer correlation: pearson={pearson:.3f} spearman={spearman:.3f}")
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    plt.savefig(png_path(tag_delta_dir, model_tag, "corr__late_qnorm_vs_heuristic"), dpi=dpi)
    plt.close()


def attach_zero_attn_mlp_for_blocks(visual: torch.nn.Module, block_idxs: List[int]) -> List[Any]:
    """
    Attach BOTH:
      - attach_zero_attn(block)
      - attach_zero_mlp(block)
    for each block idx in block_idxs.
    """
    handles: List[Any] = []
    blocks = visual.transformer.resblocks
    for bi in block_idxs:
        if not (0 <= bi < len(blocks)):
            continue
        handles.append(attach_zero_attn(blocks[bi]))
        handles.append(attach_zero_mlp(blocks[bi]))
    return handles

def attach_reg_neuron_nuke_hooks(visual: torch.nn.Module) -> List[Any]:
    """
    Zero specified MLP expanded dims at c_fc output (pre-gelu), for blocks in REG_NEURONS.
    Returns list of hook handles so we can remove them.
    """
    handles: List[Any] = []
    if not REG_NEURONS:
        return handles

    print("[INFO] Attaching register-neuron nuke hooks...")
    for block_idx, block in enumerate(visual.transformer.resblocks):
        if block_idx not in REG_NEURONS:
            continue
        idxs = torch.tensor(REG_NEURONS[block_idx], dtype=torch.long)

        c_fc = block.mlp.c_fc if hasattr(block.mlp, "c_fc") else block.mlp[0]

        def make_hook(idxs_: torch.Tensor, blk_idx: int):
            def hook(_module, _inp, output):
                out = output.clone()
                out[..., idxs_.to(out.device)] = 0.0
                return out
            hook.__name__ = f"reg_nuke_block_{blk_idx}"
            return hook

        h = c_fc.register_forward_hook(make_hook(idxs, block_idx))
        handles.append(h)
        print(f"[INFO] Hook on block {block_idx} c_fc for neurons {REG_NEURONS[block_idx]}")
    return handles


def tensor_sig(x: torch.Tensor, n_bytes: int = 4096) -> str:
    y = x.detach().to("cpu").contiguous().float()
    b = y.numpy().tobytes()
    h = hashlib.sha1(b[: min(len(b), n_bytes)]).hexdigest()
    return f"{h}|shape={tuple(y.shape)}|mean={y.mean().item():.6g}|std={y.std().item():.6g}"


def weight_diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().to("cpu").contiguous().float()
    b = b.detach().to("cpu").contiguous().float()
    da = (a - b)
    fa = torch.linalg.vector_norm(a).item()
    fb = torch.linalg.vector_norm(b).item()
    fd = torch.linalg.vector_norm(da).item()
    denom = (fa + fb) * 0.5 + 1e-12
    rel = fd / denom

    av = a.view(-1)
    bv = b.view(-1)
    cos = torch.dot(av, bv).item() / (
        torch.linalg.vector_norm(av).item() * torch.linalg.vector_norm(bv).item() + 1e-12
    )
    return {"fro_a": fa, "fro_b": fb, "fro_diff": fd, "rel_fro_diff": rel, "cos": cos}


def save_heatmap(mat: np.ndarray, out_path: str, title: str, dpi: int = 200, regs: Optional[set] = None):
    plt.figure(figsize=(16, 5))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.xlabel("Expanded feature (MLP neuron index, 0..4095)")
    plt.ylabel("Head")
    plt.title(title)
    plt.colorbar(label="|| Wk_next_head @ Wcproj_current ||_2")

    if regs:
        for r in sorted(regs):
            plt.axvline(r, linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def find_collaborators(k_norms: np.ndarray, threshold_quantile: float = 0.98):
    num_heads, num_feats = k_norms.shape
    thresholds = np.quantile(k_norms, threshold_quantile, axis=1)
    mask = k_norms >= thresholds[:, None]

    collaborator_dict: Dict[int, List[int]] = {}
    specialist_dict: Dict[int, List[int]] = {}
    per_head_dict: Dict[int, List[int]] = {h: [] for h in range(num_heads)}

    for f in range(num_feats):
        heads = np.where(mask[:, f])[0].tolist()
        if len(heads) > 1:
            collaborator_dict[f] = heads
        elif len(heads) == 1:
            specialist_dict[f] = heads
        for h in heads:
            per_head_dict[h].append(f)

    return collaborator_dict, specialist_dict, per_head_dict, thresholds


@torch.no_grad()
def mlp_neuron_to_next_k_head_scores(model, layer_id: int, device: str, apply_ln1_gamma: bool = False):
    blocks = model.visual.transformer.resblocks
    assert 0 <= layer_id < len(blocks) - 1

    blk = blocks[layer_id]
    nxt = blocks[layer_id + 1]

    Wc = blk.mlp.c_proj.weight.detach().to(device)   # [1024, 4096]
    Wk = nxt.attn.k_proj.weight.detach().to(device)  # [1024, 1024]

    n_heads = int(nxt.attn.num_heads)
    d_model = int(Wk.shape[0])
    assert d_model % n_heads == 0
    head_dim = d_model // n_heads

    if apply_ln1_gamma and hasattr(nxt, "ln_1"):
        gamma = nxt.ln_1.weight.detach().to(device)  # [1024]
        Wk = Wk * gamma.unsqueeze(0)

    Wk_heads = Wk.view(n_heads, head_dim, d_model)
    M = torch.matmul(Wk_heads, Wc)                   # [heads, head_dim, 4096]
    S = torch.linalg.vector_norm(M, dim=1)           # [heads, 4096]
    return S


def _reg_focus_score(scores_h: np.ndarray, regs: Optional[set]) -> float:
    if not regs:
        return 0.0
    return float(sum(scores_h[r] for r in regs if 0 <= r < scores_h.shape[0]))


def _topk_pairs(scores_h: np.ndarray, k: int) -> List[Tuple[int, float]]:
    idx = np.argsort(-scores_h)[:k]
    return [(int(i), float(scores_h[i])) for i in idx]


def _fmt_pairs(pairs: List[Tuple[int, float]]) -> str:
    return ", ".join([f"{i}:{s:.6f}" for (i, s) in pairs])


def _fmt_ints(xs: List[int]) -> str:
    return "[" + ", ".join(map(str, xs)) + "]"


def spearmanr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    return float((ra @ rb) / denom)


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    inter = len(ia & ib)
    union = max(1, len(ia | ib))
    return inter / union


def save_txt_report(
    out_path: str,
    k_norms: np.ndarray,
    colab: Dict[int, List[int]],
    spec: Dict[int, List[int]],
    per_head: Dict[int, List[int]],
    thresholds: np.ndarray,
    topk_per_head: int,
    regs: Optional[set],
):
    num_heads, num_feats = k_norms.shape
    regs = regs or set()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Rage attending keys report ===\n")
        f.write(f"Matrix shape: heads={num_heads}, feats={num_feats}\n")
        f.write(f"regs_used={sorted(regs)}\n\n")

        f.write("Per-head thresholds:\n")
        f.write("  " + ", ".join([f"h{h}:{thresholds[h]:.6f}" for h in range(num_heads)]) + "\n\n")

        f.write("=== Collaborator features (above threshold in >1 head): ===\n")
        for feat, heads in sorted(colab.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True):
            tag = " [REG]" if feat in regs else ""
            f.write(f"Feature {feat}: heads {heads}{tag}\n")

        f.write("\n=== Specialist features (above threshold in exactly 1 head): ===\n")
        for feat, heads in sorted(spec.items(), key=lambda kv: kv[0]):
            tag = " [REG]" if feat in regs else ""
            f.write(f"Feature {feat}: head {heads}{tag}\n")

        f.write("\n=== Top-K features per head (by score): ===\n")
        for h in range(num_heads):
            scores_h = k_norms[h]
            top_pairs = _topk_pairs(scores_h, topk_per_head)
            regs_top = sorted([i for (i, _) in top_pairs if i in regs])
            regs_above = sorted([i for i in per_head[h] if i in regs])

            f.write(f"\nHead {h} (threshold={thresholds[h]:.6f}) top{topk_per_head}:\n")
            f.write(_fmt_pairs(top_pairs) + "\n")
            f.write(f"  regs_in_top{topk_per_head}={regs_top}\n")
            f.write(f"  regs_above_threshold={regs_above}\n")

        f.write("\n=== Features above threshold per head (indices only): ===\n")
        for h in range(num_heads):
            f.write(f"Head {h}: {per_head[h]}\n")




def save_summary_txt(out_path: str, lines: List[str]):
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def make_concise_summary(
    layer: int,
    k_norms: np.ndarray,
    thresholds: np.ndarray,
    per_head: Dict[int, List[int]],
    regs: Optional[set],
    summary_topk: int,
    summary_max_heads: int,
    model_tag: str,
) -> List[str]:
    regs = regs or set()
    num_heads, _ = k_norms.shape

    focus = []
    for h in range(num_heads):
        focus.append((h, _reg_focus_score(k_norms[h], regs)))
    focus.sort(key=lambda x: x[1], reverse=True)
    focus = focus[:max(1, summary_max_heads)]

    lines = []
    lines.append(f"{model_tag}:")
    lines.append(f"[L{layer:02d}->{layer+1:02d}] regs_used={sorted(regs)}")
    for (h, sc) in focus:
        scores_h = k_norms[h]
        top_pairs = _topk_pairs(scores_h, summary_topk)
        top_idx = [i for (i, _) in top_pairs]

        regs_top = [r for r in sorted(regs) if r in top_idx]
        regs_above = [r for r in sorted(regs) if r in per_head[h]]
        regs_scores = [(r, float(scores_h[r])) for r in sorted(regs) if 0 <= r < scores_h.shape[0]]
        regs_scores.sort(key=lambda x: x[1], reverse=True)

        lines.append(
            f"  H{h:02d} thr={thresholds[h]:.6f} reg_focus_sum={sc:.6f} "
            f"regs_top{summary_topk}={regs_top} regs_above_thr={regs_above}"
        )
        if regs:
            lines.append(f"    regs_scores=" + _fmt_pairs(regs_scores))
        lines.append(f"    top{summary_topk}=" + _fmt_ints(top_idx))
    return lines


def make_pair_compare_block(
    layer: int,
    tag_a: str,
    tag_b: str,
    mats: Dict[str, np.ndarray],
    regs_for_focus: Optional[set],
    compare_topk: int,
    compare_max_heads: int,
) -> List[str]:
    regs_for_focus = regs_for_focus or set()
    MA, MB = mats[tag_a], mats[tag_b]
    num_heads = MA.shape[0]

    def top_heads(M: np.ndarray) -> List[int]:
        scores = [(h, _reg_focus_score(M[h], regs_for_focus)) for h in range(num_heads)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [h for (h, _) in scores[:max(1, compare_max_heads)]]

    focus = sorted(set(top_heads(MA) + top_heads(MB)))

    lines = []
    lines.append(
        f"[COMPARE {tag_a} vs {tag_b} | L{layer:02d}->{layer+1:02d}] compare_topk={compare_topk} regs_focus={sorted(regs_for_focus)}"
    )
    lines.append("  head |  spearman  jacc@k | worst_regΔ")
    for h in focus:
        a, b = MA[h], MB[h]
        sp = spearmanr_np(a, b)
        jc = jaccard_topk(a, b, compare_topk)

        if regs_for_focus:
            spreads = {r: float(abs(a[r] - b[r])) for r in regs_for_focus}
            r_star = max(spreads.keys(), key=lambda r: spreads[r])
            worst = f"r={r_star} |Δ|={spreads[r_star]:.6f}"
        else:
            worst = "n/a"

        lines.append(f"  H{h:02d} |   {sp: .4f}    {jc: .3f} | {worst}")
    return lines


def _maybe_load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    except Exception:
        pass
    return ImageFont.load_default()


def synth_reg_images(num: int, size: int, seed: int, font_path: str) -> List[Image.Image]:
    rng = np.random.default_rng(seed)
    imgs: List[Image.Image] = []
    words = ["STOP", "WARNING", "TEXT", "CLIP", "REGISTER", "NOISE", "HELLO", "AI", "K", "V", "Q"]
    for i in range(num):
        mode = i % 4
        im = Image.new("RGB", (size, size), (0, 0, 0))
        dr = ImageDraw.Draw(im)

        if mode == 0:
            arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
            im = Image.fromarray(arr, mode="RGB")
            dr = ImageDraw.Draw(im)
        elif mode == 1:
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            for y in range(size):
                val = int(255 * y / max(1, size - 1))
                arr[y, :, :] = (val, val, val)
            for x in range(0, size, max(4, size // 32)):
                arr[:, x:x + 2, :] = 255 - arr[:, x:x + 2, :]
            im = Image.fromarray(arr, mode="RGB")
            dr = ImageDraw.Draw(im)
        elif mode == 2:
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            step = max(8, size // 16)
            for y in range(0, size, step):
                for x in range(0, size, step):
                    c = 255 if ((x // step + y // step) % 2 == 0) else 0
                    arr[y:y + step, x:x + step, :] = c
            im = Image.fromarray(arr, mode="RGB")
            dr = ImageDraw.Draw(im)
        else:
            dr.rectangle([0, 0, size - 1, size - 1], fill=(20, 20, 20))

        font = _maybe_load_font(font_path, size=max(18, size // 9))
        w = words[int(rng.integers(0, len(words)))]
        tw, th = dr.textsize(w, font=font)
        x = max(0, (size - tw) // 2)
        y = int(rng.integers(0, max(1, size // 5)))
        color = (255, 255, 255) if mode != 0 else (0, 0, 0)
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (0, 0)]:
            dr.text((x + dx, y + dy), w, font=font, fill=color)

        imgs.append(im.convert("RGB"))
    return imgs


def load_reg_images(args, preprocess):
    paths: List[str] = []
    if args.reg_image_dir and os.path.isdir(args.reg_image_dir):
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        for root, _, files in os.walk(args.reg_image_dir):
            for fn in files:
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(root, fn))
        paths.sort()

    sel: List[str] = []
    pil_imgs: List[Image.Image] = []
    if paths:
        sel = paths
        if getattr(args, "reg_random_sample", False) and len(paths) > args.reg_num_images:
            rng = np.random.default_rng(int(args.reg_seed))
            idx = rng.choice(len(paths), size=int(args.reg_num_images), replace=False)
            sel = [paths[int(i)] for i in idx]
        else:
            sel = paths[: args.reg_num_images]

        for pth in sel:
            try:
                pil_imgs.append(Image.open(pth).convert("RGB"))
            except Exception:
                continue

    reg_paths: List[str] = list(sel)
    is_synth: List[int] = [0] * len(reg_paths)

    if len(pil_imgs) < args.reg_num_images and args.reg_use_synth:
        need = args.reg_num_images - len(pil_imgs)
        pil_imgs.extend(synth_reg_images(need, args.reg_image_size, args.reg_seed, args.font_path))
        # synth placeholders
        for i in range(need):
            reg_paths.append(f"__synth__/synth_{i:06d}.png")
            is_synth.append(1)

    if not pil_imgs:
        raise RuntimeError("No images available (even synth disabled?). Provide --reg_image_dir or keep --reg_use_synth.")

    xs = [preprocess(im) for im in pil_imgs]
    x = torch.stack(xs, dim=0).cpu()
    reg_basenames = [os.path.basename(p) for p in reg_paths]
    return x, reg_paths, reg_basenames, is_synth



def _apply_mlp_activation(blk, x: torch.Tensor) -> torch.Tensor:
    if hasattr(blk.mlp, "gelu"):
        try:
            return blk.mlp.gelu(x)
        except Exception:
            pass
    if hasattr(blk.mlp, "act"):
        try:
            return blk.mlp.act(x)
        except Exception:
            pass
    return x * torch.sigmoid(1.702 * x)  # quickgelu fallback


def _maybe_bt(x: torch.Tensor) -> torch.Tensor:
    # handle possible [T,B,C] vs [B,T,C]
    if x.dim() != 3:
        return x
    B_first = (x.shape[0] <= 512 and x.shape[1] > x.shape[0])  # weak heuristic
    if not B_first and x.shape[1] <= 512:
        return x.permute(1, 0, 2).contiguous()
    return x


import math

def _safe_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p: [..., K] with sum=1
    return -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)

def _safe_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # KL(p||q) over last dim
    p1 = p.clamp_min(eps)
    q1 = q.clamp_min(eps)
    return (p1 * (torch.log(p1) - torch.log(q1))).sum(dim=-1)

def _corr_from_sums(n: np.ndarray, sx: np.ndarray, sy: np.ndarray, sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray) -> np.ndarray:
    # Pearson corr computed elementwise for arrays (e.g. [L,H])
    denom = np.sqrt(np.maximum(0.0, n * sxx - sx * sx) * np.maximum(0.0, n * syy - sy * sy)) + 1e-12
    return (n * sxy - sx * sy) / denom

def _load_outcomes_csv(path: str, path_col: str = "path", value_col: str = "outcome") -> Dict[str, float]:
    # Maps basename -> outcome float (robust across absolute path differences)
    import csv
    out: Dict[str, float] = {}
    if not path or (not os.path.exists(path)):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if path_col not in row or value_col not in row:
                continue
            p = (row[path_col] or "").strip()
            v = (row[value_col] or "").strip()
            if not p or not v:
                continue
            try:
                out[os.path.basename(p)] = float(v)
            except Exception:
                continue
    return out


@torch.no_grad()
def compute_true_attn_contrib_diagnostics(
    model: torch.nn.Module,
    images_cpu: torch.Tensor,
    device: str,
    batch_size: int,
    reg_token_norm_threshold: float = 70.0,
    reg_token_fallback_topk: int = 16,
    outcomes_map: Optional[Dict[str, float]] = None,
    image_basenames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Effective contribution diagnostics.

    For each layer/head, recompute attention:
        A = softmax(QK^T / sqrt(d))

    Define patch-only distributions for CLS query token:
        a = A[CLS -> patches], renormalized over patches
        c_raw = a * ||V_patch||_2        (per-key scalar "effective magnitude")
        c = c_raw / sum(c_raw)

    Metrics:
      - entropy_attn_patch_cls:    H(a)
      - entropy_contrib_patch_cls: H(c)
      - kl_attn_to_contrib_cls:    KL(a || c)
      - reg_attn_mass_cls:         sum_{k in reg} a_k      (reg keys defined by high ||x_patch||_2)
      - reg_contrib_mass_cls:      sum_{k in reg} c_k
      - corr_reg_attn_vs_contrib:  corr(reg_attn_mass_cls, reg_contrib_mass_cls) over images

    Optionally:
      - corr_vs_outcome_{attn,contrib}: correlation vs provided outcome scalar.
    """
    use_outcome = bool(outcomes_map) and (image_basenames is not None)
    
    model.eval()
    blocks = model.visual.transformer.resblocks
    num_layers = len(blocks)
    num_heads = int(blocks[0].attn.num_heads)

    sums = {
        "entropy_attn_patch_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
        "entropy_contrib_patch_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
        "kl_attn_to_contrib_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
        "reg_attn_mass_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
        "reg_contrib_mass_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
    }
    den_cls = np.zeros((num_layers,), dtype=np.float64)  # counts B

    # correlation accumulators for reg_attn_mass_cls vs reg_contrib_mass_cls
    n_corr = np.zeros((num_layers, num_heads), dtype=np.float64)
    sx = np.zeros((num_layers, num_heads), dtype=np.float64)
    sy = np.zeros((num_layers, num_heads), dtype=np.float64)
    sxx = np.zeros((num_layers, num_heads), dtype=np.float64)
    syy = np.zeros((num_layers, num_heads), dtype=np.float64)
    sxy = np.zeros((num_layers, num_heads), dtype=np.float64)

    # optional correlations vs outcome
    use_outcome = bool(outcomes_map) and (image_basenames is not None)
    n_out = np.zeros((num_layers, num_heads), dtype=np.float64)
    so = np.zeros((num_layers, num_heads), dtype=np.float64)
    so2 = np.zeros((num_layers, num_heads), dtype=np.float64)

    sx_attn_o = np.zeros((num_layers, num_heads), dtype=np.float64)
    sxo_attn = np.zeros((num_layers, num_heads), dtype=np.float64)  # alias, kept for clarity
    sx_contrib_o = np.zeros((num_layers, num_heads), dtype=np.float64)

    # For corr(attn_mass, outcome) and corr(contrib_mass, outcome), we also need sum of x^2
    sx2_contrib = np.zeros((num_layers, num_heads), dtype=np.float64)


    # x = reg_attn_mass_cls, y = reg_contrib_mass_cls
    sx_attn = np.zeros((num_layers, num_heads), dtype=np.float64)
    sx2_attn = np.zeros((num_layers, num_heads), dtype=np.float64)
    sxo_attn = np.zeros((num_layers, num_heads), dtype=np.float64)  # sum(x*o)

    sy_contrib = np.zeros((num_layers, num_heads), dtype=np.float64)
    sy2_contrib = np.zeros((num_layers, num_heads), dtype=np.float64)
    syo_contrib = np.zeros((num_layers, num_heads), dtype=np.float64)  # sum(y*o)


    batch_start_idx = 0  # updated in outer loop

    def make_attn_hook(layer_idx: int):
        def hook(attn_module, inputs, output):
            nonlocal batch_start_idx
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return

            x_bt = _maybe_bt(x)  # [B,T,E]
            if x_bt.dim() != 3:
                return

            B, T, E = x_bt.shape
            H = int(attn_module.num_heads)
            if H <= 0 or (E % H != 0) or (T <= 1):
                return
            D = E // H

            q_proj, k_proj, v_proj, _Wout = _attn_get_qkv_outproj(attn_module)

            q = q_proj(x_bt).view(B, T, H, D).permute(0, 2, 1, 3)  # [B,H,T,D]
            k = k_proj(x_bt).view(B, T, H, D).permute(0, 2, 1, 3)  # [B,H,T,D]
            v = v_proj(x_bt).view(B, T, H, D).permute(0, 2, 1, 3)  # [B,H,T,D]

            attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(D))  # [B,H,T,T]
            attn_probs = torch.softmax(attn_logits, dim=-1)  # [B,H,T,T]

            # Patch-only keys (exclude CLS key at index 0)
            attn_patch = attn_probs[:, :, :, 1:]  # [B,H,T,P]
            attn_patch = attn_patch / (attn_patch.sum(dim=-1, keepdim=True) + 1e-12)

            # Define reg keys via high patch token norm in x-space (exclude CLS token)
            patch_norm = torch.linalg.vector_norm(x_bt[:, 1:, :], dim=-1)  # [B,P]
            reg_mask = patch_norm > float(reg_token_norm_threshold)
            if not reg_mask.any():
                topk = min(int(reg_token_fallback_topk), int(patch_norm.shape[1]))
                idx = torch.topk(patch_norm, k=topk, dim=1).indices
                reg_mask = torch.zeros_like(patch_norm, dtype=torch.bool)
                reg_mask.scatter_(1, idx, True)  # [B,P]

            # Effective contribution weights ~ a_k * ||V_k|| (scalar per key, avoids giant tensors)
            v_patch_norm = torch.linalg.vector_norm(v[:, :, 1:, :], dim=-1)  # [B,H,P]
            contrib_raw = attn_patch * v_patch_norm.unsqueeze(2)             # [B,H,T,P]
            contrib = contrib_raw / (contrib_raw.sum(dim=-1, keepdim=True) + 1e-12)

            # CLS query only
            a_cls = attn_patch[:, :, 0, :]     # [B,H,P]
            c_cls = contrib[:, :, 0, :]        # [B,H,P]

            ent_a = _safe_entropy(a_cls)       # [B,H]
            ent_c = _safe_entropy(c_cls)       # [B,H]
            kl_ac = _safe_kl(a_cls, c_cls)     # [B,H]

            rm = reg_mask.unsqueeze(1)         # [B,1,P]
            reg_attn = (a_cls * rm).sum(dim=-1)    # [B,H]
            reg_contrib = (c_cls * rm).sum(dim=-1) # [B,H]

            # accumulate sums
            sums["entropy_attn_patch_cls"][layer_idx] += ent_a.sum(dim=0).detach().cpu().numpy()
            sums["entropy_contrib_patch_cls"][layer_idx] += ent_c.sum(dim=0).detach().cpu().numpy()
            sums["kl_attn_to_contrib_cls"][layer_idx] += kl_ac.sum(dim=0).detach().cpu().numpy()
            sums["reg_attn_mass_cls"][layer_idx] += reg_attn.sum(dim=0).detach().cpu().numpy()
            sums["reg_contrib_mass_cls"][layer_idx] += reg_contrib.sum(dim=0).detach().cpu().numpy()
            den_cls[layer_idx] += float(B)

            # corr(reg_attn, reg_contrib) over images
            xh = reg_attn.detach().cpu().numpy()       # [B,H]
            yh = reg_contrib.detach().cpu().numpy()    # [B,H]
            n_corr[layer_idx] += float(B)
            sx[layer_idx] += xh.sum(axis=0)
            sy[layer_idx] += yh.sum(axis=0)
            sxx[layer_idx] += (xh * xh).sum(axis=0)
            syy[layer_idx] += (yh * yh).sum(axis=0)
            sxy[layer_idx] += (xh * yh).sum(axis=0)

            # corr vs outcome
            if use_outcome:
                outs = []
                for bi in range(B):
                    name = image_basenames[batch_start_idx + bi]
                    outs.append(outcomes_map.get(name, float("nan")))
                o = np.asarray(outs, dtype=np.float64)  # [B]
                ok = np.isfinite(o)
                if ok.any():
                    o2 = o[ok]              # [n_ok]
                    x_ok = xh[ok, :]        # [n_ok, H]
                    y_ok = yh[ok, :]        # [n_ok, H]

                    n_ok = float(o2.shape[0])
                    n_out[layer_idx] += n_ok
                    so[layer_idx]  += o2.sum()
                    so2[layer_idx] += (o2 * o2).sum()

                    sx_attn[layer_idx]  += x_ok.sum(axis=0)
                    sx2_attn[layer_idx] += (x_ok * x_ok).sum(axis=0)
                    sxo_attn[layer_idx] += (x_ok * o2[:, None]).sum(axis=0)

                    sy_contrib[layer_idx]  += y_ok.sum(axis=0)
                    sy2_contrib[layer_idx] += (y_ok * y_ok).sum(axis=0)
                    syo_contrib[layer_idx] += (y_ok * o2[:, None]).sum(axis=0)
        return hook

    # register hooks on every block.attn
    hooks: List[Any] = []
    for li, blk in enumerate(blocks):
        hooks.append(blk.attn.register_forward_hook(make_attn_hook(li)))

    # We need this additional accumulator for contrib/outcome cross term
    #syo_contrib = np.zeros((num_layers, num_heads), dtype=np.float64)
    try:
        N = int(images_cpu.shape[0])
        bs = max(1, int(batch_size))
        for i in range(0, N, bs):
            batch_start_idx = i
            x = images_cpu[i:i + bs].to(device)
            _ = model.encode_image(x)

            # update contrib/outcome cross term if needed (computed batchwise here)
            if use_outcome:
                B = int(x.shape[0])
                outs = []
                for bi in range(B):
                    name = image_basenames[i + bi]
                    outs.append(outcomes_map.get(name, float("nan")))
                o = np.asarray(outs, dtype=np.float64)
                ok = np.isfinite(o)
                
                if ok.any():
                    # We can’t easily recover per-layer/head reg_contrib in this outer scope without
                    # redoing work, so we compute corr(contrib,outcome) inside the hook.
                    # To keep the code clean, we’ll compute those corr values from hook-local accumulators.
                    pass
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    # finalize means
    out: Dict[str, Any] = {}
    for k, v in sums.items():
        mean = v.copy()
        for li in range(num_layers):
            mean[li] /= (den_cls[li] + 1e-12)
        out[k] = mean.astype(np.float32)

    out["corr_reg_attn_vs_contrib"] = _corr_from_sums(n_corr, sx, sy, sxx, syy, sxy).astype(np.float32)

    if use_outcome and np.any(n_out > 0):
        denom_o = np.sqrt(np.maximum(0.0, n_out * so2 - so * so)) + 1e-12

        num_attn = (n_out * sxo_attn - sx_attn * so)
        den_attn = np.sqrt(np.maximum(0.0, n_out * sx2_attn - sx_attn * sx_attn)) * denom_o + 1e-12
        out["corr_reg_attn_vs_outcome"] = (num_attn / den_attn).astype(np.float32)

        num_contrib = (n_out * syo_contrib - sy_contrib * so)
        den_contrib = np.sqrt(np.maximum(0.0, n_out * sy2_contrib - sy_contrib * sy_contrib)) * denom_o + 1e-12
        out["corr_reg_contrib_vs_outcome"] = (num_contrib / den_contrib).astype(np.float32)

    out["den_cls"] = den_cls.astype(np.float64)
    return out


# True per-head pre/post out_proj L2 measurement
def _attn_get_qkv_outproj(attn_mod):
    """
    Returns callables (q_proj, k_proj, v_proj) and out_proj weight tensor.
    Supports attention with q_proj/k_proj/v_proj/out_proj.
    """
    if hasattr(attn_mod, "q_proj") and hasattr(attn_mod, "k_proj") and hasattr(attn_mod, "v_proj") and hasattr(attn_mod, "out_proj"):
        return attn_mod.q_proj, attn_mod.k_proj, attn_mod.v_proj, attn_mod.out_proj.weight
    raise RuntimeError(
        "Attention module does not expose q_proj/k_proj/v_proj/out_proj.weight. "
        "If your attn implementation differs, we’ll adapt this accessor."
    )


@torch.no_grad()
def compute_per_head_prepost_outproj_l2(
    model: torch.nn.Module,
    images_cpu: torch.Tensor,
    device: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Computes TRUE per-head norms for the ViT vision transformer attention outputs:

      PRE (head subspace):    o_pre[h]  = (A_h @ V_h)             in R^{head_dim}
      POST (embed space):     o_post[h] = o_pre[h] @ W_h^T        in R^{d_model}
        where W_h are the per-head input slices of out_proj.weight.

    Reports two aggregation modes per layer/head:
      - *_all: mean over (batch, all query tokens)
      - *_cls: mean over (batch, CLS query token only)

    Returns dict of 4 arrays, each shaped [num_layers, num_heads]:
      {"pre_all", "post_all", "pre_cls", "post_cls"}
    """
    vision_transformer = model.visual.transformer
    blocks = vision_transformer.resblocks
    num_layers = len(blocks)
    num_heads = int(blocks[0].attn.num_heads)

    # Accumulators: sum over tokens/samples, plus denominators for true weighted means
    # Use float64 for stability.
    sums = {
        "pre_all":  np.zeros((num_layers, num_heads), dtype=np.float64),
        "post_all": np.zeros((num_layers, num_heads), dtype=np.float64),
        "pre_cls":  np.zeros((num_layers, num_heads), dtype=np.float64),
        "post_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
    }
    den_all = np.zeros((num_layers,), dtype=np.float64)  # counts B*T
    den_cls = np.zeros((num_layers,), dtype=np.float64)  # counts B

    def make_attn_hook(layer_idx: int):
        def hook(attn_module, inputs, output):
            # inputs[0] is typically x (maybe [T,B,C] or [B,T,C])
            if not inputs:
                return

            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return

            x_bt = _maybe_bt(x)  # -> [B,T,C]
            if x_bt.dim() != 3:
                return

            B, T, E = x_bt.shape
            H = int(attn_module.num_heads)
            if H <= 0:
                return
            if E % H != 0:
                # cannot define head_dim cleanly
                return
            D = E // H

            q_proj, k_proj, v_proj, W_o = _attn_get_qkv_outproj(attn_module)

            # Compute Q,K,V in float32 for numerical stability
            xb = x_bt.to(device=device, dtype=torch.float32)

            q = q_proj(xb)  # [B,T,E]
            k = k_proj(xb)
            v = v_proj(xb)

            # Reshape to [B,H,T,D]
            q = q.view(B, T, H, D).permute(0, 2, 1, 3).contiguous()
            k = k.view(B, T, H, D).permute(0, 2, 1, 3).contiguous()
            v = v.view(B, T, H, D).permute(0, 2, 1, 3).contiguous()

            # Attention probs A: [B,H,T,T]
            scale = float(1.0 / np.sqrt(D))
            attn_scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
            attn_probs = torch.softmax(attn_scores, dim=-1)

            # PRE head outputs: out_pre = A @ V -> [B,H,T,D]
            out_pre = torch.einsum("bhts,bhsd->bhtd", attn_probs, v)

            # PRE norms
            # all queries: sum over (B,T), keep H
            pre_all_sum = out_pre.norm(dim=-1).sum(dim=(0, 2))  # [H]
            # CLS query only (q=0): sum over B
            pre_cls_sum = out_pre[:, :, 0, :].norm(dim=-1).sum(dim=0)  # [H]

            # POST: apply out_proj slices W_h
            # W_o: [E,E], take per-head input slices (columns) -> [E,D] per head, transpose to [D,E]
            W_o = W_o.detach().to(device=device, dtype=torch.float32)  # [E,E]
            W_h_t = torch.stack(
                [W_o[:, h * D:(h + 1) * D].t() for h in range(H)],
                dim=0
            )  # [H, D, E]

            out_post = torch.einsum("bhtd,hde->bhte", out_pre, W_h_t)  # [B,H,T,E]

            post_all_sum = out_post.norm(dim=-1).sum(dim=(0, 2))  # [H]
            post_cls_sum = out_post[:, :, 0, :].norm(dim=-1).sum(dim=0)  # [H]

            # move to cpu numpy + accumulate with correct weighting
            sums["pre_all"][layer_idx]  += pre_all_sum.detach().cpu().numpy().astype(np.float64)
            sums["post_all"][layer_idx] += post_all_sum.detach().cpu().numpy().astype(np.float64)
            sums["pre_cls"][layer_idx]  += pre_cls_sum.detach().cpu().numpy().astype(np.float64)
            sums["post_cls"][layer_idx] += post_cls_sum.detach().cpu().numpy().astype(np.float64)

            den_all[layer_idx] += float(B * T)
            den_cls[layer_idx] += float(B)

        return hook

    # Register hooks on the *attention module* of each vision block
    hooks = []
    for l, block in enumerate(blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(l)))

    try:
        model.eval()
        N = int(images_cpu.shape[0])
        bs = max(1, int(batch_size))
        for i in range(0, N, bs):
            images = images_cpu[i:i + bs].to(device)
            _ = model.encode_image(images)
    finally:
        for h in hooks:
            h.remove()

    # Convert sums -> means
    out: Dict[str, np.ndarray] = {}
    for k in ["pre_all", "post_all"]:
        denom = den_all.reshape(-1, 1) + 1e-12
        out[k] = (sums[k] / denom).astype(np.float32)
    for k in ["pre_cls", "post_cls"]:
        denom = den_cls.reshape(-1, 1) + 1e-12
        out[k] = (sums[k] / denom).astype(np.float32)

    return out



@torch.no_grad()
def discover_regs_for_layer(
    model: torch.nn.Module,
    layer: int,
    device: str,
    reg_images_cpu: torch.Tensor,
    reg_batch_size: int,
    reg_abs_threshold: float,
    reg_topn: int,
    reg_quantile: float,
    reg_fallback_topn: int,
    reg_token_norm_threshold: float,
    reg_token_fallback_topk: int,
) -> Tuple[set, np.ndarray]:
    model.eval()
    blocks = model.visual.transformer.resblocks
    blk = blocks[layer]

    Wc = blk.mlp.c_proj.weight.detach().to(device)  # [1024,4096]
    col_norm = torch.linalg.vector_norm(Wc, dim=0)  # [4096]

    captured = {}

    def hook_c_fc(module, inp, out):
        captured["preact"] = out

    def hook_blk_out(module, inp, out):
        captured["blk_out"] = out

    h1 = blk.mlp.c_fc.register_forward_hook(hook_c_fc)
    h2 = blk.register_forward_hook(hook_blk_out)

    try:
        N = int(reg_images_cpu.shape[0])
        max_imp = torch.zeros((Wc.shape[1],), device=device)

        max_token_norm_seen = 0.0

        bs = max(1, int(reg_batch_size))
        for i in range(0, N, bs):
            x_cpu = reg_images_cpu[i:i + bs]
            x = x_cpu.to(device)

            captured.clear()
            _ = model.encode_image(x)

            if "preact" not in captured or "blk_out" not in captured:
                raise RuntimeError("Failed to capture c_fc or block output. Check attnclip module wiring.")

            preact = _maybe_bt(captured["preact"])    # [B,T,4096]
            blk_out = _maybe_bt(captured["blk_out"])  # [B,T,1024]
            h = _apply_mlp_activation(blk, preact)    # [B,T,4096]

            h_sp = h[:, 1:, :]           # [B,T-1,4096]
            x_sp = blk_out[:, 1:, :]     # [B,T-1,1024]

            token_norm = torch.linalg.vector_norm(x_sp, dim=-1)  # [B,T-1]
            max_token_norm_seen = max(max_token_norm_seen, float(token_norm.max().item()))

            mask = token_norm > float(reg_token_norm_threshold)  # [B,T-1]
            if mask.sum().item() == 0:
                k = max(1, int(reg_token_fallback_topk))
                flat = token_norm.reshape(-1)
                topk_idx = torch.topk(flat, k=min(k, flat.numel()), largest=True).indices
                mask_flat = torch.zeros_like(flat, dtype=torch.bool)
                mask_flat[topk_idx] = True
                mask = mask_flat.view_as(token_norm)

            h_flat = h_sp.reshape(-1, h_sp.shape[-1])             # [BT,4096]
            m_flat = mask.reshape(-1)                             # [BT]
            sel = h_flat[m_flat]                                  # [S,4096]
            if sel.numel() == 0:
                continue

            imp = (sel.abs() * col_norm.view(1, -1)).amax(dim=0)  # [4096]
            max_imp = torch.maximum(max_imp, imp)

        impact = max_imp.detach().float().cpu().numpy()

        if reg_topn and reg_topn > 0:
            idx = np.argsort(-impact)[: int(reg_topn)]
            regs = set(int(i) for i in idx.tolist())
        elif reg_quantile and reg_quantile > 0.0:
            thr = float(np.quantile(impact, float(reg_quantile)))
            regs = set(int(i) for i in np.where(impact >= thr)[0].tolist())
        else:
            regs = set(int(i) for i in np.where(impact >= float(reg_abs_threshold))[0].tolist())
            if len(regs) == 0 and reg_fallback_topn and reg_fallback_topn > 0:
                idx = np.argsort(-impact)[: int(reg_fallback_topn)]
                regs = set(int(i) for i in idx.tolist())

        p99 = float(np.quantile(impact, 0.99))
        p999 = float(np.quantile(impact, 0.999))
        mx = float(np.max(impact))
        print(f"    [regs:diag] max_token_norm_seen={max_token_norm_seen:.3f} impact_max={mx:.3f} p99={p99:.3f} p999={p999:.3f}")

        return regs, impact
    finally:
        h1.remove()
        h2.remove()


def print_weight_sanity_all(models: Dict[str, torch.nn.Module], layer: int):
    tags = list(models.keys())

    def get_Wc_Wk(m: torch.nn.Module):
        blocks = m.visual.transformer.resblocks
        blk = blocks[layer]
        nxt = blocks[layer + 1]
        return blk.mlp.c_proj.weight, nxt.attn.k_proj.weight

    print(f"[sanity] weight fingerprints for L{layer}->L{layer+1}")
    for t in tags:
        Wc, Wk = get_Wc_Wk(models[t])
        print(f"  {t}:")
        print(f"    Wc sig: {tensor_sig(Wc)}")
        print(f"    Wk sig: {tensor_sig(Wk)}")

    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            Wc_a, Wk_a = get_Wc_Wk(models[a])
            Wc_b, Wk_b = get_Wc_Wk(models[b])
            dc = weight_diff_stats(Wc_a, Wc_b)
            dk = weight_diff_stats(Wk_a, Wk_b)
            print(f"  diff {a} vs {b}:")
            print(f"    Wc rel_fro_diff={dc['rel_fro_diff']:.6e} cos={dc['cos']:.6f}")
            print(f"    Wk rel_fro_diff={dk['rel_fro_diff']:.6e} cos={dk['cos']:.6f}")


def regs_change_summary(pre_regs: set, other_regs: set) -> Dict[str, List[int]]:
    common = sorted(list(pre_regs & other_regs))
    dropped = sorted(list(pre_regs - other_regs))
    new = sorted(list(other_regs - pre_regs))
    return {"common": common, "dropped": dropped, "new": new}

def save_LH_metric(
    local_path: str,
    append_to_filename: str,
    mat_LH: np.ndarray,
    ylabel: str,
    title: str,
    dpi: int = 200,
):
    ensure_dir(local_path)
    num_layers, num_heads = mat_LH.shape

    # CSV
    csv_path = f"{local_path}/{append_to_filename}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "value"])
        for l in range(num_layers):
            for h in range(num_heads):
                w.writerow([l, h, float(mat_LH[l, h])])

    # NPZ
    npz_path = f"{local_path}/{append_to_filename}.npz"
    cumsum_per_head = mat_LH.cumsum(axis=0)
    final_cumsum = cumsum_per_head[-1]
    np.savez_compressed(
        npz_path,
        mat_LH=mat_LH,
        cumsum_per_head=cumsum_per_head,
        final_cumsum=final_cumsum,
    )

    # Heatmap
    plt.figure(figsize=(18, 6))
    plt.imshow(mat_LH.T, aspect="auto", cmap="viridis")
    plt.colorbar(label=ylabel)
    plt.xlabel("Block (Layer)")
    plt.ylabel("Head")
    plt.title(title)
    plt.xticks(np.arange(num_layers))
    plt.yticks(np.arange(num_heads))
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "heatmap"), dpi=dpi)
    plt.close()

    # Per-head line plot
    plt.figure(figsize=(18, 5))
    for h in range(num_heads):
        plt.plot(range(num_layers), mat_LH[:, h], label=f"Head {h}")
    plt.xlabel("Block (Layer)")
    plt.ylabel(ylabel)
    plt.title(title + " (per head)")
    plt.legend(loc="upper right", ncol=4)
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "lineplot"), dpi=dpi)
    plt.close()

    # Cumsum bar
    plt.figure(figsize=(12, 5))
    plt.bar(range(num_heads), mat_LH.cumsum(axis=0)[-1])
    plt.xlabel("Head")
    plt.ylabel(f"Cumsum({ylabel}) over layers")
    plt.title(title + " (cumsum over layers)")
    plt.xticks(range(num_heads))
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "cumsum"), dpi=dpi)
    plt.close()



# ============================
# attention diagnostics
# ============================

@torch.no_grad()
def compute_true_attn_diagnostics(
    model: torch.nn.Module,
    images_cpu: torch.Tensor,
    device: str,
    batch_size: int,
    reg_token_norm_threshold: float,
    reg_token_fallback_topk: int,
) -> Dict[str, np.ndarray]:
    """
    Returns dict of [L,H] float32 matrices:
      - entropy_all, entropy_cls
      - regmass_all, regmass_cls   (mass to "high-norm patch tokens" as reg proxy; excludes CLS as key)
      - qnorm_all, knorm_all, vnorm_all
      - qnorm_cls, knorm_cls, vnorm_cls

    regmass_* definition:
      For each sample, define reg-key tokens as patch tokens (idx 1..T-1) with ||x||_2 > threshold.
      If none, fallback to top-K patch tokens by ||x||_2.
      regmass_cls = mean_B sum_{k in reg_keys} A[CLS_query, k]
      regmass_all = mean_{B,query_tokens} sum_{k in reg_keys} A[query, k]
    """
    vision_transformer = model.visual.transformer
    blocks = vision_transformer.resblocks
    num_layers = len(blocks)
    num_heads = int(blocks[0].attn.num_heads)

    sums = {
        "entropy_all": np.zeros((num_layers, num_heads), dtype=np.float64),
        "entropy_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
        "regmass_all": np.zeros((num_layers, num_heads), dtype=np.float64),
        "regmass_cls": np.zeros((num_layers, num_heads), dtype=np.float64),
        "qnorm_all":   np.zeros((num_layers, num_heads), dtype=np.float64),
        "knorm_all":   np.zeros((num_layers, num_heads), dtype=np.float64),
        "vnorm_all":   np.zeros((num_layers, num_heads), dtype=np.float64),
        "qnorm_cls":   np.zeros((num_layers, num_heads), dtype=np.float64),
        "knorm_cls":   np.zeros((num_layers, num_heads), dtype=np.float64),
        "vnorm_cls":   np.zeros((num_layers, num_heads), dtype=np.float64),
    }
    den_all = np.zeros((num_layers,), dtype=np.float64)  # counts B*Tq
    den_cls = np.zeros((num_layers,), dtype=np.float64)  # counts B

    def make_hook(layer_idx: int):
        def hook(attn_module, inputs, output):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return

            x_bt = _maybe_bt(x)  # [B,T,E]
            if x_bt.dim() != 3:
                return
            B, T, E = x_bt.shape

            H = int(attn_module.num_heads)
            if H <= 0 or (E % H) != 0:
                return
            D = E // H

            q_proj, k_proj, v_proj, _W_o = _attn_get_qkv_outproj(attn_module)

            xb = x_bt.to(device=device, dtype=torch.float32)  # [B,T,E]

            # reg-key mask based on ||xb|| over embed dim; exclude CLS key
            token_norm = torch.linalg.vector_norm(xb, dim=-1)  # [B,T]
            # only patch tokens 1..T-1 eligible
            patch_norm = token_norm[:, 1:]  # [B,T-1]
            mask_patch = patch_norm > float(reg_token_norm_threshold)  # [B,T-1]

            if mask_patch.sum().item() == 0:
                k = max(1, int(reg_token_fallback_topk))
                # fallback per-sample
                mask_patch = torch.zeros_like(mask_patch)
                for b in range(B):
                    flat = patch_norm[b]
                    topk_idx = torch.topk(flat, k=min(k, flat.numel()), largest=True).indices
                    mask_patch[b, topk_idx] = True

            # expand to full [B,T] with CLS=False
            mask_keys = torch.zeros((B, T), device=xb.device, dtype=torch.bool)
            mask_keys[:, 1:] = mask_patch

            # Q,K,V
            q = q_proj(xb).view(B, T, H, D).permute(0, 2, 1, 3).contiguous()  # [B,H,T,D]
            k = k_proj(xb).view(B, T, H, D).permute(0, 2, 1, 3).contiguous()
            v = v_proj(xb).view(B, T, H, D).permute(0, 2, 1, 3).contiguous()

            # norms
            qn = q.norm(dim=-1)  # [B,H,T]
            kn = k.norm(dim=-1)
            vn = v.norm(dim=-1)

            q_all_sum = qn.sum(dim=(0, 2))  # [H]
            k_all_sum = kn.sum(dim=(0, 2))
            v_all_sum = vn.sum(dim=(0, 2))
            q_cls_sum = qn[:, :, 0].sum(dim=0)  # [H]
            k_cls_sum = kn[:, :, 0].sum(dim=0)
            v_cls_sum = vn[:, :, 0].sum(dim=0)

            # attention probs
            scale = float(1.0 / np.sqrt(D))
            attn_scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale  # [B,H,T,T]
            attn_probs = torch.softmax(attn_scores, dim=-1)              # [B,H,T,T]

            # entropy along keys
            p = torch.clamp(attn_probs, min=1e-9)
            ent = -(p * torch.log(p)).sum(dim=-1)  # [B,H,T]
            ent_all_sum = ent.sum(dim=(0, 2))      # [H]
            ent_cls_sum = ent[:, :, 0].sum(dim=0)  # [H]

            # regmass (sum probs to masked keys)
            # cls query: [B,H,Tk] -> sum over Tk where mask_keys True
            m = mask_keys.unsqueeze(1).expand(B, H, T)  # [B,H,Tk] for masking keys
            cls_probs = attn_probs[:, :, 0, :]          # [B,H,Tk]
            regmass_cls = (cls_probs * m).sum(dim=-1)   # [B,H]
            regmass_cls_sum = regmass_cls.sum(dim=0)    # [H]

            # all queries: [B,H,Tq,Tk] -> sum masked keys, then sum over queries
            all_regmass = (attn_probs * m.unsqueeze(2)).sum(dim=-1)  # [B,H,Tq]
            regmass_all_sum = all_regmass.sum(dim=(0, 2))            # [H]

            # accumulate
            sums["entropy_all"][layer_idx] += ent_all_sum.detach().cpu().numpy().astype(np.float64)
            sums["entropy_cls"][layer_idx] += ent_cls_sum.detach().cpu().numpy().astype(np.float64)
            sums["regmass_all"][layer_idx] += regmass_all_sum.detach().cpu().numpy().astype(np.float64)
            sums["regmass_cls"][layer_idx] += regmass_cls_sum.detach().cpu().numpy().astype(np.float64)

            sums["qnorm_all"][layer_idx] += q_all_sum.detach().cpu().numpy().astype(np.float64)
            sums["knorm_all"][layer_idx] += k_all_sum.detach().cpu().numpy().astype(np.float64)
            sums["vnorm_all"][layer_idx] += v_all_sum.detach().cpu().numpy().astype(np.float64)

            sums["qnorm_cls"][layer_idx] += q_cls_sum.detach().cpu().numpy().astype(np.float64)
            sums["knorm_cls"][layer_idx] += k_cls_sum.detach().cpu().numpy().astype(np.float64)
            sums["vnorm_cls"][layer_idx] += v_cls_sum.detach().cpu().numpy().astype(np.float64)

            den_all[layer_idx] += float(B * T)  # (B,Tq)
            den_cls[layer_idx] += float(B)

        return hook

    hooks = []
    for l, block in enumerate(blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(l)))

    try:
        model.eval()
        N = int(images_cpu.shape[0])
        bs = max(1, int(batch_size))
        for i in range(0, N, bs):
            images = images_cpu[i:i + bs].to(device)
            _ = model.encode_image(images)
    finally:
        for h in hooks:
            h.remove()

    out: Dict[str, np.ndarray] = {}
    denom_all = den_all.reshape(-1, 1) + 1e-12
    denom_cls = den_cls.reshape(-1, 1) + 1e-12

    # entropy/regmass are already “per query token” sums; divide by den_all/den_cls to make means
    for k in ["entropy_all", "regmass_all", "qnorm_all", "knorm_all", "vnorm_all"]:
        out[k] = (sums[k] / denom_all).astype(np.float32)
    for k in ["entropy_cls", "regmass_cls", "qnorm_cls", "knorm_cls", "vnorm_cls"]:
        out[k] = (sums[k] / denom_cls).astype(np.float32)

    return out


def _write_regs_discovery_file(
    out_path: str,
    layer: int,
    next_layer: int,
    args,
    regs: set,
    impact: np.ndarray,
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"layer={layer} next_layer={next_layer}\n")
        f.write(f"token_norm_thr={args.reg_token_norm_threshold} token_fallback_topk={args.reg_token_fallback_topk}\n")
        f.write(f"abs_thr={args.reg_abs_threshold} topn={args.reg_topn} quantile={args.reg_quantile} fallback_topn={args.reg_fallback_topn}\n")
        f.write(f"regs_n={len(regs)}\n")
        f.write("regs_sorted=" + str(sorted(regs)) + "\n")
        top50 = np.argsort(-impact)[:50]
        f.write("top50_by_impact=" + ", ".join([f"{int(i)}:{float(impact[i]):.6f}" for i in top50]) + "\n")


@torch.no_grad()
def compute_per_head_output_l2(model: torch.nn.Module, images_cpu: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    vision_transformer = model.visual.transformer
    num_layers = len(vision_transformer.resblocks)
    num_heads = vision_transformer.resblocks[0].attn.num_heads

    per_head_data = {l: [] for l in range(num_layers)}

    def make_hook(layer_idx):
        def hook(module, input, output):
            out = _maybe_bt(output)
            if out.dim() != 3:
                return
            batch_size_, seq_len, embed_dim = out.shape
            nheads = module.attn.num_heads
            head_dim = embed_dim // nheads
            x = out.view(batch_size_, seq_len, nheads, head_dim)
            head_norms = x.norm(dim=-1).mean(dim=(0, 1)).detach().cpu().numpy()
            per_head_data[layer_idx].append(head_norms)
        return hook

    hooks = []
    for l, block in enumerate(vision_transformer.resblocks):
        hooks.append(block.register_forward_hook(make_hook(l)))

    try:
        N = int(images_cpu.shape[0])
        bs = max(1, int(batch_size))
        for i in range(0, N, bs):
            images = images_cpu[i:i + bs].to(device)
            _ = model.encode_image(images)
    finally:
        for h in hooks:
            h.remove()

    per_head_mean = np.zeros((num_layers, num_heads), dtype=np.float64)
    for l in range(num_layers):
        arrs = per_head_data[l]
        if not arrs:
            continue
        arrs = np.stack(arrs, axis=0)
        per_head_mean[l] = arrs.mean(axis=0)

    return per_head_mean.astype(np.float32)


def save_l2_outputs(local_path: str, append_to_filename: str, per_head_mean: np.ndarray, debug: bool = False):
    ensure_dir(local_path)
    num_layers, num_heads = per_head_mean.shape

    csv_path = f"{local_path}/per_head_norms-{append_to_filename}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "mean_l2"])
        for l in range(num_layers):
            for h in range(num_heads):
                writer.writerow([l, h, float(per_head_mean[l, h])])

    npz_path = f"{local_path}/per_head_l2norms-{append_to_filename}.npz"
    cumsum_per_head = per_head_mean.cumsum(axis=0)
    final_cumsum = cumsum_per_head[-1]
    np.savez_compressed(
        npz_path,
        per_head_mean=per_head_mean,
        cumsum_per_head=cumsum_per_head,
        final_cumsum=final_cumsum,
    )

    # plots
    plt.figure(figsize=(18, 6))
    plt.imshow(per_head_mean.T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Mean L2 norm")
    plt.xlabel("Block (Layer)")
    plt.ylabel("Head")
    plt.title("Vision Transformer: Per-head mean L2 norm (output tokens)")
    plt.xticks(np.arange(num_layers))
    plt.yticks(np.arange(num_heads))
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "per_head_l2norm_heatmap"))
    plt.close()

    plt.figure(figsize=(18, 5))
    for h in range(num_heads):
        plt.plot(range(num_layers), per_head_mean[:, h], label=f"Head {h}")
    plt.xlabel("Block (Layer)")
    plt.ylabel("Mean L2 norm")
    plt.title("Vision Transformer: Per-head mean L2 norm per block")
    plt.legend(loc="upper right", ncol=4)
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "per_head_l2norm_lineplot"))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(range(num_heads), per_head_mean.cumsum(axis=0)[-1])
    plt.xlabel("Head")
    plt.ylabel("Cumulative sum of mean L2 norm")
    plt.title("Vision Transformer: Cumulative sum per head (all blocks)")
    plt.xticks(range(num_heads))
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "per_head_l2norm_cumsum"))
    plt.close()

    _log(f"[l2:{append_to_filename}] wrote CSV/NPZ/PNGs in {local_path}", debug, always=False)



def _delta_summary_lines(
    name: str,
    before: np.ndarray,
    after: np.ndarray,
    focus_layer: int,
    topk: int,
) -> List[str]:
    """
    before/after: [L,H]
    Prints:
      - max|Δ|
      - focus-layer head table (topk by |Δ|)
      - cumsum-by-head topk
    """
    assert before.shape == after.shape
    L, H = before.shape
    delta = after - before

    absmax = float(np.max(np.abs(delta)))
    where = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    lines = []
    lines.append(f"[Δ:{name}] max|Δ|={absmax:.6f} at layer={where[0]} head={where[1]} Δ={float(delta[where]):+.6f}")

    fl = int(focus_layer)
    if 0 <= fl < L:
        d = delta[fl]
        order = np.argsort(-np.abs(d))[: max(1, int(topk))]
        lines.append(f"[Δ:{name}] focus_layer={fl} top{len(order)} heads by |Δ|:")
        lines.append("  head |   before    after     delta")
        for h in order.tolist():
            lines.append(f"  {h:>4d} | {float(before[fl,h]):8.4f} {float(after[fl,h]):8.4f} {float(delta[fl,h]):+9.4f}")

    cumsum_b = before.cumsum(axis=0)[-1]
    cumsum_a = after.cumsum(axis=0)[-1]
    cumsum_d = cumsum_a - cumsum_b
    order2 = np.argsort(-np.abs(cumsum_d))[: max(1, int(topk))]
    lines.append(f"[Δ:{name}] cumsum(top{len(order2)} heads by |Δ|):")
    lines.append("  head | cumsum_before cumsum_after  cumsum_delta")
    for h in order2.tolist():
        lines.append(f"  {h:>4d} | {float(cumsum_b[h]):12.4f} {float(cumsum_a[h]):11.4f} {float(cumsum_d[h]):+12.4f}")

    return lines


@torch.no_grad()
def run_l2_suite_for_models(
    out_root: str,
    models: Dict[str, torch.nn.Module],
    model_tags: List[str],
    reg_images_cpu: torch.Tensor,
    device: str,
    batch_size: int,
    debug: bool,
    args,
    reg_basenames: Optional[List[str]] = None,
    outcomes_map: Optional[Dict[str, float]] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
]:
    """
    Returns:
      per_head_mean_by_model: tag -> [L,H]   (heuristic block-output slice L2)
      per_head_true_by_model: tag -> {kind -> [L,H]} (true pre/post outproj)
      per_head_diag_by_model: tag -> {diag_kind -> [L,H]} (entropy/regmass/qkv norms)
    """
    per_head_mean_by_model: Dict[str, np.ndarray] = {}
    per_head_true_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    per_head_diag_by_model: Dict[str, Dict[str, np.ndarray]] = {}

    for tag in model_tags:
        local_l2 = ensure_dir(os.path.join(out_root, tag, "l2_output_norms"))
        local_true = ensure_dir(os.path.join(local_l2, "outproj_prepost"))
        local_diag = ensure_dir(os.path.join(local_l2, "attn_diagnostics"))

        # heuristic
        ph = compute_per_head_output_l2(
            model=models[tag],
            images_cpu=reg_images_cpu,
            device=device,
            batch_size=batch_size,
        )
        per_head_mean_by_model[tag] = ph
        save_l2_outputs(local_path=local_l2, append_to_filename=tag, per_head_mean=ph, debug=debug)

        # true pre/post
        td = compute_per_head_prepost_outproj_l2(
            model=models[tag],
            images_cpu=reg_images_cpu,
            device=device,
            batch_size=batch_size,
        )
        per_head_true_by_model[tag] = td
        for kind, mat in td.items():
            save_l2_outputs_kind(local_path=local_true, model_tag=tag, kind=kind, per_head_mean=mat)

        
        
        # attention diagnostics (entropy/regmass/qkv norms)
        if getattr(args, "attn_diag", True):
            diag = compute_true_attn_diagnostics(
                model=models[tag],
                images_cpu=reg_images_cpu,
                device=device,
                batch_size=batch_size,
                reg_token_norm_threshold=args.reg_token_norm_threshold,
                reg_token_fallback_topk=args.reg_token_fallback_topk,
            )
            per_head_diag_by_model[tag] = diag

            for dk, mat in diag.items():
                save_LH_metric(
                    local_path=local_diag,
                    append_to_filename=f"{tag}__{dk}",
                    mat_LH=mat,
                    ylabel=dk,
                    title=f"{tag}: {dk} (true-head attention diagnostics)",
                    dpi=int(getattr(args, "dpi", 200)),
                )
        else:
            per_head_diag_by_model[tag] = {}

        # effective contribution diagnostics runs independently
        if getattr(args, "attn_contrib_diag", False):
            local_contrib = ensure_dir(os.path.join(local_l2, "attn_contrib_diagnostics"))

            contrib_diag = compute_true_attn_contrib_diagnostics(
                model=models[tag],
                images_cpu=reg_images_cpu,
                device=device,
                batch_size=batch_size,
                reg_token_norm_threshold=args.reg_token_norm_threshold,
                reg_token_fallback_topk=args.reg_token_fallback_topk,
                outcomes_map=outcomes_map,
                image_basenames=reg_basenames,
            )

            # Save NPZ
            np.savez_compressed(
                os.path.join(local_contrib, f"{tag}__attn_contrib_diag.npz"),
                **contrib_diag,
            )

            # Save CSV
            csv_path = os.path.join(local_contrib, f"{tag}__attn_contrib_diag.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                wr = csv.writer(f)
                hdr = [
                    "layer", "head",
                    "entropy_attn_patch_cls", "entropy_contrib_patch_cls", "kl_attn_to_contrib_cls",
                    "reg_attn_mass_cls", "reg_contrib_mass_cls", "corr_reg_attn_vs_contrib",
                ]
                if "corr_reg_attn_vs_outcome" in contrib_diag and "corr_reg_contrib_vs_outcome" in contrib_diag:
                    hdr += ["corr_reg_attn_vs_outcome", "corr_reg_contrib_vs_outcome"]
                wr.writerow(hdr)

                L, H = contrib_diag["entropy_attn_patch_cls"].shape
                for li in range(L):
                    for hi in range(H):
                        row = [
                            li, hi,
                            float(contrib_diag["entropy_attn_patch_cls"][li, hi]),
                            float(contrib_diag["entropy_contrib_patch_cls"][li, hi]),
                            float(contrib_diag["kl_attn_to_contrib_cls"][li, hi]),
                            float(contrib_diag["reg_attn_mass_cls"][li, hi]),
                            float(contrib_diag["reg_contrib_mass_cls"][li, hi]),
                            float(contrib_diag["corr_reg_attn_vs_contrib"][li, hi]),
                        ]
                        if "corr_reg_attn_vs_outcome" in contrib_diag and "corr_reg_contrib_vs_outcome" in contrib_diag:
                            row += [
                                float(contrib_diag["corr_reg_attn_vs_outcome"][li, hi]),
                                float(contrib_diag["corr_reg_contrib_vs_outcome"][li, hi]),
                            ]
                        wr.writerow(row)

            # Plots: head-avg + specialist overlays
            special_heads = parse_int_list_arg(getattr(args, "attn_special_heads", "10,13"))
            xs = np.arange(L)

            def _head_avg(x: np.ndarray) -> np.ndarray:
                return x.mean(axis=1)

            plt.figure(figsize=(10, 5))
            plt.plot(xs, _head_avg(contrib_diag["entropy_attn_patch_cls"]), label="attn entropy (CLS->patches, head-avg)")
            plt.plot(xs, _head_avg(contrib_diag["entropy_contrib_patch_cls"]), label="contrib entropy (CLS->patches, head-avg)")
            for h in special_heads:
                if 0 <= h < H:
                    plt.plot(xs, contrib_diag["entropy_attn_patch_cls"][:, h], linestyle="--", label=f"attn H{h:02d}")
                    plt.plot(xs, contrib_diag["entropy_contrib_patch_cls"][:, h], linestyle=":", label=f"contrib H{h:02d}")
            plt.xlabel("layer")
            plt.ylabel("entropy")
            plt.title(f"{tag}: CLS patch-only entropy (attn vs effective contribution)")
            plt.grid(True, linewidth=0.3, alpha=0.4)
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(png_path(local_contrib, tag, "attn_contrib__entropy_cls_patchonly"), dpi=int(getattr(args, "dpi", 200)))
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(xs, _head_avg(contrib_diag["kl_attn_to_contrib_cls"]), label="KL(attn||contrib), head-avg")
            for h in special_heads:
                if 0 <= h < H:
                    plt.plot(xs, contrib_diag["kl_attn_to_contrib_cls"][:, h], linestyle="--", label=f"KL H{h:02d}")
            plt.xlabel("layer")
            plt.ylabel("KL")
            plt.title(f"{tag}: KL(attn||effective contribution) for CLS (patch-only)")
            plt.grid(True, linewidth=0.3, alpha=0.4)
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(png_path(local_contrib, tag, "attn_contrib__kl_cls_patchonly"), dpi=int(getattr(args, "dpi", 200)))
            plt.close()

            # Printed summary file
            kl = contrib_diag["kl_attn_to_contrib_cls"]
            gap = contrib_diag["reg_attn_mass_cls"] - contrib_diag["reg_contrib_mass_cls"]

            flat_kl = [(float(kl[li, hi]), li, hi) for li in range(L) for hi in range(H)]
            flat_kl.sort(reverse=True, key=lambda x: x[0])

            flat_gap = [(float(abs(gap[li, hi])), float(gap[li, hi]), li, hi) for li in range(L) for hi in range(H)]
            flat_gap.sort(reverse=True, key=lambda x: x[0])

            summary_lines = []
            summary_lines.append(f"=== Effective contribution summary: {tag} ===")
            summary_lines.append(f"N_images={int(reg_images_cpu.shape[0])}")
            summary_lines.append("")
            summary_lines.append("Top-15 (layer,head) by KL(attn||contrib) (CLS, patch-only):")
            for v, li, hi in flat_kl[:15]:
                summary_lines.append(f"  L{li:02d} H{hi:02d}: KL={v:.6f}")
            summary_lines.append("")
            summary_lines.append("Top-15 by |reg_attn_mass - reg_contrib_mass| (CLS, patch-only):")
            for absv, signed, li, hi in flat_gap[:15]:
                summary_lines.append(f"  L{li:02d} H{hi:02d}: gap={signed:+.6f} |gap|={absv:.6f}")
            summary_lines.append("")
            summary_lines.append("Layerwise head-avg corr(reg_attn_mass, reg_contrib_mass):")
            corr = contrib_diag["corr_reg_attn_vs_contrib"]
            for li in range(L):
                summary_lines.append(f"  L{li:02d}: corr={float(corr[li].mean()):+.4f}")

            save_summary_txt(os.path.join(local_contrib, f"{tag}__attn_contrib_diag_summary.txt"), summary_lines)

        _log(f"[l2-suite] saved {tag} into {local_l2}", debug, always=False)

    return per_head_mean_by_model, per_head_true_by_model, per_head_diag_by_model


def save_l2_deltas_vs_pretrained(local_path: str, append_to_filename: str, delta: np.ndarray):
    ensure_dir(local_path)
    num_layers, num_heads = delta.shape

    csv_path = f"{local_path}/per_head_norms_delta_vs_pretrained-{append_to_filename}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "delta_mean_l2"])
        for l in range(num_layers):
            for h in range(num_heads):
                writer.writerow([l, h, float(delta[l, h])])
    print(f"[l2Δ:{append_to_filename}] Wrote delta CSV to {csv_path}")

    absmax = float(np.max(np.abs(delta)))
    where = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    print(f"[l2Δ:{append_to_filename}] max|Δ|={absmax:.6f} at layer={where[0]} head={where[1]} Δ={float(delta[where]):.6f}")

    cumsum_per_head = delta.cumsum(axis=0)
    final_cumsum = cumsum_per_head[-1]

    plt.figure(figsize=(18, 6))
    plt.imshow(delta.T, aspect="auto", cmap="seismic")
    plt.colorbar(label="Δ Mean L2 norm (model - pretrained)")
    plt.xlabel("Block (Layer)")
    plt.ylabel("Head")
    plt.title("Per-head mean L2 norm Δ vs pretrained")
    plt.xticks(np.arange(num_layers))
    plt.yticks(np.arange(num_heads))
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "per_head_l2norm_delta_heatmap"), dpi=200)
    plt.close()

    plt.figure(figsize=(18, 5))
    for h in range(num_heads):
        plt.plot(range(num_layers), delta[:, h], label=f"Head {h}")
    plt.xlabel("Block (Layer)")
    plt.ylabel("Δ Mean L2 norm")
    plt.title("Per-head mean L2 norm Δ vs pretrained (per block)")
    plt.legend(loc="upper right", ncol=4)
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "per_head_l2norm_delta_lineplot"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(range(num_heads), delta.cumsum(axis=0)[-1])
    plt.xlabel("Head")
    plt.ylabel("Cumulative Δ mean L2 norm")
    plt.title("Cumulative Δ per head vs pretrained (all blocks)")
    plt.xticks(range(num_heads))
    plt.tight_layout()
    plt.savefig(png_path(local_path, append_to_filename, "per_head_l2norm_delta_cumsum"), dpi=200)
    plt.close()

    npz_path = f"{local_path}/per_head_l2norms_delta_vs_pretrained-{append_to_filename}.npz"
    np.savez_compressed(
        npz_path,
        delta=delta,
        cumsum_per_head=cumsum_per_head,
        final_cumsum=final_cumsum,
    )
    print(f"[l2Δ:{append_to_filename}] Saved NPZ to {npz_path}")


def save_l2_outputs_kind(local_path: str, model_tag: str, kind: str, per_head_mean: np.ndarray):
    """
    kind examples: 'pre_all', 'post_all', 'pre_cls', 'post_cls'
    """
    append = f"{model_tag}__{kind}"
    save_l2_outputs(local_path=local_path, append_to_filename=append, per_head_mean=per_head_mean)


def save_l2_deltas_kind_vs_pretrained(local_path: str, model_tag: str, kind: str, delta: np.ndarray):
    append = f"{model_tag}__{kind}"
    save_l2_deltas_vs_pretrained(local_path=local_path, append_to_filename=append, delta=delta)


def plot_compare_special(
    out_path_png: str,
    title: str,
    focus_layers: List[int],
    pre_regs_by_layer: Dict[int, set],
    ft_regs_by_layer: Dict[int, set],
    pre_impact_by_layer: Dict[int, np.ndarray],
    ft_impact_by_layer: Dict[int, np.ndarray],
    annotate_topn_per_layer: int = 10,
):
    """
      X = feature index (0..4095), Y = layer index (focused layers).
    This spreads the dense 4096 axis horizontally (more room) and keeps Y sparse.
    """
    xs_inter, ys_inter = [], []
    xs_new, ys_new = [], []
    xs_drop, ys_drop = [], []

    for L in focus_layers:
        pre = pre_regs_by_layer.get(L, set())
        ft = ft_regs_by_layer.get(L, set())
        inter = pre & ft
        new = ft - pre
        drop = pre - ft

        xs_inter += list(sorted(inter))
        ys_inter += [L] * len(inter)

        xs_new += list(sorted(new))
        ys_new += [L] * len(new)

        xs_drop += list(sorted(drop))
        ys_drop += [L] * len(drop)

    plt.figure(figsize=(18, 6))
    if xs_drop:
        plt.scatter(xs_drop, ys_drop, s=14, marker="x", label="drop_ft (pre only)")
    if xs_new:
        plt.scatter(xs_new, ys_new, s=14, marker="^", label="new_ft (ft only)")
    if xs_inter:
        plt.scatter(xs_inter, ys_inter, s=12, marker="o", label="intersection")

    for L in focus_layers:
        pre = pre_regs_by_layer.get(L, set())
        ft = ft_regs_by_layer.get(L, set())
        uni = sorted(list(pre | ft))
        if not uni:
            continue

        imp_pre = pre_impact_by_layer.get(L, None)
        imp_ft = ft_impact_by_layer.get(L, None)
        if imp_pre is None or imp_ft is None:
            continue

        scores = []
        for r in uni:
            scores.append((float(max(imp_pre[r], imp_ft[r])), r))
        scores.sort(reverse=True)

        # small y-offsets to reduce text overlap on the same layer line
        offsets = [0.10, -0.10, 0.18, -0.18, 0.26, -0.26, 0.34, -0.34]
        for j, (_, r) in enumerate(scores[: max(0, int(annotate_topn_per_layer))]):
            dy = offsets[j % len(offsets)]
            plt.text(
                r,               # x = feature idx
                L + dy,          # y = layer idx (+tiny offset)
                str(r),          # ensure idx present
                fontsize=8,
                rotation=0,
                ha="center",
                va="center",
            )

    plt.xlabel("Feature index (MLP neuron, 0..4095)")
    plt.ylabel("Block index (layer ℓ)")
    plt.title(title)

    plt.yticks(focus_layers)

    plt.xlim(-10, 4096 + 10)

    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=200)
    plt.close()

def write_copy_paste_registers(out_path: str, regs_by_layer: Dict[int, set]):
    """
    Write:
      block{idx}_registers=[...]
    for each layer idx in sorted order.
    """
    lines: List[str] = []
    for L in sorted(regs_by_layer.keys()):
        regs = sorted(list(regs_by_layer[L]))
        lines.append(f"block{L}_registers={regs}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return lines



def main():
    args = parse_arguments()
    device = args.device
    debug = bool(args.debug)

    out_dir = ensure_dir(args.out_dir)

    special_heads = parse_int_list_arg(getattr(args, "attn_special_heads", "10,13"))

    model_tags = [a for (a, _) in MODELS]
    if "pretrained" not in set(model_tags):
        raise RuntimeError("MODELS must include ('pretrained', 'ViT-L/14' or path) for comparisons.")

    models: Dict[str, torch.nn.Module] = {}
    preprocess = None

    print(f"[init] loading {len(MODELS)} model(s) on {device} ...")
    for tag, path in MODELS:
        _log(f"[init] loading {tag} = {path} ...", debug, always=False)
        m, pre, _ = load_openai_clip_anything(clip, path, device=device, jit=False, strict=True)
        models[tag] = m.float().eval()
        if preprocess is None:
            preprocess = pre

    blocks = models["pretrained"].visual.transformer.resblocks
    n_layers = len(blocks)
    n_heads = int(blocks[0].attn.num_heads)
    print(f"[model] layers={n_layers} heads={n_heads}")

    if preprocess is None:
        raise RuntimeError("preprocess is None")

    reg_images_cpu, reg_paths, reg_basenames, is_synth = load_reg_images(args, preprocess)
    print(f"[images] N={reg_images_cpu.shape[0]} dir='{args.reg_image_dir}' synth={bool(args.reg_use_synth)}")

    manifest_csv = os.path.join(out_dir, "selected_images.csv")
    with open(manifest_csv, "w", encoding="utf-8") as f:
        f.write("idx,path,basename,is_synth\n")
        for i, (p, b, s) in enumerate(zip(reg_paths, reg_basenames, is_synth)):
            f.write(f"{i},{p},{b},{s}\n")
    print(f"[images] wrote manifest: {manifest_csv}")

    ablate_blocks = parse_int_list_arg(args.ablate_blocks)
    ablate_blocks = [b for b in ablate_blocks if 0 <= b < n_layers]
    if not ablate_blocks:
        ablate_blocks = [12] if n_layers > 12 else [max(0, n_layers // 2)]
    _log(f"[ablate] blocks={ablate_blocks}", debug, always=True)

    variants: List[Tuple[str, Dict[str, bool]]] = [
        ("baseline", {"reg_nuke": False, "zero_attn": False, "zero_mlp": False}),
        ("reg_nuke_only", {"reg_nuke": True, "zero_attn": False, "zero_mlp": False}),
        ("zero_attn_only", {"reg_nuke": False, "zero_attn": True, "zero_mlp": False}),
        ("zero_mlp_only", {"reg_nuke": False, "zero_attn": False, "zero_mlp": True}),
        ("zero_attn_mlp", {"reg_nuke": False, "zero_attn": True, "zero_mlp": True}),
        ("reg_nuke_plus_zero_attn_mlp", {"reg_nuke": True, "zero_attn": True, "zero_mlp": True}),
    ]

    # Storage for suites
    l2_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    true_by_variant: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    diag_by_variant: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for variant_name, flags in variants:
        variant_out = ensure_dir(os.path.join(out_dir, variant_name))
        print(f"\n[variant] {variant_name} flags={flags}")

        handles_all: Dict[str, List[Any]] = {tag: [] for tag in model_tags}
        try:
            if flags.get("reg_nuke", False) or flags.get("zero_attn", False) or flags.get("zero_mlp", False):
                for tag in model_tags:
                    hs: List[Any] = []

                    if flags.get("reg_nuke", False):
                        hs += attach_reg_neuron_nuke_hooks(models[tag].visual)

                    if flags.get("zero_attn", False) and flags.get("zero_mlp", False):
                        hs += attach_zero_attn_mlp_for_blocks(models[tag].visual, ablate_blocks)
                    elif flags.get("zero_attn", False):
                        hs += attach_zero_attn_for_blocks(models[tag].visual, ablate_blocks)
                    elif flags.get("zero_mlp", False):
                        hs += attach_zero_mlp_for_blocks(models[tag].visual, ablate_blocks)

                    handles_all[tag] = hs

                _log(
                    f"[variant:{variant_name}] attached hooks total={sum(len(v) for v in handles_all.values())}",
                    debug,
                    always=True,
                )

            outcomes_map = None
            if getattr(args, "outcomes_csv", ""):
                outcomes_map = _load_outcomes_csv(args.outcomes_csv, args.outcomes_path_col, args.outcomes_value_col)

            # unpack 3 returns
            per_head_mean_by_model, per_head_true_by_model, per_head_diag_by_model = run_l2_suite_for_models(
                out_root=variant_out,
                models=models,
                model_tags=model_tags,
                reg_images_cpu=reg_images_cpu,
                device=device,
                batch_size=args.reg_batch_size,
                debug=debug,
                args=args,
                reg_basenames=reg_basenames,
                outcomes_map=outcomes_map,
            )

            l2_by_variant[variant_name] = per_head_mean_by_model
            true_by_variant[variant_name] = per_head_true_by_model
            diag_by_variant[variant_name] = per_head_diag_by_model

            # baseline-only mover scan
            if variant_name == "baseline":
                scan_root = ensure_dir(os.path.join(variant_out, "register_mover_scan"))
                print("\n[mover-scan] scanning candidate register-neurons (non-invasive, baseline only)")
                for tag in model_tags:
                    scan = scan_register_mover_candidates(
                        model=models[tag],
                        images_cpu=reg_images_cpu,
                        device=device,
                        batch_size=args.reg_batch_size,
                        delta_threshold=1.0,
                        adaptive_mult=2.5,
                        patch_beg=1,
                        start_layer=0,
                        end_layer=n_layers,
                        use_max=True,
                        debug=False,
                    )

                    out_txt = os.path.join(scan_root, f"{tag}__mover_candidates.txt")
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(f"tag={tag}\n")
                        f.write(f"delta_threshold=1.0 adaptive_mult=2.5 patch_beg=1\n\n")
                        for L in sorted(scan.keys()):
                            cand = sorted(list(scan[L]["candidates"]))
                            f.write(f"L{L:02d}: candidates_n={len(cand)}\n")
                            f.write(f"  candidates={cand}\n")
                            if L in REG_NEURONS:
                                known = set(REG_NEURONS[L])
                                inter = sorted(list(known & set(cand)))
                                missing = sorted(list(known - set(cand)))
                                extra = sorted(list(set(cand) - known))
                                f.write(f"  REG_NEURONS={sorted(list(known))}\n")
                                f.write(f"  intersection={inter}\n")
                                f.write(f"  missing_vs_REG_NEURONS={missing}\n")
                                f.write(f"  extra_vs_REG_NEURONS={extra}\n")
                            f.write("\n")

                    for L in [11, 12, 20]:
                        if L in scan:
                            cand = set(scan[L]["candidates"])
                            known = set(REG_NEURONS.get(L, []))
                            inter = sorted(list(cand & known))
                            extra = sorted(list(cand - known))
                            miss = sorted(list(known - cand))
                            print(f"  [mover-scan:{tag}] L{L}: cand={len(cand)} inter={len(inter)} extra={len(extra)} missing={len(miss)}")

        finally:
            total = 0
            for tag in model_tags:
                for h in handles_all[tag]:
                    try:
                        h.remove()
                        total += 1
                    except Exception:
                        pass
            if total > 0:
                _log(f"[variant:{variant_name}] removed hooks total={total}", debug, always=True)

    # save baseline deltas vs pretrained in the per-model baseline folder
    baseline_variant = "baseline"
    if baseline_variant in l2_by_variant and baseline_variant in true_by_variant:
        baseline_root = os.path.join(out_dir, baseline_variant)

        if "pretrained" not in l2_by_variant[baseline_variant]:
            raise RuntimeError("[delta_vs_pretrained] baseline missing 'pretrained' entry")

        pre_h = l2_by_variant[baseline_variant]["pretrained"]  # [L,H]
        pre_true = true_by_variant[baseline_variant]["pretrained"]  # {kind -> [L,H]}

        for tag in model_tags:
            if tag == "pretrained":
                continue

            # baseline/{tag}/l2_output_norms/delta_vs_pretrained
            delta_dir = ensure_dir(os.path.join(baseline_root, tag, "l2_output_norms", "delta_vs_pretrained"))

            # heuristic delta: (model - pretrained)
            h = l2_by_variant[baseline_variant][tag]
            save_l2_deltas_vs_pretrained(
                local_path=delta_dir,
                append_to_filename=tag,
                delta=(h - pre_h),
            )

            # true pre/post outproj deltas: (model - pretrained), per kind
            td = true_by_variant[baseline_variant].get(tag, {})
            if not isinstance(td, dict) or not td:
                continue

            for kind in sorted(set(pre_true.keys()) & set(td.keys())):
                save_l2_deltas_kind_vs_pretrained(
                    local_path=delta_dir,
                    model_tag=tag,
                    kind=kind,
                    delta=(td[kind] - pre_true[kind]),
                )

        _log(
            f"[delta_vs_pretrained] saved baseline deltas into baseline/<tag>/l2_output_norms/delta_vs_pretrained",
            debug,
            always=True,
        )

    base = "baseline"
    if base not in l2_by_variant:
        raise RuntimeError("Missing baseline results; cannot compute deltas.")

    print("\n====================")
    print("[summary] baseline vs variant deltas (per model)")
    print("====================")

    for variant_name, _flags in variants:
        if variant_name == base:
            continue
        if variant_name not in l2_by_variant:
            continue

        print(f"\n[delta] {base} -> {variant_name}")
        delta_root = ensure_dir(os.path.join(out_dir, f"delta__{base}_vs_{variant_name}"))

        for tag in model_tags:
            print(f"\n--- {tag} ---")
            tag_delta_dir = ensure_dir(os.path.join(delta_root, tag))

            # heuristic
            b = l2_by_variant[base][tag]
            a = l2_by_variant[variant_name][tag]
            lines = _delta_summary_lines(
                name="heuristic_block_output_slice",
                before=b,
                after=a,
                focus_layer=args.delta_focus_layer,
                topk=args.delta_topk,
            )
            for ln in lines:
                print(ln)

            np.savez_compressed(
                os.path.join(tag_delta_dir, "delta__heuristic_block_output_slice.npz"),
                before=b, after=a, delta=(a - b),
            )

            # true kinds
            kinds = list(true_by_variant[base][tag].keys())
            for kind in kinds:
                b2 = true_by_variant[base][tag][kind]
                a2 = true_by_variant[variant_name][tag][kind]
                lines2 = _delta_summary_lines(
                    name=f"true_{kind}",
                    before=b2,
                    after=a2,
                    focus_layer=args.delta_focus_layer,
                    topk=args.delta_topk,
                )
                for ln in lines2:
                    print(ln)

                np.savez_compressed(
                    os.path.join(tag_delta_dir, f"delta__true_{kind}.npz"),
                    before=b2, after=a2, delta=(a2 - b2),
                )

            # effective contribution diagnostics (delta section)
            if getattr(args, "attn_contrib_diag", False):
                # keep contrib artifacts NEXT TO the delta artifacts
                contrib_out_dir = ensure_dir(os.path.join(tag_delta_dir, "attn_contrib_diagnostics"))

                outcomes_map = None
                if getattr(args, "outcomes_csv", ""):
                    outcomes_map = _load_outcomes_csv(args.outcomes_csv, args.outcomes_path_col, args.outcomes_value_col)

                contrib_diag = compute_true_attn_contrib_diagnostics(
                    model=models[tag],
                    images_cpu=reg_images_cpu,
                    device=args.device,
                    batch_size=args.reg_batch_size,
                    reg_token_norm_threshold=args.reg_token_norm_threshold,
                    reg_token_fallback_topk=args.reg_token_fallback_topk,
                    outcomes_map=outcomes_map,
                    image_basenames=reg_basenames,
                )

                # Save NPZ
                npz_path = os.path.join(contrib_out_dir, f"{tag}__attn_contrib_diag.npz")
                np.savez_compressed(npz_path, **contrib_diag)

                # Save CSV
                csv_path = os.path.join(contrib_out_dir, f"{tag}__attn_contrib_diag.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    wr = csv.writer(f)
                    hdr = [
                        "layer", "head",
                        "entropy_attn_patch_cls", "entropy_contrib_patch_cls", "kl_attn_to_contrib_cls",
                        "reg_attn_mass_cls", "reg_contrib_mass_cls", "corr_reg_attn_vs_contrib",
                    ]
                    if "corr_reg_attn_vs_outcome" in contrib_diag and "corr_reg_contrib_vs_outcome" in contrib_diag:
                        hdr += ["corr_reg_attn_vs_outcome", "corr_reg_contrib_vs_outcome"]
                    wr.writerow(hdr)

                    Lc, Hc = contrib_diag["entropy_attn_patch_cls"].shape
                    for li in range(Lc):
                        for hi in range(Hc):
                            row = [
                                li, hi,
                                float(contrib_diag["entropy_attn_patch_cls"][li, hi]),
                                float(contrib_diag["entropy_contrib_patch_cls"][li, hi]),
                                float(contrib_diag["kl_attn_to_contrib_cls"][li, hi]),
                                float(contrib_diag["reg_attn_mass_cls"][li, hi]),
                                float(contrib_diag["reg_contrib_mass_cls"][li, hi]),
                                float(contrib_diag["corr_reg_attn_vs_contrib"][li, hi]),
                            ]
                            if "corr_reg_attn_vs_outcome" in contrib_diag and "corr_reg_contrib_vs_outcome" in contrib_diag:
                                row += [
                                    float(contrib_diag["corr_reg_attn_vs_outcome"][li, hi]),
                                    float(contrib_diag["corr_reg_contrib_vs_outcome"][li, hi]),
                                ]
                            wr.writerow(row)

                def _head_avg(x: np.ndarray) -> np.ndarray:
                    return x.mean(axis=1)

                Lc = contrib_diag["entropy_attn_patch_cls"].shape[0]
                xs = np.arange(Lc)

                # Entropy curves (CLS, patch-only)
                plt.figure(figsize=(10, 5))
                plt.plot(xs, _head_avg(contrib_diag["entropy_attn_patch_cls"]), label="attn entropy (CLS->patches, head-avg)")
                plt.plot(xs, _head_avg(contrib_diag["entropy_contrib_patch_cls"]), label="contrib entropy (CLS->patches, head-avg)")
                for h in special_heads:
                    if 0 <= h < contrib_diag["entropy_attn_patch_cls"].shape[1]:
                        plt.plot(xs, contrib_diag["entropy_attn_patch_cls"][:, h], linestyle="--", label=f"attn H{h:02d}")
                        plt.plot(xs, contrib_diag["entropy_contrib_patch_cls"][:, h], linestyle=":", label=f"contrib H{h:02d}")
                plt.xlabel("layer")
                plt.ylabel("entropy")
                plt.title(f"{tag}: CLS patch-only entropy (attn vs effective contribution)")
                plt.grid(True, linewidth=0.3, alpha=0.4)
                plt.legend(fontsize=8, ncol=2)
                plt.tight_layout()
                plt.savefig(png_path(contrib_out_dir, tag, "attn_contrib__entropy_cls_patchonly"), dpi=args.dpi)  # <-- FIX
                plt.close()

                # KL curve (CLS)
                plt.figure(figsize=(10, 5))
                plt.plot(xs, _head_avg(contrib_diag["kl_attn_to_contrib_cls"]), label="KL(attn||contrib), head-avg")
                for h in special_heads:
                    if 0 <= h < contrib_diag["kl_attn_to_contrib_cls"].shape[1]:
                        plt.plot(xs, contrib_diag["kl_attn_to_contrib_cls"][:, h], linestyle="--", label=f"KL H{h:02d}")
                plt.xlabel("layer")
                plt.ylabel("KL")
                plt.title(f"{tag}: KL(attn||effective contribution) for CLS (patch-only)")
                plt.grid(True, linewidth=0.3, alpha=0.4)
                plt.legend(fontsize=8, ncol=2)
                plt.tight_layout()
                plt.savefig(png_path(contrib_out_dir, tag, "attn_contrib__kl_cls_patchonly"), dpi=args.dpi)  # <-- FIX
                plt.close()

                # Printable summary
                kl = contrib_diag["kl_attn_to_contrib_cls"]
                gap = contrib_diag["reg_attn_mass_cls"] - contrib_diag["reg_contrib_mass_cls"]

                flat_kl = [(float(kl[li, hi]), li, hi) for li in range(kl.shape[0]) for hi in range(kl.shape[1])]
                flat_kl.sort(reverse=True, key=lambda x: x[0])

                flat_gap = [(float(abs(gap[li, hi])), float(gap[li, hi]), li, hi) for li in range(gap.shape[0]) for hi in range(gap.shape[1])]
                flat_gap.sort(reverse=True, key=lambda x: x[0])

                summary_lines = []
                summary_lines.append(f"=== Effective contribution summary: {tag} ===")
                summary_lines.append(f"Images={int(contrib_diag['den_cls'].max())} (per-layer counts may differ slightly if something odd happened)")
                summary_lines.append("")
                summary_lines.append("Top-15 heads by KL(attn||contrib) (CLS, patch-only):")
                for v, li, hi in flat_kl[:15]:
                    summary_lines.append(f"  L{li:02d} H{hi:02d}: KL={v:.6f}")
                summary_lines.append("")
                summary_lines.append("Top-15 heads by |reg_attn_mass - reg_contrib_mass| (CLS, patch-only):")
                for absv, signed, li, hi in flat_gap[:15]:
                    summary_lines.append(f"  L{li:02d} H{hi:02d}: gap={signed:+.6f} |gap|={absv:.6f}")
                summary_lines.append("")

                corr = contrib_diag["corr_reg_attn_vs_contrib"]
                summary_lines.append("Layerwise head-avg corr(reg_attn_mass, reg_contrib_mass):")
                for li in range(corr.shape[0]):
                    summary_lines.append(f"  L{li:02d}: corr={float(corr[li].mean()):+.4f}")

                save_summary_txt(os.path.join(contrib_out_dir, f"{tag}__attn_contrib_diag_summary.txt"), summary_lines)
                print("\n".join(summary_lines[:40]))  # short print; full is saved

            # save attention diagnostic deltas, so correlation finds delta__attn_qnorm_all.npz
            base_diag = diag_by_variant.get(base, {}).get(tag, {})
            var_diag = diag_by_variant.get(variant_name, {}).get(tag, {})
            if base_diag and var_diag:
                for dk in sorted(set(base_diag.keys()) & set(var_diag.keys())):
                    bd = base_diag[dk]
                    ad = var_diag[dk]
                    np.savez_compressed(
                        os.path.join(tag_delta_dir, f"delta__attn_{dk}.npz"),
                        before=bd, after=ad, delta=(ad - bd),
                    )
                if "qnorm_all" in base_diag and "qnorm_all" in var_diag:
                    lines_q = _delta_summary_lines(
                        name="attn_qnorm_all",
                        before=base_diag["qnorm_all"],
                        after=var_diag["qnorm_all"],
                        focus_layer=args.delta_focus_layer,
                        topk=args.delta_topk,
                    )
                    for ln in lines_q:
                        print(ln)

            correlate_and_plot_late_qnorm_vs_heuristic(
                tag_delta_dir=tag_delta_dir,
                n_layers=n_layers,
                model_tag=tag,
                late_k=6,
                dpi=args.dpi,
                debug=debug,
            )

        print(f"[delta] artifacts in: {delta_root}")

    print(f"\n[done] outputs in: {out_dir}")

if __name__ == "__main__":
    main()