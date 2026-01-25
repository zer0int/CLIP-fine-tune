"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

CLIP ViT Raw Attention Capture + Visualization
========================================================

- Runs CLIP's visual encoder on one or more images and reads out each ViT block's raw attention
  probabilities (per layer: [heads, query_tokens, source_tokens]).
- Selects a set of query tokens to inspect:
    * CLS always
    * optional high-norm "register**" patches (via --use_reg and --reg_thr)
    * optional "normal" patches (via --get_normal and --hi_lo/--hi_hi)
- Visualization (--save_pngs): saves attention overlay PNGs for baseline / nuked / |Δ|.

** Register: High-norm patch with global information in a (supposedly local) vision token
-> See also: Vision Transformers Need Registers, https://arxiv.org/abs/2309.16588

A head isn’t just A @ V. It’s (A @ V) @ W_o (plus bias), where W_o is the head output projection.
W_o can:
- rotate contributions into directions that matter downstream,
- squash entire subspaces (so big ‖A·V‖ becomes irrelevant),
- or amplify certain components (small ‖A·V‖ becomes decisive).
"""

import os
import copy
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import cv2
import argparse
from typing import Dict, Tuple, List, Optional, Any
import textwrap
from PIL import ImageDraw, ImageFont

import attnclipdecouple as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
from utils_clip_loader.cliptools import fix_random_seed
fix_random_seed()

import warnings # stop especially torch spam
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"


# for '--nuke_regs'
REG_NEURONS: Dict[int, List[int]] = {
    11: [9, 987, 1967, 2555, 3661, 3784],
    12: [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868],
}

VIZ_SIZE = 448          # base overlay canvas (heatmap+image) size
LABEL_H  = 56           # extra header strip height
LABEL_PAD = 6
PROMPT_FONT_SIZE = 18

# Fonts to try for overlay
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
]

def parse_arguments():
    p = argparse.ArgumentParser("CLIP ViT raw attention capture + visualization")
    p.add_argument("--use_model", default="zer0int/CLIP-Regression-ViT-L-14", help="CLIP model id/path to load (OpenAI name, HF id, or local checkpoint).")
    p.add_argument("--image_folder", default="image_sets/attn_bench_images", help="Folder containing input images (.png).")
    p.add_argument("--out_folder", default="out_vis_attn/attn_raw_heads_qv", help="Root output folder for saved PNG overlays (if --save_pngs).")

    # Visualize everything as images with overlays
    p.add_argument("--save_value", action="store_true", help="Also save value-side attribution: per (layer,head,query).")
    p.add_argument("--save_pngs", action="store_true", help="Save attention overlay PNGs for baseline/nuked/|Δ|.")
    p.add_argument("--save_outproj", action="store_true", help="Also save post-out_proj contribution maps.")
    # WARNING: Below will result in *HUGE* image dumps (depending on how many heads/layers chosen):
    p.add_argument("--use_reg", action="store_true", help="Auto-select all high-norm 'register' patch tokens as attention query tokens.")
    p.add_argument("--get_normal", action="store_true", help="Include additional 'normal' (non-register) patch tokens as attention queries.")

    # Viz controls
    p.add_argument("--use_mask", action="store_true", help="Viz-only: mask register targets (set to 0 and renormalize) in overlays.")
    #p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23", help="All comma-separated ViT layers.")
    p.add_argument("--layers", default="0,1,2,5,12,22,23", help="Comma-separated ViT layers.")
    p.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", help="Comma-separated attention heads to report/save.")

    # Query selection
    p.add_argument("--select_block", type=int, default=13, help="ViT block used to compute patch norms for Query selection. Important for Registers.")
    p.add_argument("--reg_thr", type=float, default=70.0, help="Register threshold: patch norm > reg_thr at --select_block counts as register.")
    p.add_argument("--hi_lo", type=float, default=10.0, help="Lower bound for 'hi' normal patch selection: hi_lo <= norm < hi_hi.")
    p.add_argument("--hi_hi", type=float, default=50.0, help="Upper bound for 'hi' normal patch selection: hi_lo <= norm < hi_hi.")
    p.add_argument("--top_k", type=int, default=2, help="How many patches to include for each normal bucket (hi + lo).")
    
    # Intervention
    p.add_argument("--nuke_regs", action="store_true", help="Run a second pass with detected 'registers' zeroed and compare vs baseline.")

    return p.parse_args()


def get_font(font_size: int = PROMPT_FONT_SIZE) -> ImageFont.FreeTypeFont:
    for fp in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            pass
    return ImageFont.load_default()

@torch.no_grad()
def run_visual_forward(model: torch.nn.Module, img: torch.Tensor, capture_layers: Optional[set] = None):
    """
    Run the visual encoder once. If capture_layers is provided, those blocks will store last_q/k/v/probs/logits.
    """
    _ = model.visual(img.type(model.dtype), return_trace=False, capture_layers=capture_layers)


def read_attn_probs_cache(model: torch.nn.Module) -> Dict[int, torch.Tensor]:
    """
    Read already-populated blk.attn_probs from each vision block.
    Returns cache[layer] = [H,T,S] on CPU float32.
    """
    cache: Dict[int, torch.Tensor] = {}
    for li, blk in enumerate(model.visual.transformer.resblocks):
        ap = getattr(blk, "attn_probs", None)
        if ap is None:
            continue
        ap = ap.detach().float().cpu()
        # expected [B,H,T,S]
        if ap.dim() == 4:
            cache[li] = ap[0]  # [H,T,S]
    return cache

def read_probs_v_cache(model: torch.nn.Module, layers: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    For layers that were run with capture=True, read:
      - probs: [H,T,S]
      - v:     [H,S,D]
    all CPU float32 (batch=0)
    """
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for li in layers:
        if li < 0 or li >= len(model.visual.transformer.resblocks):
            continue
        blk = model.visual.transformer.resblocks[li]
        probs = getattr(blk.attn, "last_probs", None)  # [B,H,T,S]
        v = getattr(blk.attn, "last_v", None)          # [B,H,S,D]
        if probs is None or v is None:
            continue
        out[li] = {
            "probs": probs.detach().float().cpu()[0],  # [H,T,S]
            "v": v.detach().float().cpu()[0],          # [H,S,D]
        }
    return out


def value_contrib_vec_from_probs_v(probs_h_q: torch.Tensor, v_h: torch.Tensor) -> torch.Tensor:
    """
    probs_h_q: [S] attention probs for one head at one query token
    v_h:       [S,D] value vectors for that head (per source token)
    Returns contribution magnitude per source token:
      contrib[s] = || probs[s] * v[s] ||_2
    """
    # NOTE: non-negative by construction
    return (v_h * probs_h_q.unsqueeze(-1)).norm(dim=-1)  # [S]


def outproj_head_contrib_vec(
    probs_h_q: torch.Tensor,     # [S]
    v_h: torch.Tensor,           # [S,D]
    W_o: torch.Tensor,           # [E,E]  (CPU float32)
    head_idx: int,
) -> torch.Tensor:
    """
    Per-source contribution magnitude AFTER applying the head's slice of out_proj:
      u_s = probs[s] * v[s]               [D]
      y_s = u_s @ W_h^T,  W_h = W_o[:, hD:(h+1)D]  [E,D]
      contrib[s] = ||y_s||_2
    Returns: [S] (CPU float32)
    """
    H = None  # unused
    E = W_o.shape[0]
    D = v_h.shape[1]
    start = head_idx * D
    end = start + D
    W_h = W_o[:, start:end]  # [E,D]

    U = v_h * probs_h_q.unsqueeze(-1)     # [S,D]
    Y = U @ W_h.t()                       # [S,E]
    return Y.norm(dim=-1)                 # [S]


def outproj_full_before_after_contrib(
    probs_q: torch.Tensor,   # [H,S]
    v: torch.Tensor,         # [H,S,D]
    W_o: torch.Tensor,       # [E,E]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full all-head per-source contrib magnitudes:
      pre:  z_s = concat_h (probs[h,s] * v[h,s])          [E]
      post: y_s = z_s @ W_o^T                            [E]
    Returns (pre_contrib, post_contrib), each [S].
    """
    Hh, S, D = v.shape
    U = v * probs_q.unsqueeze(-1)                 # [H,S,D]
    Z = U.permute(1, 0, 2).reshape(S, Hh * D)     # [S,E]
    pre = Z.norm(dim=-1)                          # [S]
    post = (Z @ W_o.t()).norm(dim=-1)             # [S]
    return pre, post


def save_patch_grid_npy(contrib_vec: torch.Tensor, out_npy_path: str):
    patch = contrib_vec[1:].detach().float().cpu()
    P = patch.numel()
    grid = int(np.sqrt(P))
    assert grid * grid == P
    np.save(out_npy_path, patch.view(grid, grid).numpy())


def save_value_contrib_overlay_and_data(
    contrib_vec: torch.Tensor,
    orig_image: Image.Image,
    out_png_path: str,
    out_npy_path: str,
    title: str,
    query_token_idx: int,
    reg_mask_patch: Optional[torch.Tensor] = None,
    use_mask: bool = False,
):
    """
    contrib_vec: [seq_len] non-negative per-source contribution magnitudes
    Saves:
      - PNG overlay (uses same overlay logic as attention)
      - raw patch-grid data as .npy (grid,grid) in *un-normalized* units
    """
    # --- save raw data grid (patches only) ---
    patch = contrib_vec[1:].detach().float().cpu()  # [P]
    P = patch.numel()
    grid = int(np.sqrt(P))
    assert grid * grid == P, f"Expected square grid patches, got P={P}"
    patch_grid = patch.view(grid, grid).numpy()
    np.save(out_npy_path, patch_grid)

    # --- reuse existing overlay renderer ---
    save_attention_overlay(
        attn_vec=contrib_vec,
        orig_image=orig_image,
        out_path=out_png_path,
        title=title,
        query_token_idx=query_token_idx,
        reg_mask_patch=reg_mask_patch,
        use_mask=use_mask,
    )


def attach_reg_neuron_nuke_hooks(visual: torch.nn.Module) -> List[Any]:
    """
    Zero specified MLP expanded dims at c_fc output (pre-gelu), for blocks in REG_NEURONS.
    """
    handles = []
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


def remove_hooks(handles: List[Any]):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _load_label_font(size: int):
    preferred = get_font(PROMPT_FONT_SIZE)
    try:
        if os.path.exists(preferred):
            return ImageFont.truetype(preferred, size=size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def add_label_strip_above_uint8(
    rgb_uint8: np.ndarray,
    title: str,
    label_h: int = LABEL_H,
    pad: int = LABEL_PAD,
) -> np.ndarray:
    """
    Returns a new image with an extra header strip above the input.
    Input:  rgb_uint8 [H,W,3] uint8
    Output: [H+label_h, W, 3] uint8
    """
    if not title:
        return rgb_uint8

    H, W = rgb_uint8.shape[:2]
    header = Image.new("RGB", (W, label_h), (0, 0, 0))

    # Wrap title to fit the header width
    # (width ~ characters; tuned for 448px-ish)
    lines = textwrap.wrap(title, width=52)
    # keep it sane
    lines = lines[:3]
    title_wrapped = "\n".join(lines)

    # Choose a font size that fits inside LABEL_H
    font_size = PROMPT_FONT_SIZE
    font = _load_label_font(font_size)

    draw = ImageDraw.Draw(header)

    for _ in range(12):
        bbox = draw.multiline_textbbox((0, 0), title_wrapped, font=font, spacing=2)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= (W - 2 * pad) and th <= (label_h - 2 * pad):
            break
        font_size = max(10, font_size - 2)
        font = _load_label_font(font_size)

    # Draw text with outline for readability
    x, y = pad, max(pad, (label_h - th) // 2)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.multiline_text((x + dx, y + dy), title_wrapped, font=font, fill=(0, 0, 0), spacing=2)
    draw.multiline_text((x, y), title_wrapped, font=font, fill=(255, 255, 255), spacing=2)

    header_np = np.array(header, dtype=np.uint8)
    out = np.concatenate([header_np, rgb_uint8], axis=0)
    return out


def make_overlay_image_uint8(
    attn_vec: torch.Tensor,
    orig_image: Image.Image,
    query_token_idx: int,
    reg_mask_patch: Optional[torch.Tensor] = None,
    use_mask: bool = False,
    box_thickness: int = 2,
) -> np.ndarray:
    """
    Returns overlay as uint8 RGB array [VIZ_SIZE,VIZ_SIZE,3].
    """
    # fixed render target -> bounded file size
    orig_image = orig_image.resize((VIZ_SIZE, VIZ_SIZE), resample=Image.BICUBIC)

    patch_attn = attn_vec[1:].detach().float().cpu().clone()  # [num_patches]

    if use_mask and reg_mask_patch is not None:
        if reg_mask_patch.dtype != torch.bool:
            reg_mask_patch = reg_mask_patch.to(torch.bool)
        if reg_mask_patch.numel() == patch_attn.numel():
            patch_attn[reg_mask_patch] = 0
            s = patch_attn.sum()
            if float(s) > 0:
                patch_attn = patch_attn / (s + 1e-12)

    num_patches = patch_attn.numel()
    grid = int(np.sqrt(num_patches))
    assert grid * grid == num_patches, f"Expected square grid patches, got {num_patches}"

    p = patch_attn.numpy()
    p = (p - p.min()) / (p.ptp() + 1e-9)
    p = p.reshape(grid, grid)

    p = cv2.resize(p, (VIZ_SIZE, VIZ_SIZE), interpolation=cv2.INTER_CUBIC)
    p_u8 = (p * 255.0).clip(0, 255).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(p_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    img_rgb = np.array(orig_image, dtype=np.uint8)
    overlay = cv2.addWeighted(heat_rgb, 0.4, img_rgb, 0.6, 0.0)

    # draw query box (if patch query)
    if query_token_idx > 0:
        pidx = query_token_idx - 1
        py, px = divmod(pidx, grid)
        H, W = overlay.shape[:2]

        x0 = int(px * W / grid)
        x1 = int((px + 1) * W / grid) - 1
        y0 = int(py * H / grid)
        y1 = int((py + 1) * H / grid) - 1
        tmax = max(1, int(box_thickness))
        for t in range(tmax):
            cv2.rectangle(
                overlay,
                (max(0, x0 + t), max(0, y0 + t)),
                (min(W - 1, x1 - t), min(H - 1, y1 - t)),
                color=(255, 255, 255),
                thickness=1
            )

    return overlay


def save_attention_overlay(
    attn_vec: torch.Tensor,
    orig_image: Image.Image,
    out_path: str,
    title: str,
    query_token_idx: int,
    reg_mask_patch: Optional[torch.Tensor] = None,
    use_mask: bool = False,
    box_thickness: int = 2,
):
    """
    Always writes a labeled PNG:
      - base overlay is fixed VIZ_SIZE x VIZ_SIZE
      - plus LABEL_H header strip above
    """
    overlay = make_overlay_image_uint8(
        attn_vec=attn_vec,
        orig_image=orig_image,
        query_token_idx=query_token_idx,
        reg_mask_patch=reg_mask_patch,
        use_mask=use_mask,
        box_thickness=box_thickness,
    )
    overlay_labeled = add_label_strip_above_uint8(overlay, title=title)
    Image.fromarray(overlay_labeled).save(out_path)


def save_before_after_combined_png(
    pre_vec: torch.Tensor,
    post_vec: torch.Tensor,
    orig_img: Image.Image,
    out_png_path: str,
    query_token_idx: int,
    reg_mask_patch: Optional[torch.Tensor],
    use_mask: bool,
    title: str,  # required title so labels are guaranteed
):
    """
    Writes a labeled side-by-side:
      - (VIZ_SIZE x VIZ_SIZE) + (VIZ_SIZE x VIZ_SIZE) concatenated => (2*VIZ_SIZE x VIZ_SIZE)
      - plus LABEL_H header above => (2*VIZ_SIZE x (VIZ_SIZE+LABEL_H))
    """
    left = make_overlay_image_uint8(pre_vec, orig_img, query_token_idx, reg_mask_patch, use_mask)
    right = make_overlay_image_uint8(post_vec, orig_img, query_token_idx, reg_mask_patch, use_mask)
    combo = np.concatenate([left, right], axis=1)
    combo_labeled = add_label_strip_above_uint8(combo, title=title)
    Image.fromarray(combo_labeled).save(out_png_path)


@torch.no_grad()
def forward_and_get_patch_norms(model: torch.nn.Module, img: torch.Tensor, select_block: int) -> torch.Tensor:
    """
    Captures token stream output at select_block, returns patch token norms: [num_patches]
    """
    blocks = list(model.visual.transformer.resblocks)
    n_blocks = len(blocks)

    if select_block < 0:
        select_block = n_blocks + select_block
    if not (0 <= select_block < n_blocks):
        raise ValueError(f"select_block={select_block} out of range for n_blocks={n_blocks}")

    captured = {}

    def _hook(_module, _inp, out):
        captured["x"] = out

    h = blocks[select_block].register_forward_hook(_hook)
    _ = model.encode_image(img)
    h.remove()

    x = captured.get("x", None)
    if x is None:
        raise RuntimeError("Failed to capture token stream at select_block.")
    if isinstance(x, tuple):
        x = x[0]
    x = x.detach()

    seq_len = model.visual.positional_embedding.shape[0]
    if x.dim() != 3:
        raise RuntimeError(f"Unexpected captured token tensor shape: {tuple(x.shape)}")

    # [S,B,D] vs [B,S,D]
    if x.shape[0] == seq_len:
        x_bsd = x.permute(1, 0, 2)
    else:
        x_bsd = x

    tok = x_bsd[0]          # [S,D]
    patches = tok[1:].float()
    return patches.norm(dim=-1)


def build_query_tokens(
    norms_patch: torch.Tensor,
    use_reg: bool,
    get_normal: bool,
    reg_thr: float,
    hi_lo: float,
    hi_hi: float,
    top_k: int,
) -> Tuple[List[Tuple[str, int]], torch.Tensor]:
    """
    Returns (queries, reg_mask_patch)
      queries: [(label, token_idx_in_seq)] with token_idx 0=CLS, else 1..S-1
      reg_mask_patch: [num_patches] bool
    """
    norms = norms_patch.detach().cpu()
    queries: List[Tuple[str, int]] = [("cls", 0)]
    used = {0}

    reg_mask_patch = norms > reg_thr

    if use_reg:
        reg_idxs = torch.nonzero(reg_mask_patch, as_tuple=False).flatten().tolist()
        reg_idxs = sorted(reg_idxs, key=lambda i: float(norms[i]), reverse=True)
        for pi in reg_idxs:
            tidx = 1 + int(pi)
            if tidx not in used:
                queries.append((f"reg_{tidx}", tidx))
                used.add(tidx)

    if get_normal:
        hi_mask = (norms >= hi_lo) & (norms < hi_hi)
        hi_idxs = torch.nonzero(hi_mask, as_tuple=False).flatten().tolist()
        hi_idxs = sorted(hi_idxs, key=lambda i: float(norms[i]), reverse=True)[:max(0, top_k)]
        for pi in hi_idxs:
            tidx = 1 + int(pi)
            if tidx not in used:
                queries.append((f"hi_patch_{tidx}", tidx))
                used.add(tidx)

        lo_mask = norms < hi_lo
        lo_idxs = torch.nonzero(lo_mask, as_tuple=False).flatten().tolist()
        lo_idxs = sorted(lo_idxs, key=lambda i: float(norms[i]), reverse=True)[:max(0, top_k)]
        for pi in lo_idxs:
            tidx = 1 + int(pi)
            if tidx not in used:
                queries.append((f"patch_{tidx}", tidx))
                used.add(tidx)

    return queries, reg_mask_patch


@torch.no_grad()
def collect_raw_attn_probs(model: torch.nn.Module, img: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Runs encode_image(img) once, then collects per-layer raw attention probs:
      cache[layer] = attn_probs[0]  with shape [H, T, S]
    Uses block.attn_probs already populated by your ResidualAttentionBlock.
    """
    _ = model.encode_image(img)  # populates block.attn_probs as a side effect
    cache: Dict[int, torch.Tensor] = {}
    for li, blk in enumerate(model.visual.transformer.resblocks):
        ap = getattr(blk, "attn_probs", None)
        if ap is None:
            continue
        ap = ap.detach().float().cpu()
        # expected [B,H,T,S]
        if ap.dim() == 4:
            cache[li] = ap[0]  # [H,T,S]
    return cache


def diff_stats(A: torch.Tensor, B: torch.Tensor) -> Tuple[float, float]:
    D = (A - B).abs()
    return float(D.max().item()), float(D.mean().item())


def report_max_delta(base: torch.Tensor, nuke: torch.Tensor, layer: int):
    # base/nuke: [H, T, S]
    D = (base - nuke).abs()
    flat = D.view(-1)
    idx = int(flat.argmax().item())
    H, T, S = D.shape
    h = idx // (T * S)
    rem = idx % (T * S)
    q = rem // S
    s = rem % S
    print(f"  L{layer:02d}: max|Δ| at head={h}, query={q}, src={s}, |Δ|={float(D[h,q,s]):.6f}")


def report_max_delta_details(base: torch.Tensor, nuke: torch.Tensor, layer: int, grid: int = 16):
    # base/nuke: [H,T,S]
    D = (base - nuke).abs()
    H, T, S = D.shape
    idx = int(D.view(-1).argmax().item())
    h = idx // (T * S)
    rem = idx % (T * S)
    q = rem // S
    s = rem % S

    a = float(base[h, q, s].item())
    b = float(nuke[h, q, s].item())
    d = a - b

    def tok_xy(t: int):
        if t == 0:
            return "CLS"
        p = t - 1
        return f"(y={p//grid}, x={p%grid})"

    print(
        f"  L{layer:02d} argmax: head={h}, q={q} {tok_xy(q)} -> src={s} {tok_xy(s)} |Δ|={float(D[h,q,s]):.6f} "
        f"base={a:.6f} nuke={b:.6f} (base-nuke={d:+.6f})"
    )


def report_headwise_max_delta_details(
    base: torch.Tensor,
    nuke: torch.Tensor,
    layer: int,
    heads_keep: List[int],
    grid: int = 16,
):
    """
    For each head in heads_keep, find its own argmax over (q, src),
    and print detailed base/nuke values + token coords.
    """
    D = (base - nuke).abs()  # [H,T,S]
    H, T, S = D.shape

    def tok_xy(t: int):
        if t == 0:
            return "CLS"
        p = t - 1
        return f"(y={p//grid}, x={p%grid})"

    for h in heads_keep:
        if h < 0 or h >= H:
            continue
        Dh = D[h]  # [T,S]
        idx = int(Dh.view(-1).argmax().item())
        q = idx // S
        s = idx % S

        a = float(base[h, q, s].item())
        b = float(nuke[h, q, s].item())
        d = a - b
        md = float(Dh[q, s].item())

        print(
            f"    H{h:02d}: q={q:3d} {tok_xy(q)} -> s={s:3d} {tok_xy(s)} |Δ|={md:.6f} "
            f"base={a:.6f} nuke={b:.6f} (base-nuke={d:+.6f})"
        )


def hub_matrix(attn_map: torch.Tensor, hubs=(0, 18, 50)) -> torch.Tensor:
    # attn_map: [H,T,S]
    H = attn_map.shape[0]
    out = torch.zeros((H, len(hubs), len(hubs)), dtype=attn_map.dtype)
    for hi, q in enumerate(hubs):
        for sj, s in enumerate(hubs):
            out[:, hi, sj] = attn_map[:, q, s]
    return out  # [H, 3, 3]


def print_hub_deltas_per_head(
    base: torch.Tensor,
    nuke: torch.Tensor,
    layer: int,
    heads_keep: List[int],
    hubs=(0, 18, 50),
    topk_per_head: int = 5
):
    """
    Print hub-edge deltas restricted to heads_keep.
    For each head, show top-k hub edges by |Δ|.
    """
    Hb = hub_matrix(base, hubs=hubs)  # [H,3,3]
    Hn = hub_matrix(nuke, hubs=hubs)  # [H,3,3]
    D = (Hb - Hn).abs()               # [H,3,3]

    print(f"\n[L{layer:02d}] Hub-edge deltas per head (hubs={hubs}):")
    for h in heads_keep:
        if h < 0 or h >= D.shape[0]:
            continue
        flat = D[h].view(-1)
        vals, idxs = torch.topk(flat, k=min(topk_per_head, flat.numel()), largest=True)

        print(f"  Head {h:02d}:")
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            i = idx // 3
            j = idx % 3
            q = hubs[i]
            s = hubs[j]
            print(f"    q={q:3d}->s={s:3d} |Δ|={v:.6f} base={Hb[h,i,j]:.6f} nuke={Hn[h,i,j]:.6f}")


@torch.no_grad()
def capture_qk_norms(model, img, layers_to_capture: List[int]):
    # run visual forward with capture_layers
    _ = model.visual(img.type(model.dtype), return_trace=False, capture_layers=set(layers_to_capture))

    out = {}
    for li in layers_to_capture:
        blk = model.visual.transformer.resblocks[li]
        q = getattr(blk.attn, "last_q", None)  # [B,H,T,D]
        k = getattr(blk.attn, "last_k", None)  # [B,H,S,D]
        if q is None or k is None:
            continue

        qn = q.norm(dim=-1)  # [B,H,T]
        kn = k.norm(dim=-1)  # [B,H,S]

        out[li] = {
            "q_mean_per_head": qn.mean(dim=-1)[0].cpu(),  # [H]
            "k_mean_per_head": kn.mean(dim=-1)[0].cpu(),  # [H]
            "q_cls_per_head": qn[0, :, 0].cpu(),          # [H]
            "k_cls_per_head": kn[0, :, 0].cpu(),          # [H]
        }
    return out


def print_qk_norm_deltas(
    stats_base: Dict[int, Dict[str, torch.Tensor]],
    stats_nuke: Dict[int, Dict[str, torch.Tensor]],
    heads_keep: List[int],
    layers_to_capture: List[int],
    tag: str = ""
):
    hdr = f"\n[QK norms Δ]{' ' + tag if tag else ''}"
    print(hdr)
    for li in layers_to_capture:
        if li not in stats_base or li not in stats_nuke:
            print(f"  L{li:02d}: missing (base={li in stats_base}, nuke={li in stats_nuke})")
            continue

        sb = stats_base[li]
        sn = stats_nuke[li]

        print(f"  L{li:02d}:")
        for h in heads_keep:
            if h < 0 or h >= sb["q_mean_per_head"].numel():
                continue

            qmb = float(sb["q_mean_per_head"][h])
            qmn = float(sn["q_mean_per_head"][h])
            kmb = float(sb["k_mean_per_head"][h])
            kmn = float(sn["k_mean_per_head"][h])

            qcb = float(sb["q_cls_per_head"][h])
            qcn = float(sn["q_cls_per_head"][h])
            kcb = float(sb["k_cls_per_head"][h])
            kcn = float(sn["k_cls_per_head"][h])

            print(
                f"    H{h:02d}: "
                f"Qmean {qmb:7.2f}->{qmn:7.2f} Δ={qmn-qmb:+7.2f} | "
                f"Kmean {kmb:7.2f}->{kmn:7.2f} Δ={kmn-kmb:+7.2f} | "
                f"Qcls  {qcb:7.2f}->{qcn:7.2f} Δ={qcn-qcb:+7.2f} | "
                f"Kcls  {kcb:7.2f}->{kcn:7.2f} Δ={kcn-kcb:+7.2f}"
            )


def _safe_entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    return float((-p * p.log()).sum().item())


def report_baseline_attention_stats(
    base_cache: Dict[int, torch.Tensor],
    layers: List[int],
    heads_keep: List[int],
    query_list: List[Tuple[str, int]],
    image_name: str,
    grid: int = 16,
    max_queries_print: int = 4,
):
    """
    Always prints something meaningful for baseline attention, even if --nuke_regs/--save_pngs are off.
    Prints, per selected layer:
      - global max attention prob + (head, q, s) location
      - mean entropy over heads_keep for a few query tokens (CLS + first few extras)
    """
    if not layers:
        print(f"[STATS] {image_name}: no layers selected (empty --layers).")
        return

    # limit query spam
    q_print = query_list[:max(1, min(len(query_list), max_queries_print))]

    def tok_xy(t: int) -> str:
        if t == 0:
            return "CLS"
        p = t - 1
        return f"(y={p//grid}, x={p%grid})"

    print(f"\n[BASELINE STATS] {image_name}:")
    for li in layers:
        if li not in base_cache:
            print(f"  L{li:02d}: missing (no attn_probs captured)")
            continue

        A = base_cache[li]  # [H,T,S]
        H, T, S = A.shape
        flat_idx = int(A.view(-1).argmax().item())
        h = flat_idx // (T * S)
        rem = flat_idx % (T * S)
        q = rem // S
        s = rem % S
        vmax = float(A[h, q, s].item())

        print(f"  L{li:02d}: maxP={vmax:.6f} at head={h}, q={q} {tok_xy(q)}, src={s} {tok_xy(s)}")

        # mean entropy over selected heads, for a few query tokens
        for qlab, qidx in q_print:
            if qidx < 0 or qidx >= T:
                continue
            ents = []
            for hh in heads_keep:
                if 0 <= hh < H:
                    ents.append(_safe_entropy(A[hh, qidx, :]))
            if ents:
                print(f"        H∈heads: meanH(entropy) for q={qlab:>10s} (t={qidx:3d}) -> {float(np.mean(ents)):.6f}")


def main():
    args = parse_arguments()

    if args.save_pngs or args.save_value or args.save_outproj:
        os.makedirs(args.out_folder, exist_ok=True)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    heads_keep = [int(x.strip()) for x in args.heads.split(",") if x.strip()]

    # unified info print for all write modes
    if not (args.save_pngs or args.save_value or args.save_outproj):
        print("[INFO] Running in stats-only mode (no overlay images will be written).")
        print("[INFO] Add --save_pngs to write raw attention overlays.")
        print("[INFO] Add --save_value to write value-side contribution overlays + raw .npy grids.")
        print("[INFO] Add --save_outproj to write out_proj contribution overlays + raw .npy grids.")
    else:
        if not args.save_pngs:
            print("[INFO] --save_pngs is OFF: raw-attention overlays will NOT be written.")
        if args.save_value:
            print("[INFO] --save_value is ON: value-side contribution overlays + .npy grids WILL be written.")
        if args.save_outproj:
            print("[INFO] --save_outproj is ON: out_proj contribution overlays + .npy grids WILL be written.")

    model, preprocess, _ = load_openai_clip_anything(
        clip, args.use_model, device=device, jit=False, strict=True
    )
    model = model.float().eval()

    image_files = sorted(glob(os.path.join(args.image_folder, "*")))
    if not image_files:
        print(f"[WARN] No files in {args.image_folder}")
        return

    valid_image_files = []
    for p in image_files:
        if not os.path.isfile(p):
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            valid_image_files.append(p)
        except Exception:
            pass

    if not valid_image_files:
        print(f"[WARN] No readable images in {args.image_folder}")
        return

    # Choose QK capture layers
    n_layers_total = len(model.visual.transformer.resblocks)
    penult = max(0, n_layers_total - 2)
    last = max(0, n_layers_total - 1)
    qk_layers = sorted(set([0, penult, last] + layers))

    for img_file in valid_image_files:
        image_name = os.path.splitext(os.path.basename(img_file))[0]
        orig_img = Image.open(img_file).convert("RGB")
        img = preprocess(orig_img).unsqueeze(0).to(device)

        # Build queries from BASELINE norms (fixed across passes)
        norms_patch = forward_and_get_patch_norms(model, img, args.select_block)
        query_list, reg_mask_patch = build_query_tokens(
            norms_patch,
            use_reg=args.use_reg,
            get_normal=args.get_normal,
            reg_thr=args.reg_thr,
            hi_lo=args.hi_lo,
            hi_hi=args.hi_hi,
            top_k=args.top_k,
        )
        if args.use_reg:
            print(f"[{image_name}] regs@block{args.select_block} (norm>{args.reg_thr:g}): {int(reg_mask_patch.sum().item())}")

        cap_layers = set()
        need_pv = bool(args.save_value or args.save_outproj)
        if need_pv:
            cap_layers.update(layers)      # last_probs + last_v
        if args.nuke_regs:
            cap_layers.update(qk_layers)   # Q/K deltas
        run_visual_forward(model, img, capture_layers=(cap_layers if cap_layers else None))

        base_cache = read_attn_probs_cache(model)
        base_pv_cache = read_probs_v_cache(model, layers) if need_pv else {}

        # always print baseline stats
        report_baseline_attention_stats(
            base_cache=base_cache,
            layers=layers,
            heads_keep=heads_keep,
            query_list=query_list,
            image_name=image_name,
            grid=16,
            max_queries_print=4,
        )

        # Nuked pass
        nuke_cache = None
        nuke_pv_cache = None
        handles = []
        if args.nuke_regs:
            stats_qk_base = {}
            try:
                stats_qk_base = capture_qk_norms(model, img, qk_layers)
            except TypeError:
                print("[WARN] capture_qk_norms unavailable for this clip fork (no capture_layers).")

            handles = attach_reg_neuron_nuke_hooks(model.visual)

            # single nuked forward
            run_visual_forward(model, img, capture_layers=(cap_layers if cap_layers else None))
            nuke_cache = read_attn_probs_cache(model)
            nuke_pv_cache = read_probs_v_cache(model, layers) if need_pv else None

            try:
                stats_qk_nuke = capture_qk_norms(model, img, qk_layers)
            except TypeError:
                stats_qk_nuke = {}

            remove_hooks(handles)

            print(f"\n[RAW Δ] {image_name}:")
            for li in layers:
                if li not in base_cache or li not in nuke_cache:
                    print(f"  L{li:02d}: missing (base={li in base_cache}, nuke={li in nuke_cache})")
                    continue
                mx, me = diff_stats(base_cache[li], nuke_cache[li])
                print(f"  L{li:02d}: max|Δ|={mx:.8f}  mean|Δ|={me:.10f}")

            print(f"\n[MAX Δ] {image_name}:")
            for li in layers:
                if li not in base_cache or li not in nuke_cache:
                    print(f"  L{li:02d}: missing (base={li in base_cache}, nuke={li in nuke_cache})")
                    continue
                mx, me = diff_stats(base_cache[li], nuke_cache[li])
                print(f"  L{li:02d}: max|Δ|={mx:.8f}  mean|Δ|={me:.10f}")
                report_max_delta(base_cache[li], nuke_cache[li], li)
                report_max_delta_details(base_cache[li], nuke_cache[li], li)

                if mx > 0:
                    print(f"  L{li:02d} per-head argmax (restricted to --heads):")
                    report_headwise_max_delta_details(base_cache[li], nuke_cache[li], li, heads_keep=heads_keep)

                if mx > 0:
                    print_hub_deltas_per_head(
                        base_cache[li],
                        nuke_cache[li],
                        layer=li,
                        heads_keep=heads_keep,
                        hubs=(0, 18, 50),
                        topk_per_head=5
                    )

            if stats_qk_base and stats_qk_nuke:
                print_qk_norm_deltas(
                    stats_qk_base,
                    stats_qk_nuke,
                    heads_keep=heads_keep,
                    layers_to_capture=qk_layers,
                    tag=image_name
                )

        # PNG saving (raw attention)
        if args.save_pngs:
            print("\nDumping visualizations (this may take a while)...\n")
            out_root = os.path.join(args.out_folder, image_name)
            os.makedirs(out_root, exist_ok=True)

            def dump(tag: str, cache: Dict[int, torch.Tensor]):
                for li in layers:
                    if li not in cache:
                        continue
                    attn_map = cache[li]  # [H,T,S]
                    for h in heads_keep:
                        if h < 0 or h >= attn_map.shape[0]:
                            continue
                        for qlab, qidx in query_list:
                            if qidx >= attn_map.shape[1]:
                                continue

                            attn_vec = attn_map[h, qidx, :]  # [S]
                            num_patches = attn_vec.numel() - 1
                            reg_eff = reg_mask_patch if reg_mask_patch.numel() == num_patches else None

                            out_dir = os.path.join(out_root, tag, qlab)
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.png")

                            save_attention_overlay(
                                attn_vec,
                                orig_img,
                                out_path,
                                title=f"{tag} L{li} H{h} Q={qlab}",
                                query_token_idx=qidx,
                                reg_mask_patch=reg_eff,
                                use_mask=args.use_mask,
                            )

            dump("baseline", base_cache)

            if nuke_cache is not None:
                dump("nuked", nuke_cache)

                for li in layers:
                    if li not in base_cache or li not in nuke_cache:
                        continue
                    A = base_cache[li]
                    B = nuke_cache[li]
                    for h in heads_keep:
                        if h < 0 or h >= A.shape[0]:
                            continue
                        for qlab, qidx in query_list:
                            if qidx >= A.shape[1]:
                                continue
                            diff_vec = (A[h, qidx, :] - B[h, qidx, :]).abs()

                            out_dir = os.path.join(out_root, "diff_abs", qlab)
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.png")

                            save_attention_overlay(
                                diff_vec,
                                orig_img,
                                out_path,
                                title=f"|Δ| L{li} H{h} Q={qlab}",
                                query_token_idx=qidx,
                                reg_mask_patch=None,
                                use_mask=False,
                            )

        # Value-side attribution overlays + .npy raw grids
        if args.save_value:
            print("\nDumping VALUE-side contribution visualizations + data...\n")
            out_root = os.path.join(args.out_folder, image_name)
            os.makedirs(out_root, exist_ok=True)

            def dump_value(tag: str, pv_cache: Dict[int, Dict[str, torch.Tensor]]):
                for li in layers:
                    if li not in pv_cache:
                        continue
                    probs = pv_cache[li]["probs"]  # [H,T,S]
                    v = pv_cache[li]["v"]          # [H,S,D]
                    Hh, T, S = probs.shape

                    for h in heads_keep:
                        if h < 0 or h >= Hh:
                            continue
                        v_h = v[h]  # [S,D]
                        for qlab, qidx in query_list:
                            if qidx < 0 or qidx >= T:
                                continue

                            probs_h_q = probs[h, qidx, :]              # [S]
                            contrib_vec = value_contrib_vec_from_probs_v(probs_h_q, v_h)  # [S]

                            num_patches = int(contrib_vec.numel() - 1)
                            reg_eff = reg_mask_patch if reg_mask_patch.numel() == num_patches else None

                            out_dir = os.path.join(out_root, "value_contrib", tag, qlab)
                            os.makedirs(out_dir, exist_ok=True)

                            out_png = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.png")
                            out_npy = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.npy")

                            save_value_contrib_overlay_and_data(
                                contrib_vec=contrib_vec,
                                orig_image=orig_img,
                                out_png_path=out_png,
                                out_npy_path=out_npy,
                                title=f"VAL {tag} L{li} H{h} Q={qlab}  (||A*V|| per src)",
                                query_token_idx=qidx,
                                reg_mask_patch=reg_eff,
                                use_mask=args.use_mask,
                            )

            dump_value("baseline", base_pv_cache)

            if (nuke_pv_cache is not None) and (nuke_cache is not None):
                dump_value("nuked", nuke_pv_cache)

                for li in layers:
                    if li not in base_pv_cache or li not in nuke_pv_cache:
                        continue
                    probsA = base_pv_cache[li]["probs"]
                    vA = base_pv_cache[li]["v"]
                    probsB = nuke_pv_cache[li]["probs"]
                    vB = nuke_pv_cache[li]["v"]
                    Hh, T, S = probsA.shape

                    for h in heads_keep:
                        if h < 0 or h >= Hh:
                            continue
                        for qlab, qidx in query_list:
                            if qidx < 0 or qidx >= T:
                                continue

                            cA = value_contrib_vec_from_probs_v(probsA[h, qidx, :], vA[h])
                            cB = value_contrib_vec_from_probs_v(probsB[h, qidx, :], vB[h])
                            diff_vec = (cA - cB).abs()

                            num_patches = int(diff_vec.numel() - 1)
                            reg_eff = reg_mask_patch if reg_mask_patch.numel() == num_patches else None

                            out_dir = os.path.join(out_root, "value_contrib", "diff_abs", qlab)
                            os.makedirs(out_dir, exist_ok=True)
                            out_png = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.png")
                            out_npy = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.npy")

                            save_value_contrib_overlay_and_data(
                                contrib_vec=diff_vec,
                                orig_image=orig_img,
                                out_png_path=out_png,
                                out_npy_path=out_npy,
                                title=f"VAL |Δ| L{li} H{h} Q={qlab}  (||A*V|| per src)",
                                query_token_idx=qidx,
                                reg_mask_patch=reg_eff,
                                use_mask=False,
                            )

        # out_proj contribution visualizations
        if args.save_outproj:
            print("\nDumping OUT_PROJ contribution visualizations + data...\n")
            out_root = os.path.join(args.out_folder, image_name)
            os.makedirs(out_root, exist_ok=True)

            def get_Wo_cpu(layer_idx: int) -> torch.Tensor:
                blk = model.visual.transformer.resblocks[layer_idx]
                return blk.attn.out_proj.weight.detach().float().cpu()  # [E,E]

            def dump_outproj(tag: str, pv_cache: Dict[int, Dict[str, torch.Tensor]]):
                for li in layers:
                    if li not in pv_cache:
                        continue
                    probs = pv_cache[li]["probs"]  # [H,T,S]
                    v = pv_cache[li]["v"]          # [H,S,D]
                    W_o = get_Wo_cpu(li)           # [E,E]
                    Hh, T, S = probs.shape

                    # (1) per-head (slice) post-outproj maps
                    for h in heads_keep:
                        if h < 0 or h >= Hh:
                            continue
                        v_h = v[h]
                        for qlab, qidx in query_list:
                            if qidx < 0 or qidx >= T:
                                continue

                            probs_h_q = probs[h, qidx, :]
                            contrib = outproj_head_contrib_vec(probs_h_q, v_h, W_o, head_idx=h)

                            num_patches = int(contrib.numel() - 1)
                            reg_eff = reg_mask_patch if reg_mask_patch.numel() == num_patches else None

                            out_dir = os.path.join(out_root, "outproj_head", tag, qlab)
                            os.makedirs(out_dir, exist_ok=True)

                            out_png = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.png")
                            out_npy = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.npy")

                            save_attention_overlay(
                                contrib,
                                orig_img,
                                out_png,
                                title=f"OUTPROJ(head-slice) {tag} L{li} H{h} Q={qlab}",
                                query_token_idx=qidx,
                                reg_mask_patch=reg_eff,
                                use_mask=args.use_mask,
                            )
                            save_patch_grid_npy(contrib, out_npy)

                    # (2) full all-heads: BEFORE vs AFTER out_proj (combined)
                    for qlab, qidx in query_list:
                        if qidx < 0 or qidx >= T:
                            continue

                        probs_q = probs[:, qidx, :]  # [H,S]
                        pre, post = outproj_full_before_after_contrib(probs_q, v, W_o)

                        num_patches = int(pre.numel() - 1)
                        reg_eff = reg_mask_patch if reg_mask_patch.numel() == num_patches else None

                        out_dir = os.path.join(out_root, "outproj_full", tag, qlab)
                        os.makedirs(out_dir, exist_ok=True)

                        out_png = os.path.join(out_dir, f"layer{li:02d}_before_after.png")
                        pre_npy = os.path.join(out_dir, f"layer{li:02d}_pre.npy")
                        post_npy = os.path.join(out_dir, f"layer{li:02d}_post.npy")

                        save_before_after_combined_png(
                            pre_vec=pre,
                            post_vec=post,
                            orig_img=orig_img,
                            out_png_path=out_png,
                            query_token_idx=qidx,
                            reg_mask_patch=reg_eff,
                            use_mask=args.use_mask,
                            title=f"OUTPROJ full {tag} L{li} Q={qlab} | left=pre ||A*V||, right=post ||(A*V)W_o||",
                        )
                        save_patch_grid_npy(pre, pre_npy)
                        save_patch_grid_npy(post, post_npy)

            dump_outproj("baseline", base_pv_cache)

            if (nuke_pv_cache is not None) and (nuke_cache is not None):
                dump_outproj("nuked", nuke_pv_cache)

                for li in layers:
                    if li not in base_pv_cache or li not in nuke_pv_cache:
                        continue

                    probsA = base_pv_cache[li]["probs"]
                    vA = base_pv_cache[li]["v"]
                    probsB = nuke_pv_cache[li]["probs"]
                    vB = nuke_pv_cache[li]["v"]
                    W_o = model.visual.transformer.resblocks[li].attn.out_proj.weight.detach().float().cpu()

                    Hh, T, S = probsA.shape

                    # per-head slice diffs
                    for h in heads_keep:
                        if h < 0 or h >= Hh:
                            continue
                        for qlab, qidx in query_list:
                            if qidx < 0 or qidx >= T:
                                continue

                            cA = outproj_head_contrib_vec(probsA[h, qidx, :], vA[h], W_o, head_idx=h)
                            cB = outproj_head_contrib_vec(probsB[h, qidx, :], vB[h], W_o, head_idx=h)
                            diff = (cA - cB).abs()

                            num_patches = int(diff.numel() - 1)
                            reg_eff = reg_mask_patch if reg_mask_patch.numel() == num_patches else None

                            out_dir = os.path.join(out_root, "outproj_head", "diff_abs", qlab)
                            os.makedirs(out_dir, exist_ok=True)
                            out_png = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.png")
                            out_npy = os.path.join(out_dir, f"layer{li:02d}_head{h:02d}.npy")

                            save_attention_overlay(
                                diff, orig_img, out_png,
                                title=f"OUTPROJ(head-slice) |Δ| L{li} H{h} Q={qlab}",
                                query_token_idx=qidx,
                                reg_mask_patch=reg_eff,
                                use_mask=False,
                            )
                            save_patch_grid_npy(diff, out_npy)

                    # full before/after diffs (combined)
                    for qlab, qidx in query_list:
                        if qidx < 0 or qidx >= T:
                            continue

                        preA, postA = outproj_full_before_after_contrib(probsA[:, qidx, :], vA, W_o)
                        preB, postB = outproj_full_before_after_contrib(probsB[:, qidx, :], vB, W_o)

                        dpre = (preA - preB).abs()
                        dpost = (postA - postB).abs()

                        num_patches = int(dpre.numel() - 1)
                        reg_eff = reg_mask_patch if reg_mask_patch.numel() == num_patches else None

                        out_dir = os.path.join(out_root, "outproj_full", "diff_abs", qlab)
                        os.makedirs(out_dir, exist_ok=True)

                        out_png = os.path.join(out_dir, f"layer{li:02d}_before_after.png")
                        pre_npy = os.path.join(out_dir, f"layer{li:02d}_pre.npy")
                        post_npy = os.path.join(out_dir, f"layer{li:02d}_post.npy")

                        save_before_after_combined_png(
                            pre_vec=dpre,
                            post_vec=dpost,
                            orig_img=orig_img,
                            out_png_path=out_png,
                            query_token_idx=qidx,
                            reg_mask_patch=reg_eff,
                            use_mask=False,
                            title=f"OUTPROJ full |Δ| L{li} Q={qlab} | left=|Δ pre|, right=|Δ post|",
                        )
                        save_patch_grid_npy(dpre, pre_npy)
                        save_patch_grid_npy(dpost, post_npy)

        if args.save_pngs or args.save_value or args.save_outproj:
            print(f"[DONE] {image_name} -> {args.out_folder}\n")

    print("All done.")


if __name__ == "__main__":
    main()