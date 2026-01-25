"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

CLIP Patch-Norm + Attention Visualization (Unified)
Adapted from: https://github.com/hila-chefer/Transformer-MM-Explainability

Attention visualization: gradient-weighted attention rollout (image + text)

This script computes token-level relevance maps for CLIP-style transformers using a
Transformer-Interpretability / Grad-CAM-like variant applied to attention matrices.

Core idea:
- Build a scalar target = the (image, text) matching logit for each pair in the batch.
- For each transformer block (starting at start_layer / start_layer_text), take the
  attention probabilities A and their gradients d(target)/dA.
- Form a per-layer attribution map by elementwise multiplying (dA * A), clamping to
  positive contributions, and averaging over heads.
- Roll these per-layer maps through the network via attention rollout:
      R <- R + CAM @ R
  starting from the identity matrix, so relevance propagates from output tokens back
  through intermediate attentions.

Interpretation:
These maps explain which tokens are influential under this gradient-based attribution
scheme; they are not guaranteed to equal "true" causal importance, but they are a
useful diagnostic for where the model routes information.

See also: eval-visualize-attn-vit-all-blocks-qv.py

"""

from __future__ import annotations
import os
import gc
import re
import glob
import math
import argparse
from typing import List, Tuple, Optional
from contextlib import contextmanager
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

import attnclip as clip
from attnclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


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


device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FOLDER = "out_vis_attn/attn_patch_gradw_rollout"

# Attention overlay controls
ATTN_TRANSPARENT_BELOW = 0.12
ATTN_ALPHA_MAX = 0.85
ATTN_ALPHA_GAMMA = 0.8

# Background muting
BG_MUTE = 0.30
BG_MUTE_TO = 0.0

# Text canvas (draw prompt above image)
PROMPT_CANVAS_PAD_X = 10
PROMPT_CANVAS_PAD_Y = 8
PROMPT_CANVAS_BG = (255, 255, 255)
PROMPT_TEXT_COLOR = (0, 0, 0)
PROMPT_FONT_SIZE = 20
PROMPT_CANVAS_FIXED_H = 36

# Fonts to try for overlaying text prompt with visualization
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Unified CLIP patch norms + attention visualization")
    parser.add_argument("--mode", type=str, default="both", choices=["patch", "attn", "both"])
    parser.add_argument("--image_folder", type=str, default="image_sets/attn_bench_images")
    parser.add_argument("--token_folder", type=str, default="image_sets/attn_bench_texts")
    # also interesting:
    #parser.add_argument("--image_folder", type=str, default="image_sets/synpeople_images")
    #parser.add_argument("--token_folder", type=str, default="image_sets/synpeople_texts")
    parser.add_argument("--start_layers", type=str, default="22,23", help="Comma-separated list for ViT start_layer.")
    parser.add_argument("--start_layers_text", type=str, default="-1", help="Comma-separated list for text start_layer_text")
    return parser.parse_args()



def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sanitize_for_filename(s: str, max_len: int = 120) -> str:
    # return type + removed stray annotation
    s = s.strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]+', ' ', s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_")
    return s[:max_len] if len(s) > max_len else s

def get_font(font_size: int = PROMPT_FONT_SIZE) -> ImageFont.FreeTypeFont:
    for fp in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            pass
    return ImageFont.load_default()

def _wrap_prompt_lines(draw: ImageDraw.ImageDraw, prompt: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    # helper for wrapping using width only
    words = prompt.split(" ")
    lines: List[str] = []
    cur = ""
    for w in words:
        nxt = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), nxt, font=font)
        if bbox[2] <= max_w or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def _font_line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    # stable per-font line height (not glyph-dependent)
    try:
        ascent, descent = font.getmetrics()
        lh = int(ascent + descent)
        return max(1, lh)
    except Exception:
        bbox = draw.textbbox((0, 0), "Ag", font=font)
        return max(1, int(bbox[3] - bbox[1]))

def add_prompt_canvas_above(img_rgb: Image.Image, prompt: str) -> Image.Image:
    # fixed canvas height + stable line height + auto-shrink to fit
    canvas_h = PROMPT_CANVAS_FIXED_H

    draw_dummy = ImageDraw.Draw(Image.new("RGB", (10, 10), PROMPT_CANVAS_BG))
    max_w = img_rgb.width - 2 * PROMPT_CANVAS_PAD_X
    max_text_h = canvas_h - 2 * PROMPT_CANVAS_PAD_Y

    # Try to keep PROMPT_FONT_SIZE, shrink only if needed to fit into fixed canvas.
    font_size = PROMPT_FONT_SIZE
    while True:
        font = get_font(font_size)
        lines = _wrap_prompt_lines(draw_dummy, prompt, font, max_w=max_w)

        line_h = _font_line_height(draw_dummy, font)
        line_gap = max(2, int(round(0.15 * line_h)))  # mild spacing; tweak if you like
        text_h = len(lines) * line_h + max(0, (len(lines) - 1)) * line_gap

        if text_h <= max_text_h or font_size <= 10:
            break
        font_size -= 1  # shrink until it fits

    out = Image.new("RGB", (img_rgb.width, img_rgb.height + canvas_h), PROMPT_CANVAS_BG)
    out.paste(img_rgb, (0, canvas_h))

    d = ImageDraw.Draw(out)

    # vertically center the text block inside the fixed canvas (but respect padding)
    y = max(PROMPT_CANVAS_PAD_Y, (canvas_h - text_h) // 2)

    for ln in lines:
        d.text((PROMPT_CANVAS_PAD_X, y), ln, font=font, fill=PROMPT_TEXT_COLOR)
        y += line_h + line_gap

    return out

def resolve_layer_idx(layer: int, n_layers: int) -> int:
    if layer < 0:
        layer = n_layers + layer
    layer = max(0, min(n_layers - 1, layer))
    return layer

def fmt_layer_tag(prefix: str, layer: int) -> str:
    return f"{prefix}{layer}"

def vit_patch_grid_from_model_and_image(model, image_tensor: torch.Tensor) -> Tuple[int, int, int]:
    H = int(image_tensor.shape[-2])
    patch_size = None
    if hasattr(model, "visual") and hasattr(model.visual, "conv1") and hasattr(model.visual.conv1, "kernel_size"):
        patch_size = int(model.visual.conv1.kernel_size[0])
    if patch_size is None or patch_size <= 0:
        if hasattr(model.visual, "positional_embedding"):
            seq = int(model.visual.positional_embedding.shape[0])
            num_patches = seq - 1
            grid = int(round(math.sqrt(num_patches)))
            patch_size = H // grid if grid > 0 else 16
        else:
            patch_size = 16
    grid = H // patch_size
    num_patches = grid * grid
    return patch_size, grid, num_patches


# PATCH NORM PIPELINE
def get_all_layer_outputs(transformer, x: torch.Tensor) -> List[torch.Tensor]:
    all_outputs: List[torch.Tensor] = []

    def hook_fn(module, input, output):
        # Store on CPU to avoid VRAM growth / fragmentation across many images/models.
        # No clone needed: detach is enough, then move off GPU.
        all_outputs.append(output.detach().to("cpu"))

    hooks = []
    for block in transformer.resblocks:
        hooks.append(block.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = transformer(x)

    for h in hooks:
        h.remove()

    return all_outputs


def clip_encode_image_all_layers(modelorg, image_input: torch.Tensor) -> List[torch.Tensor]:
    with torch.no_grad():
        x = modelorg.visual.conv1(image_input)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = modelorg.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + modelorg.visual.positional_embedding.to(x.dtype)
        x = modelorg.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        all_layer_outputs = get_all_layer_outputs(modelorg.visual.transformer, x)
        # outputs to CPU
        all_layer_outputs = [layer_out.permute(1, 0, 2) for layer_out in all_layer_outputs]
        return all_layer_outputs

def save_patchnorm_heatmap_with_strip(
    norms: np.ndarray,
    image_name: str,
    layer_idx: int,
    out_path: str,
    grid: int,
    num_patches: int,
) -> None:
    patch_norms = norms[:num_patches].reshape(grid, grid)
    extra = norms[num_patches:]  # [CLS] + extras (if any)

    vmin, vmax = float(norms.min()), float(norms.max())
    fig, ax = plt.subplots(figsize=(6, 7))

    im = ax.imshow(patch_norms, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("L2 Norm")
    ax.set_title(f"Layer {layer_idx}: Patch Norm Heatmap: {image_name}")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(bottom=0.2)
    ax_extra = fig.add_axes([0.1, 0.05, 0.8, 0.1])
    extra_row = np.array(extra).reshape(1, -1) if len(extra) else np.zeros((1, 0), dtype=np.float32)
    ax_extra.imshow(extra_row, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)

    labels = []
    if len(extra) > 0:
        labels.append("CLS")
        extras_count = len(extra) - 1
        if extras_count == 5:
            labels += ["REG1", "REG2", "REG3", "REG4", "FUSED"]
        elif extras_count == 4:
            labels += ["REG1", "REG2", "REG3", "REG4"]
        elif extras_count > 0:
            labels += [f"EX{i+1}" for i in range(extras_count)]

    if labels:
        ax_extra.set_xticks(range(len(labels)))
        ax_extra.set_xticklabels(labels, rotation=0, fontsize=10, fontweight="bold")
    ax_extra.set_yticks([])
    ax_extra.set_title("CLS & Extra Token Norms", fontsize=12, pad=10)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_patchnorm_barplot(norms: np.ndarray, image_name: str, layer_idx: int, out_path: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(norms)), norms)
    plt.xlabel("Token ID (patches first, then CLS, then extras)")
    plt.ylabel("L2 Norm")
    plt.title(f"Layer {layer_idx}: L2 Norms of ViT Image Tokens: {image_name}")
    plt.savefig(out_path)
    plt.close()

def run_patch_norms(
    model,
    model_alias,
    preprocess,
    image_folder: str,
    out_patch_dir: str,
    layers_to_save: Optional[List[int]] = None,   # cap to layer list
) -> None:
    ensure_dir(out_patch_dir)

    csv_records = []
    img_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    img_files = [p for p in img_files if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))]

    for img_path in img_files:
        img_file = os.path.basename(img_path)
        image_name = os.path.splitext(img_file)[0]

        image_pil = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

        _, grid, num_patches = vit_patch_grid_from_model_and_image(model, image_tensor)
        all_layer_outputs = clip_encode_image_all_layers(model, image_tensor)
        n_layers = len(all_layer_outputs)

        # choose subset of layers (resolved, de-duped, order-preserving)
        if layers_to_save is None or len(layers_to_save) == 0:
            layer_indices = list(range(n_layers))
        else:
            seen = set()
            layer_indices = []
            for L in layers_to_save:
                idx = resolve_layer_idx(L, n_layers)
                if idx not in seen:
                    seen.add(idx)
                    layer_indices.append(idx)

        for layer_idx in layer_indices:
            layer_output = all_layer_outputs[layer_idx]
            x = layer_output[0]  # (seq, dim)

            cls_token = x[0:1]
            patch_tokens = x[1:1 + num_patches]
            extra_tokens = x[1 + num_patches:]

            patch_norms = torch.norm(patch_tokens, dim=-1)
            cls_norm = torch.norm(cls_token, dim=-1)
            extra_norms = torch.norm(extra_tokens, dim=-1) if extra_tokens.numel() else torch.empty((0,), device=patch_norms.device)

            norms = torch.cat([patch_norms, cls_norm, extra_norms], dim=0).detach().cpu().numpy()

            heatmap_path = os.path.join(out_patch_dir, f"{model_alias}_layer{layer_idx:02d}_heatmap_{image_name}.png")
            barplot_path = os.path.join(out_patch_dir, f"{model_alias}_layer{layer_idx:02d}_l2norm_{image_name}.png")

            save_patchnorm_heatmap_with_strip(
                norms=norms,
                image_name=image_name,
                layer_idx=layer_idx,
                out_path=heatmap_path,
                grid=grid,
                num_patches=num_patches,
            )
            save_patchnorm_barplot(norms=norms, image_name=image_name, layer_idx=layer_idx, out_path=barplot_path)

            for idx, norm in enumerate(norms):
                flag = "CLS" if idx == num_patches else ""
                csv_records.append([layer_idx, idx, float(norm), flag, image_name])

    df = pd.DataFrame(csv_records, columns=["Layer", "TokenID", "Norm", "Flag", "Image"])
    df.to_csv(os.path.join(out_patch_dir, "patch_norms_layers.csv"), index=False)
    print(f"[patch] Wrote patch norms to: {out_patch_dir}")


# ATTENTION PIPELINE
def interpret(
    image: torch.Tensor,
    texts: torch.Tensor,
    model,
    device: str,
    start_layer: int,
    start_layer_text: int,
):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)

    logits_per_image, _ = model(images, texts)
    index = [i for i in range(batch_size)]

    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).to(device)
    one_hot = torch.sum(one_hot * logits_per_image)

    # set_to_none=True drops grad buffers instead of zeroing in-place
    model.zero_grad(set_to_none=True)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    n_vit = len(image_attn_blocks)
    start_layer_i = resolve_layer_idx(start_layer, n_vit)

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype, device=device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer_i:
            continue

        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True, create_graph=False)[0]
        cam = blk.attn_probs

        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])

        cam = (grad * cam).reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)

        R = R + torch.bmm(cam, R)

    image_relevance = R[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
    n_txt = len(text_attn_blocks)
    start_layer_t = resolve_layer_idx(start_layer_text, n_txt)

    num_tokens_t = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens_t, num_tokens_t, dtype=text_attn_blocks[0].attn_probs.dtype, device=device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens_t, num_tokens_t)

    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_t:
            continue

        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True, create_graph=False)[0]
        cam = blk.attn_probs

        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])

        cam = (grad * cam).reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)

        R_text = R_text + torch.bmm(cam, R_text)

    # Drop graph roots explicitly (helps allocator behavior across many calls)
    del logits_per_image, one_hot, images
    model.zero_grad(set_to_none=True)

    return R_text, image_relevance


def relevance_vec_to_mask(image_relevance_vec: torch.Tensor, grid: int, out_size: int = 224) -> np.ndarray:
    num_patches = grid * grid
    v = image_relevance_vec[:num_patches]
    v = v.reshape(1, 1, grid, grid)
    v = torch.nn.functional.interpolate(v, size=out_size, mode="bilinear", align_corners=False)
    v = v.reshape(out_size, out_size).detach().cpu().numpy()
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v

def blend_heatmap_on_rgb_background(
    background_rgb_float: np.ndarray,
    mask01: np.ndarray,
    *,
    alpha_max: float,
    transparent_below: float,
    alpha_gamma: float,
    bg_mute: float,
    bg_mute_to: float,
) -> np.ndarray:
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * mask01), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha = np.clip((mask01 - transparent_below) / max(1e-6, (1.0 - transparent_below)), 0.0, 1.0)
    if alpha_gamma != 1.0:
        alpha = alpha ** alpha_gamma
    alpha = alpha * alpha_max
    alpha_3 = alpha[..., None]

    bg = background_rgb_float.copy()
    if bg_mute > 0.0:
        bg = (1.0 - bg_mute) * bg + bg_mute * bg_mute_to

    out = (1.0 - alpha_3) * bg + alpha_3 * heatmap_rgb
    out = np.clip(out, 0.0, 1.0)
    return np.uint8(255 * out)

def patch_norm_matrix_rgb_background(patch_norms: np.ndarray, patch_size: int) -> np.ndarray:
    import matplotlib.cm as cm
    vir = cm.get_cmap("viridis")

    v = patch_norms.astype(np.float32)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    rgb = vir(v)[..., :3].astype(np.float32)
    up = np.kron(rgb, np.ones((patch_size, patch_size, 1), dtype=np.float32))
    return up

def patch_norm_matrix_rgb_background_both(
    patch_norms: np.ndarray,
    patch_size: int,
    cmap_name: str = "gray",          # keep background hue-neutral for "both"
    clip_lo: float = 2.0,             # low percentile for floor
    typ_hi: float = 95.0,             # "top of normal" percentile
    white_hi: float = 99.7,           # outliers above this saturate to white
    floor: float = 0.08,              # darkest gray (avoid pure black)
    mid: float = 0.55,                # brightest "normal" gray (pull normals darker)
    gamma_norm: float = 1.8,          # >1 darkens normals, makes them mushier
    gamma_out: float = 0.7,           # <1 expands the near-top region (optional)
    compress: str = "none",           # "none" | "log1p" | "asinh"
) -> np.ndarray:
    """
    Background for 'both' mode:
      - normal range -> compressed into [floor, mid] (darker, mushy)
      - outliers -> ramp from mid -> 1 and saturate to white
    """
    import matplotlib.cm as cm

    p = patch_norms.astype(np.float32)

    # optional mild compression
    if compress == "log1p":
        p = np.log1p(p)
    elif compress == "asinh":
        scale = np.percentile(p, 50.0) + 1e-6
        p = np.arcsinh(p / scale)

    # percentile anchors
    vmin = np.percentile(p, clip_lo)
    vtyp = np.percentile(p, typ_hi)
    vwhite = np.percentile(p, white_hi)

    # safety
    if vtyp <= vmin:
        vtyp = vmin + 1e-6
    if vwhite <= vtyp:
        vwhite = vtyp + 1e-6

    # --- piecewise mapping ---
    # region A: [vmin, vtyp] -> [floor, mid] (dark/mushy)
    a = (p - vmin) / (vtyp - vmin)
    a = np.clip(a, 0.0, 1.0)
    a = a ** gamma_norm
    yA = floor + (mid - floor) * a

    # region B: (vtyp, vwhite] -> (mid, 1] (outliers get bright fast)
    b = (p - vtyp) / (vwhite - vtyp)
    b = np.clip(b, 0.0, 1.0)
    b = b ** gamma_out
    yB = mid + (1.0 - mid) * b

    # select region based on threshold
    y = np.where(p <= vtyp, yA, yB)

    # hard saturate anything above vwhite to full white
    y = np.where(p >= vwhite, 1.0, y)

    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(y)[..., :3].astype(np.float32)  # (grid,grid,3)

    up = np.kron(rgb, np.ones((patch_size, patch_size, 1), dtype=np.float32))
    return up


def run_attention_and_optional_overlay(
    model,
    model_alias,
    preprocess,
    image_folder: str,
    token_folder: str,
    out_attn_dir: str,
    out_both_dir: Optional[str],
    start_layers: List[int],
    start_layers_text: List[int],
) -> None:
    ensure_dir(out_attn_dir)
    if out_both_dir is not None:
        ensure_dir(out_both_dir)

    img_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    img_files = [p for p in img_files if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))]

    for img_path in img_files:
        img_file = os.path.basename(img_path)
        img_name = os.path.splitext(img_file)[0]

        token_file = os.path.join(token_folder, f"tokens_{img_name}.txt")
        if not os.path.isfile(token_file):
            print(f"[attn] Missing token file (skipping image): {token_file}")
            continue

        with open(token_file, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f.read().splitlines() if line.strip()]

        image_pil = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(image_pil).unsqueeze(0).to(device)

        patch_size, grid, num_patches = vit_patch_grid_from_model_and_image(model, img_tensor)

        all_layer_outputs = None
        if out_both_dir is not None:
            all_layer_outputs = clip_encode_image_all_layers(model, img_tensor)

        for token in tokens:
            texts = clip.tokenize([token]).to(device)

            for v_layer in start_layers:
                for t_layer in start_layers_text:
                    _, R_image = interpret(
                        model=model,
                        image=img_tensor,
                        texts=texts,
                        device=device,
                        start_layer=v_layer,
                        start_layer_text=t_layer,
                    )

                    mask01 = relevance_vec_to_mask(R_image[0], grid=grid, out_size=img_tensor.shape[-1])

                    bg = img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
                    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

                    vis_rgb_u8 = blend_heatmap_on_rgb_background(
                        background_rgb_float=bg,
                        mask01=mask01,
                        alpha_max=ATTN_ALPHA_MAX,
                        transparent_below=ATTN_TRANSPARENT_BELOW,
                        alpha_gamma=ATTN_ALPHA_GAMMA,
                        bg_mute=BG_MUTE,
                        bg_mute_to=BG_MUTE_TO,
                    )
                    vis_img = Image.fromarray(vis_rgb_u8, mode="RGB")
                    vis_img = add_prompt_canvas_above(vis_img, token)

                    token_safe = sanitize_for_filename(token)
                    v_tag = fmt_layer_tag("V", v_layer)
                    t_tag = fmt_layer_tag("T", t_layer)
                    vt_tag = f"{v_tag}_{t_tag}"
                    out_name = f"{model_alias}_{vt_tag}__{token_safe}_{img_name}.png"
                    vis_img.save(os.path.join(out_attn_dir, out_name))

                    if out_both_dir is not None and all_layer_outputs is not None:
                        v_idx = resolve_layer_idx(v_layer, len(all_layer_outputs))
                        layer_out = all_layer_outputs[v_idx][0]

                        patch_tokens = layer_out[1:1 + num_patches]
                        patch_norms = torch.norm(patch_tokens, dim=-1).detach().cpu().numpy().reshape(grid, grid)

                        patch_bg = patch_norm_matrix_rgb_background_both(
                            patch_norms,
                            patch_size=patch_size,
                            cmap_name="gray",
                            clip_lo=2.0,
                            typ_hi=95.0,
                            white_hi=99.7,     # raise to 99.9 if too many whites
                            floor=0.10,
                            mid=0.55,          # lower -> darker overall
                            gamma_norm=2.0,    # higher -> mushier normals
                            gamma_out=0.6,     # lower -> outliers brighten faster
                            compress="none",
                        )


                        H = int(img_tensor.shape[-2])
                        W = int(img_tensor.shape[-1])
                        if patch_bg.shape[0] != H or patch_bg.shape[1] != W:
                            patch_bg_img = Image.fromarray(np.uint8(255 * patch_bg), mode="RGB")
                            patch_bg_img = patch_bg_img.resize((W, H), resample=Image.NEAREST)
                            patch_bg = np.asarray(patch_bg_img).astype(np.float32) / 255.0

                        vis_on_patch_u8 = blend_heatmap_on_rgb_background(
                            background_rgb_float=patch_bg,
                            mask01=mask01,
                            alpha_max=0.9,
                            transparent_below=ATTN_TRANSPARENT_BELOW,
                            alpha_gamma=ATTN_ALPHA_GAMMA,
                            bg_mute=0.0,
                            bg_mute_to=0.0,
                        )
                        vis_on_patch = Image.fromarray(vis_on_patch_u8, mode="RGB")
                        vis_on_patch = add_prompt_canvas_above(vis_on_patch, token)
                        vis_on_patch.save(os.path.join(out_both_dir, out_name))

    print(f"[attn] Wrote attention heatmaps to: {out_attn_dir}")
    if out_both_dir is not None:
        print(f"[both] Wrote attn-on-patch overlays to: {out_both_dir}")


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]

def main():
    args = parse_args()

    mode = args.mode.lower()
    image_folder = args.image_folder
    token_folder = args.token_folder

    start_layers = parse_int_list(args.start_layers)
    start_layers_text = parse_int_list(args.start_layers_text)

    base_out = OUTPUT_FOLDER

    for model_alias, model_name_or_path in MODELS:
        print("=" * 70)
        print(f"Model: {model_alias} -> {model_name_or_path}")
        print("=" * 70)

        model, preprocess, _ = load_openai_clip_anything(clip, model_name_or_path, device=device, jit=False, strict=True)
        model = model.float()

        out_model_dir = os.path.join(base_out, model_alias)
        ensure_dir(out_model_dir)

        out_patch_dir = os.path.join(out_model_dir, "patch")
        out_attn_dir  = os.path.join(out_model_dir, "attn")
        out_both_dir  = os.path.join(out_model_dir, "both")

        if mode == "patch":
            run_patch_norms(
                model,
                model_alias,
                preprocess,
                image_folder=image_folder,
                out_patch_dir=out_patch_dir,
                layers_to_save=start_layers,
            )

        elif mode == "attn":
            run_attention_and_optional_overlay(
                model,
                model_alias,
                preprocess,
                image_folder=image_folder,
                token_folder=token_folder,
                out_attn_dir=out_attn_dir,
                out_both_dir=None,
                start_layers=start_layers,
                start_layers_text=start_layers_text,
            )

        elif mode == "both":
            run_patch_norms(
                model,
                model_alias,
                preprocess,
                image_folder=image_folder,
                out_patch_dir=out_patch_dir,
                layers_to_save=start_layers,
            )
            run_attention_and_optional_overlay(
                model,
                model_alias,
                preprocess,
                image_folder=image_folder,
                token_folder=token_folder,
                out_attn_dir=out_attn_dir,
                out_both_dir=out_both_dir,
                start_layers=start_layers,
                start_layers_text=start_layers_text,
            )

        del model        
        gc.collect()

    print("\nAll done.")

if __name__ == "__main__":
    main()