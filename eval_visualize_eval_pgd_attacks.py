"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

 Multi-attack targeted adversarial suite for CLIP-like models:
 Example scenario:
 Turn goldfinch images into "bumblebee" under 2-way CLIP decision.

 Implements (each separately, saved into per-attack subfolders):
   1) linf_pgd           (baseline targeted PGD, L∞)
   2) l2_pgd             (targeted PGD, L2)
   3) eot_linf_pgd       (EOT-PGD with random resize/crop/pad)
   4) ditimi_linf_pgd    (DI+TI+MI iterative attack)
   5) fft_low_linf_pgd   (L∞ PGD with low-frequency FFT filter)
   6) fft_high_linf_pgd  (L∞ PGD with high-frequency FFT filter)
   7) patch_attack       (optimized patch overlay; localized attack)

 Saves milestone images for "subtle vs mush" (cos sim):
   clean, near_flip, flip, solid, final

 Prints an end-of-run summary for each (attack, model).

"""
from __future__ import annotations

import os
import json
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image

import attnclipindiv as clip
from attnclipindiv.model import CLIP
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything


# ============================================================
# Models: OpenAI / local path .pt .safetensors / HuggingFace Hub
# ============================================================

# Multi-model list with alias
MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
    ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
    ("regress-norm", "zer0int/CLIP-Regression-ViT-L-14"),
    ("regress-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
]

# Images to attack
FOLDER: str = "image_sets/n01531178"
ONLY: str = "n01531178_125.JPEG,n01531178_232.JPEG"  # comma-separated; use "*" for all in folder
OUT_DIR: str = "out_vis_perturb_attacks"

# 2-way prompts
PROMPT_SOURCE: str = "a bird"
PROMPT_TARGET: str = "a bumblebee"

STEPS: int = 200
ALPHA: Optional[float] = None   # None => per-attack default
SAVE_EVERY: int = 50            # save an image every N steps <-- consider setting 20 for deepdream

# Choose which to run (comment out what you don't want)
ATTACKS_TO_RUN: List[str] = [
    "linf_pgd",
    "l2_pgd",
    "eot_linf_pgd",
    "ditimi_linf_pgd",
    "fft_low_linf_pgd",
    "fft_high_linf_pgd",
    "patch_attack",
    #"deepdream", # routes to eot_linf_pgd_attack with DREAM_EPS_LIST
]

# FFT params
FFT_CUTOFF: float = 0.3                 # Non-deepdream

# Deepdream-only params 
DREAM_FFT_CUTOFF: float = 0.8           # lower => smoother / less high-freq (very low: color blur blob)
DREAM_FFT_EVERY: int = 4                # curb high-frequency noise every n steps
DREAM_CROP_SIZE: int = 224              # feed model at 224 always
DREAM_CROPS_PER_STEP: int = 4           # more views, more compute, more quality. set to 1 for 'fast'.
DREAM_CENTER_BIAS: float = 0.6          # 0=uniform crop (no 'edge'), higher => more center-heavy
DREAM_CANVAS_SIZE: int = 336            # Larger image for deepdream (optimizes over crops of 224,224)
DREAM_EPS_LIST: List[float] = [1024]    # extreme for deepdream



# Epsilon schedules in *pixel-value* units (0..255), then converted to [0..1] internally.
#DEFAULT_EPS_LIST_255: List[float] = [1, 2, 4, 8, 16, 32, 48, 64]
DEFAULT_EPS_LIST_255: List[float] = [4, 32]

def eps255_to_pix(eps_255: float) -> float:
    return float(eps_255) / 255.0

# EOT params
EOT_SAMPLES: int = 4
EOT_SCALE_MIN: float = 0.90
EOT_SCALE_MAX: float = 1.10

# DI+TI+MI params
DI_PROB: float = 0.7
DI_SCALE_MIN: float = 0.90
DI_SCALE_MAX: float = 1.10
TI_KS: int = 15
TI_SIGMA: float = 3.0
MOMENTUM: float = 0.9

# Milestones
NEAR_FLIP_P_TARGET: float = 0.45
SOLID_P_TARGET: float = 0.90

# Patch params (only for patch attacks)
PATCH_FRAC: float = 0.20
PATCH_LOC: str = "center"  # "center" | "topleft" | "random"
PATCH_STEP_SIZE: float = 0.05

# Fixed logit_scale (no logit_scale weaseling)
FIXED_LOGIT_SCALE: float = 100.0

# Runtime
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
SEED: int = 123

# Supported attack registry
ATTACK_NAMES: List[str] = [
    "linf_pgd",
    "l2_pgd",
    "eot_linf_pgd",
    "ditimi_linf_pgd",
    "fft_low_linf_pgd",
    "fft_high_linf_pgd",
    "patch_attack",
    "deepdream",
]

def load_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    out = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(exts):
            out.append(os.path.join(folder, fn))
    return out


def fix_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Normalization constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)


def _inv_normalize(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # x_norm: [1,3,H,W] normalized -> [0,1] approx
    return (x_norm * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1))


def _normalize(x_pix: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # x_pix: [1,3,H,W] in [0,1]
    return (x_pix - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def cast_imgs_to_visual_dtype(x_norm: torch.Tensor, visual: torch.nn.Module) -> torch.Tensor:
    w = visual.conv1.weight
    if x_norm.dtype != w.dtype:
        x_norm = x_norm.to(dtype=w.dtype)
    return x_norm

@torch.no_grad()
def clip_two_way_probs_from_pix_multicrop(
    model: CLIP,
    x_pix: torch.Tensor,
    tok_2: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    crop_size: int,
    crops_per_step: int,
    center_bias: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deepdream-safe evaluation: average logits over multiple differentiable crops.
    Returns probs/logits on CPU like clip_two_way_probs_from_pix().
    """
    model.eval()

    x_pix = x_pix.clamp(0, 1)

    # text features once
    txt_feat = model.encode_text(tok_2)
    txt_feat = F.normalize(txt_feat, dim=-1)

    logits_sum = None
    K = int(max(1, crops_per_step))

    for _ in range(K):
        x_crop = _rand_crop_224(x_pix, crop_size=int(crop_size), center_bias=float(center_bias))
        x_norm = _normalize(x_crop, mean, std)
        x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

        img_feat = model.encode_image(x_norm)
        img_feat = F.normalize(img_feat, dim=-1)

        logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)  # [2]
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    logits_avg = logits_sum / float(K)
    probs = logits_avg.softmax(dim=-1)

    return probs.detach().cpu(), logits_avg.detach().cpu()


def _rand_crop_224(
    x_pix: torch.Tensor,
    crop_size: int,
    center_bias: float = 0.0,
) -> torch.Tensor:
    """
    Differentiable crop via tensor slicing.
    center_bias in [0,1-ish]: pushes crops toward center (more grad in center).
    """
    _, _, H, W = x_pix.shape
    cs = int(crop_size)
    if H < cs or W < cs:
        # if input smaller, resize up (still differentiable)
        x_pix = F.interpolate(x_pix, size=(max(H, cs), max(W, cs)), mode="bilinear", align_corners=False)
        _, _, H, W = x_pix.shape

    max_top = H - cs
    max_left = W - cs

    if max_top == 0:
        top = 0
    else:
        if center_bias > 0:
            # sample around center using a normal-ish distribution
            mu = max_top / 2.0
            sigma = max(1e-6, (max_top / 2.0) * (1.0 - min(center_bias, 0.999)))
            top_f = torch.randn((), device=x_pix.device) * sigma + mu
            top = int(torch.clamp(top_f.round(), 0, max_top).item())
        else:
            top = int(torch.randint(0, max_top + 1, (1,), device=x_pix.device).item())

    if max_left == 0:
        left = 0
    else:
        if center_bias > 0:
            mu = max_left / 2.0
            sigma = max(1e-6, (max_left / 2.0) * (1.0 - min(center_bias, 0.999)))
            left_f = torch.randn((), device=x_pix.device) * sigma + mu
            left = int(torch.clamp(left_f.round(), 0, max_left).item())
        else:
            left = int(torch.randint(0, max_left + 1, (1,), device=x_pix.device).item())

    return x_pix[:, :, top:top + cs, left:left + cs]


def Toggle_Deepdream_Canvas(x_pix: torch.Tensor, size: int) -> torch.Tensor:
    # helper to build deepdream canvas
    if int(size) <= 0:
        return x_pix
    return F.interpolate(x_pix, size=(int(size), int(size)), mode="bilinear", align_corners=False)


@torch.no_grad()
def clip_two_way_probs_from_pix(
    model: CLIP,
    x_pix: torch.Tensor,
    tok_2: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      probs: [2] softmax over [source, target]
      logits: [2] raw (cosine * FIXED_LOGIT_SCALE)
    """
    model.eval()

    x_pix = x_pix.clamp(0, 1)
    x_norm = _normalize(x_pix, mean, std)
    x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

    img_feat = model.encode_image(x_norm)
    img_feat = F.normalize(img_feat, dim=-1)

    txt_feat = model.encode_text(tok_2)
    txt_feat = F.normalize(txt_feat, dim=-1)

    logits = (img_feat @ txt_feat.T).squeeze(0)  # [2]
    logits = logits * float(FIXED_LOGIT_SCALE)

    probs = logits.softmax(dim=-1)
    return probs.detach().cpu(), logits.detach().cpu()


def _save_pix_image(
    x_pix: torch.Tensor,
    out_path: str,
) -> None:
    x = x_pix.detach().clamp(0, 1)
    im = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(im).save(out_path)


def _milestone_filename(
    tag: str,
    attack: str,
    eps_pix: float,
    step_idx: int,
    name: str,
    p_t: float,
    margin: float,
) -> str:
    return f"{tag}__{attack}__eps{eps_pix:.6f}__step{step_idx:04d}__{name}__pT{p_t:.3f}__m{margin:+.3f}.png"


def _gaussian_kernel_2d(ks: int, sigma: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    assert ks % 2 == 1
    ax = torch.arange(-(ks // 2), ks // 2 + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k = k / (k.sum() + 1e-12)
    return k


def _ti_smooth_grad(grad: torch.Tensor, ks: int = 15, sigma: float = 3.0) -> torch.Tensor:
    # grad: [1,3,H,W]
    device = str(grad.device)
    dtype = grad.dtype
    k2 = _gaussian_kernel_2d(ks, sigma, device=device, dtype=dtype)  # [ks,ks]
    k = k2.view(1, 1, ks, ks).repeat(3, 1, 1, 1)  # [3,1,ks,ks]
    return F.conv2d(grad, k, padding=ks // 2, groups=3)


def _random_resize_pad_crop(
    x_pix: torch.Tensor,
    out_hw: Tuple[int, int],
    scale_min: float,
    scale_max: float,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """
    Differentiable random resize + pad + random crop back to out_hw.
    x_pix: [1,3,H,W] in [0,1]
    """
    B, C, H, W = x_pix.shape
    oh, ow = out_hw

    s = float(torch.empty(1).uniform_(scale_min, scale_max).item())
    nh = max(1, int(round(H * s)))
    nw = max(1, int(round(W * s)))

    x = F.interpolate(x_pix, size=(nh, nw), mode="bilinear", align_corners=False)

    # pad to at least (oh,ow)
    ph = max(0, oh - nh)
    pw = max(0, ow - nw)
    if ph > 0 or pw > 0:
        pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]  # left,right,top,bottom
        x = F.pad(x, pad, mode=pad_mode)

    # random crop to (oh,ow)
    _, _, HH, WW = x.shape
    if HH == oh:
        top = 0
    else:
        top = int(torch.randint(0, HH - oh + 1, (1,)).item())
    if WW == ow:
        left = 0
    else:
        left = int(torch.randint(0, WW - ow + 1, (1,)).item())

    x = x[:, :, top : top + oh, left : left + ow]
    return x


def _fft_filter_delta(delta: torch.Tensor, keep: str, cutoff: float) -> torch.Tensor:
    """
    delta: [1,3,H,W] (pixel-space delta)
    keep: "low" or "high"
    cutoff: normalized radial cutoff in [0, 0.5]-ish; e.g. 0.08..0.20 are typical
    """
    assert keep in ("low", "high")

    # do FFT in fp32 for stability (fp16 often breaks/behaves oddly)
    in_dtype = delta.dtype
    delta32 = delta.to(torch.float32)

    B, C, H, W = delta32.shape
    device = delta32.device

    fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=torch.float32).view(H, 1)
    fx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=torch.float32).view(1, W // 2 + 1)
    rr = torch.sqrt(fy * fy + fx * fx)

    if keep == "low":
        mask = (rr <= cutoff).to(torch.float32)
    else:
        mask = (rr >= cutoff).to(torch.float32)

    out = torch.zeros_like(delta32)
    for ch in range(C):
        Fch = torch.fft.rfft2(delta32[:, ch, :, :])
        Fch = Fch * mask
        out[:, ch, :, :] = torch.fft.irfft2(Fch, s=(H, W))

    return out.to(in_dtype)


def _proj_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    # delta: [1,3,H,W], eps in pixel space
    flat = delta.view(delta.shape[0], -1)
    n = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    factor = torch.minimum(torch.ones_like(n), (eps / n))
    return (flat * factor).view_as(delta)


def _compute_margin_from_logits(logits_2: torch.Tensor) -> float:
    # logits_2: [2] on CPU
    return float((logits_2[1] - logits_2[0]).item())


def _record_step_history(
    history: List[Dict[str, float]],
    step: int,
    probs_2: torch.Tensor,
    logits_2: torch.Tensor,
    x_pix: torch.Tensor,
    x0_pix: torch.Tensor,
) -> None:
    p_src = float(probs_2[0].item())
    p_tgt = float(probs_2[1].item())
    margin = _compute_margin_from_logits(logits_2)

    delta = (x_pix.detach() - x0_pix.detach())
    linf = float(delta.abs().max().item())
    l2 = float(delta.view(-1).pow(2).sum().sqrt().item())

    history.append(
        {
            "step": float(step),
            "p_source": p_src,
            "p_target": p_tgt,
            "margin(logitT-logitS)": float(margin),
            "delta_linf_pix": float(linf),
            "delta_l2_pix": float(l2),
        }
    )


def _maybe_save_milestone(
    milestone_paths: Dict[str, str],
    save_dir: str,
    tag: str,
    attack: str,
    eps_pix: float,
    step_idx: int,
    name: str,
    x_pix: torch.Tensor,
    p_t: float,
    margin: float,
) -> None:
    if attack == "deepdream":
        return

    fn = _milestone_filename(tag, attack, eps_pix, step_idx, name, p_t, margin)
    path = os.path.join(save_dir, fn)
    _save_pix_image(x_pix, path)
    milestone_paths[name] = path


def _maybe_save_periodic(
    save_dir: str,
    tag: str,
    attack: str,
    eps_pix: float,
    step_idx: int,
    x_pix: torch.Tensor,
    p_t: float,
    margin: float,
    save_every: int = SAVE_EVERY,
) -> Optional[str]:
    if step_idx <= 0:
        return None
    if (step_idx % int(save_every)) != 0:
        return None

    fn = _milestone_filename(tag, attack, eps_pix, step_idx, "iter", p_t, margin)
    path = os.path.join(save_dir, fn)
    _save_pix_image(x_pix, path)
    return path


def _attack_common_setup(
    model: CLIP,
    preprocess,
    pil: Image.Image,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Returns:
      x0_norm: [1,3,H,W] normalized
      x0_pix:  [1,3,H,W] pixel in [0,1]
      mean/std on device
      (H,W)
    """
    mean = CLIP_MEAN.to(device=device)
    std = CLIP_STD.to(device=device)

    x0_norm = preprocess(pil).unsqueeze(0).to(device).float()  # normalized already
    # Convert to pixel for our attack parameterization
    x0_pix = _inv_normalize(x0_norm, mean, std).clamp(0, 1)
    H, W = int(x0_pix.shape[-2]), int(x0_pix.shape[-1])
    return x0_norm, x0_pix, mean, std, (H, W)

def linf_pgd_attack(
    model: CLIP,
    preprocess,
    pil: Image.Image,
    tok_2: torch.Tensor,
    eps_pix: float,
    steps: int,
    step_size_pix: Optional[float],
    device: str,
    save_dir: str,
    tag: str,
    attack: str = "linf_pgd",
    near_p: float = NEAR_FLIP_P_TARGET,
    solid_p: float = SOLID_P_TARGET,
    print_every: int = 25,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    model.eval().float()

    _, x0_pix, mean, std, _ = _attack_common_setup(model, preprocess, pil, device=device)

    if step_size_pix is None:
        step_size_pix = float(2.5 * eps_pix / max(1, steps))

    x = x0_pix.clone().detach()
    x.requires_grad_(True)

    milestone_paths: Dict[str, str] = {}
    first_step_near: Optional[int] = None
    first_step_flip: Optional[int] = None
    first_step_solid: Optional[int] = None
    history: List[Dict[str, float]] = []

    # Baseline
    probs0, logits0 = clip_two_way_probs_from_pix(model, x0_pix, tok_2, mean, std)
    p_tgt0 = float(probs0[1].item())
    margin0 = _compute_margin_from_logits(logits0)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, 0, "clean", x0_pix, p_tgt0, margin0)

    for t in range(1, steps + 1):
        # forward (must be differentiable)
        x_pix = x.clamp(0, 1)
        x_norm = _normalize(x_pix, mean, std)
        x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

        img_feat = F.normalize(model.encode_image(x_norm), dim=-1)
        txt_feat = F.normalize(model.encode_text(tok_2), dim=-1)

        logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)  # [2]
        obj = logits[1] - logits[0]  # maximize
        loss = -obj

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        with torch.no_grad():
            g = x.grad
            x_next = x - step_size_pix * g.sign()  # pixel-space ascent

            # project into Linf ball around x0
            delta = (x_next - x0_pix).clamp(-eps_pix, eps_pix)
            x_next = (x0_pix + delta).clamp(0, 1)

            x = x_next.detach()
            x.requires_grad_(True)

            probs, logits_cpu = clip_two_way_probs_from_pix(model, x, tok_2, mean, std)
            p_src = float(probs[0].item())
            p_tgt = float(probs[1].item())
            margin = _compute_margin_from_logits(logits_cpu)
            pred_is_target = (p_tgt >= p_src)

            if (t % print_every == 0) or pred_is_target or (p_tgt >= near_p) or (p_tgt >= solid_p):
                print(f"  step {t:4d}/{steps} | pT={p_tgt:.4f} pS={p_src:.4f} | margin(T-S)={margin:+.4f}")

            # periodic save
            _maybe_save_periodic(
                save_dir=save_dir,
                tag=tag,
                attack=attack,
                eps_pix=eps_pix,
                step_idx=t,
                x_pix=x,
                p_t=p_tgt,
                margin=margin,
                save_every=save_every,
            )

            _record_step_history(history, t, probs, logits_cpu, x, x0_pix)

            # milestones (save immediately when hit)
            if first_step_near is None and p_tgt >= near_p:
                first_step_near = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "near_flip", x, p_tgt, margin)
            if first_step_flip is None and pred_is_target:
                first_step_flip = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "flip", x, p_tgt, margin)
            if first_step_solid is None and p_tgt >= solid_p:
                first_step_solid = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "solid", x, p_tgt, margin)

    probsF, logitsF = clip_two_way_probs_from_pix(model, x.detach(), tok_2, mean, std)
    p_tgtF = float(probsF[1].item())
    marginF = _compute_margin_from_logits(logitsF)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, steps, "final", x.detach(), p_tgtF, marginF)

    return {
        "attack": attack,
        "eps_pix": float(eps_pix),
        "steps": int(steps),
        "p_target_clean": float(p_tgt0),
        "p_target_final": float(p_tgtF),
        "margin_clean": float(margin0),
        "margin_final": float(marginF),
        "first_step_near": first_step_near if first_step_near is not None else -1,
        "first_step_flip": first_step_flip if first_step_flip is not None else -1,
        "first_step_solid": first_step_solid if first_step_solid is not None else -1,
        "milestones": milestone_paths,
        "history": history,
    }


def eot_linf_pgd_attack(
    model: CLIP,
    preprocess,
    pil: Image.Image,
    tok_2: torch.Tensor,
    eps_pix: float,
    steps: int,
    step_size_pix: Optional[float],
    device: str,
    save_dir: str,
    tag: str,
    eot_samples: int = 4,
    scale_min: float = 0.90,
    scale_max: float = 1.10,
    attack: str = "eot_linf_pgd",
    near_p: float = NEAR_FLIP_P_TARGET,
    solid_p: float = SOLID_P_TARGET,
    print_every: int = 25,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    """
    EOT-PGD (L∞): average gradient over random transforms.

    - Periodic "iter" PNG obeys `save_every` (SAVE_EVERY).
    - Milestones are saved immediately when reached (skipped for deepdream).
    """
    model.eval().float()

    # Setup
    _, x0_pix, mean, std, hw = _attack_common_setup(model, preprocess, pil, device=device)

    # deepdream: enlarge internal canvas
    if attack == "deepdream":
        x0_pix = Toggle_Deepdream_Canvas(x0_pix, DREAM_CANVAS_SIZE)
        hw = (int(DREAM_CANVAS_SIZE), int(DREAM_CANVAS_SIZE))

    if step_size_pix is None:
        step_size_pix = float(2.0 * eps_pix / max(1, steps))

    x = x0_pix.clone().detach()
    x.requires_grad_(True)

    milestone_paths: Dict[str, str] = {}
    first_step_near: Optional[int] = None
    first_step_flip: Optional[int] = None
    first_step_solid: Optional[int] = None
    history: List[Dict[str, float]] = []

    do_milestones = (attack != "deepdream")

    # Baseline measurement
    if attack == "deepdream":
        probs0, logits0 = clip_two_way_probs_from_pix_multicrop(
            model,
            x0_pix,
            tok_2,
            mean,
            std,
            crop_size=DREAM_CROP_SIZE,
            crops_per_step=DREAM_CROPS_PER_STEP,
            center_bias=DREAM_CENTER_BIAS,
        )
    else:
        probs0, logits0 = clip_two_way_probs_from_pix(model, x0_pix, tok_2, mean, std)

    p_tgt0 = float(probs0[1].item())
    margin0 = _compute_margin_from_logits(logits0)

    if do_milestones:
        _maybe_save_milestone(
            milestone_paths, save_dir, tag, attack, eps_pix, 0, "clean", x0_pix, p_tgt0, margin0
        )

    # Main loop
    for t in range(1, steps + 1):
        grad_acc = torch.zeros_like(x)

        if attack == "deepdream":
            K_total = int(max(1, eot_samples)) * int(max(1, DREAM_CROPS_PER_STEP))

            txt_feat = model.encode_text(tok_2)
            txt_feat = F.normalize(txt_feat, dim=-1)

            for _ in range(K_total):
                x_pix = x.clamp(0, 1)
                x_crop = _rand_crop_224(
                    x_pix,
                    crop_size=int(DREAM_CROP_SIZE),
                    center_bias=float(DREAM_CENTER_BIAS),
                )

                x_norm = _normalize(x_crop, mean, std)
                x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

                img_feat = F.normalize(model.encode_image(x_norm), dim=-1)
                logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)
                obj = logits[1] - logits[0]
                loss = -obj

                model.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad.zero_()
                loss.backward(retain_graph=True)

                grad_acc += x.grad.detach()

            g = grad_acc / float(max(1, K_total))
        else:
            for _ in range(eot_samples):
                x_pix = x.clamp(0, 1)
                x_tr = _random_resize_pad_crop(
                    x_pix,
                    out_hw=hw,
                    scale_min=scale_min,
                    scale_max=scale_max,
                )

                x_norm = _normalize(x_tr, mean, std)
                x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

                img_feat = F.normalize(model.encode_image(x_norm), dim=-1)
                txt_feat = F.normalize(model.encode_text(tok_2), dim=-1)

                logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)
                obj = logits[1] - logits[0]
                loss = -obj

                model.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad.zero_()
                loss.backward(retain_graph=True)

                grad_acc += x.grad.detach()

            g = grad_acc / float(max(1, eot_samples))

        with torch.no_grad():
            x_next = x - step_size_pix * g.sign()
            delta = (x_next - x0_pix).clamp(-eps_pix, eps_pix)

            if attack == "deepdream" and (DREAM_FFT_EVERY > 0) and (t % int(DREAM_FFT_EVERY) == 0):
                delta = _fft_filter_delta(delta, keep="low", cutoff=float(DREAM_FFT_CUTOFF))
                delta = delta.clamp(-eps_pix, eps_pix)

            x_next = (x0_pix + delta).clamp(0, 1)

            x = x_next.detach()
            x.requires_grad_(True)

            if attack == "deepdream":
                probs, logits_cpu = clip_two_way_probs_from_pix_multicrop(
                    model,
                    x,
                    tok_2,
                    mean,
                    std,
                    crop_size=DREAM_CROP_SIZE,
                    crops_per_step=DREAM_CROPS_PER_STEP,
                    center_bias=DREAM_CENTER_BIAS,
                )
            else:
                probs, logits_cpu = clip_two_way_probs_from_pix(model, x, tok_2, mean, std)

            p_src = float(probs[0].item())
            p_tgt = float(probs[1].item())
            margin = _compute_margin_from_logits(logits_cpu)
            pred_is_target = (p_tgt >= p_src)

            if (t % print_every == 0) or pred_is_target or (p_tgt >= near_p) or (p_tgt >= solid_p):
                print(
                    f"  step {t:4d}/{steps} | pT={p_tgt:.4f} pS={p_src:.4f} | margin(T-S)={margin:+.4f} | EOT={eot_samples}"
                )

            # periodic save
            _maybe_save_periodic(
                save_dir=save_dir,
                tag=tag,
                attack=attack,
                eps_pix=eps_pix,
                step_idx=t,
                x_pix=x,
                p_t=p_tgt,
                margin=margin,
                save_every=save_every,
            )

            _record_step_history(history, t, probs, logits_cpu, x, x0_pix)

            if do_milestones:
                if first_step_near is None and p_tgt >= near_p:
                    first_step_near = t
                    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "near_flip", x, p_tgt, margin)
                if first_step_flip is None and pred_is_target:
                    first_step_flip = t
                    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "flip", x, p_tgt, margin)
                if first_step_solid is None and p_tgt >= solid_p:
                    first_step_solid = t
                    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "solid", x, p_tgt, margin)

    # Final measurement
    if attack == "deepdream":
        probsF, logitsF = clip_two_way_probs_from_pix_multicrop(
            model,
            x.detach(),
            tok_2,
            mean,
            std,
            crop_size=DREAM_CROP_SIZE,
            crops_per_step=DREAM_CROPS_PER_STEP,
            center_bias=DREAM_CENTER_BIAS,
        )
    else:
        probsF, logitsF = clip_two_way_probs_from_pix(model, x.detach(), tok_2, mean, std)

    p_tgtF = float(probsF[1].item())
    marginF = _compute_margin_from_logits(logitsF)

    if do_milestones:
        _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, steps, "final", x.detach(), p_tgtF, marginF)

    return {
        "attack": attack,
        "eps_pix": float(eps_pix),
        "steps": int(steps),
        "p_target_clean": float(p_tgt0),
        "p_target_final": float(p_tgtF),
        "margin_clean": float(margin0),
        "margin_final": float(marginF),
        "first_step_near": first_step_near if first_step_near is not None else -1,
        "first_step_flip": first_step_flip if first_step_flip is not None else -1,
        "first_step_solid": first_step_solid if first_step_solid is not None else -1,
        "milestones": milestone_paths,
        "history": history,
        "eot_samples": int(eot_samples),
        "eot_scale_min": float(scale_min),
        "eot_scale_max": float(scale_max),
        "save_every": int(save_every),
    }


def l2_pgd_attack(
    model: CLIP,
    preprocess,
    pil: Image.Image,
    tok_2: torch.Tensor,
    eps_pix: float,
    steps: int,
    step_size_pix: Optional[float],
    device: str,
    save_dir: str,
    tag: str,
    attack: str = "l2_pgd",
    near_p: float = NEAR_FLIP_P_TARGET,
    solid_p: float = SOLID_P_TARGET,
    print_every: int = 25,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    model.eval().float()

    _, x0_pix, mean, std, _ = _attack_common_setup(model, preprocess, pil, device=device)

    if step_size_pix is None:
        step_size_pix = float(1.0 * eps_pix / max(1, steps) * 8.0)

    x = x0_pix.clone().detach()
    x.requires_grad_(True)

    milestone_paths: Dict[str, str] = {}
    first_step_near: Optional[int] = None
    first_step_flip: Optional[int] = None
    first_step_solid: Optional[int] = None
    history: List[Dict[str, float]] = []

    probs0, logits0 = clip_two_way_probs_from_pix(model, x0_pix, tok_2, mean, std)
    p_tgt0 = float(probs0[1].item())
    margin0 = _compute_margin_from_logits(logits0)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, 0, "clean", x0_pix, p_tgt0, margin0)

    for t in range(1, steps + 1):
        x_pix = x.clamp(0, 1)
        x_norm = _normalize(x_pix, mean, std)
        x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

        img_feat = F.normalize(model.encode_image(x_norm), dim=-1)
        txt_feat = F.normalize(model.encode_text(tok_2), dim=-1)

        logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)
        obj = logits[1] - logits[0]
        loss = -obj

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        with torch.no_grad():
            g = x.grad
            g_flat = g.view(1, -1)
            g_norm = g_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            g_unit = (g_flat / g_norm).view_as(g)

            x_next = x - step_size_pix * g_unit

            delta = x_next - x0_pix
            delta = _proj_l2_ball(delta, eps_pix)
            x_next = (x0_pix + delta).clamp(0, 1)

            x = x_next.detach()
            x.requires_grad_(True)

            probs, logits_cpu = clip_two_way_probs_from_pix(model, x, tok_2, mean, std)
            p_src = float(probs[0].item())
            p_tgt = float(probs[1].item())
            margin = _compute_margin_from_logits(logits_cpu)
            pred_is_target = (p_tgt >= p_src)

            if (t % print_every == 0) or pred_is_target or (p_tgt >= near_p) or (p_tgt >= solid_p):
                print(f"  step {t:4d}/{steps} | pT={p_tgt:.4f} pS={p_src:.4f} | margin(T-S)={margin:+.4f}")

            # periodic save
            _maybe_save_periodic(save_dir, tag, attack, eps_pix, t, x, p_tgt, margin, save_every=save_every)

            _record_step_history(history, t, probs, logits_cpu, x, x0_pix)

            if first_step_near is None and p_tgt >= near_p:
                first_step_near = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "near_flip", x, p_tgt, margin)
            if first_step_flip is None and pred_is_target:
                first_step_flip = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "flip", x, p_tgt, margin)
            if first_step_solid is None and p_tgt >= solid_p:
                first_step_solid = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "solid", x, p_tgt, margin)

    probsF, logitsF = clip_two_way_probs_from_pix(model, x.detach(), tok_2, mean, std)
    p_tgtF = float(probsF[1].item())
    marginF = _compute_margin_from_logits(logitsF)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, steps, "final", x.detach(), p_tgtF, marginF)

    return {
        "attack": attack,
        "eps_pix": float(eps_pix),
        "steps": int(steps),
        "p_target_clean": float(p_tgt0),
        "p_target_final": float(p_tgtF),
        "margin_clean": float(margin0),
        "margin_final": float(marginF),
        "first_step_near": first_step_near if first_step_near is not None else -1,
        "first_step_flip": first_step_flip if first_step_flip is not None else -1,
        "first_step_solid": first_step_solid if first_step_solid is not None else -1,
        "milestones": milestone_paths,
        "history": history,
    }


def ditimi_linf_pgd_attack(
    model: CLIP,
    preprocess,
    pil: Image.Image,
    tok_2: torch.Tensor,
    eps_pix: float,
    steps: int,
    step_size_pix: Optional[float],
    device: str,
    save_dir: str,
    tag: str,
    di_prob: float = 0.7,
    di_scale_min: float = 0.90,
    di_scale_max: float = 1.10,
    ti_ks: int = 15,
    ti_sigma: float = 3.0,
    momentum: float = 0.9,
    attack: str = "ditimi_linf_pgd",
    near_p: float = NEAR_FLIP_P_TARGET,
    solid_p: float = SOLID_P_TARGET,
    print_every: int = 25,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    model.eval().float()

    _, x0_pix, mean, std, hw = _attack_common_setup(model, preprocess, pil, device=device)

    if step_size_pix is None:
        step_size_pix = float(2.0 * eps_pix / max(1, steps))

    x = x0_pix.clone().detach()
    x.requires_grad_(True)
    v = torch.zeros_like(x)

    milestone_paths: Dict[str, str] = {}
    first_step_near: Optional[int] = None
    first_step_flip: Optional[int] = None
    first_step_solid: Optional[int] = None
    history: List[Dict[str, float]] = []

    probs0, logits0 = clip_two_way_probs_from_pix(model, x0_pix, tok_2, mean, std)
    p_tgt0 = float(probs0[1].item())
    margin0 = _compute_margin_from_logits(logits0)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, 0, "clean", x0_pix, p_tgt0, margin0)

    for t in range(1, steps + 1):
        x_pix = x.clamp(0, 1)

        do_di = (float(torch.rand(1).item()) < di_prob)
        if do_di:
            x_in = _random_resize_pad_crop(x_pix, out_hw=hw, scale_min=di_scale_min, scale_max=di_scale_max)
        else:
            x_in = x_pix

        x_norm = _normalize(x_in, mean, std)
        x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

        img_feat = F.normalize(model.encode_image(x_norm), dim=-1)
        txt_feat = F.normalize(model.encode_text(tok_2), dim=-1)

        logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)
        obj = logits[1] - logits[0]
        loss = -obj

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        with torch.no_grad():
            g = x.grad.detach()
            g = _ti_smooth_grad(g, ks=ti_ks, sigma=ti_sigma)
            g = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12))

            v = momentum * v + g
            x_next = x - step_size_pix * v.sign()

            delta = (x_next - x0_pix).clamp(-eps_pix, eps_pix)
            x_next = (x0_pix + delta).clamp(0, 1)

            x = x_next.detach()
            x.requires_grad_(True)

            probs, logits_cpu = clip_two_way_probs_from_pix(model, x, tok_2, mean, std)
            p_src = float(probs[0].item())
            p_tgt = float(probs[1].item())
            margin = _compute_margin_from_logits(logits_cpu)
            pred_is_target = (p_tgt >= p_src)

            if (t % print_every == 0) or pred_is_target or (p_tgt >= near_p) or (p_tgt >= solid_p):
                print(
                    f"  step {t:4d}/{steps} | pT={p_tgt:.4f} pS={p_src:.4f} | margin(T-S)={margin:+.4f} | DI={do_di}"
                )

            # periodic save
            _maybe_save_periodic(save_dir, tag, attack, eps_pix, t, x, p_tgt, margin, save_every=save_every)

            _record_step_history(history, t, probs, logits_cpu, x, x0_pix)

            if first_step_near is None and p_tgt >= near_p:
                first_step_near = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "near_flip", x, p_tgt, margin)
            if first_step_flip is None and pred_is_target:
                first_step_flip = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "flip", x, p_tgt, margin)
            if first_step_solid is None and p_tgt >= solid_p:
                first_step_solid = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "solid", x, p_tgt, margin)

    probsF, logitsF = clip_two_way_probs_from_pix(model, x.detach(), tok_2, mean, std)
    p_tgtF = float(probsF[1].item())
    marginF = _compute_margin_from_logits(logitsF)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, steps, "final", x.detach(), p_tgtF, marginF)

    return {
        "attack": attack,
        "eps_pix": float(eps_pix),
        "steps": int(steps),
        "p_target_clean": float(p_tgt0),
        "p_target_final": float(p_tgtF),
        "margin_clean": float(margin0),
        "margin_final": float(marginF),
        "first_step_near": first_step_near if first_step_near is not None else -1,
        "first_step_flip": first_step_flip if first_step_flip is not None else -1,
        "first_step_solid": first_step_solid if first_step_solid is not None else -1,
        "milestones": milestone_paths,
        "history": history,
        "di_prob": float(di_prob),
        "ti_ks": int(ti_ks),
        "ti_sigma": float(ti_sigma),
        "momentum": float(momentum),
    }


def fft_filtered_linf_pgd_attack(
    model: CLIP,
    preprocess,
    pil: Image.Image,
    tok_2: torch.Tensor,
    eps_pix: float,
    steps: int,
    step_size_pix: Optional[float],
    device: str,
    save_dir: str,
    tag: str,
    keep: str,
    cutoff: float,
    attack: str,
    near_p: float = NEAR_FLIP_P_TARGET,
    solid_p: float = SOLID_P_TARGET,
    print_every: int = 25,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    assert keep in ("low", "high")
    model.eval().float()

    _, x0_pix, mean, std, _ = _attack_common_setup(model, preprocess, pil, device=device)

    if step_size_pix is None:
        step_size_pix = float(2.0 * eps_pix / max(1, steps))

    x = x0_pix.clone().detach()
    x.requires_grad_(True)

    milestone_paths: Dict[str, str] = {}
    first_step_near: Optional[int] = None
    first_step_flip: Optional[int] = None
    first_step_solid: Optional[int] = None
    history: List[Dict[str, float]] = []

    probs0, logits0 = clip_two_way_probs_from_pix(model, x0_pix, tok_2, mean, std)
    p_tgt0 = float(probs0[1].item())
    margin0 = _compute_margin_from_logits(logits0)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, 0, "clean", x0_pix, p_tgt0, margin0)

    for t in range(1, steps + 1):
        x_pix = x.clamp(0, 1)
        x_norm = _normalize(x_pix, mean, std)
        x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

        img_feat = F.normalize(model.encode_image(x_norm), dim=-1)
        txt_feat = F.normalize(model.encode_text(tok_2), dim=-1)

        logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)
        obj = logits[1] - logits[0]
        loss = -obj

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        with torch.no_grad():
            g = x.grad.detach()
            x_next = x - step_size_pix * g.sign()

            delta = (x_next - x0_pix).clamp(-eps_pix, eps_pix)
            delta_f = _fft_filter_delta(delta, keep=keep, cutoff=cutoff)
            delta_f = delta_f.clamp(-eps_pix, eps_pix)

            x_next = (x0_pix + delta_f).clamp(0, 1)

            x = x_next.detach()
            x.requires_grad_(True)

            probs, logits_cpu = clip_two_way_probs_from_pix(model, x, tok_2, mean, std)
            p_src = float(probs[0].item())
            p_tgt = float(probs[1].item())
            margin = _compute_margin_from_logits(logits_cpu)
            pred_is_target = (p_tgt >= p_src)

            if (t % print_every == 0) or pred_is_target or (p_tgt >= near_p) or (p_tgt >= solid_p):
                print(
                    f"  step {t:4d}/{steps} | pT={p_tgt:.4f} pS={p_src:.4f} | margin(T-S)={margin:+.4f} | FFT={keep}@{cutoff:.3f}"
                )

            # periodic save
            _maybe_save_periodic(save_dir, tag, attack, eps_pix, t, x, p_tgt, margin, save_every=save_every)

            _record_step_history(history, t, probs, logits_cpu, x, x0_pix)

            if first_step_near is None and p_tgt >= near_p:
                first_step_near = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "near_flip", x, p_tgt, margin)
            if first_step_flip is None and pred_is_target:
                first_step_flip = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "flip", x, p_tgt, margin)
            if first_step_solid is None and p_tgt >= solid_p:
                first_step_solid = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, t, "solid", x, p_tgt, margin)

    probsF, logitsF = clip_two_way_probs_from_pix(model, x.detach(), tok_2, mean, std)
    p_tgtF = float(probsF[1].item())
    marginF = _compute_margin_from_logits(logitsF)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, eps_pix, steps, "final", x.detach(), p_tgtF, marginF)

    return {
        "attack": attack,
        "eps_pix": float(eps_pix),
        "steps": int(steps),
        "p_target_clean": float(p_tgt0),
        "p_target_final": float(p_tgtF),
        "margin_clean": float(margin0),
        "margin_final": float(marginF),
        "first_step_near": first_step_near if first_step_near is not None else -1,
        "first_step_flip": first_step_flip if first_step_flip is not None else -1,
        "first_step_solid": first_step_solid if first_step_solid is not None else -1,
        "milestones": milestone_paths,
        "history": history,
        "fft_keep": keep,
        "fft_cutoff": float(cutoff),
    }


def patch_attack(
    model: CLIP,
    preprocess,
    pil: Image.Image,
    tok_2: torch.Tensor,
    steps: int,
    device: str,
    save_dir: str,
    tag: str,
    patch_frac: float = 0.20,
    patch_loc: str = "center",
    step_size: float = 0.05,
    attack: str = "patch_attack",
    near_p: float = NEAR_FLIP_P_TARGET,
    solid_p: float = SOLID_P_TARGET,
    print_every: int = 25,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    model.eval().float()

    _, x0_pix, mean, std, hw = _attack_common_setup(model, preprocess, pil, device=device)
    H, W = hw

    ph = max(1, int(round(H * patch_frac)))
    pw = max(1, int(round(W * patch_frac)))

    if patch_loc == "center":
        top = (H - ph) // 2
        left = (W - pw) // 2
    elif patch_loc == "topleft":
        top, left = 0, 0
    elif patch_loc == "random":
        top = int(torch.randint(0, max(1, H - ph + 1), (1,)).item())
        left = int(torch.randint(0, max(1, W - pw + 1), (1,)).item())
    else:
        raise ValueError(f"patch_loc={patch_loc}")

    patch = torch.rand((1, 3, ph, pw), device=device, dtype=torch.float32, requires_grad=True)

    milestone_paths: Dict[str, str] = {}
    first_step_near: Optional[int] = None
    first_step_flip: Optional[int] = None
    first_step_solid: Optional[int] = None
    history: List[Dict[str, float]] = []

    probs0, logits0 = clip_two_way_probs_from_pix(model, x0_pix, tok_2, mean, std)
    p_tgt0 = float(probs0[1].item())
    margin0 = _compute_margin_from_logits(logits0)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, patch_frac, 0, "clean", x0_pix, p_tgt0, margin0)

    def apply_patch(x_base: torch.Tensor, patch_now: torch.Tensor) -> torch.Tensor:
        x = x_base.clone()
        x[:, :, top : top + ph, left : left + pw] = patch_now.clamp(0, 1)
        return x

    for t in range(1, steps + 1):
        x_adv = apply_patch(x0_pix, patch)

        x_norm = _normalize(x_adv, mean, std)
        x_norm = cast_imgs_to_visual_dtype(x_norm, model.visual)

        img_feat = F.normalize(model.encode_image(x_norm), dim=-1)
        txt_feat = F.normalize(model.encode_text(tok_2), dim=-1)

        logits = (img_feat @ txt_feat.T).squeeze(0) * float(FIXED_LOGIT_SCALE)
        obj = logits[1] - logits[0]
        loss = -obj

        model.zero_grad(set_to_none=True)
        if patch.grad is not None:
            patch.grad.zero_()
        loss.backward()

        with torch.no_grad():
            g = patch.grad.detach()
            patch -= step_size * g.sign()
            patch.clamp_(0, 1)

            # measure on current x_adv
            probs, logits_cpu = clip_two_way_probs_from_pix(model, x_adv.detach(), tok_2, mean, std)
            p_src = float(probs[0].item())
            p_tgt = float(probs[1].item())
            margin = _compute_margin_from_logits(logits_cpu)
            pred_is_target = (p_tgt >= p_src)

            if (t % print_every == 0) or pred_is_target or (p_tgt >= near_p) or (p_tgt >= solid_p):
                print(f"  step {t:4d}/{steps} | pT={p_tgt:.4f} pS={p_src:.4f} | margin(T-S)={margin:+.4f} | patch={ph}x{pw}")

            # periodic save
            _maybe_save_periodic(save_dir, tag, attack, patch_frac, t, x_adv.detach(), p_tgt, margin, save_every=save_every)

            _record_step_history(history, t, probs, logits_cpu, x_adv.detach(), x0_pix)

            if first_step_near is None and p_tgt >= near_p:
                first_step_near = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, patch_frac, t, "near_flip", x_adv.detach(), p_tgt, margin)
            if first_step_flip is None and pred_is_target:
                first_step_flip = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, patch_frac, t, "flip", x_adv.detach(), p_tgt, margin)
            if first_step_solid is None and p_tgt >= solid_p:
                first_step_solid = t
                _maybe_save_milestone(milestone_paths, save_dir, tag, attack, patch_frac, t, "solid", x_adv.detach(), p_tgt, margin)

    x_final = apply_patch(x0_pix, patch).detach()
    probsF, logitsF = clip_two_way_probs_from_pix(model, x_final, tok_2, mean, std)
    p_tgtF = float(probsF[1].item())
    marginF = _compute_margin_from_logits(logitsF)
    _maybe_save_milestone(milestone_paths, save_dir, tag, attack, patch_frac, steps, "final", x_final, p_tgtF, marginF)

    return {
        "attack": attack,
        "eps_pix": float(patch_frac),
        "steps": int(steps),
        "p_target_clean": float(p_tgt0),
        "p_target_final": float(p_tgtF),
        "margin_clean": float(margin0),
        "margin_final": float(marginF),
        "first_step_near": first_step_near if first_step_near is not None else -1,
        "first_step_flip": first_step_flip if first_step_flip is not None else -1,
        "first_step_solid": first_step_solid if first_step_solid is not None else -1,
        "milestones": milestone_paths,
        "history": history,
        "patch_frac": float(patch_frac),
        "patch_hw": [int(ph), int(pw)],
        "patch_loc": str(patch_loc),
        "patch_xy": [int(top), int(left)],
        "patch_step_size": float(step_size),
    }


def run_attack(
    attack_name: str,
    model: CLIP,
    preprocess,
    pil: Image.Image,
    tok_2: torch.Tensor,
    eps_pix: float,
    steps: int,
    alpha: Optional[float],
    device: str,
    save_dir: str,
    tag: str,
) -> Dict[str, object]:

    # deepdream is just EOT-Linf-PGD but tagged differently
    if attack_name == "deepdream":
        return eot_linf_pgd_attack(
            model,
            preprocess,
            pil,
            tok_2,
            eps_pix,
            steps,
            alpha,
            device,
            save_dir,
            tag,
            eot_samples=EOT_SAMPLES,
            scale_min=EOT_SCALE_MIN,
            scale_max=EOT_SCALE_MAX,
            attack="deepdream",
            save_every=SAVE_EVERY,
        )

    if attack_name == "linf_pgd":
        return linf_pgd_attack(
            model, preprocess, pil, tok_2, eps_pix, steps, alpha, device, save_dir, tag,
            attack="linf_pgd",
            save_every=SAVE_EVERY,
        )

    if attack_name == "l2_pgd":
        return l2_pgd_attack(
            model, preprocess, pil, tok_2, eps_pix, steps, alpha, device, save_dir, tag,
            attack="l2_pgd",
            save_every=SAVE_EVERY,
        )

    if attack_name == "eot_linf_pgd":
        return eot_linf_pgd_attack(
            model,
            preprocess,
            pil,
            tok_2,
            eps_pix,
            steps,
            alpha,
            device,
            save_dir,
            tag,
            eot_samples=EOT_SAMPLES,
            scale_min=EOT_SCALE_MIN,
            scale_max=EOT_SCALE_MAX,
            attack="eot_linf_pgd",
            save_every=SAVE_EVERY,
        )

    if attack_name == "ditimi_linf_pgd":
        return ditimi_linf_pgd_attack(
            model,
            preprocess,
            pil,
            tok_2,
            eps_pix,
            steps,
            alpha,
            device,
            save_dir,
            tag,
            di_prob=DI_PROB,
            di_scale_min=DI_SCALE_MIN,
            di_scale_max=DI_SCALE_MAX,
            ti_ks=TI_KS,
            ti_sigma=TI_SIGMA,
            momentum=MOMENTUM,
            attack="ditimi_linf_pgd",
            save_every=SAVE_EVERY,
        )

    if attack_name == "fft_low_linf_pgd":
        return fft_filtered_linf_pgd_attack(
            model,
            preprocess,
            pil,
            tok_2,
            eps_pix,
            steps,
            alpha,
            device,
            save_dir,
            tag,
            keep="low",
            cutoff=FFT_CUTOFF,
            attack="fft_low_linf_pgd",
            save_every=SAVE_EVERY,
        )

    if attack_name == "fft_high_linf_pgd":
        return fft_filtered_linf_pgd_attack(
            model,
            preprocess,
            pil,
            tok_2,
            eps_pix,
            steps,
            alpha,
            device,
            save_dir,
            tag,
            keep="high",
            cutoff=FFT_CUTOFF,
            attack="fft_high_linf_pgd",
            save_every=SAVE_EVERY,
        )

    if attack_name == "patch_attack":
        return patch_attack(
            model,
            preprocess,
            pil,
            tok_2,
            steps=steps,
            device=device,
            save_dir=save_dir,
            tag=tag,
            patch_frac=PATCH_FRAC,
            patch_loc=PATCH_LOC,
            step_size=PATCH_STEP_SIZE,
            attack="patch_attack",
            save_every=SAVE_EVERY,
        )

    raise ValueError(f"Unknown attack_name={attack_name}")



def print_model_robustness_summary(df: pd.DataFrame) -> None:
    """
    Model-level robustness: "last to flip" across eps-swept attacks.
    We compute per (model, attack, image): the smallest eps_pix that flips (pT >= pS).
    Aggregate per model:
      - never_flip_rate: fraction of (attack,image) pairs that never flipped at any eps
      - median_flip_eps: median of first-flip eps over pairs that did flip
    Sorted most-robust first (highest never_flip_rate, then highest median_flip_eps).
    """
    if df.empty:
        print("\n[Model Robustness] (empty)")
        return

    # Only eps-swept attacks are meaningful for "last to flip"
    swept_attacks = set(df["attack"].unique().tolist()) - {"patch_attack"}
    d = df[df["attack"].isin(list(swept_attacks))].copy()
    if d.empty:
        print("\n[Model Robustness] (no eps-swept attacks)")
        return

    # first-flip eps per (model, attack, image)
    rows = []
    for (model, attack, image), g in d.groupby(["model", "attack", "image"]):
        g = g.sort_values("eps_pix", ascending=True)
        flipped = g[g["first_step_flip"].astype(int) != -1]
        if len(flipped) == 0:
            first_eps = float("inf")
        else:
            first_eps = float(flipped.iloc[0]["eps_pix"])
        rows.append({"model": model, "attack": attack, "image": image, "first_flip_eps": first_eps})

    dd = pd.DataFrame(rows)
    out_rows = []
    for model, gm in dd.groupby("model"):
        n = len(gm)
        never = int((gm["first_flip_eps"] == float("inf")).sum())
        flipped = gm[gm["first_flip_eps"] != float("inf")]["first_flip_eps"].values

        never_rate = never / max(1, n)
        if len(flipped) == 0:
            med_eps = float("inf")
            min_eps = float("inf")
        else:
            med_eps = float(pd.Series(flipped).median())
            min_eps = float(pd.Series(flipped).min())

        out_rows.append(
            {
                "model": model,
                "pairs": n,
                "never_flip_pairs": never,
                "never_flip_rate": never_rate,
                "median_first_flip_eps_pix": med_eps,
                "min_first_flip_eps_pix": min_eps,
            }
        )

    res = pd.DataFrame(out_rows)

    def sort_key(row):
        # never flip = most robust; treat median inf as very robust
        return (row["never_flip_rate"], row["median_first_flip_eps_pix"])

    res = res.sort_values(
        by=["never_flip_rate", "median_first_flip_eps_pix"],
        ascending=[False, False],
    )

    print("\n==================== MODEL ROBUSTNESS (last to flip) ====================")
    for _, r in res.iterrows():
        med_eps = r["median_first_flip_eps_pix"]
        min_eps = r["min_first_flip_eps_pix"]

        def fmt_eps(e):
            if math.isinf(e):
                return "inf"
            return f"{e:.6f}  (~{e*255.0:.1f}/255)"

        print(
            f"  {r['model']}: pairs={int(r['pairs'])} | never_flip={int(r['never_flip_pairs'])}/{int(r['pairs'])} ({r['never_flip_rate']:.1%}) "
            f"| median first-flip eps={fmt_eps(med_eps)} | min first-flip eps={fmt_eps(min_eps)}"
        )


def print_end_summary(df: pd.DataFrame) -> None:
    """
    Print a compact "success by step" summary at the end, for all models & attacks.
    """
    if df.empty:
        print("\n[Summary] (empty)")
        return

    print("\n==================== END SUMMARY ====================")

    # success definitions
    def is_success_flip(row) -> bool:
        return int(row["first_step_flip"]) != -1

    def is_success_solid(row) -> bool:
        return int(row["first_step_solid"]) != -1

    for attack in sorted(df["attack"].unique().tolist()):
        dfa = df[df["attack"] == attack].copy()
        print(f"\n[Attack: {attack}]")

        for model in sorted(dfa["model"].unique().tolist()):
            dfm = dfa[dfa["model"] == model].copy()
            if dfm.empty:
                continue

            n = len(dfm)
            succ_flip = sum(is_success_flip(r) for _, r in dfm.iterrows())
            succ_solid = sum(is_success_solid(r) for _, r in dfm.iterrows())

            # average steps on successes
            flip_steps = [int(r["first_step_flip"]) for _, r in dfm.iterrows() if int(r["first_step_flip"]) != -1]
            solid_steps = [int(r["first_step_solid"]) for _, r in dfm.iterrows() if int(r["first_step_solid"]) != -1]
            avg_flip = (sum(flip_steps) / len(flip_steps)) if flip_steps else float("nan")
            avg_solid = (sum(solid_steps) / len(solid_steps)) if solid_steps else float("nan")

            # median final p_target as a sanity signal
            med_pT = float(dfm["p_target_final"].median())

            print(
                f"  {model}: runs={n} | flip_succ={succ_flip}/{n} ({succ_flip/n:.2%}) "
                f"| solid_succ={succ_solid}/{n} ({succ_solid/n:.2%}) "
                f"| avg_flip_step={avg_flip:.1f} avg_solid_step={avg_solid:.1f} | median pT_final={med_pT:.3f}"
            )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fix_seed(SEED)

    # Validate attacks
    for a in ATTACKS_TO_RUN:
        if a not in ATTACK_NAMES:
            raise RuntimeError(f"Unknown attack '{a}'. Allowed: {ATTACK_NAMES}")

    # Build image list
    all_paths = list_images(FOLDER)
    if ONLY.strip() == "*":
        attack_paths = all_paths
    else:
        wanted = [s.strip() for s in ONLY.split(",") if s.strip()]
        wanted_set = set(wanted)
        attack_paths = [p for p in all_paths if os.path.basename(p) in wanted_set]
    if len(attack_paths) == 0:
        raise RuntimeError("No images found to attack. Check FOLDER and ONLY.")

    # Preprocess (load once from first model entry)
    m0, preprocess, _ = load_openai_clip_anything(clip, MODELS[0][1], device=DEVICE, jit=False, strict=True)
    m0.eval().float()
    del m0
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

    # Tokenize two prompts
    tok_2 = clip.tokenize([PROMPT_SOURCE, PROMPT_TARGET]).to(DEVICE)

    summary_rows: List[Dict[str, object]] = []
    details: Dict[str, object] = {}

    # Outer loop: attacks -> models -> images
    for attack_name in ATTACKS_TO_RUN:
        print(f"\n==================== ATTACK: {attack_name} ====================")
        details[attack_name] = {}

        attack_dir = os.path.join(OUT_DIR, attack_name)
        os.makedirs(attack_dir, exist_ok=True)

        # pick eps schedule
        if attack_name == "deepdream":
            eps_list_pix = [eps255_to_pix(e) for e in DREAM_EPS_LIST]
        else:
            eps_list_pix = [eps255_to_pix(e) for e in DEFAULT_EPS_LIST_255]

        for (alias, mp) in MODELS:
            print(f"\n==================== Model: {alias} ====================")
            model, _, _ = load_openai_clip_anything(clip, mp, device=DEVICE, jit=False, strict=True)
            model.eval().float()

            details[attack_name][alias] = {}

            model_dir = os.path.join(attack_dir, alias)
            os.makedirs(model_dir, exist_ok=True)

            for img_path in attack_paths:
                fn = os.path.basename(img_path)
                tag_base = os.path.splitext(fn)[0]
                pil = load_rgb(img_path)

                print(f"\n--- Image: {fn} ---")
                details[attack_name][alias][fn] = {}

                run_dir = os.path.join(model_dir, tag_base)
                os.makedirs(run_dir, exist_ok=True)
                tag = f"{alias}__{tag_base}"

                # Patch doesn’t sweep eps; run once.
                if attack_name == "patch_attack":
                    res = run_attack(
                        attack_name=attack_name,
                        model=model,
                        preprocess=preprocess,
                        pil=pil,
                        tok_2=tok_2,
                        eps_pix=float(PATCH_FRAC),  # stored as eps_pix for CSV compatibility
                        steps=STEPS,
                        alpha=ALPHA,
                        device=DEVICE,
                        save_dir=run_dir,
                        tag=tag,
                    )

                    row = {
                        "attack": attack_name,
                        "model": alias,
                        "image": fn,
                        "eps_pix": res["eps_pix"],
                        "steps": res["steps"],
                        "p_target_clean": res["p_target_clean"],
                        "p_target_final": res["p_target_final"],
                        "margin_clean": res["margin_clean"],
                        "margin_final": res["margin_final"],
                        "first_step_near": res["first_step_near"],
                        "first_step_flip": res["first_step_flip"],
                        "first_step_solid": res["first_step_solid"],
                    }
                    summary_rows.append(row)
                    details[attack_name][alias][fn]["patch"] = res

                    print(
                        f"{attack_name} | patch_frac={row['eps_pix']:.3f} | "
                        f"pT(clean→final) {row['p_target_clean']:.3f}→{row['p_target_final']:.3f} | "
                        f"flip@{row['first_step_flip']} solid@{row['first_step_solid']}"
                    )
                else:
                    for eps_pix in eps_list_pix:
                        res = run_attack(
                            attack_name=attack_name,
                            model=model,
                            preprocess=preprocess,
                            pil=pil,
                            tok_2=tok_2,
                            eps_pix=eps_pix,
                            steps=STEPS,
                            alpha=ALPHA,
                            device=DEVICE,
                            save_dir=run_dir,
                            tag=tag,
                        )

                        row = {
                            "attack": attack_name,
                            "model": alias,
                            "image": fn,
                            "eps_pix": res["eps_pix"],
                            "steps": res["steps"],
                            "p_target_clean": res["p_target_clean"],
                            "p_target_final": res["p_target_final"],
                            "margin_clean": res["margin_clean"],
                            "margin_final": res["margin_final"],
                            "first_step_near": res["first_step_near"],
                            "first_step_flip": res["first_step_flip"],
                            "first_step_solid": res["first_step_solid"],
                        }
                        summary_rows.append(row)
                        details[attack_name][alias][fn][f"eps_{eps_pix:.6f}"] = res

                        print(
                            f"{attack_name} | eps={eps_pix:.6f} (~{eps_pix*255.0:.1f}/255) | "
                            f"pT(clean→final) {row['p_target_clean']:.3f}→{row['p_target_final']:.3f} | "
                            f"flip@{row['first_step_flip']} solid@{row['first_step_solid']}"
                        )

            del model
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()

    # Write one CSV/JSON for all attacks
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(OUT_DIR, "details.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    # End summaries
    print_end_summary(df)
    print_model_robustness_summary(df)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {OUT_DIR}/<attack>/<model>/<image>/  (milestone PNGs)")


if __name__ == "__main__":
    main()