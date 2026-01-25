"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

 Quick Probe:
 Single-image probe for "REG tokens exist but aren't USED" hypothesis (fine-tune).

 For each (model, image), at EACH layer:
   A) REG token stats (count, top norms, coordinates)
   B) CLS->REG attention mass fraction
   C) V-projection norm stats (REG vs non-REG)
   D) Headwise energy ratios (pre-out and post-out via out_proj mixing)
   E) Headwise "logit alignment" scores using gradient contraction

"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import attnclipindiv as clip
from attnclipindiv.model import CLIP

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

# "REGISTER" definition: patch norm cutoff
REG_THRESHOLD = 70.0

USE_AMP = False

# Output dir
OUT_DIR = "out_eval_measure/quick_reg_gate_probe"

DEFAULT_IMAGES: List[str] = [
    "image_sets/attn_bench_images/pure_cup.png",
    "image_sets/attn_bench_images/mixed_catdog.png",
    "image_sets/attn_bench_images/mixed_plunger.png",
    "image_sets/attn_bench_images/mixed_ipod.png",
    "image_sets/attn_bench_images/text_ai.png",
    "image_sets/attn_bench_images/text_stroop.png",
]

def default_pair_for_image(path: str) -> Tuple[str, str]:
    fn = os.path.basename(path).lower()
    if fn == "pure_cup.png":
        return ("a cup of coffee", "a face")
    if fn == "mixed_catdog.png":
        return ("a cat", "a husky")
    if fn == "mixed_plunger.png":
        return ("a plunger", "a gun")
    if fn == "mixed_ipod.png":
        return ("an apple", "an ipod")
    if fn == "text_ai.png":
        return ("a text", "an ai")
    if fn == "text_stroop.png":
        return ("a green word", "a blue word")
    # fallback
    return ("object", "text")


def sanitize_filename(s: str, max_len: int = 140) -> str:
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .")
    if not s:
        s = "item"
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s

def _amp_ctx(device: str):
    if device.startswith("cuda") and USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    class _NoOp:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False

    return _NoOp()

def cast_imgs_to_visual_dtype(imgs: torch.Tensor, visual: nn.Module) -> torch.Tensor:
    w = visual.conv1.weight
    if imgs.dtype != w.dtype:
        imgs = imgs.to(dtype=w.dtype)
    return imgs


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _infer_patch_grid(num_patches: int) -> int:
    g = int(round(math.sqrt(float(num_patches))))
    if g * g == num_patches:
        return g
    return 16


def patch_rc_from_index(p: int, grid: int) -> Tuple[int, int]:
    return (p // grid, p % grid)



# Visual forward with ALL-layer capture (single forward/backward)
@dataclass
class LayerCache:
    x_before_det_cpu: torch.Tensor     # [S,B,E] detached, stored on CPU (fp16)
    x_after_seq: torch.Tensor          # [S,B,E] in-graph tensor (retain_grad)
    attn_cls_src_cpu: torch.Tensor     # [B,H,S] on CPU (fp32)


def clear_attn_probs(visual: nn.Module):
    for blk in visual.transformer.resblocks:
        blk.attn_probs = None
        blk.attn_grad = None


def forward_visual_collect_all_layers(
    visual: nn.Module,
    imgs: torch.Tensor,          # [B,3,224,224]
    retain_grads: bool,
) -> Tuple[torch.Tensor, List[LayerCache]]:
    """
    Returns:
      cls_resid_final: [B,E] final CLS residual (pre ln_post)
      caches: list[LayerCache] with length = num_blocks, aligned by layer index.
    """
    device = imgs.device
    clear_attn_probs(visual)

    x = visual.conv1(imgs)                       # [B, C, gh, gw]
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1).permute(0, 2, 1)     # [B,HW,E]

    class_emb = visual.class_embedding.to(x.dtype)
    cls_tokens = class_emb + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=device)
    x = torch.cat([cls_tokens, x], dim=1)        # [B,1+HW,E]

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                       # [S,B,E]

    caches: List[LayerCache] = []
    for li, blk in enumerate(visual.transformer.resblocks):
        x_before = x
        x = blk(x)

        attn = blk.attn_probs  # [B,H,S,S] CPU expected
        if attn is None:
            # keep a safe shape
            attn_cls_src_cpu = torch.zeros((x.shape[1], blk.attn.num_heads, x.shape[0]), dtype=torch.float32)
        else:
            attn_cls_src_cpu = attn[:, :, 0, :].contiguous().to(dtype=torch.float32)  # [B,H,S] CPU

        x_after_seq = x
        if retain_grads:
            x_after_seq.retain_grad()

        # Store x_before detached on CPU to keep GPU memory sane during sweeps
        x_before_det_cpu = x_before.detach().to("cpu", dtype=torch.float16, copy=True)

        caches.append(LayerCache(
            x_before_det_cpu=x_before_det_cpu,
            x_after_seq=x_after_seq,
            attn_cls_src_cpu=attn_cls_src_cpu,
        ))

    # final CLS resid
    x2 = x.permute(1, 0, 2)         # [B,S,E]
    cls_resid_final = x2[:, 0, :]   # [B,E]
    return cls_resid_final, caches


def encode_image_from_cls(visual: nn.Module, cls_resid: torch.Tensor) -> torch.Tensor:
    cls = visual.ln_post(cls_resid)
    if visual.proj is not None:
        cls = cls @ visual.proj
    return cls


# REG mask and probing math (per-layer from cached tensors)
def reg_mask_from_x_before_cpu(x_before_det_cpu: torch.Tensor, reg_threshold: float) -> torch.Tensor:
    """
    x_before_det_cpu: [S,B,E] detached, CPU
    returns reg_mask_patch: [B,P] bool over patches only
    """
    x = x_before_det_cpu.permute(1, 0, 2).float()  # [B,S,E]
    patches = x[:, 1:, :]                          # [B,P,E]
    norms = patches.norm(dim=-1)                   # [B,P]
    return norms >= reg_threshold

def compute_metrics_from_cache(
    model: CLIP,
    layer: int,
    cache: LayerCache,
    grad_seq: torch.Tensor,          # [S,B,E] on device
    obj_label: str,
    attack_label: str,
    margin: torch.Tensor,            # [1]
    logit_obj: torch.Tensor,         # [1]
    logit_att: torch.Tensor,         # [1]
    reg_threshold: float,
    device: str,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Returns:
      summary dict (per-layer)
      per-head DataFrame (per-layer)
    """
    visual = model.visual
    blk = visual.transformer.resblocks[layer]

    # Grad on sequence tensor at layer output -> CLS row
    grad_x_cls = grad_seq[0].detach().float()  # [B=1,E]

    # ---- shapes / params ----
    H = int(blk.attn.num_heads)
    E = int(blk.attn.embed_dim)
    hd = E // H

    # ---- A) REG mask + norms ----
    x_before_det_cpu = cache.x_before_det_cpu         # [S,B,E] CPU fp16
    x_in_cpu = x_before_det_cpu.permute(1, 0, 2).float()  # [B,S,E] CPU fp32
    P = int(x_in_cpu.shape[1] - 1)
    grid = _infer_patch_grid(P)

    reg_mask_patch = reg_mask_from_x_before_cpu(x_before_det_cpu, reg_threshold=reg_threshold)  # [1,P]
    patch_norms = x_in_cpu[:, 1:, :].norm(dim=-1)[0]  # [P]
    reg_idx = torch.nonzero(reg_mask_patch[0], as_tuple=False).flatten()
    reg_count = int(reg_idx.numel())

    top_reg_items: List[Dict[str, object]] = []
    if reg_count > 0:
        reg_norms = patch_norms[reg_idx]
        k = min(8, reg_count)
        topk = torch.topk(reg_norms, k=k, largest=True)
        for j in range(k):
            p = int(reg_idx[topk.indices[j]].item())
            n = float(topk.values[j].item())
            r, c = patch_rc_from_index(p, grid=grid)
            top_reg_items.append({"patch": p, "r": r, "c": c, "norm": n})

    # pull attention
    a = cache.attn_cls_src_cpu.to(device=device, dtype=torch.float32)  # [B,H,S]
    a = a[0]                                                           # [H,S]
    a_patch = a[:, 1:]                                                 # [H,P]

    # CLS->REG attention mass fraction
    if reg_count > 0:
        attn_reg_mass_per_head = a_patch[:, reg_idx.to(device)].sum(dim=1)          # [H]
    else:
        attn_reg_mass_per_head = torch.zeros((H,), device=device)
    attn_all_mass_per_head = a_patch.sum(dim=1) + 1e-8                               # [H]
    attn_reg_frac_per_head = (attn_reg_mass_per_head / attn_all_mass_per_head).clamp(0.0, 1.0)
    attn_reg_frac = float(attn_reg_frac_per_head.mean().item())

    # V-proj norms (REG vs non-REG)
    with torch.no_grad():
        xb = x_before_det_cpu.to(device=device, dtype=torch.float32)                # [S,B,E]
        v_lin = blk.attn.v_proj(xb)                                                 # [S,B,E]
        v = v_lin.permute(1, 0, 2).contiguous().view(1, v_lin.shape[0], H, hd).permute(0, 2, 1, 3)[0]  # [H,S,hd]
        v_patch = v[:, 1:, :]                                                       # [H,P,hd]
        vnorm_patch = v_patch.norm(dim=-1)                                          # [H,P]

        if reg_count > 0:
            reg_idx_dev = reg_idx.to(device)
            vnorm_reg = vnorm_patch[:, reg_idx_dev].mean(dim=1)                     # [H]
            reg_bool = torch.zeros((P,), device=device, dtype=torch.bool)
            reg_bool[reg_idx_dev] = True
            non_idx = torch.nonzero(~reg_bool, as_tuple=False).flatten()
            vnorm_non = vnorm_patch[:, non_idx].mean(dim=1) if non_idx.numel() else torch.zeros((H,), device=device)
        else:
            vnorm_reg = torch.zeros((H,), device=device)
            vnorm_non = vnorm_patch.mean(dim=1)

        vnorm_reg_mean = float(vnorm_reg.mean().item())
        vnorm_non_mean = float(vnorm_non.mean().item())

    # Headwise energy ratios (pre-out + post-out via out_proj)
    with torch.no_grad():
        a_p = a_patch.unsqueeze(-1)                # [H,P,1]
        v_p = v_patch                              # [H,P,hd]
        z_all_h = (a_p * v_p).sum(dim=1)           # [H,hd]

        if reg_count > 0:
            reg_mask_p = torch.zeros((P,), device=device, dtype=torch.bool)
            reg_mask_p[reg_idx.to(device)] = True
            z_reg_h = (a_p * v_p * reg_mask_p.view(1, -1, 1)).sum(dim=1)
        else:
            z_reg_h = torch.zeros_like(z_all_h)

        e_all_pre = (z_all_h.pow(2).sum(dim=-1) + 1e-8)     # [H]
        e_reg_pre = z_reg_h.pow(2).sum(dim=-1)              # [H]
        pre_ratio_h = (e_reg_pre / e_all_pre).clamp(0.0, 1.0)

        W = blk.attn.out_proj.weight.detach().to(device=device, dtype=torch.float32)  # [E,E]

        def _embed_head(z_h_vec: torch.Tensor, head_id: int) -> torch.Tensor:
            out = torch.zeros((E,), device=device, dtype=torch.float32)
            out[head_id * hd:(head_id + 1) * hd] = z_h_vec
            return out

        deltas_all_h = []
        deltas_reg_h = []
        e_all_post_h = torch.zeros((H,), device=device)
        e_reg_post_h = torch.zeros((H,), device=device)

        for h in range(H):
            z_all_E = _embed_head(z_all_h[h].float(), h).unsqueeze(0)  # [1,E]
            z_reg_E = _embed_head(z_reg_h[h].float(), h).unsqueeze(0)  # [1,E]
            d_all = z_all_E @ W.t()
            d_reg = z_reg_E @ W.t()
            deltas_all_h.append(d_all)
            deltas_reg_h.append(d_reg)
            e_all_post_h[h] = float(d_all.pow(2).sum().item() + 1e-8)
            e_reg_post_h[h] = float(d_reg.pow(2).sum().item())

        post_ratio_h = (e_reg_post_h / e_all_post_h).clamp(0.0, 1.0)

        d_all_total = torch.stack(deltas_all_h, dim=0).sum(dim=0)  # [1,E]
        d_reg_total = torch.stack(deltas_reg_h, dim=0).sum(dim=0)  # [1,E]
        post_ratio_total = float((d_reg_total.pow(2).sum().item()) / (d_all_total.pow(2).sum().item() + 1e-8))

    # Headwise logit-alignment scores
    with torch.no_grad():
        g = grad_x_cls.to(device=device, dtype=torch.float32)[0]           # [E]
        gW = (g.unsqueeze(0) @ W).view(H, hd)                              # [H,hd]

        dot_v_gW = (v_patch * gW.unsqueeze(1)).sum(dim=-1)                 # [H,P]
        contrib_patch_h = a_patch * dot_v_gW                               # [H,P]

        if reg_count > 0:
            reg_idx_dev = reg_idx.to(device)
            score_reg_h = contrib_patch_h[:, reg_idx_dev].sum(dim=1)       # [H]
            reg_bool = torch.zeros((P,), device=device, dtype=torch.bool)
            reg_bool[reg_idx_dev] = True
            non_idx = torch.nonzero(~reg_bool, as_tuple=False).flatten()
            score_non_h = contrib_patch_h[:, non_idx].sum(dim=1) if non_idx.numel() else torch.zeros((H,), device=device)
        else:
            score_reg_h = torch.zeros((H,), device=device)
            score_non_h = contrib_patch_h.sum(dim=1)

        abs_contrib_h = contrib_patch_h.abs()
        if reg_count > 0:
            abs_reg_h = abs_contrib_h[:, reg_idx_dev].sum(dim=1)
            reg_bool = torch.zeros((P,), device=device, dtype=torch.bool)
            reg_bool[reg_idx_dev] = True
            non_idx = torch.nonzero(~reg_bool, as_tuple=False).flatten()
            abs_non_h = abs_contrib_h[:, non_idx].sum(dim=1) if non_idx.numel() else torch.zeros((H,), device=device)
        else:
            abs_reg_h = torch.zeros((H,), device=device)
            abs_non_h = abs_contrib_h.sum(dim=1)

        abs_reg_total = float(abs_reg_h.sum().item())
        abs_all_total = float((abs_reg_h.sum() + abs_non_h.sum()).item() + 1e-8)
        reg_logit_share_abs = abs_reg_total / abs_all_total

        signed_reg_total = float(score_reg_h.sum().item())
        signed_all_total = float((score_reg_h.sum() + score_non_h.sum()).item() + 1e-8)
        reg_logit_share_signed = signed_reg_total / signed_all_total

    head_rows = []
    for h in range(H):
        head_rows.append({
            "layer": int(layer),
            "head": int(h),
            "attn_reg_mass": float(attn_reg_mass_per_head[h].item()),
            "attn_all_mass": float(attn_all_mass_per_head[h].item()),
            "attn_reg_frac": float(attn_reg_frac_per_head[h].item()),
            "vnorm_reg": float(vnorm_reg[h].item()) if reg_count > 0 else 0.0,
            "vnorm_non": float(vnorm_non[h].item()),
            "pre_energy_ratio": float(pre_ratio_h[h].item()),
            "post_energy_ratio": float(post_ratio_h[h].item()),
            "score_reg_signed": float(score_reg_h[h].item()),
            "score_non_signed": float(score_non_h[h].item()),
            "score_reg_abs": float(abs_reg_h[h].item()),
            "score_non_abs": float(abs_non_h[h].item()),
        })
    heads_df = pd.DataFrame(head_rows)

    summary = {
        "obj_label": obj_label,
        "attack_label": attack_label,
        "layer": int(layer),
        "reg_threshold": float(reg_threshold),

        "margin": float(margin.detach().cpu().item()),
        "logit_obj": float(logit_obj.detach().cpu().item()),
        "logit_att": float(logit_att.detach().cpu().item()),

        "reg_count": int(reg_count),
        "top_reg": top_reg_items,

        "attn_reg_frac_mean": float(attn_reg_frac),

        "vnorm_reg_mean": float(vnorm_reg_mean),
        "vnorm_non_mean": float(vnorm_non_mean),

        "pre_energy_ratio_mean": float(pre_ratio_h.mean().item()),
        "post_energy_ratio_mean": float(post_ratio_h.mean().item()),
        "post_energy_ratio_total": float(post_ratio_total),

        "reg_logit_share_abs": float(reg_logit_share_abs),
        "reg_logit_share_signed": float(reg_logit_share_signed),

        "grad_x_cls_l2": float(grad_x_cls.norm(dim=-1).item()),
    }

    return summary, heads_df


def compute_probe_metrics_all_layers(
    model: CLIP,
    image_tensor: torch.Tensor,   # [1,3,224,224] on device
    obj_label: str,
    attack_label: str,
    reg_threshold: float,
    device: str,
) -> Tuple[List[Dict[str, object]], List[pd.DataFrame]]:
    """
    Single forward/backward that yields per-layer summaries + per-layer heads DFs.
    """
    model.eval().float()
    visual = model.visual
    visual.eval()

    texts = [f"a photo of a {obj_label}", f"a photo of a {attack_label}"]
    tok = clip.tokenize(texts).to(device)

    clear_attn_probs(visual)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        with _amp_ctx(device):
            txt = model.encode_text(tok)          # [2,D]
            txt = F.normalize(txt, dim=-1)

            cls_final, caches = forward_visual_collect_all_layers(
                visual=visual,
                imgs=image_tensor,
                retain_grads=True,
            )
            img = encode_image_from_cls(visual, cls_final)  # [1,D]
            img = F.normalize(img, dim=-1)

            logit_obj = (img * txt[0:1]).sum(dim=-1)        # [1]
            logit_att = (img * txt[1:2]).sum(dim=-1)        # [1]
            margin = (logit_obj - logit_att)                # [1]
            loss = margin.sum()

        loss.backward()

    # Build per-layer outputs
    summaries: List[Dict[str, object]] = []
    heads_dfs: List[pd.DataFrame] = []

    for li, cache in enumerate(caches):
        grad_seq = cache.x_after_seq.grad
        if grad_seq is None:
            raise RuntimeError(
                f"grad_seq is None at layer={li}. "
                "This indicates retain_grad() didn't attach or autograd graph was broken."
            )

        summary, heads_df = compute_metrics_from_cache(
            model=model,
            layer=li,
            cache=cache,
            grad_seq=grad_seq,
            obj_label=obj_label,
            attack_label=attack_label,
            margin=margin,
            logit_obj=logit_obj,
            logit_att=logit_att,
            reg_threshold=reg_threshold,
            device=device,
        )
        summaries.append(summary)
        heads_dfs.append(heads_df)

    return summaries, heads_dfs


# Integrated analysis + plotting (in-memory)
def summarize_and_print(summary_df: pd.DataFrame, heads_df: pd.DataFrame) -> None:
    print("\n==================== coffee_reg_gate_probe summary (INTEGRATED) ====================\n")

    agg_cols = [
        "reg_count",
        "attn_reg_frac_mean",
        "vnorm_reg_mean",
        "vnorm_non_mean",
        "pre_energy_ratio_mean",
        "post_energy_ratio_mean",
        "post_energy_ratio_total",
        "reg_logit_share_abs",
        "reg_logit_share_signed",
        "margin",
        "grad_x_cls_l2",
    ]

    if summary_df.empty:
        print("[WARN] No summary rows found.")
        return

    g = summary_df.groupby(["layer", "model_alias"], dropna=False)[agg_cols]
    agg_mean = g.mean(numeric_only=True).reset_index()

    layers = sorted([int(x) for x in agg_mean["layer"].dropna().astype(int).unique().tolist()])
    models = agg_mean["model_alias"].unique().tolist()

    print("Per-layer means (averaged across images/pairs found):")
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        sub = agg_mean[agg_mean["layer"].astype(int) == layer].copy()
        sub = sub.sort_values(["model_alias"])

        for _, r in sub.iterrows():
            ml = r["model_alias"]
            print(
                f"{ml}\n"
                f"  reg_count={r['reg_count']:.3f} | attn_reg_frac={r['attn_reg_frac_mean']:.4f} | "
                f"vnorm_reg={r['vnorm_reg_mean']:.3f} vs nonreg={r['vnorm_non_mean']:.3f}\n"
                f"  post_energy_total={r['post_energy_ratio_total']:.4f} | reg_abs_share={r['reg_logit_share_abs']:.4f} | "
                f"margin={r['margin']:.4f} | grad_norm={r['grad_x_cls_l2']:.4f}"
            )

    # Heads summary
    if heads_df.empty:
        print("\n[WARN] No heads rows found; skipping per-head summary.")
        return

    print("\n\nPer-head localization (top heads by route_score = attn_reg_frac * post_energy_ratio):")
    heads_df2 = heads_df.copy()
    heads_df2["route_score"] = heads_df2["attn_reg_frac"] * heads_df2["post_energy_ratio"]

    hg = heads_df2.groupby(["layer", "model_alias", "head"], dropna=False)[
        ["attn_reg_frac", "post_energy_ratio", "pre_energy_ratio", "route_score"]
    ].mean(numeric_only=True).reset_index()

    layers = sorted([int(x) for x in hg["layer"].dropna().astype(int).unique().tolist()])
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        for model_alias in sorted(hg[hg["layer"].astype(int) == layer]["model_alias"].unique().tolist()):
            sub = hg[(hg["layer"].astype(int) == layer) & (hg["model_alias"] == model_alias)].copy()
            sub = sub.sort_values("route_score", ascending=False).head(6)
            print(f"{model_alias} | top heads:")
            for _, r in sub.iterrows():
                print(
                    f"  head={int(r['head']):>2d} | route_score={r['route_score']:.4f} | "
                    f"attn_reg_frac={r['attn_reg_frac']:.4f} | post_energy_ratio={r['post_energy_ratio']:.4f} "
                    f"| pre_energy_ratio={r['pre_energy_ratio']:.4f}"
                )


def plot_layer_sweeps(summary_df: pd.DataFrame, heads_df: pd.DataFrame, out_plot_dir: str) -> None:
    _ensure_dir(out_plot_dir)

    # Summary metric sweeps
    if not summary_df.empty:
        agg_cols = [
            "attn_reg_frac_mean",
            "post_energy_ratio_total",
            "reg_logit_share_abs",
            "reg_count",
            "grad_x_cls_l2",
            "margin",
        ]
        agg = summary_df.groupby(["layer", "model_alias"], dropna=False)[agg_cols].mean(numeric_only=True).reset_index()
        agg = agg.sort_values(["layer", "model_alias"])

        for metric in agg_cols:
            plt.figure(figsize=(8, 5), dpi=150)
            for model_alias in sorted(agg["model_alias"].unique().tolist()):
                sub = agg[agg["model_alias"] == model_alias].copy()
                xs = sub["layer"].astype(int).tolist()
                ys = sub[metric].astype(float).tolist()
                plt.plot(xs, ys, marker="o", label=model_alias)
            plt.xlabel("Layer")
            plt.ylabel(metric)
            plt.title(f"Layer sweep: {metric}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_plot_dir, f"layer_sweep__{metric}.png"), bbox_inches="tight")
            plt.close()

    # Headwise sweeps (top-K summaries)
    if heads_df.empty:
        return

    heads_df2 = heads_df.copy()
    heads_df2["route_score"] = heads_df2["attn_reg_frac"] * heads_df2["post_energy_ratio"]

    hg = heads_df2.groupby(["layer", "model_alias", "head"], dropna=False)[
        ["attn_reg_frac", "post_energy_ratio", "pre_energy_ratio", "route_score"]
    ].mean(numeric_only=True).reset_index()

    K = 6
    topk_rows: List[Dict[str, object]] = []
    for (layer, model_alias), sub in hg.groupby(["layer", "model_alias"], dropna=False):
        if pd.isna(layer) or sub.empty:
            continue
        sub2 = sub.sort_values("route_score", ascending=False).head(K)
        topk_rows.append({
            "layer": int(layer),
            "model_alias": str(model_alias),
            f"top{K}_route_score_mean": float(sub2["route_score"].mean()),
            f"top{K}_attn_reg_frac_mean": float(sub2["attn_reg_frac"].mean()),
            f"top{K}_post_energy_ratio_mean": float(sub2["post_energy_ratio"].mean()),
        })

    topk = pd.DataFrame(topk_rows)
    if not topk.empty:
        topk = topk.sort_values(["layer", "model_alias"])
        for metric in [f"top{K}_route_score_mean", f"top{K}_attn_reg_frac_mean", f"top{K}_post_energy_ratio_mean"]:
            plt.figure(figsize=(8, 5), dpi=150)
            for model_alias in sorted(topk["model_alias"].unique().tolist()):
                sub = topk[topk["model_alias"] == model_alias].copy()
                xs = sub["layer"].astype(int).tolist()
                ys = sub[metric].astype(float).tolist()
                plt.plot(xs, ys, marker="o", label=model_alias)
            plt.xlabel("Layer")
            plt.ylabel(metric)
            plt.title(f"Layer sweep (headwise top{K}): {metric}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_plot_dir, f"layer_sweep__top{K}__{metric}.png"), bbox_inches="tight")
            plt.close()

        topk.to_csv(os.path.join(out_plot_dir, f"top{K}_heads__layer_sweep.csv"), index=False)

    # Top-head barplots at the max layer found
    layers_ok = hg["layer"].dropna()
    if layers_ok.empty:
        return
    last_layer = int(layers_ok.astype(int).max())

    for model_alias in sorted(hg[hg["layer"].astype(int) == last_layer]["model_alias"].unique().tolist()):
        sub = hg[(hg["layer"].astype(int) == last_layer) & (hg["model_alias"] == model_alias)].copy()
        sub = sub.sort_values("route_score", ascending=False).head(12)

        plt.figure(figsize=(10, 5), dpi=150)
        x = list(range(len(sub)))
        plt.bar(x, sub["route_score"].astype(float).tolist())
        plt.xticks(x, [f"h{int(h)}" for h in sub["head"].astype(int).tolist()], rotation=0)
        plt.xlabel("Head")
        plt.ylabel("route_score (attn_reg_frac * post_energy_ratio)")
        plt.title(f"Top heads by route_score | layer {last_layer} | {model_alias}")
        plt.tight_layout()
        safe = re.sub(r"[^a-z0-9]+", "_", model_alias.strip().lower())
        plt.savefig(os.path.join(out_plot_dir, f"top_heads__layer{last_layer}__{safe}.png"),
                    bbox_inches="tight")
        plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--reg_threshold", type=float, default=REG_THRESHOLD)
    parser.add_argument("--images", type=str, nargs="*", default=DEFAULT_IMAGES,
                        help="List of image paths to probe.")
    parser.add_argument("--obj_label", type=str, default=None,
                        help="Override object label for ALL images (otherwise per-image defaults).")
    parser.add_argument("--attack_label", type=str, default=None,
                        help="Override attack label for ALL images (otherwise per-image defaults).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | reg_thr={args.reg_threshold} | amp={USE_AMP}")
    print("Models:")
    for alias, path in MODELS:
        print(f"  - {alias}: {path}")
    print("Images:")
    for im in args.images:
        print("  -", im)

    _ensure_dir(args.out_dir)
    layers_root = os.path.join(args.out_dir, "layers")
    _ensure_dir(layers_root)

    # Use preprocess from first model
    print("\nLoading preprocess from first model...")
    m0, preprocess, _ = load_openai_clip_anything(clip, MODELS[0][1], device=device, jit=False, strict=True)
    num_layers_ref = len(m0.visual.transformer.resblocks)
    m0.eval().float()
    del m0
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"[INFO] visual.resblocks = {num_layers_ref} (probing ALL layers: 0..{num_layers_ref-1})")

    # Load + preprocess images (store compact on CPU; move per model)
    image_bank: Dict[str, torch.Tensor] = {}
    for path in args.images:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing image: {path}")
        img_pil = Image.open(path).convert("RGB")
        x = preprocess(img_pil)                    # [3,224,224] float32
        x = x.to(dtype=torch.float16, copy=False)  # compact storage
        image_bank[path] = x

    # Run probe sweep
    summary_rows: List[Dict[str, object]] = []
    heads_rows: List[pd.DataFrame] = []

    for model_alias, model_path in MODELS:
        model_slug = sanitize_filename(model_alias)
        print(f"\n==================== {model_alias} ====================")

        model, _, _ = load_openai_clip_anything(clip, model_path, device=device, jit=False, strict=True)
        model.eval().float()

        # sanity: layer count matches
        n_layers = len(model.visual.transformer.resblocks)
        if n_layers != num_layers_ref:
            print(f"[WARN] {model_alias} has {n_layers} layers, reference has {num_layers_ref}. Using {n_layers} for this model.")

        for img_path, img_cpu in image_bank.items():
            image_slug = sanitize_filename(os.path.basename(img_path))
            obj, att = default_pair_for_image(img_path)

            if args.obj_label is not None:
                obj = args.obj_label
            if args.attack_label is not None:
                att = args.attack_label

            img = img_cpu.unsqueeze(0).to(device, non_blocking=True)  # [1,3,224,224]
            img = cast_imgs_to_visual_dtype(img, model.visual)

            print(f"\n--- Image: {img_path} | pair: ({obj} vs {att}) ---")

            per_layer_summaries, per_layer_heads = compute_probe_metrics_all_layers(
                model=model,
                image_tensor=img,
                obj_label=obj,
                attack_label=att,
                reg_threshold=args.reg_threshold,
                device=device,
            )

            # Save per-layer artifacts + build combined tables
            for layer_idx, (summary, heads_df) in enumerate(zip(per_layer_summaries, per_layer_heads)):
                layer_dir = os.path.join(layers_root, f"layer_{layer_idx:02d}")
                _ensure_dir(layer_dir)

                # Lightweight per-layer print (avoids exploding stdout with full A/B/C/D/E blocks)
                print(
                    f"  L{layer_idx:02d} | reg_count={summary['reg_count']:>3d} | "
                    f"attn_reg_frac={summary['attn_reg_frac_mean']:.4f} | "
                    f"post_total={summary['post_energy_ratio_total']:.4f} | "
                    f"reg_abs_share={summary['reg_logit_share_abs']:.4f}"
                )

                out_json = os.path.join(layer_dir, f"{model_slug}__{image_slug}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

                out_heads = os.path.join(layer_dir, f"{model_slug}__{image_slug}__heads.csv")
                heads_df.to_csv(out_heads, index=False)

                # summary row
                row = dict(summary)
                row.update({
                    "model_alias": model_alias,
                    "model_path": model_path,
                    "image_path": img_path,
                })
                row["top_reg_str"] = "; ".join(
                    [f"p{d['patch']}({d['r']},{d['c']}):{d['norm']:.1f}" for d in summary["top_reg"]]
                )
                summary_rows.append(row)

                # heads rows (augment with identifiers)
                hdf = heads_df.copy()
                hdf["model_alias"] = model_alias
                hdf["model_path"] = model_path
                hdf["image_path"] = img_path
                hdf["pair"] = f"{obj} vs {att}"
                heads_rows.append(hdf)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Save run-level raw tables
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    heads_df = pd.concat(heads_rows, axis=0, ignore_index=True) if heads_rows else pd.DataFrame()

    out_summary_csv = os.path.join(args.out_dir, "summary_rows__raw.csv")
    out_heads_csv = os.path.join(args.out_dir, "heads_rows__raw.csv")
    if not summary_df.empty:
        summary_df.to_csv(out_summary_csv, index=False)
    if not heads_df.empty:
        heads_df.to_csv(out_heads_csv, index=False)

    # Integrated analysis outputs
    out_summary_dir = os.path.join(args.out_dir, "__summary")
    out_plot_dir = os.path.join(out_summary_dir, "plots")
    _ensure_dir(out_summary_dir)
    _ensure_dir(out_plot_dir)

    if not summary_df.empty:
        agg = summary_df.groupby(["layer", "model_alias"], dropna=False).mean(numeric_only=True).reset_index()
        agg.to_csv(os.path.join(out_summary_dir, "summary_layer_model__mean.csv"), index=False)

    if not heads_df.empty:
        hg = heads_df.groupby(["layer", "model_alias", "head"], dropna=False).mean(numeric_only=True).reset_index()
        hg.to_csv(os.path.join(out_summary_dir, "heads_layer_model_head__mean.csv"), index=False)

    summarize_and_print(summary_df, heads_df)
    plot_layer_sweeps(summary_df, heads_df, out_plot_dir)

    print("\nSaved outputs:")
    print("  ", args.out_dir)
    print("  ", layers_root)
    if not summary_df.empty:
        print("  ", out_summary_csv)
    if not heads_df.empty:
        print("  ", out_heads_csv)
    print("  ", out_summary_dir)
    print("  ", out_plot_dir)


if __name__ == "__main__":
    main()