"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

----- VARIANT: SCAM dataset (clean vs. attack) -----

CLIP ViT-L/14: late-half probing + hidden-space reps + survival-aware head writes + plotting + dyslexify.

Adds (requested):
  1) Dyslexify JSON production:
       - Writes a block->disabled_heads JSON you can feed back via --dyslexify_json.
       - By default uses a survival-weighted score per head (configurable).

  2) Dyslexify-only mode:
       - If --dyslexify_json is provided, the script runs ONLY dyslexify evaluation + related outputs (no EXP1/EXP2 full rerun).

  3) Plots (lots):
       - Per-model line plots over blocks for:
            reg_attn_mass, reg_write_norm, reg_write_survival, delta(attn_frac - write_frac)
       - Per-model heatmaps (heads x blocks) for the same.
       - Per-block (for each probed block): side-by-side pies (attn_frac vs write_frac), plus scatter attn vs write.
       - Per-model correlation curve corr(attn_frac, write_frac) over blocks.

"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import attnclipdecouple as clip  # capture: last_probs/last_v

from utils_clip_loader.cliptools import fix_random_seed
fix_random_seed()

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


# ============================================================
# Data
# ============================================================

@dataclass
class PairSample:
    image: Any  # PIL
    correct_label: str
    distractor_label: str
    meta: Dict[str, Any]


def _variant_from_id(sid: str) -> Optional[str]:
    for v in ("NoSCAM", "SynthSCAM"):
        if sid.startswith(v):
            return v
    return None


def _pair_key_from_id(sid: str) -> str:
    """
    Robust-ish pairing key between NoSCAM and SynthSCAM.
    Drops the leading variant prefix and any leading separators.
    """
    sid2 = re.sub(r"^(NoSCAM|SynthSCAM)", "", sid)
    sid2 = sid2.lstrip("_-: ")
    return sid2


def load_scam_samples(limit_per_variant: Optional[int] = None) -> Dict[str, List[PairSample]]:
    """
    Returns dict: variant -> list[PairSample]
    variants: NoSCAM, SynthSCAM
    """
    ds = load_dataset("BLISS-e-V/SCAM", split="train")
    buckets: Dict[str, List[PairSample]] = {v: [] for v in ["NoSCAM", "SynthSCAM"]}

    for entry in ds:
        sid = str(entry["id"])
        variant = _variant_from_id(sid)
        if variant is None:
            continue

        img = entry["image"]
        obj = str(entry["object_label"])
        atk = str(entry["attack_word"])

        buckets[variant].append(
            PairSample(
                image=img,
                correct_label=obj,
                distractor_label=atk,
                meta=dict(
                    id=sid,
                    pair_key=_pair_key_from_id(sid),
                    postit_area_pct=float(entry.get("postit_area_pct", 0.0)),
                    type=str(entry.get("type", "")),
                    dataset="SCAM",
                    variant=variant,
                ),
            )
        )

        if limit_per_variant is not None:
            # stop once BOTH buckets reached limit (keeps stable ordering per variant)
            if len(buckets["NoSCAM"]) >= limit_per_variant and len(buckets["SynthSCAM"]) >= limit_per_variant:
                break

    if limit_per_variant is not None:
        for v in buckets:
            buckets[v] = buckets[v][:limit_per_variant]

    return buckets


# ============================================================
# Math helpers (PCA + subspace angles)
# ============================================================

def pca_evr_topk(X: np.ndarray, k: int = 8) -> List[float]:
    """
    X: [N, D] (centered inside)
    returns: top-k explained variance ratios
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2)
    evr = var / (var.sum() + 1e-12)
    return evr[:k].tolist()


def principal_angles_deg(U: np.ndarray, V: np.ndarray, k: int = 8) -> List[float]:
    """
    U: [D, k] orthonormal basis
    V: [D, k] orthonormal basis
    Returns k principal angles in degrees.
    """
    M = U.T @ V
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    ang = np.arccos(s)
    return (ang * (180.0 / np.pi)).tolist()


def orthonormal_basis_from_samples(X: np.ndarray, k: int = 8) -> np.ndarray:
    """
    X: [N, D], returns basis [D, k]
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    B = Vt[:k].T  # [D,k]
    return B


def decompose_delta_into_subspace(
    dZ: np.ndarray,       # [N,D]
    basis: np.ndarray,    # [D,k] orthonormal
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (dz_reg, dz_non) per-sample norms:
      dz_reg = ||P_basis(dZ)||
      dz_non = ||dZ - P_basis(dZ)||
    """
    # projection: P = B B^T (since orthonormal)
    proj = dZ @ basis @ basis.T  # [N,D]
    dz_reg = np.linalg.norm(proj, axis=1)
    dz_non = np.linalg.norm(dZ - proj, axis=1)
    return dz_reg, dz_non


# ============================================================
# Register mask helper
# ============================================================

def make_implicit_register_mask(
    patch_token_norms: torch.Tensor,   # [B, P]
    register_threshold: float = 70.0,
    max_registers: Optional[int] = 8,
    min_registers: int = 1
) -> torch.Tensor:
    """
    Returns bool mask [B, P] where True = register.
    """
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


@torch.no_grad()
def tokens_to_final_embedding(visual: torch.nn.Module, x_lnd: torch.Tensor) -> torch.Tensor:
    """
    x_lnd: [T,B,C] token stream after some block (or final block),
    runs ln_post + proj on CLS token to produce image embedding (unnormalized).
    """
    x_btn = x_lnd.permute(1, 0, 2)  # [B,T,C]
    cls_h = x_btn[:, 0, :]          # [B,C] pre-ln_post
    cls = visual.ln_post(cls_h)     # [B,C]
    if getattr(visual, "proj", None) is not None:
        cls = cls @ visual.proj     # [B,D]
    return cls


# ============================================================
# EXP plumbing via forward hooks
# ============================================================

class BlockCapture:
    """
    Captures per-block hidden reps + attention caches when capture_layers includes the block index.
    Stores only what we need (mostly CPU) to avoid VRAM bloat.
    """
    def __init__(
        self,
        blocks: List[int],
        rep_blocks: List[int],
        survival_blocks: List[int],
        register_threshold: float,
        max_registers: int,
        min_registers: int,
        n_heads: int,
        head_dim: int,
        width: int,
        keep_gpu_for_survival: bool = True,
    ):
        self.blocks = set(blocks)
        self.rep_blocks = set(rep_blocks)
        self.survival_blocks = set(survival_blocks)

        self.register_threshold = float(register_threshold)
        self.max_registers = int(max_registers)
        self.min_registers = int(min_registers)

        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.width = int(width)

        # per-batch captures (cleared each forward)
        self.batch_cls_hidden: Dict[int, torch.Tensor] = {}        # block -> [B,C] CPU float32
        self.batch_patch_pool: Dict[int, torch.Tensor] = {}        # block -> [B,C] CPU float32 (non-REG)
        self.batch_reg_pool: Dict[int, torch.Tensor] = {}          # block -> [B,C] CPU float32 (REG)

        # EXP2 stats per block/head for this batch
        self.batch_reg_attn_mass: Dict[int, torch.Tensor] = {}     # block -> [H] CPU float32
        self.batch_reg_write_norm: Dict[int, torch.Tensor] = {}    # block -> [H] CPU float32

        # survival: stash GPU tensors needed for vectorized injection
        self.keep_gpu_for_survival = bool(keep_gpu_for_survival)
        self.batch_x_out_gpu: Dict[int, torch.Tensor] = {}         # block -> [T,B,C] GPU (detached)
        self.batch_write_vecs_gpu: Dict[int, torch.Tensor] = {}    # block -> [B,H,C] GPU (detached)

        # allow disabling hooks during survival passes (expanded batch size)
        self.suspend: bool = False  # IMPORTANT


    def clear_batch(self):
        self.batch_cls_hidden.clear()
        self.batch_patch_pool.clear()
        self.batch_reg_pool.clear()
        self.batch_reg_attn_mass.clear()
        self.batch_reg_write_norm.clear()
        if self.keep_gpu_for_survival:
            self.batch_x_out_gpu.clear()
            self.batch_write_vecs_gpu.clear()

    def hook_for_block(self, block_idx: int):
        """
        Returns a forward hook function for a given resblock.
        Assumes:
          - output is x_lnd [T,B,C]
          - block.attn.last_probs and block.attn.last_v populated when capture=True
        """
        def _hook(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if self.suspend:
                return
            if not isinstance(output, torch.Tensor) or output.dim() != 3:
                return

            x_lnd = output
            T, B, C = x_lnd.shape

            # NLD for pooling
            x_btn = x_lnd.permute(1, 0, 2)                 # [B,T,C]
            cls_h = x_btn[:, 0, :]                         # [B,C] pre-ln_post
            patches = x_btn[:, 1:, :]                      # [B,P,C]
            patch_norms = patches.norm(dim=-1)             # [B,P]

            reg_mask = make_implicit_register_mask(
                patch_token_norms=patch_norms,
                register_threshold=self.register_threshold,
                max_registers=self.max_registers,
                min_registers=self.min_registers
            )                                              # [B,P] bool

            non_mask = ~reg_mask
            non_cnt = non_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            reg_cnt = reg_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            patch_pool = (patches * non_mask.unsqueeze(-1)).sum(dim=1) / non_cnt  # [B,C]
            reg_pool = (patches * reg_mask.unsqueeze(-1)).sum(dim=1) / reg_cnt    # [B,C]

            if block_idx in self.rep_blocks:
                self.batch_cls_hidden[block_idx] = cls_h.detach().float().cpu()
                self.batch_patch_pool[block_idx] = patch_pool.detach().float().cpu()
                self.batch_reg_pool[block_idx] = reg_pool.detach().float().cpu()

            # EXP2 per-head attn/write stats if attention cache exists
            attn = getattr(module, "attn", None)
            if attn is None:
                return
            probs = getattr(attn, "last_probs", None)  # [B,H,T,S]
            v = getattr(attn, "last_v", None)          # [B,H,S,D]
            if probs is None or v is None:
                return
            if probs.shape[0] != B or v.shape[0] != B:
                return

            P = reg_mask.shape[1]
            S = 1 + P
            src_reg = torch.zeros((B, S), dtype=torch.bool, device=probs.device)
            src_reg[:, 1:] = reg_mask.to(device=probs.device)

            probs_cls = probs[:, :, 0, :]                                   # [B,H,S]
            reg_mass = (probs_cls * src_reg.unsqueeze(1).float()).sum(dim=-1)  # [B,H]
            reg_mass_mean = reg_mass.mean(dim=0).detach().float().cpu()     # [H]

            # reg-sourced head output (pre out_proj): sum_{src in REG} probs * v
            v_reg = v * src_reg.unsqueeze(1).unsqueeze(-1).float()          # [B,H,S,D]
            head_out = (probs_cls.unsqueeze(-1) * v_reg).sum(dim=2)         # [B,H,D]

            out_proj = getattr(attn, "out_proj", None)
            if out_proj is None:
                return
            W = out_proj.weight  # [E,E]

            # per-head write vectors in width space [B,H,E]
            writes = []
            for h in range(self.n_heads):
                w_slice = W[:, h * self.head_dim:(h + 1) * self.head_dim]   # [E,D]
                wh = head_out[:, h, :] @ w_slice.T                          # [B,E]
                writes.append(wh)
            write_vecs = torch.stack(writes, dim=1)                          # [B,H,E]
            write_norm = write_vecs.norm(dim=-1).mean(dim=0).detach().float().cpu()  # [H]

            self.batch_reg_attn_mass[block_idx] = reg_mass_mean
            self.batch_reg_write_norm[block_idx] = write_norm

            if self.keep_gpu_for_survival and (block_idx in self.survival_blocks):
                self.batch_x_out_gpu[block_idx] = x_lnd.detach()
                self.batch_write_vecs_gpu[block_idx] = write_vecs.detach()

        return _hook


# ============================================================
# EXP1: hidden-space metrics
# ============================================================

def compute_exp1_hidden_metrics(
    reps_noscam: Dict[str, Dict[str, np.ndarray]],
    reps_synth: Dict[str, Dict[str, np.ndarray]],
    rep_blocks: List[int],
    k: int = 8
) -> Dict[str, Any]:
    """
    reps_*: pair_key -> {"cls_b22":..., "patch_b22":..., "reg_b22":..., ...}

    Returns:
      metrics dict with PCA EVR and principal angles for each rep choice.
    """
    keys = sorted(set(reps_noscam.keys()).intersection(set(reps_synth.keys())))
    if len(keys) < 32:
        raise RuntimeError(f"Too few paired samples for EXP1: {len(keys)}")

    out: Dict[str, Any] = {}

    # We compute for b22/b23 only (or whatever is in rep_blocks).
    # We also compute patch_delta_b23m22 if both present.
    def has(block: int, name: str) -> bool:
        k0 = keys[0]
        return f"{name}_b{block}" in reps_noscam[k0]

    # build patch_delta if possible
    if (22 in rep_blocks) and (23 in rep_blocks) and has(22, "patch") and has(23, "patch"):
        for d in (reps_noscam, reps_synth):
            for kk in keys:
                d[kk]["patch_delta_b23m22"] = d[kk]["patch_b23"] - d[kk]["patch_b22"]

    rep_specs: List[Tuple[str, str]] = []
    for b in rep_blocks:
        rep_specs.extend([
            (f"cls_b{b}", f"reg_b{b}"),
            (f"patch_b{b}", f"reg_b{b}"),
        ])
    if "patch_delta_b23m22" in reps_noscam[keys[0]]:
        rep_specs.append(("patch_delta_b23m22", "reg_b23"))

    for rep_name, reg_name in rep_specs:
        dZ = np.stack([reps_synth[kk][rep_name] - reps_noscam[kk][rep_name] for kk in keys], axis=0)  # [N,D]
        R = np.stack([reps_noscam[kk][reg_name] for kk in keys], axis=0)                               # [N,D]

        evr = pca_evr_topk(dZ, k=k)
        U = orthonormal_basis_from_samples(dZ, k=k)
        V = orthonormal_basis_from_samples(R, k=k)
        ang = principal_angles_deg(U, V, k=k)

        out[rep_name] = dict(
            n_pairs=len(keys),
            pca_evr_topk=evr,
            principal_angles_deg=ang,
        )

    return out


# ============================================================
# EXP2 survival-aware: inject per-head write vectors at block output, run remaining blocks once
# ============================================================

@torch.no_grad()
def compute_survival_scores_for_block(
    visual: torch.nn.Module,
    transformer: torch.nn.Module,
    block_idx: int,
    x_out_lnd: torch.Tensor,            # [T,B,C] after block block_idx
    write_vecs: torch.Tensor,           # [B,H,C] write vectors in width space
) -> torch.Tensor:
    """
    Returns survival score per head: mean ||delta(final_embedding)|| over batch. Shape [H].
    Vectorized across heads: remainder forward sees batch size B*H.
    """
    T, B, C = x_out_lnd.shape
    H = write_vecs.shape[1]

    # baseline remainder
    x_rem = x_out_lnd
    for j in range(block_idx + 1, transformer.layers):
        x_rem = transformer.resblocks[j](x_rem)
    emb_base = tokens_to_final_embedding(visual, x_rem)     # [B,D]

    # expanded stream for all heads
    x_rep = x_out_lnd.repeat_interleave(H, dim=1)           # [T, B*H, C]
    add = write_vecs.reshape(B * H, C)                      # [B*H,C]
    x_rep = x_rep.clone()
    x_rep[0, :, :] = x_rep[0, :, :] + add                  # inject into CLS token

    x_rem2 = x_rep
    for j in range(block_idx + 1, transformer.layers):
        x_rem2 = transformer.resblocks[j](x_rem2)
    emb_rep = tokens_to_final_embedding(visual, x_rem2)     # [B*H,D]
    emb_rep = emb_rep.view(B, H, emb_rep.shape[-1])         # [B,H,D]

    delta = emb_rep - emb_base.unsqueeze(1)                 # [B,H,D]
    scores = delta.norm(dim=-1).mean(dim=0)                 # [H]
    return scores.detach().float().cpu()


# ============================================================
# Dyslexify
# ============================================================

def load_head_mask_config(path: str) -> Dict[int, List[int]]:
    """
    Expected JSON format:
      { "18": [10,12], "22": [10], "23": [10,2] }
    Means: for each block, these heads are DISABLED (masked to 0), others kept.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out: Dict[int, List[int]] = {}
    for k, v in cfg.items():
        out[int(k)] = [int(x) for x in v]
    return out


def apply_head_mask(transformer: torch.nn.Module, block_to_disable_heads: Dict[int, List[int]], n_heads: int):
    """
    Sets blk.attn.head_mask for specified blocks; None otherwise.
    head_mask is 1 for keep, 0 for disable.
    """
    for i, blk in enumerate(transformer.resblocks):
        if i in block_to_disable_heads:
            disable = set(block_to_disable_heads[i])
            mask = torch.ones((n_heads,), dtype=torch.float32)
            for h in disable:
                if 0 <= h < n_heads:
                    mask[h] = 0.0
            blk.attn.head_mask = mask
        else:
            blk.attn.head_mask = None


def build_dyslexify_json_from_stats(
    blocks: List[int],
    attn: np.ndarray,      # [n_blocks, H] raw mean reg_attn_mass
    write: np.ndarray,     # [n_blocks, H] raw mean reg_write_norm
    surv: Optional[np.ndarray],  # [n_blocks, H] mean reg_write_survival, may be None
    topk: int,
    score_mode: str = "survival",
) -> Dict[int, List[int]]:
    """
    Creates mapping block->list[heads_to_disable].

    score_mode:
      - "survival": use surv directly (requires surv)
      - "attn": use attn
      - "write": use write
      - "attn_write": use attn_frac * write_frac
      - "survival_write": use surv * write_frac (requires surv)
      - "survival_attn": use surv * attn_frac (requires surv)
    """
    b2i = {b: i for i, b in enumerate(blocks)}

    # normalize to fractions for combination modes
    attn_frac = attn / (attn.sum(axis=1, keepdims=True) + 1e-12)
    write_frac = write / (write.sum(axis=1, keepdims=True) + 1e-12)

    cfg: Dict[int, List[int]] = {}
    for b in blocks:
        i = b2i[b]
        if score_mode == "survival":
            if surv is None:
                raise ValueError("score_mode=survival requires survival stats.")
            s = surv[i]
        elif score_mode == "attn":
            s = attn[i]
        elif score_mode == "write":
            s = write[i]
        elif score_mode == "attn_write":
            s = attn_frac[i] * write_frac[i]
        elif score_mode == "survival_write":
            if surv is None:
                raise ValueError("score_mode=survival_write requires survival stats.")
            s = surv[i] * write_frac[i]
        elif score_mode == "survival_attn":
            if surv is None:
                raise ValueError("score_mode=survival_attn requires survival stats.")
            s = surv[i] * attn_frac[i]
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

        idx = np.argsort(s)[::-1][:topk]
        cfg[int(b)] = [int(x) for x in idx.tolist()]
    return cfg


# ============================================================
# Plotting helpers
# ============================================================

def _head_styles(n_heads: int = 16):
    """
    Returns list of dicts: [{'color':..., 'linestyle':..., 'marker':...}, ...]
    Deterministic:
      - base colors from tab20 (first 16)
      - head10 forced black
      - even heads solid, odd heads dashed + marker 'x'
    """
    cmap = plt.get_cmap("tab20")
    styles = []
    for h in range(n_heads):
        color = cmap(h % 20)
        if h == 10:
            color = "black"
        linestyle = "-" if (h % 2 == 0) else "--"
        marker = None if (h % 2 == 0) else "x"
        styles.append(dict(color=color, linestyle=linestyle, marker=marker))
    return styles


def plot_line_over_blocks(
    blocks: List[int],
    values: np.ndarray,  # [n_blocks, H]
    out_path: str,
    title: str,
    ylabel: str,
    legend: bool = True,
):
    H = values.shape[1]
    styles = _head_styles(H)

    plt.figure(figsize=(12, 6))
    x = np.array(blocks, dtype=int)
    for h in range(H):
        st = styles[h]
        plt.plot(
            x, values[:, h],
            label=f"h{h}",
            linewidth=1.8,
            **st
        )
    plt.title(title)
    plt.xlabel("block")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if legend:
        plt.legend(ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(
    blocks: List[int],
    values: np.ndarray,  # [n_blocks, H]
    out_path: str,
    title: str,
    xlabel: str = "block",
    ylabel: str = "head",
):
    plt.figure(figsize=(12, 5))
    # imshow wants [H, n_blocks]
    M = values.T
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=np.arange(len(blocks)), labels=[str(b) for b in blocks], rotation=0)
    plt.yticks(ticks=np.arange(M.shape[0]), labels=[f"h{i}" for i in range(M.shape[0])])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pies_for_block(
    block: int,
    attn_row: np.ndarray,   # [H]
    write_row: np.ndarray,  # [H]
    out_path: str,
    title: str,
):
    H = attn_row.shape[0]
    styles = _head_styles(H)
    colors = [styles[h]["color"] for h in range(H)]

    attn_frac = attn_row / (attn_row.sum() + 1e-12)
    write_frac = write_row / (write_row.sum() + 1e-12)

    plt.figure(figsize=(12, 6))
    plt.suptitle(title)

    ax1 = plt.subplot(1, 2, 1)
    ax1.pie(attn_frac, labels=[f"h{h}" for h in range(H)], colors=colors, textprops={"fontsize": 8})
    ax1.set_title("reg_attn_mass (fraction)")

    ax2 = plt.subplot(1, 2, 2)
    ax2.pie(write_frac, labels=[f"h{h}" for h in range(H)], colors=colors, textprops={"fontsize": 8})
    ax2.set_title("reg_write_norm (fraction)")

    plt.tight_layout(rect=[0, 0.0, 1, 0.92])
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter_attn_vs_write_for_block(
    block: int,
    attn_row: np.ndarray,   # [H]
    write_row: np.ndarray,  # [H]
    out_path: str,
    title: str,
):
    H = attn_row.shape[0]
    styles = _head_styles(H)
    colors = [styles[h]["color"] for h in range(H)]

    attn_frac = attn_row / (attn_row.sum() + 1e-12)
    write_frac = write_row / (write_row.sum() + 1e-12)

    plt.figure(figsize=(7, 6))
    for h in range(H):
        plt.scatter(attn_frac[h], write_frac[h], s=60, color=colors[h])
        plt.text(attn_frac[h], write_frac[h], f"h{h}", fontsize=9)

    plt.title(title)
    plt.xlabel("attn_frac (REG mass)")
    plt.ylabel("write_frac (REG write norm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corr_over_blocks(
    blocks: List[int],
    attn: np.ndarray,   # [n_blocks,H]
    write: np.ndarray,  # [n_blocks,H]
    out_path: str,
    title: str,
):
    attn_frac = attn / (attn.sum(axis=1, keepdims=True) + 1e-12)
    write_frac = write / (write.sum(axis=1, keepdims=True) + 1e-12)

    cors = []
    for i in range(attn.shape[0]):
        a = attn_frac[i]
        w = write_frac[i]
        if np.std(a) < 1e-12 or np.std(w) < 1e-12:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(a, w)[0, 1]))

    plt.figure(figsize=(10, 4))
    plt.plot(blocks, cors, linewidth=2.0)
    plt.title(title)
    plt.xlabel("block")
    plt.ylabel("corr(attn_frac, write_frac)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def parse_int_list(spec: str) -> List[int]:
    """
    Accepts:
      "12,13,18,22,23"
      "12-23"
      "12-23,5,7"
    """
    spec = spec.strip()
    if not spec:
        return []
    out: List[int] = []
    parts = spec.split(",")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-")
            a = int(a.strip())
            b = int(b.strip())
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(p))
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def topk_heads(vals: List[float], k: int = 5) -> str:
    idx = np.argsort(np.array(vals))[::-1][:k]
    return ", ".join([f"h{int(i)}={vals[int(i)]:.4f}" for i in idx])


def dump_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ============================================================
# Dyslexify-only evaluation
# ============================================================

@torch.no_grad()
def run_dyslexify_only(
    model_tag: str,
    model_name: str,
    model: torch.nn.Module,
    preprocess,
    device: str,
    buckets: Dict[str, List[PairSample]],
    ordered_keys: List[str],
    map_n: Dict[str, PairSample],
    map_s: Dict[str, PairSample],
    blocks: List[int],
    rep_blocks: List[int],
    register_threshold: float,
    max_registers: int,
    min_registers: int,
    out_dir: str,
    dyslexify_json_path: str,
    batch_size: int,
    dz_blocks: List[int],
    dz_k: int = 8,
) -> None:
    """
    Runs ONLY dyslexify evaluation:
      - Applies head mask from JSON
      - Captures hidden reps (cls_b*, reg_b*) for dz decomposition
      - Computes dz_reg, dz_non and ratio per block in dz_blocks
      - Writes results + minimal plots

    IMPORTANT: Does not rerun full EXP1/EXP2 dumps.
    """
    ensure_dir(out_dir)
    print(Fore.YELLOW + f"[DYSLEXIFY ONLY] model={model_tag} json={dyslexify_json_path}" + Style.RESET_ALL)

    dys_cfg = load_head_mask_config(dyslexify_json_path)
    n_heads = model.visual.transformer.resblocks[0].attn.num_heads
    apply_head_mask(model.visual.transformer, dys_cfg, n_heads=n_heads)

    head_dim = model.visual.transformer.resblocks[0].attn.head_dim
    width = model.visual.transformer.width

    cap = BlockCapture(
        blocks=blocks,
        rep_blocks=rep_blocks,
        survival_blocks=[],
        register_threshold=register_threshold,
        max_registers=max_registers,
        min_registers=min_registers,
        n_heads=n_heads,
        head_dim=head_dim,
        width=width,
        keep_gpu_for_survival=False,
    )

    hooks = []
    for b in blocks:
        if 0 <= b < model.visual.transformer.layers:
            h = model.visual.transformer.resblocks[b].register_forward_hook(cap.hook_for_block(b))
            hooks.append(h)

    reps_noscam: Dict[str, Dict[str, np.ndarray]] = {}
    reps_synth: Dict[str, Dict[str, np.ndarray]] = {}

    bs = batch_size
    n_total = len(ordered_keys)

    def run_variant_batch(variant: str, key_batch: List[str]):
        cap.clear_batch()
        imgs = [map_n[k].image if variant == "NoSCAM" else map_s[k].image for k in key_batch]
        img_t = torch.stack([preprocess(im) for im in imgs], dim=0).to(device)

        _ = model.visual(
            img_t,
            return_trace=False,
            capture_layers=set(blocks),
            return_tokens=False
        )

        for b in rep_blocks:
            if b not in cap.batch_cls_hidden:
                continue
            cls_h = cap.batch_cls_hidden[b].numpy()
            reg_h = cap.batch_reg_pool[b].numpy()

            for i, k in enumerate(key_batch):
                d = reps_noscam if variant == "NoSCAM" else reps_synth
                if k not in d:
                    d[k] = {}
                d[k][f"cls_b{b}"] = cls_h[i]
                d[k][f"reg_b{b}"] = reg_h[i]

    for start in tqdm(range(0, n_total, bs), desc=f"{model_tag} dyslexify batches", leave=False):
        key_batch = ordered_keys[start:start + bs]
        run_variant_batch("NoSCAM", key_batch)
        run_variant_batch("SynthSCAM", key_batch)

    for h in hooks:
        h.remove()

    # dz decomposition per block
    keys = sorted(set(reps_noscam.keys()).intersection(set(reps_synth.keys())))
    rows = []
    for b in dz_blocks:
        rep_name = f"cls_b{b}"
        reg_name = f"reg_b{b}"
        if (keys and (rep_name not in reps_noscam[keys[0]])):
            continue

        dZ = np.stack([reps_synth[kk][rep_name] - reps_noscam[kk][rep_name] for kk in keys], axis=0)  # [N,D]
        R = np.stack([reps_noscam[kk][reg_name] for kk in keys], axis=0)                               # [N,D]
        basis = orthonormal_basis_from_samples(R, k=min(dz_k, R.shape[1], R.shape[0]))
        dz_reg, dz_non = decompose_delta_into_subspace(dZ, basis=basis)
        ratio = dz_reg / (dz_non + 1e-12)

        rows.append(dict(
            model=model_tag,
            block=b,
            n_pairs=len(keys),
            dz_reg_mean=float(np.mean(dz_reg)),
            dz_non_mean=float(np.mean(dz_non)),
            ratio_mean=float(np.mean(ratio)),
            dz_reg_median=float(np.median(dz_reg)),
            dz_non_median=float(np.median(dz_non)),
            ratio_median=float(np.median(ratio)),
        ))

    # write
    out_csv = os.path.join(out_dir, "dyslexify_dz_stats.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("model,block,n_pairs,dz_reg_mean,dz_non_mean,ratio_mean,dz_reg_median,dz_non_median,ratio_median\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['block']},{r['n_pairs']},"
                f"{r['dz_reg_mean']:.6f},{r['dz_non_mean']:.6f},{r['ratio_mean']:.6f},"
                f"{r['dz_reg_median']:.6f},{r['dz_non_median']:.6f},{r['ratio_median']:.6f}\n"
            )

    # minimal plot: ratio_mean vs block
    if rows:
        plt.figure(figsize=(8, 4))
        xs = [r["block"] for r in rows]
        ys = [r["ratio_mean"] for r in rows]
        plt.plot(xs, ys, linewidth=2.0)
        plt.title("Dyslexify dz_reg/dz_non (mean) over blocks")
        plt.xlabel("block")
        plt.ylabel("ratio_mean")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dyslexify_ratio_over_blocks.png"), dpi=200)
        plt.close()

    # summary txt
    out_txt = os.path.join(out_dir, "dyslexify_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Dyslexify-only evaluation\nmodel={model_tag} ({model_name})\njson={dyslexify_json_path}\n\n")
        for r in rows:
            f.write(
                f"block {r['block']}: dz_reg_mean={r['dz_reg_mean']:.6f} dz_non_mean={r['dz_non_mean']:.6f} "
                f"ratio_mean={r['ratio_mean']:.6f}\n"
            )

    print(Fore.CYAN + f"[DYSLEXIFY ONLY] wrote: {out_dir}" + Style.RESET_ALL)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="out_eval_reproduce/mechanistic_proof_v4_critique_regr-norm_cfg")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit_per_variant", type=int, default=None)

    ap.add_argument("--register_threshold", type=float, default=70.0)
    ap.add_argument("--max_registers", type=int, default=32)
    ap.add_argument("--min_registers", type=int, default=1)

    ap.add_argument("--blocks", type=str, default="18-23")
    # which blocks to save hidden reps for EXP1
    ap.add_argument("--rep_blocks", type=str, default="18-23")
    # which blocks to run survival-aware EXP2 on
    ap.add_argument("--survival_blocks", type=str, default="18-23")

    # Dyslexify-only: if set, ONLY run dyslexify eval + outputs (no EXP1/EXP2 rerun)
    ap.add_argument("--dyslexify_json", type=str, default=None) # Set to None, run once, then put "path/to/file.json" here

    # Dyslexify JSON production (from stats)
    ap.add_argument("--dyslexify_blocks", type=str, default="18-23")
    ap.add_argument("--dyslexify_topk", type=int, default=2)
    ap.add_argument("--dyslexify_score", type=str, default="survival",
                    choices=["survival", "attn", "write", "attn_write", "survival_write", "survival_attn"])
    ap.add_argument("--dyslexify_json_name", type=str, default="dyslexify_config.json")

    # dz decomposition blocks for dyslexify-only mode
    ap.add_argument("--dz_blocks", type=str, default="18-23")
    ap.add_argument("--dz_k", type=int, default=4)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blocks = parse_int_list(args.blocks)
    rep_blocks = parse_int_list(args.rep_blocks)
    survival_blocks = parse_int_list(args.survival_blocks)
    dyslexify_blocks = parse_int_list(args.dyslexify_blocks)
    dz_blocks = parse_int_list(args.dz_blocks)

    # Load dataset
    buckets = load_scam_samples(limit_per_variant=args.limit_per_variant)

    keys_n = {s.meta["pair_key"] for s in buckets["NoSCAM"]}
    keys_s = {s.meta["pair_key"] for s in buckets["SynthSCAM"]}
    keys = sorted(keys_n.intersection(keys_s))
    if len(keys) == 0:
        raise RuntimeError("No paired NoSCAM/SynthSCAM keys found.")
    print(Fore.CYAN + f"Paired keys: {len(keys)}" + Style.RESET_ALL)

    map_n = {s.meta["pair_key"]: s for s in buckets["NoSCAM"] if s.meta["pair_key"] in keys}
    map_s = {s.meta["pair_key"]: s for s in buckets["SynthSCAM"] if s.meta["pair_key"] in keys}
    ordered_keys = keys

    # ============================================================
    # If dyslexify_json is provided: ONLY run dyslexify mode and exit.
    # ============================================================
    if args.dyslexify_json is not None:
        # pick WHICH model to apply dyslexify to: user likely runs per-model by editing MODELS or passing different out_dir.
        # We'll run dyslexify for ALL models by default
        for tag, name in MODELS:
            print(Fore.MAGENTA + f"\n=== Dyslexify-only Model: {tag} | {name} ===" + Style.RESET_ALL)
            model, preprocess, _ = load_openai_clip_anything(clip, name, device=device, jit=False, strict=True)
            model = model.to(device).eval().float()

            model_out = os.path.join(args.out_dir, tag, "dyslexify_only")
            ensure_dir(model_out)

            run_dyslexify_only(
                model_tag=tag,
                model_name=name,
                model=model,
                preprocess=preprocess,
                device=device,
                buckets=buckets,
                ordered_keys=ordered_keys,
                map_n=map_n,
                map_s=map_s,
                blocks=blocks,
                rep_blocks=sorted(set(rep_blocks).union(set(dz_blocks))),
                register_threshold=args.register_threshold,
                max_registers=args.max_registers,
                min_registers=args.min_registers,
                out_dir=model_out,
                dyslexify_json_path=args.dyslexify_json,
                batch_size=args.batch_size,
                dz_blocks=dz_blocks,
                dz_k=args.dz_k,
            )
        return  # IMPORTANT: do not run EXP1/EXP2 again

    # ============================================================
    # Otherwise: run full EXP1/EXP2 + plots
    # ============================================================

    for tag, name in MODELS:
        print(Fore.MAGENTA + f"\n=== Model: {tag} | {name} ===" + Style.RESET_ALL)

        model, preprocess, _ = load_openai_clip_anything(clip, name, device=device, jit=False, strict=True)
        model = model.to(device).eval().float()

        attn0 = model.visual.transformer.resblocks[0].attn
        if not hasattr(attn0, "last_v") or not hasattr(attn0, "last_probs"):
            raise RuntimeError("This script requires attnclipdecouple (needs last_v/last_probs capture).")

        n_heads = model.visual.transformer.resblocks[0].attn.num_heads
        head_dim = model.visual.transformer.resblocks[0].attn.head_dim
        width = model.visual.transformer.width

        cap = BlockCapture(
            blocks=blocks,
            rep_blocks=rep_blocks,
            survival_blocks=survival_blocks,
            register_threshold=args.register_threshold,
            max_registers=args.max_registers,
            min_registers=args.min_registers,
            n_heads=n_heads,
            head_dim=head_dim,
            width=width,
            keep_gpu_for_survival=True,
        )

        hooks = []
        for b in blocks:
            if 0 <= b < model.visual.transformer.layers:
                h = model.visual.transformer.resblocks[b].register_forward_hook(cap.hook_for_block(b))
                hooks.append(h)

        reps_noscam: Dict[str, Dict[str, np.ndarray]] = {}
        reps_synth: Dict[str, Dict[str, np.ndarray]] = {}

        sum_reg_attn_mass: Dict[int, torch.Tensor] = {}
        sum_reg_write_norm: Dict[int, torch.Tensor] = {}
        count_batches: Dict[int, int] = {}

        surv_sum: Dict[int, torch.Tensor] = {}
        surv_count: Dict[int, int] = {}

        bs = args.batch_size
        n_total = len(ordered_keys)

        def run_variant_batch(variant: str, key_batch: List[str]):
            cap.clear_batch()
            imgs = [map_n[k].image if variant == "NoSCAM" else map_s[k].image for k in key_batch]
            img_t = torch.stack([preprocess(im) for im in imgs], dim=0).to(device)

            _ = model.visual(
                img_t,
                return_trace=False,
                capture_layers=set(blocks),
                return_tokens=False
            )

            # EXP1 reps (rep_blocks)
            for b in rep_blocks:
                if b not in cap.batch_cls_hidden:
                    continue
                cls_h = cap.batch_cls_hidden[b].numpy()
                patch_h = cap.batch_patch_pool[b].numpy()
                reg_h = cap.batch_reg_pool[b].numpy()

                for i, kk in enumerate(key_batch):
                    d = reps_noscam if variant == "NoSCAM" else reps_synth
                    if kk not in d:
                        d[kk] = {}
                    d[kk][f"cls_b{b}"] = cls_h[i]
                    d[kk][f"patch_b{b}"] = patch_h[i]
                    d[kk][f"reg_b{b}"] = reg_h[i]

            # EXP2 aggregate head stats
            for b in cap.batch_reg_attn_mass.keys():
                if b not in sum_reg_attn_mass:
                    sum_reg_attn_mass[b] = torch.zeros((n_heads,), dtype=torch.float32)
                    sum_reg_write_norm[b] = torch.zeros((n_heads,), dtype=torch.float32)
                    count_batches[b] = 0
                sum_reg_attn_mass[b] += cap.batch_reg_attn_mass[b]
                sum_reg_write_norm[b] += cap.batch_reg_write_norm[b]
                count_batches[b] += 1

            # EXP2 survival-aware
            for b in survival_blocks:
                if b not in cap.batch_x_out_gpu or b not in cap.batch_write_vecs_gpu:
                    continue
                cap.suspend = True  # IMPORTANT: prevent hooks from firing during expanded-batch remainder
                try:
                    scores = compute_survival_scores_for_block(
                        visual=model.visual,
                        transformer=model.visual.transformer,
                        block_idx=b,
                        x_out_lnd=cap.batch_x_out_gpu[b],
                        write_vecs=cap.batch_write_vecs_gpu[b],
                    )  # [H] CPU
                finally:
                    cap.suspend = False

                if b not in surv_sum:
                    surv_sum[b] = torch.zeros((n_heads,), dtype=torch.float32)
                    surv_count[b] = 0
                surv_sum[b] += scores
                surv_count[b] += 1

        for start in tqdm(range(0, n_total, bs), desc=f"{tag} batches", leave=False):
            key_batch = ordered_keys[start:start + bs]
            run_variant_batch("NoSCAM", key_batch)
            run_variant_batch("SynthSCAM", key_batch)

        for h in hooks:
            h.remove()

        # EXP1
        exp1 = compute_exp1_hidden_metrics(reps_noscam, reps_synth, rep_blocks=rep_blocks, k=8)

        # EXP2: means arrays aligned to blocks list order
        blocks_sorted = sorted([b for b in blocks if b in sum_reg_attn_mass])
        attn_mat = np.zeros((len(blocks_sorted), n_heads), dtype=np.float32)
        write_mat = np.zeros((len(blocks_sorted), n_heads), dtype=np.float32)
        for i, b in enumerate(blocks_sorted):
            denom = max(1, count_batches.get(b, 1))
            attn_mat[i] = (sum_reg_attn_mass[b] / denom).numpy()
            write_mat[i] = (sum_reg_write_norm[b] / denom).numpy()

        surv_blocks_sorted = sorted([b for b in survival_blocks if b in surv_sum])
        surv_mat = np.zeros((len(surv_blocks_sorted), n_heads), dtype=np.float32)
        for i, b in enumerate(surv_blocks_sorted):
            denom = max(1, surv_count.get(b, 1))
            surv_mat[i] = (surv_sum[b] / denom).numpy()

        # Build rows for CSV dumps
        exp2_rows = []
        for i, b in enumerate(blocks_sorted):
            for h in range(n_heads):
                exp2_rows.append(dict(
                    model=tag,
                    block=b,
                    head=h,
                    reg_attn_mass=float(attn_mat[i, h]),
                    reg_write_norm=float(write_mat[i, h]),
                ))

        surv_rows = []
        for i, b in enumerate(surv_blocks_sorted):
            for h in range(n_heads):
                surv_rows.append(dict(
                    model=tag,
                    block=b,
                    head=h,
                    reg_write_survival=float(surv_mat[i, h]),
                ))

        # Summary lines
        lines = []
        lines.append(f"=== Model: {tag} | {name} ===")

        # EXP1 condensed print
        for rep_name, m in exp1.items():
            evr = m["pca_evr_topk"]
            ang = m["principal_angles_deg"]
            lines.append(f"EXP1(hidden) {rep_name} PCA top-8 EVR: " + ", ".join([f"{x:.3f}" for x in evr]))
            lines.append(f"EXP1(hidden) {rep_name} principal angles typo vs reg (deg): " + ", ".join([f"{x:.1f}" for x in ang]))

        # EXP2 top heads at blocks of interest if present
        for b in [18, 22, 23]:
            if b in blocks_sorted:
                i = blocks_sorted.index(b)
                lines.append(f"EXP2 b{b} top reg_write_norm heads: " + topk_heads(write_mat[i].tolist(), 5))
                lines.append(f"EXP2 b{b} top reg_attn_mass heads: " + topk_heads(attn_mat[i].tolist(), 5))

        for b in [18, 19, 20, 21, 22, 23]:
            if b in surv_blocks_sorted:
                i = surv_blocks_sorted.index(b)
                lines.append(f"EXP2(survival) b{b} top reg_write_survival heads: " + topk_heads(surv_mat[i].tolist(), 5))

        # Write outputs
        model_out = os.path.join(args.out_dir, tag)
        ensure_dir(model_out)
        plots_dir = os.path.join(model_out, "plots")
        ensure_dir(plots_dir)

        # summary.txt
        with open(os.path.join(model_out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # exp1_hidden_metrics.csv
        with open(os.path.join(model_out, "exp1_hidden_metrics.csv"), "w", encoding="utf-8") as f:
            f.write("model,rep,n_pairs,evr_top8,angles_deg_top8\n")
            for rep_name, m in exp1.items():
                f.write(f"{tag},{rep_name},{m['n_pairs']},\"{m['pca_evr_topk']}\",\"{m['principal_angles_deg']}\"\n")

        # exp2_head_stats_blocks.csv
        with open(os.path.join(model_out, "exp2_head_stats_blocks.csv"), "w", encoding="utf-8") as f:
            f.write("model,block,head,reg_attn_mass,reg_write_norm\n")
            for r in exp2_rows:
                f.write(f"{r['model']},{r['block']},{r['head']},{r['reg_attn_mass']:.6f},{r['reg_write_norm']:.6f}\n")

        # exp2_survival_blocks.csv
        with open(os.path.join(model_out, "exp2_survival_blocks.csv"), "w", encoding="utf-8") as f:
            f.write("model,block,head,reg_write_survival\n")
            for r in surv_rows:
                f.write(f"{r['model']},{r['block']},{r['head']},{r['reg_write_survival']:.6f}\n")

        # ============================================================
        # Plots
        # ============================================================

        # line plots
        plot_line_over_blocks(
            blocks_sorted, attn_mat,
            out_path=os.path.join(plots_dir, f"{tag}_reg_attn_mass_line.png"),
            title=f"{tag}: reg_attn_mass over blocks (per head)",
            ylabel="reg_attn_mass",
            legend=True,
        )
        plot_line_over_blocks(
            blocks_sorted, write_mat,
            out_path=os.path.join(plots_dir, f"{tag}_reg_write_norm_line.png"),
            title=f"{tag}: reg_write_norm over blocks (per head)",
            ylabel="reg_write_norm",
            legend=True,
        )
        if len(surv_blocks_sorted) > 0:
            plot_line_over_blocks(
                surv_blocks_sorted, surv_mat,
                out_path=os.path.join(plots_dir, f"{tag}_reg_write_survival_line.png"),
                title=f"{tag}: reg_write_survival over blocks (per head)",
                ylabel="reg_write_survival",
                legend=True,
            )

        # delta line plot: (attn_frac - write_frac)
        attn_frac = attn_mat / (attn_mat.sum(axis=1, keepdims=True) + 1e-12)
        write_frac = write_mat / (write_mat.sum(axis=1, keepdims=True) + 1e-12)
        delta_frac = attn_frac - write_frac
        plot_line_over_blocks(
            blocks_sorted, delta_frac,
            out_path=os.path.join(plots_dir, f"{tag}_delta_attn_minus_write_frac_line.png"),
            title=f"{tag}: delta(attn_frac - write_frac) over blocks (per head)",
            ylabel="attn_frac - write_frac",
            legend=True,
        )

        # heatmaps
        plot_heatmap(
            blocks_sorted, attn_mat,
            out_path=os.path.join(plots_dir, f"{tag}_reg_attn_mass_heatmap.png"),
            title=f"{tag}: reg_attn_mass heatmap (heads x blocks)",
        )
        plot_heatmap(
            blocks_sorted, write_mat,
            out_path=os.path.join(plots_dir, f"{tag}_reg_write_norm_heatmap.png"),
            title=f"{tag}: reg_write_norm heatmap (heads x blocks)",
        )
        if len(surv_blocks_sorted) > 0:
            plot_heatmap(
                surv_blocks_sorted, surv_mat,
                out_path=os.path.join(plots_dir, f"{tag}_reg_write_survival_heatmap.png"),
                title=f"{tag}: reg_write_survival heatmap (heads x blocks)",
            )
        plot_heatmap(
            blocks_sorted, delta_frac,
            out_path=os.path.join(plots_dir, f"{tag}_delta_attn_minus_write_frac_heatmap.png"),
            title=f"{tag}: delta(attn_frac - write_frac) heatmap (heads x blocks)",
        )

        # per-block pies + scatter
        for i, b in enumerate(blocks_sorted):
            plot_pies_for_block(
                block=b,
                attn_row=attn_mat[i],
                write_row=write_mat[i],
                out_path=os.path.join(plots_dir, f"{tag}_b{b:02d}_pies_attn_vs_write.png"),
                title=f"{tag} block {b}: REG attention vs REG write (fractions)",
            )
            plot_scatter_attn_vs_write_for_block(
                block=b,
                attn_row=attn_mat[i],
                write_row=write_mat[i],
                out_path=os.path.join(plots_dir, f"{tag}_b{b:02d}_scatter_attn_vs_write.png"),
                title=f"{tag} block {b}: attn_frac vs write_frac (per head)",
            )

        # correlation curve over blocks
        plot_corr_over_blocks(
            blocks_sorted, attn_mat, write_mat,
            out_path=os.path.join(plots_dir, f"{tag}_corr_attn_write_over_blocks.png"),
            title=f"{tag}: corr(attn_frac, write_frac) over blocks",
        )

        # ============================================================
        # Dyslexify JSON production 
        # ============================================================

        # Only generate for the block range requested (intersection with computed stats).
        dblocks = [b for b in dyslexify_blocks if b in blocks_sorted]
        if len(dblocks) == 0:
            print(Fore.YELLOW + f"[dyslexify_json] no overlapping blocks between --dyslexify_blocks and computed blocks for {tag}" + Style.RESET_ALL)
        else:
            # slice arrays to those blocks
            idx = [blocks_sorted.index(b) for b in dblocks]
            attn_sub = attn_mat[idx]
            write_sub = write_mat[idx]

            # survival array must align too if needed
            surv_sub = None
            if args.dyslexify_score.startswith("survival"):
                # require survival stats for those blocks
                if not all(b in surv_blocks_sorted for b in dblocks):
                    raise RuntimeError(
                        f"--dyslexify_score={args.dyslexify_score} requires survival computed for all dyslexify_blocks. "
                        f"Missing survival blocks among: {dblocks}"
                    )
                sidx = [surv_blocks_sorted.index(b) for b in dblocks]
                surv_sub = surv_mat[sidx]

            cfg = build_dyslexify_json_from_stats(
                blocks=dblocks,
                attn=attn_sub,
                write=write_sub,
                surv=surv_sub,
                topk=args.dyslexify_topk,
                score_mode=args.dyslexify_score,
            )

            cfg_path = os.path.join(model_out, args.dyslexify_json_name)
            dump_json(cfg_path, {str(k): v for k, v in cfg.items()})

            # also dump a small readable txt with per-block top ranks
            txt_path = os.path.join(model_out, "dyslexify_config_rationale.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Dyslexify config generation\nmodel={tag}\nscore_mode={args.dyslexify_score}\ntopk={args.dyslexify_topk}\nblocks={dblocks}\n\n")
                for b in dblocks:
                    i = dblocks.index(b)
                    f.write(f"block {b} disable: {cfg[b]}\n")
                    f.write(f"  top attn:  {topk_heads((attn_sub[i]).tolist(), 8)}\n")
                    f.write(f"  top write: {topk_heads((write_sub[i]).tolist(), 8)}\n")
                    if surv_sub is not None:
                        f.write(f"  top surv:  {topk_heads((surv_sub[i]).tolist(), 8)}\n")
                    f.write("\n")

            print(Fore.CYAN + f"[dyslexify_json] wrote: {cfg_path}" + Style.RESET_ALL)

        # Console summary
        print(Fore.GREEN + "\n".join(lines[-12:]) + Style.RESET_ALL)
        print(Fore.CYAN + f"Wrote outputs to: {model_out}" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
