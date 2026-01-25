"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

 CLIP reproduction of:
   "Register and [CLS] tokens yield a decoupling of local and global features in large ViTs"
   https://arxiv.org/abs/2505.05892v2

   Registers are detected PER-IMAGE from the *unaltered* forward pass,
   using the *final layer token norms*:

     x = ln_pre(x + pos + cls)
     x = transformer(x)    output of each block
     final_layer_out = output of last block (post-attn+mlp+residual)
     reg_idx = {patch i | ||final_layer_out_patch_i||2 > threshold}

   Then we "hardcode" those indices (per-image) to define:
     - PATCH-ONLY: CLS attends only to patches that are NOT registers
     - REG-ONLY:   CLS attends only to detected register patches

 Threshold is applied per-image, per-run.

 Also includes:
   - CKA(full vs patch-only/reg-only) on flickr8k images
   - Attention "faithfulness" with two metrics:
       * TOKEN mix (attention weights over patch tokens)
       * VALUE mix (attention weights over V + out_proj)  [closer to true attention math]
   - One-shot ImageNet top-5: train linear clf on FULL, eval on FULL/PATCH/REG
   - Rich reg/patch norm stats + reg count distribution

 RESULTS interpretation: 
 Orthogonality (mean cosine among FULL/PATCH/REG). Lower => CLS summary differs more from patch-only/reg-only summaries (stronger decoupling by cosine).
 Attention Faithfulness (VALUE). Higher => CLS token aligns with last-block attention’s V+out_proj output over patches (closer to actual attention math).
 Attention Faithfulness (TOKEN). Higher => CLS token aligns with an attention-weighted mix of *patch tokens* (less faithful than VALUE; still a useful diagnostic).
 One-shot ImageNet top-5 (FULL). Higher => more linearly extractable class signal in the final embedding (proxy for representation usefulness under this protocol).
 CKA(full vs masked). Lower => masked embedding geometry differs more from full (a decoupling/perturbation readout, not inherently quality).

 Crucially: the “patch-only/reg-only” embeddings are not pooled patch embeddings. They’re still CLS embeddings, just under a masked last-layer attention counterfactual.
 -> See also: eval-reproduce-regression-teacher-zs.py
 -> Interpretation: Cosine here means: “If I force CLS (in the last layer) to only look at patches vs only look at regs, the resulting final CLS summary changes by X.”
 -> Or: “If I prevent CLS from seeing regs (or patches) in the last layer, does the final embedding rotate away?”

 Download Flickr8k: Many places, e.g.:
 https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip

"""
from __future__ import annotations

import os
import gc
import json
import math
import random
import argparse
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import attnclipdecouple as clip
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

OUT_DIR = os.path.join("out_eval_reproduce", "decoupling_cls_patch")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--run_scam", action="store_true", help="Run SCAM vs NoSCAM vs SynthSCAM probe")
    ap.add_argument("--scam_max_per_variant", type=int, default=2000)

    ap.add_argument("--register_threshold", type=float, default=70.0)
    ap.add_argument("--cls_mask_includes_self", action="store_true", default=True, help="Allow CLS->CLS when masking. Recommended for stability.")
    ap.add_argument("--fallback_when_no_regs", choices=["cls_only", "patch_only"], default="cls_only", help="If an image has 0 regs (extremely rare): reg-only fallback.")
    ap.add_argument("--reg_leak_blocks", default="12,13,14,15,16,17,18,19,20,21,22,23", help="List of visual blocks to probe for 'REG leakage into FULL' (e.g. '12,13,19-23').")

    # flickr8k
    ap.add_argument("--flickr8k_root", default=r"path/to/Flickr8k/Images")
    ap.add_argument("--flickr8k_max_images", type=int, default=5000)
    ap.add_argument("--flickr8k_batch_size", type=int, default=32)

    # ImageNet
    ap.add_argument("--imagenet_train", default=r"path/to/ILSVRC2012/train")
    ap.add_argument("--imagenet_val", default=r"path/to/ILSVRC2012/val")
    ap.add_argument("--wnid_to_class_json", default=r"path/to/ILSVRC2012/imagenet_wnid_to_class.json")
    ap.add_argument("--imagenet_batch_size", type=int, default=32)


    ap.add_argument("--imagenet_one_shot_repeats", type=int, default=3, help="How many one-shot ImageNet runs with different seeds to average (high variance protocol).")
    ap.add_argument("--imagenet_one_shot_seed_stride", type=int, default=1000, help="Stride added per repeat: seed_i = seed + i*stride.")
    ap.add_argument("--imagenet_one_shot_fixed_split", action="store_true", default=False, help="If set, sample the one-shot train/val split ONCE and reuse it across repeats.")
    
    return ap.parse_args()



IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def ok(msg: str) -> str: return f"{Fore.GREEN}[OK]{Style.RESET_ALL} {msg}"
def warn(msg: str) -> str: return f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}"
def bad(msg: str) -> str: return f"{Fore.RED}[ERR]{Style.RESET_ALL} {msg}"

def parse_int_list(s: str) -> List[int]:
    """
    Accepts: "12,13,19-23" -> [12,13,19,20,21,22,23]
    """
    s = str(s).strip()
    if not s:
        return []
    out: List[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    # dedup, keep order
    seen = set()
    out2 = []
    for x in out:
        if x not in seen:
            out2.append(x)
            seen.add(x)
    return out2

@dataclass
class PairSample:
    image: Image.Image
    correct_label: str
    distractor_label: str
    meta: Dict[str, Any]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # a,b: [B,D]
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA via feature-space formula:
      CKA = || Y^T X ||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )
    with column-wise mean centering.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XT_X = X.T @ X
    YT_Y = Y.T @ Y
    YT_X = Y.T @ X

    hsic = np.sum(YT_X ** 2)
    norm_x = math.sqrt(np.sum(XT_X ** 2)) + 1e-12
    norm_y = math.sqrt(np.sum(YT_Y ** 2)) + 1e-12
    return float(hsic / (norm_x * norm_y))

def topk_extremes(vals: np.ndarray, k: int = 5) -> Dict[str, List[Tuple[float, int]]]:
    if vals.ndim != 1:
        vals = vals.reshape(-1)
    n = vals.shape[0]
    k = min(k, n)
    order = np.argsort(vals)
    bot = [(float(vals[i]), int(i)) for i in order[:k]]
    top = [(float(vals[i]), int(i)) for i in order[-k:][::-1]]
    return {"bottom": bot, "top": top}

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _save_bar_plot(
    out_path: str,
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(max(8, int(0.9 * len(labels) + 6)), 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def _cos_to_deg(c: float) -> float:
    # Cosine -> angle in degrees. Clamp for numeric safety.
    c = float(c)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _save_orthogonality_3d_arrow_single(
    out_path: str,
    alias: str,
    x: float,  # cos(FULL,PATCH)
    y: float,  # cos(FULL,REG)
    z: float,  # cos(PATCH,REG)
) -> None:
    x_deg = _cos_to_deg(x)
    y_deg = _cos_to_deg(y)
    z_deg = _cos_to_deg(z)

    fig = plt.figure(figsize=(7.8, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    # 3 component arrows from origin in DEGREE SPACE
    ax.quiver(0.0, 0.0, 0.0, x_deg, 0.0,  0.0,  arrow_length_ratio=0.08)
    ax.quiver(0.0, 0.0, 0.0, 0.0,  y_deg, 0.0,  arrow_length_ratio=0.08)
    ax.quiver(0.0, 0.0, 0.0, 0.0,  0.0,  z_deg, arrow_length_ratio=0.08)

    # Tip labels: show both cos and deg
    ax.text(x_deg, 0.0,   0.0,  f"FULL–PATCH: cos={x:+.4f}\nθ={x_deg:.1f}°", fontsize=9)
    ax.text(0.0,   y_deg, 0.0,  f"FULL–REG:   cos={y:+.4f}\nθ={y_deg:.1f}°", fontsize=9)
    ax.text(0.0,   0.0,   z_deg, f"PATCH–REG:  cos={z:+.4f}\nθ={z_deg:.1f}°", fontsize=9)

    ax.scatter([x_deg], [y_deg], [z_deg])
    ax.text(
        x_deg, y_deg, z_deg,
        f"{alias}\n"
        f"θ=({x_deg:.1f}°, {y_deg:.1f}°, {z_deg:.1f}°)\n"
        f"cos=({x:+.4f}, {y:+.4f}, {z:+.4f})",
        fontsize=10
    )

    ax.set_title(
        "CLS/PATCH/REG Orthogonality Triplet (angles in degrees; cos shown)\n"
        f"{alias}  |  cos(F,P)={x:+.4f}→{x_deg:.1f}°,  cos(F,R)={y:+.4f}→{y_deg:.1f}°,  cos(P,R)={z:+.4f}→{z_deg:.1f}°"
    )
    ax.set_xlabel("θ(FULL, PATCH) [deg]")
    ax.set_ylabel("θ(FULL, REG) [deg]")
    ax.set_zlabel("θ(PATCH, REG) [deg]")

    max_deg = max(x_deg, y_deg, z_deg)
    lim = min(180.0, max(10.0, 1.35 * max_deg))  # margin + floor so tiny angles still show

    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_zlim(0.0, lim)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def load_scam_samples() -> Dict[str, List[PairSample]]:
    """
    Returns dict: variant -> list[PairSample]
    variants: NoSCAM, SCAM, SynthSCAM
    """
    ds = load_dataset("BLISS-e-V/SCAM", split="train")

    buckets: Dict[str, List[PairSample]] = {v: [] for v in ["NoSCAM", "SCAM", "SynthSCAM"]}

    # NOTE: dataset entries: id, image, object_label, attack_word, postit_area_pct, type, ...
    for entry in ds:
        sid = str(entry["id"])
        variant = None
        for v in buckets.keys():
            if sid.startswith(v):
                variant = v
                break
        if variant is None:
            continue

        img = entry["image"]  # PIL from datasets
        obj = str(entry["object_label"])
        atk = str(entry["attack_word"])

        buckets[variant].append(
            PairSample(
                image=img,
                correct_label=obj,
                distractor_label=atk,
                meta=dict(
                    id=sid,
                    postit_area_pct=float(entry.get("postit_area_pct", 0.0)),
                    type=str(entry.get("type", "")),
                    dataset="SCAM",
                    variant=variant,
                ),
            )
        )
    return buckets

def get_flickr8k_paths(flickr8k_root: str, max_images: Optional[int], seed: int) -> List[str]:
    root = os.path.abspath(os.path.expanduser(flickr8k_root))
    if not os.path.isdir(root):
        raise RuntimeError(f"flickr8k_root is not a directory: {root}")

    def _list_images_flat(d: str) -> List[str]:
        imgs = []
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp) and is_image_file(fp):
                imgs.append(fp)
        return imgs

    def _list_images_recursive(d: str) -> List[str]:
        imgs = []
        for r, _, files in os.walk(d):
            for fn in files:
                fp = os.path.join(r, fn)
                if is_image_file(fp):
                    imgs.append(fp)
        return imgs

    all_imgs = _list_images_flat(root)

    # Fallback: maybe you pointed at the parent folder; search recursively
    if len(all_imgs) == 0:
        all_imgs = _list_images_recursive(root)

    # Fallback: common Flickr8k extracted folder names 
    if len(all_imgs) == 0:
        common = [
            "Images", "images",
            "Flickr8k_Dataset", "Flicker8k_Dataset",
            "Flickr8k", "flickr8k",
        ]
        for name in common:
            cand = os.path.join(root, name)
            if os.path.isdir(cand):
                all_imgs = _list_images_flat(cand)
                if len(all_imgs) == 0:
                    all_imgs = _list_images_recursive(cand)
                if len(all_imgs) > 0:
                    root = cand
                    break

    # Still nothing: give useful debugging context
    if len(all_imgs) == 0:
        try:
            sample = os.listdir(root)[:25]
        except Exception:
            sample = []
        raise RuntimeError(
            f"No images found under flickr8k root: {root}\n"
            f"Dir sample (first 25 entries): {sample}\n"
            f"Expected extensions: {sorted(list(IMG_EXTS))}"
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(all_imgs)
    if max_images is not None:
        all_imgs = all_imgs[:max_images]
    return all_imgs


def list_images_in_dir(root: str) -> List[str]:
    out = []
    for fn in os.listdir(root):
        fp = os.path.join(root, fn)
        if os.path.isfile(fp) and is_image_file(fp):
            out.append(fp)
    return out

def _extras_update_batch(
    extras_accum: Dict[str, Any],
    trace: Dict[str, Any],
    ef: torch.Tensor,
    ep: torch.Tensor,
    er: torch.Tensor,
    id_list: List[str],  # length == batch
) -> None:
    """
    Update extras_accum for one batch.
    ef/ep/er: [B,D], trace tensors on GPU.
    Stores CPU tensors.
    """
    patch_norms = trace["patch_norms_final"]   # [B,P]
    reg_mask = trace["reg_mask"]               # [B,P]
    reg_count = trace["reg_count"]             # [B]

    extras_accum["patch_norms_final"].append(patch_norms.detach().cpu())
    extras_accum["reg_count"].append(reg_count.detach().cpu())
    extras_accum["cls_skip_norm"].append(trace["cls_skip_norm"].float().detach().cpu())
    extras_accum["cls_attn_norm"].append(trace["cls_attn_norm"].float().detach().cpu())
    extras_accum["paths"].extend(id_list)

    # pairwise embedding cosines (FULL/PATCH/REG) in CLIP embed space
    ef_n = ef / (ef.norm(dim=-1, keepdim=True) + 1e-8)
    ep_n = ep / (ep.norm(dim=-1, keepdim=True) + 1e-8)
    er_n = er / (er.norm(dim=-1, keepdim=True) + 1e-8)
    extras_accum["cos_full_patch"].append((ef_n * ep_n).sum(dim=-1).detach().cpu())
    extras_accum["cos_full_reg"].append((ef_n * er_n).sum(dim=-1).detach().cpu())
    extras_accum["cos_patch_reg"].append((ep_n * er_n).sum(dim=-1).detach().cpu())

    # attention faithfulness metrics
    attn = trace["last_attn_probs"]  # [B,H,T,S]
    v = trace["last_v"]              # [B,H,S,D]
    W = trace["out_proj_weight"]
    b = trace["out_proj_bias"]
    cls_attn_true = trace["cls_attn_out_full"]  # [B,C]

    # (A) TOKEN faithfulness in LN1 space
    tokens_ln1 = trace["tokens_ln1_pre_attn"]  # [B,T,C]
    cls_ln1 = tokens_ln1[:, 0, :]              # [B,C]
    patch_ln1 = tokens_ln1[:, 1:, :]           # [B,P,C]

    alpha = attn[:, :, 0, 1:].mean(dim=1)      # [B,P]
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

    z_patch_ln1 = torch.einsum("bp,bpc->bc", alpha, patch_ln1)

    alpha_noregs = alpha.masked_fill(reg_mask, 0.0)
    alpha_noregs = alpha_noregs / (alpha_noregs.sum(dim=1, keepdim=True) + 1e-8)
    z_noregs_ln1 = torch.einsum("bp,bpc->bc", alpha_noregs, patch_ln1)

    alpha_regs = alpha.masked_fill(~reg_mask, 0.0)
    alpha_regs = alpha_regs / (alpha_regs.sum(dim=1, keepdim=True) + 1e-8)
    z_regs_ln1 = torch.einsum("bp,bpc->bc", alpha_regs, patch_ln1)

    extras_accum["faith_token_cos"].append(cosine_sim_batch(cls_ln1, z_patch_ln1).detach().cpu())
    extras_accum["faith_token_cos_noregs"].append(cosine_sim_batch(cls_ln1, z_noregs_ln1).detach().cpu())
    extras_accum["faith_token_cos_regs"].append(cosine_sim_batch(cls_ln1, z_regs_ln1).detach().cpu())

    # (B) VALUE reconstructions
    probs_cls_all = attn[:, :, 0, :]        # [B,H,S]
    probs_patches = probs_cls_all[:, :, 1:] # [B,H,P]
    v_all = v                               # [B,H,S,D]
    v_patches = v_all[:, :, 1:, :]          # [B,H,P,D]

    head_out_all = torch.einsum("bhs,bhsd->bhd", probs_cls_all, v_all)  # [B,H,D]
    attn_out_all = F.linear(head_out_all.reshape(head_out_all.shape[0], -1), W, b)  # [B,C]

    probs_self = probs_cls_all[:, :, 0:1]    # [B,H,1]
    v_self = v_all[:, :, 0:1, :]             # [B,H,1,D]
    head_out_self = torch.einsum("bhi,bhid->bhd", probs_self, v_self)
    attn_out_self = F.linear(head_out_self.reshape(head_out_self.shape[0], -1), W, b)

    reg_mask_h = reg_mask[:, None, :].float()  # [B,1,P]

    head_out_regs_add = torch.einsum("bhp,bhpd->bhd", probs_patches * reg_mask_h, v_patches)
    attn_out_regs_add = F.linear(head_out_regs_add.reshape(head_out_regs_add.shape[0], -1), W, b)

    head_out_noregs_add = torch.einsum("bhp,bhpd->bhd", probs_patches * (1.0 - reg_mask_h), v_patches)
    attn_out_noregs_add = F.linear(head_out_noregs_add.reshape(head_out_noregs_add.shape[0], -1), W, b)

    probs_patch_renorm = probs_patches / (probs_patches.sum(dim=-1, keepdim=True) + 1e-8)
    head_out_patch = torch.einsum("bhp,bhpd->bhd", probs_patch_renorm, v_patches)
    attn_out_patch_renorm = F.linear(head_out_patch.reshape(head_out_patch.shape[0], -1), W, b)

    probs_regs = probs_patches * reg_mask_h
    probs_regs = probs_regs / (probs_regs.sum(dim=-1, keepdim=True) + 1e-8)
    head_out_regs = torch.einsum("bhp,bhpd->bhd", probs_regs, v_patches)
    attn_out_regs_renorm = F.linear(head_out_regs.reshape(head_out_regs.shape[0], -1), W, b)

    probs_noregs = probs_patches * (1.0 - reg_mask_h)
    probs_noregs = probs_noregs / (probs_noregs.sum(dim=-1, keepdim=True) + 1e-8)
    head_out_noregs = torch.einsum("bhp,bhpd->bhd", probs_noregs, v_patches)
    attn_out_noregs_renorm = F.linear(head_out_noregs.reshape(head_out_noregs.shape[0], -1), W, b)

    extras_accum["faith_value_cos"].append(
        cosine_sim_batch(cls_attn_true, attn_out_patch_renorm).detach().cpu()
    )
    extras_accum["faith_value_cos_vs_attnout"].append(
        cosine_sim_batch(cls_attn_true, attn_out_all).detach().cpu()
    )
    extras_accum["faith_value_cos_noregs"].append(
        cosine_sim_batch(cls_attn_true, attn_out_noregs_renorm).detach().cpu()
    )
    extras_accum["faith_value_cos_regs"].append(
        cosine_sim_batch(cls_attn_true, attn_out_regs_renorm).detach().cpu()
    )

    extras_accum["faith_value_add_cos_all"].append(
        cosine_sim_batch(cls_attn_true, attn_out_all).detach().cpu()
    )
    extras_accum["faith_value_add_cos_self"].append(
        cosine_sim_batch(cls_attn_true, attn_out_self).detach().cpu()
    )
    extras_accum["faith_value_add_cos_regs"].append(
        cosine_sim_batch(cls_attn_true, attn_out_regs_add).detach().cpu()
    )
    extras_accum["faith_value_add_cos_noregs"].append(
        cosine_sim_batch(cls_attn_true, attn_out_noregs_add).detach().cpu()
    )

    extras_accum["faith_value_add_norm_true"].append(cls_attn_true.norm(dim=-1).detach().cpu())
    extras_accum["faith_value_add_norm_all"].append(attn_out_all.norm(dim=-1).detach().cpu())
    extras_accum["faith_value_add_norm_self"].append(attn_out_self.norm(dim=-1).detach().cpu())
    extras_accum["faith_value_add_norm_regs"].append(attn_out_regs_add.norm(dim=-1).detach().cpu())
    extras_accum["faith_value_add_norm_noregs"].append(attn_out_noregs_add.norm(dim=-1).detach().cpu())


def _extras_finalize(extras_accum: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(extras_accum)
    for k in list(out.keys()):
        if k == "paths":
            continue
        if len(out[k]) > 0:
            out[k] = torch.cat(out[k], dim=0)
    return out



@torch.no_grad()
def encode_pils_per_image_regs(
    model,
    preprocess,
    pils: List[Image.Image],
    device: str,
    register_threshold: float,
    cls_mask_includes_self: bool,
    batch_size: int,
    fallback_when_no_regs: str,
    return_extras: bool = False,
    ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    fulls, patchs, regs = [], [], []

    if ids is None:
        ids = [f"pil_{i:06d}" for i in range(len(pils))]

    extras_accum = {
        "patch_norms_final": [],
        "reg_count": [],
        "cls_skip_norm": [],
        "cls_attn_norm": [],
        "faith_token_cos": [],
        "faith_token_cos_noregs": [],
        "faith_token_cos_regs": [],
        "faith_value_cos": [],
        "faith_value_cos_vs_attnout": [],
        "faith_value_cos_noregs": [],
        "faith_value_cos_regs": [],
        "faith_value_add_cos_all": [],
        "faith_value_add_cos_self": [],
        "faith_value_add_cos_regs": [],
        "faith_value_add_cos_noregs": [],
        "faith_value_add_norm_true": [],
        "faith_value_add_norm_all": [],
        "faith_value_add_norm_self": [],
        "faith_value_add_norm_regs": [],
        "faith_value_add_norm_noregs": [],
        "cos_full_patch": [],
        "cos_full_reg": [],
        "cos_patch_reg": [],
        "paths": [],
    }

    for i in range(0, len(pils), batch_size):
        chunk = pils[i:i + batch_size]
        chunk_ids = ids[i:i + batch_size]
        imgs = torch.stack([preprocess(im) for im in chunk], dim=0).to(device)

        trace = visual_forward_trace_finalnorm_regs(
            model=model,
            images=imgs,
            register_threshold=register_threshold,
            cls_mask_includes_self=cls_mask_includes_self,
            capture_attn=return_extras,
            fallback_when_no_regs=fallback_when_no_regs
        )

        ef = trace["image_embedding_full"].float()
        ep = trace["image_embedding_patch_only"].float()
        er = trace["image_embedding_reg_only"].float()

        fulls.append(ef.detach().cpu())
        patchs.append(ep.detach().cpu())
        regs.append(er.detach().cpu())

        if return_extras:
            _extras_update_batch(extras_accum, trace, ef, ep, er, chunk_ids)

    out = {
        "full": torch.cat(fulls, dim=0),
        "patch": torch.cat(patchs, dim=0),
        "reg": torch.cat(regs, dim=0),
    }

    if return_extras:
        out["extras"] = _extras_finalize(extras_accum)

    return out



def run_scam_vs_noscam_probe(
    model,
    preprocess,
    scam_buckets: Dict[str, List[Any]],  # list of PairSample-like objects with .image and .meta["id"]
    device: str,
    batch_size: int,
    register_threshold: float,
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str,
    max_per_variant: int = 1000,
    seed: int = 0
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    out = {}

    for variant in ["NoSCAM", "SCAM", "SynthSCAM"]:
        samples = scam_buckets.get(variant, [])
        if not samples:
            out[variant] = {"n": 0}
            continue

        idx = np.arange(len(samples))
        rng.shuffle(idx)
        idx = idx[:min(max_per_variant, len(samples))]

        pils = [samples[i].image for i in idx]
        ids = [str(samples[i].meta.get("id", "")) for i in idx]

        enc = encode_pils_per_image_regs(
            model=model,
            preprocess=preprocess,
            pils=pils,
            ids=ids,
            device=device,
            register_threshold=register_threshold,
            cls_mask_includes_self=cls_mask_includes_self,
            batch_size=batch_size,
            fallback_when_no_regs=fallback_when_no_regs,
            return_extras=True  # requires the full-parity extras copy
        )

        X = enc["full"].numpy()
        Xp = enc["patch"].numpy()
        Xr = enc["reg"].numpy()

        cka_fp = linear_cka(X, Xp)
        cka_fr = linear_cka(X, Xr)

        extras = enc["extras"]

        out[variant] = {
            "n": int(X.shape[0]),
            "cka_full_vs_patch_only": float(cka_fp),
            "cka_full_vs_reg_only": float(cka_fr),
            "cos_full_patch_mean": float(extras["cos_full_patch"].numpy().mean()),
            "cos_full_reg_mean": float(extras["cos_full_reg"].numpy().mean()),
            "cos_patch_reg_mean": float(extras["cos_patch_reg"].numpy().mean()),

            # TOKEN faith (LN1)
            "faith_token_cos_mean": float(extras["faith_token_cos"].numpy().mean()),
            "faith_token_cos_noregs_mean": float(extras["faith_token_cos_noregs"].numpy().mean()),
            "faith_token_cos_regs_mean": float(extras["faith_token_cos_regs"].numpy().mean()),

            # VALUE recon (renormed)
            "faith_value_cos_patch_renorm_mean": float(extras["faith_value_cos"].numpy().mean()),
            "faith_value_sanity_all_sources_mean": float(extras["faith_value_cos_vs_attnout"].numpy().mean()),
            "faith_value_cos_noregs_renorm_mean": float(extras["faith_value_cos_noregs"].numpy().mean()),
            "faith_value_cos_regs_renorm_mean": float(extras["faith_value_cos_regs"].numpy().mean()),

            # Additive decomposition
            "faith_value_add_cos_all_mean": float(extras["faith_value_add_cos_all"].numpy().mean()),
            "faith_value_add_cos_self_mean": float(extras["faith_value_add_cos_self"].numpy().mean()),
            "faith_value_add_cos_regs_mean": float(extras["faith_value_add_cos_regs"].numpy().mean()),
            "faith_value_add_cos_noregs_mean": float(extras["faith_value_add_cos_noregs"].numpy().mean()),

            "faith_value_add_norm_true_mean": float(extras["faith_value_add_norm_true"].numpy().mean()),
            "faith_value_add_norm_self_mean": float(extras["faith_value_add_norm_self"].numpy().mean()),
            "faith_value_add_norm_regs_mean": float(extras["faith_value_add_norm_regs"].numpy().mean()),
            "faith_value_add_norm_noregs_mean": float(extras["faith_value_add_norm_noregs"].numpy().mean()),
        }

    # Delta view: SCAM - NoSCAM (and SynthSCAM - NoSCAM) for the key probe numbers
    base = out.get("NoSCAM", {})
    for v in ["SCAM", "SynthSCAM"]:
        if out.get(v, {}).get("n", 0) and base.get("n", 0):
            out[v + "_minus_NoSCAM"] = {
                "faith_value_add_norm_regs_mean": out[v]["faith_value_add_norm_regs_mean"] - base["faith_value_add_norm_regs_mean"],
                "faith_value_add_norm_noregs_mean": out[v]["faith_value_add_norm_noregs_mean"] - base["faith_value_add_norm_noregs_mean"],
                "faith_value_add_cos_regs_mean": out[v]["faith_value_add_cos_regs_mean"] - base["faith_value_add_cos_regs_mean"],
                "faith_value_add_cos_noregs_mean": out[v]["faith_value_add_cos_noregs_mean"] - base["faith_value_add_cos_noregs_mean"],
                "cos_full_reg_mean": out[v]["cos_full_reg_mean"] - base["cos_full_reg_mean"],
                "cos_full_patch_mean": out[v]["cos_full_patch_mean"] - base["cos_full_patch_mean"],
            }

    return out


class ImageNetOneShotSampler:
    """
    One-shot protocol:
      - 1 training image per class from train/{WNID}/
      - 1 test image per class from val/{WNID}/
    """
    def __init__(self, train_root: str, val_root: str, wnid_to_class_json: str, seed: int):
        self.train_root = train_root
        self.val_root = val_root

        with open(wnid_to_class_json, "r", encoding="utf-8") as f:
            wnid_to_name = json.load(f)

        wnids = sorted(list(wnid_to_name.keys()))
        wnids = [w for w in wnids
                 if os.path.isdir(os.path.join(train_root, w)) and os.path.isdir(os.path.join(val_root, w))]

        if len(wnids) == 0:
            raise RuntimeError("No WNID folders found that exist in both train and val roots.")

        self.wnids = wnids
        self.wnid_to_name = wnid_to_name
        self.rng = np.random.RandomState(seed)

    def _pick_one(self, folder: str) -> str:
        files = list_images_in_dir(folder)
        if not files:
            raise RuntimeError(f"No images in folder: {folder}")
        return files[int(self.rng.randint(0, len(files)))]

    def sample_one_per_class(self) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
        train_paths, val_paths = [], []
        train_labels, val_labels = [], []
        class_names = []

        for i, wnid in enumerate(self.wnids):
            tr_dir = os.path.join(self.train_root, wnid)
            va_dir = os.path.join(self.val_root, wnid)
            train_paths.append(self._pick_one(tr_dir))
            val_paths.append(self._pick_one(va_dir))
            train_labels.append(i)
            val_labels.append(i)
            class_names.append(self.wnid_to_name.get(wnid, wnid))

        return train_paths, train_labels, val_paths, val_labels, class_names


# Classifier (one-shot)
class LinearClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def run_imagenet_one_shot_repeats(
    model,
    preprocess,
    train_root: str,
    val_root: str,
    wnid_to_class_json: str,
    base_seed: int,
    repeats: int,
    seed_stride: int,
    fixed_split: bool,
    device: str,
    batch_size: int,
    register_threshold: float,
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str,
) -> Dict[str, Any]:
    """
    Runs the one-shot ImageNet protocol multiple times with different seeds.

    If fixed_split=False:
      - each repeat re-samples (train_img, val_img) per class AND reinitializes the linear head.

    If fixed_split=True:
      - sample (train_img, val_img) per class ONCE using base_seed, reuse across repeats,
        but still reinitialize and train the linear head each time.
    """
    repeats = int(max(1, repeats))
    seed_stride = int(max(1, seed_stride))

    # optional fixed split
    fixed = None
    if fixed_split:
        sampler0 = ImageNetOneShotSampler(train_root, val_root, wnid_to_class_json, base_seed)
        fixed = sampler0.sample_one_per_class()  # (tr_paths, tr_labels, va_paths, va_labels, class_names)

    runs: List[Dict[str, Any]] = []
    for i in range(repeats):
        seed_i = int(base_seed + i * seed_stride)

        if fixed_split:
            tr_paths, tr_labels, va_paths, va_labels, _ = fixed
        else:
            sampler = ImageNetOneShotSampler(train_root, val_root, wnid_to_class_json, seed_i)
            tr_paths, tr_labels, va_paths, va_labels, _ = sampler.sample_one_per_class()

        # ensure the linear head init/training differs too
        set_seed(seed_i)

        res = run_imagenet_one_shot(
            model=model,
            preprocess=preprocess,
            train_paths=tr_paths,
            train_labels=tr_labels,
            val_paths=va_paths,
            val_labels=va_labels,
            device=device,
            batch_size=batch_size,
            register_threshold=register_threshold,
            cls_mask_includes_self=cls_mask_includes_self,
            fallback_when_no_regs=fallback_when_no_regs,
        )
        res["seed"] = seed_i
        runs.append(res)

    def _mean(xs: List[float]) -> float:
        xs = [float(x) for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    def _std(xs: List[float]) -> float:
        xs = [float(x) for x in xs if np.isfinite(x)]
        return float(np.std(xs)) if xs else float("nan")

    fulls  = [r["one_shot_top5_full"] for r in runs]
    patchs = [r["one_shot_top5_patch_only"] for r in runs]
    regs   = [r["one_shot_top5_reg_only"] for r in runs]

    run_rows = []
    for r in runs:
        run_rows.append({
            "seed": int(r["seed"]),
            "top5_full": float(r["one_shot_top5_full"]),
            "top5_patch_only": float(r["one_shot_top5_patch_only"]),
            "top5_reg_only": float(r["one_shot_top5_reg_only"]),
        })

    out = {
        "imagenet_classes": int(runs[0]["imagenet_classes"]) if runs else 0,
        "repeats": repeats,
        "base_seed": int(base_seed),
        "seed_stride": int(seed_stride),
        "fixed_split": bool(fixed_split),

        # keep full raw runs (verbatim from run_imagenet_one_shot)
        "runs": runs,

        # plus compact rows (just what you'd report)
        "run_rows": run_rows,

        # summary stats
        "one_shot_top5_full_mean": _mean(fulls),
        "one_shot_top5_full_std": _std(fulls),
        "one_shot_top5_patch_only_mean": _mean(patchs),
        "one_shot_top5_patch_only_std": _std(patchs),
        "one_shot_top5_reg_only_mean": _mean(regs),
        "one_shot_top5_reg_only_std": _std(regs),
    }
    return out



def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels[:, None]).any(dim=1).float().mean().item()
    return float(correct)

def train_one_shot_linear(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_classes: int,
    epochs: int = 200,
    lr: float = 0.2,
    wd: float = 1e-4,
    device: str = "cuda"
) -> LinearClassifier:
    in_dim = X_train.shape[1]
    clf = LinearClassifier(in_dim, n_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    # Ensure grads are enabled even if caller is under torch.no_grad()
    with torch.enable_grad():
        clf.train()
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = clf(X_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            opt.step()

    clf.eval()
    return clf

# Visual forward with per-image register indices from final layer norms
@torch.no_grad()
def visual_forward_trace_finalnorm_regs(
    model,
    images: torch.Tensor,  # [B,3,224,224] preprocessed
    register_threshold: float,
    cls_mask_includes_self: bool,
    capture_attn: bool = True,
    fallback_when_no_regs: str = "cls_only",  # "cls_only" | "patch_only"
) -> Dict[str, Any]:
    """
    Steps:
      1) Run UNALTERED visual forward through last block to get final tokens (post_last).
      2) Compute patch norms on final tokens (post_last), define reg_mask = norm > thr (per-image).
      3) Re-run ONLY the last block attention with CLS-row masking informed by those per-image indices:
           - patch-only: keep ~reg_mask
           - reg-only: keep reg_mask
         then finish last block MLP + residual (same as normal).
      4) Project CLS (ln_post + proj) for FULL/PATCH/REG outputs.

    Returns embeddings + masks + norms + attention tensors for faithfulness diagnostics.
    """
    visual = model.visual
    dtype = visual.conv1.weight.dtype
    x = images.to(dtype=dtype)

    # patchify + cls + pos + ln_pre
    x = visual.conv1(x)                              # [B, C, g, g]
    x = x.reshape(x.shape[0], x.shape[1], -1)        # [B, C, P]
    x = x.permute(0, 2, 1)                           # [B, P, C]
    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)             # [B, 1+P, C]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)                             # [B, 1+P, C]

    # transformer operates in [T,B,C]
    x_lnd = x.permute(1, 0, 2)                       # [T,B,C]

    last_idx = visual.transformer.layers - 1
    x_pre_last = visual.transformer.forward_until(x_lnd, last_idx)  # [T,B,C]
    last_block = visual.transformer.resblocks[last_idx]

    # full last block (capture attn)
    ln1 = last_block.ln_1(x_pre_last)  # [T,B,C]

    # expose LN1 tokens (the attention "working space")
    tokens_ln1 = ln1.permute(1, 0, 2).contiguous()   # [B,T,C]

    attn_out_full, attn_w_full = last_block.attn(
        ln1, ln1, ln1,
        need_weights=True,
        attn_mask=None,
        capture=capture_attn,
        cls_src_keep_mask=None,
        cls_mask_includes_self=cls_mask_includes_self
    )
    x_after_attn = x_pre_last + attn_out_full
    x_full = x_after_attn + last_block.mlp(last_block.ln_2(x_after_attn))  # [T,B,C]
    tokens_post_last = x_full.permute(1, 0, 2)  # [B,T,C]

    # Register indices from FINAL LAYER norms
    patch_tokens_final = tokens_post_last[:, 1:, :]            # [B,P,C]
    patch_norms_final = patch_tokens_final.norm(dim=-1)        # [B,P]
    reg_mask = patch_norms_final > register_threshold          # [B,P] bool
    reg_count = reg_mask.sum(dim=1)                            # [B]

    B, P = reg_mask.shape
    src_len = 1 + P

    keep_patch_only = torch.ones((B, src_len), dtype=torch.bool, device=images.device)
    keep_reg_only = torch.ones((B, src_len), dtype=torch.bool, device=images.device)

    keep_patch_only[:, 1:] = ~reg_mask
    keep_reg_only[:, 1:] = reg_mask

    # handle edge cases per-image (no regs or all regs)
    no_regs = (reg_count == 0)
    if no_regs.any():
        if fallback_when_no_regs == "patch_only":
            keep_reg_only[no_regs, 1:] = keep_patch_only[no_regs, 1:]
        else:
            if not cls_mask_includes_self:
                keep_reg_only[no_regs, 1:] = keep_patch_only[no_regs, 1:]

    all_regs = (reg_count == P)
    if all_regs.any():
        if not cls_mask_includes_self:
            keep_patch_only[all_regs, 1:] = keep_reg_only[all_regs, 1:]

    # rerun last attention with patch-only mask (CLS row only)
    ln1_shared = last_block.ln_1(x_pre_last)

    attn_out_patch, _ = last_block.attn(
        ln1_shared, ln1_shared, ln1_shared,
        need_weights=False,
        attn_mask=None,
        capture=False,
        cls_src_keep_mask=keep_patch_only,
        cls_mask_includes_self=cls_mask_includes_self
    )
    x_after_attn_patch = x_pre_last + attn_out_patch
    x_patch = x_after_attn_patch + last_block.mlp(last_block.ln_2(x_after_attn_patch))
    tokens_patch = x_patch.permute(1, 0, 2)  # [B,T,C]

    # rerun last attention with reg-only mask
    attn_out_reg, _ = last_block.attn(
        ln1_shared, ln1_shared, ln1_shared,
        need_weights=False,
        attn_mask=None,
        capture=False,
        cls_src_keep_mask=keep_reg_only,
        cls_mask_includes_self=cls_mask_includes_self
    )
    x_after_attn_reg = x_pre_last + attn_out_reg
    x_reg = x_after_attn_reg + last_block.mlp(last_block.ln_2(x_after_attn_reg))
    tokens_reg = x_reg.permute(1, 0, 2)

    # final projected embeddings (same as visual.forward)
    cls_full = visual.ln_post(tokens_post_last[:, 0, :])
    cls_patch = visual.ln_post(tokens_patch[:, 0, :])
    cls_reg = visual.ln_post(tokens_reg[:, 0, :])

    if visual.proj is not None:
        cls_full = cls_full @ visual.proj
        cls_patch = cls_patch @ visual.proj
        cls_reg = cls_reg @ visual.proj

    # skip-vs-attn norms (last block, CLS only, pre-MLP)
    cls_skip = x_pre_last[0]       # [B,C]
    cls_attn = attn_out_full[0]    # [B,C] (out_proj applied)
    cls_skip_norm = cls_skip.norm(dim=-1)
    cls_attn_norm = cls_attn.norm(dim=-1)

    # captured attention tensors
    last_probs = last_block.attn.last_probs  # [B,H,T,S]
    last_v = last_block.attn.last_v          # [B,H,S,D]
    last_logits = last_block.attn.last_logits

    return {
        "image_embedding_full": cls_full,
        "image_embedding_patch_only": cls_patch,
        "image_embedding_reg_only": cls_reg,

        "tokens_post_last_full": tokens_post_last,
        "tokens_ln1_pre_attn": tokens_ln1,
        "patch_norms_final": patch_norms_final,
        "reg_mask": reg_mask,
        "reg_count": reg_count,

        "cls_skip_norm": cls_skip_norm,
        "cls_attn_norm": cls_attn_norm,

        "last_attn_probs": last_probs,
        "last_attn_logits": last_logits,
        "last_v": last_v,
        "cls_attn_out_full": cls_attn,  # [B,C]
        "out_proj_weight": last_block.attn.out_proj.weight,
        "out_proj_bias": last_block.attn.out_proj.bias,
    }

@torch.no_grad()
def _block_forward_standard(block, x: torch.Tensor) -> torch.Tensor:
    """
    Best-effort: use block(x) if supported, else manual residual attention + MLP.
    x: [T,B,C]
    """
    try:
        return block(x)
    except TypeError:
        # manual (OpenAI-CLIP style)
        ln1 = block.ln_1(x)
        attn_out, _ = block.attn(ln1, ln1, ln1, need_weights=False, attn_mask=None)
        x2 = x + attn_out
        x3 = x2 + block.mlp(block.ln_2(x2))
        return x3


@torch.no_grad()
def _block_forward_cls_mask(
    block,
    x: torch.Tensor,                      # [T,B,C]
    cls_src_keep_mask: torch.Tensor,      # [B, 1+P] bool
    cls_mask_includes_self: bool,
) -> torch.Tensor:
    """
    Run one block, but apply CLS-row masking in attention (your attnclipdecouple API).
    """
    ln1 = block.ln_1(x)
    attn_out, _ = block.attn(
        ln1, ln1, ln1,
        need_weights=False,
        attn_mask=None,
        capture=False,
        cls_src_keep_mask=cls_src_keep_mask,
        cls_mask_includes_self=cls_mask_includes_self
    )
    x2 = x + attn_out
    x3 = x2 + block.mlp(block.ln_2(x2))
    return x3


@torch.no_grad()
def _make_keep_masks_from_reg_mask(
    reg_mask: torch.Tensor,               # [B,P] bool
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: keep_patch_only, keep_reg_only, reg_count
    Shapes:  [B,1+P], [B,1+P], [B]
    """
    B, P = reg_mask.shape
    src_len = 1 + P
    device = reg_mask.device

    reg_count = reg_mask.sum(dim=1)  # [B]
    keep_patch_only = torch.ones((B, src_len), dtype=torch.bool, device=device)
    keep_reg_only = torch.ones((B, src_len), dtype=torch.bool, device=device)

    keep_patch_only[:, 1:] = ~reg_mask
    keep_reg_only[:, 1:] = reg_mask

    no_regs = (reg_count == 0)
    if no_regs.any():
        if fallback_when_no_regs == "patch_only":
            keep_reg_only[no_regs, 1:] = keep_patch_only[no_regs, 1:]
        else:
            if not cls_mask_includes_self:
                keep_reg_only[no_regs, 1:] = keep_patch_only[no_regs, 1:]

    all_regs = (reg_count == P)
    if all_regs.any():
        if not cls_mask_includes_self:
            keep_patch_only[all_regs, 1:] = keep_reg_only[all_regs, 1:]

    return keep_patch_only, keep_reg_only, reg_count


@torch.no_grad()
def encode_paths_reg_leak_over_blocks(
    model,
    preprocess,
    paths: List[str],
    device: str,
    register_threshold: float,
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str,
    batch_size: int,
    block_indices: List[int],
) -> Dict[str, Any]:
    """
    Returns CPU tensors:
      full: [N,D]
      per_block[b]["patch"]: [N,D]
      per_block[b]["reg"]:   [N,D]
      per_block[b]["reg_count"]: [N]
      per_block[b]["cos_*"]: [N]

    NOTE: uses visual_forward_trace_reg_leak_over_blocks(), which now defines
    reg_mask ONCE from the final-layer unmodified pass, then reuses it across blocks.
    """
    fulls: List[torch.Tensor] = []

    blocks = sorted(list(dict.fromkeys([int(b) for b in block_indices])))
    per_block_accum: Dict[int, Dict[str, List[torch.Tensor]]] = {
        b: {"patch": [], "reg": [], "reg_count": [], "cos_full_patch": [], "cos_full_reg": [], "cos_patch_reg": []}
        for b in blocks
    }

    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        pil = load_pil_images(chunk)
        imgs = torch.stack([preprocess(im) for im in pil], dim=0).to(device)

        trace = visual_forward_trace_reg_leak_over_blocks(
            model=model,
            images=imgs,
            register_threshold=register_threshold,
            cls_mask_includes_self=cls_mask_includes_self,
            fallback_when_no_regs=fallback_when_no_regs,
            block_indices=blocks,
        )

        fulls.append(trace["full"].detach().cpu())

        pb = trace["per_block"]
        for b in blocks:
            if b not in pb:
                continue
            per_block_accum[b]["patch"].append(pb[b]["patch"].detach().cpu())
            per_block_accum[b]["reg"].append(pb[b]["reg"].detach().cpu())
            per_block_accum[b]["reg_count"].append(pb[b]["reg_count"].detach().cpu())
            per_block_accum[b]["cos_full_patch"].append(pb[b]["cos_full_patch"].detach().cpu())
            per_block_accum[b]["cos_full_reg"].append(pb[b]["cos_full_reg"].detach().cpu())
            per_block_accum[b]["cos_patch_reg"].append(pb[b]["cos_patch_reg"].detach().cpu())

    out: Dict[str, Any] = {"full": torch.cat(fulls, dim=0), "per_block": {}}
    for b in blocks:
        out["per_block"][b] = {k: torch.cat(v, dim=0) for k, v in per_block_accum[b].items()}
    return out


@torch.no_grad()
def visual_forward_trace_reg_leak_over_blocks(
    model,
    images: torch.Tensor,                 # [B,3,224,224]
    register_threshold: float,
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str,
    block_indices: List[int],
) -> Dict[str, Any]:
    """
    CLEAN reg-leak probe (fixes circular conditioning / "mask leakage"):

    1) Run ONE unmodified forward pass through all blocks to get:
         - final embedding FULL
         - final-layer patch norms -> reg_mask_final (paper's definition)
       Also cache x_pre at each target block b (state entering block b).

    2) For each target block b:
         - re-run ONLY block b with CLS-row masking using reg_mask_final:
              PATCH@b: CLS attends only to NON-reg patches (and optionally CLS->CLS)
              REG@b:   CLS attends only to reg patches (and optionally CLS->CLS)
         - run the suffix blocks (b+1..end) unmodified
         - read out FINAL embeddings for PATCH@b and REG@b

    This answers a non-circular question:
      "Using registers defined by the *unaltered final layer*, how much do those
       tokens influence the final embedding when CLS is restricted at block b?"

    Returns:
      full: [B,D] final FULL embedding
      per_block[b]:
        patch: [B,D] final embedding after PATCH@b masking
        reg:   [B,D] final embedding after REG@b masking
        reg_count: [B] number of regs from final-layer definition
        cos_full_patch/cos_full_reg/cos_patch_reg: [B]
      plus (debug):
        reg_mask_final: [B,P] bool
        patch_norms_final: [B,P]
    """
    visual = model.visual
    blocks = list(visual.transformer.resblocks)
    L = len(blocks)

    # sanitize
    block_indices = [int(b) for b in block_indices]
    block_indices = [b for b in block_indices if 0 <= b < L]
    block_indices = sorted(list(dict.fromkeys(block_indices)))  # dedup + stable

    dtype = visual.conv1.weight.dtype
    x = images.to(dtype=dtype)

    # patchify + cls + pos + ln_pre
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)   # [B,C,P]
    x = x.permute(0, 2, 1)                      # [B,P,C]
    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)        # [B,1+P,C]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)                         # [B,1+P,C]
    x = x.permute(1, 0, 2)                       # [T,B,C]

    def _project_cls(x_lnd: torch.Tensor) -> torch.Tensor:
        tokens = x_lnd.permute(1, 0, 2)          # [B,T,C]
        cls = visual.ln_post(tokens[:, 0, :])
        if visual.proj is not None:
            cls = cls @ visual.proj
        return cls

    # baseline full pass ONCE + cache x_pre at target b's
    x_pre_cache: Dict[int, torch.Tensor] = {}
    for i, blk in enumerate(blocks):
        if i in block_indices:
            # Safe: no in-place ops expected; store entering-state tensor
            x_pre_cache[i] = x.detach()
        x = _block_forward_standard(blk, x)

    # baseline final embedding
    emb_full = _project_cls(x).float()  # [B,D]
    ef_n = emb_full / (emb_full.norm(dim=-1, keepdim=True) + 1e-8)

    # final-layer registers (paper definition)
    tokens_final = x.permute(1, 0, 2)            # [B,T,C]
    patch_tokens_final = tokens_final[:, 1:, :]  # [B,P,C]
    patch_norms_final = patch_tokens_final.norm(dim=-1)  # [B,P]
    reg_mask_final = patch_norms_final > register_threshold  # [B,P]
    keep_patch_only, keep_reg_only, reg_count = _make_keep_masks_from_reg_mask(
        reg_mask=reg_mask_final,
        cls_mask_includes_self=cls_mask_includes_self,
        fallback_when_no_regs=fallback_when_no_regs,
    )

    # branch per target block using the SAME reg_mask_final
    per_block: Dict[int, Dict[str, Any]] = {}
    for b in block_indices:
        x_pre = x_pre_cache[b]  # [T,B,C]

        # PATCH@b
        x_patch = _block_forward_cls_mask(blocks[b], x_pre, keep_patch_only, cls_mask_includes_self)
        for j in range(b + 1, L):
            x_patch = _block_forward_standard(blocks[j], x_patch)
        emb_patch = _project_cls(x_patch).float()

        # REG@b
        x_reg = _block_forward_cls_mask(blocks[b], x_pre, keep_reg_only, cls_mask_includes_self)
        for j in range(b + 1, L):
            x_reg = _block_forward_standard(blocks[j], x_reg)
        emb_reg = _project_cls(x_reg).float()

        ep_n = emb_patch / (emb_patch.norm(dim=-1, keepdim=True) + 1e-8)
        er_n = emb_reg / (emb_reg.norm(dim=-1, keepdim=True) + 1e-8)

        per_block[b] = {
            "patch": emb_patch,
            "reg": emb_reg,
            "reg_count": reg_count.detach(),
            "cos_full_patch": (ef_n * ep_n).sum(dim=-1).detach(),
            "cos_full_reg": (ef_n * er_n).sum(dim=-1).detach(),
            "cos_patch_reg": (ep_n * er_n).sum(dim=-1).detach(),
        }

    return {
        "full": emb_full,
        "per_block": per_block,
        # debug
        "reg_mask_final": reg_mask_final,
        "patch_norms_final": patch_norms_final,
    }



def run_flickr8k_reg_leak_over_blocks(
    model,
    preprocess,
    flickr8k_paths: List[str],
    device: str,
    batch_size: int,
    register_threshold: float,
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str,
    block_indices: List[int],
) -> Dict[str, Any]:
    """
    Reg-leak probe over blocks, with CLEAN register definition:
    - regs are defined once per image from the final-layer unmodified pass
    - masking at each block b uses that same reg set
    """
    enc = encode_paths_reg_leak_over_blocks(
        model=model,
        preprocess=preprocess,
        paths=flickr8k_paths,
        device=device,
        register_threshold=register_threshold,
        cls_mask_includes_self=cls_mask_includes_self,
        fallback_when_no_regs=fallback_when_no_regs,
        batch_size=batch_size,
        block_indices=block_indices,
    )

    X_full = enc["full"].numpy()  # [N,D]
    out: Dict[str, Any] = {"blocks": {}, "block_list": list(enc["per_block"].keys())}

    for b, d in enc["per_block"].items():
        X_patch = d["patch"].numpy()
        X_reg = d["reg"].numpy()

        out["blocks"][str(b)] = {
            "n": int(X_full.shape[0]),
            "cos_full_patch_mean": float(d["cos_full_patch"].numpy().mean()),
            "cos_full_reg_mean": float(d["cos_full_reg"].numpy().mean()),
            "cos_patch_reg_mean": float(d["cos_patch_reg"].numpy().mean()),
            "reg_count_mean": float(d["reg_count"].numpy().astype(np.float64).mean()),
            "cka_full_vs_patch": float(linear_cka(X_full, X_patch)),
            "cka_full_vs_reg": float(linear_cka(X_full, X_reg)),
        }

    return out



def load_pil_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs


@torch.no_grad()
def encode_paths_per_image_regs(
    model,
    preprocess,
    paths: List[str],
    device: str,
    register_threshold: float,
    cls_mask_includes_self: bool,
    batch_size: int,
    fallback_when_no_regs: str,
    return_extras: bool = False
) -> Dict[str, Any]:
    fulls, patchs, regs = [], [], []

    extras_accum = {
        "patch_norms_final": [],
        "reg_count": [],
        "cls_skip_norm": [],
        "cls_attn_norm": [],
        "faith_token_cos": [],
        "faith_token_cos_noregs": [],
        "faith_token_cos_regs": [],
        "faith_value_cos": [],
        "faith_value_cos_vs_attnout": [],
        "faith_value_cos_noregs": [],
        "faith_value_cos_regs": [],
        "faith_value_add_cos_all": [],
        "faith_value_add_cos_self": [],
        "faith_value_add_cos_regs": [],
        "faith_value_add_cos_noregs": [],
        "faith_value_add_norm_true": [],
        "faith_value_add_norm_all": [],
        "faith_value_add_norm_self": [],
        "faith_value_add_norm_regs": [],
        "faith_value_add_norm_noregs": [],
        "cos_full_patch": [],
        "cos_full_reg": [],
        "cos_patch_reg": [],
        "paths": [],
    }

    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        pil = load_pil_images(chunk)
        imgs = torch.stack([preprocess(im) for im in pil], dim=0).to(device)

        trace = visual_forward_trace_finalnorm_regs(
            model=model,
            images=imgs,
            register_threshold=register_threshold,
            cls_mask_includes_self=cls_mask_includes_self,
            capture_attn=return_extras,
            fallback_when_no_regs=fallback_when_no_regs
        )

        ef = trace["image_embedding_full"].float()
        ep = trace["image_embedding_patch_only"].float()
        er = trace["image_embedding_reg_only"].float()

        fulls.append(ef.detach().cpu())
        patchs.append(ep.detach().cpu())
        regs.append(er.detach().cpu())

        if return_extras:
            _extras_update_batch(extras_accum, trace, ef, ep, er, id_list=chunk)

    out = {
        "full": torch.cat(fulls, dim=0),
        "patch": torch.cat(patchs, dim=0),
        "reg": torch.cat(regs, dim=0),
    }

    if return_extras:
        out["extras"] = _extras_finalize(extras_accum)

    return out



def summarize_reg_patch_stats(patch_norms: np.ndarray, reg_count: np.ndarray, threshold: float) -> Dict[str, Any]:
    """
    patch_norms: [N,P] final-layer patch norms (post last block)
    reg_count:   [N] number of patches with norm > threshold
    """
    N, P = patch_norms.shape
    reg_mask = patch_norms > threshold
    patch_mask = ~reg_mask

    reg_norms = patch_norms[reg_mask]
    patch_norms_only = patch_norms[patch_mask]

    out: Dict[str, Any] = {}
    out["reg_count_mean"] = float(reg_count.mean())
    out["reg_count_min"] = int(reg_count.min())
    out["reg_count_max"] = int(reg_count.max())
    out["reg_count_zero_frac"] = float((reg_count == 0).mean())
    out["patch_count_mean"] = float((P - reg_count).mean())
    out["patch_count_min"] = int((P - reg_count).min())
    out["patch_count_max"] = int((P - reg_count).max())

    out["patch_norm_mean"] = float(patch_norms_only.mean()) if patch_norms_only.size else float("nan")
    out["patch_norm_min"] = float(patch_norms_only.min()) if patch_norms_only.size else float("nan")
    out["patch_norm_max"] = float(patch_norms_only.max()) if patch_norms_only.size else float("nan")

    out["reg_norm_mean"] = float(reg_norms.mean()) if reg_norms.size else float("nan")
    out["reg_norm_min"] = float(reg_norms.min()) if reg_norms.size else float("nan")
    out["reg_norm_max"] = float(reg_norms.max()) if reg_norms.size else float("nan")

    # per-image min/max among patches/regs
    patch_min_per = np.full((N,), np.nan, dtype=np.float64)
    patch_max_per = np.full((N,), np.nan, dtype=np.float64)
    reg_min_per = np.full((N,), np.nan, dtype=np.float64)
    reg_max_per = np.full((N,), np.nan, dtype=np.float64)

    for i in range(N):
        pn = patch_norms[i]
        m_reg = pn > threshold
        m_patch = ~m_reg
        if m_patch.any():
            patch_min_per[i] = float(pn[m_patch].min())
            patch_max_per[i] = float(pn[m_patch].max())
        if m_reg.any():
            reg_min_per[i] = float(pn[m_reg].min())
            reg_max_per[i] = float(pn[m_reg].max())

    out["per_image_patch_min_mean"] = float(np.nanmean(patch_min_per))
    out["per_image_patch_max_mean"] = float(np.nanmean(patch_max_per))
    out["per_image_reg_min_mean"] = float(np.nanmean(reg_min_per)) if np.isfinite(reg_min_per).any() else float("nan")
    out["per_image_reg_max_mean"] = float(np.nanmean(reg_max_per)) if np.isfinite(reg_max_per).any() else float("nan")

    # reg-count histogram (0..10 and 10+)
    bins = {}
    for k in range(0, 11):
        bins[str(k)] = int((reg_count == k).sum())
    bins["11_plus"] = int((reg_count >= 11).sum())
    out["reg_count_hist"] = bins

    margins = [
        ("below_10", threshold - 10.0, threshold),       # [thr-10, thr]
        ("above_30", threshold, threshold + 30.0),       # (thr, thr+30]
    ]

    margin_out = {}
    for name, lo, hi in margins:
        if "below" in name:
            m = (patch_norms >= lo) & (patch_norms <= hi)
        else:
            m = (patch_norms > lo) & (patch_norms <= hi)

        per_img = m.sum(axis=1).astype(np.int32)  # [N]
        margin_out[f"{name}_token_count_mean"] = float(per_img.mean())
        margin_out[f"{name}_token_count_max"] = int(per_img.max())
        margin_out[f"{name}_image_frac_any"] = float((per_img > 0).mean())

    out["threshold_margins"] = margin_out

    return out


def run_flickr8k_eval(
    model,
    preprocess,
    flickr8k_paths: List[str],
    device: str,
    batch_size: int,
    register_threshold: float,
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str
) -> Dict[str, Any]:
    enc = encode_paths_per_image_regs(
        model=model,
        preprocess=preprocess,
        paths=flickr8k_paths,
        device=device,
        register_threshold=register_threshold,
        cls_mask_includes_self=cls_mask_includes_self,
        batch_size=batch_size,
        fallback_when_no_regs=fallback_when_no_regs,
        return_extras=True
    )

    X = enc["full"].numpy()
    Xp = enc["patch"].numpy()
    Xr = enc["reg"].numpy()

    cka_fp = linear_cka(X, Xp)
    cka_fr = linear_cka(X, Xr)

    extras = enc["extras"]
    patch_norms = extras["patch_norms_final"].numpy()          # [N,P]
    reg_count = extras["reg_count"].numpy().astype(np.int32)   # [N]

    stats = summarize_reg_patch_stats(patch_norms, reg_count, register_threshold)

    # skip/attn ratio
    cls_skip_norm = extras["cls_skip_norm"].numpy()
    cls_attn_norm = extras["cls_attn_norm"].numpy()
    attn_skip_ratio = cls_attn_norm / (cls_skip_norm + 1e-8)

    # faithfulness
    tok_cos = extras["faith_token_cos"].numpy()
    tok_cos_nr = extras["faith_token_cos_noregs"].numpy()
    tok_cos_r = extras["faith_token_cos_regs"].numpy()

    val_cos = extras["faith_value_cos"].numpy()
    val_cos_attn = extras["faith_value_cos_vs_attnout"].numpy()
    val_cos_nr = extras["faith_value_cos_noregs"].numpy()
    val_cos_r = extras["faith_value_cos_regs"].numpy()

    # pairwise embedding cosines
    c_fp = extras["cos_full_patch"].numpy()
    c_fr = extras["cos_full_reg"].numpy()
    c_pr = extras["cos_patch_reg"].numpy()

    # extremes with paths (for token faithfulness)
    paths = extras["paths"]
    token_ext = topk_extremes(tok_cos, k=5)
    worst_paths = [(token_ext["bottom"][i][0], paths[token_ext["bottom"][i][1]]) for i in range(len(token_ext["bottom"]))]

    # additive decomposition probe (no renorm)
    add_cos_all = extras["faith_value_add_cos_all"].numpy()
    add_cos_self = extras["faith_value_add_cos_self"].numpy()
    add_cos_regs = extras["faith_value_add_cos_regs"].numpy()
    add_cos_noregs = extras["faith_value_add_cos_noregs"].numpy()

    add_norm_true = extras["faith_value_add_norm_true"].numpy()
    add_norm_self = extras["faith_value_add_norm_self"].numpy()
    add_norm_regs = extras["faith_value_add_norm_regs"].numpy()
    add_norm_noregs = extras["faith_value_add_norm_noregs"].numpy()


    return {
        "flickr8k_n": X.shape[0],
        "cka_full_vs_patch_only": float(cka_fp),
        "cka_full_vs_reg_only": float(cka_fr),

        "attn_skip_ratio_mean": float(attn_skip_ratio.mean()),
        "attn_skip_ratio_median": float(np.median(attn_skip_ratio)),

        "faith_token_cos_mean": float(tok_cos.mean()),
        "faith_token_cos_min": float(tok_cos.min()),
        "faith_token_cos_noregs_mean": float(tok_cos_nr.mean()),
        "faith_token_cos_regs_mean": float(tok_cos_r.mean()),

        "faith_value_cos_mean": float(val_cos.mean()),
        "faith_value_cos_min": float(val_cos.min()),
        "faith_value_cos_vs_attnout_mean": float(val_cos_attn.mean()),
        "faith_value_cos_noregs_mean": float(val_cos_nr.mean()),
        "faith_value_cos_regs_mean": float(val_cos_r.mean()),

        "cos_full_patch_mean": float(c_fp.mean()),
        "cos_full_patch_min": float(c_fp.min()),
        "cos_full_patch_max": float(c_fp.max()),
        "cos_full_reg_mean": float(c_fr.mean()),
        "cos_full_reg_min": float(c_fr.min()),
        "cos_full_reg_max": float(c_fr.max()),
        "cos_patch_reg_mean": float(c_pr.mean()),
        "cos_patch_reg_min": float(c_pr.min()),
        "cos_patch_reg_max": float(c_pr.max()),

        "token_faithfulness_worst_5": worst_paths,
        "reg_patch_stats": stats,

        "faith_value_add_cos_all_mean": float(add_cos_all.mean()),
        "faith_value_add_cos_self_mean": float(add_cos_self.mean()),
        "faith_value_add_cos_regs_mean": float(add_cos_regs.mean()),
        "faith_value_add_cos_noregs_mean": float(add_cos_noregs.mean()),

        "faith_value_add_norm_true_mean": float(add_norm_true.mean()),
        "faith_value_add_norm_self_mean": float(add_norm_self.mean()),
        "faith_value_add_norm_regs_mean": float(add_norm_regs.mean()),
        "faith_value_add_norm_noregs_mean": float(add_norm_noregs.mean()),
    }


@torch.no_grad()
def run_imagenet_one_shot(
    model,
    preprocess,
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    device: str,
    batch_size: int,
    register_threshold: float,
    cls_mask_includes_self: bool,
    fallback_when_no_regs: str
) -> Dict[str, Any]:
    n_classes = len(train_paths)
    assert n_classes == len(val_paths)

    tr = encode_paths_per_image_regs(
        model=model, preprocess=preprocess, paths=train_paths, device=device,
        register_threshold=register_threshold,
        cls_mask_includes_self=cls_mask_includes_self,
        batch_size=batch_size,
        fallback_when_no_regs=fallback_when_no_regs,
        return_extras=False
    )
    va = encode_paths_per_image_regs(
        model=model, preprocess=preprocess, paths=val_paths, device=device,
        register_threshold=register_threshold,
        cls_mask_includes_self=cls_mask_includes_self,
        batch_size=batch_size,
        fallback_when_no_regs=fallback_when_no_regs,
        return_extras=False
    )

    Xtr_full = tr["full"]
    Xva_full = va["full"]
    Xva_patch = va["patch"]
    Xva_reg = va["reg"]

    ytr = torch.tensor(train_labels, dtype=torch.long)
    yva = torch.tensor(val_labels, dtype=torch.long)

    # normalize embeddings
    Xtr_full = Xtr_full / (Xtr_full.norm(dim=-1, keepdim=True) + 1e-8)
    Xva_full = Xva_full / (Xva_full.norm(dim=-1, keepdim=True) + 1e-8)
    Xva_patch = Xva_patch / (Xva_patch.norm(dim=-1, keepdim=True) + 1e-8)
    Xva_reg = Xva_reg / (Xva_reg.norm(dim=-1, keepdim=True) + 1e-8)

    Xtr_full_gpu = Xtr_full.to(device)
    ytr_gpu = ytr.to(device)

    clf = train_one_shot_linear(
        X_train=Xtr_full_gpu,
        y_train=ytr_gpu,
        n_classes=n_classes,
        epochs=200,
        lr=0.2,
        wd=1e-4,
        device=device
    )

    def eval_top5(X: torch.Tensor) -> float:
        logits = clf(X.to(device)).detach().cpu()
        return topk_accuracy(logits, yva, k=5)

    return {
        "imagenet_classes": n_classes,
        "one_shot_top5_full": eval_top5(Xva_full),
        "one_shot_top5_patch_only": eval_top5(Xva_patch),
        "one_shot_top5_reg_only": eval_top5(Xva_reg),
    }


# -----------------------------
# main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(ok(f"Using device: {device}"))

    reg_leak_blocks = parse_int_list(args.reg_leak_blocks)
    print(ok(f"Reg-leak blocks: {reg_leak_blocks}"))

    # Prepare shared datasets ONCE
    flickr8k_paths = get_flickr8k_paths(args.flickr8k_root, args.flickr8k_max_images, args.seed)
    print(ok(f"Prepared flickr8k paths: {len(flickr8k_paths)} images"))

    print(ok("Prepared ImageNet one-shot protocol (splits will be sampled per repeat unless --imagenet_one_shot_fixed_split)."))

    out_dir = OUT_DIR
    _ensure_dir(out_dir)

    out_dir_leak = os.path.join(out_dir, "reg_leak_over_blocks")
    _ensure_dir(out_dir_leak)     

    all_results: List[Dict[str, Any]] = []
    all_reg_leak_curves: Dict[str, Any] = {} 

    # Loop over models
    for alias, model_id in MODELS:
        print("\n" + ok(f"===== MODEL: {alias} ====="))

        model, preprocess, _ = load_openai_clip_anything(clip, model_id, device=device, jit=False, strict=True)
        model = model.eval().float()
        print(ok(f"Loaded model: {model_id}"))

        print("\n" + ok("Running flickr8k experiments (per-image FINAL-LAYER reg indices)..."))
        flickr8k_res = run_flickr8k_eval(
            model=model,
            preprocess=preprocess,
            flickr8k_paths=flickr8k_paths,
            device=device,
            batch_size=args.flickr8k_batch_size,
            register_threshold=args.register_threshold,
            cls_mask_includes_self=args.cls_mask_includes_self,
            fallback_when_no_regs=args.fallback_when_no_regs
        )

        print(ok(f"flickr8k N = {flickr8k_res['flickr8k_n']}"))
        print(ok(f"CKA(full, patch_only) = {flickr8k_res['cka_full_vs_patch_only']:.6f}"))
        print(ok(f"CKA(full, reg_only)   = {flickr8k_res['cka_full_vs_reg_only']:.6f}"))
        print(ok(f"attn/skip ratio mean  = {flickr8k_res['attn_skip_ratio_mean']:.6f} (median {flickr8k_res['attn_skip_ratio_median']:.6f})"))

        s = flickr8k_res["reg_patch_stats"]
        print(ok(f"Reg count (>thr) mean={s['reg_count_mean']:.3f}, min={s['reg_count_min']}, max={s['reg_count_max']}, zero_frac={s['reg_count_zero_frac']:.4f}"))
        print(ok(f"Patch count (<=thr) mean={s['patch_count_mean']:.3f}, min={s['patch_count_min']}, max={s['patch_count_max']}"))
        print(ok(f"Patch norms (<=thr): mean={s['patch_norm_mean']:.3f}, min={s['patch_norm_min']:.3f}, max={s['patch_norm_max']:.3f}"))
        print(ok(f"Reg norms  (>thr):  mean={s['reg_norm_mean']:.3f}, min={s['reg_norm_min']:.3f}, max={s['reg_norm_max']:.3f}"))
        print(ok(f"Per-image patch min/max mean: {s['per_image_patch_min_mean']:.3f} / {s['per_image_patch_max_mean']:.3f}"))
        print(ok(f"Per-image reg   min/max mean: {s['per_image_reg_min_mean']:.3f} / {s['per_image_reg_max_mean']:.3f}"))
        print(ok(f"Reg-count hist (0..10,11+): {s['reg_count_hist']}"))

        print(ok(f"faith TOKEN mean = {flickr8k_res['faith_token_cos_mean']:.6f}, min = {flickr8k_res['faith_token_cos_min']:.6f}"))
        print(ok(f"faith TOKEN NO-REGS mean = {flickr8k_res['faith_token_cos_noregs_mean']:.6f} | REGS-ONLY mean = {flickr8k_res['faith_token_cos_regs_mean']:.6f}"))
        print(ok(f"faith VALUE mean = {flickr8k_res['faith_value_cos_mean']:.6f}, min = {flickr8k_res['faith_value_cos_min']:.6f}"))
        print(ok(f"faith VALUE sanity (cos(TRUE attn_out, reconstructed ALL sources incl CLS self)) mean = {flickr8k_res['faith_value_cos_vs_attnout_mean']:.6f}"))
        print(ok(f"faith VALUE NO-REGS mean = {flickr8k_res['faith_value_cos_noregs_mean']:.6f} | REGS-ONLY mean = {flickr8k_res['faith_value_cos_regs_mean']:.6f}"))

        print(ok(f"cos(FULL,PATCH) mean={flickr8k_res['cos_full_patch_mean']:.6f}, min={flickr8k_res['cos_full_patch_min']:.6f}, max={flickr8k_res['cos_full_patch_max']:.6f}"))
        print(ok(f"cos(FULL,REG)   mean={flickr8k_res['cos_full_reg_mean']:.6f}, min={flickr8k_res['cos_full_reg_min']:.6f}, max={flickr8k_res['cos_full_reg_max']:.6f}"))
        print(ok(f"cos(PATCH,REG)  mean={flickr8k_res['cos_patch_reg_mean']:.6f}, min={flickr8k_res['cos_patch_reg_min']:.6f}, max={flickr8k_res['cos_patch_reg_max']:.6f}"))

        m = flickr8k_res["reg_patch_stats"]["threshold_margins"]
        print(ok(f"Margin stats: below[thr-10,thr] frac_any={m['below_10_image_frac_any']:.4f}, "
                 f"tokens_mean={m['below_10_token_count_mean']:.3f}, max={m['below_10_token_count_max']}"))
        print(ok(f"Margin stats: above(thr,thr+30] frac_any={m['above_30_image_frac_any']:.4f}, "
                 f"tokens_mean={m['above_30_token_count_mean']:.3f}, max={m['above_30_token_count_max']}"))

        print(warn("Worst 5 TOKEN-faithfulness examples (lowest cos(CLS, attn-weighted PATCH-TOKEN mix))"))
        for c, p in flickr8k_res["token_faithfulness_worst_5"]:
            print(f"  cos={c:+.6f}  path={p}")

        # per-block 'REG leakage into FULL' probe
        print("\n" + ok("Running REG leakage into FULL over blocks (patch/reg masking at each block, final embedding readout)..."))
        reg_leak = run_flickr8k_reg_leak_over_blocks(
            model=model,
            preprocess=preprocess,
            flickr8k_paths=flickr8k_paths,
            device=device,
            batch_size=args.flickr8k_batch_size,
            register_threshold=args.register_threshold,
            cls_mask_includes_self=args.cls_mask_includes_self,
            fallback_when_no_regs=args.fallback_when_no_regs,
            block_indices=reg_leak_blocks,
        )

        # print a compact table
        print(ok("REG leakage by block (means + CKA):"))
        for b in reg_leak["block_list"]:
            r = reg_leak["blocks"][str(b)]
            print(
                f"  b{int(b):02d} | "
                f"cos(F,P)={r['cos_full_patch_mean']:+.4f} "
                f"cos(F,R)={r['cos_full_reg_mean']:+.4f} "
                f"cos(P,R)={r['cos_patch_reg_mean']:+.4f} | "
                f"CKA(F,P)={r['cka_full_vs_patch']:.4f} "
                f"CKA(F,R)={r['cka_full_vs_reg']:.4f} | "
                f"reg_count={r['reg_count_mean']:.3f}"
            )

        # save per-model JSON + line plots over blocks
        out_json = os.path.join(out_dir_leak, f"{alias}__reg_leak_over_blocks.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(reg_leak, f, indent=2)
        print(ok(f"Saved reg-leak JSON: {out_json}"))

        # line plots: cos(F,R), cos(P,R), CKA(F,R)
        bs = [int(b) for b in reg_leak["block_list"]]
        cos_fr = [reg_leak["blocks"][str(b)]["cos_full_reg_mean"] for b in bs]
        cos_pr = [reg_leak["blocks"][str(b)]["cos_patch_reg_mean"] for b in bs]
        cka_fr = [reg_leak["blocks"][str(b)]["cka_full_vs_reg"] for b in bs]

        def _save_line(x, y, title, ylabel, path):
            plt.figure(figsize=(8.5, 4.8))
            plt.plot(x, y, marker="o")
            plt.title(title)
            plt.xlabel("visual block index")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(path, dpi=180)
            plt.close()

        safe_alias = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in alias)
        _save_line(bs, cos_fr, f"{alias}: cos(FULL, REG@block) over blocks", "mean cosine", os.path.join(out_dir_leak, f"{safe_alias}__cos_full_reg_over_blocks.png"))
        _save_line(bs, cos_pr, f"{alias}: cos(PATCH@block, REG@block) over blocks", "mean cosine", os.path.join(out_dir_leak, f"{safe_alias}__cos_patch_reg_over_blocks.png"))
        _save_line(bs, cka_fr, f"{alias}: CKA(FULL, REG@block) over blocks", "linear CKA", os.path.join(out_dir_leak, f"{safe_alias}__cka_full_reg_over_blocks.png"))

        all_reg_leak_curves[alias] = reg_leak


        print("\n" + ok("Running ImageNet one-shot top-5 experiment..."))
        inet_res = run_imagenet_one_shot_repeats(
            model=model,
            preprocess=preprocess,
            train_root=args.imagenet_train,
            val_root=args.imagenet_val,
            wnid_to_class_json=args.wnid_to_class_json,
            base_seed=args.seed,
            repeats=args.imagenet_one_shot_repeats,
            seed_stride=args.imagenet_one_shot_seed_stride,
            fixed_split=args.imagenet_one_shot_fixed_split,
            device=device,
            batch_size=args.imagenet_batch_size,
            register_threshold=args.register_threshold,
            cls_mask_includes_self=args.cls_mask_includes_self,
            fallback_when_no_regs=args.fallback_when_no_regs,
        )
        print(ok(f"ImageNet classes = {inet_res['imagenet_classes']}"))
        print(ok(f"One-shot top-5 FULL      = {inet_res['one_shot_top5_full_mean']:.6f} ± {inet_res['one_shot_top5_full_std']:.6f}"))
        print(ok(f"One-shot top-5 PATCH-ONLY= {inet_res['one_shot_top5_patch_only_mean']:.6f} ± {inet_res['one_shot_top5_patch_only_std']:.6f}"))
        print(ok(f"One-shot top-5 REG-ONLY  = {inet_res['one_shot_top5_reg_only_mean']:.6f} ± {inet_res['one_shot_top5_reg_only_std']:.6f}"))


        print(ok("One-shot per-run results:"))
        for rr in inet_res.get("run_rows", []):
            print(
                f"  seed={rr['seed']:>6d} | "
                f"FULL={rr['top5_full']:.6f}  "
                f"PATCH={rr['top5_patch_only']:.6f}  "
                f"REG={rr['top5_reg_only']:.6f}"
            )

        out_json_inet = os.path.join(out_dir, f"{safe_alias}__imagenet_one_shot_repeats.json")
        with open(out_json_inet, "w", encoding="utf-8") as f:
            json.dump(inet_res, f, indent=2)
        print(ok(f"Saved ImageNet one-shot repeats JSON: {out_json_inet}"))

        # Aggregate comparable metrics
        orth_triplet = (
            float(flickr8k_res["cos_full_patch_mean"]),
            float(flickr8k_res["cos_full_reg_mean"]),
            float(flickr8k_res["cos_patch_reg_mean"]),
        )
        orth_score = float(np.mean(orth_triplet))  # lower => more orthogonal / more decoupled (by cosine)

        scam_res = None
        if args.run_scam:
            print("\n" + ok("Running SCAM vs NoSCAM probe..."))
            scam_buckets = load_scam_samples()
            scam_res = run_scam_vs_noscam_probe(
                model=model,
                preprocess=preprocess,
                scam_buckets=scam_buckets,
                device=device,
                batch_size=args.flickr8k_batch_size,
                register_threshold=args.register_threshold,
                cls_mask_includes_self=args.cls_mask_includes_self,
                fallback_when_no_regs=args.fallback_when_no_regs,
                max_per_variant=args.scam_max_per_variant,
                seed=args.seed
            )
            for v in ["NoSCAM", "SCAM", "SynthSCAM"]:
                if scam_res.get(v, {}).get("n", 0) == 0:
                    continue
                print(ok(
                    f"{v:8s} n={scam_res[v]['n']} | "
                    f"cos_full_reg={scam_res[v]['cos_full_reg_mean']:.6f} | "
                    f"add_norm_regs={scam_res[v]['faith_value_add_norm_regs_mean']:.6f} | "
                    f"add_cos_regs={scam_res[v]['faith_value_add_cos_regs_mean']:.6f}"
                ))
            if "SCAM_minus_NoSCAM" in scam_res:
                print(warn(f"SCAM - NoSCAM delta: {scam_res['SCAM_minus_NoSCAM']}"))

        # summarize SCAM into flat, plot-friendly numbers
        scam_summary = {
            "scam_n_noscam": 0,
            "scam_n_scam": 0,
            "scam_n_synthscam": 0,

            # raw means (NoSCAM baseline + SCAM/Synth)
            "scam_cos_full_reg_mean_noscam": float("nan"),
            "scam_cos_full_reg_mean_scam": float("nan"),
            "scam_cos_full_reg_mean_synthscam": float("nan"),

            "scam_add_norm_regs_mean_noscam": float("nan"),
            "scam_add_norm_regs_mean_scam": float("nan"),
            "scam_add_norm_regs_mean_synthscam": float("nan"),

            "scam_add_norm_noregs_mean_noscam": float("nan"),
            "scam_add_norm_noregs_mean_scam": float("nan"),
            "scam_add_norm_noregs_mean_synthscam": float("nan"),

            "scam_add_cos_regs_mean_noscam": float("nan"),
            "scam_add_cos_regs_mean_scam": float("nan"),
            "scam_add_cos_regs_mean_synthscam": float("nan"),

            "scam_add_cos_noregs_mean_noscam": float("nan"),
            "scam_add_cos_noregs_mean_scam": float("nan"),
            "scam_add_cos_noregs_mean_synthscam": float("nan"),

            # deltas (SCAM - NoSCAM, SynthSCAM - NoSCAM)
            "scam_delta_add_norm_regs_mean": float("nan"),
            "synthscam_delta_add_norm_regs_mean": float("nan"),
            "scam_delta_add_norm_noregs_mean": float("nan"),
            "synthscam_delta_add_norm_noregs_mean": float("nan"),

            "scam_delta_add_cos_regs_mean": float("nan"),
            "synthscam_delta_add_cos_regs_mean": float("nan"),
            "scam_delta_add_cos_noregs_mean": float("nan"),
            "synthscam_delta_add_cos_noregs_mean": float("nan"),

            "scam_delta_cos_full_reg_mean": float("nan"),
            "synthscam_delta_cos_full_reg_mean": float("nan"),
            "scam_delta_cos_full_patch_mean": float("nan"),
            "synthscam_delta_cos_full_patch_mean": float("nan"),
        }

        if scam_res is not None:
            # counts
            for v, key in [("NoSCAM", "scam_n_noscam"), ("SCAM", "scam_n_scam"), ("SynthSCAM", "scam_n_synthscam")]:
                scam_summary[key] = int(scam_res.get(v, {}).get("n", 0))

            # raw means
            def _maybe(v: str, k: str) -> float:
                return float(scam_res[v][k]) if (v in scam_res and k in scam_res[v]) else float("nan")

            scam_summary["scam_cos_full_reg_mean_noscam"] = _maybe("NoSCAM", "cos_full_reg_mean")
            scam_summary["scam_cos_full_reg_mean_scam"] = _maybe("SCAM", "cos_full_reg_mean")
            scam_summary["scam_cos_full_reg_mean_synthscam"] = _maybe("SynthSCAM", "cos_full_reg_mean")

            scam_summary["scam_add_norm_regs_mean_noscam"] = _maybe("NoSCAM", "faith_value_add_norm_regs_mean")
            scam_summary["scam_add_norm_regs_mean_scam"] = _maybe("SCAM", "faith_value_add_norm_regs_mean")
            scam_summary["scam_add_norm_regs_mean_synthscam"] = _maybe("SynthSCAM", "faith_value_add_norm_regs_mean")

            scam_summary["scam_add_norm_noregs_mean_noscam"] = _maybe("NoSCAM", "faith_value_add_norm_noregs_mean")
            scam_summary["scam_add_norm_noregs_mean_scam"] = _maybe("SCAM", "faith_value_add_norm_noregs_mean")
            scam_summary["scam_add_norm_noregs_mean_synthscam"] = _maybe("SynthSCAM", "faith_value_add_norm_noregs_mean")

            scam_summary["scam_add_cos_regs_mean_noscam"] = _maybe("NoSCAM", "faith_value_add_cos_regs_mean")
            scam_summary["scam_add_cos_regs_mean_scam"] = _maybe("SCAM", "faith_value_add_cos_regs_mean")
            scam_summary["scam_add_cos_regs_mean_synthscam"] = _maybe("SynthSCAM", "faith_value_add_cos_regs_mean")

            scam_summary["scam_add_cos_noregs_mean_noscam"] = _maybe("NoSCAM", "faith_value_add_cos_noregs_mean")
            scam_summary["scam_add_cos_noregs_mean_scam"] = _maybe("SCAM", "faith_value_add_cos_noregs_mean")
            scam_summary["scam_add_cos_noregs_mean_synthscam"] = _maybe("SynthSCAM", "faith_value_add_cos_noregs_mean")

            # deltas (computed by probe)
            if "SCAM_minus_NoSCAM" in scam_res:
                d = scam_res["SCAM_minus_NoSCAM"]
                scam_summary["scam_delta_add_norm_regs_mean"] = float(d.get("faith_value_add_norm_regs_mean", float("nan")))
                scam_summary["scam_delta_add_norm_noregs_mean"] = float(d.get("faith_value_add_norm_noregs_mean", float("nan")))
                scam_summary["scam_delta_add_cos_regs_mean"] = float(d.get("faith_value_add_cos_regs_mean", float("nan")))
                scam_summary["scam_delta_add_cos_noregs_mean"] = float(d.get("faith_value_add_cos_noregs_mean", float("nan")))
                scam_summary["scam_delta_cos_full_reg_mean"] = float(d.get("cos_full_reg_mean", float("nan")))
                scam_summary["scam_delta_cos_full_patch_mean"] = float(d.get("cos_full_patch_mean", float("nan")))

            if "SynthSCAM_minus_NoSCAM" in scam_res:
                d = scam_res["SynthSCAM_minus_NoSCAM"]
                scam_summary["synthscam_delta_add_norm_regs_mean"] = float(d.get("faith_value_add_norm_regs_mean", float("nan")))
                scam_summary["synthscam_delta_add_norm_noregs_mean"] = float(d.get("faith_value_add_norm_noregs_mean", float("nan")))
                scam_summary["synthscam_delta_add_cos_regs_mean"] = float(d.get("faith_value_add_cos_regs_mean", float("nan")))
                scam_summary["synthscam_delta_add_cos_noregs_mean"] = float(d.get("faith_value_add_cos_noregs_mean", float("nan")))
                scam_summary["synthscam_delta_cos_full_reg_mean"] = float(d.get("cos_full_reg_mean", float("nan")))
                scam_summary["synthscam_delta_cos_full_patch_mean"] = float(d.get("cos_full_patch_mean", float("nan")))

        all_results.append({
            "alias": alias,
            "model_id": model_id,
            "orth_triplet": orth_triplet,
            "orth_score": orth_score,

            "cka_full_vs_patch_only": float(flickr8k_res["cka_full_vs_patch_only"]),
            "cka_full_vs_reg_only": float(flickr8k_res["cka_full_vs_reg_only"]),

            "faith_value_cos_mean": float(flickr8k_res["faith_value_cos_mean"]),
            "faith_value_cos_noregs_mean": float(flickr8k_res["faith_value_cos_noregs_mean"]),
            "faith_value_cos_regs_mean": float(flickr8k_res["faith_value_cos_regs_mean"]),
            "faith_token_cos_mean": float(flickr8k_res["faith_token_cos_mean"]),

            "attn_skip_ratio_mean": float(flickr8k_res["attn_skip_ratio_mean"]),
            "reg_count_mean": float(flickr8k_res["reg_patch_stats"]["reg_count_mean"]),
            "reg_count_zero_frac": float(flickr8k_res["reg_patch_stats"]["reg_count_zero_frac"]),

            "one_shot_top5_full": float(inet_res["one_shot_top5_full_mean"]),
            "one_shot_top5_patch_only": float(inet_res["one_shot_top5_patch_only_mean"]),
            "one_shot_top5_reg_only": float(inet_res["one_shot_top5_reg_only_mean"]),

            "one_shot_top5_full_std": float(inet_res["one_shot_top5_full_std"]),
            "one_shot_top5_patch_only_std": float(inet_res["one_shot_top5_patch_only_std"]),
            "one_shot_top5_reg_only_std": float(inet_res["one_shot_top5_reg_only_std"]),

            "faith_value_add_cos_all_mean": float(flickr8k_res["faith_value_add_cos_all_mean"]),
            "faith_value_add_cos_self_mean": float(flickr8k_res["faith_value_add_cos_self_mean"]),
            "faith_value_add_cos_regs_mean": float(flickr8k_res["faith_value_add_cos_regs_mean"]),
            "faith_value_add_cos_noregs_mean": float(flickr8k_res["faith_value_add_cos_noregs_mean"]),
            "faith_value_add_norm_true_mean": float(flickr8k_res["faith_value_add_norm_true_mean"]),
            "faith_value_add_norm_self_mean": float(flickr8k_res["faith_value_add_norm_self_mean"]),
            "faith_value_add_norm_regs_mean": float(flickr8k_res["faith_value_add_norm_regs_mean"]),
            "faith_value_add_norm_noregs_mean": float(flickr8k_res["faith_value_add_norm_noregs_mean"]),

            # store SCAM probe summary (flat)
            **scam_summary,
        })


        # free VRAM between models
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==============================
    # Combined results printout
    # ==============================
    print("\n" + ok("===== Combined results across models ====="))

    # 1) Orthogonality (lower first)
    print(ok("Orthogonality (mean cosine among FULL/PATCH/REG). Lower => CLS summary differs more from patch-only/reg-only summaries (stronger decoupling by cosine)."))
    for r in sorted(all_results, key=lambda x: x["orth_score"]):
        x, y, z = r["orth_triplet"]
        print(f"  {r['alias']:<16s} orth={r['orth_score']:.6f} | cos(F,P)={x:.6f} cos(F,R)={y:.6f} cos(P,R)={z:.6f}")

    # 2) Attention Faithfulness (VALUE) (higher first)
    print(ok("Attention Faithfulness (VALUE). Higher => CLS token aligns with last-block attention’s V+out_proj output over patches (closer to actual attention math)."))
    for r in sorted(all_results, key=lambda x: x["faith_value_cos_mean"], reverse=True):
        print(f"  {r['alias']:<16s} value_faith={r['faith_value_cos_mean']:.6f} | no-regs={r['faith_value_cos_noregs_mean']:.6f} regs-only={r['faith_value_cos_regs_mean']:.6f}")

    # 3) Attention Faithfulness (TOKEN) (higher first)
    print(ok("Attention Faithfulness (TOKEN). Higher => CLS token aligns with an attention-weighted mix of *patch tokens* (less faithful than VALUE; still a useful diagnostic)."))
    for r in sorted(all_results, key=lambda x: x["faith_token_cos_mean"], reverse=True):
        print(f"  {r['alias']:<16s} token_faith={r['faith_token_cos_mean']:.6f}")

    # 4) One-shot ImageNet (FULL) (higher first)
    print(ok("One-shot ImageNet top-5 (FULL). Higher => more linearly extractable class signal in the final embedding (proxy for representation usefulness under this protocol)."))
    for r in sorted(all_results, key=lambda x: x["one_shot_top5_full"], reverse=True):
        print(f"  {r['alias']:<16s} top5_full={r['one_shot_top5_full']:.6f} | patch={r['one_shot_top5_patch_only']:.6f} reg={r['one_shot_top5_reg_only']:.6f}")

    # 5) CKA changes (lower => more change) – not “quality”, but “difference”
    print(ok("CKA(full vs masked). Lower => masked embedding geometry differs more from full (a decoupling/perturbation readout, not inherently quality)."))
    for r in sorted(all_results, key=lambda x: 0.5 * (x["cka_full_vs_patch_only"] + x["cka_full_vs_reg_only"])):
        print(f"  {r['alias']:<16s} cka(F,P)={r['cka_full_vs_patch_only']:.6f} cka(F,R)={r['cka_full_vs_reg_only']:.6f}")

    # 6) Additive probe numbers (from flickr8k), ranked
    print(ok("VALUE additive probe (flickr8k)."))
    for r in sorted(all_results, key=lambda x: x["faith_value_add_norm_regs_mean"], reverse=True):
        print(
            f"  {r['alias']:<16s} "
            f"add_norm_true={r['faith_value_add_norm_true_mean']:.6f} | "
            f"self={r['faith_value_add_norm_self_mean']:.6f} regs={r['faith_value_add_norm_regs_mean']:.6f} noregs={r['faith_value_add_norm_noregs_mean']:.6f} | "
            f"add_cos_regs={r['faith_value_add_cos_regs_mean']:.6f} add_cos_noregs={r['faith_value_add_cos_noregs_mean']:.6f}"
        )

    # 7) SCAM vs NoSCAM deltas (if run)
    any_scam = any(np.isfinite(r.get("scam_delta_add_norm_regs_mean", float("nan"))) for r in all_results)
    if any_scam:
        print(ok("SCAM probe deltas (SCAM - NoSCAM). Positive add_norm_regs suggests stronger reg-value contribution under SCAM."))
        for r in sorted(all_results, key=lambda x: x.get("scam_delta_add_norm_regs_mean", float("-inf")), reverse=True):
            dv = r.get("scam_delta_add_norm_regs_mean", float("nan"))
            if not np.isfinite(dv):
                continue
            print(
                f"  {r['alias']:<16s} "
                f"Δadd_norm_regs={r['scam_delta_add_norm_regs_mean']:+.6f} "
                f"Δadd_norm_noregs={r['scam_delta_add_norm_noregs_mean']:+.6f} "
                f"Δadd_cos_regs={r['scam_delta_add_cos_regs_mean']:+.6f} "
                f"Δcos_full_reg={r['scam_delta_cos_full_reg_mean']:+.6f} "
                f"(n NoSCAM={r['scam_n_noscam']}, SCAM={r['scam_n_scam']})"
            )

        print(ok("SynthSCAM probe deltas (SynthSCAM - NoSCAM)."))
        for r in sorted(all_results, key=lambda x: x.get("synthscam_delta_add_norm_regs_mean", float("-inf")), reverse=True):
            dv = r.get("synthscam_delta_add_norm_regs_mean", float("nan"))
            if not np.isfinite(dv):
                continue
            print(
                f"  {r['alias']:<16s} "
                f"Δadd_norm_regs={r['synthscam_delta_add_norm_regs_mean']:+.6f} "
                f"Δadd_norm_noregs={r['synthscam_delta_add_norm_noregs_mean']:+.6f} "
                f"Δadd_cos_regs={r['synthscam_delta_add_cos_regs_mean']:+.6f} "
                f"Δcos_full_reg={r['synthscam_delta_cos_full_reg_mean']:+.6f} "
                f"(n NoSCAM={r['scam_n_noscam']}, Synth={r['scam_n_synthscam']})"
            )

    # Save plots
    labels = [r["alias"] for r in all_results]

    # Orthogonality bar (sorted low->high)
    ort_sorted = sorted(all_results, key=lambda x: x["orth_score"])
    _save_bar_plot(
        out_path=os.path.join(out_dir, "orthogonality_mean_cosine_bar.png"),
        labels=[r["alias"] for r in ort_sorted],
        values=[r["orth_score"] for r in ort_sorted],
        title="Orthogonality (mean cosine among FULL/PATCH/REG) — lower is more decoupled",
        ylabel="mean cosine",
    )

    # 3D arrows for the orthogonality triplet
    for r in all_results:
        x, y, z = r["orth_triplet"]
        safe_alias = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in r["alias"])
        _save_orthogonality_3d_arrow_single(
            out_path=os.path.join(out_dir, f"orthogonality_triplet_3d_arrow__{safe_alias}.png"),
            alias=r["alias"],
            x=float(x),
            y=float(y),
            z=float(z),
        )

    # Attention faithfulness (VALUE)
    val_sorted = sorted(all_results, key=lambda x: x["faith_value_cos_mean"], reverse=True)
    _save_bar_plot(
        out_path=os.path.join(out_dir, "attention_faithfulness_value_bar.png"),
        labels=[r["alias"] for r in val_sorted],
        values=[r["faith_value_cos_mean"] for r in val_sorted],
        title="Attention Faithfulness (VALUE) — higher is more faithful",
        ylabel="cosine(CLS, attention(V+out_proj over patches))",
    )

    # One-shot FULL top-5
    top_sorted = sorted(all_results, key=lambda x: x["one_shot_top5_full"], reverse=True)
    _save_bar_plot(
        out_path=os.path.join(out_dir, "imagenet_one_shot_top5_full_bar.png"),
        labels=[r["alias"] for r in top_sorted],
        values=[r["one_shot_top5_full"] for r in top_sorted],
        title="ImageNet One-shot Top-5 (FULL) — higher is better",
        ylabel="top-5 accuracy",
    )

    # reg_count_mean
    _save_bar_plot(
        out_path=os.path.join(out_dir, "reg_count_mean_bar.png"),
        labels=labels,
        values=[r["reg_count_mean"] for r in all_results],
        title="Mean register count per image (final-layer norm > threshold)",
        ylabel="mean reg_count",
    )

    # additive probe plots
    add_regs_sorted = sorted(all_results, key=lambda x: x["faith_value_add_norm_regs_mean"], reverse=True)
    _save_bar_plot(
        out_path=os.path.join(out_dir, "value_additive_norm_regs_bar.png"),
        labels=[r["alias"] for r in add_regs_sorted],
        values=[r["faith_value_add_norm_regs_mean"] for r in add_regs_sorted],
        title="VALUE additive decomposition — ||regs contribution|| (flickr8k) — higher means regs contribute more (no renorm)",
        ylabel="mean ||attn_out_regs_add||",
    )

    add_noregs_sorted = sorted(all_results, key=lambda x: x["faith_value_add_norm_noregs_mean"], reverse=True)
    _save_bar_plot(
        out_path=os.path.join(out_dir, "value_additive_norm_noregs_bar.png"),
        labels=[r["alias"] for r in add_noregs_sorted],
        values=[r["faith_value_add_norm_noregs_mean"] for r in add_noregs_sorted],
        title="VALUE additive decomposition — ||non-regs contribution|| (flickr8k)",
        ylabel="mean ||attn_out_noregs_add||",
    )

    add_cos_regs_sorted = sorted(all_results, key=lambda x: x["faith_value_add_cos_regs_mean"], reverse=True)
    _save_bar_plot(
        out_path=os.path.join(out_dir, "value_additive_cos_regs_bar.png"),
        labels=[r["alias"] for r in add_cos_regs_sorted],
        values=[r["faith_value_add_cos_regs_mean"] for r in add_cos_regs_sorted],
        title="VALUE additive decomposition — cos(TRUE, regs contribution) (flickr8k)",
        ylabel="mean cosine",
    )

    add_cos_noregs_sorted = sorted(all_results, key=lambda x: x["faith_value_add_cos_noregs_mean"], reverse=True)
    _save_bar_plot(
        out_path=os.path.join(out_dir, "value_additive_cos_noregs_bar.png"),
        labels=[r["alias"] for r in add_cos_noregs_sorted],
        values=[r["faith_value_add_cos_noregs_mean"] for r in add_cos_noregs_sorted],
        title="VALUE additive decomposition — cos(TRUE, non-regs contribution) (flickr8k)",
        ylabel="mean cosine",
    )

    # SCAM delta plots (if present)
    any_scam = any(np.isfinite(r.get("scam_delta_add_norm_regs_mean", float("nan"))) for r in all_results)
    if any_scam:
        scam_sorted = sorted(all_results, key=lambda x: x.get("scam_delta_add_norm_regs_mean", float("-inf")), reverse=True)
        _save_bar_plot(
            out_path=os.path.join(out_dir, "scam_delta_add_norm_regs_bar.png"),
            labels=[r["alias"] for r in scam_sorted if np.isfinite(r.get("scam_delta_add_norm_regs_mean", float("nan")))],
            values=[r["scam_delta_add_norm_regs_mean"] for r in scam_sorted if np.isfinite(r.get("scam_delta_add_norm_regs_mean", float("nan")))],
            title="SCAM - NoSCAM: Δ ||regs contribution|| (VALUE additive, no renorm)",
            ylabel="delta mean ||attn_out_regs_add||",
        )

        _save_bar_plot(
            out_path=os.path.join(out_dir, "scam_delta_cos_full_reg_bar.png"),
            labels=[r["alias"] for r in scam_sorted if np.isfinite(r.get("scam_delta_cos_full_reg_mean", float("nan")))],
            values=[r["scam_delta_cos_full_reg_mean"] for r in scam_sorted if np.isfinite(r.get("scam_delta_cos_full_reg_mean", float("nan")))],
            title="SCAM - NoSCAM: Δ cos(FULL, REG-only embedding)",
            ylabel="delta mean cosine",
        )

        synth_sorted = sorted(all_results, key=lambda x: x.get("synthscam_delta_add_norm_regs_mean", float("-inf")), reverse=True)
        _save_bar_plot(
            out_path=os.path.join(out_dir, "synthscam_delta_add_norm_regs_bar.png"),
            labels=[r["alias"] for r in synth_sorted if np.isfinite(r.get("synthscam_delta_add_norm_regs_mean", float("nan")))],
            values=[r["synthscam_delta_add_norm_regs_mean"] for r in synth_sorted if np.isfinite(r.get("synthscam_delta_add_norm_regs_mean", float("nan")))],
            title="SynthSCAM - NoSCAM: Δ ||regs contribution|| (VALUE additive, no renorm)",
            ylabel="delta mean ||attn_out_regs_add||",
        )


    print("\n" + ok("Done."))


if __name__ == "__main__":
    main()