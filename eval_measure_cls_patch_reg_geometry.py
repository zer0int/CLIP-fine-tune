"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

What this code measures (model geometry / token-stream behavior)
===============================================================================

1) Token-stream summaries per block (CLS / patches / “registers”)
---------------------------------------------------------------
What: for each block b, token stream X_b ∈ R^{B×T×D}, T=1+P
  CLS:            c_b = X_b[:,0,:]                         ∈ R^{B×D}
  patches:        x_{b,i} = X_b[:,1+i,:]                   ∈ R^{B×D}, i=1..P
  patch mean:     pmean_b   = (1/P) Σ_i x_{b,i}
  reg mask:       m_{b,i}   = 1[ ||x_{b,i}||_2 > τ ]        (τ=register_norm_thresh)
  regFrac:        regFrac_b = (1/P) Σ_i m_{b,i}
  nonreg mean:    pnonreg_b = Σ_i (1-m_{b,i}) x_{b,i} / max(1, Σ_i (1-m_{b,i}))
  reg mean:       preg_b    = Σ_i m_{b,i} x_{b,i} / max(1, Σ_i m_{b,i})   (only if any reg)
  patch norms:    max_i ||x_{b,i}||_2 ; mean_i ||x_{b,i}||_2
Why: isolates “normal patch content” vs high-norm outliers (“register-ish”) and how CLS relates to each.

2) Directional covariance spectra (effective dimension / anisotropy)
-------------------------------------------------------------------
What: optional per-sample norming: z ← z / ||z|| (for z ∈ {c_b,pmean_b,pnonreg_b,preg_b})
  Σ_z = E[zz^T] − E[z]E[z]^T
  eig(Σ_z): λ_1≥…≥λ_D ; p_i=λ_i/Σ_j λ_j
  topFrac = λ_1 / Σ_j λ_j
  PR      = ( (Σ_j λ_j)^2 ) / ( Σ_j λ_j^2 )
  eRank   = exp( −Σ_i p_i log p_i )
  k90/k95/k99 = min k s.t. (Σ_{i≤k} λ_i)/(Σ_j λ_j) ≥ {0.90,0.95,0.99}
  cond ≈ λ_max / max(λ_min>0,1e−30)
Why: tells how “spread out” the representation is (collapse vs diverse factors) for CLS vs patches, incl. nonreg vs reg.

3) CLS–Patch alignment (mean cosine angles)
------------------------------------------
What: per block b, mean over images of
  ang(CLS~Pmean):    θ_CP_b   = arccos( <c_b,pmean_b> /(||c_b||·||pmean_b||) )   [deg]
  ang(CLS~Pnonreg):  θ_CN_b   = arccos( <c_b,pnonreg_b>/… )
  ang(CLS~Preg):     θ_CR_b   = arccos( <c_b,preg_b>/… ) (if reg exists)
  ang(CLS~Pfinal):   θ_CPf_b  = arccos( <c_b,pmean_final>/… )
  ang(CLS~CLSfinal): θ_C2F_b  = arccos( <c_b,c_final>/… )
Why: “patch integration” proxy (CLS points toward aggregated patch content); also tests whether CLS aligns with nonreg vs reg outliers.

4) CLS–Patch subspace overlap (shared principal directions)
----------------------------------------------------------
What: from eigvecs U (CLS) and V (Pmean), take top-k (k=min(k90_CLS,k90_P))
  overlap(U,V,k) = ||U_k^T V_k||_F^2 / k    ∈ [0,1]
  rand baseline ≈ k/D
  enrich = overlap / (k/D)
  also overlap(CLS@b, Pfinal) with k=min(k90_CLS, k90_Pfinal)
Why: dataset-level “do CLS and patches vary along the same factors?” (shared subspace vs decoupled streams), normalized by random expectation.

5) Attention vs skip contribution on CLS (norm-share)
-------------------------------------------------------------------
What: per block b
  cls_in   = CLS input to block (pre-hook): c_in_b
  cls_attn = attention output at CLS (attn hook): c_attn_b
  skip_n   = ||c_in_b||_2
  attn_n   = ||c_attn_b||_2
  attnShare_b = attn_n / (attn_n + skip_n + 1e−12)
  label: SKIP_DOMINANT if <0.15 ; ATTN_HEAVY if >0.35 ; else MIXED
Why: proxy for whether CLS updates are dominated by attention mixing vs residual carry-through (block-by-block routing signature).

6) Attention-space Pattn vs uniform Pmean (selection vs averaging)
-----------------------------------------------------------------
What: using attn weights for CLS query over patches (mean over heads), re-normalized:
  w_b ∈ R^{B×P}, Σ_i w_{b,i}=1
  value tokens v_{b,i} from attention VALUE stream
  Pattn_b = Σ_i w_{b,i} v_{b,i}                      ∈ R^{B×D}
  PmeanA_b = (1/P) Σ_i v_{b,i}                       ∈ R^{B×D}
  ang(PmeanA~Pattn) = arccos( <PmeanA,Pattn>/(||·||·||·||) ) [deg]
  corr(PmeanA,Pattn) = Pearson across feature dims, per image; mean over images
Why: detects whether attention behaves like near-uniform pooling (small angle / high corr) or selective aggregation (divergence).

7) Attention mass on “register” patches (CLS→reg focus)
-------------------------------------------------------
What: reg mask in VALUE space: m^v_{b,i}=1[||v_{b,i}||_2>τ]
  regAttnMass_b = Σ_i w_{b,i} m^v_{b,i}          ∈ [0,1]  (mean + max reported)
Why: quantifies whether CLS attention concentrates on high-norm outlier patches (often rises in text-overlay / SynthSCAM cases).

8) Register invariance proxy (shared reg direction)
---------------------------------------------------
What: for images with regs, normalize preg_b:
  r = preg_b / ||preg_b|| ; μ = mean_batch(r) / ||mean_batch(r)||
  regInvCos = <r, μ> ; report mean + n
Why: measures whether “register means” converge to a consistent direction across images (shared attractor) vs being idiosyncratic.

9) Trajectory diagnostics across blocks (smooth vs kink)
-------------------------------------------------------
What: track curves over b: θ_CP_b, overlap_CP_b, θ_C2F_b, regFrac_b
  maxJump = max_b |v_{b+1}−v_b| (angles in deg; overlap/share in abs units)
  monotone fractions: θ down, overlap up
  label: SMOOTH / KINK / LATE_KINK based on jump thresholds and location
Why: flags abrupt late-stage integration/decoupling events (representation “snaps” rather than evolves smoothly).

10) Final-block patch-integration heuristic (opinionated summary)
---------------------------------------------------------------
What (final block only):
  score = 0.55·overlap_CP_final + 0.45·max(0,1−θ_CP_final/60)
  label: FULFILLED if θ≤15° and overlap≥0.60
         PARTIAL   if θ≤30° and overlap≥0.45
         else NOT_FULFILLED
Why: compact proxy for “does final CLS reflect integrated patch content?” (not a theorem; a reporting convenience).
"""

import os
import math
import argparse
from dataclasses import dataclass
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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

REG_NEURONS: Dict[int, List[int]] = {
    11: [9, 987, 1967, 2555, 3661, 3784],
    12: [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868],
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=-1, help="cap per-variant for debugging (-1 = all encountered)")

    ap.add_argument("--out_dir", type=str, default="out_eval_measure/cls_patch_reg_geometry")
    ap.add_argument("--register_norm_thresh", type=float, default=70.0)
    
    ap.add_argument("--nuke_reg", action="store_true", help="Nuke 'register inception neurons' at Block 11 and 12")
    ap.add_argument("--normalize_per_sample", action="store_true", default=True)
    ap.add_argument("--no_normalize_per_sample", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--scam_dataset", type=str, default="BLISS-e-V/SCAM")
    ap.add_argument("--scam_split", type=str, default="train")

    return ap.parse_args()


def attach_reg_neuron_nuke_hooks(visual: torch.nn.Module):
    """
    Attach forward hooks to block.mlp.c_fc for the specified blocks in REG_NEURONS.
    This zeroes specific 4096-d hidden units per block during forward.

    visual: model_clip.visual (VisionTransformer)
    """
    if not REG_NEURONS:
        return

    print("[INFO] Attaching register-neuron nuke hooks...")

    for block_idx, block in enumerate(visual.transformer.resblocks):
        if block_idx not in REG_NEURONS:
            continue

        neuron_idx = torch.tensor(REG_NEURONS[block_idx], dtype=torch.long)

        # c_fc is the first Linear in the MLP (GeLU afterwards)
        c_fc = block.mlp[0]

        def make_hook(idx_tensor: torch.Tensor, blk_idx: int):
            def hook(module, input, output):
                out = output.clone()
                out[..., idx_tensor] = 0.0
                return out
            hook.__name__ = f"reg_nuke_block_{blk_idx}"
            return hook

        c_fc.register_forward_hook(make_hook(neuron_idx, block_idx))
        print(f"[INFO] Hook attached on block {block_idx} c_fc for neurons {REG_NEURONS[block_idx]}")


class ScamVariantDataset(Dataset):
    def __init__(self, hf_ds, indices: List[int], preprocess):
        self.hf_ds = hf_ds
        self.indices = indices
        self.preprocess = preprocess

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        entry = self.hf_ds[self.indices[i]]
        img = entry["image"]  # PIL
        img_t = self.preprocess(img)
        return img_t


def build_scam_indices(hf_ds, variants: List[str], max_images: int = -1) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = {v: [] for v in variants}
    for idx, entry in enumerate(hf_ds):
        sid = str(entry.get("id", ""))
        for v in variants:
            if sid.startswith(v):
                buckets[v].append(idx)
                break

        if max_images is not None and max_images > 0:
            done = all(len(buckets[v]) >= max_images for v in variants)
            if done:
                break

    if max_images is not None and max_images > 0:
        for v in variants:
            buckets[v] = buckets[v][:max_images]

    return buckets

@dataclass
class EigenStats:
    top_eig_frac: float
    pr_effdim: float
    erank: float
    max_eig: float
    min_eig: float
    cond: float
    k50: int
    k90: int
    k95: int
    k99: int

class RunningCov:
    def __init__(self, d: int, normalize_per_sample: bool = True, dtype=torch.float64):
        self.d = d
        self.normalize_per_sample = normalize_per_sample
        self.dtype = dtype
        self.n = 0
        self.sum_x = torch.zeros(d, dtype=dtype)
        self.sum_xxT = torch.zeros(d, d, dtype=dtype)

    def update(self, X: torch.Tensor):
        if X is None or X.numel() == 0:
            return
        X = X.to(dtype=self.dtype)
        if self.normalize_per_sample:
            norms = torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(1e-12)
            X = X / norms
        self.n += X.shape[0]
        self.sum_x += X.sum(dim=0)
        self.sum_xxT += X.T @ X

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n == 0:
            mean = torch.zeros(self.d, dtype=self.dtype)
            cov = torch.zeros(self.d, self.d, dtype=self.dtype)
            return mean, cov
        mean = self.sum_x / float(self.n)
        exx = self.sum_xxT / float(self.n)
        cov = exx - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        return mean, cov


class RunningAngle:
    def __init__(self):
        self.n = 0
        self.sum_deg = 0.0

    def update(self, A: torch.Tensor, B: torch.Tensor):
        if A is None or B is None or A.numel() == 0 or B.numel() == 0:
            return
        A = A / torch.linalg.norm(A, dim=1, keepdim=True).clamp_min(1e-12)
        B = B / torch.linalg.norm(B, dim=1, keepdim=True).clamp_min(1e-12)
        cos = (A * B).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.arccos(cos) * (180.0 / math.pi)
        self.n += ang.numel()
        self.sum_deg += float(ang.sum().item())

    def mean(self) -> float:
        return self.sum_deg / max(1, self.n)


class RunningScalar:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, x: torch.Tensor):
        if x is None or x.numel() == 0:
            return
        x = x.detach().float().cpu().view(-1)
        self.n += x.numel()
        self.sum += float(x.sum().item())
        self.min = min(self.min, float(x.min().item()))
        self.max = max(self.max, float(x.max().item()))

    def mean(self) -> float:
        return self.sum / max(1, self.n)

def eig_summary_from_cov(cov: torch.Tensor) -> Tuple[EigenStats, torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    eigvals = eigvals.clamp_min(0.0)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = float(eigvals.sum().item())
    if total <= 0:
        z = EigenStats(0, 0, 0, 0, 0, float("inf"), 0, 0, 0, 0)
        return z, eigvals, eigvecs

    p = (eigvals / eigvals.sum()).clamp_min(1e-30)
    top_eig_frac = float((eigvals[0] / eigvals.sum()).item())

    pr_effdim = float((eigvals.sum() ** 2 / (eigvals.pow(2).sum().clamp_min(1e-30))).item())
    erank = float(torch.exp(-(p * torch.log(p)).sum()).item())

    max_eig = float(eigvals[0].item())
    positive = eigvals[eigvals > 0]
    min_eig = float(positive[-1].item()) if positive.numel() > 0 else float(eigvals[-1].item())
    cond = float(max_eig / max(min_eig, 1e-30))

    cumev = torch.cumsum(eigvals, dim=0) / eigvals.sum()

    def k_at(thr: float) -> int:
        return int((cumev < thr).sum().item() + 1)

    stats = EigenStats(
        top_eig_frac=top_eig_frac,
        pr_effdim=pr_effdim,
        erank=erank,
        max_eig=max_eig,
        min_eig=min_eig,
        cond=cond,
        k50=k_at(0.50),
        k90=k_at(0.90),
        k95=k_at(0.95),
        k99=k_at(0.99),
    )
    return stats, eigvals, eigvecs


def subspace_overlap(U: torch.Tensor, V: torch.Tensor, k: int) -> float:
    k = int(k)
    if k <= 0:
        return float("nan")
    Uk = U[:, :k]
    Vk = V[:, :k]
    M = Uk.T @ Vk
    return float((M.pow(2).sum() / float(k)).item())


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _standardize_TBD_to_BTD(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Accept [T,B,D] or [B,T,D] and return [B,T,D].
    """
    if x.dim() != 3:
        raise RuntimeError(f"Unexpected tensor shape: {tuple(x.shape)}")
    if x.shape[1] == batch_size:  # [T,B,D]
        return x.permute(1, 0, 2).contiguous()
    if x.shape[0] == batch_size:  # [B,T,D]
        return x.contiguous()
    return x.permute(1, 0, 2).contiguous()


# Explainability heuristic (final block only)
def patch_integration_heuristic(angle_deg: float, overlap: float) -> Tuple[str, float]:
    ang_term = max(0.0, 1.0 - (angle_deg / 60.0))
    ov_term = max(0.0, min(1.0, overlap))
    score = 0.55 * ov_term + 0.45 * ang_term

    if (angle_deg <= 15.0) and (overlap >= 0.60):
        label = "FULFILLED"
    elif (angle_deg <= 30.0) and (overlap >= 0.45):
        label = "PARTIAL"
    else:
        label = "NOT_FULFILLED"

    return label, float(max(0.0, min(1.0, score)))


def get_proxy_label(attn_share: float) -> str:
    if not np.isfinite(attn_share):
        return "NA"
    if attn_share < 0.15:
        return "SKIP_DOMINANT"
    if attn_share > 0.35:
        return "ATTN_HEAVY"
    return "MIXED"

def _max_jump(vals: List[float]) -> Tuple[float, int]:
    if len(vals) < 2:
        return 0.0, -1
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    j = int(np.argmax(diffs))
    return float(diffs[j]), j


def _monotone_fraction(vals: List[float], direction: str) -> float:
    if len(vals) < 2:
        return float("nan")
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if direction == "down":
        good = sum(1 for d in diffs if d <= 0)
    else:
        good = sum(1 for d in diffs if d >= 0)
    return float(good) / float(len(diffs))

def compute_curve_diagnostics(
    ang_cls_p: List[float],
    ov_cls_p: List[float],
    ang_cls_to_final: List[float],
) -> Dict[str, Any]:
    n = len(ang_cls_p)
    if n == 0:
        return {}

    def seg_mean(v: List[float], a: int, b: int) -> float:
        a = max(0, min(a, len(v)))
        b = max(0, min(b, len(v)))
        if b <= a:
            return float("nan")
        return float(np.mean(v[a:b]))

    thirds = max(1, n // 3)
    early = (0, thirds)
    mid = (thirds, 2 * thirds)
    late = (2 * thirds, n)

    max_jump_ang_p, idx_ang_p = _max_jump(ang_cls_p)
    max_jump_ov, idx_ov = _max_jump(ov_cls_p)

    c2f_trunc = ang_cls_to_final[:-1] if len(ang_cls_to_final) >= 2 else ang_cls_to_final
    max_jump_c2f, idx_c2f = _max_jump(c2f_trunc)

    diag = {
        "n_blocks": n,

        "angCP_early_mean": seg_mean(ang_cls_p, *early),
        "angCP_mid_mean": seg_mean(ang_cls_p, *mid),
        "angCP_late_mean": seg_mean(ang_cls_p, *late),
        "ovCP_early_mean": seg_mean(ov_cls_p, *early),
        "ovCP_mid_mean": seg_mean(ov_cls_p, *mid),
        "ovCP_late_mean": seg_mean(ov_cls_p, *late),
        "angC2F_early_mean": seg_mean(ang_cls_to_final, *early),
        "angC2F_mid_mean": seg_mean(ang_cls_to_final, *mid),
        "angC2F_late_mean": seg_mean(ang_cls_to_final, *late),

        "angCP_max_jump_deg": max_jump_ang_p,
        "angCP_max_jump_between": idx_ang_p,
        "ovCP_max_jump": max_jump_ov,
        "ovCP_max_jump_between": idx_ov,
        "angC2F_max_jump_deg": max_jump_c2f,
        "angC2F_max_jump_between": idx_c2f,

        "angCP_monotone_down_frac": _monotone_fraction(ang_cls_p, "down"),
        "ovCP_monotone_up_frac": _monotone_fraction(ov_cls_p, "up"),
        "angC2F_monotone_down_frac": _monotone_fraction(ang_cls_to_final, "down"),
    }

    kink = (max_jump_c2f >= 8.0) or (max_jump_ang_p >= 8.0) or (max_jump_ov >= 0.10)
    late_kink = (idx_c2f >= (n // 2)) or (idx_ang_p >= (n // 2)) or (idx_ov >= (n // 2))
    if kink and late_kink:
        diag["trajectory_label"] = "LATE_KINK"
    elif kink:
        diag["trajectory_label"] = "KINK"
    else:
        diag["trajectory_label"] = "SMOOTH"

    return diag

def save_trajectory_plot(
    out_plots: str,
    ang_cls_p: List[float],
    ov_cls_p: List[float],
    ang_cls_to_final: List[float],
    reg_frac: List[float],
    title: str,
):
    ensure_dir(out_plots)
    x = np.arange(len(ang_cls_p))

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    ax1.plot(x, ang_cls_p, label="ang(CLS~P_mean) [deg]")
    ax1.plot(x, ang_cls_to_final, label="ang(CLS_block~CLS_final) [deg]")
    ax1.set_xlabel("block index")
    ax1.set_ylabel("angle (deg)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, ov_cls_p, label="ov(CLS~P_mean)", linestyle="--")
    ax2.plot(x, reg_frac, label="regFracMean", linestyle=":")
    ax2.set_ylabel("overlap / regFrac")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(title)
    fig.tight_layout()

    out_png = os.path.join(out_plots, "block_trajectories.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def write_trajectory_diagnostics(out_plots: str, diag: Dict[str, Any]):
    out_txt = os.path.join(out_plots, "trajectory_diagnostics.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for k in sorted(diag.keys()):
            f.write(f"{k}: {diag[k]}\n")
    return out_txt

def overlap_random_baseline(k: int, d: int) -> float:
    k = max(1, int(k))
    d = max(1, int(d))
    return float(k) / float(d)

def overlap_enrichment(overlap: float, k: int, d: int) -> float:
    ov_rand = overlap_random_baseline(k, d)
    return float(overlap) / max(1e-12, ov_rand)

def batch_pearson_corr(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Per-sample Pearson correlation across feature dims.
    A,B: [B,D] -> corr: [B]
    """
    A = A.float()
    B = B.float()
    A = A - A.mean(dim=1, keepdim=True)
    B = B - B.mean(dim=1, keepdim=True)
    num = (A * B).sum(dim=1)
    den = torch.linalg.norm(A, dim=1) * torch.linalg.norm(B, dim=1)
    return (num / den.clamp_min(1e-12)).clamp(-1.0, 1.0)

def evaluate_one_model_one_variant(
    *,
    model_alias: str,
    model_spec: str,
    variant: str,
    hf_ds: Any,
    indices: List[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    out_dir: str,
    register_norm_thresh: float,
    normalize_per_sample: bool,
    NUKE_REG_NEURONS: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    is_file = os.path.isfile(model_spec)
    print(f"\n[load] model_alias='{model_alias}' model_spec='{model_spec}' (is_file={is_file}) | variant={variant}")
    model, preprocess, _ = load_openai_clip_anything(clip, model_spec, device=device, jit=False, strict=True)

    visual = model.visual

    # Attach reg neuron nuke hooks if requested
    if NUKE_REG_NEURONS:
        attach_reg_neuron_nuke_hooks(visual)
        print("[INFO] Nuking register neurons: ", REG_NEURONS)
    
    model.eval()
    model = model.float()

    ds = ScamVariantDataset(hf_ds, indices, preprocess=preprocess)

    pin_memory = (device.type == "cuda")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    out_plots = os.path.join(out_dir, model_alias, variant, "single_plots")
    ensure_dir(out_plots)

    resblocks = model.visual.transformer.resblocks
    n_blocks = len(resblocks)
    D = model.visual.transformer.width
    print(f"[model] n_blocks={n_blocks} token_width={D} | N_images={len(ds)}")

    cov_cls = [RunningCov(D, normalize_per_sample=normalize_per_sample) for _ in range(n_blocks)]
    cov_pmean = [RunningCov(D, normalize_per_sample=normalize_per_sample) for _ in range(n_blocks)]
    cov_pnonreg = [RunningCov(D, normalize_per_sample=normalize_per_sample) for _ in range(n_blocks)]
    cov_preg = [RunningCov(D, normalize_per_sample=normalize_per_sample) for _ in range(n_blocks)]

    ang_cls_pmean = [RunningAngle() for _ in range(n_blocks)]
    ang_cls_pfinal = [RunningAngle() for _ in range(n_blocks)]
    ang_cls_to_final = [RunningAngle() for _ in range(n_blocks)]

    ang_cls_pnonreg = [RunningAngle() for _ in range(n_blocks)]
    ang_cls_preg = [RunningAngle() for _ in range(n_blocks)]

    reg_frac = [RunningScalar() for _ in range(n_blocks)]
    max_patch_norm = [RunningScalar() for _ in range(n_blocks)]
    mean_patch_norm = [RunningScalar() for _ in range(n_blocks)]
    reg_invar_cos = [RunningScalar() for _ in range(n_blocks)]

    # Proxy: skip vs attn norms (CLS)
    skip_cls_l2 = [RunningScalar() for _ in range(n_blocks)]
    attn_cls_l2 = [RunningScalar() for _ in range(n_blocks)]
    attn_share  = [RunningScalar() for _ in range(n_blocks)]

    # Pmean~Pattn + reg-attn-mass (CLS->patch)
    ang_pmean_pattn = [RunningAngle() for _ in range(n_blocks)]
    corr_pmean_pattn = [RunningScalar() for _ in range(n_blocks)]
    reg_attn_mass = [RunningScalar() for _ in range(n_blocks)]

    # -------------------------------------------------------------------------
    # Instead of storing huge token streams, we store only per-block *summaries*
    # per batch, then update running stats in the post-forward loop.
    # -------------------------------------------------------------------------
    batch_cls_out: Dict[int, torch.Tensor] = {}
    batch_pmean_out: Dict[int, torch.Tensor] = {}
    batch_pnonreg_out: Dict[int, torch.Tensor] = {}
    batch_preg_out: Dict[int, torch.Tensor] = {}
    batch_reg_exists: Dict[int, torch.Tensor] = {}

    batch_cls_in: Dict[int, torch.Tensor] = {}
    batch_cls_attn: Dict[int, torch.Tensor] = {}

    # store attention-derived Pmean/Pattn + reg-attn-mass (small!)
    batch_pmean_attn: Dict[int, torch.Tensor] = {}
    batch_pattn: Dict[int, torch.Tensor] = {}
    batch_reg_attn_mass: Dict[int, torch.Tensor] = {}

    def make_block_hook(bi: int):
        def hook(_module, _inp, out):
            # out: [T,B,D] (typically)
            # We only store small [B,D] summaries.
            # NOTE: B is not known here, infer from out.
            if out.dim() != 3:
                raise RuntimeError(f"Block hook got unexpected out shape: {tuple(out.shape)}")
            # infer batch:
            B_local = out.shape[1] if out.shape[0] != out.shape[1] else out.shape[1]
            xb = out.permute(1, 0, 2).contiguous() if out.shape[1] == B_local else out.contiguous()
            # xb: [B,T,D]
            cls = xb[:, 0, :]
            patches = xb[:, 1:, :]

            # patch norms -> reg mask
            pnorm = torch.linalg.norm(patches.float(), dim=-1)  # [B,P]
            max_patch_norm[bi].update(pnorm.max(dim=1).values.detach().cpu())
            mean_patch_norm[bi].update(pnorm.mean(dim=1).detach().cpu())

            reg_mask = (pnorm > register_norm_thresh)  # [B,P]
            reg_count = reg_mask.sum(dim=1).float()
            reg_frac[bi].update((reg_count / float(patches.shape[1])).detach().cpu())

            pmean = patches.mean(dim=1)

            nonreg_mask = (~reg_mask)
            nonreg_count = nonreg_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pnonreg_sum = (patches * nonreg_mask.unsqueeze(-1)).sum(dim=1)
            pnonreg = pnonreg_sum / nonreg_count
            zero_nonreg = (nonreg_mask.sum(dim=1) == 0)
            if zero_nonreg.any():
                pnonreg[zero_nonreg] = pmean[zero_nonreg]

            # reg-mean only for samples with reg
            reg_exists = (reg_mask.sum(dim=1) > 0)
            preg = None
            if reg_exists.any():
                reg_count_safe = reg_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                preg_sum = (patches * reg_mask.unsqueeze(-1)).sum(dim=1)
                preg = preg_sum / reg_count_safe

                # reg invariance proxy (cos to batch mean in reg space)
                preg_use = preg[reg_exists]
                preg_use_n = preg_use / torch.linalg.norm(preg_use, dim=1, keepdim=True).clamp_min(1e-12)
                batch_mu = preg_use_n.mean(dim=0, keepdim=True)
                batch_mu = batch_mu / torch.linalg.norm(batch_mu, dim=1, keepdim=True).clamp_min(1e-12)
                cos_to_mu = (preg_use_n * batch_mu).sum(dim=1)
                reg_invar_cos[bi].update(cos_to_mu.detach().cpu())

            # store summaries (CPU)
            batch_cls_out[bi] = cls.detach().cpu()
            batch_pmean_out[bi] = pmean.detach().cpu()
            batch_pnonreg_out[bi] = pnonreg.detach().cpu()
            batch_reg_exists[bi] = reg_exists.detach().cpu()
            if preg is not None:
                batch_preg_out[bi] = preg.detach().cpu()
        return hook

    def make_block_prehook(bi: int):
        def prehook(_module, inp):
            x = inp[0]  # [T,B,D]
            if x.dim() != 3:
                raise RuntimeError(f"Prehook got unexpected x shape: {tuple(x.shape)}")
            # standardize to [B,T,D]
            # infer batch from second dim typical for CLIP [T,B,D]
            B_local = x.shape[1]
            xb = x.permute(1, 0, 2).contiguous()
            batch_cls_in[bi] = xb[:, 0, :].detach().cpu()
        return prehook

    def make_attn_hook(bi: int):
        def hook(_module, inp, out):
            """
            inp: (query, key, value, ...)
              query/key/value are [T,B,D]
            out: (attn_output, attn_weights) where:
              attn_output: [T,B,D]
              attn_weights: [B,H,tgt,src]  (per-head, no averaging)
            We store:
              - cls_attn vector (for proxy)
              - Pmean_attn and Pattn computed from VALUE tokens (attention space)
              - reg-attn-mass using reg mask from VALUE patch norms
            """
            if not isinstance(out, (tuple, list)) or len(out) < 2 or out[1] is None:
                raise RuntimeError("attn hook expected (attn_out, attn_weights) but got something else.")
            attn_out = out[0]
            attn_w = out[1]

            # proxy: cls_attn
            B_local = attn_out.shape[1]
            aout_b = attn_out.permute(1, 0, 2).contiguous()  # [B,T,D]
            batch_cls_attn[bi] = aout_b[:, 0, :].detach().cpu()

            # compute Pattn in attention space using VALUE tokens + CLS attention weights
            query = inp[0]
            value = inp[2]
            # value: [T,B,D] -> [B,T,D]
            vb = value.permute(1, 0, 2).contiguous()  # [B,T,D]
            patches_v = vb[:, 1:, :]                  # [B,P,D]

            # CLS-row, patch columns: [B,H,P] -> mean heads -> [B,P]
            w_patch = attn_w[:, :, 0, 1:].mean(dim=1)  # [B,P]
            # renormalize over patches only (exclude CLS self-attn)
            w_patch = w_patch / w_patch.sum(dim=1, keepdim=True).clamp_min(1e-12)

            pattn = (w_patch.unsqueeze(-1) * patches_v).sum(dim=1)  # [B,D]
            pmean_attn = patches_v.mean(dim=1)                      # [B,D]

            # reg mask from VALUE patch norms (attention-space regs)
            pnorm_v = torch.linalg.norm(patches_v.float(), dim=-1)   # [B,P]
            reg_mask_v = (pnorm_v > register_norm_thresh).float()    # [B,P]
            reg_mass = (w_patch * reg_mask_v).sum(dim=1)             # [B]

            batch_pattn[bi] = pattn.detach().cpu()
            batch_pmean_attn[bi] = pmean_attn.detach().cpu()
            batch_reg_attn_mass[bi] = reg_mass.detach().cpu()
        return hook

    hooks = []
    for bi in range(n_blocks):
        hooks.append(resblocks[bi].register_forward_hook(make_block_hook(bi)))
        hooks.append(resblocks[bi].register_forward_pre_hook(make_block_prehook(bi)))
        hooks.append(resblocks[bi].attn.register_forward_hook(make_attn_hook(bi)))

    total_seen = 0
    pbar = tqdm(dl, desc=f"Extracting summaries [{model_alias}|{variant}]", ncols=110)

    with torch.no_grad():
        for images in pbar:
            images = images.to(device, non_blocking=True)
            B = images.shape[0]

            # clear per-batch storage
            batch_cls_out.clear()
            batch_pmean_out.clear()
            batch_pnonreg_out.clear()
            batch_preg_out.clear()
            batch_reg_exists.clear()
            batch_cls_in.clear()
            batch_cls_attn.clear()
            batch_pmean_attn.clear()
            batch_pattn.clear()
            batch_reg_attn_mass.clear()

            _ = model.encode_image(images)

            total_seen += B
            pbar.set_postfix({"seen": total_seen})

            # final vectors for CLS-final / Pfinal (from last block summaries)
            if (n_blocks - 1) not in batch_cls_out or (n_blocks - 1) not in batch_pmean_out:
                raise RuntimeError("Final block summaries missing; hooks failed.")
            cls_final = batch_cls_out[n_blocks - 1]
            pfinal_mean = batch_pmean_out[n_blocks - 1]

            # update per-block running stats
            for bi in range(n_blocks):
                cls = batch_cls_out.get(bi, None)
                pmean = batch_pmean_out.get(bi, None)
                pnonreg = batch_pnonreg_out.get(bi, None)
                reg_exists = batch_reg_exists.get(bi, None)
                if cls is None or pmean is None or pnonreg is None or reg_exists is None:
                    raise RuntimeError(f"Missing block summaries for block {bi}.")

                # covs
                cov_cls[bi].update(cls)
                cov_pmean[bi].update(pmean)
                cov_pnonreg[bi].update(pnonreg)

                # angles (mean-space)
                ang_cls_pmean[bi].update(cls, pmean)
                ang_cls_pfinal[bi].update(cls, pfinal_mean)
                ang_cls_to_final[bi].update(cls, cls_final)
                ang_cls_pnonreg[bi].update(cls, pnonreg)

                # preg cov/angle only where reg exists
                preg = batch_preg_out.get(bi, None)
                if preg is not None and reg_exists.any():
                    cov_preg[bi].update(preg[reg_exists])
                    ang_cls_preg[bi].update(cls[reg_exists], preg[reg_exists])

                # proxy: skip/attn/share
                cls_in = batch_cls_in.get(bi, None)
                cls_attn = batch_cls_attn.get(bi, None)
                if cls_in is None or cls_attn is None:
                    raise RuntimeError(f"Missing cls_in/cls_attn for block {bi} (proxy).")
                skip_n = torch.linalg.norm(cls_in.float(), dim=1)
                attn_n = torch.linalg.norm(cls_attn.float(), dim=1)
                share = attn_n / (attn_n + skip_n + 1e-12)
                skip_cls_l2[bi].update(skip_n)
                attn_cls_l2[bi].update(attn_n)
                attn_share[bi].update(share)

                # Pmean~Pattn diagnostics (attention-space)
                pmean_a = batch_pmean_attn.get(bi, None)
                pattn = batch_pattn.get(bi, None)
                r_mass = batch_reg_attn_mass.get(bi, None)
                if pmean_a is None or pattn is None or r_mass is None:
                    raise RuntimeError(f"Missing Pmean_attn/Pattn/reg_attn_mass for block {bi}.")
                ang_pmean_pattn[bi].update(pmean_a, pattn)
                corr = batch_pearson_corr(pmean_a, pattn)  # [B]
                corr_pmean_pattn[bi].update(corr)
                reg_attn_mass[bi].update(r_mass)

    for h in hooks:
        h.remove()

    print(f"[done] model_alias='{model_alias}' variant={variant} total_seen={total_seen}")

    # Patch-final basis for CLS~Pfinal overlap comparisons
    _, cov_pf = cov_pmean[n_blocks - 1].finalize()
    pf_stats, pf_eigvals, pf_eigvecs = eig_summary_from_cov(cov_pf)
    k_pf = pf_stats.k90

    rows = []
    ckpt_tag = model_alias

    curve_ang_cp: List[float] = []
    curve_ov_cp: List[float] = []
    curve_ang_c2f: List[float] = []
    curve_regfrac: List[float] = []

    for bi in range(n_blocks):
        _, covC = cov_cls[bi].finalize()
        C_stats, C_eigvals, C_eigvecs = eig_summary_from_cov(covC)

        _, covP = cov_pmean[bi].finalize()
        P_stats, P_eigvals, P_eigvecs = eig_summary_from_cov(covP)

        _, covN = cov_pnonreg[bi].finalize()
        N_stats, N_eigvals, N_eigvecs = eig_summary_from_cov(covN)

        preg_n = cov_preg[bi].n
        if preg_n > 0:
            _, covR = cov_preg[bi].finalize()
            R_stats, R_eigvals, R_eigvecs = eig_summary_from_cov(covR)
        else:
            R_stats, R_eigvals, R_eigvecs = None, None, None

        k_overlap = max(1, min(C_stats.k90, P_stats.k90))
        overlap_CP = subspace_overlap(C_eigvecs, P_eigvecs, k_overlap)

        k_overlap_pf = max(1, min(C_stats.k90, k_pf))
        overlap_CPf = subspace_overlap(C_eigvecs, pf_eigvecs, k_overlap_pf)

        ov_rand = overlap_random_baseline(k_overlap, D)
        ov_enrich = overlap_enrichment(overlap_CP, k_overlap, D)

        # Per-block plots
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)

        def plot_eigs(ax, eigvals, label):
            s = eigvals / eigvals.sum().clamp_min(1e-30)
            ax.plot(np.arange(1, s.numel() + 1), s.cpu().numpy(), label=label)

        plot_eigs(ax1, C_eigvals, "CLS")
        plot_eigs(ax1, P_eigvals, "P_mean")
        plot_eigs(ax1, N_eigvals, "P_nonreg")
        if R_stats is not None and R_eigvals is not None:
            plot_eigs(ax1, R_eigvals, "P_reg")

        ax1.set_title(f"Block {bi:02d} eigen spectrum (fraction)")
        ax1.set_xlabel("eigen index")
        ax1.set_ylabel("eig / sum(eig)")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)

        def plot_cumev(ax, eigvals, label):
            c = torch.cumsum(eigvals, dim=0) / eigvals.sum().clamp_min(1e-30)
            ax.plot(np.arange(1, c.numel() + 1), c.cpu().numpy(), label=label)

        plot_cumev(ax2, C_eigvals, "CLS")
        plot_cumev(ax2, P_eigvals, "P_mean")
        plot_cumev(ax2, N_eigvals, "P_nonreg")
        if R_stats is not None and R_eigvals is not None:
            plot_cumev(ax2, R_eigvals, "P_reg")

        ax2.axhline(0.90, linestyle="--", linewidth=1)
        ax2.axhline(0.95, linestyle="--", linewidth=1)
        ax2.axhline(0.99, linestyle="--", linewidth=1)
        ax2.set_ylim(0.0, 1.01)
        ax2.set_title("Cumulative explained variance")
        ax2.set_xlabel("k")
        ax2.set_ylabel("cumEV")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        reg_mean = reg_frac[bi].mean()
        ang_mean = ang_cls_pmean[bi].mean()
        ang_final = ang_cls_pfinal[bi].mean()
        ang_c2f = ang_cls_to_final[bi].mean()

        ang_nonreg = ang_cls_pnonreg[bi].mean()
        ang_reg = ang_cls_preg[bi].mean() if ang_cls_preg[bi].n > 0 else float("nan")

        ang_pm_pa = ang_pmean_pattn[bi].mean()
        corr_pm_pa = corr_pmean_pattn[bi].mean()
        reg_mass_mean = reg_attn_mass[bi].mean()

        fig.suptitle(
            f"{ckpt_tag} | {variant} | block {bi:02d} | regFracMean={reg_mean:.4f} | "
            f"ang(CLS~Pmean)={ang_mean:.2f}° | ov(CLS~Pmean)={overlap_CP:.3f} (rand≈{ov_rand:.3f},×{ov_enrich:.2f}) | "
            f"ang(CLS~Pnonreg)={ang_nonreg:.2f}° | ang(CLS~Preg)={ang_reg:.2f}° | "
            f"ang(Pmean~Pattn)={ang_pm_pa:.2f}° corr={corr_pm_pa:.3f} regAttnMass={reg_mass_mean:.3f} | "  # <-- NEW
            f"ang(CLS~Pfinal)={ang_final:.2f}° | ov(CLS~Pfinal)={overlap_CPf:.3f} | "
            f"ang(CLS~CLSfinal)={ang_c2f:.2f}°",
            fontsize=9
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])

        out_png = os.path.join(out_plots, f"block_{bi:02d}_eigs.png")
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

        row = {
            "model_alias": model_alias,
            "model_spec": model_spec,
            "variant": variant,
            "ckpt": ckpt_tag,
            "block": bi,
            "N_images": total_seen,
            "token_width": D,
            "normalize_per_sample": bool(normalize_per_sample),
            "register_norm_thresh": float(register_norm_thresh),

            "regFracMean": reg_mean,
            "maxPatchNorm_min": max_patch_norm[bi].min,
            "maxPatchNorm_mean": max_patch_norm[bi].mean(),
            "maxPatchNorm_max": max_patch_norm[bi].max,
            "meanPatchNorm_min": mean_patch_norm[bi].min,
            "meanPatchNorm_mean": mean_patch_norm[bi].mean(),
            "meanPatchNorm_max": mean_patch_norm[bi].max,

            "skipCLS_L2_mean": skip_cls_l2[bi].mean(),
            "attnCLS_L2_mean": attn_cls_l2[bi].mean(),
            "attnShare_mean":  attn_share[bi].mean(),

            "reg_invar_cos_mean": reg_invar_cos[bi].mean() if reg_invar_cos[bi].n > 0 else float("nan"),
            "reg_invar_cos_n": reg_invar_cos[bi].n,

            "CLS_topFrac": C_stats.top_eig_frac,
            "CLS_prEff": C_stats.pr_effdim,
            "CLS_erank": C_stats.erank,
            "CLS_k90": C_stats.k90,

            "P_topFrac": P_stats.top_eig_frac,
            "P_prEff": P_stats.pr_effdim,
            "P_erank": P_stats.erank,
            "P_k90": P_stats.k90,

            "N_topFrac": N_stats.top_eig_frac,
            "N_prEff": N_stats.pr_effdim,
            "N_erank": N_stats.erank,
            "N_k90": N_stats.k90,

            "R_topFrac": R_stats.top_eig_frac if R_stats is not None else float("nan"),
            "R_prEff": R_stats.pr_effdim if R_stats is not None else float("nan"),
            "R_erank": R_stats.erank if R_stats is not None else float("nan"),
            "R_k90": R_stats.k90 if R_stats is not None else float("nan"),
            "R_n": preg_n,

            "angMean_CLS_P_deg": ang_mean,
            "overlap_CLS_P": overlap_CP,
            "overlap_CLS_P_rand": ov_rand,
            "overlap_CLS_P_enrich": ov_enrich,

            "angMean_CLS_Pnonreg_deg": ang_nonreg,
            "angMean_CLS_Preg_deg": ang_reg,

            "angMean_CLS_Pfinal_deg": ang_final,
            "overlap_CLS_Pfinal": overlap_CPf,

            "angMean_CLS_CLSfinal_deg": ang_c2f,

            "angMean_Pmean_Pattn_deg": ang_pm_pa,
            "corrMean_Pmean_Pattn": corr_pm_pa,
            "regAttnMass_mean": reg_mass_mean,
            "regAttnMass_max": reg_attn_mass[bi].max,
        }
        rows.append(row)

        curve_ang_cp.append(float(ang_mean))
        curve_ov_cp.append(float(overlap_CP))
        curve_ang_c2f.append(float(ang_c2f))
        curve_regfrac.append(float(reg_mean))

        print(
            f"block {bi:02d} regFracMean={row['regFracMean']:.4f} "
            f"maxPatchNorm(mean)={row['maxPatchNorm_mean']:.2f} "
            f"ang(CLS~P)={row['angMean_CLS_P_deg']:.1f}° ov={row['overlap_CLS_P']:.3f} (rand≈{row['overlap_CLS_P_rand']:.3f},×{row['overlap_CLS_P_enrich']:.2f}) "
            f"ang(CLS~Pnonreg)={row['angMean_CLS_Pnonreg_deg']:.1f}° "
            f"ang(CLS~Preg)={row['angMean_CLS_Preg_deg']:.1f}° "
            f"ang(Pmean~Pattn)={row['angMean_Pmean_Pattn_deg']:.1f}° corr={row['corrMean_Pmean_Pattn']:.3f} "
            f"regAttnMass={row['regAttnMass_mean']:.3f} "
            f"ang(CLS~CLSfinal)={row['angMean_CLS_CLSfinal_deg']:.1f}° | "
            f"attnShare={row['attnShare_mean']:.3f} "
            f"(skip||CLS||={row['skipCLS_L2_mean']:.2f}, attn||CLS||={row['attnCLS_L2_mean']:.2f}) | "
            f"regInvCos={row['reg_invar_cos_mean']:.4f} (n={row['reg_invar_cos_n']})"
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_plots, "summary_table.csv")
    df.to_csv(out_csv, index=False)

    out_txt = os.path.join(out_plots, "summary_table.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"model_alias: {model_alias}\n")
        f.write(f"model_spec: {model_spec}\n")
        f.write(f"variant: {variant}\n")
        f.write(f"N_images={total_seen} D={D} normalize_per_sample={normalize_per_sample}\n")
        f.write(f"register_norm_thresh={register_norm_thresh}\n\n")
        f.write(
            "block\tregFracMean\tmaxPatchNorm(mean)\t"
            "ang(CLS~P)\toverlap(CLS~P)\tov_rand\tov_enrich\t"
            "ang(Pmean~Pattn)\tcorr(Pmean,Pattn)\tregAttnMass\t"
            "ang(CLS~Pnonreg)\tang(CLS~Preg)\tang(CLS~CLSfinal)\t"
            "ang(CLS~Pfinal)\toverlap(CLS~Pfinal)\tregInvCosMean\n"
        )
        for r in rows:
            f.write(
                f"{r['block']}\t{r['regFracMean']:.4f}\t{r['maxPatchNorm_mean']:.2f}\t"
                f"{r['angMean_CLS_P_deg']:.1f}\t{r['overlap_CLS_P']:.3f}\t{r['overlap_CLS_P_rand']:.3f}\t{r['overlap_CLS_P_enrich']:.2f}\t"
                f"{r['angMean_Pmean_Pattn_deg']:.1f}\t{r['corrMean_Pmean_Pattn']:.3f}\t{r['regAttnMass_mean']:.3f}\t"
                f"{r['angMean_CLS_Pnonreg_deg']:.1f}\t{r['angMean_CLS_Preg_deg']:.1f}\t{r['angMean_CLS_CLSfinal_deg']:.1f}\t"
                f"{r['angMean_CLS_Pfinal_deg']:.1f}\t{r['overlap_CLS_Pfinal']:.3f}\t"
                f"{(r['reg_invar_cos_mean'] if np.isfinite(r['reg_invar_cos_mean']) else float('nan')):.4f}\n"
            )

    diag = compute_curve_diagnostics(curve_ang_cp, curve_ov_cp, curve_ang_c2f)
    traj_png_title = f"{model_alias} | {variant} | block-wise trajectories"
    save_trajectory_plot(
        out_plots=out_plots,
        ang_cls_p=curve_ang_cp,
        ov_cls_p=curve_ov_cp,
        ang_cls_to_final=curve_ang_c2f,
        reg_frac=curve_regfrac,
        title=traj_png_title,
    )
    traj_diag_txt = write_trajectory_diagnostics(out_plots, diag)

    final_row = df[df["block"] == (n_blocks - 1)].iloc[0].to_dict()

    final_bi = n_blocks - 1
    final_skip = skip_cls_l2[final_bi].mean()
    final_attn = attn_cls_l2[final_bi].mean()
    final_share = attn_share[final_bi].mean()
    infer_lbl = get_proxy_label(final_share)

    lbl, score = patch_integration_heuristic(
        angle_deg=float(final_row["angMean_CLS_P_deg"]),
        overlap=float(final_row["overlap_CLS_P"]),
    )

    diag_dir = os.path.join(out_plots, "diagnostics")
    ensure_dir(diag_dir)

    xs = np.arange(n_blocks)
    skip_means = np.array([skip_cls_l2[i].mean() for i in range(n_blocks)], dtype=np.float64)
    attn_means = np.array([attn_cls_l2[i].mean() for i in range(n_blocks)], dtype=np.float64)
    share_means = np.array([attn_share[i].mean() for i in range(n_blocks)], dtype=np.float64)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(xs, skip_means, label="||skip(CLS)||")
    ax1.plot(xs, attn_means, label="||attn(CLS)||")
    ax1.set_title("Proxy: skip vs attn (L2 norms)")
    ax1.set_xlabel("block")
    ax1.set_ylabel("L2 norm (mean over images)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(xs, share_means, label="attnShare = attn/(attn+skip)")
    ax2.set_title("attnShare over blocks")
    ax2.set_xlabel("block")
    ax2.set_ylabel("share")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "proxy_skip_vs_attn.png"), dpi=160)
    plt.close(fig)

    final_summary = {
        "model_alias": model_alias,
        "model_spec": model_spec,
        "variant": variant,
        "N_images": int(final_row["N_images"]),
        "final_block": int(final_row["block"]),

        "ang_CLS_P_deg": float(final_row["angMean_CLS_P_deg"]),
        "ov_CLS_P": float(final_row["overlap_CLS_P"]),
        "ov_CLS_P_rand": float(final_row["overlap_CLS_P_rand"]),
        "ov_CLS_P_enrich": float(final_row["overlap_CLS_P_enrich"]),

        "ang_CLS_Pnonreg_deg": float(final_row["angMean_CLS_Pnonreg_deg"]),
        "ang_CLS_Preg_deg": float(final_row["angMean_CLS_Preg_deg"]),
        "ang_CLS_CLSfinal_deg": float(final_row["angMean_CLS_CLSfinal_deg"]),
        "regFracMean": float(final_row["regFracMean"]),

        "ang_Pmean_Pattn_deg": float(final_row["angMean_Pmean_Pattn_deg"]),
        "corr_Pmean_Pattn": float(final_row["corrMean_Pmean_Pattn"]),
        "regAttnMass_mean": float(final_row["regAttnMass_mean"]),
        "regAttnMass_max": float(final_row["regAttnMass_max"]),

        "heuristic_label": lbl,
        "heuristic_score": score,
        "trajectory_label": diag.get("trajectory_label", "NA"),
        "trajectory_max_jump_angC2F_deg": diag.get("angC2F_max_jump_deg", float("nan")),
        "trajectory_max_jump_between_angC2F": diag.get("angC2F_max_jump_between", -1),
        "trajectory_max_jump_angCP_deg": diag.get("angCP_max_jump_deg", float("nan")),
        "trajectory_max_jump_between_angCP": diag.get("angCP_max_jump_between", -1),
        "trajectory_max_jump_ovCP": diag.get("ovCP_max_jump", float("nan")),
        "trajectory_max_jump_between_ovCP": diag.get("ovCP_max_jump_between", -1),

        "out_plots_dir": out_plots,
        "summary_csv": out_csv,
        "summary_txt": out_txt,
        "trajectory_plot_png": os.path.join(out_plots, "block_trajectories.png"),
        "trajectory_diag_txt": traj_diag_txt,

        "skipCLS_L2_mean": float(final_skip),
        "attnCLS_L2_mean": float(final_attn),
        "attnShare_mean": float(final_share),
        "label": infer_lbl,
    }

    print("\n[final-block explainability heuristic]")
    print(
        f"  {model_alias} | {variant} | "
        f"ang(CLS~Pmean)={final_summary['ang_CLS_P_deg']:.2f}° "
        f"ov={final_summary['ov_CLS_P']:.3f} (rand≈{final_summary['ov_CLS_P_rand']:.3f},×{final_summary['ov_CLS_P_enrich']:.2f}) | "
        f"ang(Pmean~Pattn)={final_summary['ang_Pmean_Pattn_deg']:.2f}° corr={final_summary['corr_Pmean_Pattn']:.3f} "
        f"regAttnMass={final_summary['regAttnMass_mean']:.3f} (max={final_summary['regAttnMass_max']:.3f}) | "
        f"ang(CLS~CLSfinal)={final_summary['ang_CLS_CLSfinal_deg']:.2f}° "
        f"regFracMean={final_summary['regFracMean']:.4f} "
        f"=> patch integration: {final_summary['heuristic_label']} (score={final_summary['heuristic_score']:.3f})\n"
        f"  [proxy final block] label={final_summary['label']} | "
        f"skip||CLS||={final_summary['skipCLS_L2_mean']:.3f} "
        f"attn||CLS||={final_summary['attnCLS_L2_mean']:.3f} "
        f"attnShare={final_summary['attnShare_mean']:.3f}"
    )

    print("[trajectory diagnostics] (kink ignores final forced step for CLS~CLSfinal)")
    print(
        f"  label={final_summary['trajectory_label']} | "
        f"maxJump ang(CLS~CLSfinal)={final_summary['trajectory_max_jump_angC2F_deg']:.2f}° "
        f"@({final_summary['trajectory_max_jump_between_angC2F']}→{final_summary['trajectory_max_jump_between_angC2F']+1}) | "
        f"maxJump ang(CLS~Pmean)={final_summary['trajectory_max_jump_angCP_deg']:.2f}° "
        f"@({final_summary['trajectory_max_jump_between_angCP']}→{final_summary['trajectory_max_jump_between_angCP']+1}) | "
        f"maxJump ov(CLS~Pmean)={final_summary['trajectory_max_jump_ovCP']:.3f} "
        f"@({final_summary['trajectory_max_jump_between_ovCP']}→{final_summary['trajectory_max_jump_between_ovCP']+1})"
    )

    print("[files]")
    print("  CSV:", out_csv)
    print("  TXT:", out_txt)
    print("  TRAJ PNG:", final_summary["trajectory_plot_png"])
    print("  TRAJ TXT:", final_summary["trajectory_diag_txt"])
    print("  PLOTS DIR:", out_plots)

    return df, final_summary


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    normalize_per_sample = args.normalize_per_sample and (not args.no_normalize_per_sample)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} normalize_per_sample={normalize_per_sample} reg_thr={args.register_norm_thresh}")

    print(f"[data] loading HF dataset: {args.scam_dataset} split={args.scam_split}")
    hf_ds = load_dataset(args.scam_dataset, split=args.scam_split)

    variants = ["NoSCAM", "SynthSCAM"]
    indices_map = build_scam_indices(hf_ds, variants=variants, max_images=args.max_images)
    for v in variants:
        print(f"[data] {v}: n={len(indices_map[v])}")

    all_summaries: List[Dict[str, Any]] = []
    all_rows: List[pd.DataFrame] = []
    
    
    NUKE_REG_NEURONS = False
    if args.nuke_reg:    
        NUKE_REG_NEURONS = True 

    for (model_alias, model_spec) in MODELS:
        for variant in variants:
            if len(indices_map[variant]) == 0:
                print(f"[skip] {model_alias} | {variant}: no samples found.")
                continue

            df, final_summary = evaluate_one_model_one_variant(
                model_alias=model_alias,
                model_spec=model_spec,
                variant=variant,
                hf_ds=hf_ds,
                indices=indices_map[variant],
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dir=args.out_dir,
                register_norm_thresh=args.register_norm_thresh,
                normalize_per_sample=normalize_per_sample,
                NUKE_REG_NEURONS=NUKE_REG_NEURONS,
            )
            all_rows.append(df)
            all_summaries.append(final_summary)

    if len(all_summaries) > 0:
        print("\n==================== SUMMARY (final block + Pattn diagnostics) ====================")
        all_summaries_sorted = sorted(all_summaries, key=lambda d: (d["model_alias"], d["variant"]))
        for s in all_summaries_sorted:
            print(
                f"{s['model_alias']:<18} | {s['variant']:<9} | "
                f"ang(CLS~Pmean)={s['ang_CLS_P_deg']:>6.2f}° | ov={s['ov_CLS_P']:.3f} (×{s['ov_CLS_P_enrich']:.2f}) | "
                f"ang(Pmean~Pattn)={s['ang_Pmean_Pattn_deg']:>6.2f}° corr={s['corr_Pmean_Pattn']:+.3f} | "
                f"regAttnMass={s['regAttnMass_mean']:.3f} (max={s['regAttnMass_max']:.3f}) | "
                f"AttnShare={s.get('label','NA'):<12} share={s.get('attnShare_mean',float('nan')):.3f} | "
                f"TRAJ={s['trajectory_label']:<9} "
                f"plots={s['out_plots_dir']}"
            )

        df_all = pd.concat(all_rows, ignore_index=True) if len(all_rows) > 0 else pd.DataFrame()
        out_all = os.path.join(args.out_dir, "ALL_MODELS_ALL_VARIANTS_summary_table.csv")
        ensure_dir(args.out_dir)
        df_all.to_csv(out_all, index=False)
        print("\n[combined]")
        print("  ALL CSV:", out_all)
    else:
        print("\n[summary] no evaluations ran (no data or empty MODELS).")


if __name__ == "__main__":
    main()