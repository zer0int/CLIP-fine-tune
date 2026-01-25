"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

CLIP register subspace analysis
-----------------------------------------------

Hint: Use the following config to obtain a model that will be 'interesting' to compare:

    utils_xconfigs_examples/expert_experiments/registers_explained_by_patches.json


(1) Recoverability test (reg-subspace vs non-reg-subspace)
    ------------------------------------------------------
    Given a register subspace U_reg built from a *reference* model (typically "pretrained")
    at reference blocks (e.g. 11,12), we test whether the *final* CLIP image embedding
    can be preserved when, at an intermediate block b, we replace *patch tokens* with:

      - reg_only:    x_patch := Proj_{U_reg}(x_patch)
      - nonreg_only: x_patch := x_patch - Proj_{U_reg}(x_patch)

    We compare the resulting final image embedding to the baseline embedding
    (no modification) via cosine similarity.


(2) Cross-image register token detection + diagnostics
    --------------------------------------------------
    For each model and each block, we detect register-like patch tokens using
    cross-image invariance (cluster patches across images by cosine similarity,
    excluding within-image edges). We compute:
      - registers per image (min/mean/max)
      - cosine baselines (off-diagonal means, local distant baseline)
      - PCA spectrum sanity checks (variance explained, participation ratio)
      - reg/nonreg subspace overlap probes (principal angles, max singular value)
      - component diagnostics (why clustering fails: coverage / max-per-image constraints)


(3) Projection-energy probe (reg-subspace “energy spread”) + paired A/B comparisons
    -------------------------------------------------------------------------------
    Using a *reference register basis* (prefer "pretrained" if present), for each model
    and each block we project centered patch tokens onto that register basis and measure:

      - proj_frac_mean: mean fraction ||Proj_U(x)||^2 / ||x||^2 over patches
      - proj_gini_mean: how concentrated reg-subspace energy is across patches
                        (0 = spread evenly, 1 = concentrated in few patches)
      - proj_topk_share_mean: share of reg-subspace energy in top-k patches

    Also measures overlap between each model’s patch PCA subspace (per block) and
    the reference register subspace (principal angles / max singular value).

    Paired comparisons:
      Provide an optional probe tag per model in MODELS: e.g. "A1", "B1", "A2", "B2".
      The script will compare A1 vs B1, A2 vs B2, etc, and print warnings on invalid tags.


- Heuristic auto-summary (printed at end)
- Compares each model to a "pretrained" baseline if that alias is present.
"""

from __future__ import annotations

import os
import csv
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
import matplotlib.pyplot as plt
from PIL import Image

from colorama import Fore, Style, init as colorama_init

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything


# ============================================================
# Models: OpenAI / local path .pt .safetensors / HuggingFace Hub
# ============================================================

MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
    ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
    ("regr-norm", "zer0int/CLIP-Regression-ViT-L-14", "A1"),
    ("regr-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14", "B1"),
]

"""
 MODELS supports 2-tuple or 3-tuple:
   (alias, model_id) or (alias, model_id, probe_tag)
 probe_tag is used for paired A/B comparisons: "A1"/"B1", "A2"/"B2", etc.

--rec_ref_alias to select alias for comparing recoverability against
"""

SMALL_TESTSET_DIR = "image_sets/attn_bench_images"


def parse_args():
    ap = argparse.ArgumentParser("CLIP register subspace analysis suite")
    ap.add_argument("--image_dir", type=str, default=SMALL_TESTSET_DIR)
    ap.add_argument("--out_root", type=str, default="out_eval_measure/comp_reg_subspace")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # toggles
    ap.add_argument("--no_recoverability", action="store_true", help="Skip recoverability experiment.")
    ap.add_argument("--no_crossimage", action="store_true", help="Skip cross-image detection + diagnostics.")
    ap.add_argument("--no_regproj", action="store_true", help="Skip regproj projection-energy probe.")

    # recoverability settings
    ap.add_argument("--rec_blocks", type=str, default="0-23")
    ap.add_argument("--rec_ref_alias", type=str, default="pretrained")
    ap.add_argument("--rec_ref_blocks", type=str, default="12,13,21,22,23")
    ap.add_argument("--rec_proj_k", type=int, default=16)
    ap.add_argument("--rec_center", action="store_true", default=True)
    ap.add_argument("--rec_adaptive_mult", type=float, default=3.0)
    ap.add_argument("--rec_abs_thresh", type=float, default=0.0)
    ap.add_argument("--rec_max_per_img", type=int, default=64)
    ap.add_argument("--rec_project_cls", action="store_true", default=False)

    # crossimage settings
    ap.add_argument("--cx_sim_thr", type=float, default=0.95)
    ap.add_argument("--cx_cover_frac", type=float, default=0.85)
    ap.add_argument("--cx_max_per_image", type=int, default=64)
    ap.add_argument("--cx_refine_topk", type=int, default=32)
    ap.add_argument("--cx_chunk_rows", type=int, default=2048)
    ap.add_argument("--cx_make_plots", action="store_true", default=True)
    ap.add_argument("--cx_no_plots", action="store_true", default=False)

    # regproj settings (uses a reference register basis, prefer pretrained if present)
    ap.add_argument("--rp_ref_alias", type=str, default="pretrained")
    ap.add_argument("--rp_ref_blocks", type=str, default="11,12")
    ap.add_argument("--rp_k_reg_basis", type=int, default=16)
    ap.add_argument("--rp_k_patch_basis", type=int, default=16)
    ap.add_argument("--rp_use_global_centering", action="store_true", default=True)
    ap.add_argument("--rp_use_image_centering", action="store_false", default=False)
    ap.add_argument("--rp_topk_share_k", type=int, default=16)
    
    return ap.parse_args()




def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _savefig(path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _list_images(root: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    out = []
    for fn in sorted(os.listdir(root)):
        if fn.lower().endswith(exts):
            out.append(os.path.join(root, fn))
    return out


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (a * b).sum(dim=-1)


def _proj_onto_subspace(x: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    x: [..., D]
    U: [D, k] orthonormal columns
    """
    orig = x.shape
    D = orig[-1]
    x2 = x.reshape(-1, D)
    z = x2 @ U
    return (z @ U.t()).reshape(orig)


def _parse_blocks(spec: str) -> List[int]:
    # supports "0-23" and "0-11,12-23" and "0,1,2"
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(p))
    return sorted(set(out))


def _as_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_any_clip(model_id: str, device: str) -> Tuple[torch.nn.Module, Any]:
    model, preprocess, _ = load_openai_clip_anything(clip, model_id, device=device, jit=False, strict=True)
    model.eval()
    model.float()
    return model, preprocess


def load_images(image_paths: List[str], preprocess, device: str) -> torch.Tensor:
    ims: List[torch.Tensor] = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        ims.append(preprocess(im))
    return torch.stack(ims, dim=0).to(device)


@torch.no_grad()
def encode_image_baseline(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    emb = model.encode_image(images)
    return emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)


@torch.no_grad()
def capture_visual_block_tokens(model: torch.nn.Module, images: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Returns dict: block_index -> tokens (B, T, D) captured at the output of each resblock.
    Normalizes any (T,B,D) outputs to (B,T,D).
    """
    resblocks = model.visual.transformer.resblocks
    feats: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(bi: int):
        def _hook(_module, _inp, out):
            x = out
            if x is None:
                return
            if x.dim() == 3 and x.shape[1] == images.shape[0]:
                x = x.permute(1, 0, 2).contiguous()
            feats[bi] = x.detach()
        return _hook

    for bi, blk in enumerate(resblocks):
        hooks.append(blk.register_forward_hook(make_hook(bi)))

    _ = model.encode_image(images)

    for h in hooks:
        h.remove()

    return feats


@dataclass
class RegDetectCfgNorm:
    adaptive_mult: float = 3.0
    abs_thresh: Optional[float] = None
    max_per_img: int = 64
    use_patch_only: bool = True


@torch.no_grad()
def collect_register_vectors_norm(
    model: torch.nn.Module,
    images: torch.Tensor,
    ref_blocks: Tuple[int, ...],
    cfg: RegDetectCfgNorm,
) -> torch.Tensor:
    """
    Collects candidate "register" vectors from patch tokens at given ref blocks
    using an adaptive norm threshold: norm >= max(abs_thresh, median*adaptive_mult).
    Returns stacked matrix [N, D].
    """
    v = model.visual
    all_vecs: List[torch.Tensor] = []

    def _forward_until_block(block_idx: int) -> torch.Tensor:
        x = v.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B,P,D)
        cls = v.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)  # (T,B,D)
        for i, blk in enumerate(v.transformer.resblocks):
            x = blk(x)
            if i == block_idx:
                break
        return x.permute(1, 0, 2).contiguous()  # (B,T,D)

    for b in ref_blocks:
        toks = _forward_until_block(b)
        patch = toks[:, 1:, :] if cfg.use_patch_only else toks
        norms = patch.norm(dim=-1)

        for i in range(patch.shape[0]):
            n = norms[i]
            med = n.median()
            thr = med * cfg.adaptive_mult
            if cfg.abs_thresh is not None:
                thr2 = torch.tensor(cfg.abs_thresh, device=thr.device, dtype=thr.dtype)
                thr = torch.maximum(thr, thr2)

            idx = (n >= thr).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue

            if idx.numel() > cfg.max_per_img:
                topk = torch.topk(n[idx], k=cfg.max_per_img, largest=True).indices
                idx = idx[topk]

            all_vecs.append(patch[i, idx, :].detach().clone())

        print(f"[recoverability/Ureg-norm] ref block {b:02d} vecs so far: {sum(vv.shape[0] for vv in all_vecs)}")

    if not all_vecs:
        return torch.empty(0, 0, device=images.device)

    return torch.cat(all_vecs, dim=0)


@torch.no_grad()
def build_U_from_X(X: torch.Tensor, k: int = 16, center: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [N, D]
    Returns:
      U: [D, k] orthonormal basis
      mu: [D] mean used for centering (zeros if center=False)
    """
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"Need X as [N,D] with N>=2; got {tuple(X.shape)}")

    mu = X.mean(dim=0) if center else torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
    Y = X - mu[None, :] if center else X

    _, _, Vh = torch.linalg.svd(Y, full_matrices=False)
    V = Vh.transpose(-2, -1)  # (D, rank)
    kk = min(int(k), V.shape[1])
    U = V[:, :kk].contiguous()
    U, _ = torch.linalg.qr(U, mode="reduced")
    return U, mu


class BlockProjectorHook:
    def __init__(self, block: torch.nn.Module, U: torch.Tensor, mode: str, project_cls: bool = False):
        assert mode in ("reg_only", "nonreg_only")
        self.block = block
        self.U = U
        self.mode = mode
        self.project_cls = project_cls
        self.h = None

    def __enter__(self):
        def _hook(_module, _inp, out):
            if out is None or out.ndim != 3:
                return out
            # out is usually (T,B,D); normalize to (B,T,D)
            x = out
            xb = x.permute(1, 0, 2).contiguous()  # (B,T,D)

            cls = xb[:, :1, :]
            patch = xb[:, 1:, :]

            if self.project_cls:
                cls_proj = _proj_onto_subspace(cls, self.U)
                cls2 = cls_proj if self.mode == "reg_only" else (cls - cls_proj)
            else:
                cls2 = cls

            patch_proj = _proj_onto_subspace(patch, self.U)
            patch2 = patch_proj if self.mode == "reg_only" else (patch - patch_proj)

            xb2 = torch.cat([cls2, patch2], dim=1)
            return xb2.permute(1, 0, 2).contiguous()

        self.h = self.block.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.h is not None:
            self.h.remove()
        self.h = None


@torch.no_grad()
def run_recoverability(
    model: torch.nn.Module,
    images: torch.Tensor,
    U_reg: torch.Tensor,
    blocks: List[int],
    project_cls: bool = False,
) -> Dict[str, List[float]]:
    base = encode_image_baseline(model, images)
    out: Dict[str, List[float]] = {"reg_only": [], "nonreg_only": []}

    for b in blocks:
        blk = model.visual.transformer.resblocks[b]
        for mode in ("reg_only", "nonreg_only"):
            with BlockProjectorHook(blk, U_reg, mode, project_cls=project_cls):
                emb = encode_image_baseline(model, images)
            cs = _cos_sim(emb, base)
            out[mode].append(float(cs.mean().item()))

        print(f"[recoverability] block {b:02d} reg_only={out['reg_only'][-1]:.4f} nonreg_only={out['nonreg_only'][-1]:.4f}")

    return out


def plot_recoverability_curves(model_alias: str, blocks: List[int], series: Dict[str, List[float]], out_dir: str) -> None:
    _ensure_dir(out_dir)
    plt.figure()
    plt.plot(blocks, series["reg_only"], label="reg_only (patch := Proj_U)")
    plt.plot(blocks, series["nonreg_only"], label="nonreg_only (patch := x - Proj_U)")
    plt.title(f"{model_alias} :: recoverability vs block")
    plt.xlabel("block")
    plt.ylabel("cosine(sim(modified_emb, baseline_emb))")
    plt.ylim(0.0, 1.01)
    plt.legend()
    _savefig(os.path.join(out_dir, f"{model_alias}__recoverability.png"))


def plot_recoverability_combined(blocks: List[int], per_model: Dict[str, Dict[str, List[float]]], out_dir: str) -> None:
    _ensure_dir(out_dir)
    for mode in ("reg_only", "nonreg_only"):
        plt.figure()
        for alias, series in per_model.items():
            plt.plot(blocks, series[mode], label=f"{alias}")
        plt.title(f"All models :: {mode}")
        plt.xlabel("block")
        plt.ylabel("cosine(sim(modified_emb, baseline_emb))")
        plt.ylim(0.0, 1.01)
        plt.legend()
        _savefig(os.path.join(out_dir, f"ALL__recoverability__{mode}.png"))


@dataclass
class RegDetectCfgCross:
    sim_thr: float = 0.95
    cover_frac: float = 0.85
    max_per_image: int = 64
    chunk_rows: int = 2048
    device: str = "cuda"

    refine_topk: int = 32
    local_pairs_per_image: int = 64
    seed: int = 0

    use_image_mean_centering: bool = False
    use_global_mean_centering: bool = True

    make_plots: bool = True
    plots_dir: str = "out_registers/unified/crossimage/plots"


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _infer_grid_side(P: int) -> Optional[int]:
    side = int(round(P ** 0.5))
    return side if side * side == P else None


@torch.no_grad()
def _sample_distant_pairs(P: int, side: Optional[int], num_pairs: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    if P < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    if side is None:
        i = torch.randint(0, P, (num_pairs,), generator=gen, device=device)
        j = torch.randint(0, P, (num_pairs,), generator=gen, device=device)
        neq = (i != j)
        if neq.any():
            return torch.stack([i[neq], j[neq]], dim=1)
        return torch.empty((0, 2), dtype=torch.long, device=device)

    coords = torch.stack(torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij"
    ), dim=-1).reshape(P, 2)

    out = []
    tries = 0
    max_tries = max(10_000, num_pairs * 20)
    while len(out) < num_pairs and tries < max_tries:
        tries += 1
        a = int(torch.randint(0, P, (1,), generator=gen, device=device).item())
        b = int(torch.randint(0, P, (1,), generator=gen, device=device).item())
        if a == b:
            continue
        da = coords[a]
        db = coords[b]
        if (torch.max(torch.abs(da - db)) > 1).item():
            out.append((a, b))

    if not out:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    return torch.tensor(out, dtype=torch.long, device=device)


@torch.no_grad()
def _apply_centering(patches_bpd: torch.Tensor, cfg: RegDetectCfgCross) -> Tuple[torch.Tensor, torch.Tensor]:
    B, P, D = patches_bpd.shape
    x = patches_bpd

    if cfg.use_global_mean_centering:
        mu_global = x.reshape(B * P, D).mean(dim=0)
        x = x - mu_global.view(1, 1, D)
    else:
        mu_global = torch.zeros((D,), device=x.device, dtype=x.dtype)

    if cfg.use_image_mean_centering:
        mu_img = x.mean(dim=1, keepdim=True)
        x = x - mu_img

    return x, mu_global


@torch.no_grad()
def _offdiag_mean_cos(v_pd: torch.Tensor) -> Optional[float]:
    P, D = v_pd.shape
    if P < 2:
        return None
    G = v_pd @ v_pd.T
    s = G.sum() - torch.diagonal(G).sum()
    denom = float(P * (P - 1))
    return float((s / denom).item())


@torch.no_grad()
def _offdiag_mean_cos_subset(v_pd: torch.Tensor, idx: torch.Tensor) -> Optional[float]:
    if idx.numel() < 2:
        return None
    return _offdiag_mean_cos(v_pd.index_select(0, idx))


@torch.no_grad()
def _pca_var_explained_and_pr(x_pd: torch.Tensor, topk: int = 5) -> Dict[str, Optional[float]]:
    P, D = x_pd.shape
    if P < 2:
        return {"top1": None, "topk": None, "pr": None}

    if float(x_pd.abs().max().item()) == 0.0:
        return {"top1": 0.0, "topk": 0.0, "pr": 0.0}

    s = torch.linalg.svdvals(x_pd)
    lam = (s ** 2)
    total = lam.sum()
    if float(total.item()) <= 0.0:
        return {"top1": None, "topk": None, "pr": None}

    top1 = float((lam[0] / total).item())
    k = min(int(topk), lam.numel())
    topk_sum = float((lam[:k].sum() / total).item())
    pr = float(((total ** 2) / (lam.pow(2).sum() + 1e-12)).item())
    return {"top1": top1, "topk": topk_sum, "pr": pr}


def _pca_basis(X: torch.Tensor, k: int) -> Optional[torch.Tensor]:
    if X is None or X.ndim != 2:
        return None
    N, D = X.shape
    if N < 2:
        return None
    k_eff = min(int(k), D, N)
    if k_eff < 1:
        return None
    try:
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        X_cpu = X.detach().float().cpu()
        _, _, Vt = torch.linalg.svd(X_cpu, full_matrices=False)
        Vt = Vt.to(X.device)
    return Vt.transpose(0, 1)[:, :k_eff]


def _principal_angles_deg(U: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    M = U.transpose(0, 1) @ V
    sig = torch.linalg.svdvals(M)
    sig = torch.clamp(sig, 0.0, 1.0)
    angles = torch.acos(sig) * (180.0 / math.pi)
    return sig, angles


@torch.no_grad()
def compute_reg_nonreg_subspace_overlap(
    patches_centered: torch.Tensor,  # (P,D), centered (not normalized)
    reg_mask: torch.Tensor,          # (P,) bool
    k_nonreg: int = 5,
    k_reg: int = 5,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if patches_centered is None or patches_centered.numel() == 0:
        return {
            "overlap_regcent_vs_nonreg_top1_cos": None,
            "overlap_regcent_vs_nonreg_topk_max_cos": None,
            "subspace_overlap_maxsv": None,
            "principal_angles_deg_mean": None,
            "principal_angles_deg_min": None,
            "principal_angles_deg_max": None,
        }

    if reg_mask.dtype != torch.bool:
        reg_mask = reg_mask.bool()

    X_reg = patches_centered[reg_mask]
    X_non = patches_centered[~reg_mask]

    if X_reg.shape[0] < 2 or X_non.shape[0] < 2:
        return {
            "overlap_regcent_vs_nonreg_top1_cos": None,
            "overlap_regcent_vs_nonreg_topk_max_cos": None,
            "subspace_overlap_maxsv": None,
            "principal_angles_deg_mean": None,
            "principal_angles_deg_min": None,
            "principal_angles_deg_max": None,
        }

    reg_cent = X_reg.mean(dim=0)
    reg_cent_norm = float(torch.linalg.norm(reg_cent).item())
    reg_cent_dir = None if reg_cent_norm < 1e-12 else (reg_cent / (reg_cent_norm + 1e-12))

    U_non = _pca_basis(X_non, k_nonreg)
    U_reg = _pca_basis(X_reg, k_reg)

    if reg_cent_dir is not None and U_non is not None and U_non.shape[1] >= 1:
        top1 = U_non[:, 0]
        out["overlap_regcent_vs_nonreg_top1_cos"] = float(torch.dot(reg_cent_dir, top1).abs().item())
        sims = torch.matmul(U_non.transpose(0, 1), reg_cent_dir)
        out["overlap_regcent_vs_nonreg_topk_max_cos"] = float(sims.abs().max().item())
    else:
        out["overlap_regcent_vs_nonreg_top1_cos"] = None
        out["overlap_regcent_vs_nonreg_topk_max_cos"] = None

    if U_non is not None and U_reg is not None:
        sig, angles_deg = _principal_angles_deg(U_non, U_reg)
        out["subspace_overlap_maxsv"] = float(sig.max().item()) if sig.numel() > 0 else None
        out["principal_angles_deg_mean"] = float(angles_deg.mean().item()) if angles_deg.numel() > 0 else None
        out["principal_angles_deg_min"] = float(angles_deg.min().item()) if angles_deg.numel() > 0 else None
        out["principal_angles_deg_max"] = float(angles_deg.max().item()) if angles_deg.numel() > 0 else None
    else:
        out["subspace_overlap_maxsv"] = None
        out["principal_angles_deg_mean"] = None
        out["principal_angles_deg_min"] = None
        out["principal_angles_deg_max"] = None

    return out


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0.0:
        return None
    return a / b


@torch.no_grad()
def find_register_tokens_cross_image(tokens_btd: torch.Tensor, cfg: RegDetectCfgCross) -> Dict[str, Any]:
    device = tokens_btd.device
    B, T, D = tokens_btd.shape

    patches_raw = tokens_btd[:, 1:, :]
    B, P, D = patches_raw.shape
    M = B * P

    patches_centered, mu_global = _apply_centering(patches_raw, cfg)

    patches_norm_raw = torch.nn.functional.normalize(patches_raw, dim=-1)
    patches_norm_cent = torch.nn.functional.normalize(patches_centered, dim=-1)

    X = patches_norm_cent.reshape(M, D)
    img_idx = torch.arange(B, device=device).repeat_interleave(P)
    patch_idx = torch.arange(P, device=device).repeat(B) + 1

    uf = UnionFind(M)
    thr = float(cfg.sim_thr)
    chunk = int(cfg.chunk_rows)

    for i0 in range(0, M, chunk):
        i1 = min(M, i0 + chunk)
        Xi = X[i0:i1]
        sims = Xi @ X.T
        same = (img_idx[i0:i1].unsqueeze(1) == img_idx.unsqueeze(0))
        sims = sims.masked_fill(same, -1.0)

        mask = sims >= thr
        if not mask.any():
            continue

        rows, cols = mask.nonzero(as_tuple=True)
        rows = rows + i0
        for a, b in zip(rows.tolist(), cols.tolist()):
            uf.union(a, b)

    comp: Dict[int, List[int]] = {}
    for node in range(M):
        r = uf.find(node)
        comp.setdefault(r, []).append(node)

    kept_roots: List[int] = []
    comp_info: List[Dict[str, Any]] = []

    comp_total = 0
    comp_cov_ok = 0
    comp_mpi_ok = 0
    comp_both_ok = 0

    best_cov = 0.0
    best_min_maxpi: Optional[int] = None
    biggest_size = 0
    biggest_cov = 0.0
    biggest_maxpi = 0

    for r, nodes in comp.items():
        comp_total += 1
        imgs = [int(img_idx[n].item()) for n in nodes]
        uniq_imgs = sorted(set(imgs))
        coverage = len(uniq_imgs) / float(B)

        counts: Dict[int, int] = {}
        for ii in imgs:
            counts[ii] = counts.get(ii, 0) + 1
        max_pi = max(counts.values())

        cov_ok = (coverage >= float(cfg.cover_frac))
        mpi_ok = (max_pi <= int(cfg.max_per_image))

        if cov_ok:
            comp_cov_ok += 1
        if mpi_ok:
            comp_mpi_ok += 1
        if cov_ok and mpi_ok:
            comp_both_ok += 1
            kept_roots.append(r)

        if coverage > best_cov:
            best_cov = float(coverage)
        if best_min_maxpi is None or max_pi < best_min_maxpi:
            best_min_maxpi = int(max_pi)

        if len(nodes) > biggest_size:
            biggest_size = int(len(nodes))
            biggest_cov = float(coverage)
            biggest_maxpi = int(max_pi)

        comp_info.append({
            "root": int(r),
            "size": int(len(nodes)),
            "coverage": float(coverage),
            "max_per_image": int(max_pi),
            "images": uniq_imgs,
        })

    if comp_total > 0:
        print(
            f"    [compDiag] total={comp_total} covOK={comp_cov_ok} mpiOK={comp_mpi_ok} bothOK={comp_both_ok}"
            f" | bestCov={best_cov:.3f} bestMinMaxPerImg={best_min_maxpi}"
            f" | biggest(size={biggest_size}, cov={biggest_cov:.3f}, maxPerImg={biggest_maxpi})"
        )
    else:
        print("    [compDiag] total=0")

    reg_by_image: List[List[int]] = [[] for _ in range(B)]
    for r in kept_roots:
        for n in comp[r]:
            bi = int(img_idx[n].item())
            pj = int(patch_idx[n].item())
            reg_by_image[bi].append(pj)
    for bi in range(B):
        reg_by_image[bi] = sorted(set(reg_by_image[bi]))

    centroids: List[torch.Tensor] = []
    centroid_meta: List[Dict[str, Any]] = []

    if kept_roots:
        for r in kept_roots:
            nodes = comp[r]
            V = X[nodes]
            c = torch.nn.functional.normalize(V.mean(dim=0), dim=-1)
            centroids.append(c)
            centroid_meta.append({"root": int(r), "size": int(len(nodes))})

        C = torch.stack(centroids, dim=0)
        sims = torch.einsum("bpd,kd->bpk", patches_norm_cent, C)
        refine_topk = max(1, int(cfg.refine_topk))

        for bi in range(B):
            for ki in range(C.shape[0]):
                s = sims[bi, :, ki]
                topv, topi = torch.topk(s, k=min(refine_topk, P), largest=True)
                for v, idx0 in zip(topv.tolist(), topi.tolist()):
                    if v >= thr:
                        reg_by_image[bi].append(idx0 + 1)

        for bi in range(B):
            reg_by_image[bi] = sorted(set(reg_by_image[bi]))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed))
    side = _infer_grid_side(P)

    reg_mask = torch.zeros((B, P), dtype=torch.bool, device=device)
    for bi in range(B):
        for pj in reg_by_image[bi]:
            j0 = pj - 1
            if 0 <= j0 < P:
                reg_mask[bi, j0] = True

    reg_counts = reg_mask.sum(dim=1).detach().cpu().tolist()
    nonreg_counts = (P - reg_mask.sum(dim=1)).detach().cpu().tolist()

    # distant local baseline (non-reg only)
    local_cos_vals_raw: List[float] = []
    local_cos_vals_cent: List[float] = []
    num_local_pairs_used = 0

    for bi in range(B):
        nonreg = (~reg_mask[bi]).nonzero(as_tuple=True)[0]
        if nonreg.numel() < 2:
            continue
        pairs = _sample_distant_pairs(P, side, int(cfg.local_pairs_per_image), device, gen)
        if pairs.numel() == 0:
            continue
        a = pairs[:, 0]
        b = pairs[:, 1]
        keep = (~reg_mask[bi, a]) & (~reg_mask[bi, b])
        a = a[keep]
        b = b[keep]
        if a.numel() == 0:
            continue

        va0 = patches_norm_raw[bi, a, :]
        vb0 = patches_norm_raw[bi, b, :]
        local_cos_vals_raw.extend(((va0 * vb0).sum(dim=-1)).detach().float().tolist())

        va1 = patches_norm_cent[bi, a, :]
        vb1 = patches_norm_cent[bi, b, :]
        local_cos_vals_cent.extend(((va1 * vb1).sum(dim=-1)).detach().float().tolist())

        num_local_pairs_used += int(a.numel())

    def _summ(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"mean": None, "std": None, "p95": None, "max": None, "min": None, "n": 0}
        t = torch.tensor(vals)
        return {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "p95": float(torch.quantile(t, 0.95).item()),
            "max": float(t.max().item()),
            "min": float(t.min().item()),
            "n": int(t.numel()),
        }

    local_raw = _summ(local_cos_vals_raw)
    local_cent = _summ(local_cos_vals_cent)

    mu_img_raw = patches_raw.mean(dim=1)
    mu_img_raw_norm = mu_img_raw.norm(dim=-1)
    mu_img_raw_stats = {
        "patch_mean_norm_raw_mean": float(mu_img_raw_norm.mean().item()),
        "patch_mean_norm_raw_std": float(mu_img_raw_norm.std(unbiased=False).item()),
        "patch_mean_norm_raw_min": float(mu_img_raw_norm.min().item()),
        "patch_mean_norm_raw_max": float(mu_img_raw_norm.max().item()),
    }

    if cfg.use_global_mean_centering:
        patches_g = patches_raw - mu_global.view(1, 1, D)
        mu_img_g = patches_g.mean(dim=1)
        mu_img_g_norm = mu_img_g.norm(dim=-1)
        mu_img_g_stats = {
            "patch_mean_norm_after_global_mean": float(mu_img_g_norm.mean().item()),
            "patch_mean_norm_after_global_std": float(mu_img_g_norm.std(unbiased=False).item()),
        }
    else:
        mu_img_g_stats = {"patch_mean_norm_after_global_mean": None, "patch_mean_norm_after_global_std": None}

    offdiag_raw_vals: List[float] = []
    offdiag_cent_vals: List[float] = []
    offdiag_nonreg_raw_vals: List[float] = []
    offdiag_nonreg_cent_vals: List[float] = []

    for bi in range(B):
        v0 = patches_norm_raw[bi]
        v1 = patches_norm_cent[bi]
        m0 = _offdiag_mean_cos(v0)
        m1 = _offdiag_mean_cos(v1)
        if m0 is not None:
            offdiag_raw_vals.append(m0)
        if m1 is not None:
            offdiag_cent_vals.append(m1)

        nonreg_idx = (~reg_mask[bi]).nonzero(as_tuple=True)[0]
        mn0 = _offdiag_mean_cos_subset(v0, nonreg_idx)
        mn1 = _offdiag_mean_cos_subset(v1, nonreg_idx)
        if mn0 is not None:
            offdiag_nonreg_raw_vals.append(mn0)
        if mn1 is not None:
            offdiag_nonreg_cent_vals.append(mn1)

    offdiag_raw = _summ(offdiag_raw_vals)
    offdiag_cent = _summ(offdiag_cent_vals)
    offdiag_nonreg_raw = _summ(offdiag_nonreg_raw_vals)
    offdiag_nonreg_cent = _summ(offdiag_nonreg_cent_vals)

    comp_pair_cos: List[float] = []
    for r in kept_roots:
        nodes = comp[r]
        if len(nodes) < 2:
            continue
        V = X[nodes]
        S = V @ V.T
        iu = torch.triu_indices(S.shape[0], S.shape[1], offset=1, device=device)
        comp_pair_cos.extend(S[iu[0], iu[1]].detach().float().tolist())
    reg_comp = _summ(comp_pair_cos)

    reg_to_centroid_vals: List[float] = []
    if kept_roots:
        C = torch.stack(centroids, dim=0)
        for bi in range(B):
            idxs = [pj - 1 for pj in reg_by_image[bi] if 1 <= pj <= P]
            if not idxs:
                continue
            V = patches_norm_cent[bi, torch.tensor(idxs, device=device), :]
            reg_to_centroid_vals.extend((V @ C.T).max(dim=1).values.detach().float().tolist())
    reg_cent = _summ(reg_to_centroid_vals)

    # PCA diagnostics
    pca_all_top1: List[float] = []
    pca_all_top5: List[float] = []
    pca_all_pr: List[float] = []
    pca_nonreg_top1: List[float] = []
    pca_nonreg_top5: List[float] = []
    pca_nonreg_pr: List[float] = []

    for bi in range(B):
        x_all = patches_centered[bi]
        st_all = _pca_var_explained_and_pr(x_all, topk=5)
        if st_all["top1"] is not None:
            pca_all_top1.append(float(st_all["top1"]))
            pca_all_top5.append(float(st_all["topk"]))
            pca_all_pr.append(float(st_all["pr"]))

        idx = (~reg_mask[bi]).nonzero(as_tuple=True)[0]
        if idx.numel() >= 2:
            x_nr = x_all.index_select(0, idx)
            st_nr = _pca_var_explained_and_pr(x_nr, topk=5)
            if st_nr["top1"] is not None:
                pca_nonreg_top1.append(float(st_nr["top1"]))
                pca_nonreg_top5.append(float(st_nr["topk"]))
                pca_nonreg_pr.append(float(st_nr["pr"]))

    pca_all = {
        "pca_top1_var_centered_all_mean": _summ(pca_all_top1)["mean"],
        "pca_top5_var_centered_all_mean": _summ(pca_all_top5)["mean"],
        "pca_pr_centered_all_mean": _summ(pca_all_pr)["mean"],
    }
    pca_nonreg = {
        "pca_top1_var_centered_nonreg_mean": _summ(pca_nonreg_top1)["mean"],
        "pca_top5_var_centered_nonreg_mean": _summ(pca_nonreg_top5)["mean"],
        "pca_pr_centered_nonreg_mean": _summ(pca_nonreg_pr)["mean"],
    }

    # reg/nonreg subspace overlap probes
    ov_regcent_top1_vals: List[float] = []
    ov_regcent_topk_vals: List[float] = []
    ov_maxsv_vals: List[float] = []
    ov_pa_mean_vals: List[float] = []
    ov_pa_min_vals: List[float] = []
    ov_pa_max_vals: List[float] = []

    for bi in range(B):
        over = compute_reg_nonreg_subspace_overlap(
            patches_centered=patches_centered[bi],
            reg_mask=reg_mask[bi],
            k_nonreg=5,
            k_reg=5,
        )
        if over["overlap_regcent_vs_nonreg_top1_cos"] is not None:
            ov_regcent_top1_vals.append(float(over["overlap_regcent_vs_nonreg_top1_cos"]))
        if over["overlap_regcent_vs_nonreg_topk_max_cos"] is not None:
            ov_regcent_topk_vals.append(float(over["overlap_regcent_vs_nonreg_topk_max_cos"]))
        if over["subspace_overlap_maxsv"] is not None:
            ov_maxsv_vals.append(float(over["subspace_overlap_maxsv"]))
        if over["principal_angles_deg_mean"] is not None:
            ov_pa_mean_vals.append(float(over["principal_angles_deg_mean"]))
        if over["principal_angles_deg_min"] is not None:
            ov_pa_min_vals.append(float(over["principal_angles_deg_min"]))
        if over["principal_angles_deg_max"] is not None:
            ov_pa_max_vals.append(float(over["principal_angles_deg_max"]))

    ov_regcent_top1 = _summ(ov_regcent_top1_vals)["mean"]
    ov_regcent_topk = _summ(ov_regcent_topk_vals)["mean"]
    ov_maxsv = _summ(ov_maxsv_vals)["mean"]
    ov_pa_mean = _summ(ov_pa_mean_vals)["mean"]
    ov_pa_min = _summ(ov_pa_min_vals)["mean"]
    ov_pa_max = _summ(ov_pa_max_vals)["mean"]

    compDiag = {
        "comp_total": int(comp_total),
        "comp_cov_ok": int(comp_cov_ok),
        "comp_mpi_ok": int(comp_mpi_ok),
        "comp_both_ok": int(comp_both_ok),
        "comp_best_coverage": float(best_cov) if comp_total > 0 else None,
        "comp_best_min_max_per_image": int(best_min_maxpi) if best_min_maxpi is not None else None,
        "comp_biggest_size": int(biggest_size) if comp_total > 0 else None,
        "comp_biggest_coverage": float(biggest_cov) if comp_total > 0 else None,
        "comp_biggest_max_per_image": int(biggest_maxpi) if comp_total > 0 else None,
        "comp_frac_cov_ok": _safe_div(float(comp_cov_ok), float(comp_total)),
        "comp_frac_mpi_ok": _safe_div(float(comp_mpi_ok), float(comp_total)),
        "comp_frac_both_ok": _safe_div(float(comp_both_ok), float(comp_total)),
    }

    stats = {
        "grid_side": int(side) if side is not None else None,
        "num_local_pairs_used": int(num_local_pairs_used),

        "reg_patches_per_image_min": int(min(reg_counts)) if reg_counts else None,
        "reg_patches_per_image_mean": float(sum(reg_counts) / len(reg_counts)) if reg_counts else None,
        "reg_patches_per_image_max": int(max(reg_counts)) if reg_counts else None,

        "nonreg_patches_per_image_min": int(min(nonreg_counts)) if nonreg_counts else None,
        "nonreg_patches_per_image_mean": float(sum(nonreg_counts) / len(nonreg_counts)) if nonreg_counts else None,
        "nonreg_patches_per_image_max": int(max(nonreg_counts)) if nonreg_counts else None,

        "distant_local_cos_uncentered_mean": local_raw["mean"],
        "distant_local_cos_centered_mean": local_cent["mean"],

        "offdiag_allpatch_cos_uncentered_mean": offdiag_raw["mean"],
        "offdiag_allpatch_cos_centered_mean": offdiag_cent["mean"],

        "offdiag_nonreg_cos_uncentered_mean": offdiag_nonreg_raw["mean"],
        "offdiag_nonreg_cos_centered_mean": offdiag_nonreg_cent["mean"],

        **mu_img_raw_stats,
        **mu_img_g_stats,

        "reg_component_pair_cos_mean": reg_comp["mean"],
        "reg_to_centroid_cos_mean": reg_cent["mean"],

        **pca_all,
        **pca_nonreg,

        "overlap_regcent_vs_nonreg_top1_cos_mean": ov_regcent_top1,
        "overlap_regcent_vs_nonreg_topk_max_cos_mean": ov_regcent_topk,
        "overlap_subspace_maxsv_mean": ov_maxsv,
        "overlap_principal_angles_deg_mean": ov_pa_mean,
        "overlap_principal_angles_deg_min": ov_pa_min,
        "overlap_principal_angles_deg_max": ov_pa_max,

        **compDiag,
    }

    return {
        "reg_by_image": reg_by_image,
        "components": sorted(comp_info, key=lambda x: (-x["coverage"], -x["size"])),
        "kept_component_roots": [int(r) for r in kept_roots],
        "meta": {"B": B, "T": T, "D": D, "P": P, "M": M},
        "centroids": centroid_meta,
        "stats": stats,
        "mu_global": mu_global.detach(),
        "reg_mask_bp": reg_mask.detach(),
    }


def plot_model_curves(model_alias: str, blocks: List[int], series: Dict[str, List[Optional[float]]], out_dir: str) -> None:
    _ensure_dir(out_dir)

    def _plot(names: List[str], title: str, fname: str, ylab: str) -> None:
        plt.figure()
        for n in names:
            ys = series.get(n, [])
            xs2 = [x for x, y in zip(blocks, ys) if y is not None]
            ys2 = [y for y in ys if y is not None]
            if xs2:
                plt.plot(xs2, ys2, label=n)
        plt.title(f"{model_alias} :: {title}")
        plt.xlabel("block")
        plt.ylabel(ylab)
        plt.legend()
        _savefig(os.path.join(out_dir, f"{model_alias}__{fname}.png"))

    _plot(
        ["reg_patches_per_image_mean", "nonreg_patches_per_image_mean"],
        "Register / non-register counts",
        "counts",
        "count (mean per image)",
    )

    _plot(
        [
            "distant_local_cos_uncentered_mean",
            "distant_local_cos_centered_mean",
            "offdiag_allpatch_cos_uncentered_mean",
            "offdiag_allpatch_cos_centered_mean",
            "offdiag_nonreg_cos_centered_mean",
        ],
        "Cosine baselines",
        "cosine_baselines",
        "cosine",
    )

    _plot(
        ["reg_component_pair_cos_mean", "reg_to_centroid_cos_mean"],
        "Register coherence",
        "register_coherence",
        "cosine",
    )

    _plot(
        [
            "pca_top1_var_centered_all_mean",
            "pca_top5_var_centered_all_mean",
            "pca_top1_var_centered_nonreg_mean",
            "pca_top5_var_centered_nonreg_mean",
        ],
        "PCA variance explained (centered)",
        "pca_var",
        "fraction",
    )
    _plot(
        ["pca_pr_centered_all_mean", "pca_pr_centered_nonreg_mean"],
        "Participation ratio (centered)",
        "pca_pr",
        "PR (effective dims)",
    )

    _plot(
        [
            "overlap_regcent_vs_nonreg_top1_cos_mean",
            "overlap_regcent_vs_nonreg_topk_max_cos_mean",
            "overlap_subspace_maxsv_mean",
        ],
        "Reg/nonreg overlap (higher = more collapse)",
        "overlap_strength",
        "overlap / cosine",
    )

    _plot(
        [
            "overlap_principal_angles_deg_mean",
            "overlap_principal_angles_deg_min",
            "overlap_principal_angles_deg_max",
        ],
        "Principal angles between nonreg and reg subspaces",
        "principal_angles",
        "degrees",
    )

    _plot(
        ["comp_total", "comp_cov_ok", "comp_mpi_ok", "comp_both_ok"],
        "Component diagnostics: counts (total / pass filters)",
        "compdiag_counts",
        "count",
    )

    _plot(
        ["comp_frac_cov_ok", "comp_frac_mpi_ok", "comp_frac_both_ok"],
        "Component diagnostics: pass fractions",
        "compdiag_fracs",
        "fraction",
    )


def plot_all_models_overlay(
    blocks: List[int],
    model_to_series: Dict[str, Dict[str, List[Optional[float]]]],
    metric: str,
    out_dir: str,
    fname: str,
    title: str,
    ylab: str,
) -> None:
    _ensure_dir(out_dir)
    plt.figure()
    for alias, series in model_to_series.items():
        ys = series.get(metric, [])
        xs2 = [x for x, y in zip(blocks, ys) if y is not None]
        ys2 = [y for y in ys if y is not None]
        if xs2:
            plt.plot(xs2, ys2, label=alias)
    plt.title(title)
    plt.xlabel("block")
    plt.ylabel(ylab)
    plt.legend()
    _savefig(os.path.join(out_dir, fname))


@dataclass
class RegProjCfg:
    # register detection basis construction (cross-image) on ref blocks
    sim_thr: float = 0.95
    cover_frac: float = 0.85
    max_per_image: int = 64
    chunk_rows: int = 2048
    refine_topk: int = 32

    ref_blocks: Tuple[int, ...] = (11, 12)

    k_reg_basis: int = 16
    k_patch_basis_for_angles: int = 16

    use_global_mean_centering: bool = True
    use_image_mean_centering: bool = True

    topk_share_k: int = 16
    eps: float = 1e-12

    seed: int = 0
    device: str = "cuda"

    out_dir: str = "out_registers/unified/regproj"
    plots_dir: str = "out_registers/unified/regproj/plots"


@torch.no_grad()
def _apply_centering_with_mu_global(
    patches_bpd: torch.Tensor,
    mu_global: Optional[torch.Tensor],
    use_image_mean_centering: bool,
) -> torch.Tensor:
    x = patches_bpd
    if mu_global is not None:
        x = x - mu_global.view(1, 1, -1)
    if use_image_mean_centering:
        x = x - x.mean(dim=1, keepdim=True)
    return x


@torch.no_grad()
def _gini(x: torch.Tensor, eps: float = 1e-12) -> float:
    if x.numel() == 0:
        return float("nan")
    x = x.detach().float()
    x = torch.clamp(x, min=0.0)
    s = float(x.sum().item())
    if s <= eps:
        return 0.0
    xs = torch.sort(x)[0]
    n = xs.numel()
    idx = torch.arange(1, n + 1, device=xs.device, dtype=xs.dtype)
    g = (2.0 * (idx * xs).sum() / (n * (xs.sum() + eps))) - (n + 1.0) / n
    return float(g.item())


@torch.no_grad()
def projection_metrics_per_block(
    tokens_btd: torch.Tensor,
    U_reg: torch.Tensor,
    mu_global_ref: Optional[torch.Tensor],
    cfg: RegProjCfg,
) -> Dict[str, float]:
    patches = tokens_btd[:, 1:, :].float()
    x = _apply_centering_with_mu_global(
        patches_bpd=patches,
        mu_global=mu_global_ref if cfg.use_global_mean_centering else None,
        use_image_mean_centering=cfg.use_image_mean_centering,
    )
    z = torch.einsum("bpd,dk->bpk", x, U_reg)
    e_proj = (z * z).sum(dim=-1)
    e_all = (x * x).sum(dim=-1)
    frac = e_proj / (e_all + float(cfg.eps))

    frac_mean = frac.mean(dim=1)
    frac_max = frac.max(dim=1).values

    B, P = e_proj.shape
    k = max(1, min(int(cfg.topk_share_k), P))
    topk = torch.topk(e_proj, k=k, dim=1, largest=True).values.sum(dim=1)
    share = topk / (e_proj.sum(dim=1) + float(cfg.eps))

    gini_vals = [_gini(e_proj[bi], eps=float(cfg.eps)) for bi in range(B)]
    gini = torch.tensor(gini_vals, device=frac.device, dtype=torch.float32)

    return {
        "proj_frac_mean": float(frac_mean.mean().item()),
        "proj_frac_max_mean": float(frac_max.mean().item()),
        "proj_topk_share_mean": float(share.mean().item()),
        "proj_gini_mean": float(gini.mean().item()),
    }


@torch.no_grad()
def reg_vs_patch_subspace_angles(
    all_patches_bpd: torch.Tensor,
    U_reg: torch.Tensor,
    k_patch: int,
) -> Dict[str, Optional[float]]:
    B, P, D = all_patches_bpd.shape
    X = all_patches_bpd.reshape(B * P, D)
    if X.shape[0] < 2:
        return {"ov_maxsv": None, "pa_deg_mean": None, "pa_deg_min": None, "pa_deg_max": None}
    U_patch = _pca_basis(X, k=int(k_patch))
    if U_patch is None or U_reg is None:
        return {"ov_maxsv": None, "pa_deg_mean": None, "pa_deg_min": None, "pa_deg_max": None}

    sig, ang = _principal_angles_deg(U_reg, U_patch)
    if sig.numel() == 0 or ang.numel() == 0:
        return {"ov_maxsv": None, "pa_deg_mean": None, "pa_deg_min": None, "pa_deg_max": None}

    return {
        "ov_maxsv": float(sig.max().item()),
        "pa_deg_mean": float(ang.mean().item()),
        "pa_deg_min": float(ang.min().item()),
        "pa_deg_max": float(ang.max().item()),
    }


def plot_overlay_models(
    blocks: List[int],
    alias_to_series: Dict[str, Dict[str, List[Optional[float]]]],
    metric: str,
    title: str,
    ylab: str,
    out_path: str,
) -> None:
    plt.figure()
    for alias, series in alias_to_series.items():
        ys = series.get(metric, [])
        xs2 = [x for x, y in zip(blocks, ys) if y is not None]
        ys2 = [y for y in ys if y is not None]
        if xs2:
            plt.plot(xs2, ys2, label=alias)
    plt.title(title)
    plt.xlabel("block")
    plt.ylabel(ylab)
    plt.legend()
    _savefig(out_path)


@dataclass
class ModelSpec:
    alias: str
    model_id: str
    probe_tag: Optional[str] = None


def parse_models(models_raw: List[Tuple[Any, ...]]) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for t in models_raw:
        if len(t) == 2:
            out.append(ModelSpec(alias=str(t[0]), model_id=str(t[1]), probe_tag=None))
        elif len(t) == 3:
            out.append(ModelSpec(alias=str(t[0]), model_id=str(t[1]), probe_tag=str(t[2]) if t[2] is not None else None))
        else:
            raise ValueError(f"MODELS entries must be (alias, id) or (alias, id, probe_tag). Got: {t}")
    return out


def build_pairs_by_probe_tag(specs: List[ModelSpec]) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
    """
    Returns:
      pairs: key like "1" -> (aliasA, aliasB)
      warnings: list of warning strings
    """
    warnings: List[str] = []
    tag_to_alias: Dict[str, str] = {}

    for ms in specs:
        if ms.probe_tag is None:
            continue
        tag = ms.probe_tag.strip()
        # must be A<digits> or B<digits>
        if len(tag) < 2 or tag[0] not in ("A", "B") or (not tag[1:].isdigit()):
            warnings.append(f"Illegal probe_tag '{tag}' on model '{ms.alias}'. Expected 'A1','B1',... Skipping.")
            continue
        if tag in tag_to_alias:
            warnings.append(f"Duplicate probe_tag '{tag}' for '{ms.alias}' and '{tag_to_alias[tag]}'. Using first, skipping later.")
            continue
        tag_to_alias[tag] = ms.alias

    pairs: Dict[str, Tuple[str, str]] = {}
    nums = sorted(set([t[1:] for t in tag_to_alias.keys()]))
    for n in nums:
        a = tag_to_alias.get(f"A{n}")
        b = tag_to_alias.get(f"B{n}")
        if a is None or b is None:
            warnings.append(f"Missing pair for index {n}: found A{n}={a} B{n}={b}. Skipping pair.")
            continue
        pairs[n] = (a, b)

    return pairs, warnings


def _c_green(s: str) -> str:
    return Fore.GREEN + Style.BRIGHT + s + Style.RESET_ALL

def _c_yellow(s: str) -> str:
    return Fore.YELLOW + Style.BRIGHT + s + Style.RESET_ALL

def _c_red(s: str) -> str:
    return Fore.RED + Style.BRIGHT + s + Style.RESET_ALL

def _c_cyan(s: str) -> str:
    return Fore.CYAN + Style.BRIGHT + s + Style.RESET_ALL


def _grade_3way(val: Optional[float], good_thr: float, bad_thr: float, higher_is_better: bool = True) -> str:
    """
    Returns colored string label for a scalar.
      green: good
      yellow: borderline
      red: bad
    """
    if val is None:
        return _c_yellow("NA")
    if higher_is_better:
        if val >= good_thr:
            return _c_green(f"{val:.4f}")
        if val <= bad_thr:
            return _c_red(f"{val:.4f}")
        return _c_yellow(f"{val:.4f}")
    else:
        if val <= good_thr:
            return _c_green(f"{val:.4f}")
        if val >= bad_thr:
            return _c_red(f"{val:.4f}")
        return _c_yellow(f"{val:.4f}")


def heuristic_summary(
    specs: List[ModelSpec],
    blocks: List[int],
    recoverability: Optional[Dict[str, Dict[str, List[float]]]],
    cross_series: Optional[Dict[str, Dict[str, List[Optional[float]]]]],
    regproj_series: Optional[Dict[str, Dict[str, List[Optional[float]]]]],
    pairs: Dict[str, Tuple[str, str]],
    have_pretrained: bool,
) -> None:
    print("\n" + "=" * 92)
    print(_c_cyan("Heuristic auto-summary"))
    print("=" * 92)

    baseline_alias = "pretrained" if have_pretrained else None

    # Recoverability classification per model
    if recoverability is not None:
        print("\n" + _c_cyan("[Recoverability] reg_only vs nonreg_only (higher=better cosine to baseline embedding)"))
        for ms in specs:
            if ms.alias not in recoverability:
                continue
            reg = recoverability[ms.alias]["reg_only"]
            non = recoverability[ms.alias]["nonreg_only"]
            # focus on late blocks where “global summary” effects should show up
            late_idx = [i for i, b in enumerate(blocks) if b >= max(blocks[-6], blocks[0])]
            reg_late = sum(reg[i] for i in late_idx) / max(1, len(late_idx))
            non_late = sum(non[i] for i in late_idx) / max(1, len(late_idx))
            gap = reg_late - non_late

            # rule-of-thumb labels
            if reg_late > 0.88 and non_late < 0.65 and gap > 0.20:
                label = _c_green("reg-dominant recoverability")
            elif non_late > 0.88 and reg_late < 0.65 and (-gap) > 0.20:
                label = _c_green("nonreg-dominant recoverability")
            elif (reg_late > 0.80 and non_late > 0.80):
                label = _c_yellow("both high (projection not very destructive or U_reg too broad)")
            elif (reg_late < 0.70 and non_late < 0.70):
                label = _c_red("both low (projection destroys embedding: mismatch or too-small subspace)")
            else:
                label = _c_yellow("mixed/ambiguous")

            print(
                f"  {ms.alias:20s} "
                f"late reg={_grade_3way(reg_late, 0.88, 0.75, True)} "
                f"late non={_grade_3way(non_late, 0.88, 0.75, True)} "
                f"gap={_c_cyan(f'{gap:+.3f}')}  => {label}"
            )

    # Cross-image overlap collapse hints
    if cross_series is not None:
        print("\n" + _c_cyan("[Cross-image] reg/nonreg overlap probes (collapse indicators)"))
        for ms in specs:
            s = cross_series.get(ms.alias)
            if not s:
                continue
            # Use last block’s overlap as a crude “steady state” indicator (also show mean over last 6 blocks)
            metric_sv = s.get("overlap_subspace_maxsv_mean", [])
            metric_pa = s.get("overlap_principal_angles_deg_mean", [])
            metric_regs = s.get("reg_patches_per_image_mean", [])

            def _tail_mean(vs: List[Optional[float]], tail: int = 6) -> Optional[float]:
                xs = [x for x in vs[-tail:] if x is not None]
                return None if not xs else sum(xs) / len(xs)

            sv_late = _tail_mean(metric_sv, 6)
            pa_late = _tail_mean(metric_pa, 6)
            regs_late = _tail_mean(metric_regs, 6)

            # collapse-ish if sv high and angles small
            if sv_late is not None and pa_late is not None:
                if sv_late > 0.92 and pa_late < 18.0:
                    verdict = _c_red("reg/nonreg subspaces strongly overlapping (collapse risk)")
                elif sv_late < 0.75 and pa_late > 30.0:
                    verdict = _c_green("reg/nonreg more separated (less collapse)")
                else:
                    verdict = _c_yellow("intermediate overlap")
            else:
                verdict = _c_yellow("NA")

            print(
                f"  {ms.alias:20s} "
                f"ov_maxsv(late)={_grade_3way(sv_late, 0.75, 0.92, higher_is_better=False)} "
                f"PAdeg_mean(late)={_grade_3way(pa_late, 20.0, 35.0, higher_is_better=False)} "
                f"regs/image(late)={_c_cyan('NA' if regs_late is None else f'{regs_late:.2f}')}  => {verdict}"
            )

    # Regproj metrics vs baseline register basis
    if regproj_series is not None:
        ref_note = baseline_alias if baseline_alias is not None else "ref (non-pretrained)"
        print("\n" + _c_cyan(f"[RegProj] projection-energy vs {ref_note} register basis (spread vs concentration)"))
        for ms in specs:
            s = regproj_series.get(ms.alias)
            if not s:
                continue

            def _tail_mean(vs: List[Optional[float]], tail: int = 6) -> Optional[float]:
                xs = [x for x in vs[-tail:] if x is not None]
                return None if not xs else sum(xs) / len(xs)

            frac = _tail_mean(s.get("proj_frac_mean", []), 6)
            gini = _tail_mean(s.get("proj_gini_mean", []), 6)
            topk = _tail_mean(s.get("proj_topk_share_mean", []), 6)
            ov = _tail_mean(s.get("ov_maxsv", []), 6)

            # heuristics:
            # - frac higher means more patch energy lies in reg basis
            # - gini lower means that reg-basis energy is spread across patches
            # - topk share lower means less dominated by a few patches
            if frac is not None and gini is not None:
                if frac > 0.65 and gini < 0.35:
                    verdict = _c_green("high reg-basis energy, well-spread across patches")
                elif frac > 0.70 and gini > 0.55:
                    verdict = _c_yellow("high energy but concentrated (few patches dominate)")
                elif frac < 0.40:
                    verdict = _c_red("low reg-basis energy (less aligned to ref reg basis)")
                else:
                    verdict = _c_yellow("intermediate/mixed")
            else:
                verdict = _c_yellow("NA")

            print(
                f"  {ms.alias:20s} "
                f"frac={_grade_3way(frac, 0.65, 0.45, True)} "
                f"gini={_grade_3way(gini, 0.35, 0.55, higher_is_better=False)} "
                f"topkShare={_c_cyan('NA' if topk is None else f'{topk:.3f}')} "
                f"ovMaxSV={_c_cyan('NA' if ov is None else f'{ov:.3f}')}  => {verdict}"
            )

    # Paired comparisons (A1 vs B1 etc.)
    if pairs and regproj_series is not None:
        print("\n" + _c_cyan("[Paired probe] A# vs B# deltas (late-block means; sign = A - B)"))
        for idx, (a_alias, b_alias) in pairs.items():
            sa = regproj_series.get(a_alias)
            sb = regproj_series.get(b_alias)
            if not sa or not sb:
                print(_c_yellow(f"  pair {idx}: missing series for {a_alias} or {b_alias}"))
                continue

            def _tail_mean(vs: List[Optional[float]], tail: int = 6) -> Optional[float]:
                xs = [x for x in vs[-tail:] if x is not None]
                return None if not xs else sum(xs) / len(xs)

            a_frac = _tail_mean(sa.get("proj_frac_mean", []), 6)
            b_frac = _tail_mean(sb.get("proj_frac_mean", []), 6)
            a_gini = _tail_mean(sa.get("proj_gini_mean", []), 6)
            b_gini = _tail_mean(sb.get("proj_gini_mean", []), 6)

            d_frac = None if (a_frac is None or b_frac is None) else (a_frac - b_frac)
            d_gini = None if (a_gini is None or b_gini is None) else (a_gini - b_gini)

            # just highlight magnitude + direction
            def _fmt_delta(d: Optional[float]) -> str:
                if d is None:
                    return _c_yellow("NA")
                # emphasize large deltas
                if abs(d) >= 0.10:
                    return _c_green(f"{d:+.3f}") if d > 0 else _c_red(f"{d:+.3f}")
                return _c_yellow(f"{d:+.3f}")

            print(
                f"  pair {idx}: {a_alias} vs {b_alias} | "
                f"Δfrac={_fmt_delta(d_frac)}  Δgini={_fmt_delta(d_gini)}"
            )

def main() -> None:
    colorama_init(autoreset=True)

    args = parse_args()

    device = args.device
    out_root = args.out_root
    _ensure_dir(out_root)

    img_paths = _list_images(args.image_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")
    print(f"[Images] {len(img_paths)} from {args.image_dir}")

    specs = parse_models(MODELS)
    have_pretrained = any(ms.alias == "pretrained" for ms in specs)

    pairs, pair_warnings = build_pairs_by_probe_tag(specs)
    for w in pair_warnings:
        print(_c_yellow("[probe_tag warning] ") + w)

    models: Dict[str, torch.nn.Module] = {}
    preprocessors: Dict[str, Any] = {}
    images_by_model: Dict[str, torch.Tensor] = {}

    for ms in specs:
        m, pre = load_any_clip(ms.model_id, device=device)
        models[ms.alias] = m
        preprocessors[ms.alias] = pre
        images_by_model[ms.alias] = load_images(img_paths, pre, device=device)
        print(f"[Model] {ms.alias} :: {ms.model_id} | images={tuple(images_by_model[ms.alias].shape)}")

    # Capture tokens once per model (used by crossimage + regproj)
    tokens_by_model: Dict[str, Dict[int, torch.Tensor]] = {}
    common_blocks: Optional[List[int]] = None
    for ms in specs:
        print(f"\n=== Capturing visual block tokens: {ms.alias} ===")
        tb = capture_visual_block_tokens(models[ms.alias], images_by_model[ms.alias])
        tokens_by_model[ms.alias] = tb
        blks = sorted(tb.keys())
        if common_blocks is None:
            common_blocks = blks
        else:
            if blks != common_blocks:
                raise RuntimeError(f"Block set mismatch for {ms.alias}: got {blks}, expected {common_blocks}")

    assert common_blocks is not None
    all_blocks = common_blocks

    recoverability_out: Optional[Dict[str, Dict[str, List[float]]]] = None
    if not args.no_recoverability:
        rec_dir = os.path.join(out_root, "recoverability")
        _ensure_dir(rec_dir)

        blocks = _parse_blocks(args.rec_blocks)
        ref_blocks = tuple(int(x.strip()) for x in args.rec_ref_blocks.split(",") if x.strip())

        rec_ref_alias = args.rec_ref_alias
        if rec_ref_alias not in models:
            # fallback: if pretrained missing, use first model
            rec_ref_alias = "pretrained" if ("pretrained" in models) else specs[0].alias
            print(_c_yellow(f"[recoverability] --rec_ref_alias not found; using '{rec_ref_alias}'"))

        abs_thresh = None if float(args.rec_abs_thresh) <= 0 else float(args.rec_abs_thresh)
        cfg_norm = RegDetectCfgNorm(
            adaptive_mult=float(args.rec_adaptive_mult),
            abs_thresh=abs_thresh,
            max_per_img=int(args.rec_max_per_img),
            use_patch_only=True,
        )

        print(f"\n=== [Recoverability] Building U_reg (norm-threshold) from '{rec_ref_alias}', blocks={ref_blocks} ===")
        X = collect_register_vectors_norm(models[rec_ref_alias], images_by_model[rec_ref_alias], ref_blocks, cfg_norm)
        if X.numel() == 0:
            raise RuntimeError("[Recoverability] No register vectors collected; relax thresholds or change ref blocks.")

        U_reg, mu = build_U_from_X(X, k=int(args.rec_proj_k), center=bool(args.rec_center))
        print(_c_cyan(f"[Recoverability] U_reg shape={tuple(U_reg.shape)} | reg_vecs={X.shape[0]} | D={U_reg.shape[0]}"))

        torch.save(
            {"U_reg": U_reg.detach().cpu(), "mu": mu.detach().cpu(), "ref_alias": rec_ref_alias, "ref_blocks": ref_blocks},
            os.path.join(rec_dir, f"U_reg_norm__{rec_ref_alias}__blocks_{'_'.join(map(str, ref_blocks))}.pt"),
        )

        recoverability_out = {}
        rows: List[Dict[str, Any]] = []

        for ms in specs:
            print(f"\n=== [Recoverability] {ms.alias} ===")
            series = run_recoverability(
                models[ms.alias],
                images_by_model[ms.alias],
                U_reg.to(device),
                blocks=blocks,
                project_cls=bool(args.rec_project_cls),
            )
            recoverability_out[ms.alias] = series
            plot_recoverability_curves(ms.alias, blocks, series, rec_dir)

            for i, b in enumerate(blocks):
                for mode in ("reg_only", "nonreg_only"):
                    rows.append({
                        "model": ms.alias,
                        "block": int(b),
                        "mode": mode,
                        "cos_mean": float(series[mode][i]),
                        "ref_alias": rec_ref_alias,
                        "ref_blocks": ",".join(map(str, ref_blocks)),
                        "proj_k": int(args.rec_proj_k),
                        "adaptive_mult": float(args.rec_adaptive_mult),
                        "abs_thresh": float(args.rec_abs_thresh),
                        "max_per_img": int(args.rec_max_per_img),
                        "project_cls": bool(args.rec_project_cls),
                    })

        plot_recoverability_combined(blocks, recoverability_out, rec_dir)

        csv_path = os.path.join(rec_dir, "recoverability.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[Recoverability] wrote: {csv_path}")



    cross_series: Optional[Dict[str, Dict[str, List[Optional[float]]]]] = None
    cross_json_paths: Dict[str, str] = {}
    if not args.no_crossimage:
        cx_dir = os.path.join(out_root, "crossimage")
        _ensure_dir(cx_dir)

        make_plots = bool(args.cx_make_plots) and (not bool(args.cx_no_plots))
        cfg_cx = RegDetectCfgCross(
            sim_thr=float(args.cx_sim_thr),
            cover_frac=float(args.cx_cover_frac),
            max_per_image=int(args.cx_max_per_image),
            chunk_rows=int(args.cx_chunk_rows),
            refine_topk=int(args.cx_refine_topk),
            device=device,
            make_plots=make_plots,
            plots_dir=os.path.join(cx_dir, "plots"),
        )

        if cfg_cx.make_plots:
            _ensure_dir(cfg_cx.plots_dir)

        cross_series = {}
        all_blocks_ref = all_blocks

        for ms in specs:
            print(f"\n=== [CrossImage] {ms.alias} ===")
            series: Dict[str, List[Optional[float]]] = {}
            per_block: Dict[int, Any] = {}

            def _push(k: str, v: Any) -> None:
                series.setdefault(k, []).append(None if v is None else float(v))

            for b in all_blocks_ref:
                tokens_btd = tokens_by_model[ms.alias][b].float()
                out = find_register_tokens_cross_image(tokens_btd, cfg_cx)
                st = out.get("stats", {})

                # store series for plots + later summary
                keys = [
                    "reg_patches_per_image_mean",
                    "nonreg_patches_per_image_mean",
                    "distant_local_cos_uncentered_mean",
                    "distant_local_cos_centered_mean",
                    "offdiag_allpatch_cos_uncentered_mean",
                    "offdiag_allpatch_cos_centered_mean",
                    "offdiag_nonreg_cos_centered_mean",
                    "reg_component_pair_cos_mean",
                    "reg_to_centroid_cos_mean",
                    "patch_mean_norm_raw_mean",
                    "patch_mean_norm_after_global_mean",
                    "pca_top1_var_centered_all_mean",
                    "pca_top5_var_centered_all_mean",
                    "pca_pr_centered_all_mean",
                    "pca_top1_var_centered_nonreg_mean",
                    "pca_top5_var_centered_nonreg_mean",
                    "pca_pr_centered_nonreg_mean",
                    "overlap_regcent_vs_nonreg_top1_cos_mean",
                    "overlap_regcent_vs_nonreg_topk_max_cos_mean",
                    "overlap_subspace_maxsv_mean",
                    "overlap_principal_angles_deg_mean",
                    "overlap_principal_angles_deg_min",
                    "overlap_principal_angles_deg_max",
                    "comp_total",
                    "comp_cov_ok",
                    "comp_mpi_ok",
                    "comp_both_ok",
                    "comp_frac_cov_ok",
                    "comp_frac_mpi_ok",
                    "comp_frac_both_ok",
                    "comp_best_coverage",
                    "comp_biggest_coverage",
                    "comp_best_min_max_per_image",
                    "comp_biggest_max_per_image",
                    "comp_biggest_size",
                ]
                for k in keys:
                    _push(k, st.get(k))

                counts = [len(x) for x in out["reg_by_image"]]
                print(
                    f"[CrossImage block {b:02d}] regs/image min={min(counts)} mean={sum(counts)/len(counts):.2f} max={max(counts)}"
                    f" | kept_comps={len(out['kept_component_roots'])}"
                    f" | ovMaxSV={st.get('overlap_subspace_maxsv_mean')} PAdegMean={st.get('overlap_principal_angles_deg_mean')}"
                    f" | comp(total/both_ok)={st.get('comp_total')}/{st.get('comp_both_ok')}"
                )

                # keep the full per-block structure for JSON
                # (contains masks, reg lists, and meta; can be large)
                per_block[int(b)] = {
                    "reg_by_image": out["reg_by_image"],
                    "kept_component_roots": out["kept_component_roots"],
                    "centroids": out["centroids"],
                    "stats": out["stats"],
                    "meta": out["meta"],
                    "components": out["components"],
                }

            # save JSON per model
            save = {
                "model_alias": ms.alias,
                "model_id": ms.model_id,
                "images": img_paths,
                "cfg": cfg_cx.__dict__,
                "per_block": per_block,
            }
            out_path = os.path.join(cx_dir, f"{ms.alias}__crossimage_register_tokens_by_block.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(save, f, indent=2)
            cross_json_paths[ms.alias] = out_path
            print(f"[CrossImage] saved: {out_path}")

            if cfg_cx.make_plots:
                plot_model_curves(ms.alias, all_blocks_ref, series, cfg_cx.plots_dir)

            cross_series[ms.alias] = series

        # selected all-model overlays
        if cfg_cx.make_plots and cross_series:
            plot_all_models_overlay(
                all_blocks_ref,
                cross_series,
                metric="overlap_subspace_maxsv_mean",
                out_dir=cfg_cx.plots_dir,
                fname="ALL__overlap_maxsv.png",
                title="All models :: subspace overlap max singular value (nonreg vs reg)",
                ylab="max sv (0..1)",
            )
            plot_all_models_overlay(
                all_blocks_ref,
                cross_series,
                metric="overlap_principal_angles_deg_mean",
                out_dir=cfg_cx.plots_dir,
                fname="ALL__principal_angles_mean.png",
                title="All models :: principal angles mean (nonreg vs reg subspaces)",
                ylab="degrees",
            )
            print(f"[CrossImage] plots saved to: {cfg_cx.plots_dir}")


    regproj_series: Optional[Dict[str, Dict[str, List[Optional[float]]]]] = None
    if not args.no_regproj:
        rp = RegProjCfg(
            ref_blocks=tuple(int(x.strip()) for x in args.rp_ref_blocks.split(",") if x.strip()),
            k_reg_basis=int(args.rp_k_reg_basis),
            k_patch_basis_for_angles=int(args.rp_k_patch_basis),
            use_global_mean_centering=bool(args.rp_use_global_centering),
            use_image_mean_centering=bool(args.rp_use_image_centering),
            topk_share_k=int(args.rp_topk_share_k),
            device=device,
            out_dir=os.path.join(out_root, "regproj"),
            plots_dir=os.path.join(out_root, "regproj", "plots"),
        )
        _ensure_dir(rp.out_dir)
        _ensure_dir(rp.plots_dir)

        # choose reference alias
        rp_ref_alias = args.rp_ref_alias
        if rp_ref_alias not in models:
            rp_ref_alias = "pretrained" if ("pretrained" in models) else specs[0].alias
            print(_c_yellow(f"[RegProj] --rp_ref_alias not found; using '{rp_ref_alias}'"))

        print(f"\n=== [RegProj] Building reference register basis from '{rp_ref_alias}', ref_blocks={rp.ref_blocks} ===")
        # reuse cross-image detection to build U_reg:
        cfg_tmp = RegDetectCfgCross(
            sim_thr=float(rp.sim_thr),
            cover_frac=float(rp.cover_frac),
            max_per_image=int(rp.max_per_image),
            chunk_rows=int(rp.chunk_rows),
            refine_topk=int(rp.refine_topk),
            device=device,
            use_global_mean_centering=True,
            use_image_mean_centering=False,
            make_plots=False,
            plots_dir="",
        )

        reg_vecs: List[torch.Tensor] = []
        patches_for_mu: List[torch.Tensor] = []
        diag_per_ref: Dict[int, Any] = {}

        for b in rp.ref_blocks:
            if b not in tokens_by_model[rp_ref_alias]:
                raise ValueError(f"[RegProj] reference block {b} not present for {rp_ref_alias}")
            toks = tokens_by_model[rp_ref_alias][b].float()
            out = find_register_tokens_cross_image(toks, cfg_tmp)
            diag_per_ref[int(b)] = out["stats"]

            patches = toks[:, 1:, :].float()  # (B,P,D)
            B, P, D = patches.shape
            patches_for_mu.append(patches.reshape(B * P, D))

            reg_mask_bp = out["reg_mask_bp"]  # (B,P)
            idx = reg_mask_bp.nonzero(as_tuple=False)
            if idx.numel() > 0:
                reg_vecs.append(patches[idx[:, 0], idx[:, 1], :])

            num_regs = reg_mask_bp.sum(dim=1).detach().cpu().tolist()
            print(
                f"[RegProj ref block {b:02d}] regs/image min={min(num_regs)} mean={sum(num_regs)/len(num_regs):.2f} max={max(num_regs)}"
                f" | comp(both_ok)={out['stats'].get('comp_both_ok')} bestCov={out['stats'].get('comp_best_coverage')}"
            )

        if not reg_vecs:
            raise RuntimeError("[RegProj] No register vectors found in ref blocks; lower sim_thr or change ref blocks.")

        X_mu = torch.cat(patches_for_mu, dim=0)
        mu_global_ref = X_mu.mean(dim=0) if rp.use_global_mean_centering else None

        X_reg = torch.cat(reg_vecs, dim=0)
        if rp.use_global_mean_centering and (mu_global_ref is not None):
            X_reg = X_reg - mu_global_ref.view(1, -1)

        U_reg_ref = _pca_basis(X_reg, k=int(rp.k_reg_basis))
        if U_reg_ref is None:
            raise RuntimeError("[RegProj] Failed to compute reference reg PCA basis (too few vectors?).")

        print(_c_cyan(f"[RegProj] U_reg_ref shape={tuple(U_reg_ref.shape)} | reg_vecs={X_reg.shape[0]} | ref_alias={rp_ref_alias}"))

        torch.save(
            {
                "U_reg_ref": U_reg_ref.detach().cpu(),
                "mu_global_ref": None if mu_global_ref is None else mu_global_ref.detach().cpu(),
                "ref_alias": rp_ref_alias,
                "ref_blocks": rp.ref_blocks,
            },
            os.path.join(rp.out_dir, f"U_reg_cross__{rp_ref_alias}__blocks_{'_'.join(map(str, rp.ref_blocks))}.pt"),
        )

        regproj_series = {}
        rows_csv: List[Dict[str, Any]] = []

        for ms in specs:
            print(f"\n=== [RegProj] metrics for {ms.alias} ===")
            series = {
                "proj_frac_mean": [],
                "proj_frac_max_mean": [],
                "proj_topk_share_mean": [],
                "proj_gini_mean": [],
                "ov_maxsv": [],
                "pa_deg_mean": [],
                "pa_deg_min": [],
                "pa_deg_max": [],
            }

            for b in all_blocks:
                toks = tokens_by_model[ms.alias][b].float()
                patches = toks[:, 1:, :].float()
                patches_cent = _apply_centering_with_mu_global(
                    patches_bpd=patches,
                    mu_global=mu_global_ref if rp.use_global_mean_centering else None,
                    use_image_mean_centering=rp.use_image_mean_centering,
                )

                m = projection_metrics_per_block(toks, U_reg_ref, mu_global_ref, rp)
                ang = reg_vs_patch_subspace_angles(patches_cent, U_reg_ref, k_patch=int(rp.k_patch_basis_for_angles))

                for k in ["proj_frac_mean", "proj_frac_max_mean", "proj_topk_share_mean", "proj_gini_mean"]:
                    series[k].append(float(m[k]))
                for k in ["ov_maxsv", "pa_deg_mean", "pa_deg_min", "pa_deg_max"]:
                    series[k].append(None if ang[k] is None else float(ang[k]))

                rows_csv.append({
                    "model_alias": ms.alias,
                    "model_id": ms.model_id,
                    "block": int(b),
                    "ref_alias": rp_ref_alias,
                    "ref_blocks": ",".join(map(str, rp.ref_blocks)),
                    **m,
                    **ang,
                })

                print(
                    f"[RegProj block {b:02d}] fracMean={m['proj_frac_mean']:.4f} topkShare={m['proj_topk_share_mean']:.4f} "
                    f"gini={m['proj_gini_mean']:.4f} ovMaxSV={ang['ov_maxsv']}"
                )

            regproj_series[ms.alias] = series

        # Save CSV + JSON
        csv_path = os.path.join(rp.out_dir, "regproj_rows.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            for r in rows_csv:
                w.writerow(r)
        print(f"[RegProj] saved CSV: {csv_path}")

        json_path = os.path.join(rp.out_dir, "regproj_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cfg": rp.__dict__,
                    "images": img_paths,
                    "ref_alias": rp_ref_alias,
                    "ref_blocks_diag": diag_per_ref,
                    "U_reg_shape": [int(U_reg_ref.shape[0]), int(U_reg_ref.shape[1])],
                    "series": regproj_series,
                    "blocks": all_blocks,
                    "rows": rows_csv,
                },
                f,
                indent=2,
            )
        print(f"[RegProj] saved JSON: {json_path}")

        # Plots: all models overlay (useful even without pairs)
        plot_overlay_models(
            all_blocks,
            regproj_series,
            metric="proj_frac_mean",
            title=f"All models :: proj_frac_mean onto {rp_ref_alias} reg basis",
            ylab="mean ||Proj||^2 / ||x||^2",
            out_path=os.path.join(rp.plots_dir, "ALL__proj_frac_mean.png"),
        )
        plot_overlay_models(
            all_blocks,
            regproj_series,
            metric="proj_gini_mean",
            title=f"All models :: gini(proj energy) onto {rp_ref_alias} reg basis",
            ylab="gini (0=spread,1=concentrated)",
            out_path=os.path.join(rp.plots_dir, "ALL__proj_gini_mean.png"),
        )
        plot_overlay_models(
            all_blocks,
            regproj_series,
            metric="ov_maxsv",
            title=f"All models :: ov_maxsv(patch PCA vs {rp_ref_alias} reg basis)",
            ylab="max sv (0..1)",
            out_path=os.path.join(rp.plots_dir, "ALL__overlap_maxsv.png"),
        )

        # Pair overlays (A# vs B#)
        for idx, (a_alias, b_alias) in pairs.items():
            if a_alias not in regproj_series or b_alias not in regproj_series:
                continue
            pair_series = {a_alias: regproj_series[a_alias], b_alias: regproj_series[b_alias]}
            plot_overlay_models(
                all_blocks,
                pair_series,
                metric="proj_frac_mean",
                title=f"Pair {idx}: proj_frac_mean onto {rp_ref_alias} reg basis",
                ylab="mean ||Proj||^2 / ||x||^2",
                out_path=os.path.join(rp.plots_dir, f"PAIR{idx}__proj_frac_mean.png"),
            )
            plot_overlay_models(
                all_blocks,
                pair_series,
                metric="proj_gini_mean",
                title=f"Pair {idx}: gini(proj energy) onto {rp_ref_alias} reg basis",
                ylab="gini",
                out_path=os.path.join(rp.plots_dir, f"PAIR{idx}__proj_gini_mean.png"),
            )
            plot_overlay_models(
                all_blocks,
                pair_series,
                metric="ov_maxsv",
                title=f"Pair {idx}: ov_maxsv(patch PCA vs {rp_ref_alias} reg basis)",
                ylab="max sv (0..1)",
                out_path=os.path.join(rp.plots_dir, f"PAIR{idx}__overlap_maxsv.png"),
            )

        print(f"[RegProj] plots saved to: {rp.plots_dir}")

    # Final heuristic summary
    heuristic_summary(
        specs=specs,
        blocks=all_blocks,
        recoverability=recoverability_out,
        cross_series=cross_series,
        regproj_series=regproj_series,
        pairs=pairs,
        have_pretrained=have_pretrained,
    )

    print("\n" + _c_cyan("[Outputs]"))
    print(_c_cyan(f"  out_root: {out_root}"))
    if recoverability_out is not None:
        print(_c_cyan(f"  recoverability: {os.path.join(out_root, 'recoverability')}"))
    if cross_series is not None:
        print(_c_cyan(f"  crossimage:     {os.path.join(out_root, 'crossimage')}"))
        if cross_json_paths:
            print(_c_cyan(f"  crossimage JSONs: {len(cross_json_paths)} files"))
    if regproj_series is not None:
        print(_c_cyan(f"  regproj:        {os.path.join(out_root, 'regproj')}"))


if __name__ == "__main__":
    main()