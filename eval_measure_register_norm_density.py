"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

Simple script for obtaining / plotting:

- Layerwise token-norm density maps for CLIP ViT on BLISS-e-V/SCAM
- Thresholded ("register-only") maps
- SynthSCAM-vs-NoSCAM delta maps
- Layerwise "globalness" (cosine to final) analysis

"""

from __future__ import annotations

import os
import csv
import math
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from datasets import load_dataset

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything



CONFIG: Dict[str, Any] = {
    "OUT_DIR": "out_register_norms_fixed",
    "SEED": 0,

    # Data
    "MAX_IMAGES_PER_VARIANT": None,  # e.g. 200, or None for full
    "BATCH_SIZE": 32,

    # Norm bins
    "NORM_MIN": 1.0,
    "NORM_MAX": 512.0,
    "NUM_BINS": 80,

    # "Register" threshold
    "REG_THRESHOLD": 70.0,

    # Delta: support gating
    "DELTA_MIN_COUNT_ALL": 25,
    "DELTA_MIN_COUNT_THR": 5,

    # Visualization
    "DENSITY_VMIN": 1e-4,
    "DENSITY_VMAX": 1e-1,
    "EDGE_SIGMA": 1.0,       # only smooths the *support boundary*
    "DELTA_CLIP_ABS": 2.0,

    # Globalness (ON by default)
    "DO_GLOBALNESS": True,
    "GLOBALNESS_N_IMAGES": 256,
    "GLOBALNESS_START_LAYER": 0,            # inclusive, 0-index
    "GLOBALNESS_LOCAL_POOL": (0.25, 0.75),  # IQR for local token sampling
    "GLOBALNESS_TOPK_FALLBACK": 4,          # if no regs at threshold, use top-k by norm

    # CONFIG
    "GLOBALNESS_LOW_CEIL": 50.0,
    "GLOBALNESS_TOPLOW_K": 4,


    # Models: OpenAI / local path .pt .safetensors / HuggingFace Hub
    "MODELS": [
        ("pretrained", "ViT-L/14"),
        ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
        ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
        ("regr-norm", "zer0int/CLIP-Regression-ViT-L-14"),
        ("regr-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
    ],
}





_EPS = 1e-12

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "nan"
        return f"{float(x):.{nd}f}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def _torch_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [..., D], b: [D] or [..., D]
    a = a / (a.norm(dim=-1, keepdim=True) + _EPS)
    b = b / (b.norm(dim=-1, keepdim=True) + _EPS)
    return (a * b).sum(dim=-1)


def _maybe_gaussian_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.float64)
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(mask.astype(np.float64), sigma=float(sigma), mode="nearest")
    except Exception:
        # cheap fallback
        k = 3
        out = mask.astype(np.float64)
        reps = int(max(1, round(sigma * 2)))
        for _ in range(reps):
            pad = k // 2
            p = np.pad(out, ((pad, pad), (pad, pad)), mode="edge")
            acc = np.zeros_like(out)
            for dy in range(k):
                for dx in range(k):
                    acc += p[dy:dy + out.shape[0], dx:dx + out.shape[1]]
            out = acc / float(k * k)
        return out

def load_scam_entries(max_images_per_variant: Optional[int], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
      buckets["NoSCAM"|"SCAM"|"SynthSCAM"] -> list of dicts {id, image, variant, ...}
    """
    ds = load_dataset("BLISS-e-V/SCAM", split="train")

    buckets: Dict[str, List[Dict[str, Any]]] = {v: [] for v in ["NoSCAM", "SCAM", "SynthSCAM"]}
    for e in ds:
        sid = str(e["id"])
        v = None
        for name in buckets.keys():
            if sid.startswith(name):
                v = name
                break
        if v is None:
            continue
        buckets[v].append(dict(
            id=sid,
            image=e["image"],
            variant=v,
            object_label=str(e.get("object_label", "")),
            attack_word=str(e.get("attack_word", "")),
        ))

    if max_images_per_variant is not None:
        rng = random.Random(seed)
        for v in list(buckets.keys()):
            if len(buckets[v]) > max_images_per_variant:
                rng.shuffle(buckets[v])
                buckets[v] = buckets[v][:max_images_per_variant]

    return buckets

def load_clip_model(name_or_path: str, device: str) -> Tuple[torch.nn.Module, Any]:
    model, preprocess_fn, _ = load_openai_clip_anything(
        clip, name_or_path, device=device, jit=False, strict=True
    )
    model = model.eval().float()
    return model, preprocess_fn


def get_visual_resblocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    return list(model.visual.transformer.resblocks)


def _resblock_out_to_btd(out: torch.Tensor) -> torch.Tensor:
    """
    Hooking resblocks => out is LND / [tokens, batch, width].
    Convert to BTD / [batch, tokens, width].
    """
    if not torch.is_tensor(out) or out.ndim != 3:
        raise RuntimeError(f"Expected 3D tensor from resblock hook; got {type(out)} shape={getattr(out, 'shape', None)}")
    return out.permute(1, 0, 2).contiguous()


@dataclass
class LayerStats:
    n: int = 0
    sum_: float = 0.0
    sumsq: float = 0.0
    min_: float = float("inf")
    max_: float = -float("inf")

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float64, copy=False)
        self.n += int(x.size)
        self.sum_ += float(x.sum())
        self.sumsq += float((x * x).sum())
        self.min_ = float(min(self.min_, float(x.min())))
        self.max_ = float(max(self.max_, float(x.max())))

    def mean(self) -> float:
        return self.sum_ / max(1, self.n)

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        mu = self.mean()
        var = max(0.0, (self.sumsq / self.n) - (mu * mu))
        return float(math.sqrt(var))


class NormDensityAccumulator:
    """
    Accumulates per-layer histograms and stats over patch-token norms.
    If threshold is set, also accumulates thresholded (reg-only) hist/stats.
    """
    def __init__(self, num_layers: int, bin_edges: np.ndarray, threshold: Optional[float]):
        self.num_layers = int(num_layers)
        self.bin_edges = bin_edges.astype(np.float64)
        self.num_bins = int(bin_edges.size - 1)
        self.threshold = threshold

        self.counts = np.zeros((self.num_bins, self.num_layers), dtype=np.int64)
        self.stats = [LayerStats() for _ in range(self.num_layers)]

        self.reg_counts = np.zeros((self.num_bins, self.num_layers), dtype=np.int64)
        self.reg_stats = [LayerStats() for _ in range(self.num_layers)]
        self.reg_n = np.zeros((self.num_layers,), dtype=np.int64)

    def _bincount(self, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.bin_edges, x, side="right") - 1
        idx = np.clip(idx, 0, self.num_bins - 1)
        return np.bincount(idx, minlength=self.num_bins).astype(np.int64)

    def update_layer(self, layer: int, norms_flat: np.ndarray) -> None:
        self.stats[layer].update(norms_flat)
        self.counts[:, layer] += self._bincount(norms_flat)

        if self.threshold is not None:
            regs = norms_flat[norms_flat >= float(self.threshold)]
            self.reg_n[layer] += int(regs.size)
            self.reg_stats[layer].update(regs)
            if regs.size > 0:
                self.reg_counts[:, layer] += self._bincount(regs)

    def proportions(self) -> np.ndarray:
        denom = self.counts.sum(axis=0, keepdims=True).astype(np.float64)
        denom = np.maximum(1.0, denom)
        return self.counts.astype(np.float64) / denom

    def reg_proportions(self) -> np.ndarray:
        denom = self.reg_counts.sum(axis=0, keepdims=True).astype(np.float64)
        denom = np.maximum(1.0, denom)
        return self.reg_counts.astype(np.float64) / denom


def accumulate_norms_one_variant(
    model: torch.nn.Module,
    preprocess_fn,
    device: str,
    entries: List[Dict[str, Any]],
    bin_edges: np.ndarray,
    threshold: float,
    batch_size: int,
) -> Tuple[NormDensityAccumulator, NormDensityAccumulator]:
    """
    Returns:
      acc_all: full-token accumulator (threshold=None)
      acc_thr: thresholded accumulator (threshold=REG_THRESHOLD)
    """
    resblocks = get_visual_resblocks(model)
    L = len(resblocks)

    acc_all = NormDensityAccumulator(num_layers=L, bin_edges=bin_edges, threshold=None)
    acc_thr = NormDensityAccumulator(num_layers=L, bin_edges=bin_edges, threshold=threshold)

    # Hook once, update both accumulators inside hook body.
    hooks: List[Any] = []
    expected_b = None  # set per batch

    def make_hook(layer: int):
        def _hook(_m, _inp, out):
            nonlocal expected_b
            with torch.no_grad():
                x = _resblock_out_to_btd(out)  # [B,T,D]
                if expected_b is None:
                    expected_b = int(x.shape[0])
                if int(x.shape[0]) != int(expected_b):
                    raise RuntimeError(f"Hook batch mismatch at layer {layer}: got B={int(x.shape[0])} expected {int(expected_b)}")

                patch = x[:, 1:, :]  # [B,P,D]
                norms = torch.linalg.vector_norm(patch, ord=2, dim=-1)  # [B,P]
                nf = norms.reshape(-1).detach().float().cpu().numpy()

                acc_all.update_layer(layer, nf)
                acc_thr.update_layer(layer, nf)
        return _hook

    for li, rb in enumerate(resblocks):
        hooks.append(rb.register_forward_hook(make_hook(li)))

    n_batches = (len(entries) + batch_size - 1) // batch_size
    t0 = time.time()

    with torch.inference_mode():
        for bi in range(n_batches):
            batch = entries[bi * batch_size:(bi + 1) * batch_size]
            expected_b = None

            imgs = [preprocess_fn(e["image"]) for e in batch]
            images = torch.stack(imgs, dim=0).to(device)

            _ = model.encode_image(images)  # trigger hooks

            if bi in (0, 50, 100) or (bi % 50 == 0 and bi > 0):
                print(f"  batch {bi:5d}/{n_batches}")

    for h in hooks:
        h.remove()

    dt = time.time() - t0
    # quick sanity prints (overall over all layers)
    total_n = sum(st.n for st in acc_all.stats)
    total_sum = sum(st.sum_ for st in acc_all.stats)
    total_min = min((st.min_ for st in acc_all.stats if st.n > 0), default=float("nan"))
    total_max = max((st.max_ for st in acc_all.stats if st.n > 0), default=float("nan"))
    total_mean = total_sum / max(1, total_n)

    reg_n = int(acc_thr.reg_n.sum())
    reg_sum = sum(st.sum_ for st in acc_thr.reg_stats)
    reg_min = min((st.min_ for st in acc_thr.reg_stats if st.n > 0), default=float("nan"))
    reg_max = max((st.max_ for st in acc_thr.reg_stats if st.n > 0), default=float("nan"))
    reg_mean = (reg_sum / max(1, reg_n)) if reg_n > 0 else float("nan")

    print(f"  [ALL TOKENS] n={total_n} min={total_min:.4f} max={total_max:.4f} mean={total_mean:.4f} | {dt:.1f}s")
    print(f"  [REG>= {threshold:g}] n={reg_n} min={reg_min:.4f} max={reg_max:.4f} mean={reg_mean:.4f}")

    return acc_all, acc_thr


def compute_logratio(
    p_a: np.ndarray,
    p_b: np.ndarray,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    min_count: int,
) -> np.ndarray:
    """
    log10 ratio with support gating:
      lr = log10((p_a + eps)/(p_b + eps))
      lr[support < min_count] = nan
    """
    lr = np.log10((p_a + _EPS) / (p_b + _EPS)).astype(np.float64, copy=False)
    support = (counts_a.astype(np.int64) + counts_b.astype(np.int64))
    if int(min_count) > 0:
        lr = lr.astype(np.float64, copy=True)
        lr[support < int(min_count)] = np.nan
    return lr


def summarize_logratio(lr: np.ndarray, bin_edges: np.ndarray, top_k: int = 12) -> Dict[str, Any]:
    num_bins, num_layers = lr.shape
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    finite = np.isfinite(lr)
    if not np.any(finite):
        return dict(
            per_layer_mean=[],
            per_layer_mean_abs=[],
            per_layer_max_abs=[],
            global_max_abs=dict(value=np.nan, abs_value=np.nan, layer=None, norm_center=np.nan),
            top_outliers=[],
        )

    per_layer_mean = []
    per_layer_mean_abs = []
    per_layer_max_abs = []
    for li in range(num_layers):
        col = lr[:, li]
        per_layer_mean.append(float(np.nanmean(col)))
        per_layer_mean_abs.append(float(np.nanmean(np.abs(col))))
        per_layer_max_abs.append(float(np.nanmax(np.abs(col))))

    abs_lr = np.abs(lr).astype(np.float64, copy=True)
    abs_lr[~finite] = -np.inf

    flat_abs = abs_lr.reshape(-1)
    if not np.isfinite(flat_abs).any():
        g = dict(value=np.nan, abs_value=np.nan, layer=None, norm_center=np.nan)
    else:
        idx = int(np.argmax(flat_abs))
        bi = idx // num_layers
        li = idx % num_layers
        g = dict(
            value=float(lr[bi, li]),
            abs_value=float(abs_lr[bi, li]),
            layer=int(li),  # 0-index
            norm_center=float(centers[bi]),
        )

    idxs = np.argsort(-flat_abs)[:top_k]
    top = []
    flat_lr = lr.reshape(-1)
    for idx in idxs:
        if not np.isfinite(flat_lr[idx]):
            continue
        bi = int(idx // num_layers)
        li = int(idx % num_layers)
        v = float(lr[bi, li])
        top.append(dict(layer=li, norm_center=float(centers[bi]), value=v, abs_value=float(abs(v))))

    return dict(
        per_layer_mean=per_layer_mean,
        per_layer_mean_abs=per_layer_mean_abs,
        per_layer_max_abs=per_layer_max_abs,
        global_max_abs=g,
        top_outliers=top,
    )


def plot_norm_density(
    path: str,
    bin_edges: np.ndarray,
    props: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    edge_sigma: float,
) -> None:
    """
    Correct geometry for log-spaced bin edges + edge-only alpha smoothing.

    Key idea:
      - Use pcolormesh(x_edges, bin_edges, data) so non-uniform bins are correct.
      - Let matplotlib compute colors normally.
      - Force a draw so QuadMesh facecolors are expanded.
      - Then overwrite only the alpha channel with the blurred support mask.
    """
    num_bins, num_layers = props.shape
    data = props.astype(np.float64, copy=True)
    valid = (data > 0)

    filler = float(vmin) * 0.5
    data[~valid] = filler  # for LogNorm; hidden by alpha

    alpha = _maybe_gaussian_blur(valid.astype(np.float64), sigma=float(edge_sigma))
    alpha = np.clip(alpha, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    ax.set_facecolor("black")
    cmap = plt.get_cmap("magma")
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    x_edges = np.arange(-0.5, num_layers + 0.5, 1.0, dtype=np.float64)  # len L+1
    y_edges = bin_edges.astype(np.float64)                               # len bins+1

    mesh = ax.pcolormesh(
        x_edges, y_edges, data,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel("layer (0-index)")
    ax.set_ylabel("norm")
    ax.set_title(title)
    ax.set_yscale("log")

    # force facecolors materialization, then inject per-cell alpha
    fig.canvas.draw()  # for QuadMesh facecolors to exist per-cell
    print("[DEBUG] facecolors:", mesh.get_facecolors().shape, "data.size:", data.size)

    fc = mesh.get_facecolors()  # shape should be (num_bins*num_layers, 4)

    # if something changes in matplotlib, fail loudly rather than silently
    if fc.shape[0] != data.size:
        raise RuntimeError(
            f"QuadMesh facecolors size mismatch: facecolors={fc.shape[0]} vs data.size={data.size}. "
            f"(bins={num_bins}, layers={num_layers})"
        )

    fc[:, 3] = fc[:, 3] * alpha.reshape(-1)  # apply blurred support mask only
    mesh.set_facecolors(fc)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("proportion")

    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_delta_logratio(
    path: str,
    bin_edges: np.ndarray,
    lr: np.ndarray,
    title: str,
    clip_abs: float,
) -> None:
    """
    Uses pcolormesh; x-axis uses 0-indexed layer edges [0..L].
    """
    num_bins, num_layers = lr.shape
    x_edges = np.arange(0, num_layers + 1, dtype=np.float64)
    y_edges = bin_edges

    data = np.clip(lr, -clip_abs, clip_abs)
    cmap = plt.get_cmap("gist_ncar").copy()

    fig, ax = plt.subplots(figsize=(6.7, 4.3))

    # black plot area/background
    ax.set_facecolor("black")

    im = ax.pcolormesh(
        x_edges, y_edges, data,
        norm=Normalize(vmin=-clip_abs, vmax=clip_abs),
        shading="auto",
        cmap=cmap,
    )

    ax.set_yscale("log")
    ax.set_xlabel("layer (0-index)")
    ax.set_ylabel("norm")
    ax.set_title(title)

    # make text/ticks visible on black
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 ratio")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_globalness_lines(path: str, model_alias: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    layers = np.array([r["layer"] for r in rows], dtype=np.int64)
    reg = np.array([r["reg_mean"] for r in rows], dtype=np.float64)
    loc = np.array([r["loc_mean"] for r in rows], dtype=np.float64)

    toplow = None
    if "toplow_mean" in rows[0]:
        toplow = np.array([r["toplow_mean"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(7.2, 3.9))
    plt.plot(layers, reg, label="reg_norm>70_mean", color="red")
    plt.plot(layers, loc, label="local_patch_IQR", color="blue")
    if toplow is not None:
        plt.plot(layers, toplow, label="top4_norm<50_mean", color="green")

    plt.xlabel("layer (0-index)")
    plt.ylabel("cosine vs final image embedding")
    plt.title(f"{model_alias} | globalness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def save_layer_stats_csv(path: str, acc_all: NormDensityAccumulator) -> None:
    header = ["layer", "n_tokens", "min", "max", "mean", "std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for li in range(acc_all.num_layers):
            st = acc_all.stats[li]
            w.writerow([
                li,
                st.n,
                st.min_ if st.n > 0 else "",
                st.max_ if st.n > 0 else "",
                st.mean() if st.n > 0 else "",
                st.std() if st.n > 0 else "",
            ])


def save_layer_stats_thr_csv(path: str, acc_thr: NormDensityAccumulator) -> None:
    header = ["layer", "reg_n_tokens", "reg_min", "reg_max", "reg_mean", "reg_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for li in range(acc_thr.num_layers):
            rst = acc_thr.reg_stats[li]
            reg_n = int(acc_thr.reg_n[li])
            w.writerow([
                li,
                reg_n,
                rst.min_ if rst.n > 0 else "",
                rst.max_ if rst.n > 0 else "",
                rst.mean() if rst.n > 0 else "",
                rst.std() if rst.n > 0 else "",
            ])


def save_hist_long_csv(path: str, bin_edges: np.ndarray, props: np.ndarray) -> None:
    num_bins, num_layers = props.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "bin_lo", "bin_hi", "proportion"])
        for li in range(num_layers):
            for bi in range(num_bins):
                w.writerow([li, float(bin_edges[bi]), float(bin_edges[bi + 1]), float(props[bi, li])])


def write_key_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_key_metrics(acc_all: NormDensityAccumulator, acc_thr: NormDensityAccumulator) -> Dict[str, Any]:
    L = acc_all.num_layers
    late = [max(0, L - 3), max(0, L - 2), max(0, L - 1)]

    total_n = sum(st.n for st in acc_all.stats)
    total_sum = sum(st.sum_ for st in acc_all.stats)
    overall_mean = total_sum / max(1, total_n)

    late_n = sum(acc_all.stats[i].n for i in late)
    late_sum = sum(acc_all.stats[i].sum_ for i in late)
    late_mean = late_sum / max(1, late_n)

    reg_total = int(acc_thr.reg_n.sum())
    reg_sum = sum(st.sum_ for st in acc_thr.reg_stats)
    reg_mean = (reg_sum / max(1, reg_total)) if reg_total > 0 else float("nan")

    late_reg = int(acc_thr.reg_n[late].sum())
    late_reg_frac = float(late_reg / max(1, late_n))

    return dict(
        overall_mean_norm=float(overall_mean),
        late3_mean_norm=float(late_mean),
        total_tokens=int(total_n),
        reg_total_tokens=int(reg_total),
        reg_fraction=float(reg_total / max(1, total_n)),
        reg_mean_norm=float(reg_mean),
        late3_reg_tokens=int(late_reg),
        late3_reg_fraction=float(late_reg_frac),
    )


def build_summary_report(
    model_alias: str,
    model_path: str,
    num_layers: int,
    threshold: float,
    variant_sizes: Dict[str, int],
    accs: Dict[str, Tuple[NormDensityAccumulator, NormDensityAccumulator]],
    bin_edges: np.ndarray,
    lr_all: np.ndarray,
    lr_thr: np.ndarray,
    globalness_rows: Optional[List[Dict[str, Any]]],
) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"MODEL: {model_alias} | {model_path}")
    lines.append(f"visual_layers={num_layers} (layers 0..{num_layers-1}) | reg_threshold={threshold:g}")
    lines.append("DATA:")
    for k in ["all", "NoSCAM", "SynthSCAM"]:
        lines.append(f"  - {k:9s}: images={variant_sizes[k]}")
    lines.append("")

    lines.append("NORM SUMMARY (per variant):")
    for v in ["all", "NoSCAM", "SynthSCAM"]:
        acc_all, acc_thr = accs[v]
        total_n = sum(st.n for st in acc_all.stats)
        total_sum = sum(st.sum_ for st in acc_all.stats)
        overall_mean = total_sum / max(1, total_n)

        reg_total = int(acc_thr.reg_n.sum())
        reg_sum = sum(st.sum_ for st in acc_thr.reg_stats)
        reg_mean = (reg_sum / max(1, reg_total)) if reg_total > 0 else float("nan")

        lines.append(f"  [{v}]")
        lines.append(f"    all_tokens: n={total_n} mean_norm={overall_mean:.4f}")
        lines.append(f"    reg_tokens: n={reg_total} mean_reg_norm={_fmt(reg_mean)} reg_fraction={reg_total/max(1,total_n):.4f}")

        # Top layers by reg fraction
        reg_frac = np.array([
            float(acc_thr.reg_n[li] / max(1, acc_all.stats[li].n))
            for li in range(num_layers)
        ], dtype=np.float64)
        mean_norm = np.array([acc_all.stats[li].mean() for li in range(num_layers)], dtype=np.float64)
        reg_mean_by_layer = np.array([
            acc_thr.reg_stats[li].mean() if acc_thr.reg_stats[li].n > 0 else np.nan
            for li in range(num_layers)
        ], dtype=np.float64)

        idxs = np.argsort(-reg_frac)[:8]
        lines.append("    top_layers_by_reg_fraction (layer: reg_frac | mean_norm | mean_reg_norm):")
        for li in idxs:
            lines.append(f"      L{li:02d}: {reg_frac[li]:.4f} | {mean_norm[li]:.4f} | {_fmt(reg_mean_by_layer[li])}")
        lines.append("")

    def _delta_block(name: str, lr: np.ndarray, min_count: int) -> None:
        s = summarize_logratio(lr, bin_edges, top_k=12)
        g = s["global_max_abs"]

        lines.append(f"DELTA SUMMARY ({name}): log10 ratio per (norm_bin, layer) | min_count={min_count}")
        layer_str = f"L{g['layer']}" if g.get("layer", None) is not None else "?"
        lines.append(
            f"  global_max_abs: layer={layer_str} norm≈{_fmt(g['norm_center'],2)} "
            f"value={_fmt(g['value'])} (abs={_fmt(g['abs_value'])})"
        )

        per_layer_max_abs = np.array(s["per_layer_max_abs"], dtype=np.float64)
        lines.append(
            f"  per_layer_max_abs: mean={_fmt(np.nanmean(per_layer_max_abs))} std={_fmt(np.nanstd(per_layer_max_abs))} "
            f"min={_fmt(np.nanmin(per_layer_max_abs))} max={_fmt(np.nanmax(per_layer_max_abs))}"
        )

        lines.append("  top_outlier_bins (layer, norm≈center, value):")
        for o in s["top_outliers"]:
            lines.append(f"    L{o['layer']:02d}, norm≈{o['norm_center']:.2f}: {o['value']:+.4f} (abs={o['abs_value']:.4f})")

        idxs = np.argsort(-per_layer_max_abs)[:8]
        per_layer_mean_abs = np.array(s["per_layer_mean_abs"], dtype=np.float64)
        per_layer_mean = np.array(s["per_layer_mean"], dtype=np.float64)
        lines.append("  top_layers_by_max_abs_delta (layer: max_abs | mean_abs | mean_signed):")
        for li in idxs:
            lines.append(f"    L{li:02d}: {per_layer_max_abs[li]:.4f} | {per_layer_mean_abs[li]:.4f} | {per_layer_mean[li]:+.4f}")
        lines.append("")

    _delta_block("ALL TOKENS (SynthSCAM vs NoSCAM)", lr_all, CONFIG["DELTA_MIN_COUNT_ALL"])
    _delta_block(f"REG-ONLY thr={threshold:g} (SynthSCAM vs NoSCAM)", lr_thr, CONFIG["DELTA_MIN_COUNT_THR"])

    if globalness_rows is not None and len(globalness_rows) > 0:
        reg = np.array([r["reg_mean"] for r in globalness_rows], dtype=np.float64)
        loc = np.array([r["loc_mean"] for r in globalness_rows], dtype=np.float64)
        gap = reg - loc
        layers = np.array([r["layer"] for r in globalness_rows], dtype=np.int64)

        lines.append("GLOBALNESS SUMMARY (cosine alignment vs final image embedding):")
        lines.append(f"  layers: {int(layers.min())}..{int(layers.max())} (0-index)")
        lines.append(
            f"  overall: reg_mean={reg.mean():.4f}±{reg.std():.4f} "
            f"loc_mean={loc.mean():.4f}±{loc.std():.4f} "
            f"gap={gap.mean():+.4f}±{gap.std():.4f}"
        )

        best_reg_i = int(np.argmax(reg))
        best_gap_i = int(np.argmax(gap))
        lines.append(
            f"  best_reg_layer: L{int(layers[best_reg_i])} (reg_mean={reg[best_reg_i]:.4f}) | "
            f"best_gap_layer: L{int(layers[best_gap_i])} (gap_mean={gap[best_gap_i]:+.4f})"
        )

        top = np.argsort(-gap)[:8]
        lines.append("  top_layers_by_gap (layer: reg_mean | loc_mean | gap_mean):")
        for idx in top:
            lines.append(f"    L{int(layers[idx]):02d}: {reg[idx]:.4f} | {loc[idx]:.4f} | {gap[idx]:+.4f}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def globalness_probe_layerwise(
    model: torch.nn.Module,
    preprocess_fn,
    device: str,
    entries: List[Dict[str, Any]],
    out_csv_path: str,
    out_txt_path: str,
    threshold: float,
    start_layer: int,
    n_images: int,
    batch_size: int,
    seed: int,
    local_pool_q: Tuple[float, float],
    topk_fallback: int,
) -> List[Dict[str, Any]]:
    """
    Hook-time compute: no per-layer activation caching.

    For each layer in [start_layer..L-1]:
      - reg_mean (RED): mean cosine of tokens with (norm>=threshold). If none => 0.0
      - loc_mean (BLUE): ALWAYS k=4 samples from IQR (q25..q75), register-independent
      - toplow_mean (GREEN): top-4 by norm among tokens with norm < 50. If none => 0.0
      - project tokens via visual.proj to embedding space
      - cosine vs final image embedding (explicitly normalized)
    """
    resblocks = get_visual_resblocks(model)
    L = len(resblocks)
    if not (0 <= int(start_layer) < L):
        raise ValueError(f"GLOBALNESS_START_LAYER must be in [0,{L-1}], got {start_layer}")

    proj = model.visual.proj
    if proj is None:
        raise RuntimeError("model.visual.proj is None; cannot project token states to embedding space.")

    def _proj_tokens(x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(proj):
            return x @ proj
        if isinstance(proj, torch.nn.Parameter):
            return x @ proj
        if isinstance(proj, torch.nn.Module):
            return proj(x)
        raise RuntimeError(f"Unsupported visual.proj type: {type(proj)}")

    rng = random.Random(seed)
    chosen = entries[:]
    rng.shuffle(chosen)
    chosen = chosen[:min(len(chosen), int(n_images))]

    layers = list(range(int(start_layer), L))

    @dataclass
    class _W:
        n: int = 0
        mean: float = 0.0
        m2: float = 0.0
        def add(self, v: float) -> None:
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            d2 = v - self.mean
            self.m2 += d * d2
        def std(self) -> float:
            if self.n <= 1:
                return 0.0
            return float(math.sqrt(max(0.0, self.m2 / (self.n - 1))))

    reg_w = {li: _W() for li in layers}
    loc_w = {li: _W() for li in layers}
    gap_w = {li: _W() for li in layers}
    toplow_w = {li: _W() for li in layers}

    # Hook arming + batch final embedding
    GLOBALNESS_ACTIVE: bool = False
    CURRENT_E_FINAL: Optional[torch.Tensor] = None  # [B,embed_dim] unit norm
    CURRENT_GEN = torch.Generator(device=device)
    CURRENT_GEN.manual_seed(int(seed))

    q_lo, q_hi = float(local_pool_q[0]), float(local_pool_q[1])
    thr = float(threshold)

    # fixed baseline size
    k_loc_fixed = 4
    low_ceil = 50.0
    toplow_k = 4

    hooks: List[Any] = []

    def make_hook(li: int):
        def _hook(_m, _inp, out):
            nonlocal CURRENT_E_FINAL, GLOBALNESS_ACTIVE
            if not GLOBALNESS_ACTIVE:
                return
            if CURRENT_E_FINAL is None:
                raise RuntimeError("Globalness: CURRENT_E_FINAL is None inside hook (encode_image not set?)")

            with torch.no_grad():
                x = _resblock_out_to_btd(out)  # [B,T,width]
                B = int(x.shape[0])
                if int(CURRENT_E_FINAL.shape[0]) != B:
                    raise RuntimeError(
                        f"Globalness: layer {li} B mismatch: got {B}, expected {int(CURRENT_E_FINAL.shape[0])}"
                    )

                patch = x[:, 1:, :]  # [B,P,width]
                norms = torch.linalg.vector_norm(patch, ord=2, dim=-1)  # [B,P]

                for b in range(B):
                    nb = norms[b]              # [P] (torch)
                    xb = patch[b]              # [P,width]
                    eb = CURRENT_E_FINAL[b]    # [embed_dim], unit norm

                    P = int(nb.numel())
                    if P == 0:
                        # pathological; keep everything at 0
                        reg_w[li].add(0.0)
                        loc_w[li].add(0.0)
                        gap_w[li].add(0.0)
                        toplow_w[li].add(0.0)
                        continue

                    # BLUE: IQR baseline, ALWAYS k=4, register-independent
                    lo = torch.quantile(nb, q_lo)
                    hi = torch.quantile(nb, q_hi)
                    local_pool = ((nb >= lo) & (nb <= hi)).nonzero(as_tuple=False).flatten()
                    if int(local_pool.numel()) < 1:
                        local_pool = torch.arange(P, device=nb.device)

                    kk_loc = min(int(k_loc_fixed), int(local_pool.numel()))
                    if kk_loc <= 0:
                        lmean = 0.0
                    else:
                        weights = torch.ones(int(local_pool.numel()), device=nb.device, dtype=torch.float32)
                        sample_pos = torch.multinomial(
                            weights, num_samples=kk_loc, replacement=False, generator=CURRENT_GEN
                        )
                        local_idx = local_pool[sample_pos]
                        loc_feats = xb.index_select(0, local_idx)
                        loc_proj = _proj_tokens(loc_feats)
                        loc_cos = _torch_cosine(loc_proj, eb)
                        lmean = float(loc_cos.mean().detach().cpu().item())

                    loc_w[li].add(lmean)

                    # RED: strict registers only (no fallback)
                    reg_idx = (nb >= thr).nonzero(as_tuple=False).flatten()
                    if int(reg_idx.numel()) == 0:
                        rmean = 0.0
                    else:
                        reg_feats = xb.index_select(0, reg_idx)
                        reg_proj = _proj_tokens(reg_feats)
                        reg_cos = _torch_cosine(reg_proj, eb)
                        rmean = float(reg_cos.mean().detach().cpu().item())

                    reg_w[li].add(rmean)
                    gap_w[li].add(rmean - lmean)

                    # GREEN: top-4 among norms < 50
                    low_idx = (nb < float(low_ceil)).nonzero(as_tuple=False).flatten()
                    if int(low_idx.numel()) == 0:
                        toplow_w[li].add(0.0)
                    else:
                        low_norms = nb.index_select(0, low_idx)
                        kk = min(int(toplow_k), int(low_norms.numel()))
                        sel_pos = torch.topk(low_norms, k=kk, largest=True).indices
                        sel_idx = low_idx.index_select(0, sel_pos)

                        low_feats = xb.index_select(0, sel_idx)
                        low_proj = _proj_tokens(low_feats)
                        low_cos = _torch_cosine(low_proj, eb)
                        toplow_w[li].add(float(low_cos.mean().detach().cpu().item()))
        return _hook

    for li in layers:
        hooks.append(resblocks[li].register_forward_hook(make_hook(li)))

    with torch.inference_mode():
        n_batches = (len(chosen) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            batch = chosen[bi * batch_size:(bi + 1) * batch_size]

            imgs = [preprocess_fn(e["image"]) for e in batch]
            images = torch.stack(imgs, dim=0).to(device)

            # PASS A: compute final embedding (hooks inactive)
            GLOBALNESS_ACTIVE = False
            e_final = model.encode_image(images)  # UNNORMALIZED
            e_final = e_final / (e_final.norm(dim=-1, keepdim=True) + _EPS)

            # PASS B: hooks active; use e_final
            CURRENT_E_FINAL = e_final
            GLOBALNESS_ACTIVE = True
            _ = model.encode_image(images)
            GLOBALNESS_ACTIVE = False
            CURRENT_E_FINAL = None

            if bi in (0, 10, 20) or (bi % 50 == 0 and bi > 0):
                print(f"  globalness batch {bi:4d}/{n_batches}")

    for h in hooks:
        h.remove()

    rows: List[Dict[str, Any]] = []
    for li in layers:
        rows.append(dict(
            layer=int(li),
            n=int(reg_w[li].n),
            reg_mean=float(reg_w[li].mean),
            reg_std=float(reg_w[li].std()),
            loc_mean=float(loc_w[li].mean),
            loc_std=float(loc_w[li].std()),
            gap_mean=float(gap_w[li].mean),
            gap_std=float(gap_w[li].std()),
            toplow_mean=float(toplow_w[li].mean),
            toplow_std=float(toplow_w[li].std()),
        ))

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # add full layerwise printout (ascending layer order)
    reg = np.array([r["reg_mean"] for r in rows], dtype=np.float64)
    loc = np.array([r["loc_mean"] for r in rows], dtype=np.float64)
    gap = reg - loc
    layers_np = np.array([r["layer"] for r in rows], dtype=np.int64)

    toplow = None
    if len(rows) > 0 and ("toplow_mean" in rows[0]):
        toplow = np.array([r["toplow_mean"] for r in rows], dtype=np.float64)

    best_reg_i = int(np.argmax(reg))
    best_gap_i = int(np.argmax(gap))
    top = np.argsort(-gap)[:8]

    txt_lines: List[str] = []
    txt_lines.append("GLOBALNESS SUMMARY (cosine alignment vs final image embedding):")
    txt_lines.append(f"  layers: {int(layers_np.min())}..{int(layers_np.max())} (0-index)")
    txt_lines.append(
        f"  best_reg_layer: L{int(layers_np[best_reg_i])} (reg_mean={reg[best_reg_i]:.4f}) | "
        f"best_gap_layer: L{int(layers_np[best_gap_i])} (gap_mean={gap[best_gap_i]:+.4f})"
    )
    txt_lines.append("  top_layers_by_gap (layer: reg_mean | loc_mean | gap_mean):")
    for idx in top:
        txt_lines.append(
            f"    L{int(layers_np[idx]):02d}: {reg[idx]:.4f} | {loc[idx]:.4f} | {gap[idx]:+.4f}"
        )

    # Full table (all layers in rows, ascending by layer id)
    txt_lines.append("")
    txt_lines.append("LAYERWISE (all layers, 0-index):")
    hdr = "  layer | reg_mean | loc_mean | gap_mean"
    if toplow is not None:
        hdr += " | toplow_mean"
    txt_lines.append(hdr)
    txt_lines.append("  " + "-" * (len(hdr) - 2))

    order = np.argsort(layers_np)
    for i in order:
        li = int(layers_np[i])
        line = f"  L{li:02d}  | {reg[i]:.4f}  | {loc[i]:.4f}  | {gap[i]:+.4f}"
        if toplow is not None:
            line += f"  | {toplow[i]:.4f}"
        txt_lines.append(line)

    txt = "\n".join(txt_lines)
    print(txt)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    return rows


def main() -> None:
    cfg = CONFIG
    random.seed(int(cfg["SEED"]))
    np.random.seed(int(cfg["SEED"]))
    torch.manual_seed(int(cfg["SEED"]))

    out_dir = str(cfg["OUT_DIR"])
    _ensure_dir(out_dir)

    device = _device()
    print(f"[INFO] device={device}")

    buckets = load_scam_entries(cfg["MAX_IMAGES_PER_VARIANT"], int(cfg["SEED"]))
    all_entries = buckets["NoSCAM"] + buckets["SCAM"] + buckets["SynthSCAM"]

    runs: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("all", all_entries),
        ("NoSCAM", buckets["NoSCAM"]),
        ("SynthSCAM", buckets["SynthSCAM"]),
    ]
    variant_sizes = {k: len(v) for (k, v) in runs}

    # bin edges (log-spaced in norm value)
    bin_edges = np.logspace(
        np.log10(float(cfg["NORM_MIN"])),
        np.log10(float(cfg["NORM_MAX"])),
        int(cfg["NUM_BINS"]) + 1,
        base=10.0,
        dtype=np.float64,
    )

    for model_alias, model_path in cfg["MODELS"]:
        print("\n" + "=" * 80)
        print(f"=== Model: {model_alias} ({model_path}) ===")

        model, preprocess_fn = load_clip_model(model_path, device=device)
        L = len(get_visual_resblocks(model))
        print(f"Visual layers: {L} (layers 0..{L-1})")

        model_out = os.path.join(out_dir, model_alias)
        out_norms = os.path.join(model_out, "norms")
        out_csv = os.path.join(model_out, "csv")
        _ensure_dir(out_norms)
        _ensure_dir(out_csv)

        accs: Dict[str, Tuple[NormDensityAccumulator, NormDensityAccumulator]] = {}
        key_rows: List[Dict[str, Any]] = []

        thr = float(cfg["REG_THRESHOLD"])

        for variant_name, entries in runs:
            print(f"--- Variant: {variant_name} | images={len(entries)} ---")
            acc_all, acc_thr = accumulate_norms_one_variant(
                model=model,
                preprocess_fn=preprocess_fn,
                device=device,
                entries=entries,
                bin_edges=bin_edges,
                threshold=thr,
                batch_size=int(cfg["BATCH_SIZE"]),
            )
            accs[variant_name] = (acc_all, acc_thr)

            # CSV stats + long hist
            save_layer_stats_csv(
                os.path.join(out_csv, f"{variant_name}_layer_stats.csv"),
                acc_all,
            )
            save_layer_stats_thr_csv(
                os.path.join(out_csv, f"{variant_name}_layer_stats_thr70.csv"),
                acc_thr,
            )
            save_hist_long_csv(
                os.path.join(out_csv, f"{variant_name}_hist_long.csv"),
                bin_edges,
                acc_all.proportions(),
            )
            save_hist_long_csv(
                os.path.join(out_csv, f"{variant_name}_hist_long_thr70.csv"),
                bin_edges,
                acc_thr.reg_proportions(),
            )
            plot_norm_density(
                os.path.join(out_norms, f"{variant_name}_norm_density.png"),
                bin_edges,
                acc_all.proportions(),
                title=f"{model_alias} | {variant_name} | norm density",
                vmin=float(cfg["DENSITY_VMIN"]),
                vmax=float(cfg["DENSITY_VMAX"]),
                edge_sigma=float(cfg["EDGE_SIGMA"]),
            )
            plot_norm_density(
                os.path.join(out_norms, f"{variant_name}_norm_density_thr{int(thr)}.png"),
                bin_edges,
                acc_thr.reg_proportions(),
                title=f"{model_alias} | {variant_name} | reg-only density (thr={thr:g})",
                vmin=float(cfg["DENSITY_VMIN"]),
                vmax=float(cfg["DENSITY_VMAX"]),
                edge_sigma=float(cfg["EDGE_SIGMA"]),
            )

            km = summarize_key_metrics(acc_all, acc_thr)
            km.update(dict(model=model_alias, variant=variant_name, threshold=thr))
            key_rows.append(km)

        # Delta (Synth vs NoSCAM) — full tokens
        p_s = accs["SynthSCAM"][0].proportions()
        p_n = accs["NoSCAM"][0].proportions()
        c_s = accs["SynthSCAM"][0].counts
        c_n = accs["NoSCAM"][0].counts

        lr_all = compute_logratio(
            p_a=p_s, p_b=p_n,
            counts_a=c_s, counts_b=c_n,
            min_count=int(cfg["DELTA_MIN_COUNT_ALL"]),
        )
        plot_delta_logratio(
            os.path.join(out_norms, "delta_synth_minus_noscam_logratio.png"),
            bin_edges, lr_all,
            title=f"{model_alias} | delta log10((Synth+eps)/(NoSCAM+eps)) [min_count={cfg['DELTA_MIN_COUNT_ALL']}]",
            clip_abs=float(cfg["DELTA_CLIP_ABS"]),
        )

        # Delta — reg-only
        p_s_thr = accs["SynthSCAM"][1].reg_proportions()
        p_n_thr = accs["NoSCAM"][1].reg_proportions()
        c_s_thr = accs["SynthSCAM"][1].reg_counts
        c_n_thr = accs["NoSCAM"][1].reg_counts

        lr_thr = compute_logratio(
            p_a=p_s_thr, p_b=p_n_thr,
            counts_a=c_s_thr, counts_b=c_n_thr,
            min_count=int(cfg["DELTA_MIN_COUNT_THR"]),
        )
        plot_delta_logratio(
            os.path.join(out_norms, f"delta_synth_minus_noscam_logratio_thr{int(thr)}.png"),
            bin_edges, lr_thr,
            title=f"{model_alias} | delta reg-only log10 ratio (thr={thr:g}) [min_count={cfg['DELTA_MIN_COUNT_THR']}]",
            clip_abs=float(cfg["DELTA_CLIP_ABS"]),
        )

        # Globalness
        globalness_rows: Optional[List[Dict[str, Any]]] = None
        if bool(cfg["DO_GLOBALNESS"]):
            gl_csv = os.path.join(out_csv, "globalness_layerwise_summary.csv")
            gl_txt = os.path.join(out_csv, "globalness_layerwise_summary.txt")
            gl_png = os.path.join(out_norms, "globalness_lines.png")

            print("--- Globalness (layerwise) ---")
            globalness_rows = globalness_probe_layerwise(
                model=model,
                preprocess_fn=preprocess_fn,
                device=device,
                entries=all_entries,
                out_csv_path=gl_csv,
                out_txt_path=gl_txt,
                threshold=thr,
                start_layer=int(cfg["GLOBALNESS_START_LAYER"]),
                n_images=int(cfg["GLOBALNESS_N_IMAGES"]),
                batch_size=int(cfg["BATCH_SIZE"]),
                seed=int(cfg["SEED"]),
                local_pool_q=tuple(cfg["GLOBALNESS_LOCAL_POOL"]),
                topk_fallback=int(cfg["GLOBALNESS_TOPK_FALLBACK"]),
            )
            plot_globalness_lines(gl_png, model_alias=model_alias, rows=globalness_rows)

        # Summary report
        report = build_summary_report(
            model_alias=model_alias,
            model_path=model_path,
            num_layers=L,
            threshold=thr,
            variant_sizes=variant_sizes,
            accs=accs,
            bin_edges=bin_edges,
            lr_all=lr_all,
            lr_thr=lr_thr,
            globalness_rows=globalness_rows,
        )
        print(report)
        with open(os.path.join(out_csv, "summary_report.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")

        # Key summary CSV
        write_key_summary_csv(os.path.join(out_csv, "key_summary.csv"), key_rows)

        # cleanup
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()