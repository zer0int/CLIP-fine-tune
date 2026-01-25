"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

 CLIP "what logit_scale works best for OOD gating?" calibrator.

 Download:
 https://www.image-net.org/download.php -> ILSVRC2012
 https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
 https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar

 Goal:
   Choose logit_scale to optimize OOD detection on ImageNet:
     ID  = ImageNet-val (WNID folders)
     OOD = ImageNet-A / ImageNet-R (WNID folders)

 Objective (choose one):
   - minimize FPR@95TPR   (default; practical for gating)
   - maximize AUROC

 Score type (choose one):
   - energy      (default; scale-sensitive, usually solid)
   - msp         (max softmax prob; scale-sensitive)
   - entropy     (-entropy; scale-sensitive)
   - prob_margin (p_top1 - p_top2; scale-sensitive)
   - maxlogit    (WARNING: AUROC/FPR@95TPR invariant to positive scaling => pointless to calibrate)

 Saves:
   Always updates model.logit_scale and torch.save(model, out_file)
   out_file is resolved via --out_path (dir or explicit .pt file).
   Auto-name: {model_name}_logit-scale_{factor}.pt
   factor(T) = exp(old_logit_scale) / exp(new_logit_scale)

 TEST the CALIBRATED model with:
 ----> eval-measure-imagenet-ood-logit-scale.py

"""

from __future__ import annotations

import os
import re
import math
import json
import random
import argparse
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    ap = argparse.ArgumentParser("CLIP logit_scale calibrator (OOD-based, ImageNet A/R).")
    ap.add_argument("--model", default="zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14", help="CLIP model id/path to load (OpenAI name, HF id, or local checkpoint).")
    ap.add_argument("--out_path", default="out_calibrated_model_logit_scale", help="Directory (or .pt file path) to save the calibrated full-model pickle to.")
    
    ap.add_argument("--imagenet_val_dir", default="path/to/ILSVRC2012/val", help="ImageNet val root with WNID subfolders.")
    ap.add_argument("--imagenet_a_dir", default="path/to/imagenet-a", help="ImageNet-A root with WNID subfolders.")
    ap.add_argument("--imagenet_r_dir", default="path/to/imagenet-r", help="ImageNet-R root with WNID subfolders.")
    ap.add_argument("--wnid_to_class_json", default="utils_datasets/imagenet/imagenet_wnid_to_class.json", help="JSON mapping WNID -> class name (used to build 1000-way prompts).")

    ap.add_argument("--id_mode", default="all", choices=["all", "match_a", "match_r", "match_ar"], help="Which subset of ImageNet-val to use as ID (by WNID intersection with A/R).")
    ap.add_argument("--ood_set", default="ar", choices=["a", "r", "ar"], help="Which OOD set to calibrate against: a, r, or ar (union).")
    ap.add_argument("--score_type", default="entropy", choices=["energy", "msp", "entropy", "prob_margin", "maxlogit"], help="OOD score to optimize (recommended: entropy).")
    ap.add_argument("--objective", default="auroc", choices=["fpr95", "auroc"], help="Optimization objective: minimize fpr95 or maximize auroc.")

    ap.add_argument("--sweep", action="store_true", help="Run full metric sweep and save plots.")
    ap.add_argument("--sweep_min_log", type=float, default=1.0)
    ap.add_argument("--sweep_max_log", type=float, default=8.0)
    ap.add_argument("--sweep_steps", type=int, default=120)
    ap.add_argument("--sweep_scores", default="energy,msp,entropy,prob_margin", help="Comma-separated score types to sweep.")
    ap.add_argument("--sweep_out_dir", default="out_eval_measure/logit_scale_sweep", help="Where to save sweep CSV/JSON/plots.")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--text_batch_size", type=int, default=4096)
    ap.add_argument("--amp", action="store_true", help="Use AMP autocast on CUDA.")

    ap.add_argument("--max_id", type=int, default=4000, help="Max ID images (0=all after filters).")
    ap.add_argument("--max_ood_a", type=int, default=4000, help="Max OOD-A images (0=all).")
    ap.add_argument("--max_ood_r", type=int, default=4000, help="Max OOD-R images (0=all).")
    ap.add_argument("--max_per_class_id", type=int, default=20, help="Cap per-WNID ID images (0=all).")
    ap.add_argument("--max_per_class_ood", type=int, default=20, help="Cap per-WNID OOD images (0=all).")

    ap.add_argument("--search_min_log", type=float, default=2.0, help="Min logit_scale (log-space).")
    ap.add_argument("--search_max_log", type=float, default=20, help="Max logit_scale (log-space).")
    ap.add_argument("--search_steps", type=int, default=80, help="Grid steps per search round.")
    ap.add_argument("--refine_rounds", type=int, default=2, help="Number of refinement rounds.")
    ap.add_argument("--refine_window", type=float, default=0.35, help="Refinement window (+/- in log-space).")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_json", action="store_true", default=True, help="Save a small report json next to the output model.")
    ap.add_argument("--save_fp16", action="store_true", default=True, help="Convert selected weights to fp16 before saving (storage convenience).")

    return ap.parse_args()


PROMPT_TEMPLATES_ENSEMBLE = [
    "a photo of a {c}",
    "a photo of the {c}",
    "a photo of one {c}",
    "a photo of a {c} in the scene",
    "a close-up photo of a {c}",
    "a bright photo of a {c}",
    "a cropped photo of a {c}",
    "a good photo of a {c}",
    "a photo of a small {c}",
    "a photo of a large {c}",
    "a blurry photo of a {c}",
    "a dark photo of a {c}",
    "a photo of a clean {c}",
    "a photo of a dirty {c}",
]


def _sanitize_factor(x: float) -> str:
    # turn 2.2 -> "2_2", 1.000 -> "1"
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    s = s.replace(".", "_")
    return s


def _model_id_to_name_base(model_id: str) -> str:
    s = model_id.strip().rstrip("/")

    # local file path
    if os.path.isfile(s):
        base = os.path.basename(s)
        base, _ext = os.path.splitext(base)
        return base

    # HF id: take last component
    base = s.split("/")[-1] if "/" in s else s

    # sanitize for filenames
    base = base.replace("/", "-")
    base = re.sub(r"[^A-Za-z0-9._\-]+", "-", base).strip("-")
    return base if base else "clip_model"


def _resolve_out_file(out_path: str, name_base: str, factor_s: str) -> str:
    """
    If out_path is a file (.pt/.pth/.bin), use it directly.
    Else treat out_path as a directory and auto-name the file.
    """
    op = out_path.strip()
    low = op.lower()
    if low.endswith((".pt", ".pth", ".bin")):
        parent = os.path.dirname(op)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return op

    os.makedirs(op, exist_ok=True)
    return os.path.join(op, f"{name_base}_logit-scale_{factor_s}.pt")


def fix_random_seed(seed: int = 6247423):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16 (storage convenience)."""
    def _convert_weights_to_fp16(l):
        if isinstance(l, nn.MultiheadAttention):
            for attr in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                module = getattr(l, attr, None)
                if module is not None and hasattr(module, "weight"):
                    module.weight.data = module.weight.data.half()
                    if module.bias is not None:
                        module.bias.data = module.bias.data.half()
            for attr in ["bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def list_wnid_subfolders(root: str) -> List[str]:
    out = []
    if not os.path.isdir(root):
        return out
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            out.append(d)
    out.sort()
    return out


def list_images_in_folder(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".jpe")
    out = []
    for fn in os.listdir(folder):
        low = fn.lower()
        if low.endswith(exts) or fn.endswith(".JPEG"):
            out.append(os.path.join(folder, fn))
    out.sort()
    return out


def build_paths_from_wnid_root(
    root: str,
    allowed_wnids: Optional[Set[str]] = None,
    max_per_class: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Returns (paths, wnids_used) where paths are image file paths.
    """
    paths: List[str] = []
    wnids_used: List[str] = []

    wnids = list_wnid_subfolders(root)
    for w in wnids:
        if allowed_wnids is not None and w not in allowed_wnids:
            continue
        d = os.path.join(root, w)
        files = list_images_in_folder(d)
        if max_per_class and max_per_class > 0:
            files = files[:max_per_class]
        if files:
            paths.extend(files)
            wnids_used.append(w)

    paths.sort()
    wnids_used.sort()
    return paths, wnids_used


class ImagePathDataset(Dataset):
    def __init__(self, img_paths: List[str], transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        p = self.img_paths[idx]
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img)


def collate_images(batch):
    return torch.stack(batch, dim=0)


@torch.inference_mode()
def build_text_features_imagenet_1k(
    model,
    wnid_to_class: Dict[str, str],
    templates: List[str],
    device: str,
    text_batch_size: int = 4096,
) -> torch.Tensor:
    """
    Builds [1000, D] normalized text features on device.
    Ordering: sorted WNID keys from wnid_to_class.
    """
    wnids = sorted(wnid_to_class.keys())
    classnames = [wnid_to_class[w] for w in wnids]

    prompts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for c in classnames:
        a = len(prompts)
        for t in templates:
            prompts.append(t.format(c=c))
        b = len(prompts)
        offsets.append((a, b))

    tokens_cpu = clip.tokenize(prompts, truncate=True)

    chunks = []
    for i in range(0, tokens_cpu.shape[0], text_batch_size):
        tok = tokens_cpu[i:i + text_batch_size].to(device, non_blocking=(device == "cuda"))
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = model.encode_text(tok)
        else:
            y = model.encode_text(tok)
        y = l2_normalize(y.float())
        chunks.append(y.detach())

    Y = torch.cat(chunks, dim=0)  # [P,D]

    class_feats = []
    for (a, b) in offsets:
        z = Y[a:b].mean(dim=0)
        z = l2_normalize(z)
        class_feats.append(z)

    return torch.stack(class_feats, dim=0).to(device)


@torch.inference_mode()
def encode_sims_for_paths(
    model,
    preprocess,
    img_paths: List[str],
    text_features: torch.Tensor,  # [C,D] on device
    device: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> torch.Tensor:
    """
    Returns sims [N,C] on device (float32).
    """
    ds = ImagePathDataset(img_paths, transform=preprocess)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=(device == "cuda"),
        persistent_workers=(int(num_workers) > 0),
        collate_fn=collate_images,
    )

    sims_chunks = []
    for x in dl:
        x = x.to(device, non_blocking=(device == "cuda"))
        if device == "cuda" and amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                img = model.encode_image(x)
        else:
            img = model.encode_image(x)

        img = l2_normalize(img.float())
        sims = img @ text_features.T  # [B,C]
        sims_chunks.append(sims.detach())

    if not sims_chunks:
        raise RuntimeError("No images encoded for sims (empty dataset?)")

    return torch.cat(sims_chunks, dim=0).to(torch.float32)


@torch.inference_mode()
def score_from_sims(sims: torch.Tensor, scale: float, score_type: str) -> torch.Tensor:
    """
    sims: [N,C] on device
    returns scores [N] on device, higher => more ID-like
    """
    score_type = score_type.lower().strip()

    if score_type == "maxlogit":
        # for AUROC and FPR@95TPR w/ quantile threshold this is invariant to positive scaling.
        return (sims * float(scale)).max(dim=1).values

    if score_type == "energy":
        logits = sims * float(scale)
        return torch.logsumexp(logits, dim=1)

    if score_type == "msp":
        logits = sims * float(scale)
        p = F.softmax(logits, dim=1)
        return p.max(dim=1).values

    if score_type == "entropy":
        logits = sims * float(scale)
        p = F.softmax(logits, dim=1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
        return -ent  # higher => more confident/ID-like

    if score_type == "prob_margin":
        logits = sims * float(scale)
        p = F.softmax(logits, dim=1)
        top2 = torch.topk(p, k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    raise ValueError(f"Unknown score_type={score_type}")


def auroc_rank(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """
    AUROC with ID as positive class (higher score => more ID-like).
    Rank-based (no sklearn).
    """
    y = np.concatenate([np.ones_like(scores_id, dtype=np.int32), np.zeros_like(scores_ood, dtype=np.int32)])
    s = np.concatenate([scores_id, scores_ood])

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg + 1e-12)
    return float(auc)


def auprc_average_precision(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """
    Average Precision (AP) treating ID as positive class.
    Higher score => more ID-like.

    Implementation: sort by score desc; AP = sum(precision@k * delta_recall).
    """
    y = np.concatenate([np.ones_like(scores_id, dtype=np.int32), np.zeros_like(scores_ood, dtype=np.int32)])
    s = np.concatenate([scores_id, scores_ood])

    order = np.argsort(-s)  # desc
    y_sorted = y[order]

    P = float((y_sorted == 1).sum())
    if P <= 0:
        return 0.0

    tp = 0.0
    fp = 0.0
    ap = 0.0
    prev_recall = 0.0

    for yi in y_sorted:
        if yi == 1:
            tp += 1.0
        else:
            fp += 1.0

        recall = tp / P
        precision = tp / max(1e-12, (tp + fp))

        # only add area when recall increases (i.e., at positives)
        if yi == 1:
            ap += precision * (recall - prev_recall)
            prev_recall = recall

    return float(ap)


def pr_auc_trapezoid(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """
    Trapezoidal PR-AUC (less standard than AP, but useful to plot).
    Treat ID as positive class; higher score => more ID-like.
    """
    y = np.concatenate([np.ones_like(scores_id, dtype=np.int32), np.zeros_like(scores_ood, dtype=np.int32)])
    s = np.concatenate([scores_id, scores_ood])

    order = np.argsort(-s)
    y_sorted = y[order]

    P = float((y_sorted == 1).sum())
    if P <= 0:
        return 0.0

    tp = 0.0
    fp = 0.0

    recalls = []
    precisions = []

    for yi in y_sorted:
        if yi == 1:
            tp += 1.0
        else:
            fp += 1.0
        recall = tp / P
        precision = tp / max(1e-12, tp + fp)
        recalls.append(recall)
        precisions.append(precision)

    # ensure starts at recall=0
    recalls = np.array([0.0] + recalls, dtype=np.float64)
    precisions = np.array([1.0] + precisions, dtype=np.float64)

    # trapezoid in recall-space
    area = np.trapz(precisions, recalls)
    return float(area)



def fpr_at_95_tpr(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    thr = np.quantile(scores_id, 0.05)  # accept top 95% ID => threshold at 5th percentile
    return float((scores_ood >= thr).mean())


def eval_ood_metrics(
    sims_id: torch.Tensor,
    sims_ood: torch.Tensor,
    logit_scale_log: float,
    score_type: str,
) -> Dict[str, float]:
    scale = float(math.exp(logit_scale_log))
    s_id = score_from_sims(sims_id, scale=scale, score_type=score_type).detach().cpu().numpy()
    s_ood = score_from_sims(sims_ood, scale=scale, score_type=score_type).detach().cpu().numpy()

    auroc = auroc_rank(s_id, s_ood)
    fpr95 = fpr_at_95_tpr(s_id, s_ood)

    ap = auprc_average_precision(s_id, s_ood)
    prauc = pr_auc_trapezoid(s_id, s_ood)

    return {
        "auroc": float(auroc),
        "fpr95": float(fpr95),
        "auprc_ap": float(ap),
        "auprc_trap": float(prauc),
        "scale": float(scale),
        "logit_scale_log": float(logit_scale_log),
    }


def run_metric_sweep_and_plot(
    sims_id: torch.Tensor,
    sims_ood: torch.Tensor,
    score_types: List[str],
    sweep_min_log: float,
    sweep_max_log: float,
    sweep_steps: int,
    out_dir: str,
    title_prefix: str = "",
):
    import csv
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    grid = np.linspace(float(sweep_min_log), float(sweep_max_log), int(sweep_steps), dtype=np.float64)

    rows = []
    for st in score_types:
        st = st.strip().lower()
        print(f"[Sweep] score_type={st}  steps={len(grid)}  log_range=[{sweep_min_log},{sweep_max_log}]")

        for ls in grid:
            m = eval_ood_metrics(sims_id, sims_ood, float(ls), score_type=st)
            rows.append({
                "score_type": st,
                "logit_scale_log": m["logit_scale_log"],
                "scale_exp": m["scale"],
                "auroc": m["auroc"],
                "fpr95": m["fpr95"],
                "auprc_ap": m["auprc_ap"],
                "auprc_trap": m["auprc_trap"],
            })

    # Save CSV
    csv_path = os.path.join(out_dir, "metrics_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Saved] {csv_path}")

    # Save JSON
    json_path = os.path.join(out_dir, "metrics_sweep.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[Saved] {json_path}")

    # Plot helper
    def plot_metric(metric_key: str, y_label: str, fname: str):
        plt.figure()
        for st in score_types:
            st = st.strip().lower()
            xs = [r["logit_scale_log"] for r in rows if r["score_type"] == st]
            ys = [r[metric_key] for r in rows if r["score_type"] == st]
            plt.plot(xs, ys, label=st)
        plt.xlabel("logit_scale (log space)")
        plt.ylabel(y_label)
        if title_prefix:
            plt.title(f"{title_prefix} {y_label}")
        else:
            plt.title(y_label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        outp = os.path.join(out_dir, fname)
        plt.savefig(outp, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {outp}")

    plot_metric("fpr95", "FPR@95TPR (lower is better)", "plot_fpr95_vs_logit_scale.png")
    plot_metric("auroc", "AUROC (higher is better)", "plot_auroc_vs_logit_scale.png")
    plot_metric("auprc_ap", "AUPRC (AP) (higher is better)", "plot_auprc_ap_vs_logit_scale.png")
    plot_metric("auprc_trap", "AUPRC (trap) (higher is better)", "plot_auprc_trap_vs_logit_scale.png")


@torch.inference_mode()
def search_logit_scale(
    sims_id: torch.Tensor,
    sims_ood: torch.Tensor,
    init_logit_scale: float,
    score_type: str,
    objective: str,
    search_min_log: float,
    search_max_log: float,
    steps: int,
    refine_rounds: int,
    refine_window: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Grid search over logit_scale in log-space.
    Returns (best_logit_scale, best_metrics_dict).
    """
    objective = objective.lower().strip()
    score_type = score_type.lower().strip()

    if score_type == "maxlogit":
        print("[Warn] score_type=maxlogit is invariant to positive scaling for AUROC/FPR@95TPR under this evaluation; "
              "calibration will not meaningfully change metrics.")

    def metric_for(ls_log: float) -> Tuple[float, Dict[str, float]]:
        m = eval_ood_metrics(sims_id, sims_ood, ls_log, score_type=score_type)
        if objective == "fpr95":
            return float(m["fpr95"]), m  # minimize
        elif objective == "auroc":
            return -float(m["auroc"]), m  # maximize => minimize negative
        else:
            raise ValueError(f"Unknown objective={objective} (use fpr95 or auroc)")

    # start centered around init, but clamp to given range
    lo = float(search_min_log)
    hi = float(search_max_log)

    best_ls = float(init_logit_scale)
    best_val, best_m = metric_for(best_ls)

    for r in range(refine_rounds + 1):
        grid = torch.linspace(lo, hi, steps=int(steps)).tolist()

        for ls in grid:
            v, m = metric_for(float(ls))
            if v < best_val:
                best_val = v
                best_ls = float(ls)
                best_m = m

        # refine around best
        if r < refine_rounds:
            lo = max(float(search_min_log), best_ls - float(refine_window))
            hi = min(float(search_max_log), best_ls + float(refine_window))

    return best_ls, best_m

def main():
    args = parse_args()
    fix_random_seed(int(args.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Device] {device}")
    print(f"[Load] clip.load: {args.model}")
    model, preprocess, _ = load_openai_clip_anything(clip, args.model, device=device, jit=False, strict=True)
    model = model.float().eval()

    # WNID -> class name
    with open(args.wnid_to_class_json, "r", encoding="utf-8") as f:
        wnid_to_class: Dict[str, str] = json.load(f)

    wnids_val = set(list_wnid_subfolders(args.imagenet_val_dir))
    wnids_a = set(list_wnid_subfolders(args.imagenet_a_dir))
    wnids_r = set(list_wnid_subfolders(args.imagenet_r_dir))
    wnids_ar = wnids_a | wnids_r

    wnids_match_a = wnids_val & wnids_a
    wnids_match_r = wnids_val & wnids_r
    wnids_match_ar = wnids_val & wnids_ar

    print(f"[Classes] val wnid folders: {len(wnids_val)}")
    print(f"[Classes] A wnid folders:   {len(wnids_a)}")
    print(f"[Classes] R wnid folders:   {len(wnids_r)}")
    print(f"[Classes] val∩A: {len(wnids_match_a)}  val∩R: {len(wnids_match_r)}  val∩(A∪R): {len(wnids_match_ar)}")

    # Choose ID wnids
    if args.id_mode == "all":
        allowed_id = wnids_val
    elif args.id_mode == "match_a":
        allowed_id = wnids_match_a
    elif args.id_mode == "match_r":
        allowed_id = wnids_match_r
    else:
        allowed_id = wnids_match_ar

    # Build image path lists
    id_paths, id_wnids_used = build_paths_from_wnid_root(
        args.imagenet_val_dir,
        allowed_wnids=set(allowed_id),
        max_per_class=int(args.max_per_class_id),
    )
    ood_a_paths, _ = build_paths_from_wnid_root(
        args.imagenet_a_dir,
        allowed_wnids=None,
        max_per_class=int(args.max_per_class_ood),
    )
    ood_r_paths, _ = build_paths_from_wnid_root(
        args.imagenet_r_dir,
        allowed_wnids=None,
        max_per_class=int(args.max_per_class_ood),
    )

    if int(args.max_id) > 0:
        id_paths = id_paths[: int(args.max_id)]
    if int(args.max_ood_a) > 0:
        ood_a_paths = ood_a_paths[: int(args.max_ood_a)]
    if int(args.max_ood_r) > 0:
        ood_r_paths = ood_r_paths[: int(args.max_ood_r)]

    if args.ood_set == "a":
        ood_paths = ood_a_paths
        ood_name = "A"
    elif args.ood_set == "r":
        ood_paths = ood_r_paths
        ood_name = "R"
    else:
        ood_paths = ood_a_paths + ood_r_paths
        ood_name = "A∪R"

    print(f"[ID] mode={args.id_mode}  images={len(id_paths):,}  wnids_used={len(id_wnids_used)}")
    print(f"[OOD-{ood_name}] images={len(ood_paths):,}")
    print(f"[Calib] score_type={args.score_type}  objective={args.objective}")

    # Build 1000-way text features once
    print("[Text] building 1000-way prompt-ensemble text features...")
    tf = build_text_features_imagenet_1k(
        model=model,
        wnid_to_class=wnid_to_class,
        templates=PROMPT_TEMPLATES_ENSEMBLE,
        device=device,
        text_batch_size=int(args.text_batch_size),
    )
    print(f"[Text] text_features={tuple(tf.shape)}")

    # Precompute sims ONCE
    print("[Encode] precomputing sims for ID...")
    sims_id = encode_sims_for_paths(
        model=model,
        preprocess=preprocess,
        img_paths=id_paths,
        text_features=tf,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
    )
    print("[Encode] precomputing sims for OOD...")
    sims_ood = encode_sims_for_paths(
        model=model,
        preprocess=preprocess,
        img_paths=ood_paths,
        text_features=tf,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
    )

    if args.sweep:
        score_types = [s.strip() for s in str(args.sweep_scores).split(",") if s.strip()]
        run_metric_sweep_and_plot(
            sims_id=sims_id,
            sims_ood=sims_ood,
            score_types=score_types,
            sweep_min_log=float(args.sweep_min_log),
            sweep_max_log=float(args.sweep_max_log),
            sweep_steps=int(args.sweep_steps),
            out_dir=str(args.sweep_out_dir),
            title_prefix=f"ID={args.id_mode} OOD={args.ood_set} ({args.model})",
        )

    # Old logit_scale
    old_ls = float(model.logit_scale.detach().float().cpu().item())
    old_scale = float(math.exp(old_ls))
    print(f"[logit_scale] old logit_scale={old_ls:.6f}  exp={old_scale:.6f}")

    # Metrics before
    before = eval_ood_metrics(sims_id, sims_ood, old_ls, score_type=args.score_type)
    print(f"[Before] AUROC={before['auroc']:.6f}  FPR@95TPR={before['fpr95']:.6f}  scale={before['scale']:.6f}")

    # Search for best logit_scale
    best_ls, best_m = search_logit_scale(
        sims_id=sims_id,
        sims_ood=sims_ood,
        init_logit_scale=old_ls,
        score_type=args.score_type,
        objective=args.objective,
        search_min_log=float(args.search_min_log),
        search_max_log=float(args.search_max_log),
        steps=int(args.search_steps),
        refine_rounds=int(args.refine_rounds),
        refine_window=float(args.refine_window),
    )

    new_ls = float(best_ls)
    new_scale = float(math.exp(new_ls))

    # Factor: logits_new = logits_old / factor
    factor = float(old_scale / max(1e-12, new_scale))
    factor_s = _sanitize_factor(factor)

    after = eval_ood_metrics(sims_id, sims_ood, new_ls, score_type=args.score_type)
    print(f"[Fit]  new logit_scale={new_ls:.6f}  exp={new_scale:.6f}")
    print(f"[Fit]  factor(T)={factor:.6f}  sanitized={factor_s}")
    print(f"[After] AUROC={after['auroc']:.6f}  FPR@95TPR={after['fpr95']:.6f}  scale={after['scale']:.6f}")

    # Always update model.logit_scale and save
    name_base = _model_id_to_name_base(args.model)
    out_file = _resolve_out_file(args.out_path, name_base=name_base, factor_s=factor_s)

    print(f"[Save] updating model.logit_scale and torch.save -> {out_file}")
    with torch.no_grad():
        model.logit_scale.copy_(torch.tensor(new_ls, device=model.logit_scale.device, dtype=model.logit_scale.dtype))

    convert_weights(model)

    torch.save(model, out_file)

    # Report json
    rep_path = os.path.splitext(out_file)[0] + ".json"
    rep = dict(
        model_in=str(args.model),
        model_out=str(out_file),
        out_path=str(args.out_path),
        seed=int(args.seed),

        imagenet_val_dir=str(args.imagenet_val_dir),
        imagenet_a_dir=str(args.imagenet_a_dir),
        imagenet_r_dir=str(args.imagenet_r_dir),
        wnid_to_class_json=str(args.wnid_to_class_json),

        id_mode=str(args.id_mode),
        ood_set=str(args.ood_set),
        score_type=str(args.score_type),
        objective=str(args.objective),

        n_id=int(sims_id.shape[0]),
        n_ood=int(sims_ood.shape[0]),

        old_logit_scale=old_ls,
        new_logit_scale=new_ls,
        old_exp_logit_scale=old_scale,
        new_exp_logit_scale=new_scale,
        factor_temperature=factor,

        before=before,
        after=after,

        search=dict(
            search_min_log=float(args.search_min_log),
            search_max_log=float(args.search_max_log),
            search_steps=int(args.search_steps),
            refine_rounds=int(args.refine_rounds),
            refine_window=float(args.refine_window),
        ),
    )

    if bool(args.save_json):
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"[Save] report -> {rep_path}")

    print(f"[Done] Saved calibrated model to: {out_file}")


if __name__ == "__main__":
    main()