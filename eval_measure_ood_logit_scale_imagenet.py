"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

CLIP OOD Benchmark (ImageNet-val vs ImageNet-A / ImageNet-R) (logit_scale matters)
====================================================================================
Download:
https://www.image-net.org/download.php
https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar

imagenet_wnid_to_class.json -> included in this repo

ID set modes:
  - id_mode="all"     : use all val images
  - id_mode="match_a" : only WNIDs that exist in ImageNet-A
  - id_mode="match_r" : only WNIDs that exist in ImageNet-R
  - id_mode="match_ar": only WNIDs that exist in (A ∪ R)

OOD sets:
  - OOD-A: ImageNet-A WNID folders
  - OOD-R: ImageNet-R WNID folders
  - OOD-AR: concatenation of A and R

Classifier:
  - 1000-way zero-shot using WNID->classname mapping + prompt ensemble.

Scores (logit_scale matters for the first three):
  - MSP      : max softmax probability over 1000 classes
  - ENERGY   : logsumexp(logits)
  - MAXLOGIT : max(logits)
  - MAXCOS   : max cosine similarity (temperature-invariant)

Metrics:
  - AUROC (ID=positive)
  - FPR@95TPR
"""

from __future__ import annotations

import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# Models: OpenAI / local path .pt .safetensors / HuggingFace Hub
# ============================================================

MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
    ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
    ("regress-norm", "zer0int/CLIP-Regression-ViT-L-14"),
    ("regress-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
]

IMAGENET_VAL_DIR = "path/to/ILSVRC2012/val"
IMAGENET_A_DIR = "path/to/imagenet-a"
IMAGENET_R_DIR = "path/to/imagenet-r"
WNID_TO_CLASS_JSON = "utils_datasets/imagenet/imagenet_wnid_to_class.json"

OUT_DIR = "out_eval_measure/ood_logit_scale"


def parse_args():
    ap = argparse.ArgumentParser("CLIP OOD Benchmark (WNID val)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--text_batch_size", type=int, default=4096)

    ap.add_argument("--id_mode", default="all", choices=["all", "match_a", "match_r", "match_ar"], help="Which subset of ImageNet-val to use as ID.")
    ap.add_argument("--max_id", type=int, default=0, help="limit #ID images (0=all after id_mode filter)")
    ap.add_argument("--max_ood_a", type=int, default=0, help="limit #OOD-A images (0=all)")
    ap.add_argument("--max_ood_r", type=int, default=0, help="limit #OOD-R images (0=all)")
    ap.add_argument("--max_per_class_id", type=int, default=20, help="limit per-WNID ID images (0=all)")
    ap.add_argument("--max_per_class_ood", type=int, default=20, help="limit per-WNID OOD images (0=all)")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--scores", default="msp,energy,maxlogit,maxcos,entropy,prob_margin", help="comma-separated: msp,energy,maxlogit,maxcos,entropy,prob_margin",
    )
    return ap.parse_args()


PROMPT_TEMPLATES = [
    "a photo of a {c}",
    "a photo of the {c}",
    "a close-up photo of a {c}",
    "a cropped photo of a {c}",
    "a bright photo of a {c}",
    "a blurry photo of a {c}",
]

def fix_random_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".jpe", ".jpeg", ".jxl")
    # include .JPEG (uppercase) commonly in ImageNet
    out = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts) or fn.endswith(".JPEG"):
            out.append(os.path.join(folder, fn))
    out.sort()
    return out


def build_paths_from_wnid_root(root: str, allowed_wnids: Optional[Set[str]] = None, max_per_class: int = 0) -> Tuple[List[str], List[str]]:
    """
    Returns (paths, wnids_used)
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


def compute_auc_roc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
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


def fpr_at_95_tpr(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    thr = np.quantile(scores_id, 0.05)  # accept top 95% ID
    return float((scores_ood >= thr).mean())


def plot_hist(scores_id: np.ndarray, scores_ood: np.ndarray, title: str, out_path: str):
    plt.figure()
    plt.hist(scores_id, bins=80, alpha=0.6, label="ID", density=True)
    plt.hist(scores_ood, bins=80, alpha=0.6, label="OOD", density=True)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc(scores_id: np.ndarray, scores_ood: np.ndarray, title: str, out_path: str):
    y = np.concatenate([np.ones_like(scores_id, dtype=np.int32), np.zeros_like(scores_ood, dtype=np.int32)])
    s = np.concatenate([scores_id, scores_ood])

    order = np.argsort(-s)
    y_sorted = y[order]

    tp = 0.0
    fp = 0.0
    tps = []
    fps = []
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())

    for yi in y_sorted:
        if yi == 1:
            tp += 1.0
        else:
            fp += 1.0
        tps.append(tp / max(1.0, n_pos))
        fps.append(fp / max(1.0, n_neg))

    plt.figure()
    plt.plot(fps, tps)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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


# ============================================================
# Text features (1000-way)
# ============================================================

@torch.inference_mode()
def build_text_features_imagenet(
    model,
    wnid_to_class: Dict[str, str],
    templates: List[str],
    device: str,
    text_batch_size: int = 4096,
) -> torch.Tensor:
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

    Y = torch.cat(chunks, dim=0)

    class_feats = []
    for (a, b) in offsets:
        z = Y[a:b].mean(dim=0)
        z = l2_normalize(z)
        class_feats.append(z)

    return torch.stack(class_feats, dim=0).to(device)  # [1000, D]


# ============================================================
# Scoring (logit_scale matters for logits-based scores)
# ============================================================

@torch.inference_mode()
def compute_scores(
    model,
    dl: DataLoader,
    text_feats: torch.Tensor,
    device: str,
    amp: bool,
    score_type: str,
) -> np.ndarray:
    """
    Returns scores where higher => more ID-like.

    score_type:
      - "msp"         : max softmax prob
      - "energy"      : logsumexp(logits)
      - "maxlogit"    : max(logits)              (AUROC/FPR@95 invariant to temperature scaling)
      - "maxcos"      : max cosine(sim)          (temperature-invariant)
      - "entropy"     : NEGATIVE entropy of softmax  (we return -H so higher => more ID-like)   # NEW
      - "prob_margin" : p_top1 - p_top2          (softmax margin; temperature-sensitive)       # NEW
    """
    ls = float(model.logit_scale.detach().float().cpu().item())
    scale = float(math.exp(ls))

    scores: List[np.ndarray] = []
    for x in tqdm(dl, desc=f"scores:{score_type}"):
        x = x.to(device, non_blocking=(device == "cuda"))

        if device == "cuda" and amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                img = model.encode_image(x)
        else:
            img = model.encode_image(x)

        img = l2_normalize(img.float())
        sims = img @ text_feats.T  # [B,C]

        if score_type == "maxcos":
            sc = sims.max(dim=1).values
            scores.append(sc.detach().cpu().numpy())
            continue

        logits = sims * scale  # [B,C]

        if score_type == "msp":
            p = F.softmax(logits, dim=1)
            sc = p.max(dim=1).values

        elif score_type == "energy":
            sc = torch.logsumexp(logits, dim=1)

        elif score_type == "maxlogit":
            sc = logits.max(dim=1).values

        elif score_type == "entropy":  # NEW
            # return -H(p) so larger => more confident/ID-like
            p = F.softmax(logits, dim=1)
            ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=1)
            sc = -ent

        elif score_type == "prob_margin":  # NEW
            p = F.softmax(logits, dim=1)
            top2 = torch.topk(p, k=2, dim=1).values  # [B,2]
            sc = top2[:, 0] - top2[:, 1]

        else:
            raise ValueError(f"Unknown score_type={score_type}")

        scores.append(sc.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    fix_random_seed(int(args.seed))
    ensure_dir(OUT_DIR)

    print("\n==================================")
    print("OOD Classifier: ImageNet /-a /-r")
    print("==================================\n")

    with open(WNID_TO_CLASS_JSON, "r", encoding="utf-8") as f:
        wnid_to_class: Dict[str, str] = json.load(f)

    wnids_val = set(list_wnid_subfolders(IMAGENET_VAL_DIR))
    wnids_a = set(list_wnid_subfolders(IMAGENET_A_DIR))
    wnids_r = set(list_wnid_subfolders(IMAGENET_R_DIR))

    wnids_ar = wnids_a | wnids_r
    wnids_match_a = wnids_val & wnids_a
    wnids_match_r = wnids_val & wnids_r
    wnids_match_ar = wnids_val & wnids_ar

    # save class-set info (useful to sanity-check intersections)
    class_sets_path = os.path.join(OUT_DIR, "class_sets.json")
    with open(class_sets_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts": {
                    "val_wnids": len(wnids_val),
                    "a_wnids": len(wnids_a),
                    "r_wnids": len(wnids_r),
                    "match_a": len(wnids_match_a),
                    "match_r": len(wnids_match_r),
                    "match_ar": len(wnids_match_ar),
                },
                "wnids": {
                    "val_sample": sorted(list(wnids_val))[:20],
                    "a_sample": sorted(list(wnids_a))[:20],
                    "r_sample": sorted(list(wnids_r))[:20],
                },
            },
            f,
            indent=2,
        )

    print(f"[Classes] val wnid folders: {len(wnids_val)}")
    print(f"[Classes] A wnid folders:   {len(wnids_a)}")
    print(f"[Classes] R wnid folders:   {len(wnids_r)}")
    print(f"[Classes] val∩A: {len(wnids_match_a)}  val∩R: {len(wnids_match_r)}  val∩(A∪R): {len(wnids_match_ar)}")
    print(f"[Saved] {class_sets_path}")

    # choose ID wnids
    if args.id_mode == "all":
        allowed_id = wnids_val
    elif args.id_mode == "match_a":
        allowed_id = wnids_match_a
    elif args.id_mode == "match_r":
        allowed_id = wnids_match_r
    else:
        allowed_id = wnids_match_ar

    id_paths, id_wnids_used = build_paths_from_wnid_root(
        IMAGENET_VAL_DIR,
        allowed_wnids=allowed_id,
        max_per_class=int(args.max_per_class_id),
    )
    if args.max_id and args.max_id > 0:
        id_paths = id_paths[: int(args.max_id)]

    ood_a_paths, _ = build_paths_from_wnid_root(
        IMAGENET_A_DIR,
        allowed_wnids=None,
        max_per_class=int(args.max_per_class_ood),
    )
    ood_r_paths, _ = build_paths_from_wnid_root(
        IMAGENET_R_DIR,
        allowed_wnids=None,
        max_per_class=int(args.max_per_class_ood),
    )
    if args.max_ood_a and args.max_ood_a > 0:
        ood_a_paths = ood_a_paths[: int(args.max_ood_a)]
    if args.max_ood_r and args.max_ood_r > 0:
        ood_r_paths = ood_r_paths[: int(args.max_ood_r)]

    print(f"[ID] mode={args.id_mode}  images={len(id_paths):,}  wnids_used={len(id_wnids_used)}")
    print(f"[OOD-A] images={len(ood_a_paths):,}")
    print(f"[OOD-R] images={len(ood_r_paths):,}")

    score_types = [s.strip().lower() for s in args.scores.split(",") if s.strip()]
    print(f"[Scores] {score_types}")

    all_results = []

    for mi, (alias, model_id) in enumerate(MODELS, start=1):
        print("\n" + "=" * 100)
        print(f"[Model {mi}/{len(MODELS)}] {alias}: {model_id}")
        print("=" * 100)

        device = args.device
        model, preprocess, _ = load_openai_clip_anything(clip, model_id, device=device, jit=False, strict=True)
        model = model.float().eval()

        text_feats = build_text_features_imagenet(
            model=model,
            wnid_to_class=wnid_to_class,
            templates=PROMPT_TEMPLATES,
            device=device,
            text_batch_size=int(args.text_batch_size),
        )

        ds_id = ImagePathDataset(id_paths, transform=preprocess)
        ds_a = ImagePathDataset(ood_a_paths, transform=preprocess)
        ds_r = ImagePathDataset(ood_r_paths, transform=preprocess)

        dl_id = DataLoader(ds_id, batch_size=int(args.batch_size), shuffle=False,
                           num_workers=int(args.num_workers), pin_memory=(device == "cuda"),
                           persistent_workers=(int(args.num_workers) > 0), collate_fn=collate_images)
        dl_a = DataLoader(ds_a, batch_size=int(args.batch_size), shuffle=False,
                          num_workers=int(args.num_workers), pin_memory=(device == "cuda"),
                          persistent_workers=(int(args.num_workers) > 0), collate_fn=collate_images)
        dl_r = DataLoader(ds_r, batch_size=int(args.batch_size), shuffle=False,
                          num_workers=int(args.num_workers), pin_memory=(device == "cuda"),
                          persistent_workers=(int(args.num_workers) > 0), collate_fn=collate_images)

        ls = float(model.logit_scale.detach().float().cpu().item())
        scale = float(math.exp(ls))
        print(f"[logit_scale] log={ls:.6f}  exp={scale:.6f}")

        per_model = {
            "alias": alias,
            "model": model_id,
            "id_mode": args.id_mode,
            "n_id": int(len(ds_id)),
            "n_ood_a": int(len(ds_a)),
            "n_ood_r": int(len(ds_r)),
            "logit_scale_log": ls,
            "logit_scale_exp": scale,
            "scores": {},
        }

        for st in score_types:
            print(f"\n[Compute] score={st}")
            scores_id = compute_scores(model, dl_id, text_feats, device=device, amp=bool(args.amp), score_type=st)
            scores_a = compute_scores(model, dl_a, text_feats, device=device, amp=bool(args.amp), score_type=st)
            scores_r = compute_scores(model, dl_r, text_feats, device=device, amp=bool(args.amp), score_type=st)
            scores_ar = np.concatenate([scores_a, scores_r], axis=0)

            auc_a = compute_auc_roc(scores_id, scores_a)
            fpr95_a = fpr_at_95_tpr(scores_id, scores_a)

            auc_r = compute_auc_roc(scores_id, scores_r)
            fpr95_r = fpr_at_95_tpr(scores_id, scores_r)

            auc_ar = compute_auc_roc(scores_id, scores_ar)
            fpr95_ar = fpr_at_95_tpr(scores_id, scores_ar)

            print(f"[OOD] {st:8s}  AUROC(ID vs A)={auc_a:.6f}  FPR@95TPR={fpr95_a:.6f}")
            print(f"[OOD] {st:8s}  AUROC(ID vs R)={auc_r:.6f}  FPR@95TPR={fpr95_r:.6f}")
            print(f"[OOD] {st:8s}  AUROC(ID vs A∪R)={auc_ar:.6f}  FPR@95TPR={fpr95_ar:.6f}")

            per_model["scores"][st] = {
                "id_vs_a": {"auroc": float(auc_a), "fpr95": float(fpr95_a)},
                "id_vs_r": {"auroc": float(auc_r), "fpr95": float(fpr95_r)},
                "id_vs_ar": {"auroc": float(auc_ar), "fpr95": float(fpr95_ar)},
            }

            tag = f"{alias}__{args.id_mode}__{st}"
            plot_hist(scores_id, scores_a, f"{tag}  ID vs ImageNet-A", os.path.join(OUT_DIR, f"hist_{tag}_A.png"))
            plot_hist(scores_id, scores_r, f"{tag}  ID vs ImageNet-R", os.path.join(OUT_DIR, f"hist_{tag}_R.png"))
            plot_hist(scores_id, scores_ar, f"{tag}  ID vs (A∪R)", os.path.join(OUT_DIR, f"hist_{tag}_AR.png"))

            plot_roc(scores_id, scores_a, f"{tag} ROC  ID vs A", os.path.join(OUT_DIR, f"roc_{tag}_A.png"))
            plot_roc(scores_id, scores_r, f"{tag} ROC  ID vs R", os.path.join(OUT_DIR, f"roc_{tag}_R.png"))
            plot_roc(scores_id, scores_ar, f"{tag} ROC  ID vs (A∪R)", os.path.join(OUT_DIR, f"roc_{tag}_AR.png"))

        all_results.append(per_model)

    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "paths": {
                    "imagenet_val": IMAGENET_VAL_DIR,
                    "imagenet_a": IMAGENET_A_DIR,
                    "imagenet_r": IMAGENET_R_DIR,
                    "wnid_to_class_json": WNID_TO_CLASS_JSON,
                },
                "config": {
                    "id_mode": args.id_mode,
                    "scores": score_types,
                    "prompt_templates": PROMPT_TEMPLATES,
                    "batch_size": int(args.batch_size),
                    "num_workers": int(args.num_workers),
                    "amp": bool(args.amp),
                    "max_id": int(args.max_id),
                    "max_ood_a": int(args.max_ood_a),
                    "max_ood_r": int(args.max_ood_r),
                    "max_per_class_id": int(args.max_per_class_id),
                    "max_per_class_ood": int(args.max_per_class_ood),
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print("\n" + "#" * 100)
    print(f"[Saved] {out_json}")
    print(f"[Saved] plots -> {OUT_DIR}")
    print("#" * 100)


if __name__ == "__main__":
    main()