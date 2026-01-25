"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

 Download Flickr8k: Many places, e.g.:
 https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip

- Computes multiple “center/mean” modality-gap metrics between image and text embedding clouds:
    - Euclidean distance between embedding means in normalized space.
    - Cosine distance between mean directions (1 - cos(mu_i_hat, mu_t_hat)).
    - Diagonal Mahalanobis distance between means using pooled per-dimension variances.
    - Diagonal “Fréchet-like” distance using mean and diagonal covariance terms.
    - MMD (maximum mean discrepancy) with an RBF kernel, estimated on random subsets (bandwidth via median heuristic).
    
- Computes similarity-distribution statistics using cosine similarities:
    - IT: matched image–text cosine for each caption/prompt paired with its source image.
    - TT: text–text cosine within the same image (pairwise among that image’s captions/prompts), plus optional global TT sampling.
    - II: image–image cosine sampled from random image pairs.
  Then summarizes means/stds and “gaps” such as (II_mean - IT_mean) and (TT_mean - IT_mean).
  
- Quantifies distributional differences between similarity distributions (IT vs TT vs II) using:
    - Jensen–Shannon distance (via histogrammed discrete distributions with adaptive binning).
    - Wasserstein (Earth mover’s) distance.
    - Kolmogorov–Smirnov statistic.
    
- Computes representation-learning style metrics:
    - “Alignment” proxy for IT pairs via expected squared distance: E[2 - 2*cos(IT)].
    - “Uniformity” per modality (Wang & Isola style): log E[exp(-t * ||xi-xj||^2)] estimated from random pairs.
    
- Computes richer embedding-geometry diagnostics per modality:
    - Covariance eigenspectrum on centered embeddings (full eigen-decomposition).
    - Effective rank (entropy-based), participation ratio, sphericity proxy (geometric mean / mean of eigenvalues)
    - Coefficient of variation of eigenvalues, robust condition proxy (p95/p05 of eigenvalues)
    - “Cone-ness”: norm of the mean vector ||mu|| and a heuristic vMF-like concentration kappa approximation from R = ||mu||.
    - Mean distance to the centroid ||x - mu||.
    - PCA explained-variance ratios (top-k and cumulative) for image, text, and pooled embeddings.
    - Principal angles (in degrees) between the top-k covariance eigenspaces of image vs text embeddings (subspace mismatch).
    - Distribution of cos(x, mu_hat) within each modality (how strongly samples align with their centroid direction).
    - Raw feature-norm stats (mean/std and percentiles) for unnormalized image/text features.
"""

from __future__ import annotations

import os
import math
import json
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

from utils_clip_loader.cliptools import fix_random_seed
fix_random_seed()

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
    ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
    ("regr-norm", "zer0int/CLIP-Regression-ViT-L-14"),
    ("regr-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_DATA_DIR = "path/to/Flickr8k"  # Must contain Images/ subfolder
DEFAULT_OUT_DIR = "out_eval_measure/modality_gap_geometry"

CAPTIONS_FLICKR = os.path.join("utils_datasets", "flickr8k", "captions_flickr8k.txt")

# In Flickr8k there are typically 5 captions per image; we *use what exists*.
DEFAULT_MAX_IMAGES: int = 2000  # set to e.g. 2000 for quick iteration; -1 = all

# SCAM/SynthSCAM caps
DEFAULT_MAX_SCAM: int = -1  # -1 = all
SCAM_VARIANTS_TO_RUN = ["NoSCAM", "SynthSCAM"]

# Dataloader knobs
DEFAULT_BATCH_SIZE = 128 if device == "cuda" else 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_PERSISTENT_WORKERS = True
DEFAULT_PIN_MEMORY = (device == "cuda")

# Text encoding batch
TEXT_BATCH_SIZE = 4096

# Similarity sampling (II and TT can be large if done fully)
N_IMAGE_IMAGE_PAIRS = 200_000
N_TEXT_TEXT_PAIRS_GLOBAL = 200_000  # additional global TT sampling (in addition to per-image caption-pairs)

# Uniformity sampling
UNIFORMITY_PAIRS = 200_000
UNIFORMITY_T = 2.0  # Wang & Isola use t=2

# MMD sampling
MMD_SUBSET = 1024  # from each modality

# TSNE (optional)
DEFAULT_DO_TSNE = False
TSNE_MAX_POINTS_TEXT = 6000
TSNE_MAX_POINTS_IMAGE = 3000

# Geometry / PCA knobs
PCA_TOPK = 64               # how many components to report EVR for
PRINCIPAL_ANGLES_K = 64     # top-k subspace to compare between modalities
EIG_SPECTRUM_POINTS = 256   # downsample spectrum plot to <= this many points if D larger
COS_TO_CENTROID_BINS = 120  # histogram bins for cos(x, mu_hat) plots

# 3D PCA view dumping (18 frames)
PCA3D_VIEWS = 18
PCA3D_MAX_POINTS_TEXT = 8000
PCA3D_MAX_POINTS_IMAGE = 4000


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate modality gap + geometry stats (multi-model, multi-dataset).")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Flickr8k base dir (contains Images/)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output base folder")
    parser.add_argument("--run_name", default="multi_dataset", help="Subfolder name under out_dir")

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")

    parser.add_argument("--max_images", type=int, default=DEFAULT_MAX_IMAGES, help="-1 = all")
    parser.add_argument("--max_scam", type=int, default=DEFAULT_MAX_SCAM, help="-1 = all (SCAM HF dataset)")
    parser.add_argument("--do_tsne", action="store_true", help="Compute TSNE per model per dataset (slow).")
    return parser.parse_args()



def load_clip_model(name_or_path: str, device_str: str):
    model, preprocess_fn, _ = load_openai_clip_anything(clip, name_or_path, device=device_str, jit=False, strict=True)
    model = model.eval().float()
    return model, preprocess_fn

def _read_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    """
    captions_flickr8k.txt format: image.jpg,caption text...
    """
    caps: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if "," not in ln:
                continue
            img_name, caption = ln.split(",", 1)
            img_name = img_name.strip()
            caption = caption.strip()
            if not img_name:
                continue
            caps.setdefault(img_name, []).append(caption)
    return caps

class FlickrImageDataset(Dataset):
    def __init__(self, images_dir: str, image_names: List[str], preprocess_fn):
        self.images_dir = images_dir
        self.image_names = image_names
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.images_dir, name)
        img = Image.open(path).convert("RGB")
        return self.preprocess_fn(img), idx

class PILListImageDataset(Dataset):
    def __init__(self, pil_images: List[Image.Image], preprocess_fn):
        self.pil_images = pil_images
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.pil_images)

    def __getitem__(self, idx):
        img = self.pil_images[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.preprocess_fn(img), idx

# SCAM prompt diversification
PROMPT_TEMPLATES_OBJ = [
    "a photo of a {w}",
    "there is a {w}",
    "a close-up photo of a {w}",
    "someone made an image of a {w}",
    "the appearance of a carefully crafted {w}",
    "a {w} in the scene",
]

PROMPT_TEMPLATES_POSTIT = [
    "there is a {w} with a post-it note stuck to it",
    "a photo of a {w} with a post-it stuck to it",
    "there is a {w} with a post-it note on it",
]

PROMPT_TEMPLATES_ATTACK = [
    "a post-it note that says '{atk}'",
    "there is a post-it note reading '{atk}'",
    "the word '{atk}' on a post-it note",
]

PROMPT_TEMPLATES_OBJ_PLUS_ATK = [
    "there is a {w} with a post-it that says '{atk}'",
    "a photo of a {w} and a post-it reading '{atk}'",
    "there is a {w} with the word '{atk}' on a post-it",
]

def _build_scam_prompts(obj: str, atk: str, variant: str) -> List[str]:
    """
    Deterministic prompt set per image.
    We generate BOTH object-oriented and attack-word-oriented prompts,
    plus some "obj+attack" prompts (especially meaningful for SynthSCAM/SCAM).
    """
    obj = str(obj)
    atk = str(atk)
    out: List[str] = []

    # Object-only templates
    for t in PROMPT_TEMPLATES_OBJ:
        out.append(t.format(w=obj, atk=atk))

    # Post-it templates (still object grounded)
    for t in PROMPT_TEMPLATES_POSTIT:
        out.append(t.format(w=obj, atk=atk))

    # Variant-dependent additions
    if variant in ("NoSCAM", "SynthSCAM"):
        for t in PROMPT_TEMPLATES_OBJ_PLUS_ATK:
            out.append(t.format(w=obj, atk=atk))
        for t in PROMPT_TEMPLATES_ATTACK:
            out.append(t.format(w=obj, atk=atk))

    # De-dup while preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

@dataclass
class DatasetBundle:
    """
    A unified container so the evaluation loop can treat datasets the same way.
    """
    dataset_tag: str
    dl_images: DataLoader
    tokens_cpu: torch.Tensor
    img_of_text_arr: np.ndarray  # [T] -> image idx
    captions_flat: List[str]
    n_images: int
    n_texts: int


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int,
                 pin_memory: bool, persistent_workers: bool) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_flickr_bundle(args, preprocess_fn) -> DatasetBundle:
    base_dir = args.data_dir
    images_dir = os.path.join(base_dir, "Images")
    captions_file = CAPTIONS_FLICKR

    captions_dict = _read_flickr8k_captions(captions_file)

    image_names = []
    for img_name in captions_dict.keys():
        if os.path.isfile(os.path.join(images_dir, img_name)):
            image_names.append(img_name)
    image_names = sorted(image_names)

    if args.max_images is not None and args.max_images > 0:
        image_names = image_names[:args.max_images]

    captions_flat: List[str] = []
    img_of_text: List[int] = []
    for i_idx, img_name in enumerate(image_names):
        caps = captions_dict.get(img_name, [])
        for cap in caps:
            captions_flat.append(cap)
            img_of_text.append(i_idx)

    img_of_text_arr = np.array(img_of_text, dtype=np.int32)

    print(f"[Flickr8k] images={len(image_names):,}  captions={len(captions_flat):,}")

    print("[Tokenize] Flickr8k captions (once, shared across models)...")
    tokens_cpu = clip.tokenize(captions_flat, truncate=True)

    ds_images = FlickrImageDataset(images_dir=images_dir, image_names=image_names, preprocess_fn=preprocess_fn)
    dl_images = _make_loader(
        ds_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=((not args.no_pin_memory) and (device == "cuda")),
        persistent_workers=((not args.no_persistent_workers)),
    )

    return DatasetBundle(
        dataset_tag="Flickr8k",
        dl_images=dl_images,
        tokens_cpu=tokens_cpu,
        img_of_text_arr=img_of_text_arr,
        captions_flat=captions_flat,
        n_images=len(image_names),
        n_texts=len(captions_flat),
    )


def build_scam_bundles(args, preprocess_fn) -> List[DatasetBundle]:

    print("[SCAM] loading HF dataset: BLISS-e-V/SCAM (split=train)")
    ds = load_dataset("BLISS-e-V/SCAM", split="train")

    # bucket by variant prefix in id
    buckets: Dict[str, List[Dict[str, Any]]] = {v: [] for v in ["NoSCAM", "SCAM", "SynthSCAM"]}
    for entry in ds:
        sid = str(entry.get("id", ""))
        variant = None
        for v in buckets.keys():
            if sid.startswith(v):
                variant = v
                break
        if variant is None:
            continue
        buckets[variant].append(entry)

    bundles: List[DatasetBundle] = []
    for variant in SCAM_VARIANTS_TO_RUN:
        entries = buckets.get(variant, [])
        if not entries:
            print(f"[SCAM] variant={variant}: 0 entries, skipping.")
            continue

        if args.max_scam is not None and args.max_scam > 0:
            entries = entries[:args.max_scam]

        pil_images: List[Image.Image] = []
        captions_flat: List[str] = []
        img_of_text: List[int] = []

        for i_idx, entry in enumerate(entries):
            img = entry["image"]
            obj = str(entry.get("object_label", "object"))
            atk = str(entry.get("attack_word", "word"))

            pil_images.append(img)

            prompts = _build_scam_prompts(obj=obj, atk=atk, variant=variant)
            for p in prompts:
                captions_flat.append(p)
                img_of_text.append(i_idx)

        img_of_text_arr = np.array(img_of_text, dtype=np.int32)

        print(f"[SCAM] {variant}: images={len(pil_images):,}  prompts={len(captions_flat):,}  (per image ~{len(captions_flat)/max(1,len(pil_images)):.1f})")

        print(f"[Tokenize] SCAM {variant} prompts (once, shared across models)...")
        tokens_cpu = clip.tokenize(captions_flat, truncate=True)

        ds_images = PILListImageDataset(pil_images=pil_images, preprocess_fn=preprocess_fn)
        dl_images = _make_loader(
            ds_images,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=((not args.no_pin_memory) and (device == "cuda")),
            persistent_workers=((not args.no_persistent_workers)),
        )

        bundles.append(DatasetBundle(
            dataset_tag=f"SCAM::{variant}",
            dl_images=dl_images,
            tokens_cpu=tokens_cpu,
            img_of_text_arr=img_of_text_arr,
            captions_flat=captions_flat,
            n_images=len(pil_images),
            n_texts=len(captions_flat),
        ))

    return bundles


def _c(sev: str, s: str) -> str:
    """Colorize by severity: red=warning, green=positive, yellow=neutral."""
    if sev == "!!":
        return f"{Style.BRIGHT}{Fore.RED}{s}{Style.RESET_ALL}"
    if sev == "!":
        return f"{Style.BRIGHT}{Fore.MAGENTA}{s}{Style.RESET_ALL}"
    if sev == "+":
        return f"{Style.BRIGHT}{Fore.GREEN}{s}{Style.RESET_ALL}"
    # "~"
    return f"{Style.BRIGHT}{Fore.YELLOW}{s}{Style.RESET_ALL}"

def _pct(delta: float) -> str:
    return f"{delta*100:+.1f}%"

def _safe_rel_change(val: float, base: float, eps: float = 1e-12) -> float:
    return float((val - base) / (abs(base) + eps))

def _format_delta(val: float, base: float) -> str:
    return _pct(_safe_rel_change(val, base))

def _heuristic_flags(row: pd.Series, base: pd.Series) -> List[Tuple[str, str]]:
    """
    Returns list of (severity, message):
      "!!" / "!"  = warning (printed red, bright)
      "~"         = neutral/unclear but notable (yellow, bright)
      "+"         = likely positive (green, bright)
    """
    out: List[Tuple[str, str]] = []

    # 1) Gap metric disagreement (shape vs shift)
    gap_cols = ["gap_center_euclid", "gap_center_cosine", "gap_mahalanobis_diag", "gap_mmd_rbf"]
    rels = []
    for c in gap_cols:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            rels.append(_safe_rel_change(float(row[c]), float(base[c])))
    if len(rels) >= 3:
        spread = float(np.max(rels) - np.min(rels))
        if spread > 0.75:
            out.append(("!!", f"Gap metrics strongly disagree (spread {spread*100:.0f}pp across {gap_cols}) → investigate geometric reshape."))
        elif spread > 0.45:
            out.append(("!", f"Gap metrics disagree (spread {spread*100:.0f}pp across {gap_cols}) → modality gap is not stable as a single scalar."))

    # 2) Effective rank drop severity bins ---
    for c, name in [("geom_image_eff_rank", "image"), ("geom_text_eff_rank", "text")]:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            d = _safe_rel_change(float(row[c]), float(base[c]))
            if d < -0.45:
                out.append(("!!", f"Effective rank ({name}) dropped {_pct(d)} → strong anisotropy/collapse risk."))
            elif d < -0.25:
                out.append(("!", f"Effective rank ({name}) dropped {_pct(d)} → notable anisotropy increase."))
            elif d < -0.15:
                out.append(("~", f"Effective rank ({name}) dropped {_pct(d)} → mild anisotropy increase."))

    # 3) Sphericity drop severity bins ---
    for c, name in [("geom_image_sphericity", "image"), ("geom_text_sphericity", "text")]:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            d = _safe_rel_change(float(row[c]), float(base[c]))
            if d < -0.60:
                out.append(("!!", f"Sphericity ({name}) dropped {_pct(d)} → much less spherical cloud."))
            elif d < -0.30:
                out.append(("!", f"Sphericity ({name}) dropped {_pct(d)} → increased ellipsoid-ness."))
            elif d < -0.18:
                out.append(("~", f"Sphericity ({name}) dropped {_pct(d)} → mild ellipsoid-ness increase."))


    # 4) Cone-ness increase (mean norm)
    for c, name in [("center_norm_image", "||mu_i||"), ("center_norm_text", "||mu_t||")]:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            d = _safe_rel_change(float(row[c]), float(base[c]))
            if d > 0.40:
                out.append(("!", f"Cone-ness increased: {name} {_pct(d)} → stronger mean direction (can indicate concentration/collapse)."))
            elif d < -0.30:
                out.append(("~", f"Cone-ness decreased: {name} {_pct(d)} → more centered; neutral/unclear."))

    # 5) Image↔Text subspace mismatch (principal angles)
    if "subspace_angles_mean" in row and "subspace_angles_mean" in base:
        if np.isfinite(row["subspace_angles_mean"]) and np.isfinite(base["subspace_angles_mean"]):
            delta_deg = float(row["subspace_angles_mean"] - base["subspace_angles_mean"])
            if delta_deg > 12.0:
                out.append(("!", f"Image/Text top-subspace mismatch increased by {delta_deg:+.1f}° (principal angles mean)."))
            elif delta_deg < -10.0:
                out.append(("~", f"Image/Text subspaces became more aligned ({delta_deg:+.1f}°). Neutral/unclear alone."))

    # 6) IT similarity (matched pairs)
    if "sim_it_mean" in row and "sim_it_mean" in base and np.isfinite(row["sim_it_mean"]) and np.isfinite(base["sim_it_mean"]):
        d = _safe_rel_change(float(row["sim_it_mean"]), float(base["sim_it_mean"]))
        if d > 0.08:
            out.append(("+", f"IT cosine increased {_pct(d)} → matched image-text similarity up."))
        elif d < -0.06:
            out.append(("!", f"IT cosine decreased {_pct(d)} → matched alignment down."))

    # 6b) Contradiction: IT↑ but geometry worsens
    has_it_up = any(sev == "+" and "IT cosine increased" in msg for sev, msg in out)
    has_geom_warn = any(sev in ("!!", "!") and ("Effective rank" in msg or "Sphericity" in msg or "Cone-ness increased" in msg) for sev, msg in out)
    if has_it_up and has_geom_warn:
        out.append(("!", "IT improved but geometry flags fired → possible shortcut/collapse rather than true alignment gain."))

    # 7) Similarity distribution shifts (not necessarily good/bad)
    for c, meaning in [
        ("gap_mean_ii_minus_it", "II-IT mean gap"),
        ("gap_mean_tt_minus_it", "TT-IT mean gap"),
        ("dist_jsd_it_tt", "JSD(IT,TT)"),
        ("dist_jsd_it_ii", "JSD(IT,II)"),
    ]:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            d = _safe_rel_change(float(row[c]), float(base[c]))
            if abs(d) > 0.25:
                out.append(("~", f"{meaning} shifted {_pct(d)} → distribution geometry changed; interpret with plots."))

    # 8) Raw norms drift (scale/calibration changed)
    for c, name in [("raw_norm_image_mean", "raw image norm"), ("raw_norm_text_mean", "raw text norm")]:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            d = _safe_rel_change(float(row[c]), float(base[c]))
            if abs(d) > 0.35:
                out.append(("~", f"{name} shifted {_pct(d)} → feature scale changed (may affect temps/thresholds)."))

    # A) mean-direction cosine (center_cos_mu) sign/orthogonality flags
    if "center_cos_mu" in row and "center_cos_mu" in base:
        if np.isfinite(row["center_cos_mu"]) and np.isfinite(base["center_cos_mu"]):
            c  = float(row["center_cos_mu"])
            cb = float(base["center_cos_mu"])

            did_sign_flip = False

            # sign flip: aligned -> anti-aligned
            if (cb > 0.10) and (c < 0.0):
                did_sign_flip = True
                out.append(("!", f"Mean directions flipped sign: cos(mu_i,mu_t) {cb:+.3f} → {c:+.3f} (anti-alignment; neutral/unclear alone, but often meaningful)."))

            # near-orthogonality (only if NOT already captured by sign flip)
            if (not did_sign_flip) and (cb > 0.15) and (abs(c) < 0.05):
                out.append(("!", f"Mean directions became near-orthogonal: cos(mu_i,mu_t) {cb:+.3f} → {c:+.3f}."))


    # B) condition-number flag: absolute + relative
    for c, name in [("geom_image_cond_p95_p05", "image"), ("geom_text_cond_p95_p05", "text")]:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            val = float(row[c])
            basev = float(base[c])
            ratio = float(val / (basev + 1e-12))

            if val > 1e6:
                out.append(("!!", f"Condition proxy ({name}) is {val:.2e} (>1e6) → extreme anisotropy / potential collapse."))
            elif ratio > 8.0:
                out.append(("!!", f"Condition proxy ({name}) exploded ×{ratio:.1f} → extreme anisotropy / potential collapse."))
            elif ratio > 4.0:
                out.append(("!", f"Condition proxy ({name}) increased ×{ratio:.1f} → strong anisotropy increase."))

    for c, name in [("geom_image_top_eig_frac", "image"), ("geom_text_top_eig_frac", "text")]:
        if c in row and c in base and np.isfinite(row[c]) and np.isfinite(base[c]):
            d = _safe_rel_change(float(row[c]), float(base[c]))
            if d > 0.35:
                out.append(("!", f"Top-eigen variance fraction ({name}) rose {_pct(d)} → more mass in a single direction."))

    # C) II-IT inversion flags (qualitative distribution change)
    if "gap_mean_ii_minus_it" in row and "gap_mean_ii_minus_it" in base:
        if np.isfinite(row["gap_mean_ii_minus_it"]) and np.isfinite(base["gap_mean_ii_minus_it"]):
            v = float(row["gap_mean_ii_minus_it"])
            vb = float(base["gap_mean_ii_minus_it"])

            # baseline has II > IT (positive), model has II <= IT (negative): inversion
            if (vb > 0.05) and (v < -0.05):
                out.append(("!!", f"Separation inverted: (II_mean - IT_mean) {vb:+.3f} → {v:+.3f} (IT exceeds II; big qualitative change)."))
            elif abs(v - vb) > 0.25:
                out.append(("~", f"(II_mean - IT_mean) shifted {v - vb:+.3f} → distribution geometry changed."))

    # Optional: if you want a softer version for TT vs IT too:
    if "gap_mean_tt_minus_it" in row and "gap_mean_tt_minus_it" in base:
        if np.isfinite(row["gap_mean_tt_minus_it"]) and np.isfinite(base["gap_mean_tt_minus_it"]):
            v = float(row["gap_mean_tt_minus_it"])
            vb = float(base["gap_mean_tt_minus_it"])
            if (vb > 0.10) and (v < 0.0):
                out.append(("!", f"TT-IT gap crossed below zero: (TT_mean - IT_mean) {vb:+.3f} → {v:+.3f}."))

    return out


def print_mini_healthcheck_vs_pretrained(
    df: pd.DataFrame,
    baseline_alias: str = "pretrained",
    max_lines_per_model: int = 6,
):
    """
    Prints a compact per-dataset report vs baseline:
      - prints ONLY models with at least one flagged item (no "looks similar" spam)
      - colored severities: red=warning, yellow=neutral, green=positive
      - adds clear separators per model block
    """
    if baseline_alias not in set(df["model_alias"].tolist()):
        print(_c("!", f"[MiniCheck] WARNING: baseline '{baseline_alias}' not found; skipping comparisons."))
        return

    sev_order = {"!!": 0, "!": 1, "~": 2, "+": 3}

    print("\n" + "=" * 90)
    print(f"[MiniCheck] Heuristic summary vs '{baseline_alias}' (per dataset)")
    print("=" * 90)

    for ds_tag in sorted(df["dataset"].unique().tolist()):
        dfd = df[df["dataset"] == ds_tag].copy()
        if len(dfd) == 0:
            continue

        base_rows = dfd[dfd["model_alias"] == baseline_alias]
        if len(base_rows) == 0:
            print(_c("!", f"\n[MiniCheck] dataset={ds_tag}: baseline missing; skipping."))
            continue
        base = base_rows.iloc[0]

        # Only print dataset header if we will actually print any model blocks
        # (no spam)
        blocks: List[Tuple[str, List[Tuple[str, str]], str]] = []

        for model_alias in dfd["model_alias"].tolist():
            if model_alias == baseline_alias:
                continue

            r = dfd[dfd["model_alias"] == model_alias].iloc[0]
            flags = _heuristic_flags(r, base)
            if not flags:
                continue

            flags = sorted(flags, key=lambda x: sev_order.get(x[0], 9))[:max_lines_per_model]

            # Anchor deltas: short, high-signal
            it_d = _format_delta(float(r["sim_it_mean"]), float(base["sim_it_mean"])) if ("sim_it_mean" in r and "sim_it_mean" in base and np.isfinite(r["sim_it_mean"]) and np.isfinite(base["sim_it_mean"])) else "n/a"
            er_i_d = _format_delta(float(r["geom_image_eff_rank"]), float(base["geom_image_eff_rank"])) if ("geom_image_eff_rank" in r and "geom_image_eff_rank" in base and np.isfinite(r["geom_image_eff_rank"]) and np.isfinite(base["geom_image_eff_rank"])) else "n/a"
            sph_i_d = _format_delta(float(r["geom_image_sphericity"]), float(base["geom_image_sphericity"])) if ("geom_image_sphericity" in r and "geom_image_sphericity" in base and np.isfinite(r["geom_image_sphericity"]) and np.isfinite(base["geom_image_sphericity"])) else "n/a"

            anchors = f"(anchors: ΔIT={it_d}, Δeffrank_i={er_i_d}, Δsphericity_i={sph_i_d})"
            blocks.append((model_alias, flags, anchors))

        if not blocks:
            continue

        print(f"\n[MiniCheck] dataset={ds_tag}")

        for model_alias, flags, anchors in blocks:
            sep = "-" * 38
            print(_c("~", f"{sep}\nModel: {model_alias}\n{sep}\n{anchors}"))

            for sev, msg in flags:
                print(_c(sev, f"{sev} {msg}"))



def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)

def _sample_pair_cosines(emb: np.ndarray, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """
    emb: [N, D] assumed L2-normalized.
    returns cosine samples of size ~n_pairs from random (i, j), i != j
    """
    N = emb.shape[0]
    if N < 2:
        return np.array([], dtype=np.float32)
    n_pairs = int(min(n_pairs, N * (N - 1) // 2))
    i = rng.integers(0, N, size=n_pairs, endpoint=False)
    j = rng.integers(0, N, size=n_pairs, endpoint=False)
    neq = (i != j)
    i = i[neq]
    j = j[neq]
    cos = np.sum(emb[i] * emb[j], axis=1)
    return cos.astype(np.float32)

def _per_image_caption_pairs_tt(text_emb: np.ndarray, img_of_text: np.ndarray) -> np.ndarray:
    """
    text_emb: [T, D] normalized
    img_of_text: [T] mapping each text idx -> image idx
    Returns TT cosines computed within each image (pairwise among that image's captions).
    """
    tt_vals: List[np.ndarray] = []
    by_img: Dict[int, List[int]] = {}
    for t_idx, i_idx in enumerate(img_of_text.tolist()):
        by_img.setdefault(int(i_idx), []).append(int(t_idx))

    for _, t_idxs in by_img.items():
        if len(t_idxs) < 2:
            continue
        Tm = text_emb[t_idxs]  # [k, D]
        C = Tm @ Tm.T
        k = C.shape[0]
        iu = np.triu_indices(k, k=1)
        tt_vals.append(C[iu].astype(np.float32))

    if not tt_vals:
        return np.array([], dtype=np.float32)
    return np.concatenate(tt_vals, axis=0)

def _fd_bins(values: np.ndarray) -> int:
    v = values[np.isfinite(values)]
    if v.size < 10:
        return 10
    q25, q75 = np.percentile(v, [25, 75])
    iqr = q75 - q25
    if iqr <= 1e-12:
        return 50
    bin_width = 2 * iqr / (v.size ** (1/3))
    if bin_width <= 1e-12:
        return 50
    bins = int(math.ceil((v.max() - v.min()) / bin_width))
    return int(np.clip(bins, 20, 250))

def _hist_prob(values: np.ndarray, bins: int, range_: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=range_, density=False)
    hist = hist.astype(np.float64)
    eps = 1e-12
    p = hist / (hist.sum() + eps)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p

def _mmd_rbf_subset(x: np.ndarray, y: np.ndarray, subset: int, rng: np.random.Generator) -> float:
    nx = x.shape[0]
    ny = y.shape[0]
    if nx < 10 or ny < 10:
        return float("nan")

    m = int(min(subset, nx, ny))
    ix = rng.choice(nx, size=m, replace=False)
    iy = rng.choice(ny, size=m, replace=False)
    X = x[ix].astype(np.float64)
    Y = y[iy].astype(np.float64)

    def sq_dists(A, B):
        AA = np.sum(A*A, axis=1, keepdims=True)
        BB = np.sum(B*B, axis=1, keepdims=True)
        return AA - 2.0 * (A @ B.T) + BB.T

    Dxy = sq_dists(X, Y)
    med = np.median(Dxy)
    if not np.isfinite(med) or med <= 1e-12:
        med = 1.0
    gamma = 1.0 / (2.0 * med)

    Kxx = np.exp(-gamma * sq_dists(X, X))
    Kyy = np.exp(-gamma * sq_dists(Y, Y))
    Kxy = np.exp(-gamma * Dxy)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = (Kxx.sum() / (m * (m - 1))) + (Kyy.sum() / (m * (m - 1))) - (2.0 * Kxy.mean())
    return float(mmd2)

def _uniformity(emb: np.ndarray, n_pairs: int, t: float, rng: np.random.Generator) -> float:
    cos = _sample_pair_cosines(emb, n_pairs=n_pairs, rng=rng)
    if cos.size == 0:
        return float("nan")
    d2 = 2.0 - 2.0 * cos.astype(np.float64)
    val = np.log(np.mean(np.exp(-t * d2)))
    return float(val)

# covariance eigenspectrum + EVR + principal angles
def _cov_eigh(emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    emb: [N,D] normalized; we center inside.
    Returns: mu [D], evals [D] (ascending), evecs [D,D] (columns correspond to evals)
    """
    X = emb.astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)  # ascending
    evals = np.clip(evals, 0.0, None)
    return mu, evals, evecs

def _effective_rank_entropy(evals: np.ndarray) -> float:
    tr = float(evals.sum()) + 1e-12
    p = evals / tr
    H = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(H))

def _participation_ratio(evals: np.ndarray) -> float:
    s1 = float(np.sum(evals)) + 1e-12
    s2 = float(np.sum(evals * evals)) + 1e-12
    return float((s1 * s1) / s2)

def _sphericity(evals: np.ndarray) -> float:
    """
    Sphericity proxy in [0,1] (higher = more spherical):
      (prod e_i)^(1/D) / (mean e_i)
    For degenerate spectra -> near 0.
    """
    D = evals.size
    mean = float(np.mean(evals)) + 1e-12
    # geometric mean with clipping
    gm = float(np.exp(np.mean(np.log(evals + 1e-12))))
    return float(gm / mean)

def _eig_cv(evals: np.ndarray) -> float:
    m = float(np.mean(evals)) + 1e-12
    s = float(np.std(evals))
    return float(s / m)

def _cond_p95_p05(evals: np.ndarray) -> float:
    p05 = float(np.percentile(evals, 5))
    p95 = float(np.percentile(evals, 95))
    return float((p95 + 1e-12) / (p05 + 1e-12))

def _pca_evr_from_evals(evals: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    evals: covariance eigenvalues (ascending).
    returns (evr_topk, cum_evr_topk) in descending component order.
    """
    ev = evals[::-1]  # descending
    tr = float(ev.sum()) + 1e-12
    evr = ev / tr
    topk = int(min(topk, evr.size))
    evr_topk = evr[:topk]
    cum = np.cumsum(evr_topk)
    return evr_topk.astype(np.float64), cum.astype(np.float64)

def _principal_angles_deg(evecs_i: np.ndarray, evecs_t: np.ndarray, k: int) -> np.ndarray:
    """
    evecs_* are [D,D] columns = eigenvectors aligned with ascending evals.
    We take top-k eigenvectors (descending) as subspaces and compute principal angles.
    """
    k = int(min(k, evecs_i.shape[0], evecs_t.shape[0]))
    Ui = evecs_i[:, ::-1][:, :k]  # [D,k]
    Ut = evecs_t[:, ::-1][:, :k]  # [D,k]
    M = Ui.T @ Ut
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    ang = np.degrees(np.arccos(s))
    return ang.astype(np.float64)

def _cos_to_centroid(emb: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu_norm = float(np.linalg.norm(mu)) + 1e-12
    mu_hat = mu / mu_norm
    return (emb @ mu_hat).astype(np.float32)

def _raw_norm_stats(raw: np.ndarray) -> Dict[str, float]:
    n = np.linalg.norm(raw, axis=1)
    return dict(
        raw_norm_mean=float(n.mean()),
        raw_norm_std=float(n.std()),
        raw_norm_p05=float(np.percentile(n, 5)),
        raw_norm_p50=float(np.percentile(n, 50)),
        raw_norm_p95=float(np.percentile(n, 95)),
    )

def _plot_cosine_overlay_hist(ii: np.ndarray, tt: np.ndarray, it: np.ndarray, out_png: str, title: str):
    pooled = np.concatenate([ii, tt, it], axis=0) if (ii.size or tt.size or it.size) else np.array([], dtype=np.float32)
    bins = _fd_bins(pooled) if pooled.size else 50

    plt.figure(figsize=(10, 6))
    if ii.size: plt.hist(ii, bins=bins, alpha=0.45, density=True, label="Image-Image (II)")
    if tt.size: plt.hist(tt, bins=bins, alpha=0.45, density=True, label="Text-Text (TT)")
    if it.size: plt.hist(it, bins=bins, alpha=0.45, density=True, label="Image-Text (IT)")
    plt.title(title)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_cosine_ecdf(ii: np.ndarray, tt: np.ndarray, it: np.ndarray, out_png: str, title: str):
    def ecdf(x):
        x = np.sort(x)
        y = np.linspace(0, 1, num=len(x), endpoint=True)
        return x, y

    plt.figure(figsize=(10, 6))
    for arr, lab in [(ii, "Image-Image (II)"), (tt, "Text-Text (TT)"), (it, "Image-Text (IT)")]:
        if arr.size:
            x, y = ecdf(arr)
            plt.plot(x, y, label=lab)
    plt.title(title)
    plt.xlabel("Cosine similarity")
    plt.ylabel("ECDF")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_cosine_violin(ii: np.ndarray, tt: np.ndarray, it: np.ndarray, out_png: str, title: str):
    data = []
    labels = []
    if ii.size:
        data.append(ii); labels.append("II")
    if tt.size:
        data.append(tt); labels.append("TT")
    if it.size:
        data.append(it); labels.append("IT")

    plt.figure(figsize=(8, 6))
    if data:
        plt.violinplot(data, showmeans=True, showextrema=True)
        plt.xticks(range(1, len(labels) + 1), labels)
    plt.title(title)
    plt.ylabel("Cosine similarity")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_summary_bars(df, metric_cols: List[str], out_png: str, title: str):
    aliases = df["model_alias"].tolist()
    x = np.arange(len(aliases))

    plt.figure(figsize=(max(10, 1.2 * len(aliases)), 6))
    width = 0.8 / max(1, len(metric_cols))

    for k, col in enumerate(metric_cols):
        vals = df[col].to_numpy(dtype=np.float64)
        plt.bar(x + (k - (len(metric_cols) - 1) / 2) * width, vals, width=width, label=col)

    plt.xticks(x, aliases, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_pca_evr(evr_i: np.ndarray, evr_t: np.ndarray, evr_p: np.ndarray,
                  out_png: str, title: str):
    plt.figure(figsize=(10, 6))
    for evr, lab in [(evr_i, "Image"), (evr_t, "Text"), (evr_p, "Pooled")]:
        if evr.size:
            plt.plot(np.arange(1, evr.size + 1), np.cumsum(evr), label=f"{lab} cumulative EVR")
    plt.title(title)
    plt.xlabel("PC index")
    plt.ylabel("Cumulative explained variance ratio")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_eig_spectrum(evals_i: np.ndarray, evals_t: np.ndarray, out_png: str, title: str):
    def downsample(evals):
        ev = evals[::-1]  # descending
        if ev.size <= EIG_SPECTRUM_POINTS:
            return ev
        idx = np.linspace(0, ev.size - 1, EIG_SPECTRUM_POINTS).astype(np.int64)
        return ev[idx]

    ei = downsample(evals_i)
    et = downsample(evals_t)

    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(ei + 1e-12), label="Image log10(eig)")
    plt.plot(np.log10(et + 1e-12), label="Text log10(eig)")
    plt.title(title)
    plt.xlabel("Eigen index (downsampled, descending)")
    plt.ylabel("log10(eigenvalue)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_principal_angles(angles_deg: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, angles_deg.size + 1), angles_deg, marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("Component index (1..k)")
    plt.ylabel("Principal angle (degrees)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_cos_to_centroid(ii: np.ndarray, tt: np.ndarray, out_png: str, title: str):
    # ii = cos(image, mu_i_hat), tt = cos(text, mu_t_hat)
    plt.figure(figsize=(10, 6))
    plt.hist(ii, bins=COS_TO_CENTROID_BINS, alpha=0.5, density=True, label="Image cos(x, mu_i_hat)")
    plt.hist(tt, bins=COS_TO_CENTROID_BINS, alpha=0.5, density=True, label="Text cos(x, mu_t_hat)")
    plt.title(title)
    plt.xlabel("cos(x, centroid_dir)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_pca2_pooled(img_emb: np.ndarray, txt_emb: np.ndarray, out_png: str, title: str,
                      max_i: int = 4000, max_t: int = 8000, seed: int = 123):
    rng = np.random.default_rng(seed)
    Ni = img_emb.shape[0]
    Nt = txt_emb.shape[0]
    ii = rng.choice(Ni, size=min(Ni, max_i), replace=False)
    tt = rng.choice(Nt, size=min(Nt, max_t), replace=False)

    Xi = img_emb[ii]
    Xt = txt_emb[tt]
    X = np.vstack([Xi, Xt]).astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T

    Zi = Z[:Xi.shape[0]]
    Zt = Z[Xi.shape[0]:]

    plt.figure(figsize=(10, 7))
    plt.scatter(Zt[:, 0], Zt[:, 1], alpha=0.25, s=8, label="Text")
    plt.scatter(Zi[:, 0], Zi[:, 1], alpha=0.35, s=10, label="Image")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _dump_pca3_views(img_emb: np.ndarray, txt_emb: np.ndarray, out_dir: str, prefix: str, title: str,
                     max_i: int = PCA3D_MAX_POINTS_IMAGE, max_t: int = PCA3D_MAX_POINTS_TEXT,
                     n_views: int = PCA3D_VIEWS, seed: int = 123):
    from mpl_toolkits.mplot3d import Axes3D

    rng = np.random.default_rng(seed)
    Ni = img_emb.shape[0]
    Nt = txt_emb.shape[0]
    ii = rng.choice(Ni, size=min(Ni, max_i), replace=False)
    tt = rng.choice(Nt, size=min(Nt, max_t), replace=False)

    Xi = img_emb[ii].astype(np.float64)
    Xt = txt_emb[tt].astype(np.float64)
    X = np.vstack([Xi, Xt])
    Xc = X - X.mean(axis=0, keepdims=True)

    # PCA (3D) via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:3].T

    Zi = Z[:Xi.shape[0]]
    Zt = Z[Xi.shape[0]:]

    # Fixed axis limits for all views
    mins = Z.min(axis=0)
    maxs = Z.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-12)
    mins -= pad
    maxs += pad

    # 18 views over 360 degrees
    azims = np.linspace(0, 360, num=n_views, endpoint=False)

    for k, az in enumerate(azims):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # White background, keep axes stable
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        ax.scatter(Zt[:, 0], Zt[:, 1], Zt[:, 2], alpha=0.25, s=8, label="Text")
        ax.scatter(Zi[:, 0], Zi[:, 1], Zi[:, 2], alpha=0.35, s=10, label="Image")

        ax.set_title(f"{title} | view {k+1:02d}/{n_views:02d} az={az:.1f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        # Camera motion: rotate azimuth, keep elevation fixed
        ax.view_init(elev=20, azim=float(az))

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)

        out_png = os.path.join(out_dir, f"{prefix}__pca3_view_{k+1:02d}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close(fig)


# ============================================================
# MAIN EVAL
# ============================================================
@torch.inference_mode()
def _encode_images_raw_and_norm(model, dataloader: DataLoader, device_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      raw_feats: [N,D] float32 (unnormalized)
      emb:       [N,D] float32 (L2-normalized)
    """
    raw_list = []
    emb_list = []
    for images, _idx in tqdm(dataloader, desc="Encode images", ncols=90, leave=False):
        if device_str == "cuda":
            images = images.to(device_str, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = model.encode_image(images)
        else:
            images = images.to(device_str)
            x = model.encode_image(images)

        x = x.float()
        raw = x.detach().cpu().numpy().astype(np.float32, copy=False)

        x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        emb = x.detach().cpu().numpy().astype(np.float32, copy=False)

        raw_list.append(raw)
        emb_list.append(emb)

    raw_feats = np.concatenate(raw_list, axis=0)
    emb = np.concatenate(emb_list, axis=0)
    return raw_feats, emb

@torch.inference_mode()
def _encode_texts_raw_and_norm(model, tokens_cpu: torch.Tensor, device_str: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    raw_list = []
    emb_list = []
    n = tokens_cpu.shape[0]
    for i in tqdm(range(0, n, batch_size), desc="Encode texts", ncols=90, leave=False):
        tok = tokens_cpu[i:i + batch_size].to(device_str, non_blocking=(device_str == "cuda"))
        if device_str == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = model.encode_text(tok)
        else:
            x = model.encode_text(tok)

        x = x.float()
        raw = x.detach().cpu().numpy().astype(np.float32, copy=False)

        x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        emb = x.detach().cpu().numpy().astype(np.float32, copy=False)

        raw_list.append(raw)
        emb_list.append(emb)

    raw_feats = np.concatenate(raw_list, axis=0)
    emb = np.concatenate(emb_list, axis=0)
    return raw_feats, emb


def main():
    args = parse_arguments()

    out_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    pin_memory = (not args.no_pin_memory) and (device == "cuda")
    persistent_workers = (not args.no_persistent_workers)

    rng = np.random.default_rng(42)

    print(f"[Device] {device}")
    print(f"[Out]  out_dir={out_dir}")
    print(f"[DL] batch={args.batch_size} workers={args.num_workers} pin={pin_memory} prefetch={args.prefetch_factor} persistent={persistent_workers}")

    # Load ONE preprocess (shared by all datasets, all models)
    base_model_ref = MODELS[0][1] if len(MODELS) else "ViT-L/14"
    _tmp_model, preprocess_fn = load_clip_model(base_model_ref, device_str=device)
    del _tmp_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Build dataset bundles (tokenized once per dataset, shared across models)
    bundles: List[DatasetBundle] = []
    bundles.append(build_flickr_bundle(args, preprocess_fn=preprocess_fn))
    bundles.extend(build_scam_bundles(args, preprocess_fn=preprocess_fn))

    if not bundles:
        raise SystemExit("No datasets available (check Flickr path / HF datasets install).")

    # Evaluate: per model x per dataset
    all_rows: List[Dict[str, float]] = []

    for model_alias, model_path in MODELS:
        print("\n" + "=" * 90)
        print(f"[Model] {model_alias} :: {model_path}")
        print("=" * 90)

        model, _ = load_clip_model(model_path, device_str=device)

        for bundle in bundles:
            tag = bundle.dataset_tag
            print("\n" + "-" * 90)
            print(f"[Dataset] {tag}  (images={bundle.n_images:,} texts={bundle.n_texts:,})")
            print("-" * 90)

            # Per model + dataset output folder
            per_model_dir = os.path.join(out_dir, model_alias, tag.replace("::", "_"))
            os.makedirs(per_model_dir, exist_ok=True)

            # Encode
            print("[Encode] images...")
            img_raw, img_emb = _encode_images_raw_and_norm(model, bundle.dl_images, device_str=device)
            print("[Encode] texts...")
            txt_raw, txt_emb = _encode_texts_raw_and_norm(model, tokens_cpu=bundle.tokens_cpu, device_str=device, batch_size=TEXT_BATCH_SIZE)

            img_of_text_arr = bundle.img_of_text_arr

            # Sanity
            img_emb = img_emb.astype(np.float32, copy=False)
            txt_emb = txt_emb.astype(np.float32, copy=False)
            img_raw = img_raw.astype(np.float32, copy=False)
            txt_raw = txt_raw.astype(np.float32, copy=False)

            N, D = img_emb.shape
            Tn, Dt = txt_emb.shape
            assert D == Dt, f"Dim mismatch image D={D} vs text D={Dt}"

            # Original modality gap: Euclid between means (normalized space)
            mu_i = img_emb.mean(axis=0)
            mu_t = txt_emb.mean(axis=0)
            gap_center_euclid = float(np.linalg.norm(mu_i - mu_t))

            mu_i_hat = mu_i / (np.linalg.norm(mu_i) + 1e-12)
            mu_t_hat = mu_t / (np.linalg.norm(mu_t) + 1e-12)
            cos_mu = float(np.dot(mu_i_hat, mu_t_hat))
            gap_center_cosine = float(1.0 - cos_mu)

            var_i = img_emb.var(axis=0) + 1e-6
            var_t = txt_emb.var(axis=0) + 1e-6
            pooled = 0.5 * (var_i + var_t)
            gap_mahalanobis_diag = float(np.sqrt(np.sum(((mu_i - mu_t) ** 2) / pooled)))

            gap_frechet_diag = float(
                np.sum((mu_i - mu_t) ** 2) + np.sum(var_i + var_t - 2.0 * np.sqrt(var_i * var_t))
            )

            gap_mmd_rbf = _mmd_rbf_subset(img_emb, txt_emb, subset=MMD_SUBSET, rng=rng)


            # Similarity distributions: IT (matched), TT (within-image), II (sampled)
            it_cos = np.sum(img_emb[img_of_text_arr] * txt_emb, axis=1).astype(np.float32)
            tt_cos_within = _per_image_caption_pairs_tt(txt_emb, img_of_text_arr)
            tt_cos_global = _sample_pair_cosines(txt_emb, n_pairs=N_TEXT_TEXT_PAIRS_GLOBAL, rng=rng)
            tt_cos = tt_cos_within

            ii_cos = _sample_pair_cosines(img_emb, n_pairs=N_IMAGE_IMAGE_PAIRS, rng=rng)

            it_mean, it_std = float(it_cos.mean()), float(it_cos.std())
            tt_mean, tt_std = (float(tt_cos.mean()), float(tt_cos.std())) if tt_cos.size else (float("nan"), float("nan"))
            ii_mean, ii_std = (float(ii_cos.mean()), float(ii_cos.std())) if ii_cos.size else (float("nan"), float("nan"))

            gap_mean_ii_minus_it = float(ii_mean - it_mean) if np.isfinite(ii_mean) else float("nan")
            gap_mean_tt_minus_it = float(tt_mean - it_mean) if np.isfinite(tt_mean) else float("nan")

            pooled_for_bins = np.concatenate([ii_cos, tt_cos if tt_cos.size else tt_cos_global, it_cos], axis=0)
            bins = _fd_bins(pooled_for_bins) if pooled_for_bins.size else 50

            p_ii = _hist_prob(ii_cos, bins=bins) if ii_cos.size else None
            p_tt = _hist_prob(tt_cos, bins=bins) if tt_cos.size else _hist_prob(tt_cos_global, bins=bins)
            p_it = _hist_prob(it_cos, bins=bins)

            jsd_it_tt = float(jensenshannon(p_it, p_tt)) if (p_it is not None and p_tt is not None) else float("nan")
            jsd_it_ii = float(jensenshannon(p_it, p_ii)) if (p_it is not None and p_ii is not None) else float("nan")
            jsd_tt_ii = float(jensenshannon(p_tt, p_ii)) if (p_tt is not None and p_ii is not None) else float("nan")

            wass_it_tt = float(wasserstein_distance(it_cos, tt_cos if tt_cos.size else tt_cos_global))
            wass_it_ii = float(wasserstein_distance(it_cos, ii_cos)) if ii_cos.size else float("nan")
            wass_tt_ii = float(wasserstein_distance(tt_cos if tt_cos.size else tt_cos_global, ii_cos)) if ii_cos.size else float("nan")

            ks_it_tt = float(ks_2samp(it_cos, tt_cos if tt_cos.size else tt_cos_global).statistic)
            ks_it_ii = float(ks_2samp(it_cos, ii_cos).statistic) if ii_cos.size else float("nan")
            ks_tt_ii = float(ks_2samp(tt_cos if tt_cos.size else tt_cos_global, ii_cos).statistic) if ii_cos.size else float("nan")

            alignment_it = float(np.mean(2.0 - 2.0 * it_cos.astype(np.float64)))
            uniformity_i = _uniformity(img_emb, n_pairs=UNIFORMITY_PAIRS, t=UNIFORMITY_T, rng=rng)
            uniformity_t = _uniformity(txt_emb, n_pairs=UNIFORMITY_PAIRS, t=UNIFORMITY_T, rng=rng)

            # --------------------------------------------------------
            # Richer geometry diagnostics (cone/sphere/ellipsoid)
            # Compute covariance eigens + centroid-direction distributions
            # --------------------------------------------------------
            mu_i2, evals_i, evecs_i = _cov_eigh(img_emb)
            mu_t2, evals_t, evecs_t = _cov_eigh(txt_emb)

            # “Cone-ness”
            cone_R_i = float(np.linalg.norm(mu_i))
            cone_R_t = float(np.linalg.norm(mu_t))

            # vMF-ish concentration proxy (rough)
            # For large D, kappa ≈ (R*(D - R^2)) / (1 - R^2)  (heuristic)
            def kappa_approx(R, D):
                R = float(np.clip(R, 1e-6, 1 - 1e-6))
                return float((R * (D - R * R)) / (1.0 - R * R))

            kappa_i = kappa_approx(cone_R_i, D)
            kappa_t = kappa_approx(cone_R_t, D)

            # Ellipsoid-ness / anisotropy summaries
            eff_rank_i = _effective_rank_entropy(evals_i)
            eff_rank_t = _effective_rank_entropy(evals_t)
            part_ratio_i = _participation_ratio(evals_i)
            part_ratio_t = _participation_ratio(evals_t)
            sphericity_i = _sphericity(evals_i)
            sphericity_t = _sphericity(evals_t)
            eig_cv_i = _eig_cv(evals_i)
            eig_cv_t = _eig_cv(evals_t)
            cond_i = _cond_p95_p05(evals_i)
            cond_t = _cond_p95_p05(evals_t)
            top_frac_i = float(evals_i[-1] / (evals_i.sum() + 1e-12))
            top_frac_t = float(evals_t[-1] / (evals_t.sum() + 1e-12))
            trace_i = float(evals_i.sum())
            trace_t = float(evals_t.sum())

            mean_dist_i = float(np.linalg.norm(img_emb - mu_i[None, :], axis=1).mean())
            mean_dist_t = float(np.linalg.norm(txt_emb - mu_t[None, :], axis=1).mean())

            # PCA EVR
            evr_i, cum_i = _pca_evr_from_evals(evals_i, topk=PCA_TOPK)
            evr_t, cum_t = _pca_evr_from_evals(evals_t, topk=PCA_TOPK)

            # Pooled EVR (cheap via covariance of pooled)
            Xp = np.vstack([img_emb, txt_emb]).astype(np.float64)
            mu_p = Xp.mean(axis=0)
            Xpc = Xp - mu_p[None, :]
            Cp = (Xpc.T @ Xpc) / max(1, Xpc.shape[0] - 1)
            evals_p = np.clip(np.linalg.eigvalsh(Cp), 0.0, None)
            evr_p, cum_p = _pca_evr_from_evals(evals_p, topk=PCA_TOPK)

            # Principal angles between top-k covariance eigenspaces (image vs text)
            angles_deg = _principal_angles_deg(evecs_i, evecs_t, k=PRINCIPAL_ANGLES_K)
            angles_mean = float(np.mean(angles_deg))
            angles_med = float(np.median(angles_deg))
            angles_min = float(np.min(angles_deg))
            angles_max = float(np.max(angles_deg))

            # cos(x, centroid_dir) distributions
            cos_to_mu_i = _cos_to_centroid(img_emb, mu_i)
            cos_to_mu_t = _cos_to_centroid(txt_emb, mu_t)

            # Raw feature norms
            raw_i = _raw_norm_stats(img_raw)
            raw_t = _raw_norm_stats(txt_raw)


            # PRINT block (per model + dataset)
            print(f"[Gap:orig] gap_center_euclid                = {gap_center_euclid:.6f}")
            print(f"[Gap:add ] gap_center_cosine                = {gap_center_cosine:.6f}")
            print(f"[Gap:add ] gap_mahalanobis_diag             = {gap_mahalanobis_diag:.6f}")
            print(f"[Gap:add ] gap_frechet_diag                 = {gap_frechet_diag:.6f}")
            print(f"[Gap:add ] gap_mmd_rbf                      = {gap_mmd_rbf:.6f}")
            print(f"[Sims] IT mean={it_mean:.4f} std={it_std:.4f} | TT(mean within)={tt_mean:.4f} std={tt_std:.4f} | II mean={ii_mean:.4f} std={ii_std:.4f}")
            print(f"[Gap:sims] gap_mean_ii_minus_it             = {gap_mean_ii_minus_it:.6f}")
            print(f"[Gap:sims] gap_mean_tt_minus_it             = {gap_mean_tt_minus_it:.6f}")
            print(f"[Dist] JSD IT-TT={jsd_it_tt:.4f} IT-II={jsd_it_ii:.4f} TT-II={jsd_tt_ii:.4f}")
            print(f"[Dist] WAS IT-TT={wass_it_tt:.4f} IT-II={wass_it_ii:.4f} TT-II={wass_tt_ii:.4f}")
            print(f"[Dist] KS  IT-TT={ks_it_tt:.4f} IT-II={ks_it_ii:.4f} TT-II={ks_tt_ii:.4f}")
            print(f"[Align/Uni] alignment_it={alignment_it:.6f} uniformity_i={uniformity_i:.6f} uniformity_t={uniformity_t:.6f}")
            print(f"[Centers] ||mu_i||={cone_R_i:.6f}  ||mu_t||={cone_R_t:.6f}  cos(mu_i,mu_t)={cos_mu:.6f}")
            print(f"[RawNorm:I] mean={raw_i['raw_norm_mean']:.4f} std={raw_i['raw_norm_std']:.4f} p05={raw_i['raw_norm_p05']:.4f} p50={raw_i['raw_norm_p50']:.4f} p95={raw_i['raw_norm_p95']:.4f}")
            print(f"[RawNorm:T] mean={raw_t['raw_norm_mean']:.4f} std={raw_t['raw_norm_std']:.4f} p05={raw_t['raw_norm_p05']:.4f} p50={raw_t['raw_norm_p50']:.4f} p95={raw_t['raw_norm_p95']:.4f}")
            print(f"[Geom:I] cone_R={cone_R_i:.6f} kappa~={kappa_i:.2f} sphericity={sphericity_i:.4f} part_ratio={part_ratio_i:.2f} eff_rank(ent)={eff_rank_i:.2f}")
            print(f"[Geom:I] trace={trace_i:.4f} top_eig_frac={top_frac_i:.4f} eig_cv={eig_cv_i:.4f} cond(p95/p05)={cond_i:.2f} mean||x-mu||={mean_dist_i:.4f}")
            print(f"[Geom:T] cone_R={cone_R_t:.6f} kappa~={kappa_t:.2f} sphericity={sphericity_t:.4f} part_ratio={part_ratio_t:.2f} eff_rank(ent)={eff_rank_t:.2f}")
            print(f"[Geom:T] trace={trace_t:.4f} top_eig_frac={top_frac_t:.4f} eig_cv={eig_cv_t:.4f} cond(p95/p05)={cond_t:.2f} mean||x-mu||={mean_dist_t:.4f}")
            print(f"[Subspace] principal_angles_deg k={PRINCIPAL_ANGLES_K}: mean={angles_mean:.2f} med={angles_med:.2f} min={angles_min:.2f} max={angles_max:.2f}")


            # Save per-model+dataset metrics row
            row = dict(
                model_alias=model_alias,
                model_path=model_path,
                dataset=tag,
                n_images=int(N),
                n_texts=int(Tn),
                dim=int(D),

                gap_center_euclid=gap_center_euclid,
                gap_center_cosine=gap_center_cosine,
                gap_mahalanobis_diag=gap_mahalanobis_diag,
                gap_frechet_diag=gap_frechet_diag,
                gap_mmd_rbf=gap_mmd_rbf,

                sim_it_mean=it_mean,
                sim_it_std=it_std,
                sim_tt_mean=tt_mean,
                sim_tt_std=tt_std,
                sim_ii_mean=ii_mean,
                sim_ii_std=ii_std,

                sim_tt_global_mean=float(tt_cos_global.mean()) if tt_cos_global.size else float("nan"),
                sim_tt_global_std=float(tt_cos_global.std()) if tt_cos_global.size else float("nan"),

                gap_mean_ii_minus_it=gap_mean_ii_minus_it,
                gap_mean_tt_minus_it=gap_mean_tt_minus_it,

                dist_jsd_it_tt=jsd_it_tt,
                dist_jsd_it_ii=jsd_it_ii,
                dist_jsd_tt_ii=jsd_tt_ii,

                dist_wass_it_tt=wass_it_tt,
                dist_wass_it_ii=wass_it_ii,
                dist_wass_tt_ii=wass_tt_ii,

                dist_ks_it_tt=ks_it_tt,
                dist_ks_it_ii=ks_it_ii,
                dist_ks_tt_ii=ks_tt_ii,

                align_it_l2=alignment_it,
                unif_image=uniformity_i,
                unif_text=uniformity_t,

                center_norm_image=cone_R_i,
                center_norm_text=cone_R_t,
                center_cos_mu=cos_mu,

                raw_norm_image_mean=raw_i["raw_norm_mean"],
                raw_norm_image_std=raw_i["raw_norm_std"],
                raw_norm_text_mean=raw_t["raw_norm_mean"],
                raw_norm_text_std=raw_t["raw_norm_std"],

                geom_image_kappa=kappa_i,
                geom_text_kappa=kappa_t,
                geom_image_sphericity=sphericity_i,
                geom_text_sphericity=sphericity_t,
                geom_image_participation_ratio=part_ratio_i,
                geom_text_participation_ratio=part_ratio_t,
                geom_image_eff_rank=eff_rank_i,
                geom_text_eff_rank=eff_rank_t,
                geom_image_top_eig_frac=top_frac_i,
                geom_text_top_eig_frac=top_frac_t,
                geom_image_eig_cv=eig_cv_i,
                geom_text_eig_cv=eig_cv_t,
                geom_image_cond_p95_p05=cond_i,
                geom_text_cond_p95_p05=cond_t,
                geom_image_mean_dist_to_mu=mean_dist_i,
                geom_text_mean_dist_to_mu=mean_dist_t,

                subspace_angles_mean=angles_mean,
                subspace_angles_median=angles_med,
                subspace_angles_min=angles_min,
                subspace_angles_max=angles_max,

                pca_cumEVR10_image=float(np.cumsum(evr_i)[min(9, evr_i.size-1)]) if evr_i.size else float("nan"),
                pca_cumEVR10_text=float(np.cumsum(evr_t)[min(9, evr_t.size-1)]) if evr_t.size else float("nan"),
                pca_cumEVR10_pooled=float(np.cumsum(evr_p)[min(9, evr_p.size-1)]) if evr_p.size else float("nan"),
            )
            all_rows.append(row)


            # Plots per model+dataset (ALWAYS include II/TT/IT)
            _plot_cosine_overlay_hist(
                ii=ii_cos,
                tt=tt_cos if tt_cos.size else tt_cos_global,
                it=it_cos,
                out_png=os.path.join(per_model_dir, f"{model_alias}__cosine_density_overlay.png"),
                title=f"{model_alias} | {tag}: Cosine similarities (II vs TT vs IT) [density]",
            )
            _plot_cosine_ecdf(
                ii=ii_cos,
                tt=tt_cos if tt_cos.size else tt_cos_global,
                it=it_cos,
                out_png=os.path.join(per_model_dir, f"{model_alias}__cosine_ecdf.png"),
                title=f"{model_alias} | {tag}: Cosine similarities (II vs TT vs IT) [ECDF]",
            )
            _plot_cosine_violin(
                ii=ii_cos,
                tt=tt_cos if tt_cos.size else tt_cos_global,
                it=it_cos,
                out_png=os.path.join(per_model_dir, f"{model_alias}__cosine_violin.png"),
                title=f"{model_alias} | {tag}: Cosine similarities (II/TT/IT) [violin]",
            )

            _plot_pca_evr(
                evr_i=evr_i,
                evr_t=evr_t,
                evr_p=evr_p,
                out_png=os.path.join(per_model_dir, f"{model_alias}__pca_cumEVR.png"),
                title=f"{model_alias} | {tag}: PCA cumulative EVR (Image/Text/Pooled)",
            )
            _plot_eig_spectrum(
                evals_i=evals_i,
                evals_t=evals_t,
                out_png=os.path.join(per_model_dir, f"{model_alias}__eig_spectrum_log.png"),
                title=f"{model_alias} | {tag}: Covariance eigen spectrum (log10)",
            )
            _plot_principal_angles(
                angles_deg=angles_deg,
                out_png=os.path.join(per_model_dir, f"{model_alias}__principal_angles_deg.png"),
                title=f"{model_alias} | {tag}: Principal angles (Image vs Text subspace, k={PRINCIPAL_ANGLES_K})",
            )
            _plot_cos_to_centroid(
                ii=cos_to_mu_i,
                tt=cos_to_mu_t,
                out_png=os.path.join(per_model_dir, f"{model_alias}__cos_to_centroid.png"),
                title=f"{model_alias} | {tag}: cos(x, centroid_dir) distribution",
            )

            _plot_pca2_pooled(
                img_emb=img_emb,
                txt_emb=txt_emb,
                out_png=os.path.join(per_model_dir, f"{model_alias}__pca2_pooled.png"),
                title=f"{model_alias} | {tag}: PCA2 pooled (downsampled)",
            )
            _dump_pca3_views(
                img_emb=img_emb,
                txt_emb=txt_emb,
                out_dir=per_model_dir,
                prefix=f"{model_alias}__{tag.replace('::','_')}",
                title=f"{model_alias} | {tag}: PCA3 pooled",
            )

            # Optional TSNE (slow per model/dataset)
            do_tsne = bool(args.do_tsne)
            if do_tsne:
                from sklearn.manifold import TSNE
                rng_ts = np.random.default_rng(123)
                n_t = min(Tn, TSNE_MAX_POINTS_TEXT)
                n_i = min(N, TSNE_MAX_POINTS_IMAGE)
                ti = rng_ts.choice(Tn, size=n_t, replace=False)
                ii = rng_ts.choice(N, size=n_i, replace=False)

                X = np.vstack([txt_emb[ti], img_emb[ii]]).astype(np.float32)
                labels = np.array([0] * n_t + [1] * n_i)

                tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                X2 = tsne.fit_transform(X)

                plt.figure(figsize=(10, 7))
                plt.scatter(X2[labels == 0, 0], X2[labels == 0, 1], alpha=0.35, label="Text", s=8)
                plt.scatter(X2[labels == 1, 0], X2[labels == 1, 1], alpha=0.35, label="Image", s=8)
                plt.title(f"{model_alias} | {tag}: t-SNE (downsampled) text vs image")
                plt.legend()
                plt.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.savefig(os.path.join(per_model_dir, f"{model_alias}__tsne_text_image.png"), dpi=160)
                plt.close()

            # Save per-model+dataset metrics JSON
            with open(os.path.join(per_model_dir, f"{model_alias}__metrics.json"), "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

        # Cleanup model per alias
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save summary CSV + summary plots across models (per dataset)
    import pandas as pd
    df = pd.DataFrame(all_rows)
    summary_csv = os.path.join(out_dir, "summary_metrics_all.csv")
    df.to_csv(summary_csv, index=False)
    print("\n" + "#" * 90)
    print("[SUMMARY] saved:", summary_csv)
    print("#" * 90)

    print_mini_healthcheck_vs_pretrained(df, baseline_alias="pretrained", max_lines_per_model=15)

    # Make dataset-specific summary plots
    datasets = sorted(df["dataset"].unique().tolist())
    for ds_tag in datasets:
        dfd = df[df["dataset"] == ds_tag].copy()
        if len(dfd) == 0:
            continue
        ds_safe = ds_tag.replace("::", "_")
        out_ds = os.path.join(out_dir, f"summary__{ds_safe}")
        os.makedirs(out_ds, exist_ok=True)

        # print in MODELS order
        print(f"\n[SUMMARY] dataset={ds_tag}")
        order = {a: i for i, (a, _) in enumerate(MODELS)}
        dfd["__order"] = dfd["model_alias"].map(order).fillna(1e9).astype(int)
        dfd = dfd.sort_values("__order")

        for _, r in dfd.iterrows():
            print(f"{str(r['model_alias']):>16s}  gap_euclid={float(r['gap_center_euclid']):.6f}  "
                  f"IT={float(r['sim_it_mean']):.4f}  TT={float(r['sim_tt_mean']):.4f}  II={float(r['sim_ii_mean']):.4f}  "
                  f"sphericity_i={float(r['geom_image_sphericity']):.4f}  effrank_i={float(r['geom_image_eff_rank']):.2f}")

        # summary plots
        _plot_summary_bars(
            dfd,
            metric_cols=["gap_center_euclid", "gap_center_cosine", "gap_mahalanobis_diag", "gap_mmd_rbf"],
            out_png=os.path.join(out_ds, "gap_metrics.png"),
            title=f"{ds_tag}: Modality gap metrics across models",
        )
        _plot_summary_bars(
            dfd,
            metric_cols=["sim_ii_mean", "sim_tt_mean", "sim_it_mean"],
            out_png=os.path.join(out_ds, "sim_means.png"),
            title=f"{ds_tag}: Cosine similarity means across models (II/TT/IT)",
        )
        _plot_summary_bars(
            dfd,
            metric_cols=["center_norm_image", "center_norm_text", "center_cos_mu"],
            out_png=os.path.join(out_ds, "centers.png"),
            title=f"{ds_tag}: Center norms and mean-direction alignment",
        )
        _plot_summary_bars(
            dfd,
            metric_cols=["geom_image_sphericity", "geom_text_sphericity"],
            out_png=os.path.join(out_ds, "sphericity.png"),
            title=f"{ds_tag}: Sphericity proxy (higher=more spherical)",
        )
        _plot_summary_bars(
            dfd,
            metric_cols=["geom_image_eff_rank", "geom_text_eff_rank"],
            out_png=os.path.join(out_ds, "eff_rank.png"),
            title=f"{ds_tag}: Effective rank (entropy-based)",
        )
        _plot_summary_bars(
            dfd,
            metric_cols=["subspace_angles_mean", "subspace_angles_median"],
            out_png=os.path.join(out_ds, "principal_angles_summary.png"),
            title=f"{ds_tag}: Image/Text subspace mismatch (principal angles)",
        )

        # delta-to-baseline (if oai_vitl present)
        if "oai_vitl" in set(dfd["model_alias"].tolist()):
            base = dfd[dfd["model_alias"] == "oai_vitl"].iloc[0]
            for col in [
                "gap_center_euclid", "gap_center_cosine",
                "sim_it_mean", "sim_tt_mean", "sim_ii_mean",
                "geom_image_sphericity", "geom_text_sphericity",
                "geom_image_eff_rank", "geom_text_eff_rank",
                "center_norm_image", "center_norm_text",
                "subspace_angles_mean",
            ]:
                dfd[f"delta_{col}"] = dfd[col] - float(base[col])

            _plot_summary_bars(
                dfd,
                metric_cols=[
                    "delta_gap_center_euclid",
                    "delta_gap_center_cosine",
                    "delta_sim_it_mean",
                    "delta_geom_image_sphericity",
                    "delta_subspace_angles_mean",
                ],
                out_png=os.path.join(out_ds, "deltas_vs_oai_vitl.png"),
                title=f"{ds_tag}: Deltas vs oai_vitl (baseline)",
            )

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()