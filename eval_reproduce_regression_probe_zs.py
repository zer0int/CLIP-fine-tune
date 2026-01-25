"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

SCAM / SynthSCAM robustness + representation-geometry probes for CLIP models.
+ ImageNet/ObjectNet MVT - a very hard benchmark for zero-shot.

Dataset download (free, no sign-up): https://objectnet.dev/mvt/
- 5k labels as used here already included with this repo

This script evaluates multiple image-side representations derived from intermediate ViT token states
on the BLISS-e-V/SCAM dataset (variants: NoSCAM, SCAM, SynthSCAM) using a 2-way forced-choice
CLIP scoring setup (object-label prompt vs. attack-word prompt).

Image-side representations (all in CLIP’s final image embedding space) include:
  - CLS: final-layer CLS embedding (standard CLIP image embedding).
  - PATCH-L23: pooled non-register patch-token embedding from layer 23
    (mean over patches with token_norm < REG_THRESHOLD, with fallback to mean over all patches if empty).
  - REG-L23-*PC*: pooled "register-like" patch-token embedding from layer 23
    (mean over patches with token_norm >= REG_THRESHOLD), optionally mean-centered and with top principal
    components removed (PC subtraction) to suppress dominant nuisance modes.
    PC subtraction is performed in the pre-ln_post hidden space (width), then ln_post + projection is applied.
  - PATCHΔ: normalized difference between mean-centered, PC-filtered PATCH embeddings at layers 23 and 22.

Reported metrics include:
  - Accuracy and probability margins for each representation under forced-choice scoring.
  - Pairwise cosine alignments: cos(CLS, PATCH-L23), cos(CLS, REG-L23), cos(PATCH-L23, REG-L23).
    REG-related cosines are defined only for samples that contain at least one REG-like patch.
  - Three CLS-to-patch-subspace overlap fractions, using a PCA basis learned from centered PATCH embeddings
    on NoSCAM (basis spans dominant directions of (PATCH - patch_mean)):

      cls_patch_frac_patch_center:
        energy fraction of (CLS - patch_mean) that lies in the PATCH PCA subspace.

      cls_patch_frac_cls_center:
        energy of Proj_basis(CLS - patch_mean), normalized by energy of (CLS - cls_mean).
        This answers: "how large is the patch-subspace component relative to CLS's own variation?"

      cls_patch_frac_cls_center_sym:
        energy fraction of (CLS - cls_mean) that lies in the PATCH PCA subspace
        (project CLS deviations, not CLS-minus-patch-mean).

A linear least-squares "CLS from PATCH" regression map is fit on NoSCAM:
  - Train: solve (PATCH - patch_mean) -> (CLS - cls_mean) via least squares.
  - Use: CLS-PATCHREG embedding is the predicted CLS vector from PATCH at test time.
Teacher diagnostics: mean cosine(predicted_CLS, true_CLS) and MSE on the NoSCAM build set.
"""

from __future__ import annotations

import os
import json
import math
import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import attnclipindiv as clip
from attnclipindiv.model import CLIP

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

out_dir = "out_eval_reproduce/regression-teacher-zeroshot_mvt"
os.makedirs(out_dir, exist_ok=True)


# dataloader / worker tuning
CPU_COUNT = os.cpu_count() or 8
NUM_WORKERS = max(4, min(8, 6 if CPU_COUNT >= 12 else 4))
PERSISTENT_WORKERS = True
PIN_MEMORY = True
PREFETCH_FACTOR = 4

# Batch sizes
PREPROCESS_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 64
STATS_BATCH_SIZE = 64
NOCS_BUILD_BATCH_SIZE = 64

# Reg/Patch definitions + sampling for PCA
REG_THRESHOLD = 70.0
MAX_REG_SAMPLES_PER_LAYER = 25000
MAX_PATCH_SAMPLES_PER_LAYER = 25000
REG_PCS_K = 8


# ImageNet teacher-build options
use_imagenet_for_teacher: bool = False # if False, build from NoSCAM
imagenet_per_class: int = 3
IMAGENET_TRAIN_DIR = r"path/to/ILSVRC2012/train"
IMAGENET_VAL_DIR   = r"path/to/ILSVRC2012/val"
IMAGENET_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENET_TEACHER_SEED = 1337


# MVT dataset
MVT_CSV_FILE = "utils_datasets/mvt/human_responses_5k.csv"
#MVT_CSV_FILE = "path/to/dataset-difficulty-CLIP/data_release_2023/human_responses.csv" # full, takes long
MVT_IMAGE_FOLDER = "path/to/dataset-difficulty-CLIP/data_release_2023/all/"
MVT_PROMPT_TEMPLATE = "a photo of a {}"




# CLS–patch model hyperparams
CLS_PATCH_PCA_K = 128

# Which transformer block outputs to capture (post-block token states).
#TARGET_LAYERS: List[int] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

TARGET_LAYERS: List[int] = [22, 23] # That's all we need for patch-delta, etc.
TARGET_LAYERS_TUP: Tuple[int, ...] = tuple(TARGET_LAYERS)

LAYER_22 = 22
LAYER_23 = 23

METHODS: List[str] = [
    "CLS",
    "CLS-PATCHSUB",
    "CLS-PATCHREG",
    "REG-L23-NOPC",
    "REG-L23-1PC",
    "REG-L23-8PC",
    "PATCH-L23",
    "PATCHΔ",
]

VARIANTS = ["NoSCAM", "SCAM", "SynthSCAM"]

# AMP for speed
USE_AMP = False

# Plot config
PLOT_DPI = 150

# orthogonality thresholds / metrics
ORTH_ABS_THRESH = 0.05  # "near-orthogonal" if |cos| < this

# Keep SCAM variants separate (avoid mixing semantics)
SCAM_VARIANTS = VARIANTS
MVT_VARIANT_NAME = "MVT"


def _c(msg: str, fore: str = Fore.WHITE, bright: bool = False) -> str:
    return f"{Style.BRIGHT if bright else ''}{fore}{msg}{Style.RESET_ALL}"

def c_ok(msg: str, bright: bool = True) -> str:      # expected / supports hypothesis
    return _c(msg, Fore.GREEN, bright=bright)

def c_dev(msg: str, bright: bool = True) -> str:     # interesting deviation
    return _c(msg, Fore.MAGENTA, bright=bright)

def c_bad(msg: str, bright: bool = True) -> str:     # severe deviation / regression
    return _c(msg, Fore.RED, bright=bright)

def c_note(msg: str, bright: bool = True) -> str:    # neutral remark
    return _c(msg, Fore.YELLOW, bright=bright)

def c_hi(msg: str, bright: bool = True) -> str:      # highlight/headline
    return _c(msg, Fore.CYAN, bright=bright)


def cast_imgs_to_visual_dtype(imgs: torch.Tensor, visual: nn.Module) -> torch.Tensor:
    # Cast inputs to match model's conv1 weight dtype
    w = visual.conv1.weight
    target_dtype = w.dtype
    if imgs.dtype != target_dtype:
        imgs = imgs.to(dtype=target_dtype)
    return imgs

def sanitize_filename(s: str, max_len: int = 140) -> str:
    """
    - Replaces illegal chars: <>:"/\\|?*
    - Collapses whitespace
    - Strips trailing dots/spaces
    """
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .")
    if not s:
        s = "model"
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s

# alternate dataset for teacher
class ImageNetSampledDataset(Dataset):
    """
    Samples up to `per_class` images per wnid folder from one or more ImageNet roots.
    Returns (idx, preprocessed_tensor_float32) so it matches _collate_idx_img and
    existing teacher builder loops.
    """
    def __init__(
        self,
        roots: List[str],
        transform,
        per_class: int = 3,
        seed: int = 1337,
        exts: Tuple[str, ...] = IMAGENET_EXTS,
    ):
        self.transform = transform
        self.per_class = int(per_class)
        self.exts = tuple(e.lower() for e in exts)
        rng = random.Random(seed)

        samples: List[str] = []
        class_counts: Dict[str, int] = {}
        missing_roots: List[str] = []

        for root in roots:
            if not root or not os.path.isdir(root):
                missing_roots.append(str(root))
                continue

            wnids = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            wnids.sort()

            for wnid in wnids:
                cdir = os.path.join(root, wnid)
                try:
                    files = [
                        os.path.join(cdir, fn)
                        for fn in os.listdir(cdir)
                        if os.path.isfile(os.path.join(cdir, fn)) and fn.lower().endswith(self.exts)
                    ]
                except Exception:
                    files = []

                if not files:
                    continue

                k = min(self.per_class, len(files))
                chosen = files if k == len(files) else rng.sample(files, k=k)
                samples.extend(chosen)
                class_counts[wnid] = class_counts.get(wnid, 0) + len(chosen)

        self.samples = samples
        self.class_counts = class_counts
        self.missing_roots = missing_roots

        # I/O robustness
        self.io_retries = 12
        self.io_sleep_base = 0.02
        self.io_sleep_max = 0.50

    def __len__(self) -> int:
        return len(self.samples)

    def _safe_open_rgb(self, image_path: str) -> Image.Image:
        last_exc = None
        for attempt in range(self.io_retries):
            try:
                with open(image_path, "rb") as f:
                    img = Image.open(f)
                    img = img.convert("RGB")
                return img
            except PermissionError as e:
                last_exc = e
            except OSError as e:
                last_exc = e

            t = min(self.io_sleep_max, self.io_sleep_base * (2 ** attempt))
            t = t + random.random() * 0.01
            time.sleep(t)

        raise last_exc

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = self._safe_open_rgb(path)
        x = self.transform(img)  # float32 [3,224,224]
        return idx, x


def build_imagenet_teacher_loader(
    preprocess,
    device: str,
    per_class: int = 3,
    batch_size: int = NOCS_BUILD_BATCH_SIZE,
) -> Tuple[DataLoader, Dict[str, object]]:
    """
    Creates a small ImageNet teacher-build loader by sampling per-class from:
      - ILSVRC2012 train wnids
      - ILSVRC2012 val wnids
    """
    roots = [IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR]
    ds = ImageNetSampledDataset(
        roots=roots,
        transform=preprocess,
        per_class=per_class,
        seed=IMAGENET_TEACHER_SEED,
        exts=IMAGENET_EXTS,
    )

    info = {
        "teacher_build_source": "ImageNet(train+val)",
        "imagenet_per_class": int(per_class),
        "imagenet_n_classes": int(len(ds.class_counts)),
        "imagenet_n_images": int(len(ds)),
        "imagenet_missing_roots": list(ds.missing_roots),
    }

    # Summary prints
    print(c_hi("\nTeacher-build source: ImageNet sample", bright=True))
    print(
        c_note("  roots: ", bright=True) +
        f"train={IMAGENET_TRAIN_DIR} | val={IMAGENET_VAL_DIR}"
    )
    print(
        c_note("  sampled: ", bright=True) +
        f"classes={info['imagenet_n_classes']} | images={info['imagenet_n_images']} | per_class={info['imagenet_per_class']}"
    )
    if info["imagenet_missing_roots"]:
        print(c_bad(f"  missing roots: {info['imagenet_missing_roots']}"))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=_collate_idx_img,
    )
    return loader, info


# MVT dataset (ImageNet/ObjectNet "dataset-difficulty-CLIP")
class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_file: str, image_folder: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

        self._images = self.data["image"].astype(str).tolist()
        self._labels = self.data["label"].astype(str).tolist()

        # I/O robustness
        self.io_retries = 12          # how many times to retry a transient lock
        self.io_sleep_base = 0.02     # seconds
        self.io_sleep_max = 0.50      # seconds
        self.skip_on_fail = False     # set True if you prefer “log+skip”

    def _safe_open_rgb(self, image_path: str) -> Image.Image:
        """
        Windows can throw transient PermissionError due to AV/indexer locks.
        Retry with exponential backoff + tiny jitter.
        Also ensures file handle closes promptly (use 'with open').
        """
        last_exc = None
        for attempt in range(self.io_retries):
            try:
                with open(image_path, "rb") as f:
                    img = Image.open(f)
                    img = img.convert("RGB")
                return img
            except PermissionError as e:
                last_exc = e
            except OSError as e:
                last_exc = e

            t = min(self.io_sleep_max, self.io_sleep_base * (2 ** attempt))
            t = t + random.random() * 0.01
            time.sleep(t)

        raise last_exc

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image_name = self._images[idx]
        image_path = os.path.join(self.image_folder, image_name)

        try:
            image = self._safe_open_rgb(image_path)
        except Exception as e:
            if self.skip_on_fail:
                return None
            raise

        if self.transform is not None:
            image = self.transform(image)

        label = self._labels[idx]
        return image, label


def build_label_vocab(csv_path: str) -> Tuple[List[str], Dict[str, int]]:
    df = pd.read_csv(csv_path, usecols=["label"])
    labels = df["label"].astype(str)
    uniq = labels.unique().tolist()
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    return uniq, label_to_idx


def _collate_img_label(batch):
    # Drop Nones if user enables skip_on_fail in dataset
    batch = [b for b in batch if b is not None]
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = [str(b[1]) for b in batch]
    return imgs, labels



class PreprocessAllDataset(Dataset):
    """Returns (global_idx, preprocessed_tensor_float32)."""
    def __init__(self, hf_ds, preprocess):
        self.ds = hf_ds
        self.preprocess = preprocess

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img = self.ds[idx]["image"]
        x = self.preprocess(img)  # float32, [3,224,224]
        return idx, x

class PreloadedVariantDataset(Dataset):
    """Variant view over a shared [N,3,224,224] tensor. Returns (global_idx, img_tensor)."""
    def __init__(self, all_imgs: torch.Tensor, indices: List[int]):
        self.all_imgs = all_imgs
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        gi = self.indices[i]
        return gi, self.all_imgs[gi]

def _collate_idx_img(batch):
    idxs = torch.tensor([b[0] for b in batch], dtype=torch.long)
    imgs = torch.stack([b[1] for b in batch], dim=0)
    return idxs, imgs


@torch.no_grad()
def forward_visual_with_tokens(
    visual: nn.Module,
    x: torch.Tensor,
    layers_to_capture: Tuple[int, ...],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Returns:
      cls_embed_raw: [B, D] (after ln_post and proj)
      post_tokens: dict[layer_idx] = [B, seq, width] (post block)
    """
    # Enforce input dtype to match conv1
    w = visual.conv1.weight
    if x.dtype != w.dtype:
        x = x.to(dtype=w.dtype)

    capture = set(layers_to_capture)
    post_tokens: Dict[int, torch.Tensor] = {}

    # stem
    x = visual.conv1(x)
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, HW, width]

    class_emb = visual.class_embedding.to(x.dtype)
    cls_tokens = class_emb + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+HW, width]

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # [seq, B, width]

    # transformer blocks
    for li, block in enumerate(visual.transformer.resblocks):
        x = block(x)
        if li in capture:
            post_tokens[li] = x.permute(1, 0, 2).detach()  # [B, seq, width]

    # final CLS head (CLIP-style)
    x = x.permute(1, 0, 2)      # [B, seq, width]
    cls = x[:, 0, :]            # [B, width]
    cls = visual.ln_post(cls)   # [B, width]
    if visual.proj is not None:
        cls = cls @ visual.proj  # [B, D]
    return cls, post_tokens


def _masked_mean_with_fallback_all(
    x: torch.Tensor,            # [B, HW, d]
    mask: torch.Tensor,         # [B, HW] bool
) -> torch.Tensor:
    """
    Mean over mask; if a row has 0 trues, fallback to mean over all HW.
    """
    B, HW, d = x.shape
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)         # [B, HW, 1]
    sum_sel = (x * mask_f).sum(dim=1)                     # [B, d]
    cnt = mask.sum(dim=1)                                 # [B]
    cnt_f = cnt.clamp(min=1).to(dtype=x.dtype).unsqueeze(-1)

    mean_sel = sum_sel / cnt_f                            # [B, d]

    fallback = (cnt == 0)
    if fallback.any():
        mean_all = x.mean(dim=1)                          # [B, d]
        mean_sel = torch.where(fallback.unsqueeze(-1), mean_all, mean_sel)
    return mean_sel


def _masked_mean_no_fallback(
    x: torch.Tensor,            # [B, HW, d]
    mask: torch.Tensor,         # [B, HW] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mean over mask; if empty => returns zeros and has_any=False.
    """
    B, HW, d = x.shape
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    sum_sel = (x * mask_f).sum(dim=1)                     # [B, d]
    cnt = mask.sum(dim=1)                                 # [B]
    has_any = (cnt > 0)
    cnt_f = cnt.clamp(min=1).to(dtype=x.dtype).unsqueeze(-1)
    mean_sel = sum_sel / cnt_f
    mean_sel = torch.where(has_any.unsqueeze(-1), mean_sel, torch.zeros_like(mean_sel))
    return mean_sel, has_any


def _project_out_subspace(x: torch.Tensor, pcs: torch.Tensor) -> torch.Tensor:
    """
    x: [B, d]
    pcs: [k, d] assumed unit-norm rows
    returns: x - (x @ pcs.T) @ pcs
    """
    coeff = x @ pcs.transpose(0, 1)       # [B, k]
    return x - (coeff @ pcs)              # [B, d]


@torch.no_grad()
def build_patch_embed_from_tokens(
    tokens: torch.Tensor,                 # [B, seq, width]
    ln_post: nn.Module,
    proj: torch.Tensor,
    reg_threshold: float,
    patch_mean: Optional[torch.Tensor] = None,   # [width]
    patch_pc: Optional[torch.Tensor] = None,     # [width] or [k,width]
) -> torch.Tensor:
    """
    Returns raw PATCH embedding in final embed space: [B, D] (NOT normalized).
    PATCH = mean(non-REG patches), fallback to all patches.
    """
    device = tokens.device
    patches = tokens[:, 1:, :]                         # [B, HW, width]
    norms = patches.norm(dim=-1)                       # [B, HW]
    reg_mask = norms >= reg_threshold
    patch_mask = ~reg_mask

    patch_hidden = _masked_mean_with_fallback_all(patches, patch_mask)  # [B, width]

    if patch_mean is not None:
        patch_hidden = patch_hidden - patch_mean.to(device=device, dtype=patch_hidden.dtype)

    if patch_pc is not None:
        pcs = patch_pc.to(device=device, dtype=patch_hidden.dtype)
        if pcs.ndim == 1:
            pcs = pcs.view(1, -1)
        pcs = pcs / (pcs.norm(dim=1, keepdim=True) + 1e-8)
        patch_hidden = _project_out_subspace(patch_hidden, pcs)

    patch_hidden = ln_post(patch_hidden)               # [B, width]
    return patch_hidden @ proj                         # [B, D]


@torch.no_grad()
def build_reg_embed_from_tokens(
    tokens: torch.Tensor,                 # [B, seq, width]
    ln_post: nn.Module,
    proj: torch.Tensor,
    reg_threshold: float,
    reg_mean: Optional[torch.Tensor] = None,    # [width]
    reg_pc: Optional[torch.Tensor] = None,      # [width] or [k,width]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      reg_embed_raw: [B, D] (NOT normalized). Zero vector if no REG in sample.
      has_reg: [B] bool
    """
    device = tokens.device
    patches = tokens[:, 1:, :]
    norms = patches.norm(dim=-1)
    reg_mask = norms >= reg_threshold

    reg_hidden, has_reg = _masked_mean_no_fallback(patches, reg_mask)   # [B,width], [B]

    if reg_mean is not None:
        reg_hidden = reg_hidden - reg_mean.to(device=device, dtype=reg_hidden.dtype)

    if reg_pc is not None:
        pcs = reg_pc.to(device=device, dtype=reg_hidden.dtype)
        if pcs.ndim == 1:
            pcs = pcs.view(1, -1)
        pcs = pcs / (pcs.norm(dim=1, keepdim=True) + 1e-8)
        reg_hidden = _project_out_subspace(reg_hidden, pcs)

    reg_hidden = ln_post(reg_hidden)
    reg_embed = reg_hidden @ proj
    return reg_embed, has_reg


def _pca_lowrank_rows(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    X: [N, d] (CPU float32)
    Returns pcs: [k, d], unit-norm rows
    """
    X = X.float()
    N, d = X.shape
    q = min(d, max(k + 8, k))
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(Xc, q=q, center=False, niter=2)
    pcs = V[:, :k].transpose(0, 1).contiguous()  # [k, d]
    pcs = pcs / (pcs.norm(dim=1, keepdim=True) + 1e-8)
    return pcs


def collect_reg_and_patch_hidden_stats(
    visual: nn.Module,
    loader: DataLoader,
    device: str,
    reg_threshold: float = REG_THRESHOLD,
    target_layers: Tuple[int, ...] = TARGET_LAYERS_TUP,
    max_reg_samples_per_layer: int = MAX_REG_SAMPLES_PER_LAYER,
    max_patch_samples_per_layer: int = MAX_PATCH_SAMPLES_PER_LAYER,
    reg_pcs_k: int = REG_PCS_K,
):
    """
    Computes per-layer pooled patch and “register-like” hidden statistics for `target_layers`.
    For each requested layer ℓ:
      - patch_mean_ℓ: mean of pooled non-REG patches (||token|| < τ), with per-sample fallback to mean(all patches).
      - reg_mean_ℓ:   mean of pooled REG-like patches (||token|| ≥ τ), defined only for samples with ≥1 REG patch.
      - patch_pc_ℓ:   top-1 PCA direction of pooled patch means (optional nuisance direction).
      - reg_pcs_ℓ:    top-k PCA directions of pooled REG means (optional nuisance subspace).
    """
    num_blocks = len(visual.transformer.resblocks)
    width = visual.conv1.weight.shape[0]

    reg_sum = [torch.zeros(width, device=device) for _ in range(num_blocks)]
    reg_count = [0 for _ in range(num_blocks)]
    reg_samples: List[List[torch.Tensor]] = [[] for _ in range(num_blocks)]

    patch_sum = [torch.zeros(width, device=device) for _ in range(num_blocks)]
    patch_count = [0 for _ in range(num_blocks)]
    patch_samples: List[List[torch.Tensor]] = [[] for _ in range(num_blocks)]

    with torch.no_grad():
        pbar = tqdm(loader, desc="Pass 1: REG/PATCH stats", ncols=100)
        for _, imgs_cpu in pbar:
            imgs = imgs_cpu.to(device, non_blocking=True)
            imgs = cast_imgs_to_visual_dtype(imgs, visual)
            _, post = forward_visual_with_tokens(visual, imgs, target_layers)

            for li in target_layers:
                tokens = post[li]                          # [B, seq, width]
                patches = tokens[:, 1:, :]                 # [B, HW, width]
                norms = patches.norm(dim=-1)               # [B, HW]
                reg_mask = norms >= reg_threshold
                patch_mask = ~reg_mask

                reg_mean_b, has_reg = _masked_mean_no_fallback(patches, reg_mask)          # [B,width]
                patch_mean_b = _masked_mean_with_fallback_all(patches, patch_mask)         # [B,width]

                if has_reg.any():
                    reg_sum[li] += reg_mean_b[has_reg].sum(dim=0)
                    reg_count[li] += int(has_reg.sum().item())

                patch_sum[li] += patch_mean_b.sum(dim=0)
                patch_count[li] += patch_mean_b.shape[0]

                if len(reg_samples[li]) < max_reg_samples_per_layer and has_reg.any():
                    take = min(max_reg_samples_per_layer - len(reg_samples[li]), int(has_reg.sum().item()))
                    reg_samples[li].extend(reg_mean_b[has_reg][:take].detach().float().cpu().unbind(0))

                if len(patch_samples[li]) < max_patch_samples_per_layer:
                    take = min(max_patch_samples_per_layer - len(patch_samples[li]), patch_mean_b.shape[0])
                    patch_samples[li].extend(patch_mean_b[:take].detach().float().cpu().unbind(0))

    reg_means: List[Optional[torch.Tensor]] = [None for _ in range(num_blocks)]
    reg_pcs:   List[Optional[torch.Tensor]] = [None for _ in range(num_blocks)]
    patch_means: List[Optional[torch.Tensor]] = [None for _ in range(num_blocks)]
    patch_pcs:   List[Optional[torch.Tensor]] = [None for _ in range(num_blocks)]

    for li in target_layers:
        if reg_count[li] > 0:
            reg_means[li] = (reg_sum[li] / reg_count[li]).detach()
        if patch_count[li] > 0:
            patch_means[li] = (patch_sum[li] / patch_count[li]).detach()

        if len(reg_samples[li]) >= 256:
            X = torch.stack(reg_samples[li], dim=0)  # [N,width] CPU
            k = min(reg_pcs_k, X.shape[1])
            pcs = _pca_lowrank_rows(X, k=k)          # [k,width] CPU
            reg_pcs[li] = pcs.to(device)

        if len(patch_samples[li]) >= 256:
            Xp = torch.stack(patch_samples[li], dim=0)
            top1 = _pca_lowrank_rows(Xp, k=1)[0]     # [width]
            patch_pcs[li] = top1.to(device)

    return reg_means, reg_pcs, patch_means, patch_pcs


# CLS–PATCH subspace & regression model (ImageNet OR NoSCAM only)
def build_cls_patch_model(
    model: CLIP,
    visual: nn.Module,
    loader_noscam: DataLoader,
    device: str,
    reg_threshold: float = REG_THRESHOLD,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, float]]:
    """
    Batched build on NoSCAM loader (fast), plus teacher quality stats.

    Returns:
      cls_patch_model dict (on device) or None
      teacher_stats dict (python floats)
    """
    ln_post = visual.ln_post
    proj = visual.proj
    if proj is None:
        raise ValueError("visual.proj is None; this script assumes a CLIP-style projection matrix exists.")

    cls_list: List[torch.Tensor] = []
    patch_list: List[torch.Tensor] = []

    with torch.no_grad():
        for _, imgs_cpu in tqdm(loader_noscam, desc="Build CLS–PATCH model (NoSCAM)", ncols=100):
            imgs = imgs_cpu.to(device, non_blocking=True)
            imgs = cast_imgs_to_visual_dtype(imgs, visual)

            cls_raw, post = forward_visual_with_tokens(visual, imgs, (LAYER_23,))
            tokens_23 = post[LAYER_23]  # [B,seq,width]

            patch_raw = build_patch_embed_from_tokens(
                tokens_23, ln_post=ln_post, proj=proj,
                reg_threshold=reg_threshold,
                patch_mean=None, patch_pc=None,
            )

            cls_list.append(cls_raw.detach().float().cpu())
            patch_list.append(patch_raw.detach().float().cpu())

    if len(cls_list) == 0:
        return None, {
            "n_noscam": 0.0,
            "teacher_cos_mean": float("nan"),
            "teacher_cos_std": float("nan"),
            "teacher_mse": float("nan"),
            "patch_pca_k": float("nan"),
        }

    C = torch.cat(cls_list, dim=0)   # [N,D]
    P = torch.cat(patch_list, dim=0) # [N,D]
    N, D = C.shape

    C_mean = C.mean(dim=0, keepdim=True)
    P_mean = P.mean(dim=0, keepdim=True)
    Cc = C - C_mean
    Pc = P - P_mean

    k_pca = min(CLS_PATCH_PCA_K, D)
    patch_basis = _pca_lowrank_rows(Pc, k=k_pca)  # [k,D] CPU

    lstsq = torch.linalg.lstsq(Pc, Cc)
    X = lstsq.solution          # [D,D]
    W = X.T                     # [D,D]

    C_hat = (Pc @ X) + C_mean   # [N,D]
    cos = F.cosine_similarity(C_hat, C, dim=-1).detach().cpu()
    mse = (C_hat - C).pow(2).mean().item()

    teacher_stats = {
        "n_noscam": float(N),
        "teacher_cos_mean": float(cos.mean().item()),
        "teacher_cos_std": float(cos.std(unbiased=False).item()),
        "teacher_mse": float(mse),
        "patch_pca_k": float(k_pca),
    }

    cls_patch_model = {
        "patch_mean": P_mean.squeeze(0).to(device),
        "cls_mean": C_mean.squeeze(0).to(device),
        "patch_basis": patch_basis.to(device),
        "W": W.to(device),
    }
    return cls_patch_model, teacher_stats

# Tokenization helper (batched)
def tokenize_pair_prompts(object_labels: List[str], attack_words: List[str], device: str) -> torch.Tensor:
    """
    Returns tokens [B,2,77]
    """
    B = len(object_labels)
    texts: List[str] = []
    texts_extend = texts.extend
    for i in range(B):
        texts_extend([f"a photo of a {object_labels[i]}", f"a photo of a {attack_words[i]}"])
    tok = clip.tokenize(texts).to(device)
    return tok.view(B, 2, -1)

# MVT text features builder (multi-class)
@torch.no_grad()
def build_text_features_for_labels(
    model: CLIP,
    labels: List[str],
    device: str,
    prompt_template: str = MVT_PROMPT_TEMPLATE,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Returns:
      text_features: [C, D] normalized, on device.
      Dtype policy:
        - If USE_AMP and CUDA: return float16 (fast path)
        - Else: return float32 (matches float32 reps when USE_AMP=False)
    """
    model.eval()

    use_half = (device.startswith("cuda") and USE_AMP)

    feats: List[torch.Tensor] = []
    for i in tqdm(range(0, len(labels), batch_size), desc="Encode MVT text", ncols=100, leave=False):
        chunk = labels[i:i + batch_size]
        texts = [prompt_template.format(lab) for lab in chunk]
        tok = clip.tokenize(texts).to(device)

        if device.startswith("cuda") and USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                t = model.encode_text(tok)
            t = t.float()
        else:
            t = model.encode_text(tok).float()

        t = F.normalize(t, dim=-1)

        if use_half:
            t = t.half()

        feats.append(t.detach())

    return torch.cat(feats, dim=0)


def eval_mvt_for_model(
    model: CLIP,
    visual: nn.Module,
    loader_mvt: DataLoader,
    device: str,
    labels: List[str],
    label_to_idx: Dict[str, int],
    text_features: torch.Tensor,  # [C,D] normalized
    reg_means: List[Optional[torch.Tensor]],
    reg_pcs: List[Optional[torch.Tensor]],
    patch_means: List[Optional[torch.Tensor]],
    patch_pcs: List[Optional[torch.Tensor]],
    cls_patch_model: Optional[Dict[str, torch.Tensor]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    MVT multi-class zero-shot evaluation for all METHODS.

    Returns:
      df_details: per-sample details (pred/correct + geometry diagnostics)
      verbose: per-method aggregate stats (margin stats)
    """
    ln_post = visual.ln_post
    proj = visual.proj

    if cls_patch_model is not None:
        patch_mean_D = cls_patch_model["patch_mean"].view(1, -1)  # [1,D]
        cls_mean_D   = cls_patch_model["cls_mean"].view(1, -1)    # [1,D]
        basis        = cls_patch_model["patch_basis"]             # [k,D]
        Wmat         = cls_patch_model["W"]                       # [D,D]
    else:
        patch_mean_D = None
        cls_mean_D   = None
        basis        = None
        Wmat         = None

    rows: List[Dict[str, object]] = []
    margin_accum: Dict[str, List[float]] = {m: [] for m in METHODS}
    correct_count: Dict[str, int] = {m: 0 for m in METHODS}
    total_count = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (device.startswith("cuda") and USE_AMP)
        else nullcontext()
    )

    with torch.no_grad():
        pbar = tqdm(loader_mvt, desc="Eval MVT", ncols=100)
        for imgs_cpu, batch_labels in pbar:
            B = imgs_cpu.shape[0]
            imgs = imgs_cpu.to(device, non_blocking=True)
            imgs = cast_imgs_to_visual_dtype(imgs, visual)

            with amp_ctx:
                cls_raw, post = forward_visual_with_tokens(visual, imgs, TARGET_LAYERS_TUP)
                cls_base = F.normalize(cls_raw, dim=-1)

                tokens_22 = post[LAYER_22]
                tokens_23 = post[LAYER_23]

                patch_l23_raw = build_patch_embed_from_tokens(
                    tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                    patch_mean=None, patch_pc=None,
                )
                patch_l23 = F.normalize(patch_l23_raw, dim=-1)

                if patch_means[LAYER_22] is not None and patch_pcs[LAYER_22] is not None and \
                   patch_means[LAYER_23] is not None and patch_pcs[LAYER_23] is not None:
                    p22_raw = build_patch_embed_from_tokens(
                        tokens_22, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        patch_mean=patch_means[LAYER_22], patch_pc=patch_pcs[LAYER_22],
                    )
                    p23_raw = build_patch_embed_from_tokens(
                        tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        patch_mean=patch_means[LAYER_23], patch_pc=patch_pcs[LAYER_23],
                    )
                    pdelta = F.normalize(p23_raw - p22_raw, dim=-1)
                else:
                    pdelta = cls_base

                def _reg_variant(reg_pc: Optional[torch.Tensor]) -> torch.Tensor:
                    if reg_means[LAYER_23] is None:
                        return cls_base
                    reg_raw, has_reg = build_reg_embed_from_tokens(
                        tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        reg_mean=reg_means[LAYER_23], reg_pc=reg_pc,
                    )
                    reg_norm = F.normalize(reg_raw, dim=-1)
                    out = cls_base.clone()
                    if has_reg.any():
                        out[has_reg] = reg_norm[has_reg]
                    return out

                reg_nopc = _reg_variant(None)
                reg_1pc = _reg_variant(reg_pcs[LAYER_23][0] if reg_pcs[LAYER_23] is not None else None)
                reg_8pc = _reg_variant(
                    reg_pcs[LAYER_23][:min(8, reg_pcs[LAYER_23].shape[0])] if reg_pcs[LAYER_23] is not None else None
                )

                # CLS-PATCHSUB + overlap fractions + k80 PCs
                if cls_patch_model is not None:
                    basis_use = basis.to(dtype=cls_raw.dtype)

                    x_center_patch = cls_raw - patch_mean_D
                    coeff_patch = x_center_patch @ basis_use.T
                    proj_x_patch = coeff_patch @ basis_use
                    cls_patchsub = F.normalize(proj_x_patch, dim=-1)

                    numer_patch = proj_x_patch.pow(2).sum(dim=-1)
                    denom_patch = x_center_patch.pow(2).sum(dim=-1).add_(1e-8)
                    cls_patch_frac_patch_center = numer_patch / denom_patch

                    x_center_cls = cls_raw - cls_mean_D
                    denom_cls = x_center_cls.pow(2).sum(dim=-1).add_(1e-8)
                    cls_patch_frac_cls_center = numer_patch / denom_cls

                    coeff_cls = x_center_cls @ basis_use.T
                    proj_x_cls = coeff_cls @ basis_use
                    numer_cls = proj_x_cls.pow(2).sum(dim=-1)
                    cls_patch_frac_cls_center_sym = numer_cls / denom_cls

                    energy_pc = coeff_cls.pow(2)
                    total_proj = energy_pc.sum(dim=-1)
                    cum = torch.cumsum(energy_pc, dim=-1)
                    frac = cum / (total_proj.unsqueeze(-1) + 1e-8)

                    mask = frac >= 0.80
                    mask[:, -1] = True
                    k_idx0 = mask.float().argmax(dim=-1)
                    cls_patch_k80 = (k_idx0 + 1).to(dtype=torch.int64)
                    zero_proj = total_proj <= 1e-12
                    if zero_proj.any():
                        cls_patch_k80 = cls_patch_k80.clone()
                        cls_patch_k80[zero_proj] = -1
                else:
                    cls_patchsub = cls_base
                    cls_patch_frac_patch_center = None
                    cls_patch_frac_cls_center = None
                    cls_patch_frac_cls_center_sym = None
                    cls_patch_k80 = None

                # CLS-PATCHREG
                if cls_patch_model is not None:
                    p_center = patch_l23_raw - patch_mean_D
                    c_center_hat = p_center @ Wmat.T
                    c_hat = c_center_hat + cls_mean_D
                    cls_patchreg = F.normalize(c_hat, dim=-1)
                else:
                    cls_patchreg = cls_base

                # Orthogonality cosines
                cos_cls_patch = (cls_base * patch_l23).sum(dim=-1)

                if reg_means[LAYER_23] is not None:
                    reg_raw_for_cos, has_reg_for_cos = build_reg_embed_from_tokens(
                        tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        reg_mean=reg_means[LAYER_23], reg_pc=None,
                    )
                    reg_norm_for_cos = F.normalize(reg_raw_for_cos, dim=-1)
                    cos_cls_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)
                    cos_patch_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)
                    if has_reg_for_cos.any():
                        cos_cls_reg[has_reg_for_cos] = (
                            cls_base[has_reg_for_cos] * reg_norm_for_cos[has_reg_for_cos]
                        ).sum(dim=-1)
                        cos_patch_reg[has_reg_for_cos] = (
                            patch_l23[has_reg_for_cos] * reg_norm_for_cos[has_reg_for_cos]
                        ).sum(dim=-1)
                else:
                    has_reg_for_cos = torch.zeros((B,), device=cls_base.device, dtype=torch.bool)
                    cos_cls_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)
                    cos_patch_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)

                reps: Dict[str, torch.Tensor] = {
                    "CLS": cls_base,
                    "CLS-PATCHSUB": cls_patchsub,
                    "CLS-PATCHREG": cls_patchreg,
                    "REG-L23-NOPC": reg_nopc,
                    "REG-L23-1PC": reg_1pc,
                    "REG-L23-8PC": reg_8pc,
                    "PATCH-L23": patch_l23,
                    "PATCHΔ": pdelta,
                }

                # Score each method against full label vocab
                # text_features: [C,D] normalized
                # logits: [B,C]
                pred_idx: Dict[str, torch.Tensor] = {}
                top1_logit: Dict[str, torch.Tensor] = {}
                margin12: Dict[str, torch.Tensor] = {}

                text_features = text_features

                if text_features.dtype != cls_base.dtype:
                    text_features = text_features.to(dtype=cls_base.dtype)

                for m in METHODS:
                    logits = reps[m] @ text_features.T
                    v2, i2 = torch.topk(logits, k=2, dim=-1)
                    pred = i2[:, 0]
                    pred_idx[m] = pred
                    top1_logit[m] = v2[:, 0]
                    margin12[m] = (v2[:, 0] - v2[:, 1])

            # CPU bookkeeping
            for bi in range(B):
                lab = str(batch_labels[bi])
                gt = label_to_idx.get(lab, None)
                if gt is None:
                    continue

                total_count += 1
                row: Dict[str, object] = {
                    "variant": MVT_VARIANT_NAME,
                    "gt_label": lab,
                    "gt_idx": int(gt),

                    "cos_cls_patch": float(cos_cls_patch[bi].item()),
                    "cos_cls_reg": float(cos_cls_reg[bi].item()),
                    "cos_patch_reg": float(cos_patch_reg[bi].item()),
                    "has_reg": bool(has_reg_for_cos[bi].item()),

                    "cls_patch_frac_patch_center": float(cls_patch_frac_patch_center[bi].item())
                        if cls_patch_frac_patch_center is not None else float("nan"),
                    "cls_patch_frac_cls_center": float(cls_patch_frac_cls_center[bi].item())
                        if cls_patch_frac_cls_center is not None else float("nan"),
                    "cls_patch_frac_cls_center_sym": float(cls_patch_frac_cls_center_sym[bi].item())
                        if cls_patch_frac_cls_center_sym is not None else float("nan"),
                    "cls_patch_k80": int(cls_patch_k80[bi].item()) if cls_patch_k80 is not None else -1,
                }

                for m in METHODS:
                    pi = int(pred_idx[m][bi].item())
                    is_corr = (pi == gt)
                    if is_corr:
                        correct_count[m] += 1

                    row[f"pred_idx_{m}"] = pi
                    row[f"pred_label_{m}"] = labels[pi] if (0 <= pi < len(labels)) else "<?>"
                    row[f"is_correct_{m}"] = bool(is_corr)
                    row[f"top1_logit_{m}"] = float(top1_logit[m][bi].item())
                    row[f"margin12_{m}"] = float(margin12[m][bi].item())
                    margin_accum[m].append(float(margin12[m][bi].item()))

                rows.append(row)

    df = pd.DataFrame(rows)

    verbose: Dict[str, Dict[str, object]] = {}
    for m in METHODS:
        margins = torch.tensor(margin_accum[m], dtype=torch.float32)
        verbose[m] = {
            "n_total": int(total_count),
            "n_correct": int(correct_count[m]),
            "accuracy": (float(correct_count[m]) / float(total_count)) if total_count > 0 else float("nan"),
            "margin12_mean": float(margins.mean().item()) if margins.numel() else float("nan"),
            "margin12_std": float(margins.std(unbiased=False).item()) if margins.numel() else float("nan"),
            "margin12_min": float(margins.min().item()) if margins.numel() else float("nan"),
            "margin12_max": float(margins.max().item()) if margins.numel() else float("nan"),
        }

    return df, verbose


@dataclass
class MethodResult:
    correct: torch.Tensor      # [B] bool
    conf_obj: torch.Tensor     # [B] float
    conf_att: torch.Tensor     # [B] float


def eval_variant_for_model(
    model: CLIP,
    visual: nn.Module,
    loader_variant: DataLoader,
    meta: Dict[str, List],
    device: str,
    reg_means: List[Optional[torch.Tensor]],
    reg_pcs: List[Optional[torch.Tensor]],
    patch_means: List[Optional[torch.Tensor]],
    patch_pcs: List[Optional[torch.Tensor]],
    cls_patch_model: Optional[Dict[str, torch.Tensor]],
    variant_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Returns:
      per-sample dataframe (compact but detailed)
      verbose dict (fooled words + margin stats)
    """
    ln_post = visual.ln_post
    proj = visual.proj

    # Unpack meta lists (global indexing)
    ids = meta["id"]
    obj_labels = meta["object_label"]
    attack_words = meta["attack_word"]
    postit_area_pct = meta["postit_area_pct"]
    types = meta["type"]

    # unpack CLS/PATCH regression model tensors once
    if cls_patch_model is not None:
        patch_mean_D = cls_patch_model["patch_mean"].view(1, -1)  # [1,D]
        cls_mean_D   = cls_patch_model["cls_mean"].view(1, -1)    # [1,D]
        basis        = cls_patch_model["patch_basis"]             # [k,D]
        Wmat         = cls_patch_model["W"]                       # [D,D]
    else:
        patch_mean_D = None
        cls_mean_D   = None
        basis        = None
        Wmat         = None

    rows: List[Dict[str, object]] = []
    fooled: Dict[str, Dict[str, int]] = {}
    margin_accum: Dict[str, List[float]] = {}

    methods = METHODS  # keep single source of truth
    for m in methods:
        fooled[m] = {}
        margin_accum[m] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (device.startswith("cuda") and USE_AMP)
        else nullcontext()
    )


    with torch.no_grad():
        pbar = tqdm(loader_variant, desc=f"Eval {variant_name}", ncols=100)
        for batch_global_idxs, imgs_cpu in pbar:
            B = batch_global_idxs.shape[0]
            batch_global_idxs_list = batch_global_idxs.tolist()

            obj_b = [obj_labels[i] for i in batch_global_idxs_list]
            att_b = [attack_words[i] for i in batch_global_idxs_list]

            imgs = imgs_cpu.to(device, non_blocking=True)
            imgs = cast_imgs_to_visual_dtype(imgs, visual)

            tok = tokenize_pair_prompts(obj_b, att_b, device=device)  # [B,2,77]
            tok_flat = tok.view(2 * B, -1)

            # 3
            with amp_ctx:
                txt = model.encode_text(tok_flat)                     # [2B,D]
                txt = F.normalize(txt, dim=-1).view(B, 2, -1)         # [B,2,D]

                cls_raw, post = forward_visual_with_tokens(visual, imgs, TARGET_LAYERS_TUP)  # [B,D]
                cls_base = F.normalize(cls_raw, dim=-1)

                tokens_22 = post[LAYER_22]
                tokens_23 = post[LAYER_23]

                patch_l23_raw = build_patch_embed_from_tokens(
                    tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                    patch_mean=None, patch_pc=None,
                )
                patch_l23 = F.normalize(patch_l23_raw, dim=-1)

                if patch_means[LAYER_22] is not None and patch_pcs[LAYER_22] is not None and \
                   patch_means[LAYER_23] is not None and patch_pcs[LAYER_23] is not None:
                    p22_raw = build_patch_embed_from_tokens(
                        tokens_22, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        patch_mean=patch_means[LAYER_22], patch_pc=patch_pcs[LAYER_22],
                    )
                    p23_raw = build_patch_embed_from_tokens(
                        tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        patch_mean=patch_means[LAYER_23], patch_pc=patch_pcs[LAYER_23],
                    )
                    pdelta = F.normalize(p23_raw - p22_raw, dim=-1)
                else:
                    pdelta = cls_base

                # REG variants at L23
                def _reg_variant(reg_pc: Optional[torch.Tensor]) -> torch.Tensor:
                    if reg_means[LAYER_23] is None:
                        return cls_base
                    reg_raw, has_reg = build_reg_embed_from_tokens(
                        tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        reg_mean=reg_means[LAYER_23], reg_pc=reg_pc,
                    )
                    reg_norm = F.normalize(reg_raw, dim=-1)
                    out = cls_base.clone()
                    if has_reg.any():
                        out[has_reg] = reg_norm[has_reg]
                    return out

                reg_nopc = _reg_variant(None)
                reg_1pc = _reg_variant(reg_pcs[LAYER_23][0] if reg_pcs[LAYER_23] is not None else None)
                reg_8pc = _reg_variant(
                    reg_pcs[LAYER_23][:min(8, reg_pcs[LAYER_23].shape[0])] if reg_pcs[LAYER_23] is not None else None
                )

                # CLS-PATCHSUB + overlap fractions + k80 PCs
                if cls_patch_model is not None:
                    basis_use = basis.to(dtype=cls_raw.dtype)

                    # (A) Project (CLS - patch_mean) onto PATCH PCA subspace
                    x_center_patch = cls_raw - patch_mean_D            # [B,D]
                    coeff_patch = x_center_patch @ basis_use.T         # [B,k]
                    proj_x_patch = coeff_patch @ basis_use             # [B,D]
                    cls_patchsub = F.normalize(proj_x_patch, dim=-1)

                    numer_patch = proj_x_patch.pow(2).sum(dim=-1)       # [B]
                    denom_patch = x_center_patch.pow(2).sum(dim=-1).add_(1e-8)
                    cls_patch_frac_patch_center = numer_patch / denom_patch

                    # (B) CLS-centered denom
                    x_center_cls = cls_raw - cls_mean_D                # [B,D]
                    denom_cls = x_center_cls.pow(2).sum(dim=-1).add_(1e-8)
                    cls_patch_frac_cls_center = numer_patch / denom_cls

                    # (C) Fully symmetric: project (CLS - cls_mean) onto PATCH basis
                    coeff_cls = x_center_cls @ basis_use.T             # [B,k]
                    proj_x_cls = coeff_cls @ basis_use                 # [B,D]
                    numer_cls = proj_x_cls.pow(2).sum(dim=-1)           # [B]
                    cls_patch_frac_cls_center_sym = numer_cls / denom_cls

                    # (D) NEW heuristic: how many PATCH PCs needed to reach 80% of projection energy (symmetric)
                    # Energy per PC is coeff_cls^2. We use cumulative / total_proj_energy.
                    energy_pc = coeff_cls.pow(2)                        # [B,k]
                    total_proj = energy_pc.sum(dim=-1)                  # [B]
                    cum = torch.cumsum(energy_pc, dim=-1)               # [B,k]
                    frac = cum / (total_proj.unsqueeze(-1) + 1e-8)      # [B,k]

                    # Ensure existence of a True so argmax yields a valid index for rows with nonzero proj energy
                    mask = frac >= 0.80
                    mask[:, -1] = True

                    k_idx0 = mask.float().argmax(dim=-1)                # [B] 0-based
                    cls_patch_k80 = (k_idx0 + 1).to(dtype=torch.int64)   # [B] 1..k

                    # If projection energy is ~0, k80 is not meaningful → mark as -1 sentinel
                    zero_proj = total_proj <= 1e-12
                    if zero_proj.any():
                        cls_patch_k80 = cls_patch_k80.clone()
                        cls_patch_k80[zero_proj] = -1
                else:
                    cls_patchsub = cls_base
                    cls_patch_frac_patch_center = None
                    cls_patch_frac_cls_center = None
                    cls_patch_frac_cls_center_sym = None
                    cls_patch_k80 = None

                # CLS-PATCHREG
                if cls_patch_model is not None:
                    p_center = patch_l23_raw - patch_mean_D
                    c_center_hat = p_center @ Wmat.T
                    c_hat = c_center_hat + cls_mean_D
                    cls_patchreg = F.normalize(c_hat, dim=-1)
                else:
                    cls_patchreg = cls_base

                # Orthogonality cosines (CLS/PATCH/REG)
                cos_cls_patch = (cls_base * patch_l23).sum(dim=-1)  # [B]

                if reg_means[LAYER_23] is not None:
                    reg_raw_for_cos, has_reg_for_cos = build_reg_embed_from_tokens(
                        tokens_23, ln_post=ln_post, proj=proj, reg_threshold=REG_THRESHOLD,
                        reg_mean=reg_means[LAYER_23], reg_pc=None,
                    )
                    reg_norm_for_cos = F.normalize(reg_raw_for_cos, dim=-1)

                    cos_cls_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)
                    cos_patch_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)
                    if has_reg_for_cos.any():
                        cos_cls_reg[has_reg_for_cos] = (
                            cls_base[has_reg_for_cos] * reg_norm_for_cos[has_reg_for_cos]
                        ).sum(dim=-1)
                        cos_patch_reg[has_reg_for_cos] = (
                            patch_l23[has_reg_for_cos] * reg_norm_for_cos[has_reg_for_cos]
                        ).sum(dim=-1)
                else:
                    has_reg_for_cos = torch.zeros((B,), device=cls_base.device, dtype=torch.bool)
                    cos_cls_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)
                    cos_patch_reg = torch.full((B,), float("nan"), device=cls_base.device, dtype=cls_base.dtype)

                def _score(img_feat: torch.Tensor) -> MethodResult:
                    logits = (img_feat.unsqueeze(1) * txt).sum(dim=-1)  # [B,2]
                    probs = logits.softmax(dim=-1)                      # [B,2]
                    pred = probs.argmax(dim=-1)                         # 0=object, 1=attack
                    correct = (pred == 0)
                    return MethodResult(correct=correct, conf_obj=probs[:, 0], conf_att=probs[:, 1])

                scored = {
                    "CLS": _score(cls_base),
                    "CLS-PATCHSUB": _score(cls_patchsub),
                    "CLS-PATCHREG": _score(cls_patchreg),
                    "REG-L23-NOPC": _score(reg_nopc),
                    "REG-L23-1PC": _score(reg_1pc),
                    "REG-L23-8PC": _score(reg_8pc),
                    "PATCH-L23": _score(patch_l23),
                    "PATCHΔ": _score(pdelta),
                }

            # move to CPU once
            for bi, gi in enumerate(batch_global_idxs_list):
                row: Dict[str, object] = {
                    "id": ids[gi],
                    "variant": variant_name,
                    "object_label": obj_labels[gi],
                    "attack_word": attack_words[gi],
                    "postit_area_pct": float(postit_area_pct[gi]),
                    "type": types[gi],

                    # orthogonality / overlap signals (per-image)
                    "cos_cls_patch": float(cos_cls_patch[bi].item()),
                    "cos_cls_reg": float(cos_cls_reg[bi].item()),
                    "cos_patch_reg": float(cos_patch_reg[bi].item()),
                    "has_reg": bool(has_reg_for_cos[bi].item()),

                    # overlap metrics
                    "cls_patch_frac_patch_center": float(cls_patch_frac_patch_center[bi].item())
                        if cls_patch_frac_patch_center is not None else float("nan"),
                    "cls_patch_frac_cls_center": float(cls_patch_frac_cls_center[bi].item())
                        if cls_patch_frac_cls_center is not None else float("nan"),
                    "cls_patch_frac_cls_center_sym": float(cls_patch_frac_cls_center_sym[bi].item())
                        if cls_patch_frac_cls_center_sym is not None else float("nan"),

                    # k80 heuristic (int; -1 means undefined)
                    "cls_patch_k80": int(cls_patch_k80[bi].item()) if cls_patch_k80 is not None else -1,
                }

                for m in methods:
                    mr = scored[m]
                    is_corr = bool(mr.correct[bi].item())
                    c0 = float(mr.conf_obj[bi].item())
                    c1 = float(mr.conf_att[bi].item())
                    row[f"is_correct_{m}"] = is_corr
                    row[f"conf_obj_{m}"] = c0
                    row[f"conf_att_{m}"] = c1

                    margin_accum[m].append(c0 - c1)

                    if not is_corr:
                        aw = attack_words[gi]
                        fooled[m][aw] = fooled[m].get(aw, 0) + 1

                rows.append(row)

    df = pd.DataFrame(rows)

    # 8
    verbose: Dict[str, Dict[str, object]] = {}
    for m in methods:
        vc = sorted(fooled[m].items(), key=lambda kv: kv[1], reverse=True)[:50]
        margins = torch.tensor(margin_accum[m], dtype=torch.float32)
        verbose[m] = {
            "top_fooled_attack_words": vc,
            "margin_mean": float(margins.mean().item()) if margins.numel() else float("nan"),
            "margin_std": float(margins.std(unbiased=False).item()) if margins.numel() else float("nan"),
            "margin_min": float(margins.min().item()) if margins.numel() else float("nan"),
            "margin_max": float(margins.max().item()) if margins.numel() else float("nan"),
        }

    return df, verbose


def plot_mvt_cls_vs_method(
    overview_df: pd.DataFrame,
    method: str,                 # method != "CLS"
    out_path: str,
    title: str,
):
    sub = overview_df[
        (overview_df["variant"] == MVT_VARIANT_NAME) &
        (overview_df["method"].isin(["CLS", method]))
    ].copy()
    if sub.empty:
        return

    models = list(dict.fromkeys(sub["model_label"].tolist()))
    acc_cls = []
    acc_m = []
    for ml in models:
        a0 = sub[(sub["model_label"] == ml) & (sub["method"] == "CLS")]["accuracy"].values
        a1 = sub[(sub["model_label"] == ml) & (sub["method"] == method)]["accuracy"].values
        acc_cls.append(float(a0[0]) if len(a0) else float("nan"))
        acc_m.append(float(a1[0]) if len(a1) else float("nan"))

    x = list(range(len(models)))
    w = 0.38

    plt.figure(figsize=(max(10, 0.6 * len(models)), 6), dpi=PLOT_DPI)
    b0 = plt.bar([xi - w / 2 for xi in x], acc_cls, width=w, label="CLS")
    b1 = plt.bar([xi + w / 2 for xi in x], acc_m, width=w, label=method)

    plt.ylim(0.0, 1.0)
    plt.xticks(x, models, rotation=90)
    plt.title(title)
    plt.ylabel("Accuracy (MVT)")
    plt.legend()

    # Delta annotations on the METHOD bars
    for i, rect in enumerate(b1):
        y = rect.get_height()
        if not (y == y) or not (acc_cls[i] == acc_cls[i]):
            continue
        d = acc_m[i] - acc_cls[i]
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            min(0.99, y + 0.015),
            f"{d:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.subplots_adjust(bottom=0.45)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def summarize_metric(series: pd.Series, abs_thresh: Optional[float] = ORTH_ABS_THRESH) -> Dict[str, float]:
    s = series.dropna()
    if len(s) == 0:
        out = {
            "n_valid": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
        if abs_thresh is not None:
            out["p_abs_lt_thresh"] = float("nan")
        return out

    out = {
        "n_valid": float(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p05": float(s.quantile(0.05)),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
    }
    if abs_thresh is not None:
        out["p_abs_lt_thresh"] = float((s.abs() < abs_thresh).mean())
    return out


def plot_orth_metric_by_variant(
    orth_df: pd.DataFrame,
    metric: str,
    variant: str,
    out_path: str,
    title: str,
):
    sub = orth_df[(orth_df["metric"] == metric) & (orth_df["variant"] == variant)].copy()
    if sub.empty:
        return

    models = sub["model_label"].tolist()
    x = list(range(len(models)))
    means = sub["mean"].tolist()
    stds = sub["std"].tolist()

    plt.figure(figsize=(max(10, 0.6 * len(models)), 5), dpi=PLOT_DPI)
    plt.bar(x, means, yerr=stds, capsize=3)
    plt.xticks(x, models, rotation=90)
    plt.title(title)
    plt.ylabel(f"{metric} (mean ± std)")
    plt.subplots_adjust(bottom=0.45)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _safe_model_label(alias: str) -> str:
    # labels are now aliases; keep function so you don't touch plot code
    return alias

def plot_method_pair(
    overview_df: pd.DataFrame,
    method: str,
    v_a: str,
    v_b: str,
    out_path: str,
    title: str,
):
    """
    Bar chart: accuracy(v_a) and accuracy(v_b) per model for one method.
    """
    sub = overview_df[(overview_df["method"] == method) & (overview_df["variant"].isin([v_a, v_b]))].copy()
    if sub.empty:
        return

    models = list(dict.fromkeys(sub["model_label"].tolist()))  # preserve order
    acc_a = []
    acc_b = []
    for ml in models:
        a = sub[(sub["model_label"] == ml) & (sub["variant"] == v_a)]["accuracy"].values
        b = sub[(sub["model_label"] == ml) & (sub["variant"] == v_b)]["accuracy"].values
        acc_a.append(float(a[0]) if len(a) else float("nan"))
        acc_b.append(float(b[0]) if len(b) else float("nan"))

    x = list(range(len(models)))
    w = 0.38

    plt.figure(figsize=(max(10, 0.6 * len(models)), 6), dpi=PLOT_DPI)
    plt.bar([xi - w / 2 for xi in x], acc_a, width=w, label=v_a)
    plt.bar([xi + w / 2 for xi in x], acc_b, width=w, label=v_b)
    plt.ylim(0.0, 1.0)
    plt.xticks(x, models, rotation=90)
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplots_adjust(bottom=0.45)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_teacher_stats(teacher_df: pd.DataFrame, out_dir_plot: str):
    os.makedirs(out_dir_plot, exist_ok=True)
    if teacher_df.empty:
        return

    models = teacher_df["model_label"].tolist()
    x = list(range(len(models)))

    def _plot_col(col: str, title: str, fname: str):
        vals = teacher_df[col].tolist()
        plt.figure(figsize=(max(10, 0.6 * len(models)), 5), dpi=PLOT_DPI)
        plt.bar(x, vals)
        plt.xticks(x, models, rotation=90)
        plt.title(title)
        plt.ylabel(col)
        plt.subplots_adjust(bottom=0.45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_plot, fname), bbox_inches="tight")
        plt.close()

    _plot_col("teacher_cos_mean", "Teacher quality: mean cosine(C_hat, C) on NoSCAM build set", "teacher_cos_mean.png")
    _plot_col("teacher_mse", "Teacher quality: MSE(C_hat, C) on NoSCAM build set", "teacher_mse.png")




def main():

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | workers={NUM_WORKERS}")

    print("Loading BLISS-e-V/SCAM train split...")
    ds = load_dataset("BLISS-e-V/SCAM", split="train")
    N = len(ds)
    print(f"Total samples: {N}")

    meta = {
        "id": ds["id"],
        "object_label": ds["object_label"],
        "attack_word": ds["attack_word"],
        "postit_area_pct": ds["postit_area_pct"],
        "type": ds["type"],
    }

    idxs_by_variant: Dict[str, List[int]] = {}
    for v in VARIANTS:
        idxs_by_variant[v] = [i for i, s in enumerate(meta["id"]) if str(s).startswith(v)]
        print(f"  {v}: {len(idxs_by_variant[v])}")

    overview_rows: List[Dict[str, object]] = []
    orth_rows: List[Dict[str, object]] = []
    verbose_all: Dict[str, object] = {}
    teacher_rows: List[Dict[str, object]] = []

    # preprocess model (shared preprocessing)
    print("\nLoading preprocess from first model (shared preprocessing)...")
    model0_alias, model0_path = MODELS[0]
    model0, preprocess0, _ = load_openai_clip_anything(clip, model0_path, device=device, jit=False, strict=True)
    model0.eval().float()
    del model0

    sample = preprocess0(ds[0]["image"])
    C, H, W = sample.shape
    bytes_per = C * H * W * 2  # float16
    est_gb = (bytes_per * N) / (1024 ** 3)
    print(f"Preloading images to RAM as float16: shape=({N},{C},{H},{W}) ~ {est_gb:.2f} GB")

    all_imgs = torch.empty((N, C, H, W), dtype=torch.float16)  # CPU
    pp_ds = PreprocessAllDataset(ds, preprocess0)
    pp_loader = DataLoader(
        pp_ds,
        batch_size=PREPROCESS_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=_collate_idx_img,
    )

    for idxs, imgs in tqdm(pp_loader, desc="Preprocess -> RAM", ncols=100):
        all_imgs[idxs] = imgs.to(dtype=torch.float16, copy=False)

    all_imgs = all_imgs.share_memory_()

    variant_loaders: Dict[str, DataLoader] = {}
    for v in VARIANTS:
        vds = PreloadedVariantDataset(all_imgs, idxs_by_variant[v])
        variant_loaders[v] = DataLoader(
            vds,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            collate_fn=_collate_idx_img,
        )

    stats_ds = PreloadedVariantDataset(all_imgs, list(range(N)))
    loader_stats = DataLoader(
        stats_ds,
        batch_size=STATS_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=_collate_idx_img,
    )

    # teacher-build loader selection: ImageNet or NoSCAM
    teacher_build_info_global: Dict[str, object] = {}

    if use_imagenet_for_teacher:
        loader_teacher, teacher_build_info_global = build_imagenet_teacher_loader(
            preprocess=preprocess0,
            device=device,
            per_class=imagenet_per_class,
            batch_size=NOCS_BUILD_BATCH_SIZE,
        )
    else:
        noc_ds = PreloadedVariantDataset(all_imgs, idxs_by_variant["NoSCAM"])
        loader_teacher = DataLoader(
            noc_ds,
            batch_size=NOCS_BUILD_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            collate_fn=_collate_idx_img,
        )
        teacher_build_info_global = {
            "teacher_build_source": "NoSCAM(build set)",
            "imagenet_per_class": int(imagenet_per_class),
            "imagenet_n_classes": 0,
            "imagenet_n_images": 0,
            "imagenet_missing_roots": [],
        }


    # Build MVT loader
    print("\nLoading MVT (dataset-difficulty-CLIP) CSV + label vocab...")
    mvt_labels, mvt_label_to_idx = build_label_vocab(MVT_CSV_FILE)
    print(f"MVT unique labels: {len(mvt_labels)}")

    mvt_ds = CroppedImageCSVFileDataset(
        csv_file=MVT_CSV_FILE,
        image_folder=MVT_IMAGE_FOLDER,
        transform=preprocess0,
    )

    mvt_loader = DataLoader(
        mvt_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=_collate_img_label,
    )


    # Evaluate each model
    for mi, (model_alias, model_path) in enumerate(MODELS):
        model_label = _safe_model_label(model_alias)
        model_slug = sanitize_filename(model_alias)

        print(f"\n==================== Model {mi+1}/{len(MODELS)}: {model_label} ====================")

        model_clip, _, _ = load_openai_clip_anything(clip, model_path, device=device, jit=False, strict=True)
        model_clip.eval().float()
        model: CLIP = model_clip
        visual = model.visual

        reg_means, reg_pcs, patch_means, patch_pcs = collect_reg_and_patch_hidden_stats(
            visual=visual,
            loader=loader_stats,
            device=device,
            reg_threshold=REG_THRESHOLD,
            target_layers=TARGET_LAYERS_TUP,
        )

        cls_patch_model, teacher_stats = build_cls_patch_model(
            model=model,
            visual=visual,
            loader_noscam=loader_teacher,
            device=device,
            reg_threshold=REG_THRESHOLD,
        )
        teacher_stats_row = {
            "model_alias": model_alias,
            "model_path": model_path,
            "model_label": model_label,
            **teacher_build_info_global,
            **teacher_stats,
        }
        teacher_rows.append(teacher_stats_row)

        model_verbose: Dict[str, object] = {
            "model_alias": model_alias,
            "model_path": model_path,
            "model_label": model_label,
            "teacher_stats": teacher_stats,
            "variants": {},
        }

        for v in VARIANTS:
            if len(idxs_by_variant[v]) == 0:
                continue

            df_v, verbose_v = eval_variant_for_model(
                model=model,
                visual=visual,
                loader_variant=variant_loaders[v],
                meta=meta,
                device=device,
                reg_means=reg_means,
                reg_pcs=reg_pcs,
                patch_means=patch_means,
                patch_pcs=patch_pcs,
                cls_patch_model=cls_patch_model,
                variant_name=v,
            )

            out_csv = os.path.join(out_dir, f"details__{model_slug}__{v}.csv")
            df_v.to_csv(out_csv, index=False)

            n_total = len(df_v)
            methods = METHODS

            acc_line_parts = [f"{v} n={n_total}"]
            for m in methods:
                col = f"is_correct_{m}"
                acc = float(df_v[col].mean())
                acc_line_parts.append(f"{m}:{acc:.4f}")
                overview_rows.append({
                    "model_alias": model_alias,
                    "model_path": model_path,
                    "model_label": model_label,
                    "variant": v,
                    "method": m,
                    "accuracy": acc,
                    "n_correct": int(df_v[col].sum()),
                    "n_total": n_total,
                    "margin_mean": float(verbose_v[m]["margin_mean"]),
                    "margin_std": float(verbose_v[m]["margin_std"]),
                })

            # orthogonality summaries per (model, variant)
            has_reg_rate = float(df_v["has_reg"].mean()) if "has_reg" in df_v.columns else float("nan")
            orth_metrics = [
                "cos_cls_patch",
                "cos_cls_reg",
                "cos_patch_reg",
                "cls_patch_frac_patch_center",
                "cls_patch_frac_cls_center",
                "cls_patch_frac_cls_center_sym",
                "cls_patch_k80",
            ]
            orth_summary: Dict[str, object] = {"has_reg_rate": has_reg_rate, "abs_thresh": ORTH_ABS_THRESH}

            for met in orth_metrics:
                if met.startswith("cos_"):
                    summ = summarize_metric(df_v[met], abs_thresh=ORTH_ABS_THRESH)
                    orth_summary[met] = summ
                    orth_rows.append({
                        "model_path": model_path,
                        "model_label": model_label,
                        "variant": v,
                        "metric": met,
                        "abs_thresh": ORTH_ABS_THRESH,
                        "has_reg_rate": has_reg_rate,
                        **summ,
                    })
                else:
                    summ = summarize_metric(df_v[met], abs_thresh=None)
                    orth_summary[met] = summ
                    orth_rows.append({
                        "model_path": model_path,
                        "model_label": model_label,
                        "variant": v,
                        "metric": met,
                        "abs_thresh": float("nan"),
                        "has_reg_rate": has_reg_rate,
                        **summ,
                    })

            # brief print line
            print(" | ".join(acc_line_parts))

            model_verbose["variants"][v] = {
                "details_csv": out_csv,
                "verbose": verbose_v,
                "orthogonality": orth_summary,
            }

        # MVT evaluation (multi-class)
        print(c_hi("\nRunning MVT zero-shot evaluation (CLS vs METHODS)..."))

        # Build text features for THIS model (text encoder can differ across models)
        text_features_mvt = build_text_features_for_labels(
            model=model,
            labels=mvt_labels,
            device=device,
            prompt_template=MVT_PROMPT_TEMPLATE,
            batch_size=256,
        )

        df_mvt, verbose_mvt = eval_mvt_for_model(
            model=model,
            visual=visual,
            loader_mvt=mvt_loader,
            device=device,
            labels=mvt_labels,
            label_to_idx=mvt_label_to_idx,
            text_features=text_features_mvt,
            reg_means=reg_means,
            reg_pcs=reg_pcs,
            patch_means=patch_means,
            patch_pcs=patch_pcs,
            cls_patch_model=cls_patch_model,
        )

        out_csv_mvt = os.path.join(out_dir, f"details__{model_slug}__{MVT_VARIANT_NAME}.csv")
        df_mvt.to_csv(out_csv_mvt, index=False)

        n_total_mvt = len(df_mvt)
        acc_line_parts = [f"{MVT_VARIANT_NAME} n={n_total_mvt}"]
        for m in METHODS:
            col = f"is_correct_{m}"
            acc = float(df_mvt[col].mean()) if n_total_mvt > 0 else float("nan")
            acc_line_parts.append(f"{m}:{acc:.4f}")

            overview_rows.append({
                "model_alias": model_alias,
                "model_path": model_path,
                "model_label": model_label,
                "variant": MVT_VARIANT_NAME,
                "method": m,
                "accuracy": acc,
                "n_correct": int(df_mvt[col].sum()) if n_total_mvt > 0 else 0,
                "n_total": n_total_mvt,
                # reuse columns: margin_mean/std -> MVT uses top1-top2 margin12
                "margin_mean": float(verbose_mvt[m]["margin12_mean"]),
                "margin_std": float(verbose_mvt[m]["margin12_std"]),
            })

        # Orthogonality summaries for MVT
        has_reg_rate = float(df_mvt["has_reg"].mean()) if "has_reg" in df_mvt.columns and n_total_mvt > 0 else float("nan")
        orth_metrics = [
            "cos_cls_patch",
            "cos_cls_reg",
            "cos_patch_reg",
            "cls_patch_frac_patch_center",
            "cls_patch_frac_cls_center",
            "cls_patch_frac_cls_center_sym",
            "cls_patch_k80",
        ]

        orth_summary_mvt: Dict[str, object] = {"has_reg_rate": has_reg_rate, "abs_thresh": ORTH_ABS_THRESH}
        for met in orth_metrics:
            if met.startswith("cos_"):
                summ = summarize_metric(df_mvt[met], abs_thresh=ORTH_ABS_THRESH)
                orth_summary_mvt[met] = summ
                orth_rows.append({
                    "model_path": model_path,
                    "model_label": model_label,
                    "variant": MVT_VARIANT_NAME,
                    "metric": met,
                    "abs_thresh": ORTH_ABS_THRESH,
                    "has_reg_rate": has_reg_rate,
                    **summ,
                })
            else:
                summ = summarize_metric(df_mvt[met], abs_thresh=None)
                orth_summary_mvt[met] = summ
                orth_rows.append({
                    "model_path": model_path,
                    "model_label": model_label,
                    "variant": MVT_VARIANT_NAME,
                    "metric": met,
                    "abs_thresh": float("nan"),
                    "has_reg_rate": has_reg_rate,
                    **summ,
                })

        print(" | ".join(acc_line_parts))

        model_verbose["variants"][MVT_VARIANT_NAME] = {
            "details_csv": out_csv_mvt,
            "verbose": verbose_mvt,
            "orthogonality": orth_summary_mvt,
        }


        verbose_all[model_label] = model_verbose

        del model_clip
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Save overview + verbose + teacher stats + orthogonality
    overview_df = pd.DataFrame(overview_rows)
    overview_path = os.path.join(out_dir, "overview.csv")
    overview_df.to_csv(overview_path, index=False)

    teacher_df = pd.DataFrame(teacher_rows)
    teacher_path = os.path.join(out_dir, "teacher_stats.csv")
    teacher_df.to_csv(teacher_path, index=False)

    orth_df = pd.DataFrame(orth_rows)
    orth_path = os.path.join(out_dir, "orthogonality_overview.csv")
    orth_df.to_csv(orth_path, index=False)

    verbose_path = os.path.join(out_dir, "verbose.json")
    with open(verbose_path, "w", encoding="utf-8") as f:
        json.dump(verbose_all, f, indent=2)

    # Summary comparisons
    print(Fore.CYAN + Style.BRIGHT + "\n\n==================== SUMMARY (ranked by method) ====================" + Style.RESET_ALL)
    if not overview_df.empty:
        methods = sorted(overview_df["method"].unique().tolist())
        variants = [v for v in VARIANTS if v in set(overview_df["variant"].unique().tolist())]

        def _variant_acc_str(sub_m: pd.DataFrame, model_alias: str) -> str:
            parts = []
            for v in variants:
                vv = sub_m[(sub_m["model_alias"] == model_alias) & (sub_m["variant"] == v)]
                if len(vv) == 1:
                    parts.append(f"{v}={float(vv['accuracy'].iloc[0]):.4f}")
            return ", ".join(parts)

        for m in methods:
            sub_m = overview_df[overview_df["method"] == m].copy()
            if sub_m.empty:
                continue

            # Weighted mean across variants (weights = n_total), per model
            grp = (
                sub_m.groupby(["model_alias", "model_label"], as_index=False)
                     .apply(lambda g: pd.Series({
                         "acc_weighted": float((g["accuracy"] * g["n_total"]).sum() / max(1, g["n_total"].sum())),
                         "n_total_sum": int(g["n_total"].sum()),
                     }))
                     .reset_index(drop=True)
            )

            grp = grp.sort_values("acc_weighted", ascending=False)

            print(f"\n{m}")  # show all models ranked
            for _, r in grp.iterrows():
                alias = str(r["model_alias"])
                label = str(r["model_label"])
                accw = float(r["acc_weighted"])
                nsum = int(r["n_total_sum"])
                per_v = _variant_acc_str(sub_m, alias)
                # Keep it readable: alias first, then score, then per-variant breakdown
                print(f"  {label:<16s}  acc={accw:.4f}  N={nsum}  [{per_v}]")


    if not teacher_df.empty:
        best_cos = teacher_df.loc[teacher_df["teacher_cos_mean"].idxmax()]
        best_mse = teacher_df.loc[teacher_df["teacher_mse"].idxmin()]
        print("\nTeacher (regression) comparison:")
        print(f"  Best cos mean: {best_cos['teacher_cos_mean']:.4f} ({best_cos['model_label']})")
        print(f"  Best MSE:      {best_mse['teacher_mse']:.6f} ({best_mse['model_label']})")

    # Orthogonality summary table
    if not orth_df.empty:
        print(Fore.CYAN + Style.BRIGHT + "\n\n==================== Orthogonality (CLS/PATCH/REG) ====================" + Style.RESET_ALL)

        # preserve MODELS order (model_label == alias)
        _order = {alias: i for i, (alias, _path) in enumerate(MODELS)}
        model_labels = sorted(orth_df["model_label"].unique().tolist(), key=lambda x: _order.get(x, 10**9))

        metrics = [
            "cos_cls_patch",
            "cos_cls_reg",
            "cos_patch_reg",
            "cls_patch_frac_patch_center",
            "cls_patch_frac_cls_center",
            "cls_patch_frac_cls_center_sym",
            "cls_patch_k80",  # NEW
        ]

    # Plots
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if not overview_df.empty:
        for method in sorted(overview_df["method"].unique().tolist()):
            plot_method_pair(
                overview_df=overview_df,
                method=method,
                v_a="NoSCAM",
                v_b="SCAM",
                out_path=os.path.join(plot_dir, sanitize_filename(f"{method}__SCAM_vs_NoSCAM") + ".png"),
                title=f"{method}: SCAM vs NoSCAM",
            )
            plot_method_pair(
                overview_df=overview_df,
                method=method,
                v_a="NoSCAM",
                v_b="SynthSCAM",
                out_path=os.path.join(plot_dir, sanitize_filename(f"{method}__SynthSCAM_vs_NoSCAM") + ".png"),
                title=f"{method}: SynthSCAM vs NoSCAM",
            )

    plot_teacher_stats(teacher_df, out_dir_plot=plot_dir)


    # Orthogonality plots (mean ± std)
    if not orth_df.empty:
        orth_plot_dir = os.path.join(plot_dir, "orthogonality")
        os.makedirs(orth_plot_dir, exist_ok=True)
        for metric in [
            "cos_cls_patch",
            "cos_cls_reg",
            "cos_patch_reg",
            "cls_patch_frac_patch_center",
            "cls_patch_frac_cls_center",
            "cls_patch_frac_cls_center_sym",
            "cls_patch_k80",  # NEW
        ]:
            for v in VARIANTS:
                outp = os.path.join(orth_plot_dir, sanitize_filename(f"{metric}__{v}") + ".png")
                plot_orth_metric_by_variant(
                    orth_df=orth_df,
                    metric=metric,
                    variant=v,
                    out_path=outp,
                    title=f"{metric} (mean±std) | {v}",
                )


    # MVT plots (CLS vs method)
    if not overview_df.empty:
        mvt_methods = [m for m in METHODS if m != "CLS"]
        for m in mvt_methods:
            plot_mvt_cls_vs_method(
                overview_df=overview_df,
                method=m,
                out_path=os.path.join(plot_dir, sanitize_filename(f"{MVT_VARIANT_NAME}__CLS_vs__{m}") + ".png"),
                title=f"{MVT_VARIANT_NAME}: CLS vs {m}",
            )

    # full orthogonality report per model
    if not orth_df.empty:
        print("\nOrthogonality (CLS/PATCH/REG) full report per model:")

        # preserve MODELS order (model_label == alias)
        _order = {alias: i for i, (alias, _path) in enumerate(MODELS)}
        model_labels = sorted(orth_df["model_label"].unique().tolist(), key=lambda x: _order.get(x, 10**9))

        metrics = [
            "cos_cls_patch",
            "cos_cls_reg",
            "cos_patch_reg",
            "cls_patch_frac_patch_center",
            "cls_patch_frac_cls_center",
            "cls_patch_frac_cls_center_sym",
            "cls_patch_k80",
        ]

        for ml in model_labels:
            print(Fore.CYAN + Style.BRIGHT + f"\n--- Model: {ml} ---" + Style.RESET_ALL)

            for v in VARIANTS:
                sub_mv = orth_df[(orth_df["model_label"] == ml) & (orth_df["variant"] == v)].copy()
                if sub_mv.empty:
                    continue

                # has_reg_rate is duplicated across metric rows; take first non-null
                has_reg_rate_vals = sub_mv["has_reg_rate"].dropna().values
                has_reg_rate = float(has_reg_rate_vals[0]) if len(has_reg_rate_vals) else float("nan")

                print(f"  Variant: {v} | has_reg_rate={has_reg_rate:.4f}")

                for met in metrics:
                    row = sub_mv[sub_mv["metric"] == met]
                    if row.empty:
                        continue
                    r = row.iloc[0]

                    n_valid = int(r["n_valid"]) if pd.notna(r["n_valid"]) else 0
                    mean = float(r["mean"]) if pd.notna(r["mean"]) else float("nan")
                    std = float(r["std"]) if pd.notna(r["std"]) else float("nan")
                    p05 = float(r["p05"]) if pd.notna(r["p05"]) else float("nan")
                    p50 = float(r["p50"]) if pd.notna(r["p50"]) else float("nan")
                    p95 = float(r["p95"]) if pd.notna(r["p95"]) else float("nan")

                    if met.startswith("cos_"):
                        # p_abs_lt_thresh only exists for cos_ metrics (summarize_metric called with abs_thresh)
                        p_abs = float(r["p_abs_lt_thresh"]) if ("p_abs_lt_thresh" in r and pd.notna(r["p_abs_lt_thresh"])) else float("nan")
                        abs_thresh = ORTH_ABS_THRESH
                        print(
                            f"    {met:<24s} n={n_valid:<6d} mean={mean:+.4f} std={std:.4f} "
                            f"p05={p05:+.4f} p50={p50:+.4f} p95={p95:+.4f} | "
                            f"P(|cos|<{abs_thresh:g})={p_abs:.4f}"
                        )
                    elif met == "cls_patch_k80":
                        print(
                            f"    {met:<24s} n={n_valid:<6d} mean={mean:.2f} std={std:.2f} "
                            f"p05={p05:.2f} p50={p50:.2f} p95={p95:.2f}  "
                            f"(PCs to reach 80% proj energy; -1=undefined)"
                        )
                    else:
                        print(
                            f"    {met:<24s} n={n_valid:<6d} mean={mean:.4f} std={std:.4f} "
                            f"p05={p05:.4f} p50={p50:.4f} p95={p95:.4f}"
                        )

    # styled HEURISTICS SUMMARY
    def _get_orth_row(orth_df: pd.DataFrame, model_label: str, variant: str, metric: str) -> Optional[pd.Series]:
        sub = orth_df[
            (orth_df["model_label"] == model_label) &
            (orth_df["variant"] == variant) &
            (orth_df["metric"] == metric)
        ]
        return None if sub.empty else sub.iloc[0]

    def _safe_float(v) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _fmt_delta(x: float) -> str:
        return "nan" if not (x == x) else f"{x:+.3f}"

    def _band_frac(x: float) -> str:
        if not (x == x):
            return "unknown"
        if x >= 0.60:
            return "high"
        if x >= 0.35:
            return "moderate"
        return "low"

    def _band_teacher(cos_mean: float, mse: float) -> str:
        if not (cos_mean == cos_mean) or not (mse == mse):
            return "unknown"
        if cos_mean >= 0.97 and mse <= 0.020:
            return "very strong"
        if cos_mean >= 0.94 and mse <= 0.050:
            return "strong"
        if cos_mean >= 0.90:
            return "moderate"
        return "weak"

    def _mean_metric(ml: str, v: str, met: str) -> float:
        r = _get_orth_row(orth_df, ml, v, met)
        return _safe_float(r["mean"]) if r is not None else float("nan")

    def _pabs_metric(ml: str, v: str, met: str) -> float:
        r = _get_orth_row(orth_df, ml, v, met)
        if r is None or "p_abs_lt_thresh" not in r:
            return float("nan")
        return _safe_float(r["p_abs_lt_thresh"])

    def _acc(ml: str, v: str, m: str) -> float:
        sub = overview_df[
            (overview_df["model_label"] == ml) &
            (overview_df["variant"] == v) &
            (overview_df["method"] == m)
        ]
        return float(sub["accuracy"].iloc[0]) if len(sub) else float("nan")

    def _lift_line(ml: str, v: str) -> str:
        a_cls = _acc(ml, v, "CLS")
        lifts = []
        for m in ["CLS-PATCHSUB", "CLS-PATCHREG", "PATCHΔ"]:
            a_m = _acc(ml, v, m)
            lifts.append(f"{m} {_fmt_delta(a_m - a_cls)}")
        return " | ".join(lifts)

    baseline_label = "pretrained" if any(a == "pretrained" for a, _ in MODELS) else None
    focus_variants = [v for v in ["NoSCAM", "SCAM", "SynthSCAM"] if v in set(orth_df["variant"].unique())]
    v0 = "NoSCAM" if "NoSCAM" in focus_variants else (focus_variants[0] if focus_variants else "NoSCAM")

    sig_frac = 0.08
    sig_cos  = 0.05
    sig_lift = 0.05
    sig_teach_cos = 0.02
    sig_teach_mse = 0.02

    print("\n" + c_hi("==================== HEURISTICS SUMMARY ===================="))
    print(c_note(f"Focus variant for coupling/lifts: {v0}", bright=False))

    for model_alias, _path in MODELS:
        ml = _safe_model_label(model_alias)

        # Teacher stats (NoSCAM build)
        trow = teacher_df[teacher_df["model_label"] == ml]
        tcos = _safe_float(trow["teacher_cos_mean"].iloc[0]) if len(trow) else float("nan")
        tmse = _safe_float(trow["teacher_mse"].iloc[0]) if len(trow) else float("nan")
        tband = _band_teacher(tcos, tmse)

        print("\n" + c_hi(f"--- {ml} ---"))
        print(
            c_note("Teacher fit (NoSCAM CLS from PATCH): ", bright=True) +
            f"cos_mean={tcos:.4f} | mse={tmse:.6f} | strength={tband}"
        )

        # Geometry (means from v0)
        frac_sym = _mean_metric(ml, v0, "cls_patch_frac_cls_center_sym")
        frac_pc  = _mean_metric(ml, v0, "cls_patch_frac_patch_center")
        frac_cc  = _mean_metric(ml, v0, "cls_patch_frac_cls_center")
        k80      = _mean_metric(ml, v0, "cls_patch_k80")

        cos_cp   = _mean_metric(ml, v0, "cos_cls_patch")
        cos_cr   = _mean_metric(ml, v0, "cos_cls_reg")
        cos_pr   = _mean_metric(ml, v0, "cos_patch_reg")

        pabs_cp  = _pabs_metric(ml, v0, "cos_cls_patch")
        pabs_cr  = _pabs_metric(ml, v0, "cos_cls_reg")

        # Headline: patch-subspace coupling + k80
        msg = (
            f"Patch-subspace coupling ({v0}): "
            f"sym={frac_sym:.3f} ({_band_frac(frac_sym)}) | "
            f"patch_center={frac_pc:.3f} | cls_normed={frac_cc:.3f} | "
            f"k80_mean={k80:.2f} PCs"
        )
        if (frac_sym == frac_sym) and frac_sym >= 0.60:
            print(c_ok(msg))
        elif (frac_sym == frac_sym) and frac_sym < 0.35:
            print(c_dev(msg))
        else:
            print(c_note(msg))

        # EXPLANATORY SENTENCE (k80 meaning)
        if (k80 == k80) and k80 >= 0:
            # small k80 => patch-subspace is low-dimensional (energy concentrates quickly)
            if k80 <= 8:
                print(c_ok(f"Interpretation: patch-subspace is low-dimensional (≤~{int(round(k80))} PCs capture 80%);\n-> strong concentration / shared modes."))
            elif k80 >= 32:
                print(c_dev(f"Interpretation: patch-subspace is high-dimensional (~{int(round(k80))} PCs for 80%);\n-> patch variation is spread out (less compressible)."))
            else:
                print(c_note(f"Interpretation: patch-subspace dimensionality is moderate (~{int(round(k80))} PCs for 80%)."))

        # EXPLANATORY SENTENCE (mean-offset / variance mismatch)
        if (frac_sym == frac_sym) and (frac_pc == frac_pc):
            d_m = frac_pc - frac_sym
            if abs(d_m) >= 0.10:
                sign = "larger" if d_m > 0 else "smaller"
                print(
                    c_note(
                        f"Heuristic: large difference (patch_center - sym = {d_m:+.3f}) suggests mean-offset effects\nor "
                        f"CLS-vs-PATCH variance mismatch; patch-centered projection is {sign} than symmetric projection.",
                        bright=False
                    )
                )

        # Alignment (means) + near-orthogonality rates
        print(
            c_note(f"Alignment ({v0} means): ", bright=True) +
            f"cos(CLS,PATCH)={cos_cp:+.3f} | cos(CLS,REG)={cos_cr:+.3f} | cos(PATCH,REG)={cos_pr:+.3f}"
        )
        print(
            c_note(f"Near-orthogonality P(|cos|<{ORTH_ABS_THRESH:g}) ({v0}): ", bright=True) +
            f"CLS/PATCH={pabs_cp:.3f} | CLS/REG={pabs_cr:.3f}"
        )

        # EXPLANATORY SENTENCE (teacher-vs-geometry meaning)
        if (tcos == tcos) and (frac_sym == frac_sym):
            if tcos >= 0.95 and frac_sym >= 0.60:
                print(
                    c_ok(
                        "Interpretation: CLS is largely patch-subspace dominated and a linear CLS-from-PATCH map fits very well\n"
                        "(strong coupling; teacher succeeds because geometry already matches)."
                    )
                )
            elif tcos >= 0.95 and frac_sym < 0.35:
                print(
                    c_dev(
                        "Interpretation: linear CLS-from-PATCH fits well despite weak symmetric patch-subspace overlap;\n"
                        "CLS likely uses additional directions outside the dominant PATCH PCA subspace\nthat are still predictable "
                        "from PATCH (structured cross-covariances)."
                    )
                )
            elif tcos < 0.90 and frac_sym >= 0.60:
                print(
                    c_bad(
                        "Interpretation: CLS lies in patch subspace but is not well predicted by a simple linear map from PATCH;\n"
                        "this can indicate nonlinearity, dataset shift, or that your PATCH embedding is missing information CLS uses."
                    )
                )
            else:
                print(
                    c_note(
                        "Interpretation: moderate/weak coupling and/or predictability; CLS and PATCH are partially decoupled.",
                        bright=False
                    )
                )

        # Lifts: does PATCHREG/PATCHSUB/PATCHΔ help vs CLS?
        lift_str = _lift_line(ml, v0)
        d_patchreg = _acc(ml, v0, "CLS-PATCHREG") - _acc(ml, v0, "CLS")
        if d_patchreg >= sig_lift:
            print(c_ok(f"Method lifts vs CLS ({v0}): {lift_str}"))
        elif d_patchreg <= -sig_lift:
            print(c_dev(f"Method lifts vs CLS ({v0}): {lift_str}"))
        else:
            print(c_note(f"Method lifts vs CLS ({v0}): {lift_str}"))

        # Attack drops + SCAM shift explanatory sentences
        if "SCAM" in focus_variants and v0 != "SCAM":
            drop_cls = _acc(ml, v0, "CLS") - _acc(ml, "SCAM", "CLS")
            drop_pr  = _acc(ml, v0, "CLS-PATCHREG") - _acc(ml, "SCAM", "CLS-PATCHREG")
            drop_pd  = _acc(ml, v0, "PATCHΔ") - _acc(ml, "SCAM", "PATCHΔ")

            drop_line = f"SCAM drops ({v0} - SCAM): CLS={drop_cls:+.3f} | PATCHREG={drop_pr:+.3f} | PATCHΔ={drop_pd:+.3f}"
            if (drop_cls == drop_cls) and drop_cls <= 0.08:
                print(c_ok(drop_line))
            elif (drop_cls == drop_cls) and drop_cls >= 0.20:
                print(c_bad(drop_line))
            else:
                print(c_note(drop_line))

            # SCAM shift interpretation based on geometry deltas
            frac_sym_scam = _mean_metric(ml, "SCAM", "cls_patch_frac_cls_center_sym")
            cos_cp_scam   = _mean_metric(ml, "SCAM", "cos_cls_patch")

            if (frac_sym_scam == frac_sym_scam) and (frac_sym == frac_sym):
                d = frac_sym_scam - frac_sym
                if abs(d) >= sig_frac:
                    print(
                        c_note(
                            f"SCAM shift: symmetric overlap delta={d:+.3f} (SCAM - {v0}). "
                            f"{'More' if d > 0 else 'Less'} CLS variation lies in PATCH subspace under SCAM.",
                            bright=False
                        )
                    )
            if (cos_cp_scam == cos_cp_scam) and (cos_cp == cos_cp):
                d = cos_cp_scam - cos_cp
                if abs(d) >= sig_cos:
                    print(
                        c_note(
                            f"SCAM shift: cos(CLS,PATCH) delta={d:+.3f} (SCAM - {v0}).",
                            bright=False
                        )
                    )

        # Baseline comparison vs pretrained (signal, not spam)
        if baseline_label is not None and ml != baseline_label and baseline_label in set(orth_df["model_label"].unique()):
            base_frac_sym = _mean_metric(baseline_label, v0, "cls_patch_frac_cls_center_sym")
            base_cos_cp   = _mean_metric(baseline_label, v0, "cos_cls_patch")

            parts = []
            d_sym = float("nan")
            if (frac_sym == frac_sym) and (base_frac_sym == base_frac_sym):
                d_sym = frac_sym - base_frac_sym
                if abs(d_sym) >= sig_frac:
                    parts.append(f"sym_overlap {_fmt_delta(d_sym)}")
            if (cos_cp == cos_cp) and (base_cos_cp == base_cos_cp):
                d_cos = cos_cp - base_cos_cp
                if abs(d_cos) >= sig_cos:
                    parts.append(f"cos(CLS,PATCH) {_fmt_delta(d_cos)}")

            base_trow = teacher_df[teacher_df["model_label"] == baseline_label]
            base_tcos = _safe_float(base_trow["teacher_cos_mean"].iloc[0]) if len(base_trow) else float("nan")
            base_tmse = _safe_float(base_trow["teacher_mse"].iloc[0]) if len(base_trow) else float("nan")
            if (tcos == tcos) and (base_tcos == base_tcos):
                d = tcos - base_tcos
                if abs(d) >= sig_teach_cos:
                    parts.append(f"teacher_cos {_fmt_delta(d)}")
            if (tmse == tmse) and (base_tmse == base_tmse):
                d = tmse - base_tmse
                if abs(d) >= sig_teach_mse:
                    parts.append(f"teacher_mse {_fmt_delta(d)}")

            if parts:
                line = f"vs pretrained ({v0}): " + " | ".join(parts)
                if (d_sym == d_sym) and d_sym >= sig_frac:
                    print(c_ok(line))
                elif (d_sym == d_sym) and d_sym <= -sig_frac:
                    print(c_bad(line))
                else:
                    print(c_note(line))

    print("\nSaved:")
    print(f"  overview: {overview_path}")
    print(f"  teacher:  {teacher_path}")
    print(f"  orth:     {orth_path}")
    print(f"  verbose:  {verbose_path}")
    print(f"  plots:    {plot_dir}")


if __name__ == "__main__":
    main()