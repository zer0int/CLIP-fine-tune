"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

Typographic Attack Benchmarks
------------------------------------------------------------
Unified benchmark script for:
  - BLISS-e-V/SCAM (HF dataset): variants {NoSCAM, SCAM, SynthSCAM}
  - RTA-100 (local folder): filenames encode correct vs distractor labels

# ------------------------------------------------------------------
# RTA-100:
# 1000 photos with post-it notes stuck to objects.
# Download:
# https://github.com/azuma164/Defense-Prefix/blob/main/rta100.zip
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# BLISS-e-V/SCAM:
# Conveniently loaded from HuggingFace (will be auto-downloaded):
# https://huggingface.co/datasets/BLISS-e-V/SCAM
# ------------------------------------------------------------------

Features:
  - Multiple models evaluated sequentially (alias + path/name)
  - Prints per-model intermediate results and a final summary table
  - Efficient DataLoader: pin_memory, prefetch_factor, persistent_workers, non_blocking
  - Pre-tokenize all unique labels ONCE per model (cache encoded text features)
  - Dataset option: --dataset scam | rta | both
  - Saves per-model CSVs + summary CSV
------------------------------------------------------------
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import Counter
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

from utils_clip_loader.cliptools import fix_random_seed
fix_random_seed()

device = "cuda" if torch.cuda.is_available() else "cpu"

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


# RTA-100 local path
DEFAULT_RTA_DIR = "path/to/rta100"

# RTA-100 (must be available locally) / SCAM (auto-loaded from HF):
dataset = "both" # ["scam", "rta", "both"] <- use "scam", will be pulled from HF hub

# Output base dir
OUT_BASE = "out_eval_benchmarks/zeroshot_typo_attack"
os.makedirs(OUT_BASE, exist_ok=True)

# Dataloader
BATCH_SIZE = 128 if device == "cuda" else 64
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
PIN_MEMORY = (device == "cuda")

# Text encoding chunk size
TEXT_BATCH_SIZE = 4096

# Ablation experiments [sets to 0]
ablate_head = False
ablate_neurons = False

REG_NEURONS: Dict[int, List[int]] = {
    11: [9, 987, 1967, 2555, 3661, 3784],
    12: [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868],
}

BLOCK_HEADS: Dict[int, List[int]] = {
    12: [10, 5],
    22: [10, 9],
}

def attach_reg_neuron_nuke_hooks(visual: torch.nn.Module) -> List[Any]:
    """
    Zero specified MLP expanded dims at c_fc output (pre-gelu), for blocks in REG_NEURONS.
    Returns list of hook handles so we can remove them.
    """
    handles: List[Any] = []
    if not REG_NEURONS:
        return handles

    print("[INFO] Attaching register-neuron nuke hooks...")
    for block_idx, block in enumerate(visual.transformer.resblocks):
        if block_idx not in REG_NEURONS:
            continue
        idxs = torch.tensor(REG_NEURONS[block_idx], dtype=torch.long)

        c_fc = block.mlp.c_fc if hasattr(block.mlp, "c_fc") else block.mlp[0]

        def make_hook(idxs_: torch.Tensor, blk_idx: int):
            def hook(_module, _inp, output):
                out = output.clone()
                out[..., idxs_.to(out.device)] = 0.0
                return out
            hook.__name__ = f"reg_nuke_block_{blk_idx}"
            return hook

        h = c_fc.register_forward_hook(make_hook(idxs, block_idx))
        handles.append(h)
        print(f"[INFO] Hook on block {block_idx} c_fc for neurons {REG_NEURONS[block_idx]}")
    return handles


def ablate_head_output_all_layers(model, head_idx: int = None, block_heads: Dict[int, List[int]] = None):
    """
    "Real" head ablation for torch.nn.MultiheadAttention-style CLIP blocks by zeroing
    the *input to out_proj* (i.e., concatenated heads BEFORE mixing), via a forward_pre_hook
    on block.attn.out_proj.

    Supports either:
      (A) legacy: ablate_head_output_all_layers(model, head_idx=10) -> ablates that head in ALL layers
      (B) new:    ablate_head_output_all_layers(model, block_heads={11:[10],12:[3,7]}) -> per-block heads

    Returns list of hook handles so we can remove them.
    """
    handles = []
    if not hasattr(model, "visual") or not hasattr(model.visual, "transformer"):
        raise ValueError("Head ablation requested, but model.visual.transformer not found (CNN visual backbone?).")

    resblocks = list(model.visual.transformer.resblocks)

    # decide mode
    if block_heads is None:
        if head_idx is None:
            # fall back to global BLOCK_HEADS if provided, otherwise error
            if "BLOCK_HEADS" in globals() and BLOCK_HEADS:
                block_heads = BLOCK_HEADS
            else:
                raise ValueError("Head ablation requested but neither head_idx nor block_heads/BLOCK_HEADS provided.")
        else:
            # legacy: apply to all blocks
            block_heads = {i: [int(head_idx)] for i in range(len(resblocks))}

    # sanity: normalize + validate basic structure
    norm_block_heads: Dict[int, List[int]] = {}
    for blk_idx, heads in block_heads.items():
        if heads is None:
            continue
        if not isinstance(heads, (list, tuple)):
            raise TypeError(f"block_heads[{blk_idx}] must be a list/tuple of head indices, got {type(heads)}")
        heads_int = [int(h) for h in heads]
        norm_block_heads[int(blk_idx)] = heads_int

    for block_idx, block in enumerate(resblocks):
        if block_idx not in norm_block_heads:
            continue

        if not hasattr(block, "attn"):
            raise ValueError(f"Block {block_idx} has no .attn; cannot ablate heads.")

        attn = block.attn
        heads_this = norm_block_heads[block_idx]

        # prehook on out_proj for true per-head zeroing
        if hasattr(attn, "out_proj") and attn.out_proj is not None:
            out_proj = attn.out_proj

            def make_outproj_prehook(heads_list, blk_idx: int, attn_module):
                def prehook(_module, inputs):
                    # Linear gets (x,) where x shape is (..., D). D == embed_dim.
                    if not isinstance(inputs, (tuple, list)) or len(inputs) < 1:
                        return inputs

                    x = inputs[0]
                    if not torch.is_tensor(x):
                        return inputs

                    if not hasattr(attn_module, "num_heads"):
                        raise ValueError("Attention module missing num_heads. Update hook for your CLIP impl.")

                    num_heads = int(attn_module.num_heads)
                    D = int(x.shape[-1])
                    if D % num_heads != 0:
                        raise ValueError(
                            f"out_proj input dim {D} not divisible by num_heads {num_heads} (block {blk_idx})."
                        )
                    head_dim = int(getattr(attn_module, "head_dim", D // num_heads))

                    bad = [h for h in heads_list if h < 0 or h >= num_heads]
                    if bad:
                        raise ValueError(f"Invalid head indices {bad} for num_heads={num_heads} (block {blk_idx}).")

                    # x: (..., D) -> (..., num_heads, head_dim)
                    x2 = x.clone().reshape(*x.shape[:-1], num_heads, head_dim)
                    x2[..., heads_list, :] = 0.0
                    x_new = x2.reshape(*x.shape[:-1], D)

                    # return same structure, replacing only the first arg
                    return (x_new, *inputs[1:])

                prehook.__name__ = f"ablate_heads_outproj_in_block_{blk_idx}"
                return prehook

            h = out_proj.register_forward_pre_hook(make_outproj_prehook(heads_this, block_idx, attn))
            handles.append(h)
            print(f"[INFO] Hook on block {block_idx} attn.out_proj (pre) for heads {heads_this}")

        else:
            # Fallback: if there's no out_proj to hook, we can't guarantee true "head" semantics.
            raise ValueError(
                f"Block {block_idx} attn has no out_proj; cannot do true per-head ablation for this CLIP implementation."
            )

    return handles



def load_clip_model(name_or_path: str, device: str):
    model, preprocess_fn, _ = load_openai_clip_anything(clip, name_or_path, device=device, jit=False, strict=True)
    model = model.eval().float()
    return model, preprocess_fn


@dataclass
class PairSample:
    """
    A single binary choice:
      - image: PIL Image or already transformed Tensor (we'll keep PIL in dataset)
      - correct_label: the intended object label
      - distractor_label: the typographic attack word / distractor
      - meta: optional extra info for saving/debugging
    """
    image: Any
    correct_label: str
    distractor_label: str
    meta: Dict[str, Any]


class PairDataset(Dataset):
    """
    Wraps a list of PairSample and applies CLIP preprocess in __getitem__.
    """
    def __init__(self, samples: List[PairSample], preprocess_fn):
        self.samples = samples
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_tensor = self.preprocess_fn(s.image)  # CLIP preprocess expects PIL Image
        return img_tensor, s.correct_label, s.distractor_label, s.meta


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


def load_rta_samples(rta_dir: str) -> List[PairSample]:
    """
    Parses filenames like: label=<CORRECT>_text=<DISTRACTOR>.jpg|png|...
    """
    pattern = re.compile(r"label=(.+?)_text=(.+?)\.(jpg|jpeg|png)$", re.IGNORECASE)
    samples: List[PairSample] = []

    if not os.path.isdir(rta_dir):
        raise FileNotFoundError(f"RTA dir not found: {rta_dir}")

    for fname in os.listdir(rta_dir):
        m = pattern.match(fname)
        if not m:
            continue
        correct_label = m.group(1)
        distractor_label = m.group(2)
        path = os.path.join(rta_dir, fname)

        img = Image.open(path).convert("RGB")
        samples.append(
            PairSample(
                image=img,
                correct_label=str(correct_label),
                distractor_label=str(distractor_label),
                meta=dict(
                    filename=fname,
                    dataset="RTA100",
                    variant="RTA100",
                ),
            )
        )
    return samples

# TEXT FEATURE CACHE (per model)
def _prompt(label: str) -> str:
    return f"a photo of a {label}"

def compute_text_feature_cache(
    model,
    unique_labels: List[str],
    device: str,
    text_batch_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    """
    Pre-tokenize & encode each unique label once, returning a dict label -> normalized feature [D].
    Cached on device (half on CUDA).
    """
    prompts = [_prompt(lab) for lab in unique_labels]
    tokens = clip.tokenize(prompts)  # CPU int64 [C, 77]

    feats: List[torch.Tensor] = []
    model.eval()

    with torch.inference_mode():
        for i in range(0, tokens.shape[0], text_batch_size):
            tok = tokens[i:i + text_batch_size].to(device)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    t = model.encode_text(tok)
            else:
                t = model.encode_text(tok)

            t = t.float()
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
            feats.append(t)

    text_features = torch.cat(feats, dim=0).to(device)
    if device == "cuda":
        text_features = text_features.half()

    # Build dict
    cache: Dict[str, torch.Tensor] = {}
    for i, lab in enumerate(unique_labels):
        cache[lab] = text_features[i]
    return cache


@torch.inference_mode()
def evaluate_pair_dataset(
    model,
    dataloader: DataLoader,
    text_cache: Dict[str, torch.Tensor],
    device: str,
    desc: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate binary choice per sample using cached text features.
    Returns:
      - per-sample results df
      - summary dict
    """
    rows = []
    correct = 0
    total = 0

    model.eval()

    for batch in tqdm(dataloader, desc=desc, ncols=90):
        images, correct_labels, distractor_labels, metas = batch

        if device == "cuda":
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                img_feat = model.encode_image(images)
        else:
            images = images.to(device)
            img_feat = model.encode_image(images)

        img_feat = img_feat.float()
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)
        if device == "cuda":
            img_feat = img_feat.half()

        if isinstance(metas, dict):
            # metas is dict-of-lists; reconstruct list-of-dicts
            metas = [{k: metas[k][i] for k in metas.keys()} for i in range(len(correct_labels))]

        # Build per-sample logits against 2 cached text vectors
        for i in range(img_feat.shape[0]):
            obj = str(correct_labels[i])
            atk = str(distractor_labels[i])

            t_obj = text_cache[obj]  # [D]
            t_atk = text_cache[atk]  # [D]

            # cosine similarities
            s_obj = float((img_feat[i] @ t_obj).item())
            s_atk = float((img_feat[i] @ t_atk).item())

            # softmax over two
            # stable softmax for 2 elements
            m = max(s_obj, s_atk)
            e0 = torch.exp(torch.tensor(s_obj - m))
            e1 = torch.exp(torch.tensor(s_atk - m))
            p_obj = float((e0 / (e0 + e1)).item())
            p_atk = float((e1 / (e0 + e1)).item())

            pred_is_obj = (p_obj >= p_atk)
            is_correct = bool(pred_is_obj)

            if is_correct:
                correct += 1
            total += 1

            meta = metas[i]
            # metas can arrive as dict-like or as a python object depending on collate;
            # safest: keep only a few known fields if present
            row = {
                "correct_label": obj,
                "distractor_label": atk,
                "pred_label": obj if pred_is_obj else atk,
                "is_correct": is_correct,
                "confidence_correct": p_obj,
                "confidence_distractor": p_atk,
                "cos_sim_correct": s_obj,
                "cos_sim_distractor": s_atk,
                "margin_prob": p_obj - p_atk,
                "margin_cos": s_obj - s_atk,
            }

            # Try to unpack meta
            if isinstance(meta, dict):
                for k in ["id", "filename", "dataset", "variant", "postit_area_pct", "type"]:
                    if k in meta:
                        row[k] = meta[k]

            rows.append(row)

    acc = (correct / total) if total > 0 else float("nan")
    df = pd.DataFrame(rows)

    summary = {
        "n_total": int(total),
        "n_correct": int(correct),
        "accuracy": float(acc),
        "margin_cos_mean": float(df["margin_cos"].mean()) if len(df) else float("nan"),
        "margin_cos_std": float(df["margin_cos"].std()) if len(df) else float("nan"),
        "margin_prob_mean": float(df["margin_prob"].mean()) if len(df) else float("nan"),
        "margin_prob_std": float(df["margin_prob"].std()) if len(df) else float("nan"),
    }
    return df, summary


def pair_collate_fn(batch):
    """
    Keep metas as a list-of-dicts (do NOT let default_collate turn it into dict-of-lists).
    """
    images, correct_labels, distractor_labels, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(correct_labels), list(distractor_labels), list(metas)


def _make_loader(dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int,
                 pin_memory: bool, persistent_workers: bool) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        collate_fn=pair_collate_fn,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def make_model_color_map(model_aliases: List[str]) -> Dict[str, str]:
    """
    Deterministic per-alias colors using Matplotlib qualitative palettes.
    Uses tab10 for <=10, tab20 for <=20, otherwise hsv fallback.
    Returns: alias -> hex color string (e.g. '#1f77b4')
    """
    n = len(model_aliases)
    if n <= 10:
        cmap = get_cmap("tab10")
        cols = [to_hex(cmap(i)) for i in range(n)]
    elif n <= 20:
        cmap = get_cmap("tab20")
        cols = [to_hex(cmap(i)) for i in range(n)]
    else:
        cmap = get_cmap("hsv")
        cols = [to_hex(cmap(i / max(1, n - 1))) for i in range(n)]
    return {alias: cols[i] for i, alias in enumerate(model_aliases)}


def save_plot_accuracy_task(
    out_dir: str,
    dataset_tag: str,
    title: str,
    model_aliases: List[str],
    acc_by_model: List[float],
    color_map: Optional[Dict[str, str]] = None,
):
    xs = list(range(len(model_aliases)))

    # wider if many models (prevents crowded x labels)
    fig_w = max(10.0, 0.65 * len(model_aliases))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    base_colors = None
    if color_map is not None:
        base_colors = [color_map.get(a, None) for a in model_aliases]

    bars = ax.bar(xs, acc_by_model, label="Accuracy", color=base_colors, alpha=0.85)

    # headroom above 1.0 but last tick at 1.0
    y_top = 1.05
    ax.set_ylim(0.0, y_top)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # annotation offset scales with axis range
    y_off = 0.015 * y_top

    # annotate each bar with value (rotated 90°)
    for b in bars:
        h = float(b.get_height())
        if not pd.notna(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(h + y_off, y_top - 1e-6),
            f"{h:.2f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            clip_on=False,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(model_aliases, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Accuracy")

    # reserve margins for rotated labels + above-bar annotations
    fig.subplots_adjust(top=0.90, bottom=0.28)
    fig.tight_layout()

    safe_tag = dataset_tag.replace("::", "_").replace("/", "_").replace(" ", "_")
    path = os.path.join(out_dir, f"acc__{safe_tag}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)



def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = OUT_BASE
    os.makedirs(out_dir, exist_ok=True)

    pin_memory = (device == "cuda")
    persistent_workers = True

    print("\n===================================================")
    print("Typographic Attack: BLISS-e-V/SCAM & RTA-100 (ZS)")
    print("===================================================\n")

    print(f"\n[Device] {device}")
    print(f"[Config] dataset={dataset}  batch={BATCH_SIZE}  workers={NUM_WORKERS}  pin={pin_memory}")

    # Load datasets -> build sample lists
    scam_buckets: Dict[str, List[PairSample]] = {}
    rta_samples: List[PairSample] = []

    if dataset in ("scam", "both"):
        print("[Data] loading SCAM from HuggingFace: BLISS-e-V/SCAM")
        scam_buckets = load_scam_samples()
        for v, lst in scam_buckets.items():
            print(f"  [SCAM] {v:9s}: {len(lst)} samples")

    if dataset in ("rta", "both"):
        print(f"[Data] loading RTA-100 from: {DEFAULT_RTA_DIR}")
        rta_samples = load_rta_samples(DEFAULT_RTA_DIR)
        print(f"  [RTA100] {len(rta_samples)} samples")

    # Global label set across selected datasets (for per-model text cache)
    global_labels = set()
    for v, lst in scam_buckets.items():
        for s in lst:
            global_labels.add(s.correct_label)
            global_labels.add(s.distractor_label)
    for s in rta_samples:
        global_labels.add(s.correct_label)
        global_labels.add(s.distractor_label)
    global_labels = sorted(list(global_labels))
    print(f"[Labels] unique labels across selected datasets: {len(global_labels):,}")

    # Load ONE preprocess pipeline and reuse for all models
    base_model_ref = MODELS[0][1] if len(MODELS) else "ViT-L/14"
    _tmp_model, preprocess_fn = load_clip_model(base_model_ref, device=device)
    del _tmp_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Build datasets + loaders once (preprocess is shared)
    loaders: Dict[str, DataLoader] = {}

    if dataset in ("scam", "both"):
        for variant, samples in scam_buckets.items():
            if len(samples) == 0:
                continue
            ds_variant = PairDataset(samples, preprocess_fn=preprocess_fn)
            loaders[f"SCAM::{variant}"] = _make_loader(
                ds_variant,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

    if dataset in ("rta", "both") and len(rta_samples) > 0:
        ds_rta = PairDataset(rta_samples, preprocess_fn=preprocess_fn)
        loaders["RTA100"] = _make_loader(
            ds_rta,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    if len(loaders) == 0:
        raise SystemExit("No datasets to evaluate (empty loaders). Check --dataset and paths.")


    all_model_records: List[Dict[str, Any]] = []

    for alias, model_ref in MODELS:
        print("\n" + "=" * 80)
        print(f"[Run] {alias}  ::  {model_ref}")
        print("=" * 80)

        model, _ = load_clip_model(model_ref, device=device)

        hook_handles = None
        if ablate_head:
            print("---------------------------------------")
            print(f"WARNING: Ablating Attention Heads")
            print("---------------------------------------")
            hook_handles = ablate_head_output_all_layers(model, block_heads=BLOCK_HEADS)

        neuron_hooks = None
        if ablate_neurons:
            print("---------------------------------------")
            print(f"WARNING: Ablating Register Neurons")
            print("---------------------------------------")
            neuron_hooks = attach_reg_neuron_nuke_hooks(model.visual)

        # Precompute ALL text features once per model
        print("[Text] encoding all unique labels (cached per model)...")
        text_cache = compute_text_feature_cache(
            model=model,
            unique_labels=global_labels,
            device=device,
            text_batch_size=TEXT_BATCH_SIZE,
        )

        # Evaluate each loader
        per_model_summaries: List[Dict[str, Any]] = []
        for tag, loader in loaders.items():
            print(f"\n--- Evaluating: {tag} ---")
            df, summ = evaluate_pair_dataset(
                model=model,
                dataloader=loader,
                text_cache=text_cache,
                device=device,
                desc=f"{alias} | {tag}",
            )

            # Save per-sample CSV
            safe_tag = tag.replace("::", "_").replace("/", "_").replace(" ", "_")
            out_csv = os.path.join(out_dir, f"{alias}__{safe_tag}__results.csv")
            df.to_csv(out_csv, index=False)
            print(f"[Save] {out_csv}")

            # Print summary + a few extra stats
            print(f"[ZS] {tag}: {summ['n_correct']}/{summ['n_total']} = {summ['accuracy']:.4f}")
            print("[Margins] cosine margin describe:")
            if len(df):
                print(df["margin_cos"].describe())

                # Top attack words (when fooled)
                wrong = df[~df["is_correct"]]
                if "distractor_label" in wrong.columns and len(wrong):
                    print("\nTop 20 distractor labels that fooled CLIP:")
                    print(wrong["distractor_label"].value_counts().head(20))

            # Record for final summary
            record = {
                "model_alias": alias,
                "model": model_ref,
                "dataset": tag,
                **summ,
            }
            per_model_summaries.append(record)
            all_model_records.append(record)

        # Intermediate per-model print (consistent order)
        print("\n" + "-" * 80)
        print(f"[Model Summary] {alias}")
        for r in per_model_summaries:
            print(f"{r['dataset']:18s}  acc={r['accuracy']:.4f}  n={r['n_total']}  cos_margin_mean={r['margin_cos_mean']:.4f}")
        print("-" * 80)

        # Cleanup
        if hook_handles is not None:
            for h in hook_handles:
                h.remove()
        if neuron_hooks is not None:
            for h in neuron_hooks:
                h.remove()

        del text_cache
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "#" * 80)
    print("[FINAL SUMMARY] All models x datasets")
    print("#" * 80)

    df_all = pd.DataFrame(all_model_records)
    if len(df_all) == 0:
        print("No results produced.")
        return

    # Print in original run order (MODELS order, then dataset order)
    model_order = [a for a, _ in MODELS]
    dataset_order = list(loaders.keys())

    def _key(row):
        return (model_order.index(row["model_alias"]), dataset_order.index(row["dataset"]))

    rows_sorted = sorted(all_model_records, key=_key)
    for r in rows_sorted:
        print(f"{r['model_alias']:>12s}  {r['dataset']:<18s}  acc={r['accuracy']:.4f}  n={r['n_total']:4d}")

    print("\n[Sorted] by accuracy (desc):")
    df_sorted = df_all.sort_values("accuracy", ascending=False)
    for _, r in df_sorted.iterrows():
        print(f"{str(r['model_alias']):>12s}  {str(r['dataset']):<18s}  acc={float(r['accuracy']):.4f}  n={int(r['n_total']):4d}")

    out_summary_csv = os.path.join(out_dir, "summary_all_models.csv")
    df_all.to_csv(out_summary_csv, index=False)
    print(f"\n[Save] {out_summary_csv}")

    out_summary_csv = os.path.join(out_dir, "summary_all_models.csv")
    df_all.to_csv(out_summary_csv, index=False)
    print(f"\n[Save] {out_summary_csv}")

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model_order = [a for a, _ in MODELS]
    dataset_order = list(loaders.keys())

    color_map = make_model_color_map(model_order)

    for ds_tag in dataset_order:
        accs: List[float] = []
        for alias in model_order:
            sub = df_all[(df_all["model_alias"] == alias) & (df_all["dataset"] == ds_tag)]
            if len(sub) == 0:
                accs.append(float("nan"))
            else:
                accs.append(float(sub["accuracy"].iloc[0]))

        save_plot_accuracy_task(
            out_dir=plots_dir,
            dataset_tag=ds_tag,
            title=f"Typographic Attack Accuracy — {ds_tag}",
            model_aliases=model_order,
            acc_by_model=accs,
            color_map=color_map,
        )

    print(f"[Save] Plots -> {plots_dir}")
    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()