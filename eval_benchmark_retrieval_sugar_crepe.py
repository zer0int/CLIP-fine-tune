"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

SugarCrepe Benchmark (Image -> Text binary retrieval)
=====================================================

Official SugarCrepe repo :
  https://github.com/RAIVNLab/sugar-crepe/tree/main
  SugarCrepe evaluates faithful vision-language compositionality: given an image, a model must
  choose the correct (positive) caption over a hard negative caption that differs only by small
  compositional changes (add/replace/swap object/attribute/relation).

Images used by SugarCrepe:
  COCO 2017 val images (val2017.zip)
  Download: http://images.cocodataset.org/zips/val2017.zip

-> Labels are included in this repo! You only need the images from 'val2017.zip'.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
import time
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import math

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
from utils_clip_loader.cliptools import fix_random_seed
fix_random_seed()


# Your COCO val2017 images 
COCO_IMAGE_ROOT = "path/to/COCO/val2017"


DATA_ROOT = "utils_datasets/sugar_crepe"
OUT_BASE = "out_eval_benchmarks/retrieval_sugar_crepe"


MODELS: List[Tuple[str, str]] = [
    ("pretrained", "ViT-L/14"),
    ("gmp-clip", "zer0int/CLIP-GmP-ViT-L-14"),
    ("ko-clip", "zer0int/CLIP-KO-LITE-TypoAttack-Attn-Dropout-ViT-L-14"),
    ("regr-norm", "zer0int/CLIP-Regression-ViT-L-14"),
    ("regr-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
]

# If True: build dataset once using preprocess from MODELS[0] and reuse it.
SHARE_PREPROCESS_ACROSS_MODELS = True

# DataLoader knobs
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128 if device == "cuda" else 64
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
PIN_MEMORY = (device == "cuda")

# Save per-sample CSVs? (not required; can be large)
SAVE_PER_SAMPLE_CSV = False

# SugarCrepe files (expected in DATA_ROOT)
SPLITS: Dict[str, str] = {
    "add_obj": "add_obj.json",
    "add_att": "add_att.json",
    "replace_obj": "replace_obj.json",
    "replace_att": "replace_att.json",
    "replace_rel": "replace_rel.json",
    "swap_obj": "swap_obj.json",
    "swap_att": "swap_att.json",
}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_clip_model(name_or_path: str, device_str: str):
    """
    Returns: model, preprocess_fn
    """
    model, preprocess_fn, _ = load_openai_clip_anything(
        clip, name_or_path, device=device_str, jit=False, strict=True
    )
    model = model.eval().float()
    return model, preprocess_fn


@dataclass
class SugarCrepeItem:
    category: str
    key: str
    filename: str
    caption: str
    negative_caption: str


def load_sugar_crepe(data_root: str) -> Dict[str, List[SugarCrepeItem]]:
    """
    Loads SugarCrepe JSONs into: category -> list[items]
    """
    buckets: Dict[str, List[SugarCrepeItem]] = {}

    for cat, fname in SPLITS.items():
        path = os.path.join(data_root, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing SugarCrepe JSON: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items: List[SugarCrepeItem] = []
        for k, ex in raw.items():
            items.append(
                SugarCrepeItem(
                    category=cat,
                    key=str(k),
                    filename=str(ex["filename"]),
                    caption=str(ex["caption"]),
                    negative_caption=str(ex["negative_caption"]),
                )
            )
        buckets[cat] = items

    return buckets


def _open_image_rgb_with_retries(
    img_path: str,
    retries: int = 8,
    base_delay_s: float = 0.02,
    backoff: float = 1.8,
    jitter_s: float = 0.01,
):
    """
    Windows-friendly image open:
      - open file handle explicitly
      - force PIL to fully decode via .load()
      - return an image that no longer depends on the underlying file handle
      - retry on transient PermissionError / OSError
    """
    last_exc: Optional[BaseException] = None
    delay = base_delay_s

    for attempt in range(retries):
        try:
            # Use a real file handle so closing is deterministic (PIL can be lazy otherwise).
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img.load()                 # force decode while handle is open
                img = img.convert("RGB")   # materialize RGB
                return img                 # safe: handle is closed on exiting `with`
        except (PermissionError, OSError) as e:
            last_exc = e
            # exponential backoff + jitter
            time.sleep(delay + random.random() * jitter_s)
            delay *= backoff

    # If we got here, all retries failed.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to open image (unknown error): {img_path}")


class SugarCrepeDataset(Dataset):
    """
    Returns:
      image_tensor, pos_caption, neg_caption, meta(dict)
    """
    def __init__(self, coco_image_root: str, items: List[SugarCrepeItem], preprocess_fn):
        self.coco_image_root = coco_image_root
        self.items = items
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img_path = os.path.join(self.coco_image_root, it.filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"COCO image not found: {img_path}")

        # use robust open with retries + immediate close
        img = _open_image_rgb_with_retries(img_path)

        img_tensor = self.preprocess_fn(img)

        meta = {
            "category": it.category,
            "key": it.key,
            "filename": it.filename,
        }
        return img_tensor, it.caption, it.negative_caption, meta


def sugar_collate_fn(batch):
    """
    Keep captions as lists of strings and metas as list-of-dicts.
    """
    images, pos_caps, neg_caps, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(pos_caps), list(neg_caps), list(metas)


def _make_loader(ds: Dataset) -> DataLoader:
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        collate_fn=sugar_collate_fn,
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(ds, **kwargs)


@torch.inference_mode()
def evaluate_sugar_crepe_loader(
    model,
    loader: DataLoader,
    device_str: str,
    desc: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Batched evaluation:
      - Encode images (B,D)
      - Tokenize+encode pos captions (B,D)
      - Tokenize+encode neg captions (B,D)
      - Compare cosine sims: pos_sim > neg_sim

    Returns:
      df_per_sample, summary_dict
    """
    rows: List[Dict[str, Any]] = []
    n_correct = 0
    n_total = 0

    for images, pos_caps, neg_caps, metas in tqdm(loader, desc=desc, ncols=90):
        # --- images ---
        if device_str == "cuda":
            images = images.to(device_str, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                img_feat = model.encode_image(images)
        else:
            images = images.to(device_str)
            img_feat = model.encode_image(images)

        img_feat = img_feat.float()
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)

        # text (pos/neg)
        pos_tok = clip.tokenize(pos_caps)  # CPU
        neg_tok = clip.tokenize(neg_caps)  # CPU

        if device_str == "cuda":
            pos_tok = pos_tok.to(device_str, non_blocking=True)
            neg_tok = neg_tok.to(device_str, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pos_feat = model.encode_text(pos_tok)
                neg_feat = model.encode_text(neg_tok)
        else:
            pos_tok = pos_tok.to(device_str)
            neg_tok = neg_tok.to(device_str)
            pos_feat = model.encode_text(pos_tok)
            neg_feat = model.encode_text(neg_tok)

        pos_feat = pos_feat.float()
        neg_feat = neg_feat.float()
        pos_feat = pos_feat / (pos_feat.norm(dim=-1, keepdim=True) + 1e-12)
        neg_feat = neg_feat / (neg_feat.norm(dim=-1, keepdim=True) + 1e-12)

        # cosine sims per sample
        pos_sim = (img_feat * pos_feat).sum(dim=-1)  # (B,)
        neg_sim = (img_feat * neg_feat).sum(dim=-1)  # (B,)

        correct_mask = pos_sim > neg_sim
        n_correct += int(correct_mask.sum().item())
        n_total += int(correct_mask.numel())

        # margins for diagnostics
        margin = (pos_sim - neg_sim).detach().cpu()

        for i in range(len(metas)):
            meta = metas[i]
            is_correct = bool(correct_mask[i].item())
            rows.append(
                {
                    "is_correct": is_correct,
                    "pos_sim": float(pos_sim[i].item()),
                    "neg_sim": float(neg_sim[i].item()),
                    "margin": float(margin[i].item()),
                    "pos_caption": pos_caps[i],
                    "neg_caption": neg_caps[i],
                    "category": meta.get("category", ""),
                    "key": meta.get("key", ""),
                    "filename": meta.get("filename", ""),
                }
            )

    acc = (n_correct / n_total) if n_total else float("nan")
    df = pd.DataFrame(rows)

    summary = {
        "n_total": int(n_total),
        "n_correct": int(n_correct),
        "accuracy": float(acc),
        "margin_mean": float(df["margin"].mean()) if len(df) else float("nan"),
        "margin_std": float(df["margin"].std()) if len(df) else float("nan"),
    }
    return df, summary


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


def save_plot_sugarcrepe_accuracy(
    out_dir: str,
    plot_key: str,
    title: str,
    model_aliases: List[str],
    accuracies: List[float],
    color_map: Optional[Dict[str, str]] = None,
):
    xs = list(range(len(model_aliases)))

    fig_w = max(10.0, 0.65 * len(model_aliases))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    # apply per-model colors
    colors = None
    if color_map is not None:
        colors = [color_map.get(a, None) for a in model_aliases]

    bars = ax.bar(xs, accuracies, width=0.6, color=colors, label="Accuracy")

    # headroom so saturated bars + labels don't get clipped
    y_top = 1.05
    ax.set_ylim(0.0, y_top)

    # ticks: ensure 1.0 is the last labeled tick (no 1.05 tick)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{t:.1f}" for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])

    # offset scales with axis range
    y_off = 0.015 * y_top

    def _annotate_bars(bars):
        for b in bars:
            h = float(b.get_height())
            if not math.isfinite(h):
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

    _annotate_bars(bars) 

    ax.set_xticks(xs)
    ax.set_xticklabels(model_aliases, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Accuracy")

    # Single-series legend tends to be redundant, but keeping it "same type of plot".
    # Move it above so it never overlaps labels.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        frameon=False,
    )

    # reserve margins for legend + x tick labels
    fig.subplots_adjust(top=0.82, bottom=0.28)
    fig.tight_layout()

    fn = str(plot_key).lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(out_dir, f"{fn}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)



def main():
    _ensure_dir(OUT_BASE)

    print("\n===================================================")
    print("SugarCrepe (COCO val2017) — Image->Text Binary Choice")
    print("===================================================\n")

    print(f"[Device] {device}")
    print(f"[Paths] COCO_IMAGE_ROOT={COCO_IMAGE_ROOT}")
    print(f"[Paths] DATA_ROOT={DATA_ROOT}")
    print(f"[Out]   OUT_BASE={OUT_BASE}")
    print(f"[DL]    batch={BATCH_SIZE} workers={NUM_WORKERS} pin={PIN_MEMORY} persist={PERSISTENT_WORKERS}")

    # Load SugarCrepe jsons
    buckets = load_sugar_crepe(DATA_ROOT)
    for cat, items in buckets.items():
        print(f"[Data] {cat:12s}: {len(items):5d} samples")
    if not buckets:
        raise SystemExit("No SugarCrepe categories loaded (empty buckets).")

    # Build loaders
    shared_preprocess_fn = None
    shared_loaders: Optional[Dict[str, DataLoader]] = None

    if SHARE_PREPROCESS_ACROSS_MODELS:
        base_ref = MODELS[0][1]
        print(f"\n[Preprocess] Building shared preprocess from: {base_ref}")
        tmp_model, shared_preprocess_fn = load_clip_model(base_ref, device_str=device)
        del tmp_model
        if device == "cuda":
            torch.cuda.empty_cache()

        shared_loaders = {}
        for cat, items in buckets.items():
            ds = SugarCrepeDataset(COCO_IMAGE_ROOT, items, preprocess_fn=shared_preprocess_fn)
            shared_loaders[cat] = _make_loader(ds)

    all_records: List[Dict[str, Any]] = []

    # Evaluate models
    for alias, model_ref in MODELS:
        print("\n" + "=" * 80)
        print(f"[Run] {alias}  ::  {model_ref}")
        print("=" * 80)

        model, preprocess_fn = load_clip_model(model_ref, device_str=device)

        # Build loaders (per model) if not shared
        loaders: Dict[str, DataLoader] = {}
        if SHARE_PREPROCESS_ACROSS_MODELS:
            loaders = shared_loaders or {}
        else:
            for cat, items in buckets.items():
                ds = SugarCrepeDataset(COCO_IMAGE_ROOT, items, preprocess_fn=preprocess_fn)
                loaders[cat] = _make_loader(ds)

        per_model_metrics: Dict[str, Any] = {}
        per_model_rows: List[Dict[str, Any]] = []

        for cat, loader in loaders.items():
            df, summ = evaluate_sugar_crepe_loader(
                model=model,
                loader=loader,
                device_str=device,
                desc=f"{alias} | {cat}",
            )

            print(f"[ZS] {cat:12s}: {summ['n_correct']:4d}/{summ['n_total']:4d} = {summ['accuracy']:.4f} "
                  f"(margin μ={summ['margin_mean']:.4f}, σ={summ['margin_std']:.4f})")

            per_model_metrics[cat] = {
                "accuracy": summ["accuracy"],
                "n_total": summ["n_total"],
                "n_correct": summ["n_correct"],
                "margin_mean": summ["margin_mean"],
                "margin_std": summ["margin_std"],
            }

            if SAVE_PER_SAMPLE_CSV:
                out_csv = os.path.join(OUT_BASE, f"{alias}__{cat}__per_sample.csv")
                df.to_csv(out_csv, index=False)
                print(f"[Save] {out_csv}")

            # for combined summary table
            per_model_rows.append(
                {
                    "model_alias": alias,
                    "model": model_ref,
                    "category": cat,
                    **summ,
                }
            )

        # Aggregate mean across categories
        accs = [float(per_model_metrics[c]["accuracy"]) for c in per_model_metrics.keys()]
        mean_acc = sum(accs) / len(accs) if accs else float("nan")
        per_model_metrics["mean"] = mean_acc

        # Save per-model JSON
        out_json = os.path.join(OUT_BASE, f"{alias}__metrics.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(per_model_metrics, f, indent=2)
        print(f"[Save] {out_json}")

        # Append to global records
        all_records.extend(per_model_rows)

        # Cleanup
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    print("\n" + "#" * 80)
    print("[FINAL SUMMARY] All models x SugarCrepe categories")
    print("#" * 80)

    df_all = pd.DataFrame(all_records)
    if len(df_all) == 0:
        print("No results produced.")
        return

    # Save combined CSV
    out_summary_csv = os.path.join(OUT_BASE, "summary_all_models.csv")
    df_all.to_csv(out_summary_csv, index=False)
    print(f"[Save] {out_summary_csv}")

    # Print in run order
    model_order = [a for a, _ in MODELS]
    cat_order = list(SPLITS.keys())

    def _key(r):
        return (model_order.index(r["model_alias"]), cat_order.index(r["category"]))

    print("\n[By run order]")
    for r in sorted(all_records, key=_key):
        print(
            f"{r['model_alias']:>12s}  {r['category']:<12s}  "
            f"acc={r['accuracy']:.4f}  n={r['n_total']:4d}  "
            f"margin μ={r['margin_mean']:.4f}  σ={r['margin_std']:.4f}"
        )

    # Ranked by mean accuracy, with mean margin as tie-breaker / extra signal
    means = (
        df_all.groupby(["model_alias", "model"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), mean_margin=("margin_mean", "mean"))
        .sort_values(["mean_accuracy", "mean_margin"], ascending=[False, False])
    )

    print("\n[Ranked] by mean accuracy (desc), tie-break by mean margin (desc):")
    for _, r in means.iterrows():
        print(
            f"{str(r['model_alias']):>12s}  "
            f"mean_acc={float(r['mean_accuracy']):.4f}  "
            f"mean_margin={float(r['mean_margin']):.4f}  "
            f"model={str(r['model'])}"
        )

    out_rank_csv = os.path.join(OUT_BASE, "ranked_by_mean_accuracy_and_margin.csv")
    means.to_csv(out_rank_csv, index=False)
    print(f"[Save] {out_rank_csv}")

    # ------------------------------------------------------------
    # Plots: per-category accuracy + overall mean accuracy
    # ------------------------------------------------------------
    model_aliases = model_order  # keep same order as your runs
    color_map = make_model_color_map(model_aliases)

    # build quick lookup: (alias, category) -> accuracy
    acc_map: Dict[Tuple[str, str], float] = {}
    for _, r in df_all.iterrows():
        acc_map[(str(r["model_alias"]), str(r["category"]))] = float(r["accuracy"])

    # per-category plots
    for cat in cat_order:
        accs = [acc_map.get((alias, cat), float("nan")) for alias in model_aliases]
        save_plot_sugarcrepe_accuracy(
            out_dir=OUT_BASE,
            plot_key=f"acc__{cat}",
            title=f"SugarCrepe Accuracy — {cat}",
            model_aliases=model_aliases,
            accuracies=accs,
            color_map=color_map,
        )

    # mean-accuracy plot (aligned to run order, not sorted order)
    mean_map = {str(r["model_alias"]): float(r["mean_accuracy"]) for _, r in means.iterrows()}
    mean_accs = [mean_map.get(alias, float("nan")) for alias in model_aliases]
    save_plot_sugarcrepe_accuracy(
        out_dir=OUT_BASE,
        plot_key="acc__mean_over_categories",
        title="SugarCrepe Accuracy — Mean over categories",
        model_aliases=model_aliases,
        accuracies=mean_accs,
    )

    print(f"[Saved] Plots           -> {OUT_BASE}")

    # Per-category Top-5 / Bottom-5 models
    print("\n" + "#" * 80)
    print("[PER-CATEGORY RANKINGS] Top-5 and Bottom-5 by accuracy")
    print("#" * 80)

    for cat in cat_order:
        df_cat = df_all[df_all["category"] == cat].copy()
        if len(df_cat) == 0:
            continue

        df_cat = df_cat.sort_values(["accuracy", "margin_mean"], ascending=[False, False])

        top_k = min(5, len(df_cat))
        bot_k = min(5, len(df_cat))

        print(f"\n[{cat}] Top-{top_k}")
        for _, r in df_cat.head(top_k).iterrows():
            print(
                f"{str(r['model_alias']):>12s}  "
                f"acc={float(r['accuracy']):.4f}  "
                f"margin μ={float(r['margin_mean']):.4f}  "
                f"n={int(r['n_total']):4d}"
            )

        print(f"[{cat}] Bottom-{bot_k}")
        for _, r in df_cat.tail(bot_k).sort_values(["accuracy", "margin_mean"], ascending=[True, True]).iterrows():
            print(
                f"{str(r['model_alias']):>12s}  "
                f"acc={float(r['accuracy']):.4f}  "
                f"margin μ={float(r['margin_mean']):.4f}  "
                f"n={int(r['n_total']):4d}"
            )

    print(f"\nDone. Outputs in: {OUT_BASE}")



if __name__ == "__main__":
    main()