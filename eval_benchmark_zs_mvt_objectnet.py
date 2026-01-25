"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

ImageNet/ObjectNet MVT - a very hard benchmark for zero-shot.

Dataset download (free, no sign-up): https://objectnet.dev/mvt/
- 5k labels included with this repo: utils_datasets/mvt/human_responses_5k.csv
- Full labels (recommended) -> use from downloaded dataset

"""
import os
import time
import random
import torch
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
from tqdm import tqdm
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

csv_file = 'path/to/dataset-difficulty-CLIP/data_release_2023/human_responses.csv'
image_folder = "path/to/dataset-difficulty-CLIP/data_release_2023/all/"

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

# Output dir
local_path = "out_eval_benchmarks/zeroshot_mvt"
os.makedirs(local_path, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 48
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True
PIN_MEMORY = (device == "cuda")
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

# ============================================================
# DATASET
# ============================================================
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
                # close handle immediately after decode/convert
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

        # give up
        raise last_exc

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image_name = self._images[idx]
        image_path = os.path.join(self.image_folder, image_name)

        try:
            # use resilient open
            image = self._safe_open_rgb(image_path)
        except Exception as e:
            if self.skip_on_fail:
                return None
            raise

        if self.transform is not None:
            image = self.transform(image)

        label = self._labels[idx]
        return image, label

def drop_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def build_label_vocab(csv_path: str) -> Tuple[List[str], Dict[str, int]]:
    df = pd.read_csv(csv_path, usecols=["label"])
    labels = df["label"].astype(str)
    uniq = labels.unique().tolist()
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    return uniq, label_to_idx


def compute_text_features(
    model,
    classnames: List[str],
    device: str,
    text_batch_size: int = 4096,
) -> torch.Tensor:
    """
    Pre-tokenize once, encode texts once per model
    Returns normalized text features [C, D] on device.
    """

    tokens = clip.tokenize(classnames)  # [C, 77] int64 on CPU
    feats = []

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
    return text_features


def evaluate_model_zeroshot(
    model,
    dataloader: DataLoader,
    label_to_idx: Dict[str, int],
    text_features: torch.Tensor,
    device: str,
) -> float:
    """
    Correct zero-shot eval against a fixed label vocabulary
    Predict argmax over ALL labels
    """
    correct = 0
    total = 0

    model.eval()
    with torch.inference_mode():
        for batch_images, batch_labels in tqdm(dataloader, desc="eval", leave=False):
            # Move images
            if device == "cuda":
                batch_images = batch_images.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = model.encode_image(batch_images)
            else:
                batch_images = batch_images.to(device)
                image_features = model.encode_image(batch_images)

            image_features = image_features.float()
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)

            if device == "cuda":
                image_features = image_features.half()

            # Similarity
            logits = image_features @ text_features.T  # [B, C]
            pred = logits.argmax(dim=-1).tolist()

            # Compare
            for i, lab in enumerate(batch_labels):
                gt = label_to_idx.get(str(lab), None)
                if gt is not None and pred[i] == gt:
                    correct += 1
                total += 1

    return (correct / total) if total > 0 else 0.0

def load_clip_model(name_or_path: str, device: str):
    model, preprocess_fn, _ = load_openai_clip_anything(clip, name_or_path, device=device, jit=False, strict=True)
    model = model.eval().float()
    return model, preprocess_fn


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


def save_plot_mvt_zs_accuracy(
    out_dir: str,
    title: str,
    model_aliases: List[str],
    accuracies_by_model: List[float],   # 0..1
    color_map: Optional[Dict[str, str]] = None,
):
    xs = list(range(len(model_aliases)))

    # Wider if many models (prevents crowded x labels)
    fig_w = max(10.0, 0.65 * len(model_aliases))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    colors = None
    if color_map is not None:
        colors = [color_map.get(a, None) for a in model_aliases]

    bars = ax.bar(xs, accuracies_by_model, color=colors)

    # headroom above 1.0 but last tick at 1.0
    y_top = 1.05
    ax.set_ylim(0.0, y_top)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Annotation offset scales with axis range
    y_off = 0.015 * y_top

    for b in bars:
        h = float(b.get_height())
        if not math.isfinite(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(h + y_off, y_top - 1e-6),
            f"{h:.3f}",
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

    fig.subplots_adjust(top=0.92, bottom=0.28)
    fig.tight_layout()

    path = os.path.join(out_dir, "mvt_zs_accuracy.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n=====================================")
    print("ImageNet / ObjectNet MVT (ZS)")
    print("=====================================\n")
    
    print(f"[Device] {device}")
    classnames, label_to_idx = build_label_vocab(csv_file)
    print(f"[Vocab] unique labels: {len(classnames):,}")

    # load ONE preprocess pipeline and reuse for all models
    base_model_ref = MODELS[0][1] if len(MODELS) > 0 else "ViT-L/14"
    _tmp_model, preprocess_fn = load_clip_model(base_model_ref, device=device)
    del _tmp_model
    if device == "cuda":
        torch.cuda.empty_cache()

    dataset = CroppedImageCSVFileDataset(csv_file, image_folder, transform=preprocess_fn)

    # DataLoader with pin_memory + prefetch + persistent_workers
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
    )
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    dataloader = DataLoader(dataset, **loader_kwargs, collate_fn=drop_none_collate)

    results = []

    for alias, model_ref in MODELS:
        print("\n" + "=" * 70)
        print(f"[Run] {alias}  ::  {model_ref}")
        print("=" * 70)

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

        # Precompute text features once per model
        print("[Text] encoding label vocabulary...")
        text_features = compute_text_features(
            model=model,
            classnames=classnames,
            device=device,
            text_batch_size=TEXT_BATCH_SIZE,
        )

        print("[Eval] running zero-shot eval...")
        acc = evaluate_model_zeroshot(
            model=model,
            dataloader=dataloader,
            label_to_idx=label_to_idx,
            text_features=text_features,
            device=device,
        )

        rec = dict(
            alias=alias,
            model=model_ref,
            accuracy=float(acc),
            ablate_head=bool(ablate_head),
            head=int(target_head) if ablate_head else None,
        )
        results.append(rec)

        # Intermediate print + per-model save
        print(f"[Done] {alias}: accuracy={acc:.6f}")
        out_txt = os.path.join(local_path, f"imagenet-objectnet_{alias}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"alias: {alias}\n")
            f.write(f"model: {model_ref}\n")
            f.write(f"accuracy: {acc:.8f}\n")
            f.write(f"ablate_head: {ablate_head}\n")
            f.write(f"head: {target_head if ablate_head else 'n/a'}\n")
            f.write(f"num_images: {len(dataset)}\n")
            f.write(f"num_labels: {len(classnames)}\n")
        print(f"[Save] {out_txt}")

        # Cleanup
        if hook_handles is not None:
            for h in hook_handles:
                h.remove()
        if neuron_hooks is not None:
            for h in neuron_hooks:
                h.remove()

        del text_features
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final summary print + CSV save
    print("\n" + "#" * 70)
    print("[Summary] All model results")
    print("#" * 70)

    df = pd.DataFrame(results)
    for r in results:
        print(f"{r['alias']:>16}  acc={r['accuracy']:.6f}  ablate={r['ablate_head']}  model={r['model']}")

    # also print sorted summary (best-first)
    if len(df) > 0:
        print("\n[Summary] sorted by accuracy (desc):")
        df_sorted = df.sort_values("accuracy", ascending=False)
        for _, r in df_sorted.iterrows():
            print(f"{str(r['alias']):>16}  acc={float(r['accuracy']):.6f}")

        out_csv = os.path.join(local_path, "imagenet-objectnet_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[Save] {out_csv}")

        model_aliases = [a for a, _ in MODELS]
        color_map = make_model_color_map(model_aliases)

        res_by_alias = {r["alias"]: r for r in results}
        acc_list = [float(res_by_alias[a]["accuracy"]) if a in res_by_alias else float("nan") for a in model_aliases]

        save_plot_mvt_zs_accuracy(
            out_dir=local_path,
            title="ImageNet/ObjectNet MVT — Zero-shot Accuracy (all models)",
            model_aliases=model_aliases,
            accuracies_by_model=acc_list,
            color_map=color_map,
        )
        print(f"[Save] Plot -> {os.path.join(local_path, 'mvt_zs_accuracy.png')}")

    print(f"\nResults saved to {local_path}.")


if __name__ == "__main__":
    main()