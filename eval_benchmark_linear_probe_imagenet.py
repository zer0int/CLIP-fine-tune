"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

Linear Probe evaluation.

Requires ImageNet ILSVRC2012; download:
https://www.image-net.org/download.php -> ILSVRC2012

Labels 'imagenet_wnid_to_class.json' included with this repo!
"""
import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

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
    ("regr-norm", "zer0int/CLIP-Regression-ViT-L-14"),
    ("regr-brut", "zer0int/CLIP-Regression-BRUT-ViT-L-14"),
]


VAL_DIR = "path/to/ILSVRC2012/val/" # edit this

JSON_FILE = "utils_datasets/imagenet/imagenet_wnid_to_class.json"
local_path = "out_eval_benchmarks/linear_probe_imagenet"
os.makedirs(local_path, exist_ok=True)
CSV_OUTPUT = f"{local_path}/imagenet_clip_linear_probe.csv"

BATCH_SIZE = 400
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-3
SEED = 42
NUM_WORKERS = 4

NUM_WORKERS_IO = NUM_WORKERS
NUM_WORKERS_PROBE = 0               # avoid spawn overhead for TensorDataset
PREPROCESS_FROM_FIRST_MODEL = True  # re-use for all models



class ImagePathDataset(Dataset):
    """Loads + preprocesses images in worker processes."""
    def __init__(self, image_files, image_labels, preprocess):
        self.image_files = image_files
        self.image_labels = image_labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        y = self.image_labels[idx]
        # ensure file handle closes promptly, avoid angering Windows :)
        with Image.open(path) as im:
            img = im.convert("RGB")
        x = self.preprocess(img)
        return x, y


def collate_pil(batch):
    imgs, ys = zip(*batch)
    return list(imgs), torch.tensor(ys, dtype=torch.long)


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_clip_model(mp: str, device: str):
    model, preprocess, _ = load_openai_clip_anything(
        clip, mp, device=device, jit=False, strict=True
    )
    model = model.eval().float()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess


def build_imagenet_filelist():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        wnid_to_label = json.load(f)

    assert len(wnid_to_label) == 1000, "Error: Filtered JSON does not contain exactly 1000 classes!"

    unique_labels = list(wnid_to_label.values())
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    image_files, image_labels = [], []
    for wnid, label in wnid_to_label.items():
        folder_path = os.path.join(VAL_DIR, wnid)
        for path in sorted(glob.glob(os.path.join(folder_path, "*"))):
            image_files.append(path)
            image_labels.append(label_to_index[label])

    assert len(image_files) == 50000, f"Expected 50,000 images, found {len(image_files)}"
    image_labels = torch.tensor(image_labels, dtype=torch.long)
    return image_files, image_labels


@torch.inference_mode()
def extract_image_features(model, device, loader):
    features = []
    for batch_images, _ in tqdm(loader, desc="Extracting CLIP Features"):
        batch_images = batch_images.to(device, non_blocking=True)
        image_features = model.encode_image(batch_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features.append(image_features.cpu())
    return torch.cat(features, dim=0)

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(image_embeddings, image_labels, device, batch_size, epochs, num_workers):
    g = torch.Generator()
    g.manual_seed(SEED)

    num_train = int(0.8 * len(image_embeddings))
    num_val = len(image_embeddings) - num_train
    train_dataset, val_dataset = random_split(
        TensorDataset(image_embeddings, image_labels),
        [num_train, num_val],
        generator=g,
    )

    # avoid passing prefetch_factor=None; build kwargs conditionally
    train_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
        generator=g,
    )
    if num_workers > 0:
        train_kwargs.update(
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
        )
    else:
        train_kwargs.update(num_workers=0)

    val_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    if num_workers > 0:
        val_kwargs.update(
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
        )
    else:
        val_kwargs.update(num_workers=0)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    torch.manual_seed(SEED)

    input_dim = int(image_embeddings.shape[-1])
    linear_probe = LinearProbe(input_dim=input_dim, num_classes=1000).to(device)
    optimizer = optim.AdamW(linear_probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    print("Training Linear Probe...")
    for epoch in range(epochs):
        linear_probe.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = linear_probe(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
            train_loss += float(loss.item())

        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss = {train_loss / max(1, len(train_loader)):.4f}, Train Accuracy = {train_acc:.2f}%")

        linear_probe.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                outputs = linear_probe(batch_x)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = 100.0 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in linear_probe.state_dict().items()}
        else:
            print("Early stopping triggered, no improvement.")

    if best_state is not None:
        linear_probe.load_state_dict(best_state)

    return linear_probe, val_loader, best_val_acc


def evaluate_probe(linear_probe, val_loader, device):
    linear_probe.eval()

    top1_correct, top5_correct, total = 0, 0, 0
    with torch.inference_mode():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = linear_probe(batch_x)
            top1_pred = outputs.argmax(dim=-1)
            top5_pred = outputs.topk(5, dim=-1).indices

            top1_correct += top1_pred.eq(batch_y).sum().item()
            top5_correct += top5_pred.eq(batch_y.unsqueeze(1)).any(dim=1).sum().item()
            total += batch_y.size(0)

    top1_acc = top1_correct / total * 100.0
    top5_acc = top5_correct / total * 100.0
    return top1_acc, top5_acc


def _print_ranked_overview(results):
    def _fmt_row(i, r, key):
        return f"{i:>2}. {r['Alias']:<14} | {r['Model']:<48} | {r[key]:6.2f}%"

    print("\n" + "=" * 80)
    print("RANKED OVERVIEW (Top-1)")
    print("=" * 80)
    top1_sorted = sorted(results, key=lambda x: x["Top-1 Accuracy"], reverse=True)
    for i, r in enumerate(top1_sorted, start=1):
        print(_fmt_row(i, r, "Top-1 Accuracy"))

    print("\n" + "=" * 80)
    print("RANKED OVERVIEW (Top-5)")
    print("=" * 80)
    top5_sorted = sorted(results, key=lambda x: x["Top-5 Accuracy"], reverse=True)
    for i, r in enumerate(top5_sorted, start=1):
        print(_fmt_row(i, r, "Top-5 Accuracy"))


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


def save_plot_lp_top1_top5(
    out_dir: str,
    title: str,
    model_aliases: List[str],
    top1_by_model: List[float],   # percent (0..100)
    top5_by_model: List[float],   # percent (0..100)
    color_map: Optional[Dict[str, str]] = None,
):
    xs = list(range(len(model_aliases)))
    width = 0.35

    # Wider if many models (prevents crowded x labels)
    fig_w = max(10.0, 0.65 * len(model_aliases))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))

    colors = None
    if color_map is not None:
        colors = [color_map.get(a, None) for a in model_aliases]

    # Use per-model color; distinguish Top-1 vs Top-5 by alpha + hatch
    bars_top1 = ax.bar(
        [x - width / 2 for x in xs],
        top1_by_model,
        width=width,
        label="Top-1",
        color=colors,
        alpha=0.90,
    )
    bars_top5 = ax.bar(
        [x + width / 2 for x in xs],
        top5_by_model,
        width=width,
        label="Top-5",
        color=colors,
        alpha=0.35,
        hatch="///",
        linewidth=0.8,
    )

    # headroom above 100 but last tick at 100
    y_top = 105.0
    ax.set_ylim(0.0, y_top)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Annotation offset scales with axis range
    y_off = 0.015 * y_top

    def _annotate(bars):
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

    _annotate(bars_top1)
    _annotate(bars_top5)

    ax.set_xticks(xs)
    ax.set_xticklabels(model_aliases, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")

    # Put legend above so it never overlaps labels
    ax.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )

    fig.subplots_adjust(top=0.82, bottom=0.28)
    fig.tight_layout()

    path = os.path.join(out_dir, "lp_top1_top5.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    torch.backends.cudnn.benchmark = True

    batch_size = BATCH_SIZE
    epochs = EPOCHS
    num_workers = NUM_WORKERS

    print("\n==================================")
    print("Linear Probe (LP): ImageNet")
    print("==================================\n")

    print(f"Will evaluate {len(MODELS)} model(s).")
    image_files, image_labels = build_imagenet_filelist()

    results = []


    # build ONE persistent feature_loader reused across models
    if PREPROCESS_FROM_FIRST_MODEL:
        _alias0, mp0 = MODELS[0]
        _m0, preprocess_shared = load_clip_model(mp0, device=device)
        del _m0
        if device == "cuda":
            torch.cuda.empty_cache()

        dataset = ImagePathDataset(
            image_files=image_files,
            image_labels=image_labels,
            preprocess=preprocess_shared,
        )
        feature_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=seed_worker if num_workers > 0 else None,
        )

    for (alias, mp) in MODELS:
        print("\n" + "=" * 80)
        print(f"MODEL: {alias}  |  {mp}")
        print("=" * 80)

        model, preprocess = load_clip_model(mp, device=device)

        # reuse the persistent loader if enabled
        if not PREPROCESS_FROM_FIRST_MODEL:
            dataset = ImagePathDataset(image_files=image_files, image_labels=image_labels, preprocess=preprocess)
            feature_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
                persistent_workers=(num_workers > 0),
                prefetch_factor=4 if num_workers > 0 else None,
                worker_init_fn=seed_worker if num_workers > 0 else None,
            )

        print("Extracting CLIP image embeddings (in-RAM only; no disk cache)...")
        image_embeddings = extract_image_features(model=model, device=device, loader=feature_loader)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        # avoid spawning workers for TensorDataset
        linear_probe, val_loader, best_val_acc = train_linear_probe(
            image_embeddings=image_embeddings,
            image_labels=image_labels,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=0,
        )

        top1_acc, top5_acc = evaluate_probe(linear_probe=linear_probe, val_loader=val_loader, device=device)
        print(f"\nLinear Probe Results: Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")

        results.append(
            {
                "Alias": alias,
                "Model": mp,
                "Best Val Acc (Top-1, during training)": best_val_acc,
                "Top-1 Accuracy": top1_acc,
                "Top-5 Accuracy": top5_acc,
            }
        )

        del linear_probe, image_embeddings
        if device == "cuda":
            torch.cuda.empty_cache()

    _print_ranked_overview(results)

    df = pd.DataFrame(results)
    df.to_csv(CSV_OUTPUT, index=False)

    with open(f"{local_path}/linear-probe.txt", "a", encoding="utf-8") as f:
        for r in results:
            f.write(
                f"ALIAS={r['Alias']} | MODEL={r['Model']} | Top-1={r['Top-1 Accuracy']:.2f}% | Top-5={r['Top-5 Accuracy']:.2f}%\n"
            )

    model_aliases = [a for a, _ in MODELS]
    color_map = make_model_color_map(model_aliases)

    # keep plot order identical to MODELS
    top1_list = []
    top5_list = []
    res_by_alias = {r["Alias"]: r for r in results}
    for a in model_aliases:
        r = res_by_alias.get(a, None)
        top1_list.append(float(r["Top-1 Accuracy"]) if r is not None else float("nan"))
        top5_list.append(float(r["Top-5 Accuracy"]) if r is not None else float("nan"))

    save_plot_lp_top1_top5(
        out_dir=local_path,
        title="ImageNet Linear Probe — Top-1 / Top-5 (all models)",
        model_aliases=model_aliases,
        top1_by_model=top1_list,
        top5_by_model=top5_list,
        color_map=color_map,
    )
    print(f"[Save] Plot -> {os.path.join(local_path, 'lp_top1_top5.png')}")

    print(f"\nWrote results to: {CSV_OUTPUT}")
    print(f"Appended log to: {local_path}/linear-probe.txt")


if __name__ == "__main__":
    main()