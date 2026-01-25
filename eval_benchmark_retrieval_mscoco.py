"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

COCO Retrieval (II / I2T / T2I / T2T) for CLIP
========================================================================

Download:
http://images.cocodataset.org/zips/val2014.zip
-> Labels 'coco_val_karpathy.json' included in this repo!

Supports two JSON formats:

(1) Karpathy split JSON:
    {"images":[{"split":"val","filename":"...","sentences":[{"raw":"..."}...]}...]}

(2) Official COCO captions annotations JSON:
    {"images":[{"id":..., "file_name":"COCO_val2014_....jpg"}...],
     "annotations":[{"image_id":..., "caption":"..."}...]}

Computes:
  I2T: image -> text
  T2I: text -> image
  T2T: text -> text (captions of same image)
  II : image -> image (optionally with augmentation for view-B)
"""

from __future__ import annotations

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import math 

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

OUT_DIR = "out_eval_benchmarks/retrieval_mscoco"


def parse_args():
    ap = argparse.ArgumentParser("COCO Retrieval Multi-Model (II / I2T / T2I / T2T + extras)")
    ap.add_argument("--coco_img_dir", default="path/to/COCO/val2014")
    ap.add_argument("--json_path", default="utils_datasets/coco/coco_val_karpathy.json", help="Karpathy JSON OR official COCO captions annotations JSON.")
    
    
    ap.add_argument("--split", default="val", help="Karpathy split to use (val/test/train/restval). Ignored for COCO captions annotations JSON.")
    ap.add_argument("--max_images", type=int, default=0, help="If >0, limit to first N images after filtering.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--text_batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cand_block", type=int, default=4096)
    ap.add_argument("--query_block", type=int, default=256)

    ap.add_argument("--skip_ii", action="store_true")
    ap.add_argument("--skip_t2i", action="store_true")
    ap.add_argument("--skip_t2t", action="store_true")
    ap.add_argument("--skip_i2t", action="store_true")

    ap.add_argument("--ii_augment", action="store_true", help="Use stochastic augmentation for II candidate view.")
    ap.add_argument("--run_tfidf", action="store_true", help="Also run TF-IDF T2T baseline (requires scikit-learn).")
    ap.add_argument("--tfidf_query_block", type=int, default=64, help="TF-IDF query block size (RAM tradeoff).")
    return ap.parse_args()



def fix_random_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def median_rank_from_ranks(ranks_1based: List[int]) -> float:
    s = sorted(ranks_1based)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return float(s[n // 2])
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def recall_at_k_from_ranks(ranks_1based: List[int], k: int) -> float:
    if not ranks_1based:
        return 0.0
    hit = sum(1 for r in ranks_1based if r <= k)
    return hit / float(len(ranks_1based))


@dataclass
class CocoEntry:
    img_path: str
    captions: List[str]


def _looks_like_coco_captions_json(d: dict) -> bool:
    return isinstance(d, dict) and ("images" in d) and ("annotations" in d)


def load_coco_entries_auto(
    json_path: str,
    coco_img_dir: str,
    split: Optional[str] = "val",
) -> List[CocoEntry]:
    """
    Auto-detects JSON format:
      - Karpathy split JSON
      - Official COCO captions annotations JSON

    If format is COCO captions annotations JSON, `split` is ignored.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if _looks_like_coco_captions_json(data):
        images = data["images"]
        ann = data["annotations"]

        id_to_file: Dict[int, str] = {}
        for im in images:
            if "id" in im and ("file_name" in im or "filename" in im):
                fn = im.get("file_name", im.get("filename"))
                id_to_file[int(im["id"])] = str(fn)

        id_to_caps: Dict[int, List[str]] = {k: [] for k in id_to_file.keys()}
        for a in ann:
            img_id = int(a.get("image_id"))
            cap = a.get("caption", None)
            if cap is None:
                continue
            if img_id in id_to_caps:
                id_to_caps[img_id].append(str(cap))

        out: List[CocoEntry] = []
        for img_id, fn in id_to_file.items():
            caps = id_to_caps.get(img_id, [])
            if len(caps) == 0:
                continue
            img_path = os.path.join(coco_img_dir, fn)
            if not os.path.isfile(img_path):
                continue
            out.append(CocoEntry(img_path=img_path, captions=caps))
        return out

    # Karpathy: {"images":[...]}
    images = data["images"] if isinstance(data, dict) and "images" in data else data

    out: List[CocoEntry] = []
    for it in images:
        if not isinstance(it, dict):
            continue

        if split is not None and "split" in it:
            if str(it.get("split", "")).lower() != str(split).lower():
                continue

        fn = it.get("filename", None) or it.get("file_name", None)
        if fn is None:
            continue

        img_path = os.path.join(coco_img_dir, str(fn))
        if not os.path.isfile(img_path):
            continue

        sents = it.get("sentences", it.get("captions", []))
        caps: List[str] = []
        for s in sents:
            if isinstance(s, dict):
                raw = s.get("raw", None) or s.get("caption", None) or s.get("text", None)
                if raw is not None:
                    caps.append(str(raw))
            elif isinstance(s, str):
                caps.append(s)

        if len(caps) == 0:
            continue

        out.append(CocoEntry(img_path=img_path, captions=caps))

    return out


class CocoImageDataset(Dataset):
    def __init__(self, entries: List[CocoEntry], transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        p = self.entries[idx].img_path
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        x = self.transform(img)
        return x, idx


def collate_images(batch):
    xs, idxs = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(idxs, dtype=torch.long)


def build_ii_aug(preprocess):
    import torchvision.transforms as T

    size = 224
    try:
        for tr in getattr(preprocess, "transforms", []):
            if hasattr(tr, "size"):
                s = tr.size
                if isinstance(s, (tuple, list)):
                    size = int(s[0])
                else:
                    size = int(s)
                break
    except Exception:
        pass

    normalize = None
    for tr in getattr(preprocess, "transforms", []):
        if tr.__class__.__name__.lower() == "normalize":
            normalize = tr
            break

    aug_list = [
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
    ]
    if normalize is not None:
        aug_list.append(normalize)

    return T.Compose(aug_list)


def build_caption_index(entries: List[CocoEntry]) -> Tuple[List[str], List[int], Dict[int, List[int]]]:
    captions: List[str] = []
    cap_to_img: List[int] = []
    img_to_caps: Dict[int, List[int]] = {}

    for img_idx, e in enumerate(entries):
        img_to_caps[img_idx] = []
        for c in e.captions:
            cap_idx = len(captions)
            captions.append(c)
            cap_to_img.append(img_idx)
            img_to_caps[img_idx].append(cap_idx)

    return captions, cap_to_img, img_to_caps


def build_tt_positives_excl_self(cap_to_img: List[int], img_to_caps: Dict[int, List[int]]) -> List[List[int]]:
    """
    For each caption i, positives are the OTHER captions of the same image (self excluded).
    """
    positives: List[List[int]] = []
    for cap_idx, img_idx in enumerate(cap_to_img):
        caps = img_to_caps[int(img_idx)]
        positives.append([j for j in caps if j != cap_idx])
    return positives


@torch.inference_mode()
def encode_images(model, dl: DataLoader, device: str, amp: bool, desc: str) -> torch.Tensor:
    feats = []
    for x, _idx in tqdm(dl, desc=desc):
        x = x.to(device, non_blocking=(device == "cuda"))
        if device == "cuda" and amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model.encode_image(x)
        else:
            z = model.encode_image(x)
        z = l2_normalize(z.float())
        feats.append(z.detach().cpu())
    return torch.cat(feats, dim=0)


@torch.inference_mode()
def encode_texts(model, captions: List[str], device: str, amp: bool, batch_size: int, desc: str) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(captions), batch_size), desc=desc):
        chunk = captions[i:i + batch_size]
        tok = clip.tokenize(chunk, truncate=True).to(device, non_blocking=(device == "cuda"))
        if device == "cuda" and amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model.encode_text(tok)
        else:
            z = model.encode_text(tok)
        z = l2_normalize(z.float())
        feats.append(z.detach().cpu())
    return torch.cat(feats, dim=0)


# Blockwise ranks (best-positive rank), with optional exclude-self
@torch.inference_mode()
def ranks_query_to_candidates_best_positive(
    Q: torch.Tensor,               # [Nq, D] CPU
    C: torch.Tensor,               # [Nc, D] CPU
    positives: List[List[int]],    # positives[q] -> list of candidate indices
    device: str,
    cand_block: int = 4096,
    query_block: int = 256,
    exclude_cand: Optional[List[int]] = None,  # exclude one candidate per query (e.g. self)
) -> List[int]:
    Qd = Q.to(device)
    Cd = C.to(device)

    ranks: List[int] = []

    for qb in tqdm(range(0, Q.shape[0], query_block), desc="ranks (2-pass)"):
        q_end = min(Q.shape[0], qb + query_block)
        q = Qd[qb:q_end]  # [Bq,D]
        bq = q.shape[0]

        # pass 1: best positive
        best_pos = torch.full((bq,), -1e9, device=device, dtype=torch.float32)
        for bi, qi in enumerate(range(qb, q_end)):
            pos = positives[qi]
            if not pos:
                continue
            pj = Cd[pos]  # [P,D]
            sims_pos = (q[bi:bi + 1] @ pj.T).squeeze(0)  # [P]
            best_pos[bi] = sims_pos.max()

        # pass 2: count how many candidates beat best_pos
        better = torch.zeros((bq,), device=device, dtype=torch.int64)
        for cb in range(0, Cd.shape[0], cand_block):
            c_end = min(Cd.shape[0], cb + cand_block)
            c = Cd[cb:c_end]
            sims = q @ c.T
            better += (sims > best_pos[:, None]).sum(dim=1).to(torch.int64)

        # exclude per-query candidate (e.g. the query itself for T2T)
        if exclude_cand is not None:
            ex = exclude_cand[qb:q_end]
            ex_t = torch.tensor(ex, device=device, dtype=torch.long)  # [Bq]
            # gather candidate vectors for excluded indices
            c_ex = Cd[ex_t]  # [Bq, D]
            sim_ex = (q * c_ex).sum(dim=1).to(torch.float32)  # [Bq]
            better = better - (sim_ex > best_pos).to(torch.int64)

        rb = (better + 1).detach().cpu().tolist()
        ranks.extend([int(r) for r in rb])

    return ranks


# TF-IDF baseline (caption -> caption)
def ranks_tfidf_t2t(
    captions: List[str],
    positives: List[List[int]],
    query_block: int = 64,
) -> List[int]:
    """
    Computes T2T ranks using TF-IDF cosine similarity.
    Requires scikit-learn.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        raise RuntimeError(
            "TF-IDF baseline requires scikit-learn. Install with: pip install scikit-learn"
        ) from e

    # sensible defaults for COCO captions; tweak if you want
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200_000,
        dtype=float,
    )
    X = vec.fit_transform(captions)  # CSR [M,V], L2-normalized by default
    M = X.shape[0]

    ranks: List[int] = []
    for qb in tqdm(range(0, M, query_block), desc="TF-IDF ranks"):
        q_end = min(M, qb + query_block)
        B = q_end - qb

        # dense block similarities (B x M)
        S = (X[qb:q_end] @ X.T).toarray()

        # exclude self
        for bi in range(B):
            S[bi, qb + bi] = -1e9

        # best positive + rank
        for bi, qi in enumerate(range(qb, q_end)):
            pos = positives[qi]
            if not pos:
                ranks.append(M)  # degenerate
                continue
            best_pos = max(S[bi, j] for j in pos)
            better = int((S[bi, :] > best_pos).sum())
            ranks.append(better + 1)

    return ranks


def compute_metrics_from_ranks(ranks: List[int]) -> Dict[str, float]:
    return {
        "R@1": recall_at_k_from_ranks(ranks, 1),
        "R@5": recall_at_k_from_ranks(ranks, 5),
        "R@10": recall_at_k_from_ranks(ranks, 10),
        "MedR": median_rank_from_ranks(ranks),
        "MeanR": (float(sum(ranks)) / float(len(ranks))) if ranks else float("nan"),
        "n": float(len(ranks)),
    }


def print_task_metrics(task_name: str, m: Dict[str, float]):
    print(f"\n[{task_name}] n={int(m['n'])}")
    print(f"  R@1  = {m['R@1']:.6f}")
    print(f"  R@5  = {m['R@5']:.6f}")
    print(f"  R@10 = {m['R@10']:.6f}")
    print(f"  MedR = {m['MedR']:.2f}")
    print(f"  MeanR= {m['MeanR']:.2f}")


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


def save_plot_task(
    out_dir: str,
    task_key: str,
    title: str,
    model_aliases: List[str],
    metrics_by_model: List[Dict[str, float]],
    color_map: Optional[Dict[str, str]] = None,
):
    xs = list(range(len(model_aliases)))
    width = 0.25

    r1 = [m.get("R@1", float("nan")) for m in metrics_by_model]
    r5 = [m.get("R@5", float("nan")) for m in metrics_by_model]
    r10 = [m.get("R@10", float("nan")) for m in metrics_by_model]

    fig_w = max(10.0, 0.65 * len(model_aliases))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    # per-model colors (same alias => same color)
    base_colors = None
    if color_map is not None:
        base_colors = [color_map.get(a, None) for a in model_aliases]

    # For grouped bars, shift brightness via alpha (keeps "same model color" identity)
    bars_r1 = ax.bar([x - width for x in xs], r1, width=width, label="R@1",
                     color=base_colors, alpha=0.55)
    bars_r5 = ax.bar(xs, r5, width=width, label="R@5",
                     color=base_colors, alpha=0.85)
    bars_r10 = ax.bar([x + width for x in xs], r10, width=width, label="R@10",
                      color=base_colors, alpha=0.70)

    # fixed headroom above 1.0 for annotations, but keep last tick at 1.0
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

    _annotate_bars(bars_r1)
    _annotate_bars(bars_r5)
    _annotate_bars(bars_r10)

    ax.set_xticks(xs)
    ax.set_xticklabels(model_aliases, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Recall")

    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)

    fig.subplots_adjust(top=0.82, bottom=0.28)
    fig.tight_layout()

    fn = task_key.lower().replace(" ", "_").replace("/", "_").replace("->", "to")
    path = os.path.join(out_dir, f"{fn}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)



TASK_ABBREV = {
    "I2T": "IT",
    "T2I": "TI",
    "T2T": "TT",
    "T2T_IMG": "TT_IMG",
    "II": "II",
    "II_aug": "II",
    "T2T_TFIDF": "TT_TFIDF",
}


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _metric_sort_key(m: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    """
    Higher is better for recalls; lower is better for ranks.
    Sort priority (per your request):
      1) max R@5   (PRIMARY)
      2) max R@1
      3) max R@10
      4) min MedR
      5) min MeanR
    """
    r5 = float(m.get("R@5", float("-inf")))
    r1 = float(m.get("R@1", float("-inf")))
    r10 = float(m.get("R@10", float("-inf")))
    medr = float(m.get("MedR", float("inf")))
    meanr = float(m.get("MeanR", float("inf")))
    return (-r5, -r1, -r10, medr, meanr)


def build_task_rankings(
    all_results: List[Dict[str, Any]],
    task_keys: List[str],
    pretty: Dict[str, str],
    tfidf_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
      rankings[task_key] = sorted list of dicts:
        {"rank": int, "alias": str, "model": str, "metrics": dict}
    """
    pool: List[Dict[str, Any]] = list(all_results)
    if tfidf_result is not None:
        pool = pool + [tfidf_result]

    rankings: Dict[str, List[Dict[str, Any]]] = {}

    for task_key in task_keys:
        rows: List[Dict[str, Any]] = []
        for res in pool:
            m = res.get("tasks", {}).get(task_key, None)
            if not isinstance(m, dict):
                continue
            # drop missing/degenerate metrics
            if not _is_finite(m.get("R@1", float("nan"))):
                continue
            if float(m.get("n", 0.0)) <= 0.0:
                continue

            rows.append({
                "alias": str(res.get("alias", "")),
                "model": str(res.get("model", "")),
                "task": task_key,
                "task_pretty": pretty.get(task_key, task_key),
                "abbr": TASK_ABBREV.get(task_key, task_key),
                "metrics": m,
            })

        rows.sort(key=lambda r: _metric_sort_key(r["metrics"]))
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
        rankings[task_key] = rows

    return rankings


def save_rankings_files(
    out_dir: str,
    rankings: Dict[str, List[Dict[str, Any]]],
):
    # distinct filenames for ranked-only outputs
    out_json = os.path.join(out_dir, "ranked_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)

    out_csv = os.path.join(out_dir, "ranked_results.csv")
    header = ["task", "abbr", "rank", "alias", "model", "R@1", "R@5", "R@10", "MedR", "MeanR", "n"]
    lines = [",".join(header)]
    for task_key, rows in rankings.items():
        for r in rows:
            m = r["metrics"]
            lines.append(",".join([
                str(task_key),
                str(r.get("abbr", "")),
                str(r.get("rank", "")),
                str(r.get("alias", "")),
                str(r.get("model", "")),
                f"{float(m.get('R@1', float('nan'))):.8f}",
                f"{float(m.get('R@5', float('nan'))):.8f}",
                f"{float(m.get('R@10', float('nan'))):.8f}",
                f"{float(m.get('MedR', float('nan'))):.4f}",
                f"{float(m.get('MeanR', float('nan'))):.4f}",
                f"{float(m.get('n', 0.0)):.0f}",
            ]))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_json, out_csv



# Eval per model
def eval_one_model(
    alias: str,
    model_id: str,
    entries: List[CocoEntry],
    captions: List[str],
    cap_to_img: List[int],
    img_to_caps: Dict[int, List[int]],
    positives_t2t: List[List[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = args.device
    print("\n" + "=" * 90)
    print(f"[Model] {alias}: {model_id}")
    print("=" * 90)

    model, preprocess, _ = load_openai_clip_anything(clip, model_id, device=device, jit=False, strict=True)
    model = model.float().eval()

    # dataloaders
    ds_img = CocoImageDataset(entries, transform=preprocess)
    dl_img = DataLoader(
        ds_img,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=collate_images,
    )

    # encode images A
    I = encode_images(model, dl_img, device=device, amp=bool(args.amp), desc=f"{alias}: encode images (A)")

    # encode images B for II
    I2 = None
    if not args.skip_ii:
        if args.ii_augment:
            aug = build_ii_aug(preprocess)
            ds_img_b = CocoImageDataset(entries, transform=aug)
            dl_img_b = DataLoader(
                ds_img_b,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=(device == "cuda"),
                persistent_workers=(int(args.num_workers) > 0),
                collate_fn=collate_images,
            )
            I2 = encode_images(model, dl_img_b, device=device, amp=bool(args.amp), desc=f"{alias}: encode images (B, aug)")
        else:
            I2 = I

    # encode texts
    T = encode_texts(
        model,
        captions,
        device=device,
        amp=bool(args.amp),
        batch_size=int(args.text_batch_size),
        desc=f"{alias}: encode captions",
    )

    results: Dict[str, Any] = {
        "alias": alias,
        "model": model_id,
        "tasks": {}
    }

    n_images = len(entries)
    n_caps = len(captions)

    # I2T
    if not args.skip_i2t:
        positives_i2t = [img_to_caps[i] for i in range(n_images)]
        ranks_i2t = ranks_query_to_candidates_best_positive(
            Q=I, C=T, positives=positives_i2t,
            device=device,
            cand_block=int(args.cand_block),
            query_block=int(args.query_block),
            exclude_cand=None,
        )
        m = compute_metrics_from_ranks(ranks_i2t)
        results["tasks"]["I2T"] = m
        print_task_metrics("I2T (image->text)", m)

    # T2I
    if not args.skip_t2i:
        positives_t2i = [[int(cap_to_img[i])] for i in range(n_caps)]
        ranks_t2i = ranks_query_to_candidates_best_positive(
            Q=T, C=I, positives=positives_t2i,
            device=device,
            cand_block=int(args.cand_block),
            query_block=int(args.query_block),
            exclude_cand=None,
        )
        m = compute_metrics_from_ranks(ranks_t2i)
        results["tasks"]["T2I"] = m
        print_task_metrics("T2I (text->image)", m)

    # T2T (CLIP) exclude self so R@1 isn't trivially ~0
    if not args.skip_t2t:
        exclude_self = list(range(n_caps))
        ranks_t2t = ranks_query_to_candidates_best_positive(
            Q=T, C=T, positives=positives_t2t,
            device=device,
            cand_block=int(args.cand_block),
            query_block=int(args.query_block),
            exclude_cand=exclude_self,
        )
        m = compute_metrics_from_ranks(ranks_t2t)
        results["tasks"]["T2T"] = m
        print_task_metrics("T2T (text->text, same-image captions; excl self)", m)

        # T2T (image-anchored): query=caption's image embedding, candidates=captions, positives=mates
        cap_to_img_t = torch.tensor(cap_to_img, dtype=torch.long)
        Q_imganch = I[cap_to_img_t]  # [n_caps, D] CPU
        ranks_t2t_img = ranks_query_to_candidates_best_positive(
            Q=Q_imganch, C=T, positives=positives_t2t,
            device=device,
            cand_block=int(args.cand_block),
            query_block=int(args.query_block),
            exclude_cand=exclude_self,  # exclude the query caption itself
        )
        m_img = compute_metrics_from_ranks(ranks_t2t_img)
        results["tasks"]["T2T_IMG"] = m_img
        print_task_metrics("T2T_IMG (image-anchored; excl self)", m_img)

    # II
    if not args.skip_ii:
        assert I2 is not None
        positives_ii = [[i] for i in range(n_images)]
        ranks_ii = ranks_query_to_candidates_best_positive(
            Q=I, C=I2, positives=positives_ii,
            device=device,
            cand_block=int(args.cand_block),
            query_block=int(args.query_block),
            exclude_cand=None,
        )
        m = compute_metrics_from_ranks(ranks_ii)
        results["tasks"]["II_aug" if args.ii_augment else "II"] = m
        tag = "II (image->image, aug view)" if args.ii_augment else "II (image->image, identical view)"
        print_task_metrics(tag, m)

    return results


def main():
    args = parse_args()
    fix_random_seed(int(args.seed))

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n===================================================")
    print("Retrieval: MSCOCO-Captions (COCO-val-2014)")
    print("===================================================\n")

    split = None if str(args.split).lower() in ("none", "null", "off", "") else str(args.split)

    print(f"[Data] json_path={args.json_path}")
    print(f"[Data] coco_img_dir={args.coco_img_dir}")
    entries = load_coco_entries_auto(args.json_path, args.coco_img_dir, split=split)

    if args.max_images and args.max_images > 0:
        entries = entries[: int(args.max_images)]

    if len(entries) == 0:
        raise RuntimeError(
            "No entries found after parsing/filtering.\n"
            "Most common causes:\n"
            "  - coco_img_dir doesn't contain the filenames in JSON (file_name mismatch)\n"
            "  - split filter removed everything (try --split none)\n"
        )

    captions, cap_to_img, img_to_caps = build_caption_index(entries)
    positives_t2t = build_tt_positives_excl_self(cap_to_img, img_to_caps)

    print(
        f"[Data] images={len(entries):,}  captions={len(captions):,}  "
        f"avg_caps/img={len(captions) / max(1, len(entries)):.2f}"
    )

    all_results: List[Dict[str, Any]] = []

    for mi, (alias, model_id) in enumerate(MODELS, start=1):
        print(f"\n[Run] Model {mi}/{len(MODELS)}: {alias}")
        res = eval_one_model(
            alias=alias,
            model_id=model_id,
            entries=entries,
            captions=captions,
            cap_to_img=cap_to_img,
            img_to_caps=img_to_caps,
            positives_t2t=positives_t2t,
            args=args,
        )
        all_results.append(res)

    # Optional TF-IDF baseline (one-time, not per model)
    tfidf_result: Optional[Dict[str, Any]] = None
    if args.run_tfidf and (not args.skip_t2t):
        print("\n" + "=" * 90)
        print("[TF-IDF] T2T baseline (caption->caption; excl self)")
        print("=" * 90)
        ranks = ranks_tfidf_t2t(
            captions=captions,
            positives=positives_t2t,
            query_block=int(args.tfidf_query_block),
        )
        m = compute_metrics_from_ranks(ranks)
        print_task_metrics("T2T_TFIDF (TF-IDF; excl self)", m)
        tfidf_result = {"alias": "tfidf", "model": "tfidf", "tasks": {"T2T_TFIDF": m}}

    # Print combined overview (in order of list)
    print("\n" + "#" * 90)
    print("[Overview] All models (in list order)")
    print("#" * 90)

    for res in all_results:
        alias = res["alias"]
        model_id = res["model"]
        print("\n" + "-" * 90)
        print(f"[{alias}] {model_id}")
        print("-" * 90)
        for task_name, m in res["tasks"].items():
            print(
                f"{task_name:9s}  "
                f"R@1={m['R@1']:.6f}  R@5={m['R@5']:.6f}  R@10={m['R@10']:.6f}  "
                f"MedR={m['MedR']:.2f}  MeanR={m['MeanR']:.2f}"
            )

    if tfidf_result is not None:
        m = tfidf_result["tasks"]["T2T_TFIDF"]
        print("\n" + "-" * 90)
        print("[tfidf] tfidf")
        print("-" * 90)
        print(
            f"T2T_TFIDF  "
            f"R@1={m['R@1']:.6f}  R@5={m['R@5']:.6f}  R@10={m['R@10']:.6f}  "
            f"MedR={m['MedR']:.2f}  MeanR={m['MeanR']:.2f}"
        )

    # compute task_union + pretty
    task_union: List[str] = []
    seen = set()
    for res in all_results:
        for t in res["tasks"].keys():
            if t not in seen:
                seen.add(t)
                task_union.append(t)

    pretty = {
        "I2T": "I2T (image->text)",
        "T2I": "T2I (text->image)",
        "T2T": "T2T (text->text; same-image captions; excl self)",
        "T2T_IMG": "T2T_IMG (image-anchored; excl self)",
        "II": "II (image->image)",
        "II_aug": "II (image->image, aug view)",
        "T2T_TFIDF": "T2T_TFIDF (TF-IDF; excl self)",
    }

    # rankings (sorted by R@5; CLI prints R@5 ONLY)
    ranking_tasks = task_union + (["T2T_TFIDF"] if tfidf_result is not None else [])

    rankings = build_task_rankings(
        all_results=all_results,
        task_keys=ranking_tasks,
        pretty=pretty,
        tfidf_result=tfidf_result,
    )

    print("\n" + "#" * 90)
    print("[Ranking] Top-10 models per task (sorted by R@5; CLI shows R@5 only)")
    print("#" * 90)

    for task_key in ranking_tasks:
        rows_rank = rankings.get(task_key, [])
        if not rows_rank:
            continue

        abbr = TASK_ABBREV.get(task_key, task_key)
        title = pretty.get(task_key, task_key)

        print("\n" + "-" * 90)
        print(f"[{abbr}] {title}  |  task_key={task_key}")
        print("-" * 90)

        topk = rows_rank[: min(10, len(rows_rank))]
        for r in topk:
            m = r["metrics"]
            print(
                f"  {int(r['rank']):2d}. {r['alias']:<12s}  "
                f"R@5={m['R@5']:.6f}  |  {r['model']}"
            )

    # ranked-only files (all metrics preserved in the file)
    rank_json, rank_csv = save_rankings_files(OUT_DIR, rankings)

    # Save results + plots
    payload = {
        "out_dir": OUT_DIR,
        "data": {
            "json_path": args.json_path,
            "coco_img_dir": args.coco_img_dir,
            "split": args.split,
            "max_images": int(args.max_images),
            "images": int(len(entries)),
            "captions": int(len(captions)),
        },
        "models": all_results,
    }
    if tfidf_result is not None:
        payload["baselines"] = [tfidf_result]

    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # CSV (flatten)
    out_csv = os.path.join(OUT_DIR, "results.csv")
    header = ["alias", "model", "task", "R@1", "R@5", "R@10", "MedR", "MeanR", "n"]
    rows = [",".join(header)]

    for res in all_results:
        for task_name, m in res["tasks"].items():
            rows.append(",".join([
                res["alias"],
                res["model"],
                task_name,
                f"{m['R@1']:.8f}",
                f"{m['R@5']:.8f}",
                f"{m['R@10']:.8f}",
                f"{m['MedR']:.4f}",
                f"{m['MeanR']:.4f}",
                f"{m['n']:.0f}",
            ]))

    if tfidf_result is not None:
        m = tfidf_result["tasks"]["T2T_TFIDF"]
        rows.append(",".join([
            "tfidf",
            "tfidf",
            "T2T_TFIDF",
            f"{m['R@1']:.8f}",
            f"{m['R@5']:.8f}",
            f"{m['R@10']:.8f}",
            f"{m['MedR']:.4f}",
            f"{m['MeanR']:.4f}",
            f"{m['n']:.0f}",
        ]))

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))

    # plots: one plot per task (all models)
    model_aliases = [r["alias"] for r in all_results]
    color_map = make_model_color_map(model_aliases)

    for task_key in task_union:
        metrics_by_model = []
        for res in all_results:
            m = res["tasks"].get(
                task_key,
                {
                    "R@1": float("nan"),
                    "R@5": float("nan"),
                    "R@10": float("nan"),
                    "MedR": float("nan"),
                    "MeanR": float("nan"),
                    "n": 0.0,
                },
            )
            metrics_by_model.append(m)

        save_plot_task(
            OUT_DIR,
            task_key=task_key,
            title=pretty.get(task_key, task_key),
            model_aliases=model_aliases,
            metrics_by_model=metrics_by_model,
            color_map=color_map,
        )

    # TF-IDF plot
    if tfidf_result is not None:
        save_plot_task(
            OUT_DIR,
            task_key="T2T_TFIDF",
            title=pretty.get("T2T_TFIDF", "T2T_TFIDF"),
            model_aliases=["tfidf"],
            metrics_by_model=[tfidf_result["tasks"]["T2T_TFIDF"]],
            color_map={"tfidf": "#7f7f7f"},  # neutral gray (optional)
        )

    print("\n" + "#" * 90)
    print(f"[Saved] JSON           -> {out_json}")
    print(f"[Saved] CSV            -> {out_csv}")
    print(f"[Saved] Ranked JSON     -> {rank_json}")
    print(f"[Saved] Ranked CSV      -> {rank_csv}")
    print(f"[Saved] Plots           -> {OUT_DIR}")
    print("#" * 90)


if __name__ == "__main__":
    main()