"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

So, your Text Encoder embeddings just collapsed, but you think
the ViT may be fine...?

Try it here, in the roundhouse-swap Encoder-Encoder arena!
-> You'll be prompted which combo to keep
-> Merge best config ('transplant' Encoder), then:
-> Retrain only top layers of transplant, e.g. Text: 10+11, ViT: 20-23

Labels: Full: with dataset, reduced: included here.
Dataset download (free, no sign-up):
https://objectnet.dev/mvt/

"""

import os
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    #("my_wreck", "path/to/model.pt"),
]

OUT_DIR_DEFAULT = "out_model_encoders_merged"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark CLIP checkpoints + swap/merge vision/text encoders across a model list")

    parser.add_argument("--data_dir", default=r"path/to/dataset-difficulty-CLIP/data_release_2023/all", help="Root folder containing the images referenced by the CSV.")
    parser.add_argument("--data_csv", default=r"utils_datasets/mvt/human_responses_5k.csv", help="CSV with columns including: image,label")
    parser.add_argument("--typo_root", default=".", help="Root containing image_sets/. If omitted, uses --data_dir.")

    parser.add_argument("--keep_vit", action="store_true", help="Pin the vision encoder to MODELS[0] and only swap text encoders.")
    parser.add_argument("--keep_text", action="store_true", help="Pin the text encoder to MODELS[0] and only swap vision encoders.")

    parser.add_argument("--use_fp16", action="store_true", help="If set, convert applicable weights to fp16 before saving merged models (OpenAI-style storage convenience).")

    return parser.parse_args()


def fix_random_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_df, image_folder: str, transform=None):
        self.data = csv_df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]["image"]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = int(self.data.iloc[idx]["label_idx"])
        return image, label_idx


class FolderImageDataset(Dataset):
    def __init__(self, folder: str, preprocess):
        self.folder = folder
        allowed_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(allowed_ext)
        ]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.preprocess(img), os.path.basename(p)


def encode_all_images(model, dataloader: DataLoader, device: str) -> torch.Tensor:
    model.eval()
    image_embs = []
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Encoding images", leave=False):
            imgs = imgs.to(device)
            emb = model.encode_image(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            image_embs.append(emb.detach().cpu())
    return torch.cat(image_embs, dim=0)


def compute_linear_probe(image_embs: torch.Tensor, label_indices: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    X = image_embs.cpu().numpy()
    clf.fit(X, label_indices)
    preds = clf.predict(X)
    return float(accuracy_score(label_indices, preds))


def _stat_str(arr: List[float]) -> str:
    if len(arr) == 0:
        return "n/a"
    return f"min={min(arr):.4f}, max={max(arr):.4f}, mean={float(np.mean(arr)):.4f}"

def _zs_margin_stats_from_sims(sims: torch.Tensor) -> Tuple[float, float, float]:
    """
    sims: [N, C] similarity matrix (cosine sims here).
    Returns (mean, median, p05) of (top1 - top2) over N samples.
    """
    # topk is efficient and avoids a full sort
    top2 = torch.topk(sims, k=2, dim=1).values  # [N,2], descending
    margins = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy().astype(np.float64)
    return (
        float(np.mean(margins)),
        float(np.median(margins)),
        float(np.percentile(margins, 5.0)),
    )


@dataclass
class TypoFolderStats:
    folder: str
    total: int
    correct: int
    misclassified: int
    correct_margin: str
    misclassified_margin: str
    mean_p_text: float
    predicted_counts: Dict[str, int]


def _resolve_typo_folders(typo_root: str, dataset_wnid: str) -> Dict[str, str]:
    base = os.path.join(typo_root, "image_sets") if typo_root is not None else "image_sets"
    return {
        "normal": os.path.join(base, dataset_wnid),
        "adv": os.path.join(base, f"{dataset_wnid}_adv"),
    }


def _typo_stats_from_image_embs(
    img_embs: torch.Tensor,  # [N,D] normalized, CPU ok
    text_model,
    device: str,
    choices: List[str],
    gt_label: int = 0,
) -> Tuple[int, int, List[float], List[float], List[float], Dict[str, int]]:
    text_model.eval()
    with torch.no_grad():
        text_tokens = clip.tokenize(choices).to(device)
        text_embs = text_model.encode_text(text_tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # logit_scale is taken from the text side (matches your merge_vision_and_text behavior)
        scale = text_model.logit_scale.exp()

        logits = (img_embs.to(device) @ text_embs.T) * scale
        probs = logits.softmax(dim=-1)

        preds = torch.argmax(probs, dim=1).detach().cpu().numpy()

        margin = torch.abs(probs[:, 0] - probs[:, 1]).detach().cpu().numpy()
        p_text = probs[:, 2].detach().cpu().numpy()

    correct = 0
    incorrect = 0
    margins_correct: List[float] = []
    margins_incorrect: List[float] = []
    p_text_vals: List[float] = []
    pred_counts = {"bird": 0, "bee": 0, "text": 0}

    for i in range(len(preds)):
        pred_i = int(preds[i])
        if pred_i == 0:
            pred_counts["bird"] += 1
        elif pred_i == 1:
            pred_counts["bee"] += 1
        else:
            pred_counts["text"] += 1

        p_text_vals.append(float(p_text[i]))
        if pred_i == gt_label:
            correct += 1
            margins_correct.append(float(margin[i]))
        else:
            incorrect += 1
            margins_incorrect.append(float(margin[i]))

    return correct, incorrect, margins_correct, margins_incorrect, p_text_vals, pred_counts


def _copy_param(dst: torch.nn.Parameter, src: torch.nn.Parameter, name: str) -> None:
    if dst.shape != src.shape:
        raise ValueError(f"Shape mismatch for {name}: dst={tuple(dst.shape)} src={tuple(src.shape)}")
    with torch.no_grad():
        dst.copy_(src)


def _infer_embed_dim_from_visual_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    """
    Infer CLIP embed_dim (the shared image/text embedding dim) from the VISUAL side.
    Supports ViT ('visual.proj') and ResNet ('visual.attnpool.c_proj').
    """
    if "visual.proj" in sd:
        # ViT: proj is [vision_width, embed_dim]
        return int(sd["visual.proj"].shape[1])
    if "visual.attnpool.c_proj.weight" in sd:
        # ResNet: c_proj is Linear(embed_dim_in, embed_dim_out); weight is [out, in]
        return int(sd["visual.attnpool.c_proj.weight"].shape[0])
    raise KeyError("Could not infer embed_dim from visual state_dict (missing visual.proj or visual.attnpool.c_proj.weight).")


def _build_hybrid_state_dict(vision_model: nn.Module, text_model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Build a hybrid CLIP state_dict:
      - take visual.* from vision_model
      - take everything else from text_model (token embedding, transformer, ln_final, text_projection, logit_scale, etc.)
    """
    sd_v = vision_model.state_dict()
    sd_t = text_model.state_dict()

    # Hard safety: embed_dim must match (prevents ViT-B/16 onto ViT-L/14, etc.)
    embed_dim_v = _infer_embed_dim_from_visual_state_dict(sd_v)
    embed_dim_t = int(sd_t["text_projection"].shape[1])  # text_projection is [transformer_width, embed_dim]
    if embed_dim_v != embed_dim_t:
        raise ValueError(
            f"Cannot transplant: embed_dim mismatch (vision embed_dim={embed_dim_v} vs text embed_dim={embed_dim_t})."
        )

    hybrid: Dict[str, torch.Tensor] = {}

    # Visual donor: ONLY visual.*
    for k, v in sd_v.items():
        if k.startswith("visual."):
            hybrid[k] = v

    # Text donor: EVERYTHING except visual.*
    for k, v in sd_t.items():
        if not k.startswith("visual."):
            hybrid[k] = v

    # Keep optional metadata keys if they exist (build_model will ignore/remove them if present)
    for meta_k in ["input_resolution", "context_length", "vocab_size"]:
        if meta_k in sd_t and meta_k not in hybrid:
            hybrid[meta_k] = sd_t[meta_k]
        elif meta_k in sd_v and meta_k not in hybrid:
            hybrid[meta_k] = sd_v[meta_k]

    return hybrid


def merge_vision_and_text(vision_model, text_model, use_fp16: bool = False):
    """
    Flexible encoder transplant:
    - builds a hybrid state_dict (visual from vision_model, text stack from text_model)
    - instantiates a new CLIP via oaiclip -> build_model(hybrid_state_dict)

    This supports:
      - ViT-L/14@336 vision + ViT-L/14 text (same text stack)
      - 77-token text -> Long-CLIP 248-token text (context length changes)
    while refusing incompatible embed_dim merges (e.g., ViT-B/16 onto ViT-L/14).
    """
    hybrid_sd = _build_hybrid_state_dict(vision_model, text_model)

    # IMPORTANT: build_model internally does OpenAI-style fp16 conversion.
    merged_model = clip.model.build_model(hybrid_sd).eval()  # <-- MODIFIED: build instead of deepcopy+copy

    # You control final storage dtype here
    if not use_fp16:
        merged_model = merged_model.float()  # <-- MODIFIED
    return merged_model


def tokenize_for_model(model, texts: List[str], device: str):
    """
    Tokenize respecting the model's context_length if possible.
    Falls back to manual pad/trim if clip.tokenize doesn't accept context_length.
    """
    ctx = int(getattr(model, "context_length", 77))
    try:
        tok = clip.tokenize(texts, context_length=ctx)
    except TypeError:
        tok = clip.tokenize(texts)
        # Manual pad/trim to ctx
        if tok.shape[1] < ctx:
            pad = torch.zeros((tok.shape[0], ctx - tok.shape[1]), dtype=tok.dtype)
            tok = torch.cat([tok, pad], dim=1)
        elif tok.shape[1] > ctx:
            tok = tok[:, :ctx]
    return tok.to(device)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16 (storage convenience)."""

    def _convert_weights_to_fp16(l):
        # classic OpenAI-style modules
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if hasattr(l, "weight") and l.weight is not None:
                l.weight.data = l.weight.data.half()
            if hasattr(l, "bias") and l.bias is not None:
                l.bias.data = l.bias.data.half()

        # torch.nn.MultiheadAttention (old style)
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # "MultiheadAttention" (some CLIP forks) with q_proj/k_proj/v_proj/out_proj modules
        if l.__class__.__name__ == "MultiheadAttention":
            for attr in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                module = getattr(l, attr, None)
                if module is not None and hasattr(module, "weight") and module.weight is not None:
                    module.weight.data = module.weight.data.half()
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data = module.bias.data.half()
            for attr in ["bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # CLIP projections
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None and hasattr(attr, "data"):
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


@dataclass
class BenchResult:
    vit_alias: str
    text_alias: str
    zero_shot_acc: float
    linear_probe_acc: float
    typo_normal_acc: float
    typo_adv_acc: float
    typo_normal_mean_p_text: float
    typo_adv_mean_p_text: float
    logit_scale: float
    overall: float
    gain_vs_vit_base: float = float("nan")
    gain_vs_text_base: float = float("nan")
    gain_vs_best_parent: float = float("nan")
    zs_margin_mean: float = float("nan")
    zs_margin_median: float = float("nan")
    zs_margin_p05: float = float("nan")

def _overall_score(
    zs: float,
    lp: float,
    typo_n: float,
    typo_a: float,
) -> float:
    # simple, monotone, and transparent: average of 4 accuracies
    return float((zs + lp + typo_n + typo_a) / 4.0)


def _print_results_table(results: List[BenchResult]) -> None:
    hdr = (
        f"{'kind':6s} {'vit':16s} {'text':16s} | "
        f"{'overall':>7s} {'ZS':>7s} {'LP':>7s} {'Clean':>7s} {'TypoX':>7s} | "
        f"{'g_vit':>7s} {'g_txt':>7s} {'g_best':>7s} | "
        f"{'m_mean':>7s} {'m_med':>7s} {'m_p05':>7s} | "
        f"{'p_text_n':>9s} {'p_text_a':>9s} | {'logit':>8s}"
    )
    print("\n" + "#" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        kind = "BASE" if (r.vit_alias == r.text_alias) else "MERGE"
        print(
            f"{kind:6s} {r.vit_alias:16s} {r.text_alias:16s} | "
            f"{r.overall:7.4f} {r.zero_shot_acc:7.3f} {r.linear_probe_acc:7.3f} {r.typo_normal_acc:7.2f} {r.typo_adv_acc:7.2f} | "
            f"{r.gain_vs_vit_base:7.2f} {r.gain_vs_text_base:7.2f} {r.gain_vs_best_parent:7.2f} | "
            f"{r.zs_margin_mean:7.4f} {r.zs_margin_median:7.4f} {r.zs_margin_p05:7.4f} | "
            f"{r.typo_normal_mean_p_text:9.2f} {r.typo_adv_mean_p_text:9.2f} | {r.logit_scale:8.2f}"
        )
    print("#" * len(hdr) + "\n")


@dataclass
class VisionCache:
    alias: str
    source: str
    preprocess: any
    image_embs_main: torch.Tensor  # [N,D] CPU
    linear_probe_acc: float
    typo_folder_embs: Dict[str, torch.Tensor]  # tag -> [N,D] CPU
    typo_folder_paths: Dict[str, str]  # tag -> path


@dataclass
class TextCache:
    alias: str
    source: str
    text_embs_classes: torch.Tensor  # [C,D] CUDA/CPU ok (we keep CPU and move as needed)
    text_embs_choices: torch.Tensor  # [3,D]
    logit_scale_exp: float


def _load_model_any(source: str, device: str):
    # normalize all loading through the same utility (so hub/local/OpenAI are treated consistently)
    model, preprocess, _meta = load_openai_clip_anything(clip, source, device=device, jit=False, strict=True)
    model = model.float().eval()
    return model, preprocess


def _encode_folder_images(vision_model, preprocess, folder: str, device: str, batch_size: int = 64) -> torch.Tensor:
    ds = FolderImageDataset(folder, preprocess=preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    embs = []
    with torch.no_grad():
        for imgs, _names in tqdm(dl, desc=f"Encoding folder: {os.path.basename(folder)}", leave=False):
            imgs = imgs.to(device)
            emb = vision_model.encode_image(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.detach().cpu())
    if len(embs) == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(embs, dim=0)


def main():
    fix_random_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_arguments()
    csv_file = args.data_csv
    image_folder = args.data_dir
    typo_root = args.typo_root if args.typo_root is not None else args.data_dir

    # Load CSV + dataset once
    import pandas as pd
    df = pd.read_csv(csv_file)
    classnames = sorted(list(set(df["label"])))
    class2idx = {c: i for i, c in enumerate(classnames)}
    df["label_idx"] = df["label"].map(class2idx)
    label_indices = df["label_idx"].values.astype(np.int64)

    tfm = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    dataset = CroppedImageCSVFileDataset(df, image_folder=image_folder, transform=tfm)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0, pin_memory=True)

    # Precompute caches
    choices = ["a photo of a bird", "a photo of a bumblebee", "a photo of a text"]
    typo_folders = _resolve_typo_folders(typo_root=typo_root, dataset_wnid="n01531178")

    vision_caches: Dict[str, VisionCache] = {}
    text_caches: Dict[str, TextCache] = {}

    print("\n" + "=" * 96)
    print("Building caches (vision embeddings, text embeddings) ...")
    print("=" * 96)

    # Vision caches
    for alias, source in MODELS:
        print(f"[Vision cache] {alias} <- {source}")
        model_v, preprocess_v = _load_model_any(source, device=device)
        model_v = model_v.to(device)

        image_embs_main = encode_all_images(model_v, dataloader, device=device)
        lin_probe_acc = compute_linear_probe(image_embs_main, label_indices)

        typo_folder_embs: Dict[str, torch.Tensor] = {}
        typo_folder_paths: Dict[str, str] = {}

        for tag, folder in typo_folders.items():
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Typographic folder not found: {folder}")
            typo_folder_paths[tag] = folder
            typo_folder_embs[tag] = _encode_folder_images(model_v, preprocess_v, folder, device=device, batch_size=64)

        # move model off GPU
        del model_v
        torch.cuda.empty_cache()

        vision_caches[alias] = VisionCache(
            alias=alias,
            source=source,
            preprocess=preprocess_v,
            image_embs_main=image_embs_main,
            linear_probe_acc=float(lin_probe_acc),
            typo_folder_embs=typo_folder_embs,
            typo_folder_paths=typo_folder_paths,
        )

    # Text caches
    for alias, source in MODELS:
        print(f"[Text cache]   {alias} <- {source}")
        model_t, _preprocess_t = _load_model_any(source, device=device)
        model_t = model_t.to(device)

        with torch.no_grad():
            # MODIFIED: model-aware context length
            tok_classes = tokenize_for_model(model_t, classnames, device=device)
            text_embs_classes = model_t.encode_text(tok_classes)
            text_embs_classes = text_embs_classes / text_embs_classes.norm(dim=-1, keepdim=True)

            tok_choices = tokenize_for_model(model_t, choices, device=device)
            text_embs_choices = model_t.encode_text(tok_choices)
            text_embs_choices = text_embs_choices / text_embs_choices.norm(dim=-1, keepdim=True)

            logit_scale_exp = float(model_t.logit_scale.exp().detach().cpu().item())

        # store on CPU to keep VRAM clean
        text_caches[alias] = TextCache(
            alias=alias,
            source=source,
            text_embs_classes=text_embs_classes.detach().cpu(),
            text_embs_choices=text_embs_choices.detach().cpu(),
            logit_scale_exp=logit_scale_exp,
        )

        del model_t
        torch.cuda.empty_cache()


    # Decide which swaps to run
    aliases = [a for a, _s in MODELS]
    vit_aliases = aliases
    text_aliases = aliases

    if args.keep_vit:
        vit_aliases = [aliases[0]]
    if args.keep_text:
        text_aliases = [aliases[0]]

    # always include baselines for reference (vit==text)
    baseline_aliases = sorted(set(vit_aliases) | set(text_aliases))
    baseline_pairs: List[Tuple[str, str]] = [(a, a) for a in baseline_aliases]

    # merged candidates exclude native pairs
    swap_pairs: List[Tuple[str, str]] = [
        (va, ta)
        for va in vit_aliases
        for ta in text_aliases
        if va != ta
    ]

    # evaluate baselines + swaps together
    all_pairs: List[Tuple[str, str]] = baseline_pairs + swap_pairs

    # Evaluate all swaps quickly from caches
    results: List[BenchResult] = []

    print("\n" + "=" * 96)
    print(f"Evaluating pairs: {len(all_pairs)} (baselines={len(baseline_pairs)} + swaps={len(swap_pairs)})")
    print("=" * 96)

    for vit_alias, text_alias in all_pairs:
        vc = vision_caches[vit_alias]
        tc = text_caches[text_alias]

        # zero-shot (argmax invariant to scale)
        # MODIFIED: skip incompatible embedding dims (e.g. ViT-B/16 vs ViT-L/14)
        if vc.image_embs_main.shape[1] != tc.text_embs_classes.shape[1]:
            print(
                f"[Skip] vit={vit_alias} text={text_alias} :: embed_dim mismatch "
                f"(img={vc.image_embs_main.shape[1]} vs txt={tc.text_embs_classes.shape[1]})"
                f"Sorry, can't build this chimera - use models of the *same architecture*!"
            )
            continue


        sims = vc.image_embs_main @ tc.text_embs_classes.T  # CPU
        preds = torch.argmax(sims, dim=1).cpu().numpy()
        zs_acc = float(accuracy_score(label_indices, preds))

        # ZS margin diagnostics (top1 - top2)
        zs_m_mean, zs_m_med, zs_m_p05 = _zs_margin_stats_from_sims(sims)

        # probs = softmax((img @ text.T) * scale)
        def _typo_one(tag: str) -> TypoFolderStats:
            img_embs = vc.typo_folder_embs[tag]  # CPU, normalized
            folder = vc.typo_folder_paths[tag]
            scale = tc.logit_scale_exp

            logits = (img_embs @ tc.text_embs_choices.T) * scale  # CPU
            probs = logits.softmax(dim=-1)

            preds_local = torch.argmax(probs, dim=1).cpu().numpy()

            margin = torch.abs(probs[:, 0] - probs[:, 1]).cpu().numpy()
            p_text = probs[:, 2].cpu().numpy()

            correct = 0
            incorrect = 0
            margins_correct: List[float] = []
            margins_incorrect: List[float] = []
            pred_counts = {"bird": 0, "bee": 0, "text": 0}

            gt_label = 0
            for i in range(len(preds_local)):
                pred_i = int(preds_local[i])
                if pred_i == 0:
                    pred_counts["bird"] += 1
                elif pred_i == 1:
                    pred_counts["bee"] += 1
                else:
                    pred_counts["text"] += 1

                if pred_i == gt_label:
                    correct += 1
                    margins_correct.append(float(margin[i]))
                else:
                    incorrect += 1
                    margins_incorrect.append(float(margin[i]))

            return TypoFolderStats(
                folder=folder,
                total=int(img_embs.shape[0]),
                correct=correct,
                misclassified=incorrect,
                correct_margin=_stat_str(margins_correct),
                misclassified_margin=_stat_str(margins_incorrect),
                mean_p_text=float(np.mean(p_text)) if len(p_text) else float("nan"),
                predicted_counts=pred_counts,
            )

        normal = _typo_one("normal")
        adv = _typo_one("adv")

        typo_normal_acc = float(normal.correct / max(1, normal.total))
        typo_adv_acc = float(adv.correct / max(1, adv.total))

        overall = _overall_score(zs_acc, vc.linear_probe_acc, typo_normal_acc, typo_adv_acc)

        results.append(
            BenchResult(
                vit_alias=vit_alias,
                text_alias=text_alias,
                zero_shot_acc=zs_acc,
                linear_probe_acc=float(vc.linear_probe_acc),
                typo_normal_acc=typo_normal_acc,
                typo_adv_acc=typo_adv_acc,
                typo_normal_mean_p_text=float(normal.mean_p_text),
                typo_adv_mean_p_text=float(adv.mean_p_text),
                logit_scale=float(tc.logit_scale_exp),
                overall=overall,
                zs_margin_mean=zs_m_mean,
                zs_margin_median=zs_m_med,
                zs_margin_p05=zs_m_p05,
            )
        )

    # compute synergy deltas using BASE rows
    base_overall: Dict[str, float] = {
        r.vit_alias: r.overall
        for r in results
        if (r.vit_alias == r.text_alias)
    }

    for r in results:
        vit_base = base_overall.get(r.vit_alias, float("nan"))
        text_base = base_overall.get(r.text_alias, float("nan"))

        r.gain_vs_vit_base = float(r.overall - vit_base) if np.isfinite(vit_base) else float("nan")
        r.gain_vs_text_base = float(r.overall - text_base) if np.isfinite(text_base) else float("nan")

        if np.isfinite(vit_base) and np.isfinite(text_base):
            r.gain_vs_best_parent = float(r.overall - max(vit_base, text_base))
        else:
            r.gain_vs_best_parent = float("nan")


    # Overview table
    results_sorted = sorted(results, key=lambda r: r.overall, reverse=True)
    _print_results_table(results_sorted)


    # Interactive save prompt
    os.makedirs(OUT_DIR_DEFAULT, exist_ok=True)

    def _save_merged(vit_alias: str, text_alias: str) -> str:
        vc = vision_caches[vit_alias]
        tc = text_caches[text_alias]

        # load models fresh for an actual merged pickle
        model_v, _pre_v = _load_model_any(vc.source, device="cpu")
        model_t, _pre_t = _load_model_any(tc.source, device="cpu")

        merged_model = merge_vision_and_text(model_v, model_t, use_fp16=args.use_fp16).eval()

        out_path = os.path.join(OUT_DIR_DEFAULT, f"model_{vit_alias}-vit_{text_alias}-text.pt")
        torch.save(merged_model, out_path)
        return out_path

    print("\n" + "=" * 96)
    print("Save merged models (sorted best-overall-first).")
    print("y = save this model and continue | n = skip | x = save all remaining (no more prompts) | a = abort/exit")
    print("=" * 96)

    save_all_mode = False
    for r in results_sorted:
        print(
            f"\nCandidate: vit={r.vit_alias}  text={r.text_alias} | "
            f"overall={r.overall:.4f}  ZS={r.zero_shot_acc:.4f}  LP={r.linear_probe_acc:.4f}  "
            f"Typo(n)={r.typo_normal_acc:.4f}  Typo(a)={r.typo_adv_acc:.4f}  "
            f"logit_scale.exp={r.logit_scale:.6f}"
        )
        # never prompt/save baselines (native vit==text); keep them in ranking as reference
        if r.vit_alias == r.text_alias:
            print("  -> Baseline (native model). Not saving/prompting.")
            continue

        if save_all_mode:
            out_path = _save_merged(r.vit_alias, r.text_alias)
            print(f"[Saved] merged_model -> {out_path}")
            continue

        while True:
            ans = input("Save? [y]es / [n]o / [x]save-all / [a]abort-and-exit -- [y/n/x/a]: ").strip().lower()
            if ans == "y":
                out_path = _save_merged(r.vit_alias, r.text_alias)
                print(f"[Saved] merged_model -> {out_path}")
                break
            elif ans == "n":
                break
            elif ans == "x":
                out_path = _save_merged(r.vit_alias, r.text_alias)
                print(f"[Saved] merged_model -> {out_path}")
                save_all_mode = True
                break
            elif ans == "a":
                print("[Abort] exiting now.")
                return
            else:
                print("Invalid input. Use y/n/x/a.")

    print("\n" + "=" * 96)
    print(f"Done. Output directory: {OUT_DIR_DEFAULT}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()