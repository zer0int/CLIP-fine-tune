from __future__ import annotations
import os
import torch
from typing import List, Tuple, Dict, Iterable, Set
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from colorama import Fore, Style
import inspect
from PIL import Image

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def print_grad_status_summary(
    model,
    *,
    text_block_prefix: str = "transformer.resblocks",
    vision_block_prefix: str = "visual.transformer.resblocks",
    title_text: str = "Text Transformer Blocks",
    title_vision: str = "Vision Transformer Blocks",
    max_suffix_show: int = 4,
) -> None:
    """
    Pretty-print HOT/COLD params by requires_grad, with special summaries for
    transformer block families.

    Intended usage: call this ONLY from unfreeze_layers(..., verbose=True),
    after you've set requires_grad flags.
    """

    hot = f"{Fore.RED}{Style.BRIGHT}"
    cold = f"{Fore.BLUE}{Style.BRIGHT}"
    block = f"{Fore.GREEN}{Style.BRIGHT}"
    reset = Style.RESET_ALL

    def _fmt_layers(xs: Set[int]) -> str:
        if not xs:
            return "[]"
        xs_sorted = sorted(xs)
        ranges: List[Tuple[int, int]] = []
        a = b = xs_sorted[0]
        for v in xs_sorted[1:]:
            if v == b + 1:
                b = v
            else:
                ranges.append((a, b))
                a = b = v
        ranges.append((a, b))
        out = []
        for aa, bb in ranges:
            out.append(str(aa) if aa == bb else f"{aa}-{bb}")
        return "[" + ",".join(out) + "]"

    def _fmt_suffixes(sufs: Set[str], max_show: int) -> str:
        sufs_sorted = sorted(sufs)
        if len(sufs_sorted) <= max_show:
            return ", ".join(sufs_sorted)
        head = ", ".join(sufs_sorted[:max_show])
        return f"{head}, ... (+{len(sufs_sorted) - max_show} more)"

    def _summarize_block_family(prefix: str, title: str) -> None:
        hot_layers: Set[int] = set()
        cold_layers: Set[int] = set()
        hot_suffixes: Set[str] = set()
        cold_suffixes: Set[str] = set()

        prefix_dot = prefix + "."

        for name, p in model.named_parameters():
            if not name.startswith(prefix_dot):
                continue

            # parse: {prefix}.{i}.{suffix}
            rest = name[len(prefix_dot):]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                continue

            try:
                layer_i = int(parts[0])
            except Exception:
                continue

            suffix = parts[1]

            if bool(getattr(p, "requires_grad", False)):
                hot_layers.add(layer_i)
                hot_suffixes.add(suffix)
            else:
                cold_layers.add(layer_i)
                cold_suffixes.add(suffix)

        print(Fore.YELLOW + f"{title}:" + reset)
        print(
            f"{hot}HOT {reset}  {prefix}.{block}{_fmt_layers(hot_layers)}{reset}  "
            f"{{{_fmt_suffixes(hot_suffixes, max_suffix_show)}}}"
        )
        print(
            f"{cold}COLD{reset} {prefix}.{block}{_fmt_layers(cold_layers)}{reset} "
            f"{{{_fmt_suffixes(cold_suffixes, max_suffix_show)}}}"
        )

    print("-----------Params-----------")
    _summarize_block_family(text_block_prefix, title_text)
    _summarize_block_family(vision_block_prefix, title_vision)

    print(Fore.YELLOW + "Other Parameters:" + reset)

    hot_params: List[str] = []
    cold_params: List[str] = []

    for name, param in model.named_parameters():
        if name.startswith(text_block_prefix + ".") or name.startswith(vision_block_prefix + "."):
            continue

        if bool(getattr(param, "requires_grad", False)):
            hot_params.append(name)
        else:
            cold_params.append(name)

    for name in hot_params:
        print(f"{hot}HOT{reset}  {name}")

    for name in cold_params:
        print(f"{cold}COLD{reset}  {name}")

    print("----------------------------")


def _call_with_signature(fn, **kwargs):
    """
    Calls fn with only the kwargs it accepts (by name).
    Prevents crashes if probe_utils.run_quick_probe has a different signature.
    """
    sig = inspect.signature(fn)
    accepted = {}
    for name, p in sig.parameters.items():
        if name in kwargs:
            accepted[name] = kwargs[name]
    return fn(**accepted)

def _get_model_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype

def _get_image_dtype(model: nn.Module) -> torch.dtype:
    return model.visual.conv1.weight.dtype

def compute_linear_probe(image_embs: torch.Tensor, label_indices: torch.Tensor, max_iter: int) -> float:
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    
    clf = LogisticRegression(max_iter=max_iter, multi_class='multinomial', solver='lbfgs')
    X = image_embs.cpu().numpy()
    y = label_indices.cpu().numpy()
    clf.fit(X, y)
    preds = clf.predict(X)
    return float(accuracy_score(y, preds))

@torch.no_grad()
def compute_zero_shot(image_embs: torch.Tensor, class_prompts: List[str], label_indices: torch.Tensor, model, clip, device: str) -> float:
    text_tokens = clip.tokenize(class_prompts, truncate=True).to(device)
    from sklearn.metrics import accuracy_score
    with autocast():
        text_embs = model.encode_text(text_tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        sims = image_embs @ text_embs.t()
        preds = torch.argmax(sims, dim=1).cpu()
    return float(accuracy_score(label_indices.cpu().numpy(), preds.numpy()))

@torch.no_grad()
def run_quick_probe(model, clip, probe_loader, classnames: List[str], label_indices: torch.Tensor, device: str, max_iter: int) -> Tuple[float, float]:
    model.eval()
    embs = []

    for imgs, _ in tqdm(probe_loader, desc="QuickProbe encode", ncols=100):
        imgs = imgs.to(device, non_blocking=True, dtype=_get_image_dtype(model))
        with autocast():
            emb = model.encode_image(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb.float().cpu())

    image_embs = torch.cat(embs, dim=0)
    lin_acc = compute_linear_probe(image_embs, label_indices, max_iter=max_iter)
    zs_acc = compute_zero_shot(image_embs.to(device), classnames, label_indices, model, clip, device)
    return lin_acc, zs_acc

class TinyImageFolderDataset(Dataset):
    """
    Typographic attack val dataset.
    """    
    def __init__(self, folder: str, preprocess):
        self.folder = folder
        self.preprocess = preprocess
        files = []
        if os.path.isdir(folder):
            for root, _, fnames in os.walk(folder):
                for fn in fnames:
                    if fn.lower().endswith(_IMG_EXTS):
                        files.append(os.path.join(root, fn))
        self.files = sorted(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = self.preprocess(img)
        return img, p

@torch.no_grad()
def run_tiny_benchmark(
    model,
    clip,
    preprocess,
    device: str,
    folders: List[str],
    choices: List[str],
    correct_choice_idx: int = 0,
    batch_size: int = 64,
) -> Dict[str, Dict[str, float]]:
    """
    Returns dict per folder:
      acc, mean_margin, mean_logit_correct, mean_logit_othermax
    """
    model.eval()

    text_tokens = clip.tokenize(choices, truncate=True).to(device)
    with autocast():
        text_emb = model.encode_text(text_tokens).float()
    text_emb = F.normalize(text_emb, dim=-1)

    logit_scale = None
    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.detach().float().exp().item()

    out: Dict[str, Dict[str, float]] = {}
    for folder in folders:
        ds = TinyImageFolderDataset(folder, preprocess=preprocess)
        if len(ds) == 0:
            out[folder] = {"acc": float("nan"), "mean_margin": float("nan"), "n": 0}
            continue

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        preds_all = []
        margins = []
        logits_c = []
        logits_o = []

        for imgs, paths in loader:
            imgs = imgs.to(device, non_blocking=True, dtype=_get_image_dtype(model))
            with autocast():
                img_emb = model.encode_image(imgs).float()
            img_emb = F.normalize(img_emb, dim=-1)

            logits = img_emb @ text_emb.t()
            if logit_scale is not None:
                logits = logits * logit_scale

            pred = torch.argmax(logits, dim=1)
            preds_all.append(pred.cpu())

            c = logits[:, correct_choice_idx]
            mask = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
            mask[correct_choice_idx] = False
            other_max = logits[:, mask].max(dim=1).values

            margin = (c - other_max)
            margins.append(margin.cpu())
            logits_c.append(c.cpu())
            logits_o.append(other_max.cpu())

        preds_all = torch.cat(preds_all)
        margins = torch.cat(margins)
        logits_c = torch.cat(logits_c)
        logits_o = torch.cat(logits_o)

        acc = float((preds_all == correct_choice_idx).float().mean().item())
        out[folder] = {
            "acc": acc,
            "mean_margin": float(margins.mean().item()),
            "mean_logit_correct": float(logits_c.mean().item()),
            "mean_logit_othermax": float(logits_o.mean().item()),
            "n": int(len(ds)),
        }

    return out