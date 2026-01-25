from __future__ import annotations

import torch
from colorama import Fore, Style
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Dict


def _get_model_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype

def _get_image_dtype(model: nn.Module) -> torch.dtype:
    return model.visual.conv1.weight.dtype

def calculate_metrics(logits, ground_truth):
    from sklearn.metrics import f1_score, accuracy_score
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1
    
@torch.no_grad()
def compute_gapness_metrics(model, val_dataloader, device: str, max_batches: int = 30) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    
    model.eval()
    imgs = []
    txts = []

    for b_idx, (images, texts) in enumerate(val_dataloader):
        if b_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True, dtype=_get_image_dtype(model))
        texts = texts.to(device, non_blocking=True)

        with autocast():
            img_e = model.encode_image(images).float()
            txt_e = model.encode_text(texts).float()

        img_e = F.normalize(img_e, dim=-1).cpu()
        txt_e = F.normalize(txt_e, dim=-1).cpu()
        imgs.append(img_e)
        txts.append(txt_e)

    if not imgs:
        return {"centroid_gap_l2": float("nan"), "linsep_acc": float("nan"), "n": 0}

    img_all = torch.cat(imgs, dim=0)
    txt_all = torch.cat(txts, dim=0)
    n = img_all.shape[0]

    # centroid euclidean distance
    centroid_gap = float((img_all.mean(dim=0) - txt_all.mean(dim=0)).norm().item())

    # linear separability (image=0, text=1) with a simple train/test split
    X = torch.cat([img_all, txt_all], dim=0).numpy()
    y = torch.cat([torch.zeros(n), torch.ones(n)], dim=0).numpy()

    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(X.shape[0], generator=rng).numpy()
    split = int(0.8 * X.shape[0])
    tr, te = perm[:split], perm[split:]

    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X[tr], y[tr])
    acc = float((clf.predict(X[te]) == y[te]).mean())

    return {"centroid_gap_l2": centroid_gap, "linsep_acc": acc, "n": int(X.shape[0])}

def monitor_gradient_norms(
    gradient_norms_raw,
    gradient_norms_unscaled,
    gradient_rms_unscaled,
    scale: float,
    threshold=1e-5,
    explode_threshold=1000.0,
):
    for name, norms_u in gradient_norms_unscaled.items():
        mean_u = sum(norms_u) / max(1, len(norms_u))
        mean_raw = sum(gradient_norms_raw.get(name, [0.0])) / max(1, len(gradient_norms_raw.get(name, [1.0])))

        rms_list = gradient_rms_unscaled.get(name, [])
        mean_rms = sum(rms_list) / max(1, len(rms_list)) if rms_list else 0.0

        if mean_u < threshold:
            print(
                Fore.RED
                + f"Vanishing gradient in {name}: raw={mean_raw:.2e} unscaled={mean_u:.2e} rms_unscaled={mean_rms:.2e} (scale={scale:.0f})"
                + Style.RESET_ALL
            )
        elif mean_u > explode_threshold:
            print(
                Fore.RED
                + f"Exploding gradient in {name}: raw={mean_raw:.2e} unscaled={mean_u:.2e} rms_unscaled={mean_rms:.2e} (scale={scale:.0f})"
                + Style.RESET_ALL
            )

def _assert_vit_visual_or_abort(model, cfg: TrainConfig):
    """
    Abort if ViT-specific features are enabled but the visual backbone is not ViT.

    ViT required for:
      - KO config (decatt / kproj) because it assumes visual.transformer.resblocks + attn internals
      - Regression teachers because they assume tokenization, class_embedding, positional_embedding, ln_pre/ln_post, proj
    """
    needs_vit = bool(cfg.use_ko_config) or bool(cfg.use_regression_teachers)
    if not needs_vit:
        return

    visual = getattr(model, "visual", None)
    has_vit = (visual is not None) and hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks")

    if has_vit:
        return

    looks_cnn = False
    if visual is not None:
        looks_cnn = any(hasattr(visual, k) for k in ["layer1", "layer2", "layer3", "layer4"])

    feat_flags = []
    if cfg.use_ko_config:
        feat_flags.append("use_ko_config=True")
    if cfg.use_regression_teachers:
        feat_flags.append("use_regression_teachers=True")

    cnn_hint = " (visual looks like a CNN/ResNet: has layer1..layer4)" if looks_cnn else ""
    raise SystemExit(
        "[Abort] ViT visual backbone required for enabled features: "
        + ", ".join(feat_flags)
        + ".\n"
        + f"Detected visual backbone without visual.transformer.resblocks{cnn_hint}.\n"
        + "Fix: disable KO/teacher features, or load a ViT-based CLIP model."
    )