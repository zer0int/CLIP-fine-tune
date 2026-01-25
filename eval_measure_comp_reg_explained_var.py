"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

Singular register analysis for CLIP ViT-L/14-style vision transformers.

Inspired by:

SINDER: Repairing the Singular regs of DINOv2
arXiv:2407.16826v

This script measures whether high-norm patch tokens (“register-like” patches) form a low-rank / stable
direction in CLIP’s image embedding space, and whether that direction is specifically involved in
mapping local patch content to the global CLS image embedding.

Core idea (CLIP adaptation of SINDER / singular-feature diagnostics):
  1) Pick a set of resblock layers (e.g., late layers).
  2) For each image and each target layer, collect the token stream after that resblock.
  3) Define “register patches” implicitly as patch tokens whose L2 norm exceeds a threshold:
        reg(i) = { p | ||x_patch(i,p)||_2 >= reg_threshold }
     The remaining patches are treated as “normal” patches.

For each (model, layer), the script computes:

A) Register prevalence (how often these high-norm patches appear)
   - mean_n_register: mean number of register patches per image
   - frac_with_any_register: fraction of images with ≥ 1 register patch

B) Register geometry / “singularity”
   - Intra-register angle: within-image mean pairwise angle between register vectors (in proj space)
   - Intra-normal angle: same diagnostic for a matched set of “normal” patches
     (matched count: top-k highest-norm normals with k = #registers)
   - Inter-image angle: mean pairwise angle between per-image mean register directions
   - Global register direction v1: first right-singular vector of the centered matrix of all
     register vectors across images; and its explained variance:
        explained = s1^2 / sum_j s_j^2
   - Alignment: |cos(v1, mean_def)| where mean_def is the mean of per-image register directions

C) Patch→CLS “teacher” probe, and effect of removing the register direction
   - Build a patch summary embedding by averaging (optionally excluding register patches),
     then applying CLIP’s ln_post and proj to get into the same D-dim space as CLS:
        patch_embed(i) = proj( ln_post( mean_{p in normal(i)} x_patch(i,p) ) )
     where normal(i) excludes patches with ||x_patch|| >= reg_threshold.
   - Fit a linear least-squares map (“teacher”) from patch_embed to CLS embedding on a fixed split:
        (P - μP) X ≈ (C - μC),   W = X^T
     and evaluate on held-out data with mean cosine similarity + MSE.
   - Project out the global register direction v1 from patch embeddings only:
        P_proj = P - (P·v1) v1
     refit the teacher, and re-evaluate. This tests whether the register direction is a key
     component in predicting CLS from the patch summary.

Practical interpretation:
  - High reg_dir_explained_var + high |cos(v1, mean_def)| + low inter_def_angle_deg suggests
    register vectors collapse to a stable global direction (a “singular” feature).
  - A large change between teacher_cos and teacher_cos_proj suggests that direction contributes
    meaningfully to patch→CLS predictability (or that removing it forces the probe to use other cues).

"""
import os
import re
import json
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import oaiclip as clip
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


def parse_args():
    parser = argparse.ArgumentParser("SINDER-style singular register test for CLIP ViT-L/14 (threshold registers)")
    parser.add_argument("--models", type=str, default="all", help="Model aliases from MODELS to run. Comma-separated, or 'all'",
    )
    parser.add_argument("--csv_file", type=str, default="utils_datasets/mvt/human_responses_5k.csv")
    parser.add_argument("--image_folder", type=str, default="path/to/dataset-difficulty-CLIP/data_release_2023/all/")
    parser.add_argument("--out_dir", type=str, default="out_eval_measure/comp_reg_explained_var")

    parser.add_argument("--layers", type=str, default="12,13,20,21,22,23", help="Comma-separated resblock indices.")
    parser.add_argument("--reg_threshold", type=float, default=70.0, help="Patches with norm>=threshold are treated as registers.")
    
    parser.add_argument("--max_images", type=int, default=200, help="0 = all rows; else cap.")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--hard_only", action="store_true", help="Keep only the 'hard' subset according to hard_mode/hard_frac.")
    parser.add_argument("--hard_mode", type=str, default="duration", choices=["duration", "rt", "wrong", "duration_wrong", "rt_wrong"])
    parser.add_argument("--hard_frac", type=float, default=0.2, help="Fraction of rows to keep when hard_mode uses quantiles.")
    parser.add_argument("--only_objectnet", action="store_true", help="Keep only objectnet==True rows")
    parser.add_argument("--only_imagenet", action="store_true", help="Keep only objectnet==False rows")
    
    return parser.parse_args()


def fix_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make deterministic-ish (may reduce throughput)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Ensures dataloader workers are deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_file_or_df, image_folder: str, transform=None):
        if isinstance(csv_file_or_df, str):
            self.data = pd.read_csv(csv_file_or_df)
        else:
            self.data = csv_file_or_df.reset_index(drop=True)

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
        label = self.data.iloc[idx]["label"]
        return image, label, image_name


@torch.no_grad()
def encode_image_blocks_tokens(
    visual: torch.nn.Module,
    images: torch.Tensor,
    target_layers: List[int],
    detach: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Returns dict: layer_idx -> tokens [B, seq, d] AFTER that resblock.
    (Matches "post_block_tokens[layer]" style hooks.)
    """
    x = visual.conv1(images)
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

    cls = visual.class_embedding.to(x.dtype)
    cls_tokens = cls + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+HW, C]

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # [seq, B, d]

    want = set(target_layers)
    out: Dict[int, torch.Tensor] = {}

    for i, block in enumerate(visual.transformer.resblocks):
        x = block(x)
        if i in want:
            t = x.permute(1, 0, 2)  # [B, seq, d]
            out[i] = t.detach() if detach else t

    return out


@torch.no_grad()
def tokens_to_patch_embed(
    tokens: torch.Tensor,
    ln_post: torch.nn.Module,
    proj: torch.Tensor,
    reg_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    tokens: [B, seq, d] after some block
    Returns patch summary embedding in proj-space: [B, D].

    If reg_threshold is set:
      exclude patches with norm >= threshold (treat as "REG/register").
    """
    patch = tokens[:, 1:, :]       # [B, HW, d]
    norms = patch.norm(dim=-1)     # [B, HW]

    if reg_threshold is None:
        mask = torch.ones_like(norms, dtype=torch.bool)
    else:
        mask = norms < reg_threshold

    patch_hidden_list = []
    for b in range(tokens.shape[0]):
        sel = patch[b][mask[b]] if mask[b].any() else patch[b]
        mean_tok = sel.mean(dim=0, keepdim=True)  # [1, d]
        mean_tok = ln_post(mean_tok)              # [1, d]
        patch_hidden_list.append(mean_tok)

    patch_hidden = torch.cat(patch_hidden_list, dim=0)  # [B, d]
    patch_embed = patch_hidden @ proj                   # [B, D]
    return patch_embed


def pairwise_mean_angle_deg(unit_vectors: torch.Tensor) -> float:
    """
    unit_vectors: [K, D] normalized
    Returns mean pairwise angle in degrees among all pairs.
    """
    K = unit_vectors.shape[0]
    if K < 2:
        return float("nan")
    cos = unit_vectors @ unit_vectors.T
    iu = torch.triu_indices(K, K, offset=1)
    vals = cos[iu[0], iu[1]].clamp(-1.0, 1.0)
    ang = torch.acos(vals) * (180.0 / math.pi)
    return float(ang.mean().item())


def threshold_register_patch_vectors(
    tokens: torch.Tensor,
    ln_post: torch.nn.Module,
    proj: torch.Tensor,
    reg_threshold: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    Per image:
      register = ALL patches with norm >= reg_threshold
      normal    = ALL patches with norm <  reg_threshold
      comparison normals = top-(#register) highest-norm normals (< threshold), deterministic

    Returns:
      register_unit_vecs_per_image: list of [Kdef_i, D] unit (Kdef varies per image)
      normal_unit_vecs_per_image:    list of [Kdef_i, D] unit (matched count; may be smaller if too few normals)
      n_register_per_image:         list[int]
    """
    patch = tokens[:, 1:, :]             # [B, HW, d]
    norms = patch.norm(dim=-1)           # [B, HW]
    B, HW, d = patch.shape

    register_list: List[torch.Tensor] = []
    normal_list: List[torch.Tensor] = []
    n_def_list: List[int] = []

    D_out = int(proj.shape[1])

    for b in range(B):
        norms_b = norms[b]  # [HW]
        def_mask = norms_b >= reg_threshold
        nor_mask = ~def_mask

        def_idx = torch.nonzero(def_mask, as_tuple=False).squeeze(1)  # [Kdef]
        nor_idx = torch.nonzero(nor_mask, as_tuple=False).squeeze(1)  # [Knor]

        Kdef = int(def_idx.numel())
        n_def_list.append(Kdef)

        if Kdef == 0:
            register_list.append(torch.empty((0, D_out), dtype=torch.float32))
            normal_list.append(torch.empty((0, D_out), dtype=torch.float32))
            continue

        # deterministic normal selection: take the highest-norm normals (< threshold)
        if nor_idx.numel() > 0:
            nor_norms = norms_b[nor_idx]  # [Knor]
            k_take = min(Kdef, int(nor_norms.numel()))
            take_local = torch.topk(nor_norms, k=k_take, largest=True).indices
            nor_sel_idx = nor_idx[take_local]  # [k_take]
        else:
            # pathological: no normals, fall back to first k patches
            k_take = min(Kdef, HW)
            nor_sel_idx = torch.arange(HW, device=patch.device)[:k_take]

        def_tok = patch[b, def_idx, :]           # [Kdef, d]
        nor_tok = patch[b, nor_sel_idx, :]       # [k_take, d]

        def_h = ln_post(def_tok) @ proj          # [Kdef, D]
        nor_h = ln_post(nor_tok) @ proj          # [k_take, D]

        def_u = F.normalize(def_h.float(), dim=-1)
        nor_u = F.normalize(nor_h.float(), dim=-1)

        register_list.append(def_u.cpu())
        normal_list.append(nor_u.cpu())

    return register_list, normal_list, n_def_list


def solve_lstsq_teacher(P: torch.Tensor, C: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    P: [N, D] patch embeds
    C: [N, D] CLS embeds
    Returns dict with means and W (same convention as your training code).
    """
    P_mean = P.mean(dim=0, keepdim=True)
    C_mean = C.mean(dim=0, keepdim=True)
    P_center = P - P_mean
    C_center = C - C_mean
    res = torch.linalg.lstsq(P_center, C_center)
    X = res.solution           # [D, D] where P_center @ X ~ C_center
    W = X.T                    # so (p_center @ W.T) ~ c_center
    return {"patch_mean": P_mean, "cls_mean": C_mean, "W": W}


@torch.no_grad()
def eval_teacher(P: torch.Tensor, C: torch.Tensor, teacher: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    """
    Returns: (mean cosine, MSE)
    """
    P_mean = teacher["patch_mean"]
    C_mean = teacher["cls_mean"]
    W = teacher["W"]

    P_center = P - P_mean
    C_hat = (P_center @ W.T) + C_mean

    mse = torch.mean((C_hat - C) ** 2).item()
    cos = (F.normalize(C_hat, dim=-1) * F.normalize(C, dim=-1)).sum(dim=-1).mean().item()
    return float(cos), float(mse)


def project_out_direction(X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project out direction v from rows of X.
    X: [N, D], v: [D] (assumed unit)
    """
    v = v.view(1, -1)
    coeff = (X @ v.T)          # [N, 1]
    return X - coeff * v


@dataclass
class RunResult:
    model_tag: str
    layer: int
    n_images: int
    reg_threshold: float

    mean_n_register: float
    frac_with_any_register: float

    intra_def_angle_deg: float
    intra_norm_angle_deg: float
    inter_def_angle_deg: float

    reg_dir_explained_var: float

    teacher_cos: float
    teacher_mse: float
    teacher_cos_proj: float
    teacher_mse_proj: float

    reg_dir_cos_with_mean_def: float


def sanitize_tag(s: str) -> str:
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:180]


def load_model_and_preprocess(model_spec: str, device: str):
    model, preprocess, _ = load_openai_clip_anything(clip, model_spec, device=device, jit=False, strict=True)
    model = model.float().eval()
    return model, preprocess


def run_sinder_style_analysis(
    model_tag: str,
    model_spec: str,
    csv_file: str,
    image_folder: str,
    out_dir: str,
    device: str,
    layers: List[int],
    max_images: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    reg_threshold: float,
    hard_only: bool,
    hard_mode: str,
    hard_frac: float,
    only_objectnet: bool,
    only_imagenet: bool,
) -> pd.DataFrame:
    fix_random_seed(seed)

    model, preprocess = load_model_and_preprocess(model_spec, device=device)
    visual = model.visual
    ln_post = visual.ln_post
    proj = visual.proj

    df = pd.read_csv(csv_file)

    # optional dataset slice
    if "objectnet" in df.columns:
        if only_objectnet:
            df = df[df["objectnet"] == True].copy()
        if only_imagenet:
            df = df[df["objectnet"] == False].copy()

    # correctness flag
    if "label" in df.columns and "response" in df.columns:
        df["is_correct"] = (df["label"].astype(str) == df["response"].astype(str))
    else:
        df["is_correct"] = True

    if hard_only:
        mode = hard_mode

        if "image_duration" in df.columns:
            df["image_duration"] = pd.to_numeric(df["image_duration"], errors="coerce")
        if "response_time" in df.columns:
            df["response_time"] = pd.to_numeric(df["response_time"], errors="coerce")

        if mode in ["wrong", "duration_wrong", "rt_wrong"]:
            df = df[df["is_correct"] == False].copy()

        if mode in ["duration", "duration_wrong"]:
            if "image_duration" not in df.columns:
                raise ValueError("hard_mode=duration* requires column 'image_duration'")
            thr = df["image_duration"].quantile(hard_frac)
            df = df[df["image_duration"] <= thr].copy()

        if mode in ["rt", "rt_wrong"]:
            if "response_time" not in df.columns:
                raise ValueError("hard_mode=rt* requires column 'response_time'")
            thr = df["response_time"].quantile(1.0 - hard_frac)
            df = df[df["response_time"] >= thr].copy()

        df = df.reset_index(drop=True)

    if max_images > 0:
        df = df.iloc[:max_images].copy().reset_index(drop=True)

    dataset = CroppedImageCSVFileDataset(df, image_folder, transform=preprocess)

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    os.makedirs(out_dir, exist_ok=True)
    model_out = os.path.join(out_dir, sanitize_tag(model_tag))
    os.makedirs(model_out, exist_ok=True)

    results: List[RunResult] = []

    for layer in layers:
        all_cls = []
        all_patch = []

        intra_def_angles = []
        intra_norm_angles = []
        mean_def_dirs = []
        all_def_vecs = []

        n_images = 0
        n_register_total = 0
        n_register_any = 0

        pbar = tqdm(dataloader, desc=f"[{model_tag}] layer={layer}", ncols=110)
        for images, labels, image_names in pbar:
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                cls_embed = model.encode_image(images).float()  # [B, D]

                toks = encode_image_blocks_tokens(
                    visual=visual,
                    images=images,
                    target_layers=[layer],
                    detach=True,
                )[layer]  # [B, seq, d]

                patch_embed = tokens_to_patch_embed(
                    tokens=toks,
                    ln_post=ln_post,
                    proj=proj,
                    reg_threshold=reg_threshold,  # exclude norm>=threshold
                ).float()  # [B, D]

                # reg selection: threshold-based, variable count per image
                register_list, normal_list, n_def_list = threshold_register_patch_vectors(
                    tokens=toks,
                    ln_post=ln_post,
                    proj=proj,
                    reg_threshold=float(reg_threshold),
                )

            all_cls.append(cls_embed.cpu())
            all_patch.append(patch_embed.cpu())
            n_images += cls_embed.shape[0]

            for nd in n_def_list:
                n_register_total += int(nd)
                if nd > 0:
                    n_register_any += 1

            for def_u, nor_u in zip(register_list, normal_list):
                intra_def_angles.append(pairwise_mean_angle_deg(def_u))
                intra_norm_angles.append(pairwise_mean_angle_deg(nor_u))

                if def_u.shape[0] > 0:
                    m = F.normalize(def_u.mean(dim=0), dim=-1)
                    mean_def_dirs.append(m.unsqueeze(0))
                    all_def_vecs.append(def_u)

        # Stack CLS/PATCH embeddings
        C = torch.cat(all_cls, dim=0)   # [N, D]
        P = torch.cat(all_patch, dim=0) # [N, D]

        # Global reg direction via SVD on register vectors
        v1: Optional[torch.Tensor] = None
        explained = float("nan")
        inter_angle = float("nan")
        cos_v_avg = float("nan")

        if len(all_def_vecs) > 0 and len(mean_def_dirs) > 0:
            mean_def = torch.cat(mean_def_dirs, dim=0)  # [Ndef_img, D] unit
            def_mat = torch.cat(all_def_vecs, dim=0)    # [M, D] unit

            X = def_mat.float()
            X = X - X.mean(dim=0, keepdim=True)

            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            v1 = F.normalize(Vh[0], dim=-1)  # unit

            s2 = (S ** 2)
            explained = float((s2[0] / (s2.sum() + 1e-12)).item())

            Ndef = mean_def.shape[0]
            if Ndef >= 2:
                if Ndef <= 2000:
                    cos = (mean_def @ mean_def.T).clamp(-1, 1)
                    iu = torch.triu_indices(Ndef, Ndef, offset=1)
                    vals = cos[iu[0], iu[1]]
                    inter_angle = float((torch.acos(vals) * (180.0 / math.pi)).mean().item())
                else:
                    # deterministic under fixed seed
                    idx = torch.randint(0, Ndef, (200000,))
                    jdx = torch.randint(0, Ndef, (200000,))
                    vals = (mean_def[idx] * mean_def[jdx]).sum(dim=-1).clamp(-1, 1)
                    inter_angle = float((torch.acos(vals) * (180.0 / math.pi)).mean().item())

            avg_def_dir = F.normalize(mean_def.mean(dim=0), dim=-1)
            cos_v_avg = float(torch.dot(v1, avg_def_dir).abs().item())

        # Teacher fit patch -> CLS (train/val split deterministic)
        fix_random_seed(seed)  # ensure deterministic split
        perm = torch.randperm(C.shape[0])
        train_N = int(0.9 * C.shape[0])
        tr = perm[:train_N]
        va = perm[train_N:]

        C_tr, C_va = C[tr], C[va]
        P_tr, P_va = P[tr], P[va]

        teacher = solve_lstsq_teacher(P_tr, C_tr)
        cos0, mse0 = eval_teacher(P_va, C_va, teacher)

        # Project out reg direction from PATCH embeddings only
        if v1 is not None:
            P_tr_proj = project_out_direction(P_tr, v1)
            P_va_proj = project_out_direction(P_va, v1)
            teacher_proj = solve_lstsq_teacher(P_tr_proj, C_tr)
            cos1, mse1 = eval_teacher(P_va_proj, C_va, teacher_proj)
        else:
            cos1, mse1 = cos0, mse0

        rr = RunResult(
            model_tag=model_tag,
            layer=layer,
            n_images=int(C.shape[0]),
            reg_threshold=float(reg_threshold),
            mean_n_register=float(n_register_total / max(1, n_images)),
            frac_with_any_register=float(n_register_any / max(1, n_images)),
            intra_def_angle_deg=float(np.nanmean(intra_def_angles)),
            intra_norm_angle_deg=float(np.nanmean(intra_norm_angles)),
            inter_def_angle_deg=float(inter_angle),
            reg_dir_explained_var=float(explained),
            teacher_cos=float(cos0),
            teacher_mse=float(mse0),
            teacher_cos_proj=float(cos1),
            teacher_mse_proj=float(mse1),
            reg_dir_cos_with_mean_def=float(cos_v_avg),
        )
        results.append(rr)

        # Save layer artifacts
        layer_dir = os.path.join(model_out, f"{model_tag}_layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        with open(os.path.join(layer_dir, f"{model_tag}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(rr.__dict__, f, indent=2)

        # Save reg direction vector (if present)
        if v1 is not None:
            torch.save({"v1": v1.cpu()}, os.path.join(layer_dir, f"{model_tag}_reg_direction_v1.pt"))
        else:
            torch.save({"v1": None}, os.path.join(layer_dir, f"{model_tag}_reg_direction_v1.pt"))

    # Save overall CSV
    rows = [r.__dict__ for r in results]
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(model_out, f"{model_tag}_results.csv"), index=False)

    # (1) Teacher cosine before/after projection (single model)
    plt.figure(figsize=(12, 6))
    plt.plot(df_res["layer"], df_res["teacher_cos"], marker="o", label="teacher_cos")
    plt.plot(df_res["layer"], df_res["teacher_cos_proj"], marker="o", label="teacher_cos_proj")
    plt.title(f"{model_tag}: patch→CLS teacher cosine across layers")
    plt.xlabel("layer")
    plt.ylabel("mean cosine (val)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_out, f"{model_tag}_teacher_cos_vs_layer.png"))
    plt.close()

    # (2) reg prevalence
    plt.figure(figsize=(12, 6))
    plt.plot(df_res["layer"], df_res["mean_n_register"], marker="o", label="mean_n_register")
    plt.plot(df_res["layer"], df_res["frac_with_any_register"], marker="o", label="frac_with_any_register")
    plt.title(f"{model_tag}: register prevalence across layers (threshold={reg_threshold})")
    plt.xlabel("layer")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_out, f"{model_tag}_reg_prevalence_vs_layer.png"))
    plt.close()

    # (3) reg direction "geometry": explained variance + inter-image angle + alignment to mean direction
    plt.figure(figsize=(12, 6))
    plt.plot(df_res["layer"], df_res["reg_dir_explained_var"], marker="o", label="reg_dir_explained_var")
    plt.plot(df_res["layer"], df_res["reg_dir_cos_with_mean_def"], marker="o", label="reg_dir_cos_with_mean_def")
    plt.title(f"{model_tag}: reg direction stability (SVD explained var, |cos(v1, mean_def)|)")
    plt.xlabel("layer")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_out, f"{model_tag}_reg_direction_stability_vs_layer.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(df_res["layer"], df_res["inter_def_angle_deg"], marker="o", label="inter_def_angle_deg")
    plt.plot(df_res["layer"], df_res["intra_def_angle_deg"], marker="o", label="intra_def_angle_deg")
    plt.plot(df_res["layer"], df_res["intra_norm_angle_deg"], marker="o", label="intra_norm_angle_deg")
    plt.title(f"{model_tag}: angle diagnostics (degrees)")
    plt.xlabel("layer")
    plt.ylabel("degrees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_out, f"{model_tag}_angles_vs_layer.png"))
    plt.close()

    print(f"\nWrote results to: {model_out}")
    print(
        df_res[
            [
                "layer",
                "n_images",
                "reg_threshold",
                "mean_n_register",
                "frac_with_any_register",
                "intra_def_angle_deg",
                "intra_norm_angle_deg",
                "inter_def_angle_deg",
                "reg_dir_explained_var",
                "reg_dir_cos_with_mean_def",
                "teacher_cos",
                "teacher_cos_proj",
            ]
        ].to_string(index=False)
    )

    # Explicitly free
    del model, visual, ln_post, proj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df_res


def parse_layers(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _select_models(models_arg: str) -> List[Tuple[str, str]]:
    arg = (models_arg or "").strip()
    if arg.lower() in ["all", "*", ""]:
        return MODELS

    wanted = [a.strip() for a in arg.split(",") if a.strip()]
    wanted_set = set(wanted)

    sel = [(alias, spec) for (alias, spec) in MODELS if alias in wanted_set]
    missing = [a for a in wanted if a not in {m[0] for m in MODELS}]
    if missing:
        print(f"[WARN] Unknown model aliases in --models (ignored): {missing}")
    if not sel:
        raise ValueError("No valid model aliases selected. Use --models all or provide valid aliases from MODELS.")
    return sel


def _combined_plots(out_dir: str, reg_threshold: float, per_model_df: Dict[str, pd.DataFrame]) -> None:
    """
    Create combined (multi-model) plots if >1 model.
    """
    if len(per_model_df) <= 1:
        return

    comp_dir = os.path.join(out_dir, "compare_models")
    os.makedirs(comp_dir, exist_ok=True)

    # Teacher cosine proj across layers
    plt.figure(figsize=(12, 6))
    for alias, df in per_model_df.items():
        plt.plot(df["layer"], df["teacher_cos_proj"], marker="o", label=alias)
    plt.title("Compare models: teacher_cos_proj across layers")
    plt.xlabel("layer")
    plt.ylabel("mean cosine (val)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "compare_teacher_cos_proj_vs_layer.png"))
    plt.close()

    # Mean register count across layers
    plt.figure(figsize=(12, 6))
    for alias, df in per_model_df.items():
        plt.plot(df["layer"], df["mean_n_register"], marker="o", label=alias)
    plt.title(f"Compare models: mean_n_register across layers (threshold={reg_threshold})")
    plt.xlabel("layer")
    plt.ylabel("mean # register")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "compare_mean_n_register_vs_layer.png"))
    plt.close()

    # Fraction with any reg across layers
    plt.figure(figsize=(12, 6))
    for alias, df in per_model_df.items():
        plt.plot(df["layer"], df["frac_with_any_register"], marker="o", label=alias)
    plt.title(f"Compare models: frac_with_any_register across layers (threshold={reg_threshold})")
    plt.xlabel("layer")
    plt.ylabel("fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "compare_frac_with_any_register_vs_layer.png"))
    plt.close()

    # reg direction explained variance across layers
    plt.figure(figsize=(12, 6))
    for alias, df in per_model_df.items():
        plt.plot(df["layer"], df["reg_dir_explained_var"], marker="o", label=alias)
    plt.title("Compare models: reg_dir_explained_var across layers")
    plt.xlabel("layer")
    plt.ylabel("explained variance (v1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "compare_reg_dir_explained_var_vs_layer.png"))
    plt.close()

    # Inter-image register direction angle
    plt.figure(figsize=(12, 6))
    for alias, df in per_model_df.items():
        plt.plot(df["layer"], df["inter_def_angle_deg"], marker="o", label=alias)
    plt.title("Compare models: inter_def_angle_deg across layers")
    plt.xlabel("layer")
    plt.ylabel("degrees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "compare_inter_def_angle_vs_layer.png"))
    plt.close()


def _final_diagnostic(per_model_df: Dict[str, pd.DataFrame]) -> None:
    """
    Print a compact end-of-run diagnostic comparing models (>1).
    Focus: patch->CLS recoverability (teacher_cos_proj), reg prevalence, and reg-direction stability.
    """
    if len(per_model_df) <= 1:
        return

    rows = []
    for alias, df in per_model_df.items():
        rows.append(
            {
                "model": alias,
                "teacher_cos_proj_mean": float(df["teacher_cos_proj"].mean()),
                "teacher_cos_proj_min": float(df["teacher_cos_proj"].min()),
                "mean_n_register_mean": float(df["mean_n_register"].mean()),
                "frac_any_def_mean": float(df["frac_with_any_register"].mean()),
                "reg_explained_mean": float(df["reg_dir_explained_var"].mean()),
                "inter_def_angle_mean_deg": float(df["inter_def_angle_deg"].mean()),
                "cos_v1_mean_def_mean": float(df["reg_dir_cos_with_mean_def"].mean()),
            }
        )

    summary = pd.DataFrame(rows)

    # If pretrained exists, compute delta columns vs pretrained means
    if "pretrained" in per_model_df:
        base = summary[summary["model"] == "pretrained"].iloc[0]
        for col in [
            "teacher_cos_proj_mean",
            "mean_n_register_mean",
            "frac_any_def_mean",
            "reg_explained_mean",
            "inter_def_angle_mean_deg",
            "cos_v1_mean_def_mean",
        ]:
            summary[f"Δpretrained_{col}"] = summary[col] - float(base[col])

    # Rank (primary) by teacher_cos_proj_mean descending, then by mean_n_register_mean ascending
    summary = summary.sort_values(
        by=["teacher_cos_proj_mean", "mean_n_register_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)

    print("\n" + "=" * 110)
    print("FINAL DIAGNOSTIC (multi-model comparison)")
    print("- Interpretation hints:")
    print("  • Higher teacher_cos_proj_mean => patch-summary predicts CLS better after removing reg dir (if present).")
    print("  • Lower mean_n_register_mean / frac_any_def_mean => fewer high-norm 'regs' under your threshold.")
    print("  • Higher reg_explained_mean + higher cos_v1_mean_def_mean => regs collapse to a stable global direction.")
    print("=" * 110)
    with pd.option_context("display.max_columns", 999, "display.width", 240):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = parse_layers(args.layers)

    reg_threshold = float(args.reg_threshold)
    if reg_threshold < 0:
        raise ValueError("For CLIP register-token test, reg_threshold must be >= 0.")

    selected = _select_models(args.models)

    per_model_df: Dict[str, pd.DataFrame] = {}

    for model_tag, model_spec in selected:
        df_res = run_sinder_style_analysis(
            model_tag=model_tag,
            model_spec=model_spec,
            csv_file=args.csv_file,
            image_folder=args.image_folder,
            out_dir=args.out_dir,
            device=device,
            layers=layers,
            max_images=args.max_images,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            reg_threshold=reg_threshold,
            hard_only=args.hard_only,
            hard_mode=args.hard_mode,
            hard_frac=args.hard_frac,
            only_objectnet=args.only_objectnet,
            only_imagenet=args.only_imagenet,
        )
        per_model_df[model_tag] = df_res

    # Combined plots + end diagnostic if >1 model
    _combined_plots(args.out_dir, reg_threshold=reg_threshold, per_model_df=per_model_df)
    _final_diagnostic(per_model_df)


if __name__ == "__main__":
    main()