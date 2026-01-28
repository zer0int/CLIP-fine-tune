from __future__ import annotations

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from colorama import Fore, Style
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any
import re

_NUMERIC_RE = re.compile(r"^\s*[+-]?(\d+(\.\d*)?|\.\d+)\s*$")

def _phase_banner(title: str, color=Fore.MAGENTA):
    bar = "=" * 78
    print(color + bar + Style.RESET_ALL)
    print(color + f"[PHASE] {title}" + Style.RESET_ALL)
    print(color + bar + Style.RESET_ALL)

def _subphase(msg: str, color=Fore.MAGENTA):
    print(color + f"  -> {msg}" + Style.RESET_ALL)

def _get_model_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype

def _get_image_dtype(model: nn.Module) -> torch.dtype:
    return model.visual.conv1.weight.dtype


# ============================================================
# J-L Projection (see: -> Johnson–Lindenstrauss lemma)
# ============================================================
class FixedJLProjector(nn.Module):
    """
    Fixed random Gaussian projection (Johnson–Lindenstrauss style).
    - P is frozen (no grads).
    - Forward: x -> x @ P^T
    """
    def __init__(self, d_in: int, d_out: int, seed: int = 0):
        super().__init__()
        assert d_out > 0 and d_in > 0
        self.d_in = int(d_in)
        self.d_out = int(d_out)

        g = torch.Generator()
        g.manual_seed(int(seed))

        # Gaussian N(0, 1/sqrt(d_out)) is a common scaling for JL
        P = torch.randn(self.d_out, self.d_in, generator=g) / (self.d_out ** 0.5)

        self.register_buffer("P", P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.P.t()

class JLProjectorEnsemble(nn.Module):
    """
    Ensemble of fixed JL projectors to reduce 'cheating' via the complement space.
    Losses/cosines can be averaged over members.
    """
    def __init__(self, d_in: int, d_out: int, n_proj: int, seed_base: int = 0, seed_stride: int = 1):
        super().__init__()
        assert n_proj > 0
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.n_proj = int(n_proj)

        self.projectors = nn.ModuleList([
            FixedJLProjector(d_in=d_in, d_out=d_out, seed=int(seed_base + i * seed_stride))
            for i in range(self.n_proj)
        ])

    def project_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: [n_proj, B, d_out]
        """
        outs = []
        for pj in self.projectors:
            outs.append(pj(x))
        return torch.stack(outs, dim=0)

def _normalize_jl_spec(jl: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    jl = dict(jl or {})
    jl.setdefault("enabled", False)
    jl.setdefault("dim", 128)
    jl.setdefault("seed", 1234)
    jl.setdefault("num_proj", 8)
    jl.setdefault("seed_stride", 1)
    return jl


# ============================================================
# Regression Teacher: Register threshold parsing
# ============================================================

def normalize_reg_threshold_spec(reg_threshold: Any) -> Dict[str, Any]:
    """
    Normalize reg_threshold into a canonical internal spec.

    Semantics (per request):
      - If reg_threshold is int/float OR a numeric string with NO colon -> absolute threshold.
      - If reg_threshold is ANY string containing a colon ":" -> adaptive threshold:
            thr_i = median(norms_i) * factor
        where factor is the float parsed from the substring AFTER the colon.

    Examples:
      - 70.0            -> abs
      - "70.0"          -> abs
      - "median:2.5"    -> median-adaptive with factor=2.5
      - "abs:2.5"       -> median-adaptive with factor=2.5   (colon => adaptive)
      - "wordsword:3"   -> median-adaptive with factor=3.0
    """
    # absolute numeric
    if isinstance(reg_threshold, (int, float)):
        v = float(reg_threshold)
        if not (v > 0):
            raise ValueError(f"reg_threshold absolute must be > 0, got {reg_threshold!r}")
        # canonical string is informational only; mode governs behavior
        spec = f"{v:.6f}".rstrip("0").rstrip(".")
        return {"mode": "abs", "value": v, "mult": None, "spec": spec}

    if isinstance(reg_threshold, str):
        s = reg_threshold.strip()

        # colon => adaptive median * factor (tag ignored)
        if ":" in s:
            _tag, rhs = s.split(":", 1)
            rhs = rhs.strip()
            mult = float(rhs)
            if not (mult > 0):
                raise ValueError(f"reg_threshold adaptive factor must be > 0, got {reg_threshold!r}")
            spec = f"median:{mult:.6f}".rstrip("0").rstrip(".")
            return {"mode": "median", "value": None, "mult": mult, "spec": spec}

        # no colon: must be numeric string => absolute
        if _NUMERIC_RE.match(s):
            v = float(s)
            if not (v > 0):
                raise ValueError(f"reg_threshold absolute must be > 0, got {reg_threshold!r}")
            spec = f"{v:.6f}".rstrip("0").rstrip(".")
            return {"mode": "abs", "value": v, "mult": None, "spec": spec}

        # safer to reject ambiguous non-numeric strings without colon
        raise ValueError(
            f"reg_threshold string must be numeric (absolute) or contain ':' (adaptive), got {reg_threshold!r}"
        )

    raise TypeError(f"Unsupported reg_threshold type/value: {type(reg_threshold)} / {reg_threshold!r}")



def compute_reg_mask_from_norms(norms: torch.Tensor, reg_threshold: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    norms: [B, P] patch token norms
    reg_threshold:
      - float/int or numeric string with no ':' => absolute
      - any string containing ':' => adaptive median * factor (factor after ':')

    Returns:
      reg_mask: [B, P] bool
      thr_used: [B, 1] float threshold used per sample (abs expands to per-sample)
    """
    spec = normalize_reg_threshold_spec(reg_threshold)

    if spec["mode"] == "abs":
        thr = torch.tensor(float(spec["value"]), device=norms.device, dtype=norms.dtype).view(1, 1)
        thr = thr.expand(norms.shape[0], 1)
        reg_mask = norms >= thr
        return reg_mask, thr

    # adaptive median * mult (per-sample)
    mult = float(spec["mult"])
    # median on fp16 can be quirky; do it in fp32 and cast back
    med = norms.float().median(dim=1).values.to(dtype=norms.dtype)  # [B]
    thr = (med * mult).view(-1, 1)                                  # [B, 1]
    reg_mask = norms >= thr
    return reg_mask, thr



# =========================
# Collect Embeddings
# =========================
def build_global_embeddings_from_tokens(
    tokens: torch.Tensor,
    ln_post: nn.Module,
    proj: torch.Tensor,
    reg_threshold: Any,
    cls_attn_weights: torch.Tensor | None = None,
    return_reg: bool = False,
    return_reg_stats: bool = False,
):
    # ============================================================
    # Attention Uber Alles!
    # Terrible idea, with catastrophic but interesting results.
    # Best leave this set to False. :)
    # ============================================================
    use_attn_weighted = False   # <-- set False for mean pooling

    cls_tok = tokens[:, 0, :]
    patch_toks = tokens[:, 1:, :]

    norms = patch_toks.norm(dim=-1)

    # abs OR adaptive thresholding (adaptive if any ':' in string)
    reg_mask, thr_used = compute_reg_mask_from_norms(norms, reg_threshold)
    patch_mask = ~reg_mask

    cls_hidden = ln_post(cls_tok)
    cls_embed = cls_hidden @ proj

    # ============================================================
    # Attention-weighted patch pooling (CLS query weights)
    # ============================================================
    if use_attn_weighted and (cls_attn_weights is not None):
        if cls_attn_weights.dim() == 3:
            w = cls_attn_weights.mean(dim=1)
        else:
            w = cls_attn_weights

        w = w[:, 1:]  # patch tokens only
        w = w.masked_fill(reg_mask, 0.0)

        wsum = w.sum(dim=1, keepdim=True)
        need_fallback = (wsum.squeeze(1) <= 0)

        w = w / (wsum + 1e-12)

        if need_fallback.any():
            w_fb = patch_mask.float()
            w_fb_sum = w_fb.sum(dim=1, keepdim=True)
            w_fb = w_fb / (w_fb_sum + 1e-12)
            w = torch.where(need_fallback[:, None], w_fb, w)

        patch_hidden = (w.unsqueeze(-1) * patch_toks).sum(dim=1, keepdim=False)
        patch_hidden = ln_post(patch_hidden)
        patch_embed = patch_hidden @ proj

        out = {"cls": cls_embed, "patch": patch_embed}

        # optionally also return REG mean-pool (not attn-weighted)
        if return_reg:
            reg_hidden_list = []
            for b in range(tokens.shape[0]):
                patches_b = patch_toks[b]
                mask_r = reg_mask[b]
                sel_r = patches_b[mask_r] if mask_r.any() else patches_b
                reg_hidden_b = sel_r.mean(dim=0, keepdim=True)
                reg_hidden_b = ln_post(reg_hidden_b)
                reg_hidden_list.append(reg_hidden_b)
            reg_hidden = torch.cat(reg_hidden_list, dim=0)
            out["reg"] = reg_hidden @ proj

        # optional stats
        if return_reg_stats:
            out["reg_count"] = reg_mask.sum(dim=1)    # [B]
            out["thr_used"]  = thr_used.squeeze(1)    # [B]

        return out

    # Default: mean pooling over non-register patches
    patch_hidden_list = []
    reg_hidden_list = [] if return_reg else None

    for b in range(tokens.shape[0]):
        patches_b = patch_toks[b]
        mask_p = patch_mask[b]
        sel_p = patches_b[mask_p] if mask_p.any() else patches_b

        patch_hidden_b = sel_p.mean(dim=0, keepdim=True)
        patch_hidden_b = ln_post(patch_hidden_b)
        patch_hidden_list.append(patch_hidden_b)

        # REG mean pooling
        if return_reg:
            mask_r = reg_mask[b]
            sel_r = patches_b[mask_r] if mask_r.any() else patches_b
            reg_hidden_b = sel_r.mean(dim=0, keepdim=True)
            reg_hidden_b = ln_post(reg_hidden_b)
            reg_hidden_list.append(reg_hidden_b)

    patch_hidden = torch.cat(patch_hidden_list, dim=0)
    patch_embed = patch_hidden @ proj

    out = {"cls": cls_embed, "patch": patch_embed}
    if return_reg:
        reg_hidden = torch.cat(reg_hidden_list, dim=0)
        out["reg"] = reg_hidden @ proj

    # optional stats
    if return_reg_stats:
        out["reg_count"] = reg_mask.sum(dim=1)    # [B]
        out["thr_used"]  = thr_used.squeeze(1)    # [B]

    return out


def _infer_visual_num_layers(model) -> int:
    """
    Infer number of visual transformer blocks from the loaded CLIP-like model.

    Tries common structures:
      - model.visual.transformer.resblocks   (OpenAI CLIP ViT)
      - model.visual.resblocks               (some forks)
    """
    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("Cannot infer visual depth: model has no .visual")

    # OpenAI CLIP-style ViT
    tr = getattr(visual, "transformer", None)
    if tr is not None and hasattr(tr, "resblocks"):
        return len(tr.resblocks)

    # Some forks expose resblocks directly
    if hasattr(visual, "resblocks"):
        return len(visual.resblocks)

    raise ValueError(
        "Cannot infer visual depth: expected model.visual.transformer.resblocks or model.visual.resblocks"
    )


def encode_image_tokens_after_block(
    visual: nn.Module,
    x: torch.Tensor,
    block_idx: int,
    detach: bool,
) -> torch.Tensor:

    # MODIFIED: strict bounds check (prevents silent last-block fallback)
    n_blocks = len(visual.transformer.resblocks)
    if not (0 <= int(block_idx) < n_blocks):
        raise ValueError(
            f"encode_image_tokens_after_block: block_idx={block_idx} out of range "
            f"(n_blocks={n_blocks}; valid=[0..{n_blocks-1}])."
        )

    x = x.to(dtype=visual.conv1.weight.dtype)

    x = visual.conv1(x)
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1).permute(0, 2, 1)

    class_emb = visual.class_embedding.to(x.dtype)
    cls_tokens = class_emb + torch.zeros(
        x.shape[0], 1, x.shape[-1],
        dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_tokens, x], dim=1)

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # [seq, B, d]

    use_attn_weighted = False

    for i, block in enumerate(visual.transformer.resblocks):
        # (unchanged body...)
        x = block(x)
        if i == block_idx:
            t_post = x.permute(1, 0, 2)  # [B, seq, d]
            return t_post.detach() if detach else t_post

    # Should be unreachable due to bounds check.
    t_post = x.permute(1, 0, 2)
    return t_post.detach() if detach else t_post


def compute_cls_patch_embeddings(
    images: torch.Tensor,
    visual: nn.Module,
    reg_threshold: Any,
    teacher_layer: int,
    detach: bool,
    return_reg: bool = False,
    return_reg_stats: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      - {"cls": [B, embed_dim], "patch": [B, embed_dim]}
      - plus "reg": [B, embed_dim] if return_reg=True
      - plus "reg_count": [B] and "thr_used": [B] if return_reg_stats=True
    (computed from tokens after that block).
    """
    # ============================================================
    # Attention Uber Alles!
    # Terrible idea, with catastrophic but interesting results.
    # Best leave this set to False. :)
    # ============================================================
    use_attn_weighted = False  # <-- set False to mean-pool

    if use_attn_weighted:
        tokens, cls_attn_w = encode_image_tokens_after_block(
            visual=visual,
            x=images,
            block_idx=int(teacher_layer),
            detach=bool(detach),
        )
        reps = build_global_embeddings_from_tokens(
            tokens,
            ln_post=visual.ln_post,
            proj=visual.proj,
            reg_threshold=reg_threshold,
            cls_attn_weights=cls_attn_w,
            return_reg=bool(return_reg),
            return_reg_stats=bool(return_reg_stats),
        )
        return reps

    tokens = encode_image_tokens_after_block(
        visual=visual,
        x=images,
        block_idx=int(teacher_layer),
        detach=bool(detach),
    )
    reps = build_global_embeddings_from_tokens(
        tokens,
        ln_post=visual.ln_post,
        proj=visual.proj,
        reg_threshold=reg_threshold,
        cls_attn_weights=None,
        return_reg=bool(return_reg),
        return_reg_stats=bool(return_reg_stats),
    )
    return reps


# ============================================================
# REG delta + optional diagonal whitening (teacher-side)
# ============================================================
def reg_delta_whiten(
    reg_embed: torch.Tensor,
    reg_mean: torch.Tensor,
    reg_var: Optional[torch.Tensor],
    use_reg_whitening: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    reg_embed: [B, D]
    reg_mean : [D] or [1, D]
    reg_var  : [D] or [1, D] (diagonal variance), or None if whitening disabled
    Returns:
      r_center            if use_reg_whitening=False
      r_center / sqrt(var) if use_reg_whitening=True
    """
    if reg_mean.dim() == 1:
        reg_mean = reg_mean.view(1, -1)
    r_center = reg_embed - reg_mean

    if not use_reg_whitening:
        return r_center

    if reg_var is None:
        raise ValueError("use_reg_whitening=True but reg_var is None")

    if reg_var.dim() == 1:
        reg_var = reg_var.view(1, -1)

    denom = torch.sqrt(reg_var + eps)
    return r_center / denom

# =========================
# Build Regression Teacher
# =========================

def build_cls_patch_regression_teacher(
    model,
    visual,
    dataloader,
    device: str,
    max_samples: int,
    val_frac: float,
    cache_path: Optional[str],
    reg_threshold: Any,
    teacher_layer: int,
    teacher_seed: int,
    extra_image_loaders=None,
    force_rebuild: bool = False,
    cls_mix: float = 0.0,
    clipmodel_id: Optional[str] = None,
    is_reg_teacher: bool = False,
    use_reg_whitening: bool = False,
    cls_mix_use_reg: bool = False,  # if True, cls_mix blends toward REG (same block) instead of final encode_image
):
    model_dtype = _get_model_dtype(model)
    teacher_id = str(teacher_layer)
    cls_mix = float(max(0.0, min(1.0, cls_mix)))
    is_reg_teacher = bool(is_reg_teacher)
    use_reg_whitening = bool(use_reg_whitening)
    cls_mix_use_reg = bool(cls_mix_use_reg)

    # canonical threshold spec for cache key + printing
    rt_spec = normalize_reg_threshold_spec(reg_threshold)
    reg_threshold_spec = rt_spec["spec"]

    loaded_from_cache = False

    if cache_path is not None and os.path.exists(cache_path) and not force_rebuild:
        teacher = torch.load(cache_path, map_location=device)

        cached_mix = float(teacher.get("cls_mix", 0.0))
        cached_layer = int(teacher.get("layer", teacher_layer))
        cached_seed = int(teacher.get("teacher_seed", teacher_seed))
        cached_model_id = teacher.get("clipmodel_id", None)
        cached_is_reg_teacher = bool(teacher.get("is_reg_teacher", False))
        cached_use_reg_whitening = bool(teacher.get("use_reg_whitening", False))
        cached_cls_mix_use_reg = bool(teacher.get("cls_mix_use_reg", False))

        # threshold cache key
        cached_spec = teacher.get("reg_threshold_spec", None)
        if cached_spec is None:
            # backward compat: old caches stored float reg_threshold; interpret as absolute
            cached_thr = float(teacher.get("reg_threshold", 0.0))
            cached_spec = f"{cached_thr:.6f}".rstrip("0").rstrip(".")
            cached_spec = f"abs:{cached_spec}"

        mismatch = (
            (abs(cached_mix - cls_mix) > 1e-9)
            or (str(cached_spec) != str(reg_threshold_spec))
            or (cached_layer != int(teacher_layer))
            or (cached_seed != int(teacher_seed))
            or (cached_is_reg_teacher != bool(is_reg_teacher))
            or (cached_use_reg_whitening != bool(use_reg_whitening))
            or (cached_cls_mix_use_reg != bool(cls_mix_use_reg))
        )
        if (clipmodel_id is not None) and (cached_model_id is not None) and (str(clipmodel_id) != str(cached_model_id)):
            mismatch = True

        if not mismatch:
            loaded_from_cache = True
            _phase_banner(f"TEACHER {teacher_layer} | load cache", color=Fore.MAGENTA)
            _subphase(f"cache_path={cache_path}", color=Fore.MAGENTA)
            _subphase(
                f"layer={teacher_layer}  reg_threshold={reg_threshold_spec}  cls_mix={cls_mix}  cls_mix_use_reg={cls_mix_use_reg}  "
                f"seed={teacher_seed}  is_reg_teacher={is_reg_teacher}  use_reg_whitening={use_reg_whitening}",
                color=Fore.MAGENTA,
            )

            teacher["patch_mean"] = teacher["patch_mean"].to(device=device, dtype=model_dtype)
            teacher["cls_mean"]   = teacher["cls_mean"].to(device=device, dtype=model_dtype)
            teacher["W"]          = teacher["W"].to(device=device, dtype=model_dtype)

            if teacher.get("is_reg_teacher", False):
                teacher["reg_mean"] = teacher["reg_mean"].to(device=device, dtype=model_dtype)
                if teacher.get("use_reg_whitening", False):
                    teacher["reg_var"] = teacher["reg_var"].to(device=device, dtype=model_dtype)

            teacher["loaded_from_cache"] = True
            return teacher

        print(Fore.YELLOW + f"[Teacher {teacher_id}] Cache mismatch -> rebuilding." + Style.RESET_ALL)

    _phase_banner(f"TEACHER {teacher_layer} | build", color=Fore.MAGENTA)
    _subphase(f"cache_path={cache_path}", color=Fore.MAGENTA)
    _subphase(
        f"layer={teacher_layer}  reg_threshold={reg_threshold_spec}  cls_mix={cls_mix}  cls_mix_use_reg={cls_mix_use_reg}  "
        f"is_reg_teacher={is_reg_teacher}  use_reg_whitening={use_reg_whitening}",
        color=Fore.MAGENTA,
    )
    _subphase(f"max_samples={max_samples}  val_frac={val_frac}  seed={teacher_seed}", color=Fore.MAGENTA)

    model.eval()

    cls_list = []
    patch_list = []
    reg_list = [] if is_reg_teacher else None

    # collect per-sample REG token counts across dataset
    regcount_list = []

    def _collect_from_loader(loader, max_samples_remaining):
        if loader is None or max_samples_remaining <= 0:
            return max_samples_remaining

        for batch in tqdm(loader, desc=f"Collect CLS/PATCH ({teacher_layer})", ncols=100):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True, dtype=_get_image_dtype(model))

            reps = compute_cls_patch_embeddings(
                images=images,
                visual=visual,
                reg_threshold=reg_threshold,
                teacher_layer=teacher_layer,
                detach=True,
                return_reg=bool(is_reg_teacher or cls_mix_use_reg),
                return_reg_stats=True,
            )

            if "reg_count" in reps:
                regcount_list.append(reps["reg_count"].cpu())

            patch_emb = reps["patch"].float()
            cls_local = reps["cls"].float()

            # get REG for mixing if requested
            reg_for_mix = None
            if cls_mix_use_reg:
                reg_for_mix = reps["reg"].float()

            if is_reg_teacher:
                reg_emb = reps["reg"].float()
                reg_list.append(reg_emb.cpu())

            # mixing (normalize inputs, optional REG mix, renormalize output)
            if cls_mix > 0.0:
                if cls_mix_use_reg:
                    # Use REG as a residual signal, not a raw second CLS.
                    # Cheap anti-anisotropy trick: remove batch mean, then normalize.
                    r = reg_for_mix - reg_for_mix.mean(dim=0, keepdim=True)

                    cls_a = F.normalize(cls_local, dim=-1)
                    cls_b = F.normalize(r, dim=-1)
                    cls_target = F.normalize((1.0 - cls_mix) * cls_a + cls_mix * cls_b, dim=-1)
                else:
                    with autocast():
                        cls_final = model.encode_image(images).float()
                    cls_a = F.normalize(cls_local, dim=-1)
                    cls_b = F.normalize(cls_final, dim=-1)
                    cls_target = F.normalize((1.0 - cls_mix) * cls_a + cls_mix * cls_b, dim=-1)
            else:
                cls_target = cls_local

            cls_list.append(cls_target.cpu())
            patch_list.append(patch_emb.cpu())

            max_samples_remaining -= cls_target.shape[0]
            if max_samples_remaining <= 0:
                break

        return max_samples_remaining

    max_remaining = max_samples
    with torch.no_grad():
        max_remaining = _collect_from_loader(dataloader, max_remaining)
        if extra_image_loaders is not None:
            for extra_loader in extra_image_loaders:
                if max_remaining <= 0:
                    break
                max_remaining = _collect_from_loader(extra_loader, max_remaining)

    if len(cls_list) == 0:
        print(Fore.YELLOW + f"[Teacher {teacher_id}] No samples collected, skipping." + Style.RESET_ALL)
        return None

    C = torch.cat(cls_list, dim=0)
    P = torch.cat(patch_list, dim=0)
    R = torch.cat(reg_list, dim=0) if is_reg_teacher else None

    N = C.shape[0]
    print(Fore.CYAN + f"[Teacher {teacher_id}] Collected {N} CLS/PATCH pairs." + Style.RESET_ALL)
    if is_reg_teacher:
        print(Fore.CYAN + f"[Teacher {teacher_id}] Collected {N} REG embeddings (delta/whiten)." + Style.RESET_ALL)

    # print REG token stats + warning
    if regcount_list:
        rc = torch.cat(regcount_list, dim=0).to(dtype=torch.float32)
        rc_min = int(rc.min().item())
        rc_max = int(rc.max().item())
        rc_mean = float(rc.mean().item())
        print(
            Fore.CYAN
            + f"[Teacher {teacher_id}] REG token counts (per-image): min={rc_min} max={rc_max} mean={rc_mean:.3f}  (thr={reg_threshold_spec})"
            + Style.RESET_ALL
        )
        if (rc_mean < 1.0) or (rc_mean > 16.0):
            print(
                Fore.RED + Style.BRIGHT
                + f"[Teacher {teacher_id}] WARNING: mean REG count is {rc_mean:.3f} (expected ~[1..16]). "
                  "This likely means your thresholding is off (no registers or too many tokens flagged)."
                + Style.RESET_ALL
            )

    if N < 100:
        print(Fore.YELLOW + f"[Teacher {teacher_id}] Not enough samples (<100), skipping." + Style.RESET_ALL)
        return None

    g = torch.Generator()
    g.manual_seed(teacher_seed)
    perm = torch.randperm(N, generator=g)

    train_N = int(N * (1.0 - val_frac))
    train_idx = perm[:train_N]
    val_idx = perm[train_N:]

    C_train = C[train_idx]
    P_train = P[train_idx]
    C_val = C[val_idx]
    P_val = P[val_idx]

    if is_reg_teacher:
        R_train = R[train_idx]
        R_val = R[val_idx]

    C_mean = C_train.mean(dim=0, keepdim=True)
    P_mean = P_train.mean(dim=0, keepdim=True)

    C_center = C_train - C_mean
    P_center = P_train - P_mean

    if is_reg_teacher:
        R_mean = R_train.mean(dim=0, keepdim=True)
        R_center = R_train - R_mean

        R_var = None
        if use_reg_whitening:
            R_var = R_center.var(dim=0, unbiased=False, keepdim=True)

        R_in = reg_delta_whiten(
            reg_embed=R_train,
            reg_mean=R_mean,
            reg_var=R_var,
            use_reg_whitening=use_reg_whitening,
            eps=1e-4,
        )

        X_in = torch.cat([P_center, R_in], dim=1)

        print(
            Fore.CYAN
            + f"[Teacher {teacher_id}] Solving least-squares [PATCH, REG_delta{'_wh' if use_reg_whitening else ''}]→CLS on {train_N} samples..."
            + Style.RESET_ALL
        )
        lstsq_res = torch.linalg.lstsq(X_in, C_center)
        Xsol = lstsq_res.solution
        W = Xsol.T
    else:
        print(Fore.CYAN + f"[Teacher {teacher_id}] Solving least-squares P→C on {train_N} samples..." + Style.RESET_ALL)
        lstsq_res = torch.linalg.lstsq(P_center, C_center)
        Xsol = lstsq_res.solution
        W = Xsol.T

    print(Fore.CYAN + f"[Teacher {teacher_id}] Validation metrics:" + Style.RESET_ALL)
    with torch.no_grad():
        if is_reg_teacher:
            P_val_center = P_val - P_mean

            R_in_val = reg_delta_whiten(
                reg_embed=R_val,
                reg_mean=R_mean,
                reg_var=R_var,
                use_reg_whitening=use_reg_whitening,
                eps=1e-4,
            )

            X_val = torch.cat([P_val_center, R_in_val], dim=1)
            C_val_center_pred = X_val @ W.T
        else:
            P_val_center = P_val - P_mean
            C_val_center_pred = P_val_center @ W.T

        C_val_pred = C_val_center_pred + C_mean

        mse = torch.mean((C_val_pred - C_val) ** 2).item()
        C_val_norm = F.normalize(C_val, dim=-1)
        C_val_pred_norm = F.normalize(C_val_pred, dim=-1)
        cos_sim = (C_val_norm * C_val_pred_norm).sum(dim=-1).mean().item()

        print(Fore.CYAN + f"  MSE (val): {mse:.6f}" + Style.RESET_ALL)
        print(Fore.CYAN + f"  mean cos(C_true, C_pred) (val): {cos_sim:.4f}" + Style.RESET_ALL)

    teacher = {
        "patch_mean": P_mean.squeeze(0).to(device=device, dtype=model_dtype),
        "cls_mean": C_mean.squeeze(0).to(device=device, dtype=model_dtype),
        "W": W.to(device=device, dtype=model_dtype),
        "fit_mse_val": float(mse),
        "fit_cos_val": float(cos_sim),
        "n_pairs": int(N),
        "teacher_id": str(teacher_id),
        "layer": int(teacher_layer),

        # keep a stable spec string for cache matching
        "reg_threshold_spec": str(reg_threshold_spec),

        # backward-compat / informational only:
        # - if abs: store numeric threshold
        # - if adaptive: store NaN (because a single float is meaningless)
        "reg_threshold": float(rt_spec["value"]) if rt_spec["mode"] == "abs" else float("nan"),

        "cls_mix": float(cls_mix),
        "cls_mix_use_reg": bool(cls_mix_use_reg),
        "teacher_seed": int(teacher_seed),
        "clipmodel_id": (str(clipmodel_id) if clipmodel_id is not None else None),
        "is_reg_teacher": bool(is_reg_teacher),
        "use_reg_whitening": bool(use_reg_whitening),
        "loaded_from_cache": False,
    }

    if is_reg_teacher:
        teacher["reg_mean"] = R_mean.squeeze(0).to(device=device, dtype=model_dtype)
        if use_reg_whitening:
            teacher["reg_var"] = R_var.squeeze(0).to(device=device, dtype=model_dtype)

    if cache_path is not None:
        payload = {
            "patch_mean": teacher["patch_mean"].cpu(),
            "cls_mean": teacher["cls_mean"].cpu(),
            "W": teacher["W"].cpu(),
            "fit_mse_val": teacher["fit_mse_val"],
            "fit_cos_val": teacher["fit_cos_val"],
            "n_pairs": teacher["n_pairs"],
            "teacher_id": teacher["teacher_id"],
            "layer": teacher["layer"],

            # save the spec (primary) + legacy float (secondary)
            "reg_threshold_spec": teacher["reg_threshold_spec"],
            "reg_threshold": teacher["reg_threshold"],

            "cls_mix": teacher["cls_mix"],
            "cls_mix_use_reg": teacher["cls_mix_use_reg"],
            "teacher_seed": teacher["teacher_seed"],
            "clipmodel_id": teacher["clipmodel_id"],
            "is_reg_teacher": teacher["is_reg_teacher"],
            "use_reg_whitening": teacher["use_reg_whitening"],
        }
        if is_reg_teacher:
            payload["reg_mean"] = teacher["reg_mean"].cpu()
            if use_reg_whitening:
                payload["reg_var"] = teacher["reg_var"].cpu()
        torch.save(payload, cache_path)
        print(Fore.GREEN + f"[Teacher {teacher_id}] Saved cache -> {cache_path}" + Style.RESET_ALL)

    del cls_list, patch_list, C, P, C_train, P_train, C_val, P_val
    if is_reg_teacher:
        del reg_list, R, R_train, R_val
    return teacher



@torch.no_grad()
def evaluate_teacher_cosine(
    model,
    visual,
    dataloader,
    teacher: dict,
    device: str,
    reg_threshold: Any,
    teacher_layer: int,
    max_batches: int = 50,
    projector: Optional[nn.Module] = None,
    cls_mix: float = 0.0,
    cls_mix_use_reg: Optional[bool] = None,  # if None, read from teacher dict
) -> float:
    model_dtype = _get_model_dtype(model)

    patch_mean = teacher["patch_mean"].view(1, -1).to(device=device, dtype=model_dtype)
    cls_mean   = teacher["cls_mean"].view(1, -1).to(device=device, dtype=model_dtype)
    W          = teacher["W"].to(device=device, dtype=model_dtype)

    is_reg_teacher     = bool(teacher.get("is_reg_teacher", False))
    use_reg_whitening  = bool(teacher.get("use_reg_whitening", False))

    reg_mean = None
    reg_var  = None
    if is_reg_teacher:
        reg_mean = teacher["reg_mean"].view(1, -1).to(device=device, dtype=model_dtype)
        if use_reg_whitening:
            reg_var = teacher["reg_var"].view(1, -1).to(device=device, dtype=model_dtype)

    cls_mix = float(max(0.0, min(1.0, cls_mix)))

    if cls_mix_use_reg is None:
        cls_mix_use_reg = bool(teacher.get("cls_mix_use_reg", False))
    else:
        cls_mix_use_reg = bool(cls_mix_use_reg)

    cos_list = []
    model.eval()

    for b_idx, (images, texts) in enumerate(dataloader):
        if b_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True, dtype=_get_image_dtype(model))

        with autocast():
            reps = compute_cls_patch_embeddings(
                images=images,
                visual=visual,
                reg_threshold=reg_threshold,  # may be string
                teacher_layer=teacher_layer,
                detach=True,
                return_reg=bool(is_reg_teacher or cls_mix_use_reg),
            )
            cls_local = reps["cls"].float()
            patch_emb = reps["patch"].float()

            p_center = patch_emb - patch_mean

            if is_reg_teacher:
                reg_emb = reps["reg"].float()
                r_in = reg_delta_whiten(
                    reg_embed=reg_emb,
                    reg_mean=reg_mean,
                    reg_var=reg_var,
                    use_reg_whitening=use_reg_whitening,
                    eps=1e-4,
                )
                x_center = torch.cat([p_center, r_in], dim=1)
                c_center_hat = x_center @ W.T
            else:
                c_center_hat = p_center @ W.T

            c_hat = c_center_hat + cls_mean

            # mixed target cls
            if cls_mix > 0.0:
                if cls_mix_use_reg:
                    r = reps["reg"].float()
                    r = r - r.mean(dim=0, keepdim=True)
                    cls_a = F.normalize(cls_local, dim=-1)
                    cls_b = F.normalize(r, dim=-1)
                    cls_embed = F.normalize((1.0 - cls_mix) * cls_a + cls_mix * cls_b, dim=-1)
                else:
                    cls_final = model.encode_image(images).float()
                    cls_a = F.normalize(cls_local, dim=-1)
                    cls_b = F.normalize(cls_final, dim=-1)
                    cls_embed = F.normalize((1.0 - cls_mix) * cls_a + cls_mix * cls_b, dim=-1)
            else:
                cls_embed = cls_local

            if projector is not None and hasattr(projector, "project_all"):
                cls_all = projector.project_all(cls_embed.float())  # [K, B, d_out]
                hat_all = projector.project_all(c_hat.float())      # [K, B, d_out]
                cls_all = F.normalize(cls_all, dim=-1)
                hat_all = F.normalize(hat_all, dim=-1)
                cos_kb = (cls_all * hat_all).sum(dim=-1)            # [K, B]
                cos_batch = cos_kb.mean(dim=0)                      # [B]
            else:
                if projector is not None:
                    cls_use = projector(cls_embed.float())
                    hat_use = projector(c_hat.float())
                else:
                    cls_use = cls_embed.float()
                    hat_use = c_hat.float()

                cls_n = F.normalize(cls_use, dim=-1)
                hat_n = F.normalize(hat_use, dim=-1)
                cos_batch = (cls_n * hat_n).sum(dim=-1)

        cos_list.append(cos_batch.cpu())

    if not cos_list:
        return 0.0
    return float(torch.cat(cos_list, dim=0).mean().item())


def resolve_regression_teacher_specs(cfg: TrainConfig, model) -> Dict[int, Dict[str, Any]]:
    """
    Strict behavior:
      - if cfg.use_regression_teachers is False: return {}
      - if True: validate cfg.regression_teachers against the *actual model depth*
        inferred from len(model.visual.transformer.resblocks) (or compatible fallback).

    reg_threshold semantics:
      - numeric (int/float) OR numeric string with no ':' => absolute
      - any string containing ':' => adaptive median * factor (factor after ':')
    """
    if not getattr(cfg, "use_regression_teachers", False):
        return {}

    raw = dict(getattr(cfg, "regression_teachers", {}) or {})
    if len(raw) == 0:
        raise SystemExit("[Abort] cfg.use_regression_teachers=True but cfg.regression_teachers is empty.")

    # MODIFIED: infer depth from model
    n_layers = _infer_visual_num_layers(model)
    max_layer = n_layers - 1

    specs: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        try:
            layer = int(k)
        except Exception:
            raise SystemExit(f"[Abort] regression_teachers key {k!r} is not an int-like layer index.")
        if not isinstance(v, dict):
            raise SystemExit(f"[Abort] regression_teachers[{layer}] must be a dict, got {type(v)}.")
        specs[layer] = dict(v)

    norm: Dict[int, Dict[str, Any]] = {}
    for layer, spec in specs.items():
        if not (0 <= int(layer) <= max_layer):
            raise SystemExit(
                f"[Abort] regression teacher layer must be in [0..{max_layer}] for this model, got {layer}. "
                f"(model visual depth={n_layers})"
            )

        reg_threshold_raw = spec.get("reg_threshold", 70.0)
        try:
            _ = normalize_reg_threshold_spec(reg_threshold_raw)
        except Exception as e:
            raise SystemExit(f"[Abort] regression_teachers[{layer}].reg_threshold invalid: {reg_threshold_raw!r} ({e})")

        lam = float(spec.get("lam", 0.0))
        cls_mix = float(spec.get("cls_mix", 0.0))
        cls_mix_use_reg = bool(spec.get("cls_mix_use_reg", False))
        jl = _normalize_jl_spec(spec.get("jl", None))

        is_reg_teacher = bool(spec.get("is_reg_teacher", False))
        use_reg_whitening = bool(spec.get("use_reg_whitening", False))

        if lam < 0:
            raise SystemExit(f"[Abort] regression_teachers[{layer}].lam must be >= 0, got {lam}.")
        if jl["enabled"]:
            if int(jl["dim"]) <= 0:
                raise SystemExit(f"[Abort] regression_teachers[{layer}].jl.dim must be > 0.")
            if int(jl["num_proj"]) <= 0:
                raise SystemExit(f"[Abort] regression_teachers[{layer}].jl.num_proj must be > 0.")
            if int(jl["seed_stride"]) <= 0:
                raise SystemExit(f"[Abort] regression_teachers[{layer}].jl.seed_stride must be > 0.")
        if not (0.0 <= cls_mix <= 1.0):
            raise SystemExit(f"[Abort] regression_teachers[{layer}].cls_mix must be in [0,1], got {cls_mix}.")
        if use_reg_whitening and (not is_reg_teacher):
            raise SystemExit(
                f"[Abort] regression_teachers[{layer}].use_reg_whitening=True requires is_reg_teacher=True."
            )

        norm[int(layer)] = {
            "reg_threshold": reg_threshold_raw,
            "lam": lam,
            "cls_mix": cls_mix,
            "cls_mix_use_reg": cls_mix_use_reg,
            "jl": jl,
            "is_reg_teacher": is_reg_teacher,
            "use_reg_whitening": use_reg_whitening,
        }

    return {k: norm[k] for k in sorted(norm.keys())}


def make_teacher_cache_paths(cfg: TrainConfig, teacher_specs: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    fmt = getattr(cfg, "teacher_cache_name_fmt", "cls_patch_teacher_val_b{layer}.pt")
    out = {}
    for layer in teacher_specs.keys():
        name = fmt.format(layer=int(layer))
        out[int(layer)] = os.path.join(cfg.teacher_folder, name)
    return out

def make_jl_projector_for_teacher(
    embed_dim: int,
    spec: Dict[str, Any],
    device: str,
) -> Optional[nn.Module]:
    jl = spec.get("jl", None) or {}
    if not jl.get("enabled", False):
        return None

    proj = JLProjectorEnsemble(
        d_in=int(embed_dim),
        d_out=int(jl["dim"]),
        n_proj=int(jl["num_proj"]),
        seed_base=int(jl["seed"]),
        seed_stride=int(jl["seed_stride"]),
    ).to(device)
    return proj

def should_rebuild_teacher(
    epoch_idx: int,
    cfg: TrainConfig,
    last_teacher_update_epoch: int,
    current_teacher_cos: Optional[float],
    best_teacher_cos: Optional[float],
) -> Tuple[bool, str]:

    if cfg.teacher_retrain_every_epoch:
        return True, "every_epoch=True"

    if epoch_idx < cfg.teacher_early_retrain_epochs:
        return True, f"early_retrain<({cfg.teacher_early_retrain_epochs})"

    if current_teacher_cos is None:
        return False, "no_teacher_cos"

    if epoch_idx < cfg.teacher_min_epoch:
        return False, "before_teacher_min_epoch"

    if (epoch_idx - last_teacher_update_epoch) < cfg.teacher_update_cooldown:
        return False, "cooldown"

    if current_teacher_cos < cfg.teacher_abs_cos_min:
        return True, f"abs_cos<{cfg.teacher_abs_cos_min}"

    if best_teacher_cos is not None and current_teacher_cos < (best_teacher_cos - cfg.teacher_cos_drop):
        return True, f"drop>{cfg.teacher_cos_drop}"

    return False, "fresh"