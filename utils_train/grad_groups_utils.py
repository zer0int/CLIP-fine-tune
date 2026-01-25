from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

def _resolve_block_indices(idxs: Sequence[int], n_blocks: int) -> List[int]:
    """
    Convert a list of block indices into canonical [0..n_blocks-1] indices.
    Supports negative indexing: -1 => last block, -2 => second last, etc.
    Deduplicates while preserving order.
    """
    out: List[int] = []
    seen = set()
    for i in idxs:
        j = i if i >= 0 else (n_blocks + i)
        if j < 0 or j >= n_blocks:
            raise IndexError(f"Block index {i} resolves to {j}, but valid range is [0, {n_blocks-1}].")
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def get_visual_block_params(model: nn.Module, block_indices: Sequence[int]) -> List[nn.Parameter]:
    """
    Return parameters for the specified visual transformer resblocks.
    """
    blocks = model.visual.transformer.resblocks
    idxs = _resolve_block_indices(block_indices, n_blocks=len(blocks))
    params: List[nn.Parameter] = []
    for bi in idxs:
        params.extend(list(blocks[bi].parameters()))
    return params


def get_text_block_params(model: nn.Module, block_indices: Sequence[int]) -> List[nn.Parameter]:
    """
    Return parameters for the specified text transformer resblocks.
    """
    blocks = model.transformer.resblocks
    idxs = _resolve_block_indices(block_indices, n_blocks=len(blocks))
    params: List[nn.Parameter] = []
    for bi in idxs:
        params.extend(list(blocks[bi].parameters()))
    return params


def get_visual_from_params(model: nn.Module, start_block: int) -> List[nn.Parameter]:
    """
    Visual: resblocks[start_block:] plus post-block (ln_post, proj).
    Does NOT include pre-block (conv1, ln_pre, class/pos embedding) unless you add them manually.
    """
    blocks = model.visual.transformer.resblocks
    start = _resolve_block_indices([start_block], n_blocks=len(blocks))[0]
    params: List[nn.Parameter] = []
    for bi in range(start, len(blocks)):
        params.extend(list(blocks[bi].parameters()))
    params.extend(list(model.visual.ln_post.parameters()))
    # model.visual.proj is usually nn.Parameter
    if hasattr(model.visual, "proj") and isinstance(model.visual.proj, torch.nn.Parameter):
        params.append(model.visual.proj)
    elif hasattr(model.visual, "proj") and isinstance(model.visual.proj, torch.Tensor) and model.visual.proj.requires_grad is not None:
        # rare variant; keep it robust
        params.append(model.visual.proj)  # type: ignore[arg-type]
    return params


def get_text_from_params(model: nn.Module, start_block: int) -> List[nn.Parameter]:
    """
    Text: resblocks[start_block:] plus post-block (ln_final, text_projection).
    Does NOT include token_embedding/positional_embedding unless you add them manually.
    """
    blocks = model.transformer.resblocks
    start = _resolve_block_indices([start_block], n_blocks=len(blocks))[0]
    params: List[nn.Parameter] = []
    for bi in range(start, len(blocks)):
        params.extend(list(blocks[bi].parameters()))
    params.extend(list(model.ln_final.parameters()))
    if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
        params.append(model.text_projection)
    elif hasattr(model, "text_projection") and isinstance(model.text_projection, torch.Tensor):
        params.append(model.text_projection)  # type: ignore[arg-type]
    return params


def _dedupe_params(params: Iterable[nn.Parameter]) -> List[nn.Parameter]:
    out: List[nn.Parameter] = []
    seen = set()
    for p in params:
        if p is None:
            continue
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            out.append(p)
    return out


def _param_ids_from_groups(param_groups: List[Dict[str, Any]]) -> set:
    ids = set()
    for g in param_groups:
        for p in g.get("params", []):
            ids.add(id(p))
    return ids


def apply_param_groups_trainability(model: nn.Module, param_groups: List[Dict[str, Any]], *, verbose: bool = False) -> None:
    """
    Freeze everything, then unfreeze exactly the params present in param_groups.
    This guarantees optimizer groups and requires_grad stay consistent.

    Also sanity-checks:
      - empty param_groups
      - params not belonging to model
      - duplicates across groups (allowed, but warned)
    """
    if not param_groups:
        raise ValueError("param_groups is empty. Nothing to train.")

    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False

    group_param_ids = _param_ids_from_groups(param_groups)

    # Ensure every group param is actually a model param (by id)
    model_param_ids = {id(p) for p in model.parameters()}
    missing = [pid for pid in group_param_ids if pid not in model_param_ids]
    if missing:
        raise ValueError(
            f"param_groups contains {len(missing)} params not found in model.parameters(). "
            "This often happens if you passed tensors that are not nn.Parameter attributes."
        )

    # unfreeze
    for p in model.parameters():
        if id(p) in group_param_ids:
            p.requires_grad = True

    if verbose:
        n_train = sum(int(p.requires_grad) for p in model.parameters())
        n_all = sum(1 for _ in model.parameters())
        print(f"[GradPreset] trainable_params={n_train}/{n_all}")


def build_param_groups_automatic(model: nn.Module, cfg: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Automatic presets.
      - Does NOT set per-group 'lr'. Uniform LR comes from optimizer defaults
        (cfg.optimizer_kwargs['lr']).
      - Per-group LR should only be used in manual presets when explicitly set.
    """

    def _one_group(params: List[nn.Parameter]) -> List[Dict[str, Any]]:
        params = _dedupe_params([p for p in params if p is not None])
        params = [p for p in params if isinstance(p, torch.nn.Parameter)]
        if len(params) == 0:
            raise ValueError("Automatic preset produced 0 parameters.")
        return [{"params": params}]

    # 1) full all
    if getattr(cfg, "grad_full_text_vit", False):
        name = "grad_full_text_vit"
        return name, _one_group(list(model.parameters()))

    # 2) vit full
    if getattr(cfg, "grad_vit_full", False):
        name = "grad_vit_full"
        return name, _one_group(list(model.visual.parameters()))

    # 3) text full
    if getattr(cfg, "grad_text_full", False):
        name = "grad_text_full"
        params: List[nn.Parameter] = []
        params.extend(list(model.token_embedding.parameters()))
        if hasattr(model, "positional_embedding") and isinstance(model.positional_embedding, torch.nn.Parameter):
            params.append(model.positional_embedding)
        params.extend(list(model.transformer.parameters()))
        params.extend(list(model.ln_final.parameters()))
        if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
            params.append(model.text_projection)
        return name, _one_group(params)

    # 4) "from" mode (additive, non-exclusive)
    vit_from = getattr(cfg, "grad_vit_from", None)
    text_from = getattr(cfg, "grad_text_from", None)

    from_params: List[nn.Parameter] = []
    from_name_bits: List[str] = []

    if vit_from is not None:
        v = int(vit_from)
        from_name_bits.append(f"vit_from_{v}")
        if v == 0:
            from_params.extend(list(model.visual.parameters()))
        else:
            from_params.extend(get_visual_from_params(model, v))

    if text_from is not None:
        t = int(text_from)
        from_name_bits.append(f"text_from_{t}")
        if t == 0:
            from_params.extend(list(model.token_embedding.parameters()))
            if hasattr(model, "positional_embedding") and isinstance(model.positional_embedding, torch.nn.Parameter):
                from_params.append(model.positional_embedding)
            from_params.extend(list(model.transformer.parameters()))
            from_params.extend(list(model.ln_final.parameters()))
            if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
                from_params.append(model.text_projection)
        else:
            from_params.extend(get_text_from_params(model, t))


    if len(from_name_bits) > 0:
        name = "grad_" + "_".join(from_name_bits)
        return name, _one_group(from_params)

    # 5) nothing selected => error
    raise ValueError(
        "No grad preset selected. Set one of: "
        "grad_full_text_vit / grad_vit_full / grad_text_full / grad_vit_from / grad_text_from "
        "(or grad_set_manual=True)."
    )