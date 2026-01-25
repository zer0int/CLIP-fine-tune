from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

import torch
from torch import nn

OptimGroupMode = Literal["flat", "component_decay", "blockwise", "manual"]

@dataclass(frozen=True)
class GradPolicyReport:
    mode: str
    vit_from: Optional[int] = None
    text_from: Optional[int] = None
    num_trainable: int = 0
    num_total: int = 0
    trainable_names_preview: Tuple[str, ...] = ()


def _set_vit_full_like(model: nn.Module) -> None:
    # Treat as "grad_vit_full" (full visual encoder), excluding logit_scale.
    for name, p in model.named_parameters():
        if name == "logit_scale":
            continue
        if _is_vit_param_name(name):
            p.requires_grad = True


def _set_text_full_like(model: nn.Module) -> None:
    # Treat as "grad_text_full" (full text encoder), excluding logit_scale.
    for name, p in model.named_parameters():
        if name == "logit_scale":
            continue
        if _is_text_param_name(name):
            p.requires_grad = True


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

def _force_no_logit_scale(model: nn.Module) -> None:
    # CLIP has model.logit_scale (Parameter). Presets never train it.
    if hasattr(model, "logit_scale"):
        ls = getattr(model, "logit_scale")
        if isinstance(ls, torch.nn.Parameter):
            ls.requires_grad = False


def _named_params_in_order(model: nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    out = []
    for n, p in model.named_parameters():
        if isinstance(p, torch.nn.Parameter):
            out.append((n, p))
    return out


def _is_text_param_name(name: str) -> bool:
    return (
        name.startswith("transformer.")
        or name.startswith("token_embedding")
        or name.startswith("positional_embedding")
        or name.startswith("ln_final")
        or name.startswith("text_projection")
    )

def _is_vit_param_name(name: str) -> bool:
    return name.startswith("visual.")

def _is_no_decay_name(name: str) -> bool:
    # Common, sane default:
    # - no decay for biases + LayerNorm params + positional/class embeddings
    if name.endswith(".bias"):
        return True
    if ".ln_" in name or name.startswith("ln_") or name.startswith("ln_final") or name.startswith("visual.ln_"):
        return True
    if "positional_embedding" in name or "class_embedding" in name:
        return True
    return False

def _set_vit_from(model: nn.Module, vit_from: int) -> None:
    # Train: visual.transformer.resblocks[vit_from:] + visual.ln_post + visual.proj
    blocks = model.visual.transformer.resblocks
    n_blocks = len(blocks)
    vit_from = max(0, min(int(vit_from), n_blocks))

    for i in range(vit_from, n_blocks):
        for p in blocks[i].parameters():
            p.requires_grad = True

    for n, p in model.named_parameters():
        if n.startswith("visual.ln_post") or n.startswith("visual.proj"):
            p.requires_grad = True

def _set_text_from(model: nn.Module, text_from: int) -> None:
    # Train: transformer.resblocks[text_from:] + ln_final + text_projection
    blocks = model.transformer.resblocks
    n_blocks = len(blocks)
    text_from = max(0, min(int(text_from), n_blocks))

    for i in range(text_from, n_blocks):
        for p in blocks[i].parameters():
            p.requires_grad = True

    for n, p in model.named_parameters():
        if n.startswith("ln_final") or n.startswith("text_projection"):
            p.requires_grad = True


def print_trainable_summary(
    model: nn.Module,
    *,
    max_other_show: int = 40,
) -> None:
    # Pretty HOT/COLD summary similar to your existing one, but generic.
    hot_names = []
    cold_names = []
    for name, p in model.named_parameters():
        (hot_names if p.requires_grad else cold_names).append(name)

    # block families
    def _summarize_block_family(prefix: str, title: str) -> None:
        hot_layers, cold_layers = set(), set()
        for name, p in model.named_parameters():
            if not name.startswith(prefix + "."):
                continue
            rest = name[len(prefix) + 1 :]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                continue
            try:
                layer_i = int(parts[0])
            except Exception:
                continue
            (hot_layers if p.requires_grad else cold_layers).add(layer_i)

        def _fmt_layers(xs: set) -> str:
            if not xs:
                return "[]"
            xs2 = sorted(xs)
            ranges = []
            a = b = xs2[0]
            for v in xs2[1:]:
                if v == b + 1:
                    b = v
                else:
                    ranges.append((a, b))
                    a = b = v
            ranges.append((a, b))
            out = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
            return "[" + ",".join(out) + "]"

        print(f"{title}: HOT {prefix}.{_fmt_layers(hot_layers)} | COLD {prefix}.{_fmt_layers(cold_layers)}")

    print("-----------Params-----------")
    _summarize_block_family("transformer.resblocks", "Text Transformer Blocks")
    _summarize_block_family("visual.transformer.resblocks", "Vision Transformer Blocks")

    print("Other Parameters (HOT first):")
    # Other params only
    other_hot = []
    other_cold = []
    for name, p in model.named_parameters():
        if name.startswith("transformer.resblocks.") or name.startswith("visual.transformer.resblocks."):
            continue
        (other_hot if p.requires_grad else other_cold).append(name)

    for n in other_hot[:max_other_show]:
        print(f"HOT  {n}")
    if len(other_hot) > max_other_show:
        print(f"... (+{len(other_hot) - max_other_show} more HOT)")

    for n in other_cold[:max_other_show]:
        print(f"COLD {n}")
    if len(other_cold) > max_other_show:
        print(f"... (+{len(other_cold) - max_other_show} more COLD)")
    print("----------------------------")


def apply_grad_policy(
    model: nn.Module,
    cfg: Any,
    *,
    set_grad_model_params: Optional[Callable[[nn.Module, Any], None]] = None,
    verbose: bool = False,
) -> GradPolicyReport:
    """
    Priority rules (exactly):
    1) grad_full_text_vit -> all params require grad (except logit_scale)
    2) grad_vit_full      -> ViT only
    3) grad_text_full     -> Text only
       If both vit_full and text_full -> behave like full_text_vit
    4) grad_manual_layer  -> use grad_vit_from / grad_text_from (each may be None)
    5) grad_set_manual    -> call user's set_grad_model_params(model, cfg)
    """

    # always start from a known state for presets
    _freeze_all(model)

    # Resolve priority
    full = bool(getattr(cfg, "grad_full_text_vit", False))
    vit_full = bool(getattr(cfg, "grad_vit_full", False))
    text_full = bool(getattr(cfg, "grad_text_full", False))
    manual_layer = bool(getattr(cfg, "grad_manual_layer", False))
    set_manual = bool(getattr(cfg, "grad_set_manual", False))

    mode = "none"
    vit_from = None
    text_from = None

    if full or (vit_full and text_full):
        mode = "full_text_vit"
        for name, p in model.named_parameters():
            # everything except logit_scale
            if name == "logit_scale":
                continue
            p.requires_grad = True
        _force_no_logit_scale(model)

    elif vit_full:
        mode = "vit_full"
        for name, p in model.named_parameters():
            if name == "logit_scale":
                continue
            if _is_vit_param_name(name):
                p.requires_grad = True
        _force_no_logit_scale(model)

    elif text_full:
        mode = "text_full"
        for name, p in model.named_parameters():
            if name == "logit_scale":
                continue
            if _is_text_param_name(name):
                p.requires_grad = True
        _force_no_logit_scale(model)

    elif manual_layer:
        mode = "manual_layer"
        vit_from = getattr(cfg, "grad_vit_from", None)
        text_from = getattr(cfg, "grad_text_from", None)

        if vit_from is not None:
            if int(vit_from) == 0:
                _set_vit_full_like(model)
            else:
                _set_vit_from(model, int(vit_from))

        if text_from is not None:
            if int(text_from) == 0:
                _set_text_full_like(model)
            else:
                _set_text_from(model, int(text_from))


        _force_no_logit_scale(model)

    elif set_manual:
        mode = "set_manual"
        if set_grad_model_params is None:
            raise RuntimeError("cfg.grad_set_manual=True but no set_grad_model_params callback was provided.")
        # IMPORTANT: In manual mode we do NOT auto-freeze/unfreeze further beyond your callback,
        # but we already froze everything above; so your function should set grads explicitly.
        set_grad_model_params(model, cfg)

    else:
        mode = "none"
        _force_no_logit_scale(model)

    named = _named_params_in_order(model)
    trainable = [n for (n, p) in named if p.requires_grad]
    rep = GradPolicyReport(
        mode=mode,
        vit_from=vit_from,
        text_from=text_from,
        num_trainable=len(trainable),
        num_total=len(named),
        trainable_names_preview=tuple(trainable[:25]),
    )

    if verbose:
        print(f"[GradPolicy] mode={rep.mode} vit_from={rep.vit_from} text_from={rep.text_from}")
        print(f"[GradPolicy] trainable={rep.num_trainable}/{rep.num_total}")
        print_trainable_summary(model)

    return rep


# ----------------------------
# optimizer param_groups presets
# ----------------------------
def build_optimizer_param_groups(
    model: nn.Module,
    cfg: Any,
    *,
    mode: OptimGroupMode = "component_decay",
    set_manual_param_groups: Optional[Callable[[nn.Module, Any], List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns PyTorch optimizer param_groups in a stable order.

    Preset modes:
      - flat:            2 groups (decay / no_decay) across all trainable params
      - component_decay: per-component split (vit/text/other) x (decay/no_decay)
      - blockwise:       per-trainable-resblock groups + (vit tail) + (text tail)
      - manual:          call user's set_manual_param_groups(model, cfg)

    Note: For preset modes we *exclude* logit_scale by name.
    """

    mode_cfg = getattr(cfg, "optim_group_mode", mode)
    mode = mode_cfg  # allow cfg override

    lr_base = float(getattr(cfg, "learning_rate", 1e-6))
    lr_mult_vit = float(getattr(cfg, "optim_lr_mult_vit", 1.0))
    lr_mult_text = float(getattr(cfg, "optim_lr_mult_text", 1.0))
    lr_mult_other = float(getattr(cfg, "optim_lr_mult_other", 1.0))

    named = _named_params_in_order(model)

    def _iter_trainable_named():
        for name, p in named:
            if name == "logit_scale":
                continue
            if p.requires_grad:
                yield name, p

    def _push(groups: List[Dict[str, Any]], params: List[torch.nn.Parameter], lr: float, wd: Optional[float]) -> None:
        if not params:
            return
        g: Dict[str, Any] = {"params": params, "lr": lr}
        if wd is not None:
            g["weight_decay"] = float(wd)
        groups.append(g)

    if mode == "manual":
        if set_manual_param_groups is None:
            raise RuntimeError("optim_group_mode='manual' but no set_manual_param_groups callback was provided.")
        return set_manual_param_groups(model, cfg)

    # If you want per-group weight_decay, keep optimizer(weight_decay=0) and set here.
    # But you currently pass weight_decay=1e-3 to AdaBelief; leaving that is OK.
    # We'll only set weight_decay here if cfg.optim_set_wd_in_groups=True.
    set_wd_in_groups = bool(getattr(cfg, "optim_set_wd_in_groups", False))
    wd = float(getattr(cfg, "optim_weight_decay", 1e-3))
    wd_decay = wd if set_wd_in_groups else None
    wd_nodecay = 0.0 if set_wd_in_groups else None

    if mode == "flat":
        decay_params: List[torch.nn.Parameter] = []
        nodecay_params: List[torch.nn.Parameter] = []
        for name, p in _iter_trainable_named():
            (nodecay_params if _is_no_decay_name(name) else decay_params).append(p)

        groups: List[Dict[str, Any]] = []
        _push(groups, decay_params, lr_base, wd_decay)
        _push(groups, nodecay_params, lr_base, wd_nodecay)
        return groups

    if mode == "component_decay":
        vit_decay: List[torch.nn.Parameter] = []
        vit_nodecay: List[torch.nn.Parameter] = []
        text_decay: List[torch.nn.Parameter] = []
        text_nodecay: List[torch.nn.Parameter] = []
        other_decay: List[torch.nn.Parameter] = []
        other_nodecay: List[torch.nn.Parameter] = []

        for name, p in _iter_trainable_named():
            no_decay = _is_no_decay_name(name)
            if _is_vit_param_name(name):
                (vit_nodecay if no_decay else vit_decay).append(p)
            elif _is_text_param_name(name):
                (text_nodecay if no_decay else text_decay).append(p)
            else:
                (other_nodecay if no_decay else other_decay).append(p)

        groups = []
        _push(groups, vit_decay, lr_base * lr_mult_vit, wd_decay)
        _push(groups, vit_nodecay, lr_base * lr_mult_vit, wd_nodecay)

        _push(groups, text_decay, lr_base * lr_mult_text, wd_decay)
        _push(groups, text_nodecay, lr_base * lr_mult_text, wd_nodecay)

        _push(groups, other_decay, lr_base * lr_mult_other, wd_decay)
        _push(groups, other_nodecay, lr_base * lr_mult_other, wd_nodecay)
        return groups

    if mode == "blockwise":
        groups: List[Dict[str, Any]] = []

        # --- ViT blocks
        if hasattr(model, "visual") and hasattr(model.visual, "transformer"):
            vblocks = model.visual.transformer.resblocks
            for i, blk in enumerate(vblocks):
                params = [p for p in blk.parameters() if p.requires_grad]
                if params:
                    _push(groups, params, lr_base * lr_mult_vit, None)

        # --- ViT tail (ln_post / proj)
        vit_tail = []
        for name, p in _iter_trainable_named():
            if name.startswith("visual.ln_post") or name.startswith("visual.proj"):
                vit_tail.append(p)
        _push(groups, vit_tail, lr_base * lr_mult_vit, None)

        # --- Text blocks
        if hasattr(model, "transformer"):
            tblocks = model.transformer.resblocks
            for i, blk in enumerate(tblocks):
                params = [p for p in blk.parameters() if p.requires_grad]
                if params:
                    _push(groups, params, lr_base * lr_mult_text, None)

        # --- Text tail (ln_final / text_projection)
        text_tail = []
        for name, p in _iter_trainable_named():
            if name.startswith("ln_final") or name.startswith("text_projection"):
                text_tail.append(p)
        _push(groups, text_tail, lr_base * lr_mult_text, None)

        # --- Remaining trainables
        used = set(id(p) for g in groups for p in g["params"])
        other = [p for _, p in _iter_trainable_named() if id(p) not in used]
        _push(groups, other, lr_base * lr_mult_other, None)

        return groups

    raise ValueError(f"Unknown optim_group_mode: {mode}")