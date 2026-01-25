from __future__ import annotations

import os
import torch
from typing import Dict
from torch import nn
import torch.nn.functional as F
from typing import Optional

def _gather_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().to("cpu")
    return out

# For EMA model:
def _merge_state_dict_cpu(
    model: nn.Module,
    override_state_dict_cpu: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    base = _gather_state_dict_cpu(model)  # includes frozen keys not in EMA
    for k, v in override_state_dict_cpu.items():
        base[k] = v.detach().to("cpu")
    return base

# ============================================================
# FP32, QKV, GmP -> back to OpenAI/CLIP weight conversion
# ============================================================
def convert_weights(model: nn.Module):
    """
    Convert applicable model parameters to fp16
    """
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def convert_back_to_original(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert back from Geometric Parametrization .theta + .r -> .weight
    """
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".theta"):
            base_key = key.replace(".theta", "")
            r_key = base_key + ".r"
            if r_key not in state_dict:
                raise KeyError(f"Missing paired key {r_key} for {key}")

            theta = value
            r = state_dict[r_key]
            new_weight = r * F.normalize(theta, p=2, dim=1)
            new_state_dict[base_key + ".weight"] = new_weight

        elif key.endswith(".r") or key.endswith(".theta"):
            continue
        else:
            new_state_dict[key] = value
    return new_state_dict


def convert_state_dict_qkv_to_inproj(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert Attention QKV back to concatenated form for OpenAI CLIP.

    Supports two common layouts:
      1) Linear modules:    ...attn.q_proj.weight / k_proj / v_proj (+ optional .bias)
      2) MHA split weights: ...attn.q_proj_weight / k_proj_weight / v_proj_weight (+ optional *_bias)

    If biases are missing (partially or fully), we synthesize in_proj_bias as zeros
    """
    out: Dict[str, torch.Tensor] = dict(state_dict)  # shallow copy; do NOT mutate input
    attn_bases = set()

    # Detect attention bases that have q_proj in split form
    for k in list(out.keys()):
        if k.endswith(".attn.q_proj.weight"):
            attn_bases.add(k[:-len(".q_proj.weight")])
        elif k.endswith(".attn.q_proj_weight"):
            attn_bases.add(k[:-len(".q_proj_weight")])

    for base in sorted(attn_bases):
        # Prefer Linear-style keys if present, else MHA split keys
        q_w_key = base + ".q_proj.weight" if (base + ".q_proj.weight") in out else (base + ".q_proj_weight")
        k_w_key = base + ".k_proj.weight" if (base + ".k_proj.weight") in out else (base + ".k_proj_weight")
        v_w_key = base + ".v_proj.weight" if (base + ".v_proj.weight") in out else (base + ".v_proj_weight")

        if q_w_key not in out or k_w_key not in out or v_w_key not in out:
            # If we detected q_proj but can't find a full set, that's big trouble.
            missing = [kk for kk in [q_w_key, k_w_key, v_w_key] if kk not in out]
            raise KeyError(f"[QKV->in_proj] Missing required keys for base={base}: {missing}")

        in_w_key = base + ".in_proj_weight"
        if in_w_key not in out:
            qW = out[q_w_key]
            kW = out[k_w_key]
            vW = out[v_w_key]
            out[in_w_key] = torch.cat([qW, kW, vW], dim=0)

        # Bias handling: create in_proj_bias if missing
        in_b_key = base + ".in_proj_bias"
        if in_b_key not in out:
            # Linear-style bias keys, then MHA-style bias keys
            q_b_key = base + ".q_proj.bias" if (base + ".q_proj.bias") in out else (base + ".q_proj_bias")
            k_b_key = base + ".k_proj.bias" if (base + ".k_proj.bias") in out else (base + ".k_proj_bias")
            v_b_key = base + ".v_proj.bias" if (base + ".v_proj.bias") in out else (base + ".v_proj_bias")

            # Reference tensor for dtype/device + embed_dim
            refW = out[q_w_key]
            embed_dim = int(refW.shape[0])  # q weight is [embed_dim, embed_dim] for Linear-style

            def _get_bias_or_zeros(bkey: str) -> torch.Tensor:
                if bkey in out:
                    return out[bkey]
                return refW.new_zeros((embed_dim,))

            qb = _get_bias_or_zeros(q_b_key)
            kb = _get_bias_or_zeros(k_b_key)
            vb = _get_bias_or_zeros(v_b_key)
            out[in_b_key] = torch.cat([qb, kb, vb], dim=0)

        # Remove split keys so strict=True load into OpenAI CLIP works
        keys_to_drop = [
            q_w_key, k_w_key, v_w_key,
            # Drop both possible bias naming schemes if present
            base + ".q_proj.bias", base + ".k_proj.bias", base + ".v_proj.bias",
            base + ".q_proj_bias", base + ".k_proj_bias", base + ".v_proj_bias",
        ]
        for kk in keys_to_drop:
            if kk in out:
                del out[kk]

    return out


class GmPconverter:
    """
    Orchestrator
    """
    @staticmethod
    def build_config_from_model(modelft):
        return {
            'embed_dim': modelft.text_projection.shape[1],
            'image_resolution': modelft.visual.input_resolution,
            'vision_layers': modelft.visual.transformer.layers,
            'vision_width': modelft.visual.conv1.out_channels,
            'vision_patch_size': modelft.visual.conv1.kernel_size[0],
            'context_length': modelft.context_length,
            'vocab_size': modelft.vocab_size,
            'transformer_width': modelft.transformer.width,
            'transformer_heads': modelft.transformer.resblocks[0].attn.num_heads,
            'transformer_layers': modelft.transformer.layers
        }

    @staticmethod
    def convert_state_dict_to_openai_weight_model(modelft, state_dict_cpu: Dict[str, torch.Tensor]) -> nn.Module:
        config = GmPconverter.build_config_from_model(modelft)

        original_state_dict = convert_back_to_original(state_dict_cpu)
        original_state_dict = convert_state_dict_qkv_to_inproj(original_state_dict)

        from clip.model import CLIP
        original_model = CLIP(**config)
        original_model.load_state_dict(original_state_dict, strict=True)
        return original_model


def ModelSaver(
    model: nn.Module,
    epoch: int,
    device: str,
    ft_checkpoints_folder: str,
    state_dict_cpu: Optional[Dict[str, torch.Tensor]] = None,
    tag: str = "raw",
    save_fp16: bool = True,
    save_full: bool = True,
    save_dict: bool = False,
):
    """
    Saves model in OpenAI/CLIP format (full pickle and/or state_dict)
    """
    
    # always pass a FULL state_dict into the converter (for EMA model)
    if state_dict_cpu is None:
        state_dict_cpu = _gather_state_dict_cpu(model)
    else:
        state_dict_cpu = _merge_state_dict_cpu(model, state_dict_cpu)

    # always save as OpenAI-weight model
    model_to_save = GmPconverter.convert_state_dict_to_openai_weight_model(model, state_dict_cpu)

    if save_fp16:
        convert_weights(model_to_save)

    suffix = "as-weight"
    ckpt_name = f"clip_ft_e{epoch}_{tag}_full_{suffix}.pt"

    if save_full:
        torch.save(model_to_save, os.path.join(ft_checkpoints_folder, ckpt_name))

    if save_dict:
        torch.save(
            model_to_save.state_dict(),
            os.path.join(ft_checkpoints_folder, f"clip_ft_e{epoch}_{tag}_dict_{suffix}.pt")
        )

    del model_to_save


# ============================================================
# EMA (RAM-backed, sparse updates)
# ============================================================
class EMAState:
    def __init__(self, model: nn.Module, decay_step: float, store_fp16: bool):
        self.decay_step = float(decay_step)
        self.store_dtype = (torch.float16 if store_fp16 else torch.float32)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._steps_since_update = 0

        # debug counters
        self.num_updates: int = 0
        self.last_decay_eff: float = 1.0

        # pick some debug parameter to verify EMA works (this should have grad, obviously)
        self.debug_key: str = "visual.ln_post.weight" if any(n == "visual.ln_post.weight" for n, _ in model.named_parameters()) else ""

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().to("cpu", dtype=self.store_dtype).clone()

        # fallback debug key
        if not self.debug_key:
            for k in self.shadow.keys():
                self.debug_key = k
                break

    def step_counter_inc(self, n: int = 1):
        self._steps_since_update += int(n)

    @torch.no_grad()
    def update(self, model: nn.Module, force: bool = False):
        if not force and self._steps_since_update <= 0:
            return

        steps = max(1, self._steps_since_update)
        decay_eff = self.decay_step ** steps
        one_minus = 1.0 - decay_eff

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            src = p.detach().to("cpu", dtype=self.store_dtype)
            self.shadow[name].mul_(decay_eff).add_(src, alpha=one_minus)

        self._steps_since_update = 0
        self.num_updates += 1
        self.last_decay_eff = float(decay_eff)

    @torch.no_grad()
    def debug_delta(self, model: nn.Module) -> float:
        if not self.debug_key or self.debug_key not in self.shadow:
            return float("nan")

        if self.debug_key == "logit_scale" and hasattr(model, "logit_scale"):
            cur = float(model.logit_scale.detach().cpu())
            ema = float(self.shadow["logit_scale"].detach().cpu())
            return abs(cur - ema)

        p = dict(model.named_parameters()).get(self.debug_key, None)
        if p is None:
            return float("nan")
        cur0 = float(p.detach().view(-1)[0].cpu())
        ema0 = float(self.shadow[self.debug_key].detach().view(-1)[0].cpu())
        return abs(cur0 - ema0)

    # used by ModelSaver for EMA checkpoint
    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in self.shadow.items():
            out[k] = v.detach().cpu()
        return out