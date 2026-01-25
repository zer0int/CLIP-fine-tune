"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
  ------------------------------------------------------------

 Modification by zer0int for https://github.com/zer0int/CLIP-fine-tune
 OpenAI/CLIP (+ Long-CLIP variants) 
 -> HuggingFace Hub / transformers model.safetensors
 -> Extract Text Encoder (e.g. for generative models)

 NOTE! If you see a ModuleNotFoundError when loading a pickle (full model)         <----
 you trained a while ago, and with other code...       Search / CTRL + F:  GOTO

"""

from __future__ import annotations

import os
import shutil
from typing import Dict, Any, Optional, Tuple
import sys
import types
from contextlib import contextmanager
import torch
from transformers import CLIPConfig, CLIPModel


import warnings # stop spam so we can read important outputs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Your model path here:
model_checkpoint_path: str = "path/to/clip_ft_e20_raw_full_as-weight.pt"  # fine-tuned model pickle / checkpoint

# You shouldn't need to change any of what's below.
configs_root: str = r"utils_convert_hf"  # root folder containing all HF config subfolders (included!)
config_path: Optional[str] = None        # None = auto-infer from model (recommended)

save_converted_model_to: str = "CLIP_Regression_HF_converted/my_model"

to_huggingface_safetensors: bool = True
to_huggingface_text_encoder: bool = True
to_openai_clip_state_dict_safetensors: bool = False
to_openai_clip_text_encoder_only_safetensors: bool = False

# Optional: verify logits equality like the original script
run_sanity_check: bool = True
sanity_image_size: int = 224  # will be overridden to model's native res if mismatched
sanity_atol: float = 1e-3

# Long-CLIP convention (used when joining positional embeddings)
longclip_keep_len_default: int = 20  # used if model lacks .longclip_keep_len


def _try_torch_load_weights_only(path: str) -> Optional[Any]:
    """
    Try torch.load(weights_only=True) if supported by this torch version.
    Returns loaded object on success, else None.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch without weights_only
        return None
    except Exception:
        return None

@contextmanager
def _temporary_sys_modules(overrides: Dict[str, Any]):
    """
    Temporarily inject/override sys.modules entries.
    """
    old = {}
    for name, mod in overrides.items():
        old[name] = sys.modules.get(name, None)
        sys.modules[name] = mod
    try:
        yield
    finally:
        for name, prior in old.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _make_longclip_alias_modules() -> Dict[str, Any]:
    """
    Create aliases so old pickles referencing `longclip.*` can unpickle
    using current `oaiclip` code.
    """
    import oaiclip  # your package
    import oaiclip.model as oai_model
    import oaiclip.simple_tokenizer as oai_tok

    longclip_pkg = types.ModuleType("longclip")                         # <--- GOTO 
    longclip_pkg.model = oai_model
    longclip_pkg.simple_tokenizer = oai_tok

    # NOTE: We alias the submodules that are most commonly referenced in pickles.
    # Add more here if you see future ModuleNotFoundError's (e.g. longclip.clip).
    return {
        "longclip": longclip_pkg,
        "longclip.model": oai_model,
        "longclip.simple_tokenizer": oai_tok,
        #"mypickle":  mymodel_pkg,                                      # <---------- GOTO ---   ADD HERE
    }


def _safe_torch_load_with_aliases(path: str) -> Any:
    """
    Try normal torch.load; if it fails due to missing `longclip`, retry with aliases.
    WARNING: Unpickling arbitrary files is unsafe—only do this on your own checkpoints.
    """
    try:
        return torch.load(path, map_location="cpu")
    except ModuleNotFoundError as e:
        # only apply the hack for the known legacy namespace
        missing = getattr(e, "name", "") or ""
        if missing == "longclip" or missing.startswith("longclip."):    # <--- GOTO ---> if missing == "mypickle" ...
            overrides = _make_longclip_alias_modules()
            with _temporary_sys_modules(overrides):
                return torch.load(path, map_location="cpu")
        raise

def _maybe_import_safetensors():
    try:
        from safetensors.torch import save_file  # noqa: F401
    except Exception as e:
        raise ImportError("This output path needs safetensors. Install via: pip install safetensors") from e


def save_safetensors_state_dict(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    _maybe_import_safetensors()
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_file(state_dict, path)


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    out_proj_weights = pt_attn_layer.out_proj.weight
    out_proj_bias = pt_attn_layer.out_proj.bias

    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    hf_attn_layer.out_proj.weight = out_proj_weights
    hf_attn_layer.out_proj.bias = out_proj_bias


def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    hf_linear.weight = pt_linear.weight
    hf_linear.bias = pt_linear.bias


def copy_layer(hf_layer, pt_layer):
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)
    copy_mlp(hf_layer.mlp, pt_layer.mlp)
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model):
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T.contiguous()
    copy_encoder(hf_model.text_model, pt_model)


def copy_vison_model_and_projection(hf_model, pt_model):
    hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T.contiguous()
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight.data
    hf_model.vision_model.embeddings.class_embedding = pt_model.visual.class_embedding
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding.data

    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)


def _looks_like_full_model_pickle(obj: Any) -> bool:
    if isinstance(obj, torch.nn.Module):
        return True
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")) and not isinstance(obj, dict):
        return True
    return False


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        # direct state_dict?
        if obj and all(isinstance(k, str) for k in obj.keys()) and all(torch.is_tensor(v) for v in obj.values() if v is not None):
            return obj  # type: ignore[return-value]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]  # type: ignore[return-value]
        for k in ("model", "model_state", "model_state_dict", "net", "network"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]  # type: ignore[return-value]
    raise TypeError("Could not extract state_dict-like dict[str, Tensor] from loaded object.")


def _get_text_context_length_from_model(pt_model: torch.nn.Module) -> int:
    ctx = getattr(pt_model, "context_length", None)
    if ctx is not None:
        try:
            return int(ctx)
        except Exception:
            pass
    # fallback: positional_embedding length
    pe = getattr(pt_model, "positional_embedding", None)
    if pe is None:
        raise ValueError("Could not infer text context length (missing .context_length and .positional_embedding).")
    return int(pe.shape[0])


def _get_vision_patch_and_resolution(pt_model: torch.nn.Module) -> Tuple[int, int]:
    vis = getattr(pt_model, "visual", None)
    if vis is None or not hasattr(vis, "conv1"):
        raise ValueError("Could not infer vision arch (missing pt_model.visual.conv1).")
    patch = int(vis.conv1.weight.shape[-1])
    img_res = getattr(vis, "input_resolution", None)
    if img_res is None:
        raise ValueError("Could not infer image resolution (missing pt_model.visual.input_resolution).")
    return patch, int(img_res)


def infer_config_path_from_model(pt_model: torch.nn.Module, configs_root: str) -> str:
    """
    Infer one of:
      vit_b_16, vit_b_32, vit_l_14, vit_l_14_336,
      long_vit_b_16, long_vit_b_32, long_vit_l_14, long_vit_l_14_336
    """
    ctx = _get_text_context_length_from_model(pt_model)
    is_long = (ctx > 77)

    patch, img_res = _get_vision_patch_and_resolution(pt_model)

    # Determine base family by patch + resolution.
    # OpenAI conventions:
    #   B/16 -> patch 16, 224
    #   B/32 -> patch 32, 224
    #   L/14 -> patch 14, 224
    #   L/14@336 -> patch 14, 336
    if patch == 16 and img_res == 224:
        base = "vit_b_16"
    elif patch == 32 and img_res == 224:
        base = "vit_b_32"
    elif patch == 14 and img_res == 224:
        base = "vit_l_14"
    elif patch == 14 and img_res == 336:
        base = "vit_l_14_336"
    else:
        raise ValueError(f"Unsupported / unknown vision setup for auto-config: patch={patch}, image_res={img_res}")

    cfg_name = f"long_{base}" if is_long else base
    cfg_dir = os.path.join(configs_root, cfg_name)
    if not os.path.isdir(cfg_dir):
        raise FileNotFoundError(f"Auto-inferred config dir not found: {cfg_dir}")
    return cfg_dir


@torch.no_grad()
def join_longclip_positional_embeddings_inplace(
    pt_model: torch.nn.Module,
    keep_len_default: int = 20,
) -> bool:
    """
    If pt_model has positional_embedding_res, merge into positional_embedding and delete *_res.
    Returns True if a merge happened, else False.
    """
    if not hasattr(pt_model, "positional_embedding"):
        return False

    # Some Long-CLIP checkpoints have only positional_embedding (already merged)
    if not hasattr(pt_model, "positional_embedding_res"):
        # ensure flag off if present
        if hasattr(pt_model, "use_positional_embedding_res"):
            try:
                pt_model.use_positional_embedding_res = False
            except Exception:
                pass
        return False

    pe = pt_model.positional_embedding
    per = pt_model.positional_embedding_res

    if pe.shape != per.shape:
        raise ValueError(f"positional_embedding_res shape mismatch: pe={tuple(pe.shape)} vs per={tuple(per.shape)}")

    keep_len = int(getattr(pt_model, "longclip_keep_len", keep_len_default))
    keep_len = max(0, min(keep_len, int(pe.shape[0])))

    merged = pe.detach().clone()
    if keep_len < merged.shape[0]:
        merged[keep_len:] = per.detach()[keep_len:]

    # Replace positional_embedding with merged
    pt_model.positional_embedding = torch.nn.Parameter(merged)

    # Remove the res parameter from module internals (so state_dict() and attribute access stop seeing it)
    if hasattr(pt_model, "_parameters") and "positional_embedding_res" in pt_model._parameters:
        pt_model._parameters.pop("positional_embedding_res", None)
    # Also remove Long-CLIP masks if present (buffers, not persisted, but keep it clean)
    if hasattr(pt_model, "_buffers"):
        pt_model._buffers.pop("mask1", None)
        pt_model._buffers.pop("mask2", None)
    # And remove attribute if still present in __dict__
    pt_model.__dict__.pop("positional_embedding_res", None)
    pt_model.__dict__.pop("mask1", None)
    pt_model.__dict__.pop("mask2", None)

    # Turn off the flag if present
    if hasattr(pt_model, "use_positional_embedding_res"):
        try:
            pt_model.use_positional_embedding_res = False
        except Exception:
            pass

    return True


def _build_openai_model_from_state_dict(sd: Dict[str, torch.Tensor]) -> torch.nn.Module:
    """
    Build an OpenAI-style CLIP python module directly from a state_dict
    (supports Long-CLIP because your oaiclip.model.build_model detects 77 vs 248 and *_res).
    """
    from oaiclip.model import build_model as build_oai_model
    return build_oai_model(sd).eval()


def load_openai_clip_model_anything(checkpoint_path: str) -> Tuple[torch.nn.Module, Optional[Dict[str, torch.Tensor]]]:
    """
    Returns:
      - pt_model (OpenAI-style clip model)
      - openai_state_dict if we loaded a state_dict checkpoint (else None for full pickles / name loads)
    """
    import oaiclip as clip

    # OpenAI name passthrough
    try:
        available = set(clip.available_models())
    except Exception:
        available = set()

    if checkpoint_path in available:
        pt_model, _ = clip.load(checkpoint_path, device="cpu", jit=False)
        return pt_model.eval(), None

    # Local .pt/.pth only
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")

    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext not in (".pt", ".pth"):
        raise ValueError(f"Unsupported input file (expected .pt/.pth pickle/checkpoint): {checkpoint_path}")

    # 1) Try weights_only=True first (avoids legacy module import issues in many cases)
    wo = _try_torch_load_weights_only(checkpoint_path)
    if wo is not None:
        try:
            sd = _extract_state_dict(wo)
            pt_model = _build_openai_model_from_state_dict(sd)
            return pt_model.eval(), sd
        except Exception:
            # weights_only returned something non-state-dict-like; fall through
            pass

    # 2) Fall back to normal unpickle, with legacy `longclip` aliasing if needed
    obj = _safe_torch_load_with_aliases(checkpoint_path)

    # If it is a full nn.Module pickle, we can use it directly for conversion.
    if isinstance(obj, torch.nn.Module):
        return obj.eval(), None

    # Otherwise: treat as state_dict-like checkpoint (possibly wrapped)
    sd = _extract_state_dict(obj)
    pt_model = _build_openai_model_from_state_dict(sd)
    return pt_model.eval(), sd


def filter_hf_text_encoder_state_dict(hf_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Filter out keys containing "vision_" or "visual_"
    return {k: v for k, v in hf_state_dict.items() if ("vision_" not in k and "visual_" not in k)}


def filter_openai_text_encoder_state_dict(openai_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # For OpenAI CLIP ViT, vision keys start with "visual."
    return {k: v for k, v in openai_state_dict.items() if not k.startswith("visual.")}


@torch.no_grad()
def convert_clip_checkpoint(
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    config_path: Optional[str] = None,
    force_safetensors: bool = True,
    extract_hf_text_encoder: bool = True,
    save_openai_state_dict_safetensors: bool = False,
    save_openai_te_only_safetensors: bool = False,
) -> None:
    """
    Copy/paste/tweak model's weights to HF 'transformers' design, plus extra outputs.
    """

    # OpenAI CLIP load (name or pickle/checkpoint)
    pt_model, openai_sd = load_openai_clip_model_anything(checkpoint_path)
    pt_model = pt_model.eval().float()

    # capture original OpenAI state dict BEFORE we potentially join Long-CLIP embeddings
    openai_sd_for_saving: Optional[Dict[str, torch.Tensor]] = None
    if save_openai_state_dict_safetensors or save_openai_te_only_safetensors:
        openai_sd_for_saving = openai_sd if openai_sd is not None else pt_model.state_dict()

    # auto-infer config folder if config_path is None
    if config_path is None:
        config_path = infer_config_path_from_model(pt_model, configs_root=configs_root)

    # config load
    config = CLIPConfig.from_pretrained(config_path)

    # Long-CLIP join (only affects HF conversion path)
    merged = join_longclip_positional_embeddings_inplace(
        pt_model,
        keep_len_default=longclip_keep_len_default,
    )

    # Validate config vs model context length early (nicer failure mode than shape mismatch deep inside)
    model_ctx = _get_text_context_length_from_model(pt_model)
    cfg_ctx = int(getattr(config.text_config, "max_position_embeddings", 77))
    if model_ctx != cfg_ctx:
        raise ValueError(
            f"Context length mismatch: model_ctx={model_ctx} vs config.text_config.max_position_embeddings={cfg_ctx}. "
            f"Auto-config picked: {config_path}. If you overrode config_path manually, fix it."
        )

    hf_model = CLIPModel(config).eval()

    # Copy weights into HF model
    copy_text_model_and_projection(hf_model, pt_model)
    copy_vison_model_and_projection(hf_model, pt_model)
    hf_model.logit_scale = pt_model.logit_scale

    # Optional sanity check
    if run_sanity_check:
        # build input_ids with correct context length (77 or 248)
        bos = int(config.text_config.bos_token_id)
        eos = int(config.text_config.eos_token_id)
        pad = int(config.text_config.pad_token_id)
        ctx = int(config.text_config.max_position_embeddings)

        input_ids = torch.full((1, ctx), pad, dtype=torch.long)
        input_ids[0, 0] = bos
        if ctx > 3:
            # fill with small ids < eos so argmax lands on eos (OpenAI encode_text convention)
            fill_len = max(0, ctx - 3)  # positions [1 .. ctx-3] inclusive
            if fill_len > 0:
                input_ids[0, 1:1 + fill_len] = torch.arange(3, 3 + fill_len, dtype=torch.long)
        input_ids[0, ctx - 2] = eos
        # input_ids[0, ctx - 1] is pad

        # ensure pixel_values match model resolution (esp. 336px variants)
        native_res = getattr(pt_model.visual, "input_resolution", sanity_image_size)
        if int(native_res) != int(sanity_image_size):
            # override silently; you can still set sanity_image_size=native_res if you prefer
            sanity_sz = int(native_res)
        else:
            sanity_sz = int(sanity_image_size)

        pixel_values = torch.randn(1, 3, sanity_sz, sanity_sz)

        hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
        hf_logits_per_image = hf_outputs.logits_per_image
        hf_logits_per_text = hf_outputs.logits_per_text
        pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)

        assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=sanity_atol), "Sanity check failed (image logits)."
        assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=sanity_atol), "Sanity check failed (text logits)."

        if merged:
            print("[Info] Long-CLIP positional embeddings were joined for HF conversion.")

    # Prepare output folder
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    # Save HF full model (optionally as safetensors)
    hf_model.save_pretrained(
        pytorch_dump_folder_path,
        safe_serialization=force_safetensors,
    )

    # Copy "config payload" files (tokenizer, preprocessor, vocab, merges, etc.)
    if config_path is not None and os.path.isdir(config_path):
        _copy_hf_payload_files(config_path, pytorch_dump_folder_path)

    # HF TE-only extraction (always based on converted HF model)
    if extract_hf_text_encoder:
        hf_sd = hf_model.state_dict()
        hf_te = filter_hf_text_encoder_state_dict(hf_sd)
        te_path = os.path.join(pytorch_dump_folder_path, "model_hf_te-only.safetensors")
        save_safetensors_state_dict(hf_te, te_path)
        print(f"[Saved] HF text-encoder-only -> {te_path}")

    # Save OpenAI state_dict as-is (optional)
    if save_openai_state_dict_safetensors:
        assert openai_sd_for_saving is not None
        out_path = os.path.join(pytorch_dump_folder_path, "model_openai.safetensors")
        save_safetensors_state_dict(openai_sd_for_saving, out_path)
        print(f"[Saved] OpenAI state_dict -> {out_path}")

    # Save OpenAI TE-only (optional)
    if save_openai_te_only_safetensors:
        assert openai_sd_for_saving is not None
        te_only = filter_openai_text_encoder_state_dict(openai_sd_for_saving)
        out_path = os.path.join(pytorch_dump_folder_path, "model_openai_te-only.safetensors")
        save_safetensors_state_dict(te_only, out_path)
        print(f"[Saved] OpenAI text-encoder-only -> {out_path}")

    print(f"[Done] Converted + saved -> {pytorch_dump_folder_path}")
    print(f"[Info] HF config used -> {config_path}")


def _copy_hf_payload_files(src_cfg_dir: str, dst_dir: str) -> None:
    """
    Copy extra HF files if present (tokenizer.json, vocab, merges, preprocessor_config.json, etc.)
    without clobbering the freshly saved model/config unless needed.
    """
    skip_ext = {".bin", ".safetensors", ".pt", ".pth"}
    for root, _dirs, files in os.walk(src_cfg_dir):
        rel = os.path.relpath(root, src_cfg_dir)
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in skip_ext:
                continue
            src = os.path.join(root, fn)
            dst = os.path.join(dst_dir, rel, fn) if rel != "." else os.path.join(dst_dir, fn)

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # Don't overwrite model's config.json that save_pretrained just wrote.
            if os.path.basename(dst) == "config.json" and os.path.isfile(os.path.join(dst_dir, "config.json")):
                continue

            shutil.copy2(src, dst)


def main() -> None:
    convert_clip_checkpoint(
        checkpoint_path=model_checkpoint_path,
        pytorch_dump_folder_path=save_converted_model_to,
        config_path=config_path,
        force_safetensors=to_huggingface_safetensors,
        extract_hf_text_encoder=to_huggingface_text_encoder,
        save_openai_state_dict_safetensors=to_openai_clip_state_dict_safetensors,
        save_openai_te_only_safetensors=to_openai_clip_text_encoder_only_safetensors,
    )


if __name__ == "__main__":
    main()