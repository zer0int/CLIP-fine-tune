# ------------------------------------------------------------
# "Whatever you throw at this module will come out as an OpenAI/CLIP model."
#
# Supports:
# - OpenAI CLIP names: "ViT-L/14", ... (clip.available_models) -> passthrough
# - Local .pt / .pth:
#     * full model pickle ("danger pickle") -> passthrough to clip.load(path, jit=False)
#     * state_dict -> detect OpenAI vs HF -> convert if needed -> load into OpenAI CLIP
# - Local .safetensors -> detect OpenAI vs HF -> convert if needed
# - HF Hub repo id (or local HF folder):
#     * downloads weights (prefers safetensors) -> detect OpenAI vs HF -> convert if needed
#
# Notes:
# - Uses key-based heuristics to decide formats + infer CLIP variant.
# - Reverse HF -> OpenAI conversion implemented for CLIP ViT vision tower + CLIP text tower.

from __future__ import annotations

import os
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import importlib
from typing import Callable
from colorama import Fore, Style


class ClipAnythingLoadError(RuntimeError):
    """Raised when load_openai_clip_anything() can’t resolve the input as local path or HF repo."""
    pass


def _looks_like_local_path(s: str) -> bool:
    """
    Heuristic: treat strings that *look like* filesystem paths as paths even if they don't exist.
    This avoids accidentally feeding 'D:/foo/bar.pt' into HF repo-id validation.
    """
    if not isinstance(s, str) or not s:
        return False

    s2 = s.strip()

    # Windows drive letter / UNC paths
    if len(s2) >= 2 and s2[1] == ":":
        return True
    if s2.startswith("\\\\"):
        return True

    # Common path separators / relative / home
    if ("/" in s2) or ("\\" in s2):
        # BUT: HF repo ids contain exactly one "/" usually; those should still be allowed.
        # If it contains backslash, it's definitely a path. If it contains multiple "/" or a leading "./", treat as path.
        if "\\" in s2:
            return True
        if s2.startswith("./") or s2.startswith("../") or s2.startswith("~/"):
            return True
        if s2.count("/") >= 2:
            return True

    # File extensions commonly used for local checkpoints
    ext = os.path.splitext(s2)[1].lower()
    if ext in (".pt", ".pth", ".safetensors", ".bin", ".ckpt"):
        return True

    return False


def _get_openai_transform_fn(clip_module) -> Optional[Callable[[int], Any]]:
    """
    Find OpenAI-style _transform(n_px) in the provided clip_module, without assuming packaging.
    Returns callable or None.
    """
    # 1) clip_module._transform
    cand = getattr(clip_module, "_transform", None)
    if callable(cand):
        return cand

    # clip_module.clip._transform
    sub = getattr(clip_module, "clip", None)
    cand = getattr(sub, "_transform", None) if sub is not None else None
    if callable(cand):
        return cand

    # import <clip_module.__name__>.clip and use _transform
    mod_name = getattr(clip_module, "__name__", None)
    if isinstance(mod_name, str) and mod_name:
        try:
            clip_sub = importlib.import_module(f"{mod_name}.clip")
            cand = getattr(clip_sub, "_transform", None)
            if callable(cand):
                return cand
        except Exception:
            pass

    return None

def _set_tokenize_default_context_length(clip_module, ctx: int) -> None:
    """
    Force the default context length used by clip.tokenize() by editing the tokenize()
    function's module globals (robust across package layouts).
    """
    ctx = int(ctx)

    # try a few common locations for tokenize()
    candidates = []
    candidates.append(getattr(clip_module, "tokenize", None))
    sub = getattr(clip_module, "clip", None)
    if sub is not None:
        candidates.append(getattr(sub, "tokenize", None))

    for tok in candidates:
        if callable(tok):
            try:
                tok.__globals__["_DEFAULT_CONTEXT_LENGTH"] = ctx
            except Exception:
                pass

    # also keep a best-effort attribute for debugging / introspection
    try:
        setattr(clip_module, "_DEFAULT_CONTEXT_LENGTH", ctx)
    except Exception:
        pass
    if sub is not None:
        try:
            setattr(sub, "_DEFAULT_CONTEXT_LENGTH", ctx)
        except Exception:
            pass


def _build_fallback_transform(n_px: int) -> Any:
    """
    Fallback preprocessing equivalent to OpenAI CLIP:
      Resize -> CenterCrop -> RGB -> ToTensor -> Normalize
    """
    try:
        from PIL import Image
        from torchvision import transforms as T
    except Exception as e:
        raise ImportError(Fore.RED + 
            "\nCould not build preprocess transform because torchvision (and PIL) is missing. "
            "Install torchvision or expose clip_module._transform." + Style.RESET_ALL
        ) from e

    # torchvision interpolation enum differs by version; be permissive
    try:
        interp = T.InterpolationMode.BICUBIC
    except Exception:
        interp = Image.BICUBIC  # older torchvision

    return T.Compose([
        T.Resize(n_px, interpolation=interp),
        T.CenterCrop(n_px),
        T.Lambda(lambda image: image.convert("RGB")),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def _get_openai_build_model_fn(clip_module) -> Callable[[Dict[str, torch.Tensor]], torch.nn.Module]:
    """
    Find build_model in the provided clip_module, without assuming packaging.
    """
    fn = None

    # clip_module.model.build_model
    try:
        model_mod = getattr(clip_module, "model", None)
        if model_mod is not None:
            cand = getattr(model_mod, "build_model", None)
            if callable(cand):
                fn = cand
    except Exception:
        pass

    # import <clip_module.__name__>.model and use build_model
    if fn is None:
        mod_name = getattr(clip_module, "__name__", None)
        if isinstance(mod_name, str) and mod_name:
            try:
                model_mod = importlib.import_module(f"{mod_name}.model")
                cand = getattr(model_mod, "build_model", None)
                if callable(cand):
                    fn = cand
            except Exception:
                pass

    if fn is None:
        raise RuntimeError(Fore.RED + 
            "\nCould not locate build_model. Expected either:\n"
            "  clip_module.model.build_model\n"
            "or\n"
            "  import <clip_module.__name__>.model; model.build_model" + Style.RESET_ALL
        )

    return fn

@dataclass
class ClipLoadInfo:
    source_kind: str  # "openai_name" | "openai_pickle" | "openai_state_dict" | "hf_state_dict"
    resolved_path: Optional[str]
    detected_format: str  # "openai" | "hf" | "unknown"
    inferred_openai_model: Optional[str]
    inferred_signature: Dict[str, Any]

def _maybe_convert_inproj_to_qkv(clip_module, state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool, str]:
    """
    If clip_module provides:
        clip_module.model.convert_state_dict_inproj_to_qkv
    or importable as:
        <clip_module.__name__>.model.convert_state_dict_inproj_to_qkv
    then apply it and return (new_sd, True, "how_found").

    If not found, return (state_dict, False, "").

    IMPORTANT: If the function is found but errors, we raise.
    """
    fn = None
    how = ""

    # Attribute path: clip_module.model.convert_state_dict_inproj_to_qkv
    try:
        model_mod = getattr(clip_module, "model", None)
        if model_mod is not None:
            cand = getattr(model_mod, "convert_state_dict_inproj_to_qkv", None)
            if callable(cand):
                fn = cand
                how = "clip_module.model.convert_state_dict_inproj_to_qkv"
    except Exception:
        pass

    # Import path: import <clip_module.__name__>.model
    if fn is None:
        try:
            mod_name = getattr(clip_module, "__name__", None)
            if isinstance(mod_name, str) and mod_name:
                model_mod = importlib.import_module(f"{mod_name}.model")
                cand = getattr(model_mod, "convert_state_dict_inproj_to_qkv", None)
                if callable(cand):
                    fn = cand
                    how = f"{mod_name}.model.convert_state_dict_inproj_to_qkv"
        except Exception:
            pass

    if fn is None:
        return state_dict, False, ""

    # Found converter => apply
    try:
        out = fn(state_dict)
        if out is None:
            return state_dict, True, how
        if not isinstance(out, dict):
            raise TypeError(Fore.RED + f"\nconvert_state_dict_inproj_to_qkv returned {type(out)} (expected dict or None)." + Style.RESET_ALL)
        return out, True, how
    except Exception as e:
        raise RuntimeError(Fore.RED + f"\nFound {how} but conversion failed: {e}" + Style.RESET_ALL) from e


def load_openai_clip_anything(
    clip_module,
    model_or_path: str,
    device: str = "cpu",
    jit: bool = False,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Any, ClipLoadInfo]:

    # OpenAI model name passthrough
    try:
        available = set(clip_module.available_models())
    except Exception:
        available = set()

    if model_or_path in available:
        model, preprocess = clip_module.load(model_or_path, device=device, jit=jit)
        info = ClipLoadInfo(
            source_kind="openai_name",
            resolved_path=None,
            detected_format="openai",
            inferred_openai_model=model_or_path,
            inferred_signature={"note": "passed through clip.available_models()"},
        )
        return model, preprocess, info

    # Local path handling (.pt pickle / state_dict / safetensors / folder)
    if os.path.exists(model_or_path):
        if os.path.isdir(model_or_path):
            state_dict, resolved = _load_state_dict_from_hf_folder(model_or_path)
            return _instantiate_and_load_openai_clip_from_state_dict(
                clip_module,
                state_dict=state_dict,
                device=device,
                jit=jit,
                strict=strict,
                resolved_path=resolved,
            )

        ext = os.path.splitext(model_or_path)[1].lower()
        if ext in (".pt", ".pth"):
            obj = torch.load(model_or_path, map_location="cpu")
            # Full model pickle passthrough
            if _looks_like_full_model_pickle(obj):
                model, preprocess = clip_module.load(model_or_path, device=device, jit=jit)
                info = ClipLoadInfo(
                    source_kind="openai_pickle",
                    resolved_path=model_or_path,
                    detected_format="openai",
                    inferred_openai_model=None,
                    inferred_signature={"note": "passed through full model pickle"},
                )
                return model, preprocess, info

            state_dict = _extract_state_dict(obj)
            return _instantiate_and_load_openai_clip_from_state_dict(
                clip_module,
                state_dict=state_dict,
                device=device,
                jit=jit,
                strict=strict,
                resolved_path=model_or_path,
            )

        if ext == ".safetensors":
            state_dict = _load_safetensors_state_dict(model_or_path)
            return _instantiate_and_load_openai_clip_from_state_dict(
                clip_module,
                state_dict=state_dict,
                device=device,
                jit=jit,
                strict=strict,
                resolved_path=model_or_path,
            )

        raise ValueError(Fore.RED + f"\nUnsupported local file extension: {ext} (path={model_or_path})" + Style.RESET_ALL)

    # path-looking guard
    if _looks_like_local_path(model_or_path):
        msg = (Fore.RED + 
            f"\nLooks like a local path, but it does not exist:\n"
            f"  {model_or_path}\n"
            f"Fix the typo or pass a valid HF repo id like 'namespace/repo_name'." + Style.RESET_ALL
        )
        raise ClipAnythingLoadError(msg)


    # HF loading wrapped to catch errors
    try:
        state_dict, resolved = _load_state_dict_from_hf_repo(
            repo_id=model_or_path,
            cache_dir=cache_dir,
            revision=revision,
        )
    except Exception as e:
        # keep message compact; caller decides whether to print traceback
        raise ClipAnythingLoadError(str(e)) from None

    return _instantiate_and_load_openai_clip_from_state_dict(
        clip_module,
        state_dict=state_dict,
        device=device,
        jit=jit,
        strict=strict,
        resolved_path=resolved,
    )


def resolve_to_openai_state_dict(
    model_or_path: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], ClipLoadInfo]:

    if os.path.exists(model_or_path):
        if os.path.isdir(model_or_path):
            sd, resolved = _load_state_dict_from_hf_folder(model_or_path)
        else:
            ext = os.path.splitext(model_or_path)[1].lower()
            if ext in (".pt", ".pth"):
                obj = torch.load(model_or_path, map_location="cpu")
                if _looks_like_full_model_pickle(obj):
                    raise ValueError(Fore.RED + "\nresolve_to_openai_state_dict(): got a full model pickle; no state_dict to return." + Style.RESET_ALL)
                sd = _extract_state_dict(obj)
                resolved = model_or_path
            elif ext == ".safetensors":
                sd = _load_safetensors_state_dict(model_or_path)
                resolved = model_or_path
            else:
                raise ValueError(Fore.RED + f"\nUnsupported local file extension: {ext} (path={model_or_path})" + Style.RESET_ALL)
    else:
        if _looks_like_local_path(model_or_path):
            raise ClipAnythingLoadError(Fore.RED +
                f"\nThis looks like a local path, but it does not exist:\n  {model_or_path}" + Style.RESET_ALL
            )

        try:
            sd, resolved = _load_state_dict_from_hf_repo(model_or_path, cache_dir=cache_dir, revision=revision)
        except Exception as e:
            raise ClipAnythingLoadError(str(e)) from None

    openai_sd, info = _convert_any_state_dict_to_openai(sd, resolved_path=resolved)
    return openai_sd, info


def _load_state_dict_from_hf_repo(
    repo_id: str,
    cache_dir: Optional[str],
    revision: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], str]:

    try:
        from huggingface_hub import HfApi, hf_hub_download
        # errors live in huggingface_hub.errors in modern versions
        try:
            from huggingface_hub.errors import (
                HFValidationError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
                EntryNotFoundError,
                LocalEntryNotFoundError,
                HfHubHTTPError,
            )
        except Exception:
            HFValidationError = RepositoryNotFoundError = RevisionNotFoundError = EntryNotFoundError = LocalEntryNotFoundError = HfHubHTTPError = Exception
    except Exception as e:
        raise ImportError(Fore.RED + "\nhuggingface_hub is required for HF repo loading. pip install huggingface_hub" + Style.RESET_ALL) from e

    api = HfApi()

    # List files without downloading them
    try:
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, revision=revision)
        except TypeError:
            # older versions may not accept revision in list_repo_files
            repo_files = api.list_repo_files(repo_id=repo_id)
    except HFValidationError as e:
        raise ClipAnythingLoadError(Fore.RED + 
            f"\nInvalid HuggingFace repo id: '{repo_id}'. Expected 'repo_name' or 'namespace/repo_name'." + Style.RESET_ALL
        ) from None
    except RepositoryNotFoundError as e:
        raise ClipAnythingLoadError(Fore.RED + 
            f"\nHuggingFace repo not found (or private without auth): '{repo_id}'." + Style.RESET_ALL
        ) from None
    except RevisionNotFoundError as e:
        raise ClipAnythingLoadError(Fore.RED + 
            f"\nHuggingFace revision not found for repo '{repo_id}': revision={revision!r}." + Style.RESET_ALL
        ) from None
    except (HfHubHTTPError, EntryNotFoundError, LocalEntryNotFoundError) as e:
        raise ClipAnythingLoadError(Fore.RED + 
            f"\nHuggingFace hub error while querying '{repo_id}' (revision={revision!r}): {e}" + Style.RESET_ALL
        ) from None
    except Exception as e:
        raise ClipAnythingLoadError(Fore.RED + 
            f"\nUnexpected error while querying HuggingFace repo '{repo_id}' (revision={revision!r}): {e}" + Style.RESET_ALL
        ) from None

    files = set(repo_files)

    # pick a single "best" weights entry
    def _first_existing(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in files:
                return c
        return None

    preferred = (
        ["model.safetensors.index.json", "model.safetensors"] +
        ["pytorch_model.bin.index.json", "pytorch_model.bin"]
    )
    picked = _first_existing(preferred)

    if picked is None:
        st_indexes = sorted([f for f in files if f.endswith(".safetensors.index.json")])
        st_singles = sorted([f for f in files if f.endswith(".safetensors") and not f.endswith(".safetensors.index.json")])
        bin_indexes = sorted([f for f in files if f.endswith(".bin.index.json")])
        bin_singles = sorted([f for f in files if f.endswith(".bin") and not f.endswith(".bin.index.json")])

        picked = (st_indexes[0] if st_indexes else None) \
              or (st_singles[0] if st_singles else None) \
              or (bin_indexes[0] if bin_indexes else None) \
              or (bin_singles[0] if bin_singles else None)

    if picked is None:
        raise ClipAnythingLoadError(Fore.RED + 
            f"\nNo supported weights found in repo '{repo_id}'. "
            f"Expected model.safetensors / model.safetensors.index.json / pytorch_model.bin / *.safetensors / *.bin." + Style.RESET_ALL
        )

    print(Fore.BLUE + f"[hf DEBUG] repo_id={repo_id} revision={revision} picked_weights='{picked}'" + Style.RESET_ALL)

    def _dl(filename: str) -> str:
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as e:
            raise ClipAnythingLoadError(Fore.RED + 
                f"\nFailed to download '{filename}' from '{repo_id}' (revision={revision!r}): {e}" + Style.RESET_ALL
            ) from None

    if picked.endswith(".safetensors.index.json"):
        index_path = _dl(picked)
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map: Dict[str, str] = idx.get("weight_map", {})
        if not weight_map:
            raise ClipAnythingLoadError(Fore.RED + f"\nInvalid safetensors index (no weight_map): {picked}" + Style.RESET_ALL)

        shard_files = sorted(set(weight_map.values()))
        out: Dict[str, torch.Tensor] = {}

        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise ImportError(Fore.RED + "\nsafetensors is required to read .safetensors. pip install safetensors" + Style.RESET_ALL) from e

        for shard in shard_files:
            shard_path = _dl(shard)
            out.update(load_file(shard_path, device="cpu"))

        return out, index_path

    if picked.endswith(".safetensors"):
        st_path = _dl(picked)
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise ImportError(Fore.RED + "\nsafetensors is required to read .safetensors. pip install safetensors" + Style.RESET_ALL) from e
        return load_file(st_path, device="cpu"), st_path

    if picked.endswith(".bin.index.json"):
        index_path = _dl(picked)
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map: Dict[str, str] = idx.get("weight_map", {})
        if not weight_map:
            raise ClipAnythingLoadError(Fore.RED + f"\nInvalid bin index (no weight_map): {picked}" + Style.RESET_ALL)

        shard_files = sorted(set(weight_map.values()))
        out: Dict[str, torch.Tensor] = {}
        for shard in shard_files:
            shard_path = _dl(shard)
            shard_obj = torch.load(shard_path, map_location="cpu")
            out.update(_extract_state_dict(shard_obj))
        return out, index_path

    bin_path = _dl(picked)
    obj = torch.load(bin_path, map_location="cpu")
    return _extract_state_dict(obj), bin_path


def _load_state_dict_from_hf_folder(folder: str) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Loads weights from a local HF folder.
    Handles:
      - model.safetensors
      - sharded safetensors: model.safetensors.index.json
      - pytorch_model.bin
      - sharded bin: pytorch_model.bin.index.json
    """
    # safetensors (single)
    st_path = os.path.join(folder, "model.safetensors")
    if os.path.isfile(st_path):
        return _load_safetensors_state_dict(st_path), st_path

    # safetensors (sharded)
    st_index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(st_index):
        return _load_sharded_safetensors(folder, st_index), st_index

    # bin (single)
    bin_path = os.path.join(folder, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
        return _extract_state_dict(sd), bin_path

    # bin (sharded)
    bin_index = os.path.join(folder, "pytorch_model.bin.index.json")
    if os.path.isfile(bin_index):
        return _load_sharded_bin(folder, bin_index), bin_index

    # fallback: any safetensors file in folder
    for fn in os.listdir(folder):
        if fn.endswith(".safetensors") and os.path.isfile(os.path.join(folder, fn)):
            p = os.path.join(folder, fn)
            return _load_safetensors_state_dict(p), p

    raise FileNotFoundError(Fore.RED + f"\nNo supported weights found in folder: {folder}" + Style.RESET_ALL)


def _load_safetensors_state_dict(path: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except Exception as e:
        raise ImportError(Fore.RED + "\nsafetensors is required to read .safetensors. pip install safetensors" + Style.RESET_ALL) from e
    return load_file(path, device="cpu")


def _load_sharded_safetensors(folder: str, index_json_path: str) -> Dict[str, torch.Tensor]:
    with open(index_json_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx.get("weight_map", {})
    if not weight_map:
        raise ValueError(Fore.RED + f"\nInvalid safetensors index.json (no weight_map): {index_json_path}" + Style.RESET_ALL)

    shards = sorted(set(weight_map.values()))
    out: Dict[str, torch.Tensor] = {}
    for shard in shards:
        shard_path = os.path.join(folder, shard)
        shard_sd = _load_safetensors_state_dict(shard_path)
        out.update(shard_sd)
    return out


def _load_sharded_bin(folder: str, index_json_path: str) -> Dict[str, torch.Tensor]:
    with open(index_json_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx.get("weight_map", {})
    if not weight_map:
        raise ValueError(Fore.RED + f"\nInvalid bin index.json (no weight_map): {index_json_path}" + Style.RESET_ALL)

    shards = sorted(set(weight_map.values()))
    out: Dict[str, torch.Tensor] = {}
    for shard in shards:
        shard_path = os.path.join(folder, shard)
        shard_sd = torch.load(shard_path, map_location="cpu")
        out.update(_extract_state_dict(shard_sd))
    return out


# Format detection + conversion
def _convert_any_state_dict_to_openai(
    state_dict: Dict[str, torch.Tensor],
    resolved_path: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], ClipLoadInfo]:
    sd = _canonicalize_state_dict_keys(state_dict)

    detected = _detect_clip_state_dict_format(sd)

    if detected == "openai":
        openai_sd = _canonicalize_openai_like_state_dict(sd)
        sig = _infer_openai_signature(openai_sd)
        inferred_name = _infer_openai_model_name(sig)
        info = ClipLoadInfo(
            source_kind="openai_state_dict",
            resolved_path=resolved_path,
            detected_format="openai",
            inferred_openai_model=inferred_name,
            inferred_signature=sig,
        )
        return openai_sd, info

    if detected == "hf":
        openai_sd = _convert_hf_clip_state_dict_to_openai(sd)
        openai_sd = _canonicalize_openai_like_state_dict(openai_sd)
        sig = _infer_openai_signature(openai_sd)
        inferred_name = _infer_openai_model_name(sig)
        info = ClipLoadInfo(
            source_kind="hf_state_dict",
            resolved_path=resolved_path,
            detected_format="hf",
            inferred_openai_model=inferred_name,
            inferred_signature=sig,
        )
        return openai_sd, info

    # unknown
    raise ValueError(Fore.RED + 
        "\nCould not confidently detect CLIP state_dict format as OpenAI or HF transformers.\n"
        "If this is a CLIP-ish model, check whether keys include e.g.\n"
        "  OpenAI: 'visual.conv1.weight', 'transformer.resblocks.0.attn.in_proj_weight'\n"
        "  HF:     'vision_model.embeddings.patch_embedding.weight', 'text_model.encoder.layers.0.self_attn.q_proj.weight'\n" + Style.RESET_ALL
    )


def _detect_clip_state_dict_format(sd: Dict[str, torch.Tensor]) -> str:
    """
    Returns: "openai" | "hf" | "unknown"
    """
    # OpenAI markers
    openai_markers = [
        "token_embedding.weight",
        "positional_embedding",
        "ln_final.weight",
        "text_projection",
        "visual.conv1.weight",
    ]
    openai_prefix_hits = 0
    if any(k.startswith("visual.transformer.resblocks.") for k in sd):
        openai_prefix_hits += 2
    if any(k.startswith("transformer.resblocks.") for k in sd):
        openai_prefix_hits += 2

    openai_hits = sum(1 for m in openai_markers if m in sd) + openai_prefix_hits

    # HF transformers markers
    hf_markers = [
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "text_projection.weight",
        "visual_projection.weight",
    ]
    hf_hits = sum(1 for m in hf_markers if _has_key_or_suffix(sd, m))

    if openai_hits >= 3 and openai_hits >= hf_hits + 1:
        return "openai"
    if hf_hits >= 3 and hf_hits >= openai_hits + 1:
        return "hf"
    return "unknown"


def _convert_hf_clip_state_dict_to_openai(hf_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Converts *HF transformers CLIPModel* style keys -> OpenAI CLIP style keys.
    Implemented to mirror the HuggingFace 'convert_openai_clip_to_hf' logic, but reversed.

    This expects a fairly standard CLIPModel naming scheme.
    """
    out: Dict[str, torch.Tensor] = {}

    # Remove noisy buffers if present
    for noisy in (
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    ):
        _pop_by_suffix(hf_sd, noisy)

    # Text (embeddings + final LN + projection)
    out["token_embedding.weight"] = _get_by_suffix(hf_sd, "text_model.embeddings.token_embedding.weight")
    out["positional_embedding"] = _get_by_suffix(hf_sd, "text_model.embeddings.position_embedding.weight")
    out["ln_final.weight"] = _get_by_suffix(hf_sd, "text_model.final_layer_norm.weight")
    out["ln_final.bias"] = _get_by_suffix(hf_sd, "text_model.final_layer_norm.bias")

    # HF text_projection is Linear(out_dim=projection_dim, in_dim=hidden_size) => weight [proj, hidden]
    # OpenAI text_projection is Parameter [hidden, proj]
    text_proj_w = _get_by_suffix(hf_sd, "text_projection.weight")
    out["text_projection"] = text_proj_w.T.contiguous()

    # Text transformer layers
    text_layer_ids = _collect_layer_indices(hf_sd, prefix_suffix="text_model.encoder.layers.")
    if not text_layer_ids:
        raise ValueError("HF->OpenAI: could not find any text_model.encoder.layers.* in state_dict")

    for i in text_layer_ids:
        # LayerNorms
        out[f"transformer.resblocks.{i}.ln_1.weight"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.layer_norm1.weight")
        out[f"transformer.resblocks.{i}.ln_1.bias"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.layer_norm1.bias")
        out[f"transformer.resblocks.{i}.ln_2.weight"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.layer_norm2.weight")
        out[f"transformer.resblocks.{i}.ln_2.bias"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.layer_norm2.bias")

        # MLP
        out[f"transformer.resblocks.{i}.mlp.c_fc.weight"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.mlp.fc1.weight")
        out[f"transformer.resblocks.{i}.mlp.c_fc.bias"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.mlp.fc1.bias")
        out[f"transformer.resblocks.{i}.mlp.c_proj.weight"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.mlp.fc2.weight")
        out[f"transformer.resblocks.{i}.mlp.c_proj.bias"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.mlp.fc2.bias")

        # Attention: q,k,v -> in_proj
        q_w = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.q_proj.weight")
        k_w = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.k_proj.weight")
        v_w = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.v_proj.weight")
        q_b = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.q_proj.bias")
        k_b = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.k_proj.bias")
        v_b = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.v_proj.bias")

        out[f"transformer.resblocks.{i}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
        out[f"transformer.resblocks.{i}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0).contiguous()

        out[f"transformer.resblocks.{i}.attn.out_proj.weight"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.out_proj.weight")
        out[f"transformer.resblocks.{i}.attn.out_proj.bias"] = _get_by_suffix(hf_sd, f"text_model.encoder.layers.{i}.self_attn.out_proj.bias")

    # Vision (ViT-style)
    # Projection: HF visual_projection.weight [proj, hidden] -> OpenAI visual.proj [hidden, proj]
    vis_proj_w = _get_by_suffix(hf_sd, "visual_projection.weight")
    out["visual.proj"] = vis_proj_w.T.contiguous()

    # Layer norms
    pre_ln_w = _get_by_any_suffix(hf_sd, ["vision_model.pre_layrnorm.weight", "vision_model.pre_layernorm.weight"])
    pre_ln_b = _get_by_any_suffix(hf_sd, ["vision_model.pre_layrnorm.bias", "vision_model.pre_layernorm.bias"])
    post_ln_w = _get_by_suffix(hf_sd, "vision_model.post_layernorm.weight")
    post_ln_b = _get_by_suffix(hf_sd, "vision_model.post_layernorm.bias")

    out["visual.ln_pre.weight"] = pre_ln_w
    out["visual.ln_pre.bias"] = pre_ln_b
    out["visual.ln_post.weight"] = post_ln_w
    out["visual.ln_post.bias"] = post_ln_b

    # Embeddings
    out["visual.conv1.weight"] = _get_by_suffix(hf_sd, "vision_model.embeddings.patch_embedding.weight")
    out["visual.class_embedding"] = _get_by_suffix(hf_sd, "vision_model.embeddings.class_embedding")
    out["visual.positional_embedding"] = _get_by_suffix(hf_sd, "vision_model.embeddings.position_embedding.weight")

    # Vision transformer layers
    vis_layer_ids = _collect_layer_indices(hf_sd, prefix_suffix="vision_model.encoder.layers.")
    if not vis_layer_ids:
        raise ValueError("HF->OpenAI: could not find any vision_model.encoder.layers.* in state_dict")

    for i in vis_layer_ids:
        out[f"visual.transformer.resblocks.{i}.ln_1.weight"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.layer_norm1.weight")
        out[f"visual.transformer.resblocks.{i}.ln_1.bias"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.layer_norm1.bias")
        out[f"visual.transformer.resblocks.{i}.ln_2.weight"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.layer_norm2.weight")
        out[f"visual.transformer.resblocks.{i}.ln_2.bias"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.layer_norm2.bias")

        out[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.mlp.fc1.weight")
        out[f"visual.transformer.resblocks.{i}.mlp.c_fc.bias"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.mlp.fc1.bias")
        out[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.mlp.fc2.weight")
        out[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.mlp.fc2.bias")

        q_w = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight")
        k_w = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight")
        v_w = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight")
        q_b = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias")
        k_b = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias")
        v_b = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias")

        out[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
        out[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
        out[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight")
        out[f"visual.transformer.resblocks.{i}.attn.out_proj.bias"] = _get_by_suffix(hf_sd, f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias")

    # logit_scale is compatible
    if "logit_scale" in hf_sd:
        out["logit_scale"] = hf_sd["logit_scale"]

    return out


def _instantiate_and_load_openai_clip_from_state_dict(
    clip_module,
    state_dict: Dict[str, torch.Tensor],
    device: str,
    jit: bool,
    strict: bool,
    resolved_path: Optional[str],
) -> Tuple[torch.nn.Module, Any, ClipLoadInfo]:

    openai_sd, info = _convert_any_state_dict_to_openai(state_dict, resolved_path=resolved_path)

    # JIT is not meaningful for arbitrary external state_dicts
    if jit:
        # omitted
        jit = False

    # instantiate the *correct* CLIP variant directly from the incoming state_dict
    build_model_fn = _get_openai_build_model_fn(clip_module)
    model = build_model_fn(openai_sd).to(device)

    # in_proj -> qkv conversion if clip_module supports it (the attn** ones)
    openai_sd, applied, how = _maybe_convert_inproj_to_qkv(clip_module, openai_sd)
    if applied:
        info.inferred_signature = dict(info.inferred_signature)
        info.inferred_signature["applied_inproj_to_qkv"] = True
        info.inferred_signature["inproj_to_qkv_source"] = how

        # If we converted weights, we must reload them into the already-built model.
        missing, unexpected = model.load_state_dict(openai_sd, strict=strict)
        if (missing or unexpected) and strict:
            raise RuntimeError(f"strict load had issues. missing={missing} unexpected={unexpected}")

    # update the module-global default context length for tokenize()
    try:
        setter = getattr(clip_module, "_set_default_context_length_from_model", None)

        # Determine ctx from the built model (prefer model.context_length)
        ctx: Optional[int] = None
        try:
            if hasattr(model, "context_length"):
                ctx = int(getattr(model, "context_length"))
        except Exception:
            ctx = None

        if ctx is None:
            try:
                sd2 = model.state_dict()
                if "positional_embedding" in sd2:
                    ctx = int(sd2["positional_embedding"].shape[0])
            except Exception:
                ctx = None

        # calls into clip.py helper if present
        if callable(setter):
            setter(model)
        else:
            # fallback: best effort attribute set
            if ctx is not None:
                setattr(clip_module, "_DEFAULT_CONTEXT_LENGTH", ctx)

        if ctx is not None:
            _set_tokenize_default_context_length(clip_module, ctx)

    except Exception:
        pass

    # preprocess without relying on clip.load()
    n_px = None
    if hasattr(model, "visual") and hasattr(model.visual, "input_resolution"):
        n_px = int(model.visual.input_resolution)

    transform_fn = _get_openai_transform_fn(clip_module)
    if n_px is not None:
        if callable(transform_fn):
            preprocess = transform_fn(n_px)
        else:
            preprocess = _build_fallback_transform(n_px)
    else:
        preprocess = None  # keep as None if model has no visual tower

    if str(device) == "cpu":
        model.float()

    # ==================
    # TEMP DEBUG PRINTS 
    # ==================
    debug_very_verbose = False
    
    if debug_very_verbose:
        try:
            model_ctx = getattr(model, "context_length", None)
            if model_ctx is None:
                # infer from weights if needed
                sd3 = model.state_dict()
                model_ctx = int(sd3["positional_embedding"].shape[0]) if "positional_embedding" in sd3 else None

            global_ctx = getattr(clip_module, "_DEFAULT_CONTEXT_LENGTH", None)

            # also introspect tokenize globals (this is the thing that was silently wrong)
            tok_global_ctx = None
            try:
                tok_fn = getattr(clip_module, "tokenize", None)
                if callable(tok_fn):
                    tok_global_ctx = tok_fn.__globals__.get("_DEFAULT_CONTEXT_LENGTH", None)
                else:
                    sub = getattr(clip_module, "clip", None)
                    tok_fn2 = getattr(sub, "tokenize", None) if sub is not None else None
                    if callable(tok_fn2):
                        tok_global_ctx = tok_fn2.__globals__.get("_DEFAULT_CONTEXT_LENGTH", None)
            except Exception:
                tok_global_ctx = None

            in_res = None
            if hasattr(model, "visual") and hasattr(model.visual, "input_resolution"):
                in_res = model.visual.input_resolution

            print(
                f"[clip_anything DEBUG] built model ctx={model_ctx} | "
                f"clip_module._DEFAULT_CONTEXT_LENGTH={global_ctx} | "
                f"tokenize.__globals__._DEFAULT_CONTEXT_LENGTH={tok_global_ctx} | "
                f"device={device} | strict={strict} | resolved_path={resolved_path}"
            )
            print(
                f"[clip_anything DEBUG] preprocess={'OK' if preprocess is not None else 'None'} | "
                f"visual.input_resolution={in_res}"
            )
            if applied:
                print(f"[clip_anything DEBUG] applied inproj->qkv via: {how}")
        except Exception as _e:
            print(f"[clip_anything DEBUG] debug-print failed: {_e}")
    else:
        try:
            model_ctx = getattr(model, "context_length", None)
            if model_ctx is None:
                # infer from weights if needed
                sd3 = model.state_dict()
                model_ctx = int(sd3["positional_embedding"].shape[0]) if "positional_embedding" in sd3 else None

            global_ctx = getattr(clip_module, "_DEFAULT_CONTEXT_LENGTH", None)

            # also introspect tokenize globals (this is the thing that was silently wrong)
            tok_global_ctx = None
            try:
                tok_fn = getattr(clip_module, "tokenize", None)
                if callable(tok_fn):
                    tok_global_ctx = tok_fn.__globals__.get("_DEFAULT_CONTEXT_LENGTH", None)
                else:
                    sub = getattr(clip_module, "clip", None)
                    tok_fn2 = getattr(sub, "tokenize", None) if sub is not None else None
                    if callable(tok_fn2):
                        tok_global_ctx = tok_fn2.__globals__.get("_DEFAULT_CONTEXT_LENGTH", None)
            except Exception:
                tok_global_ctx = None

            in_res = None
            if hasattr(model, "visual") and hasattr(model.visual, "input_resolution"):
                in_res = model.visual.input_resolution

            print(Fore.BLUE + 
                f"[clip_anything DEBUG] built model ctx={model_ctx} | "
                f"clip_module._DEFAULT_CONTEXT_LENGTH={global_ctx} | "
                f"tokenize()={tok_global_ctx}" + Style.RESET_ALL
            )
            print(Fore.BLUE + 
                f"[clip_anything DEBUG] preprocess={'OK' if preprocess is not None else 'None'} | "
                f"visual.input_resolution={in_res}" + Style.RESET_ALL
            )
        except Exception as _e:
            print(Fore.RED + f"\n[clip_anything DEBUG] debug-print failed: {_e}" + Style.RESET_ALL)

    return model, preprocess, info



# OpenAI signature inference (key-based)
def _infer_openai_signature(openai_sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Follows the standard OpenAI/OpenCLIP logic: infer vit vs resnet and core dims from weights.
    """
    sig: Dict[str, Any] = {}
    vit = "visual.proj" in openai_sd
    sig["vision_is_vit"] = vit

    # text tower
    if "text_projection" in openai_sd:
        sig["embed_dim"] = int(openai_sd["text_projection"].shape[1])
    if "positional_embedding" in openai_sd:
        sig["context_length"] = int(openai_sd["positional_embedding"].shape[0])
    if "token_embedding.weight" in openai_sd:
        sig["vocab_size"] = int(openai_sd["token_embedding.weight"].shape[0])
    if "ln_final.weight" in openai_sd:
        sig["text_width"] = int(openai_sd["ln_final.weight"].shape[0])

    # number of text layers
    text_layers = _count_openai_resblocks(openai_sd, prefix="transformer.resblocks.")
    sig["text_layers"] = text_layers

    if vit:
        sig["vision_width"] = int(openai_sd["visual.conv1.weight"].shape[0])
        sig["vision_patch_size"] = int(openai_sd["visual.conv1.weight"].shape[-1])

        # count vision layers
        vis_layers = _count_openai_resblocks(openai_sd, prefix="visual.transformer.resblocks.")
        sig["vision_layers"] = vis_layers

        # infer image_size from positional embedding length (1 + grid^2)
        pe = openai_sd.get("visual.positional_embedding", None)
        if pe is not None:
            seq = int(pe.shape[0])
            grid = int(round((seq - 1) ** 0.5))
            sig["vision_grid"] = grid
            sig["image_size"] = int(grid * sig["vision_patch_size"])
    else:
        # ResNet OpenAI CLIP: infer layers tuple from state dict
        counts: List[int] = []
        for b in [1, 2, 3, 4]:
            # keys like visual.layer{b}.{i}.*
            layer_prefix = f"visual.layer{b}."
            blocks = set()
            for k in openai_sd:
                if k.startswith(layer_prefix):
                    parts = k.split(".")
                    if len(parts) >= 3:
                        blocks.add(parts[2])
            counts.append(len(blocks))
        sig["vision_layers"] = tuple(counts)
        if "visual.layer1.0.conv1.weight" in openai_sd:
            sig["vision_width"] = int(openai_sd["visual.layer1.0.conv1.weight"].shape[0])

    return sig


def _infer_openai_model_name(sig: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort mapping of inferred signature -> OpenAI CLIP model name.
    If ambiguous, returns None (caller may default to ViT-L/14).
    """
    if sig.get("vision_is_vit", False):
        patch = sig.get("vision_patch_size")
        layers = sig.get("vision_layers")
        width = sig.get("vision_width")
        image_size = sig.get("image_size")

        vit_table = [
            ("ViT-B/32", dict(patch=32, layers=12, width=768, image_size=224)),
            ("ViT-B/16", dict(patch=16, layers=12, width=768, image_size=224)),
            ("ViT-L/14", dict(patch=14, layers=24, width=1024, image_size=224)),
            ("ViT-L/14@336px", dict(patch=14, layers=24, width=1024, image_size=336)),
        ]
        for name, ref in vit_table:
            if patch == ref["patch"] and layers == ref["layers"] and width == ref["width"] and image_size == ref["image_size"]:
                return name

        # partial matching -> correct for resized pos-embeds
        for name, ref in vit_table:
            if patch == ref["patch"] and layers == ref["layers"] and width == ref["width"]:
                # if resolution differs, still prefer the closest named base
                if name == "ViT-L/14@336px" and image_size != 336:
                    continue
                return name

        return None

    # ResNet mapping (best-effort)
    layers = sig.get("vision_layers")
    embed_dim = sig.get("embed_dim")
    if layers == (3, 4, 6, 3):
        if embed_dim == 512:
            return "RN50"
        # common OpenAI scaling dims (best-effort)
        if embed_dim == 640:
            return "RN50x4"
        if embed_dim == 768:
            return "RN50x16"
        if embed_dim == 1024:
            return "RN50x64"
        return "RN50"
    if layers == (3, 4, 23, 3):
        return "RN101"
    return None


# Utilities
def _looks_like_full_model_pickle(obj: Any) -> bool:
    # If it's a torch.nn.Module instance (or behaves like one), treat as full model pickle.
    if isinstance(obj, torch.nn.Module):
        return True
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")) and not isinstance(obj, dict):
        return True
    return False


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Accepts:
      - state_dict directly
      - checkpoint dict with 'state_dict' key
      - checkpoint dict with nested keys (best effort)
    """
    if isinstance(obj, dict):
        # direct state_dict?
        if obj and all(isinstance(k, str) for k in obj.keys()) and all(torch.is_tensor(v) for v in obj.values() if v is not None):
            return obj  # type: ignore[return-value]

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if sd and all(isinstance(k, str) for k in sd.keys()):
                return sd  # type: ignore[return-value]

        # common training checkpoints
        for k in ("model", "model_state", "model_state_dict", "net", "network"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                if sd and all(isinstance(kk, str) for kk in sd.keys()):
                    return sd  # type: ignore[return-value]

    raise TypeError(Fore.RED + "\nCould not extract a state_dict-like dict[str, Tensor] from the loaded object." + Style.RESET_ALL)


def _canonicalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strips obvious wrappers: 'module.' (DDP), 'model.' etc, but only when it improves marker detection.
    """
    # First strip 'module.' if present broadly
    if sd and sum(1 for k in sd if k.startswith("module.")) > 0.8 * len(sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    # Some checkpoints store everything under 'model.'
    if sd and sum(1 for k in sd if k.startswith("model.")) > 0.8 * len(sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}

    return sd


def _canonicalize_openai_like_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Handles OpenCLIP / HF "text." prefix (CustomTextCLIP) by stripping it back to OpenAI CLIP keyspace.
    """
    if any(k.startswith("text.token_embedding.weight") for k in sd) and "token_embedding.weight" not in sd:
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("text."):
                out[k[len("text."):]] = v
            else:
                out[k] = v
        sd = out
    return sd


def _has_key_or_suffix(sd: Dict[str, torch.Tensor], suffix: str) -> bool:
    if suffix in sd:
        return True
    return any(k.endswith(suffix) for k in sd)


def _get_by_suffix(sd: Dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    # exact first
    if suffix in sd:
        return sd[suffix]
    # else suffix match (choose shortest key to avoid weird duplicates)
    matches = [k for k in sd.keys() if k.endswith(suffix)]
    if not matches:
        raise KeyError(Fore.RED + f"\nMissing required key (or suffix): {suffix}" + Style.RESET_ALL)
    best = min(matches, key=len)
    return sd[best]


def _get_by_any_suffix(sd: Dict[str, torch.Tensor], suffixes: List[str]) -> torch.Tensor:
    last_err: Optional[Exception] = None
    for s in suffixes:
        try:
            return _get_by_suffix(sd, s)
        except Exception as e:
            last_err = e
    raise KeyError(Fore.RED + f"\nMissing required key among suffixes: {suffixes}" + Style.RESET_ALL) from last_err


def _pop_by_suffix(sd: Dict[str, torch.Tensor], suffix: str) -> None:
    if suffix in sd:
        sd.pop(suffix, None)
        return
    matches = [k for k in list(sd.keys()) if k.endswith(suffix)]
    for k in matches:
        sd.pop(k, None)


def _collect_layer_indices(sd: Dict[str, torch.Tensor], prefix_suffix: str) -> List[int]:
    """
    Collects unique layer indices from keys that contain e.g.:
        text_model.encoder.layers.{i}.
        vision_model.encoder.layers.{i}.
    """
    ids = set()
    for k in sd.keys():
        # allow prefixes: something.text_model.encoder.layers.0....
        if prefix_suffix in k:
            tail = k.split(prefix_suffix, 1)[1]
            # tail starts with "{i}."
            parts = tail.split(".", 1)
            if parts and parts[0].isdigit():
                ids.add(int(parts[0]))
    return sorted(ids)

def _count_openai_resblocks(sd: Dict[str, torch.Tensor], prefix: str) -> int:
    """
    Counts resblock indices for OpenAI-style keys:
      transformer.resblocks.{i}.*
      visual.transformer.resblocks.{i}.*
    Works for any prefix that ends with ".resblocks." (or similar) by slicing.
    """
    ids = set()
    for k in sd.keys():
        if k.startswith(prefix):
            rest = k[len(prefix):]          # e.g. "0.attn.in_proj_weight"
            idx = rest.split(".", 1)[0]     # e.g. "0"
            if idx.isdigit():
                ids.add(int(idx))
    return (max(ids) + 1) if ids else 0