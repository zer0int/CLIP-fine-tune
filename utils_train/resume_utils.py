from __future__ import annotations

import os
import torch
import random
from typing import List, Tuple, Optional
from colorama import Fore, Style
from torch import nn

def import_clip_module(use_ko_config: bool):
    if use_ko_config:
        import gmpclipheaddropout as clip_mod
    else:
        import gmpclipregression as clip_mod
    return clip_mod

def _list_pt_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".pt")])

def _resume_paths(cfg: TrainConfig) -> Tuple[str, str]:
    bundle_path = os.path.join(cfg.optimizer_state_folder, cfg.resume_bundle_name)
    model_path = os.path.join(cfg.optimizer_state_folder, cfg.resume_model_name)
    return bundle_path, model_path

def _init_log_file(path: str, header: str, continue_run: bool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if continue_run:
        # append; if file doesn't exist yet, create + write header
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)

def _prompt_yes_no(msg: str, default_yes: bool = False) -> bool:
    default = "Y" if default_yes else "N"
    suffix = "[Y/n]" if default_yes else "[y/N]"
    ans = input(f"{msg} {suffix} ").strip().lower()
    if ans == "":
        return default_yes
    return ans.startswith("y")

def _read_last_nonheader_line(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("epoch\t"):
                continue
            last = ln
    return last

def decide_resume_or_overwrite(cfg: TrainConfig) -> Tuple[bool, str]:
    """
    Returns (continue_run, reason).
    continue_run=True => resume from bundle
    continue_run=False => start new/overwrite
    """
    bundle_path, model_path = _resume_paths(cfg)
    have_bundle = os.path.exists(bundle_path)
    have_ckpts = len(_list_pt_files(cfg.ft_checkpoints_folder)) > 0

    if cfg.resume_policy == "always_overwrite":
        if have_bundle or have_ckpts:
            return False, "policy=always_overwrite"
        return False, "fresh"

    if cfg.resume_policy == "abort_if_exists":
        if have_bundle or have_ckpts:
            raise SystemExit(f"[Abort] Existing state detected (bundle={have_bundle}, ckpts={have_ckpts}) under resume_policy=abort_if_exists.")
        return False, "fresh"

    if cfg.resume_policy == "always_resume":
        if have_bundle:
            return True, "policy=always_resume"
        raise SystemExit("[Abort] resume_policy=always_resume but no resume bundle found.")

    # prompt
    if have_bundle:
        print(Fore.RED + "\n------------ ATTENTION! ------------"  + Style.RESET_ALL)
        do_resume = _prompt_yes_no(
            f"[Resume] Found resume bundle at:\n  {bundle_path}\nContinue from it ('no' overwrites; hit CTRL+C to abort)?",
            default_yes=True
        )
        return (do_resume, "prompt_bundle_resume" if do_resume else "prompt_bundle_overwrite")

    if (not have_bundle) and have_ckpts:
        do_overwrite = _prompt_yes_no(
            f"[Warning] Found checkpoints in:\n  {cfg.ft_checkpoints_folder}\nBut no resume bundle.\nOVERWRITE and start fresh?",
            default_yes=False
        )
        if do_overwrite:
            return False, "prompt_ckpt_overwrite"
        raise SystemExit("[Abort] User aborted to avoid overwriting existing checkpoints.")

    return False, "fresh"

def save_resume_state(
    cfg: TrainConfig,
    epoch_idx: int,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    ema: Optional["EMAState"],
    global_optim_step: int,
):
    """
    saves a full-model pickle + a separate resume bundle dict
    - latest_model.pt: torch.save(model) (full pickle)
    - latest_resume.pt: optimizer/scheduler/scaler/RNG/epoch + model_state_dict fallback
    """
    os.makedirs(cfg.optimizer_state_folder, exist_ok=True)
    bundle_path, model_path = _resume_paths(cfg)

    # Full model pickle
    torch.save(model, model_path)

    # Fallback state_dict (so resume still works if pickle load breaks)
    model_sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    bundle = {
        "epoch_idx": int(epoch_idx),
        "global_optim_step": int(global_optim_step),
        "use_ko_config": bool(cfg.use_ko_config),
        "clipmodel": str(cfg.clipmodel),

        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,

        "ema": (ema.state_dict_cpu() if (ema is not None) else None),

        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "py_random_state": random.getstate(),

        "model_pickle_path": model_path,
        "model_state_dict_cpu": model_sd_cpu,
    }
    torch.save(bundle, bundle_path)

def load_resume_bundle(cfg: TrainConfig, device: str) -> dict:
    bundle_path, _ = _resume_paths(cfg)
    return torch.load(bundle_path, map_location=device)

def try_load_model_from_resume(bundle: dict, device: str) -> Optional[nn.Module]:
    mp = bundle.get("model_pickle_path", None)
    if mp is None or (not os.path.exists(mp)):
        return None
    try:
        m = torch.load(mp, map_location=device)
        return m
    except Exception as e:
        print(Fore.YELLOW + f"[Resume] torch.load(full model) failed: {e} — will fall back to state_dict." + Style.RESET_ALL)
        return None

def _coerce_rng_state_to_uint8(x):
    # torch.set_rng_state requires uint8 (ByteTensor)
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        # CPU, uint8
        return x.detach().to(device="cpu", dtype=torch.uint8)

    if isinstance(x, (bytes, bytearray)):
        return torch.tensor(list(x), dtype=torch.uint8)

    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.uint8)

    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                x = x.astype(np.uint8, copy=False)
            return torch.from_numpy(x).to(dtype=torch.uint8, device="cpu")
    except Exception:
        pass

    raise TypeError(f"Unsupported RNG state type: {type(x)}")