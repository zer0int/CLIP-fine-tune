from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import json
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from adabelief_pytorch import AdaBelief
import math
from dataclasses import asdict, fields as dataclass_fields
from typing import get_args, get_origin, Union
from colorama import Fore, Style

# ============================================================
# JSON config
# ============================================================
def _jsonify(obj: Any) -> Any:
    """Make dataclass config JSON-serializable (tuples->lists, etc.)."""
    if isinstance(obj, tuple):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        # JSON requires string keys; we stringify for safety.
        return {str(k): _jsonify(v) for k, v in obj.items()}
    return obj

def _coerce_value(tp: Any, v: Any) -> Any:
    """Coerce JSON-loaded values into the dataclass field type where feasible."""
    if v is None:
        return None

    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[T] / Union
    if origin is Union:
        # If Optional, try non-None branch
        non_none = [a for a in args if a is not type(None)]
        for a in non_none:
            try:
                return _coerce_value(a, v)
            except Exception:
                pass
        return v

    # Literal[...] -> trust JSON (no strict validation here)
    if origin is None and str(tp).startswith("typing.Literal"):
        return v

    # Containers
    if origin in (list, List):
        (et,) = args if args else (Any,)
        if isinstance(v, list):
            return [_coerce_value(et, x) for x in v]
        return v

    if origin in (dict, Dict):
        kt, vt = args if len(args) == 2 else (Any, Any)
        if isinstance(v, dict):
            out = {}
            for kk, vv in v.items():
                # JSON keys are strings; try coercion (esp. int keys)
                try:
                    ck = _coerce_value(kt, kk)
                except Exception:
                    ck = kk
                out[ck] = _coerce_value(vt, vv)
            return out
        return v

    if origin in (tuple, Tuple):
        # Tuple[T, ...]
        if len(args) == 2 and args[1] is Ellipsis:
            et = args[0]
            if isinstance(v, (list, tuple)):
                return tuple(_coerce_value(et, x) for x in v)
            return v
        # Tuple[T1, T2, ...] fixed length
        if isinstance(v, (list, tuple)) and len(args) == len(v):
            return tuple(_coerce_value(a, x) for a, x in zip(args, v))
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

    # Primitives
    try:
        if tp is int and isinstance(v, str):
            return int(v)
        if tp is float and isinstance(v, str):
            return float(v)
        if tp is bool and isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "y", "t")
    except Exception:
        pass

    return v

def _apply_cfg_overrides_from_dict(cfg: "TrainConfig", d: Dict[str, Any]) -> "TrainConfig":
    """Override cfg fields from dict (only fields present in d). Recomputes derived folders."""
    for f in dataclass_fields(cfg):
        if not f.init:
            continue  # do not override derived folders; they come from run_dir/subdirs
        if f.name in d:
            try:
                setattr(cfg, f.name, _coerce_value(f.type, d[f.name]))
            except Exception:
                # If coercion fails, fall back to raw JSON value.
                setattr(cfg, f.name, d[f.name])

    # Normalize a few common JSON->python gotchas (Dict[str, Any] fields hide types).
    if isinstance(getattr(cfg, "optimizer_kwargs", None), dict):
        b = cfg.optimizer_kwargs.get("betas", None)
        if isinstance(b, list) and len(b) == 2:
            try:
                cfg.optimizer_kwargs["betas"] = (float(b[0]), float(b[1]))
            except Exception:
                pass

    # regression_teachers keys: JSON forces strings, but your code expects int layers
    if isinstance(getattr(cfg, "regression_teachers", None), dict):
        fixed = {}
        for k, v in cfg.regression_teachers.items():
            try:
                kk = int(k)
            except Exception:
                kk = k
            fixed[kk] = v
        cfg.regression_teachers = fixed

    # Recompute derived folders (plots_folder, etc.)
    cfg.__post_init__()
    return cfg

def maybe_load_cfg_from_json(load_path: str, cfg: "TrainConfig") -> "TrainConfig":
    """Load JSON overrides if valid; otherwise warn and return cfg unchanged."""
    if not load_path:
        return cfg

    ok = (isinstance(load_path, str) and load_path.lower().endswith(".json") and os.path.isfile(load_path))
    if not ok:
        print(Fore.RED + Style.BRIGHT + f"[ConfigJSON] invalid path: {load_path!r} (using code defaults)" + Style.RESET_ALL)
        return cfg

    try:
        with open(load_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("top-level JSON must be an object/dict")
        cfg = _apply_cfg_overrides_from_dict(cfg, d)
        print(Fore.CYAN + f"[ConfigJSON] loaded overrides from: {load_path}" + Style.RESET_ALL)
        return cfg
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"[ConfigJSON] failed to load {load_path!r}: {e} (using code defaults)" + Style.RESET_ALL)
        return cfg

def save_cfg_json(cfg: "TrainConfig", out_path: str, nd: int = 6) -> None:
    """Save effective config to JSON (rounded floats, JSON-safe)."""
    try:
        d = asdict(cfg)
        d = _jsonify(d)
        d = _round_floats(d, nd=nd)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(Fore.YELLOW + f"[ConfigJSON] save failed: {e}" + Style.RESET_ALL)

def _safe_float(x: str, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        x = str(x).strip()
        if x == "":
            return default
        return float(x)
    except Exception:
        return default

def _safe_int(x: str, default: int = -1) -> int:
    try:
        if x is None:
            return default
        x = str(x).strip()
        if x == "":
            return default
        return int(x)
    except Exception:
        return default

# ================================
# reg_threshold spec handling
# ================================
def _regthr_to_str(x: Any) -> str:
    """
    Convert reg_threshold payload/log values to a stable string.
    Keeps numeric thresholds printable, and preserves "median:2.5"-style specs.
    """
    if x is None:
        return ""
    try:
        s = str(x).strip()
        return s
    except Exception:
        return ""

def _regthr_to_float_for_legacy(x: Any) -> float:
    """
    Best-effort float conversion for legacy consumers/plots.
    If x is "tag:factor", try to parse the suffix as float (factor),
    otherwise return NaN.
    """
    if x is None:
        return float("nan")
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return float("nan")
        # allow "median:2.5" -> 2.5 (legacy)
        if ":" in s:
            _, rhs = s.split(":", 1)
            rhs = rhs.strip()
            return float(rhs)
        return float(s)
    except Exception:
        return float("nan")

def load_last_logged_teacher_epoch_by_layer(teacher_log_path: str) -> Dict[int, int]:
    """
    (16-col) teacher_log.txt:
      epoch layer rebuilt reason tcos best fit_cos fit_mse n_pairs reg_threshold lam cls_mix jl_dim jl_num jl_seed jl_stride

    Returns: last_logged[layer] = max epoch seen for that layer.
    """
    if (not teacher_log_path) or (not os.path.exists(teacher_log_path)):
        return {}

    last: Dict[int, int] = {}
    with open(teacher_log_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.lower().startswith("epoch\t"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 2:
                e = _safe_int(parts[0], default=None)
                l = _safe_int(parts[1], default=None)
                if e is None or l is None:
                    continue
                last[int(l)] = max(int(e), int(last.get(int(l), -10**9)))
    return last

def load_quick_probe_history_from_log(path: str) -> Tuple[List[int], List[float], List[float]]:
    """
    quick_probe_log.txt:
      epoch\tlinear_probe_acc\tzero_shot_acc

    Returns (epochs, lp_accs, zs_accs) sorted by epoch, last-write-wins per epoch.
    """
    if (not path) or (not os.path.exists(path)):
        return [], [], []

    # epoch -> (lp, zs)
    store: Dict[int, Tuple[float, float]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.lower().startswith("epoch\t"):
                continue

            parts = ln.split("\t")
            if len(parts) < 3:
                continue

            e = _safe_int(parts[0], default=None)
            if e is None:
                continue

            lp = _safe_float(parts[1])
            zs = _safe_float(parts[2])

            store[int(e)] = (float(lp), float(zs))

    epochs = sorted(store.keys())
    lp_hist = [store[e][0] for e in epochs]
    zs_hist = [store[e][1] for e in epochs]
    return [int(e) for e in epochs], lp_hist, zs_hist


def load_tiny_benchmark_history_from_log(path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    tiny_benchmark_log.txt:
      epoch\tfolder\tn\tacc\tmean_margin\tmean_logit_correct\tmean_logit_othermax
    Returns:
      history[folder] = {"epochs":[...], "acc":[...], "margin":[...]} sorted by epoch, last-write-wins per (folder, epoch).
    """
    if (not path) or (not os.path.exists(path)):
        return {}

    # folder -> epoch -> (acc, margin)
    store: Dict[str, Dict[int, Tuple[float, float]]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("epoch"):
                continue
            parts = ln.split("\t")
            if len(parts) < 5:
                continue
            e = _safe_int(parts[0], default=None)
            if e is None:
                continue
            folder = parts[1]
            acc = _safe_float(parts[3])
            margin = _safe_float(parts[4])

            store.setdefault(folder, {})[int(e)] = (float(acc), float(margin))

    out: Dict[str, Dict[str, List[float]]] = {}
    for folder, by_epoch in store.items():
        epochs = sorted(by_epoch.keys())
        out[folder] = {
            "epochs": [int(e) for e in epochs],
            "acc":    [by_epoch[e][0] for e in epochs],
            "margin": [by_epoch[e][1] for e in epochs],
        }
    return out


def load_training_history_from_log(path: str):
    """
    training_log.txt is blocky, but we can parse the key lines you write:
      epoch=E
      train_loss=...\tval_loss=...
      logits_train_diag=...\tlogits_train_off=...
      logits_val_diag=...\tlogits_val_off=...

    Returns dict with lists:
      epoch_ids, training_losses, validation_losses,
      logits_diag_train_hist, logits_off_train_hist,
      logits_diag_val_hist, logits_off_val_hist
    """
    if (not path) or (not os.path.exists(path)):
        return {
            "epoch_ids": [],
            "training_losses": [],
            "validation_losses": [],
            "logits_diag_train_hist": [],
            "logits_off_train_hist": [],
            "logits_diag_val_hist": [],
            "logits_off_val_hist": [],
        }

    # last-write-wins by epoch
    recs: Dict[int, Dict[str, float]] = {}
    cur_epoch = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("epoch="):
                cur_epoch = _safe_int(ln.split("=", 1)[1], default=None)
                if cur_epoch is None:
                    cur_epoch = None
                    continue
                recs.setdefault(int(cur_epoch), {})
                continue

            if cur_epoch is None:
                continue

            # parse tab-separated key=val tokens
            if "\t" in ln and "=" in ln:
                for tok in ln.split("\t"):
                    tok = tok.strip()
                    if "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    recs[int(cur_epoch)][k.strip()] = _safe_float(v.strip())

    epochs = sorted(recs.keys())

    def _get_series(key: str) -> List[float]:
        return [float(recs[e].get(key, float("nan"))) for e in epochs]

    return {
        "epoch_ids": [int(e) for e in epochs],
        "training_losses": _get_series("train_loss"),
        "validation_losses": _get_series("val_loss"),
        "logits_diag_train_hist": _get_series("logits_train_diag"),
        "logits_off_train_hist": _get_series("logits_train_off"),
        "logits_diag_val_hist": _get_series("logits_val_diag"),
        "logits_off_val_hist": _get_series("logits_val_off"),
    }


def _round_floats(obj: Any, nd: int = 5) -> Any:
    """Recursively round floats for cleaner JSON logs (JSON-safe: no NaN/Inf)."""
    # accept numpy/torch scalar-ish floats
    try:
        import numpy as np
        np_floats = (np.floating,)
    except Exception:
        np_floats = tuple()

    try:
        import torch
        torch_scalar = (torch.Tensor,)
    except Exception:
        torch_scalar = tuple()

    if isinstance(obj, torch_scalar):
        if getattr(obj, "numel", lambda: 0)() == 1:
            obj = float(obj.item())
        else:
            return str(obj)

    if isinstance(obj, (float,) + np_floats):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return float(f"{v:.{nd}f}")

    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, nd) for v in obj]

    return obj

def build_optimizer_from_cfg(cfg: "TrainConfig", param_groups: List[Dict[str, Any]]):
    name = (cfg.optimizer_name or "").lower().strip()
    kw = dict(cfg.optimizer_kwargs or {})

    # resolve LR authority once, here
    any_group_has_lr = any(("lr" in g) for g in (param_groups or []))

    if name in ("adabelief", "adabelief_pytorch"):
        # Prevent clobbering: only pass 'lr' via kwargs if NO group defines an lr.
        if any_group_has_lr:
            kw.pop("lr", None)  # do not allow global lr to override per-group lrs
        else:
            # Ensure groups have an lr (use optimizer_kwargs['lr'] as global default)
            lr = float(kw.pop("lr", 1e-3))
            for g in param_groups:
                g["lr"] = lr

        return AdaBelief(param_groups, **kw)

    try:
        import torch.optim as optim
        cls = getattr(optim, cfg.optimizer_name)
    except Exception as e:
        raise SystemExit(f"[Abort] Unknown optimizer_name={cfg.optimizer_name!r}: {e}")

    # Torch optimizers usually behave, keep the same “single-source” policy:
    if any_group_has_lr:
        kw.pop("lr", None)  # avoid ambiguity; per-group wins
    else:
        lr = float(kw.pop("lr", 1e-3))
        for g in param_groups:
            g["lr"] = lr

    return cls(param_groups, **kw)


def build_scheduler_from_cfg(cfg: "TrainConfig", optimizer):
    name = (cfg.scheduler_name or "").strip()
    if name == "" or name.lower() in ("none", "null", "no", "disabled"):
        return None

    kw = dict(cfg.scheduler_kwargs or {})
    if name == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(optimizer, **kw)

    try:
        import torch.optim.lr_scheduler as lrs
        cls = getattr(lrs, name)
    except Exception as e:
        raise SystemExit(f"[Abort] Unknown scheduler_name={cfg.scheduler_name!r}: {e}")

    return cls(optimizer, **kw)

def _append_teacher_log_rows(
    teacher_log_path: str,
    epoch_i: int,
    rebuilt_by_layer: Dict[int, int],
    reason_by_layer: Dict[int, str],
    payload_by_layer: Dict[int, Dict[str, Any]],
    last_logged_epoch_by_layer: Optional[Dict[int, int]] = None,
    nd: int = 5,
):
    # auto-load guard map if caller didn't pass it
    if last_logged_epoch_by_layer is None:
        last_logged_epoch_by_layer = load_last_logged_teacher_epoch_by_layer(teacher_log_path)

    with open(teacher_log_path, "a", encoding="utf-8") as f:
        for layer, payload in payload_by_layer.items():
            layer = int(layer)

            # skip duplicates (one row per layer per epoch)
            last_e = last_logged_epoch_by_layer.get(layer, None)
            if last_e is not None and int(epoch_i) <= int(last_e):
                continue

            tcos = payload.get("tcos", None)
            best = payload.get("best", None)
            fit_cos = payload.get("fit_cos", None)
            fit_mse = payload.get("fit_mse", None)
            n_pairs = int(payload.get("n", 0))

            # reg_threshold logging stays numeric
            # payload["reg_threshold"] may be:
            #   - "70.0"  (absolute)
            #   - "median:2.5" (adaptive; log just 2.5)
            reg_thr_str = _regthr_to_str(payload.get("reg_threshold", ""))
            reg_thr_num = _regthr_to_float_for_legacy(reg_thr_str)
            reg_thr_out = "" if (not math.isfinite(float(reg_thr_num))) else str(round(float(reg_thr_num), nd))

            lam = float(payload.get("lam", float("nan")))
            cls_mix = float(payload.get("cls_mix", 0.0))

            jl = payload.get("jl", {}) or {}
            jl_dim = int(jl.get("dim", 0) or 0)
            jl_num = int(jl.get("num_proj", 0) or 0)
            jl_seed = int(jl.get("seed", 0) or 0)
            jl_stride = int(jl.get("seed_stride", 0) or 0)

            f.write(
                f"{int(epoch_i)}\t{layer}\t{int(rebuilt_by_layer.get(layer,0))}\t{reason_by_layer.get(layer,'')}\t"
                f"{'' if tcos is None else round(float(tcos), nd)}\t"
                f"{'' if best is None else round(float(best), nd)}\t"
                f"{'' if fit_cos is None else round(float(fit_cos), nd)}\t"
                f"{'' if fit_mse is None else round(float(fit_mse), nd)}\t"
                f"{n_pairs}\t{reg_thr_out}\t{round(lam, nd)}\t{round(cls_mix, nd)}\t"
                f"{jl_dim}\t{jl_num}\t{jl_seed}\t{jl_stride}\n"
            )

            # update in-memory guard
            last_logged_epoch_by_layer[layer] = int(epoch_i)



def load_teacher_history_from_log(teacher_log_path: str) -> Dict[int, Dict[str, List[float]]]:
    """
    Returns:
      hist[layer] = {"epochs": [...], "tcos": [...], "fit_cos": [...], "fit_mse": [...]}

    Supports:
      - 16-col flattened format (preferred)
      - OLD 5-col JSON-in-cell
      - OLD 4-col JSON-in-cell
    Last-write-wins per (layer, epoch).

    NOTE: reg_threshold is now allowed to be a string spec; this function ignores it.
    """
    if (not teacher_log_path) or (not os.path.exists(teacher_log_path)):
        return {}

    # temp[layer][epoch] = (tcos, fit_cos, fit_mse)
    temp: Dict[int, Dict[int, Tuple[float, float, float]]] = {}

    def _to_float(x: Any) -> float:
        try:
            if x is None:
                return float("nan")
            return float(x)
        except Exception:
            return float("nan")

    with open(teacher_log_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.lower().startswith("epoch\t"):
                continue

            parts = ln.split("\t")

            # --- NEW 16-col format ---
            if len(parts) >= 16:
                epoch_i = _safe_int(parts[0], default=None)
                layer_i = _safe_int(parts[1], default=None)
                if epoch_i is None or layer_i is None:
                    continue
                tcos = _safe_float(parts[4])
                fit_cos = _safe_float(parts[6])
                fit_mse = _safe_float(parts[7])
                temp.setdefault(int(layer_i), {})[int(epoch_i)] = (float(tcos), float(fit_cos), float(fit_mse))
                continue

            # --- OLD 5-col JSON format ---
            if len(parts) == 5:
                epoch_s, _rebuilt_s, layer_s, _reason, j = parts
                try:
                    epoch_i = int(epoch_s)
                    layer_i = int(layer_s)
                except Exception:
                    continue
                try:
                    payload = json.loads(j)
                except Exception:
                    continue
                tcos = _to_float(payload.get("tcos"))
                fit_cos = _to_float(payload.get("fit_cos"))
                fit_mse = _to_float(payload.get("fit_mse"))
                temp.setdefault(layer_i, {})[epoch_i] = (tcos, fit_cos, fit_mse)
                continue

            # --- OLD 4-col JSON format ---
            if len(parts) == 4:
                epoch_s, _rebuilt_s, _reason, j = parts
                try:
                    epoch_i = int(epoch_s)
                except Exception:
                    continue
                try:
                    payload = json.loads(j)
                except Exception:
                    continue
                if "teachers" in payload and isinstance(payload["teachers"], dict):
                    payload = payload["teachers"]
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        try:
                            layer_i = int(k)
                        except Exception:
                            continue
                        if not isinstance(v, dict):
                            continue
                        tcos = _to_float(v.get("tcos"))
                        fit_cos = _to_float(v.get("fit_cos"))
                        fit_mse = _to_float(v.get("fit_mse"))
                        temp.setdefault(layer_i, {})[epoch_i] = (tcos, fit_cos, fit_mse)

    hist: Dict[int, Dict[str, List[float]]] = {}
    for layer_i, by_epoch in temp.items():
        epochs = sorted(by_epoch.keys())
        hist[layer_i] = {
            "epochs": [int(e) for e in epochs],
            "tcos": [by_epoch[e][0] for e in epochs],
            "fit_cos": [by_epoch[e][1] for e in epochs],
            "fit_mse": [by_epoch[e][2] for e in epochs],
        }
    return hist


def load_teacher_state_from_log(teacher_log_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Reconstruct per-layer state:
      state[layer] = {
        "last_epoch": int,
        "last_rebuild_epoch": int or None,
        "best_tcos": float or None,
        "last_tcos": float or None,
      }

    Supports NEW 16-col format (preferred), and OLD JSON formats as fallback.

    NOTE: reg_threshold is now allowed to be a string spec; this function ignores it.
    """
    if (not teacher_log_path) or (not os.path.exists(teacher_log_path)):
        return {}

    state: Dict[int, Dict[str, Any]] = {}

    def _upd(layer_i: int, epoch_i: int, rebuilt_i: int, tcos: Optional[float]):
        st = state.setdefault(int(layer_i), {
            "last_epoch": None,
            "last_rebuild_epoch": None,
            "best_tcos": None,
            "last_tcos": None,
        })
        st["last_epoch"] = int(epoch_i)
        st["last_tcos"] = (None if tcos is None or (not math.isfinite(float(tcos))) else float(tcos))
        if int(rebuilt_i) == 1:
            st["last_rebuild_epoch"] = int(epoch_i)
        if st["last_tcos"] is not None:
            if (st["best_tcos"] is None) or (st["last_tcos"] > st["best_tcos"]):
                st["best_tcos"] = st["last_tcos"]

    with open(teacher_log_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.lower().startswith("epoch\t"):
                continue
            parts = ln.split("\t")

            # --- NEW 16-col format ---
            if len(parts) >= 16:
                epoch_i = _safe_int(parts[0], default=None)
                layer_i = _safe_int(parts[1], default=None)
                rebuilt_i = _safe_int(parts[2], default=0)
                if epoch_i is None or layer_i is None:
                    continue
                tcos = _safe_float(parts[4], default=float("nan"))
                tcos = None if (not math.isfinite(float(tcos))) else float(tcos)
                _upd(int(layer_i), int(epoch_i), int(rebuilt_i), tcos)
                continue

            # --- OLD 5-col JSON format ---
            if len(parts) == 5:
                epoch_s, rebuilt_s, layer_s, _reason, j = parts
                try:
                    epoch_i = int(epoch_s)
                    rebuilt_i = int(rebuilt_s)
                    layer_i = int(layer_s)
                    payload = json.loads(j)
                except Exception:
                    continue
                tcos = payload.get("tcos", None)
                try:
                    tcos = float(tcos) if tcos is not None and math.isfinite(float(tcos)) else None
                except Exception:
                    tcos = None
                _upd(layer_i, epoch_i, rebuilt_i, tcos)
                continue

    # sanitize
    for layer_i in list(state.keys()):
        if state[layer_i].get("last_epoch", None) is None:
            del state[layer_i]
    return state
