"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
________________________________________________
"""
from __future__ import annotations
import os
import json
import torch
import random
import gc
from dataclasses import dataclass, field
from colorama import Fore, Style
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from adabelief_pytorch import AdaBelief
from torchvision import transforms
from typing import List, Tuple, Optional, Dict, Any, Literal, Callable
import pandas as pd
import inspect
from PIL import Image

import warnings # stop especially torch spam
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ResumePolicy = Literal["prompt", "always_resume", "always_overwrite", "abort_if_exists"]
SchedulerStepMode = Literal["fractional_epoch", "per_optim", "per_epoch"]

from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything # loads HF HUB CLIP, Long-CLIP, OpenAI-CLIP, local safetensors.

from utils_train.config_utils import _round_floats, build_optimizer_from_cfg, build_scheduler_from_cfg, _append_teacher_log_rows, load_teacher_history_from_log
from utils_train.config_utils import save_cfg_json, maybe_load_cfg_from_json, _apply_cfg_overrides_from_dict, _coerce_value, _jsonify
from utils_train.heuristic_dataset import HeuristicDatasetSpec, build_heuristic_dataset
from utils_train.dataset_only import ImageTextDataset, BalancedImageFolderTeacher, CroppedImageCSVFileDataset, AdversarialTripletTextDataset, infinite_dataloader
from utils_train.probe_utils import run_quick_probe, run_tiny_benchmark, _call_with_signature, print_grad_status_summary
from utils_train.plot_utils import plot_gradient_norms, plot_training_info, plot_teacher_info, plot_probe_info, plot_tiny_benchmark
from utils_train.metrics_utils import calculate_metrics, compute_gapness_metrics, monitor_gradient_norms, _get_model_dtype, _get_image_dtype, _assert_vit_visual_or_abort
from utils_train.saver_ema_utils import EMAState, ModelSaver
from utils_train.resume_utils import import_clip_module, _list_pt_files, _resume_paths, _init_log_file, _prompt_yes_no, _read_last_nonheader_line
from utils_train.resume_utils import decide_resume_or_overwrite, save_resume_state, load_resume_bundle, try_load_model_from_resume, _coerce_rng_state_to_uint8
from utils_train.losses_only import BaseContrastiveLoss, GapContrastiveLoss, apply_gap_loss_schedule, k_proj_orthogonality_loss, decatt_loss
from utils_train.losses_only import regression_consistency_loss, geometry_preserving_loss
from utils_train.teacher_utils import _phase_banner, _subphase, build_global_embeddings_from_tokens, encode_image_tokens_after_block, should_rebuild_teacher
from utils_train.teacher_utils import FixedJLProjector, JLProjectorEnsemble, _normalize_jl_spec, make_jl_projector_for_teacher, build_cls_patch_regression_teacher
from utils_train.teacher_utils import evaluate_teacher_cosine, resolve_regression_teacher_specs, make_teacher_cache_paths, compute_cls_patch_embeddings
from utils_train.grad_groups_utils import _resolve_block_indices, get_visual_block_params, get_text_block_params, get_visual_from_params, get_text_from_params 
from utils_train.grad_groups_utils import _dedupe_params, _param_ids_from_groups, apply_param_groups_trainability, build_param_groups_automatic




"""
 ┌─────────────────────────────────────────────────────────────────────────────────────┐
 │ CONFIG (.json will be saved with model automatically -> load here manually.)        │
 │ Else, the TrainConfig below applies. PS: you'll be prompted to resume optimizer     │
 │ saved state (if applicable) -> so you *can* e.g. extend 20 epochs -> 30 epochs!     │
 └─────────────────────────────────────────────────────────────────────────────────────┘
"""
load_json_config: str = ""  #  <<< Set to a .json path -> override TrainConfig defaults


@dataclass
class TrainConfig:
    # ------------------------------------------------------------
    # Run root / path config
    # ------------------------------------------------------------
    run_dir: str = "CLIP-fine-tune" # Root folder, models and logs get saved here       <--- SET THIS

    # Derived folders (AUTO)
    plots_folder: str = field(init=False)
    ft_checkpoints_folder: str = field(init=False)
    text_logs_folder: str = field(init=False)
    teacher_folder: str = field(init=False)
    optimizer_state_folder: str = field(init=False)

    # Subfolder names
    plots_subdir: str = "ft-plots"
    ckpt_subdir: str = "ft-checkpoints"
    logs_subdir: str = "ft-logs"
    teacher_subdir: str = "ft-teachers"
    optim_subdir: str = "ft-optim"

    def __post_init__(self):
        self.plots_folder = os.path.join(self.run_dir, self.plots_subdir)
        self.ft_checkpoints_folder = os.path.join(self.run_dir, self.ckpt_subdir)
        self.text_logs_folder = os.path.join(self.run_dir, self.logs_subdir)
        self.teacher_folder = os.path.join(self.run_dir, self.teacher_subdir)
        self.optimizer_state_folder = os.path.join(self.run_dir, self.optim_subdir)

    # ------------------------------------------------------------
    # Grad / Param-group presets    
    # Mutually exclusive, in order of priority:
    # grad_set_manual > grad_full_text_vit > grad_vit_full > grad_text_full
    # ------------------------------------------------------------
    # Manual mode: You set the parameters and LR/param. -> JUMP to settings: CTRL+F for GOTO
    grad_set_manual: bool = False    # 0) manual mode
    # Auto-mode (exclusive): 
    grad_full_text_vit: bool = True  # 1) all of CLIP
    grad_vit_full: bool = False      # 2) ViT only
    grad_text_full: bool = False     # 3) Text only
    # These can be *combined*:
    grad_vit_from: Optional[int] = None      # 4) ViT from block idx (inclusive) to end (+ post)         None, [0-23]
    grad_text_from: Optional[int] = None     # 5) Text from block idx. For penultimate + final -> = 10   None, [0-11]

    # Debug prints
    grad_debug_print: bool = True           # Get a occasional friendly red warning if a gradient explodes / vanishes

    # ------------------------------------------------------------
    # Model ~ SUPPORTS Long-CLIP, too! Replace "ViT-L/14" with e.g. zer0int/CLIP-GmP-ViT-L-14 or zer0int/LongCLIP-GmP-ViT-L-14
    # ------------------------------------------------------------
    clipmodel: str = "ViT-L/14"     # or /path/to/your/model.pt, or HuggingFace, e.g. openai/clip-vit-large-patch14
    epochs: int = 20                # Total Epochs to train (10-30 usually works)
    batch_size: int = 24            # watch your VRAM -> memory log is in 'run_dir'!
    accumulation_steps: int = 2     # gradient accumulation ('fake larger batch size')
    
    show_losses_every: int = 200   # n batches: How often to show more losses (adversarial, teacher, etc.)

    # ------------------------------------------------------------
    # CUSTOM dataset (overrides COCO SPRIGHT below)
    # Uses HEURISTICS! (I'll figure out what your dataset is)
    # Minimum:
    # Provide a local path or HF dataset --> path_or_hf_dataset 
    # - HF dataset with >1 text label: I'll prompt / ask you which labels to use.
    # - Local path: I'll try and find images + labels
    # -> Finds: .txt sidecars, .tsv, .csv, .json, parquet, .mat
    #
    # SPRIGHT (my main dataset) (HUGE! ~40k) : SPRIGHT-T2I/spright_coco
    # ------------------------------------------------------------
    use_custom_dataset: bool = True                   # <---- Set True to use *your* dataset (disables COCO below)

    # If use_custom_dataset=True, specify TRAIN and VAL datasets here:
    custom_dataset_train: Dict[str, Any] = field(default_factory=lambda: {  # dataset you want to train the model on
        "path_or_hf_dataset": "lianghsun/pokemon-blip-captions-en-zh_tw",   # <--- random example of a HF dataset :)
        "text_labels_path": None,
        "split": "train",
        "image_root": None,
        "deterministic_text": False,
    })
    custom_dataset_val: Dict[str, Any] = field(default_factory=lambda: {    # <--- set 'validation' split here
        "path_or_hf_dataset": "lianghsun/pokemon-blip-captions-en-zh_tw",   # used for val, not training
        "text_labels_path": None,
        "split": "val",
        "image_root": None,
        "deterministic_text": True,
    })

    # ------------------------------------------------------------
    # Data (COCO SPRIGHT) ~ zer0int's standard CLIP dataset
    # Recommended: Download it first, THEN use here in coco:
    # https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco
    # But if you set use_custom_dataset: bool = True above,
    # you can also just put this HF hub ID into custom dataset above:
    # SPRIGHT-T2I/spright_coco
    # ------------------------------------------------------------    
    coco_root: str = "path/to/COCO-SPRIGHT/images"
    coco_train_json: str = "utils_datasets/coco_spright/short-coco-spright-train-0_9.json" # included with this repo
    coco_val_json: str = "utils_datasets/coco_spright/short-coco-spright-val-10_11.json" # included with this repo

    # ------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------
    optimizer_name: str = "AdaBelief"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 3e-6,                 # <------ Set *GLOBAL* learning rate here (auto mode)               <--- Set LR
        "eps": 1e-16,               # can be overridden in set_manual_param_groups (manual mode) 
        "betas": (0.9, 0.998),      # 1e-7 to 1e-5 can work. Check run_dir -> gradient norms logs!
        "weight_decay": 1e-3,       # Rule of thumb: Larger batch size => Higher learning rate.
        "weight_decouple": True,
        "rectify": True,
        "print_change_log": False,
    })

    scheduler_name: str = "CosineAnnealingWarmRestarts"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "T_0": 1,
        "T_mult": 1,
        "eta_min": 1e-8
    })
    # "fractional_epoch", "per_optim", "per_epoch"
    # -- Strongly recommended to just leave as-is.
    scheduler_step_mode: SchedulerStepMode = "fractional_epoch"

    # ------------------------------------------------------------
    # Contrastive loss controls: Label smoothing (0.0 disables)
    # ------------------------------------------------------------
    contrastive_smoothing: float = 0.1

    # ------------------------------------------------------------
    # Model additional settings
    # ------------------------------------------------------------
    # Model saving
    model_save_full: bool = True                    # Leave as-is and use all-pytorch-model-convert-to-huggingface.py
    model_save_dict: bool = False                   # after training for state_dict, safetensors, text encoder, etc.
    save_checkpoints_fp16: bool = True

    # Optional EMA Model (RAM-backed)
    use_ema: bool = False                           # <----- EMA 'smooth updates to weights' model (in RAM, not VRAM)    
    ema_decay_step: float = 0.9999                  # If your training has spiky gradients, this model may be better
    ema_update_every_n_optim_steps: int = 10        # It will lag behind with progress, though (that's the whole point)
    ema_store_dtype_fp16: bool = False
    save_ema_checkpoint: bool = True

    # ------------------------------------------------------------
    # Resume from checkpoint / overwrite behavior
    # ------------------------------------------------------------
    save_resume_state: bool = True                  # <----- if True, you can interrupt & continue training
    resume_policy: ResumePolicy = "prompt"          # "prompt" | "always_resume" | "always_overwrite" | "abort_if_exists"
    resume_bundle_name: str = "latest_resume.pt"    # optimizer/scheduler/scaler/RNG/epoch
    resume_model_name: str = "latest_model.pt"      # full model pickle via torch.save(model)

    # ------------------------------------------------------------
    # Workers / Threading
    # ------------------------------------------------------------
    num_workers: int = 4                            # choose based num CPU cores available. 2-8 is often good.
    persistent_workers: bool = True
    prefetch_factor: int = 2

    log_cuda_memory: bool = True
    cuda_empty_cache_each_epoch: bool = False       # is usually auto-handled fine (True just slows stuff down)

    # ------------------------------------------------------------
    # Adversarial triplet injection ~~ dataset included w/ repo
    # ------------------------------------------------------------
    use_adv_triplet: bool = False                   # <----- False for classic training
    adv_inject_every: int = 25
    adv_lambda: float = 0.1
    adv_batch_size: int = 1
    adv_annotations_file: str = "image_sets/adv_dataset/adversarial_labels_min_cos_sim.json"
    adversarial_log_name: str = "adversarial_log.txt"

    # ------------------------------------------------------------
    # KO extra losses
    # ------------------------------------------------------------
    use_ko_config: bool = False                     # <----- False for classic training
    
    ko_decatt_lambda: float = 1.0
    ko_kproj_lambda: float = 10.0
    ko_layers_decatt: Tuple[int, ...] = (0, 1, 2, 9, 10, 11, 12, 13, 14)
    ko_layers_kproj_ortho: Tuple[int, ...] = (8, 9, 10, 11, 12)
    ko_log_name: str = "ko_losses_log.txt"

    # ------------------------------------------------------------
    # Gap Contrastive Loss / Gap schedule
    # 'It’s Not a Modality Gap: Characterizing and Addressing the Contrastive Gap'
    # https://arxiv.org/abs/2405.18570v1
    # ------------------------------------------------------------
    use_gap_schedule: bool = False                  # <----- False for classic training
    
    gap_w_uniform: float = 0.0025                   # 'close modality gap' + 'nudge into ellipsoid' (due to InfoNCE loss)
    gap_w_align: float = 0.0    # try 0.00025       # DANGER 'close modality gap' +  ^^^ both: risk of embeddings collapse
    gap_w_xuniform: float = 0.00025                 # 'widen modality gap' + 'nudge to hypersphere' (vs ellipsoid)
    gap_uniform_start_epoch: int = 0
    gap_align_start_epoch: int = 0
    gap_xuniform_start_epoch: int = 0

    # ------------------------------------------------------------
    # Regression Consistency Teachers
    # Broadly inspired by / adapted from:
    # 'Register and [CLS]tokens induce a decoupling of local and global features in large ViTs'
    # https://arxiv.org/abs/2505.05892v2
    # ------------------------------------------------------------
    use_regression_teachers: bool = False        # <----- False for classic training

    geometry_preserving_lam: float = 0.0        # try 1e-4 to 0.01; set 0.0 to disable
    geometry_preserving_on: str = "img"         # "img" | "txt" | "both"    (very experimental, maybe leave disabled)

    regression_teachers: Dict[int, Dict[str, Any]] = field(
        default_factory=lambda: {
            21: {
                "reg_threshold": 60.0,
                "lam": 0.05,
                "cls_mix": 0.30,
                "cls_mix_use_reg": True,
                "is_reg_teacher": False,
                "use_reg_whitening": False,
                "jl": {"enabled": True, "dim": 128, "seed": 1234, "num_proj": 8, "seed_stride": 1},
            },
            22: {
                "reg_threshold": "median:2.5",      # <---- if "median:factor", we use that -> median_norm*factor <- as threshold instead of
                "lam": 0.05,                        # absolute; "reg_threshold": 60.0 or "reg_threshold": "60.0" <- this will be *absolute*.
                "cls_mix": 0.30,                    # -> if norms slip during train, but registers are still global info, *median* will catch it.
                "cls_mix_use_reg": True,
                "is_reg_teacher": False,
                "use_reg_whitening": False,
                "jl": {"enabled": False, "dim": 128, "seed": 1234, "num_proj": 8, "seed_stride": 1},
            },
            23: {
                "reg_threshold": 70.0,
                "lam": 0.15,
                "cls_mix": 1.00,
                "cls_mix_use_reg": False,
                "is_reg_teacher": False,
                "use_reg_whitening": False,
                "jl": {"enabled": False, "dim": 128, "seed": 1234, "num_proj": 8, "seed_stride": 1},
            },
        }
    )  
    """
     ┌──────────────────────────────────────────────────────────────────────────┐
     │ ----> For my Regression Fine-Tune, I trained 3 separate runs:            │
     │ 1. Classic Training, on COCO, 20 Epochs, full model (TE & ViT)           │
     │ 2. ViT only: full, with above teachers, with use_adv_triplet=True        │
     │ 3. Only last blocks: Text: 10,11 -- ViT: 20,21,22,23 - with teachers     │
     └──────────────────────────────────────────────────────────────────────────┘
    """
    # ------------------------------------------------------------
    # Teacher extra images (ImageNet)
    # https://www.image-net.org/download.php
    # ------------------------------------------------------------
    use_imagenet: bool = False                   # <----- Set to False if not available -> will use your 'val' dataset.
    imagenet_max_per_class: int = 3
    teacher_seed: int = 42
    imagenet_train_root: str = r"path/to/ILSVRC2012/train"
    imagenet_val_root: str = r"path/to/ILSVRC2012/val"

    # ------------------------------------------------------------
    # Teacher training / update policy (staleness)
    # ------------------------------------------------------------
    cls_patch_max_samples: int = 50000
    cls_patch_val_frac: float = 0.1
    teacher_min_epoch: int = 0
    teacher_update_cooldown: int = 2
    teacher_abs_cos_min: float = 0.97
    teacher_cos_drop: float = 0.01
    teacher_retrain_every_epoch: bool = True        # <----- True for rebuilding every epoch (recommended)
    teacher_early_retrain_epochs: int = 0
    teacher_log_name: str = "teacher_log.txt"
    
    teacher_rebuild_epochs: List[int] = field(default_factory=lambda: [])
    teacher_rebuild_epochs_mode: Literal["only", "extra"] = "only"
    teacher_cache_name_fmt: str = "cls_patch_teacher_val_b{layer}.pt"
    
    # ------------------------------------------------------------
    # Logs, Probe
    # Dataset download for quick_probe (or just disable it below)
    # (free, no sign-up): https://objectnet.dev/mvt/
    # ------------------------------------------------------------
    grad_log_every_n_steps: int = 50
    monitor_grad_every_n_steps: int = 100
    grad_plot_topk: int = 40

    run_quick_probe: bool = False                # <----- True for benchmarking on-the-fly (needs dataset below!)
    probe_every_n_epochs: int = 1
    probe_batch_size: int = 50
    probe_num_workers: int = 4
    probe_max_iter: int = 1000
    probe_csv: str = "utils_datasets/mvt/human_responses-mini.csv"
    probe_image_dir: str = "path/to/dataset-difficulty-CLIP/data_release_2023/all/"

    run_tiny_benchmark: bool = False            # <----- True for adversarial attack benchmarking at val, too
    tiny_batch_size: int = 64
    tiny_dataset: str = "n01531178" # included with this repo
    tiny_choices: Tuple[str, str, str] = ("a photo of a bird", "a photo of a bumblebee", "a photo of a text")
    tiny_folders: Tuple[str, str] = ("image_sets/{dataset}", "image_sets/{dataset}_adv")    
    #_________________________________________________________________________________________________________ END


# ============================================================
# Manual param groups (user-editable) # GOTO :)
# ============================================================
def set_manual_param_groups(model, cfg):
    """
    ViT (visual) [ViT-L/14: 0-23]:
      Pre-block:
        - model.visual.class_embedding                (Parameter)
        - model.visual.positional_embedding           (Parameter)
        - model.visual.conv1                          (nn.Conv2d -> use .parameters())
        - model.visual.ln_pre                         (LayerNorm -> use .parameters())
      Blocks:
        - model.visual.transformer.resblocks[i]       (nn.Module -> use .parameters())
      Post-block:
        - model.visual.ln_post                        (LayerNorm -> use .parameters())
        - model.visual.proj                           (Parameter)

    Text: [ViT-L/14: 0-11]
      Pre-block:
        - model.token_embedding                       (nn.Embedding -> use .parameters())
        - model.positional_embedding                  (Parameter)
      Blocks:
        - model.transformer.resblocks[i]              (nn.Module -> use .parameters())
      Post-block:
        - model.ln_final                              (LayerNorm -> use .parameters())
        - model.text_projection                       (Parameter)
        - model.logit_scale                           (Parameter)
    """
    from utils_train.grad_groups_utils import get_visual_block_params, get_text_block_params

    visual_list = [18,19,20,21,22,23]   # ViT blocks
    text_list = [10,11]                 # Text blocks / PS:  -> [-1] or [11] = same difference

    visual_parameters = get_visual_block_params(model, visual_list)
    text_parameters = get_text_block_params(model, text_list)

    # Example for last blocks of both encoders, with custom learning rates:
    param_groups = [
        # --- ViT-blocks from visual_list ---
        {"params": [p for p in visual_parameters], "lr": 1e-6},
        # --- ViT-'post' ---   
        {"params": [model.visual.ln_post.weight, model.visual.ln_post.bias], "lr": 8e-7},
        {"params": [model.visual.proj], "lr": 1e-6},
        # --- Text-blocks from text_list ---
        {"params": [p for p in text_parameters], "lr": 1e-7},   # <--- remove "lr" to use global from optimizer_kwargs.
        # --- Text-'post' ---
        {"params": [model.ln_final.weight, model.ln_final.bias], "lr": 1e-7},
        {"params": [model.text_projection], "lr": 1e-7},
        # -- Logit Scale Parameter ---
        # Could help or hurt to include it; 'auto mode' does *not* include logit_scale grad.
        #{"params": [model.logit_scale], "lr": 1e-8},
    ]


    param_groups = [g for g in param_groups if len(g.get("params", [])) > 0]
    return param_groups



"""
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ > You have reached the end of the user-configurable part of this code.   │
 └──────────────────────────────────────────────────────────────────────────┘
"""




# ============================================================
# Enable grad for model parameters
# ============================================================
def unfreeze_layers(model, param_groups, verbose: bool = False):
    """
    Freeze everything, then unfreeze the parameters referenced by param_groups.
    If verbose=True, print HOT/COLD summary via print_grad_status_summary(model).
    """
    # param_groups defines trainability
    from utils_train.grad_groups_utils import apply_param_groups_trainability

    apply_param_groups_trainability(model, param_groups, verbose=True)

    if verbose:
        print_grad_status_summary(model)

# ============================================================
# Heuristic Dataset
# ============================================================
def _seed_worker(worker_id: int):
    """
    Reproducible per-worker seeding for python/numpy RNG.
    torch RNG is already set per-worker by PyTorch via torch.initial_seed().
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except Exception:
        pass

def _build_heuristic_spec_from_cfg_dict(d: Dict[str, Any], seed: int) -> HeuristicDatasetSpec:
    porh = (d.get("path_or_hf_dataset", "") or "").strip()
    tlp = d.get("text_labels_path", None)
    split = d.get("split", None)
    image_root = d.get("image_root", None)

    return HeuristicDatasetSpec(
        path_or_hf_dataset=porh,
        text_labels_path=tlp,
        split=split,
        image_root=image_root,
        seed=int(seed),
        deterministic_text=bool(d.get("deterministic_text", False)),
    )

# ============================================================
# MAIN
# ============================================================
def main(cfg: TrainConfig):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ------------------------
    # Run folder init
    # ------------------------
    os.makedirs(cfg.plots_folder, exist_ok=True)
    os.makedirs(cfg.ft_checkpoints_folder, exist_ok=True)
    os.makedirs(cfg.text_logs_folder, exist_ok=True)
    os.makedirs(cfg.teacher_folder, exist_ok=True)
    os.makedirs(cfg.optimizer_state_folder, exist_ok=True)
    teacher_log_path = os.path.join(cfg.text_logs_folder, cfg.teacher_log_name)

    # json dump the *effective* config for reproducibility
    cfg_dump_path = os.path.join(cfg.run_dir, "train_config_effective.json")
    save_cfg_json(cfg, cfg_dump_path, nd=6)
    print(Fore.CYAN + f"[ConfigJSON] saved effective config -> {cfg_dump_path}" + Style.RESET_ALL)


    # ------------------------
    # Resume decision
    # ------------------------
    continue_run, continue_reason = decide_resume_or_overwrite(cfg)
    print(Fore.CYAN + f"[Run] continue_run={continue_run} reason={continue_reason}" + Style.RESET_ALL)

    start_epoch = 0
    loaded_bundle = None

    if continue_run:
        loaded_bundle = load_resume_bundle(cfg, device=device)
        start_epoch = int(loaded_bundle.get("epoch_idx", -1)) + 1
        print(Fore.CYAN + f"[Resume] start_epoch={start_epoch}" + Style.RESET_ALL)

        bundle_ko = bool(loaded_bundle.get("use_ko_config", cfg.use_ko_config))
        if bundle_ko != cfg.use_ko_config:
            raise SystemExit(
                f"[Abort] use_ko_config mismatch: bundle={bundle_ko} vs cfg={cfg.use_ko_config}. "
                "Set cfg.use_ko_config to match the saved run."
            )

        if cfg.use_regression_teachers and os.path.exists(teacher_log_path):
            last_ln = _read_last_nonheader_line(teacher_log_path)
            if last_ln is not None:
                print(Fore.CYAN + f"[TeacherLog] loaded last state from file." + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + "[TeacherLog] no previous teacher log line found." + Style.RESET_ALL)

    # ------------------------
    # Seeds
    # ------------------------
    dl_generator = torch.Generator()
    dl_generator.manual_seed(int(cfg.teacher_seed))

    torch.manual_seed(cfg.teacher_seed)
    random.seed(cfg.teacher_seed)
    try:
        import numpy as np
        np.random.seed(cfg.teacher_seed)
    except Exception:
        pass

    # ------------------------------------------------
    # CLIP module (KO-GmP or GmP), load CLIP model
    # ------------------------------------------------
    clip = import_clip_module(cfg.use_ko_config)
    print(Fore.CYAN + f"[CLIP] module={'gmpclipheaddropout' if cfg.use_ko_config else 'gmpclipregression'}" + Style.RESET_ALL)

    base_model, preprocess, _ = load_openai_clip_anything(clip, cfg.clipmodel, device=device, jit=False, strict=True)

    if continue_run and loaded_bundle is not None:
        model_from_pickle = try_load_model_from_resume(loaded_bundle, device=device)
        if model_from_pickle is not None:
            model = model_from_pickle
            print(Fore.CYAN + "[Resume] Loaded full model pickle." + Style.RESET_ALL)
        else:
            model = base_model
            sd_cpu = loaded_bundle.get("model_state_dict_cpu", None)
            if sd_cpu is None:
                raise SystemExit("[Abort] Resume bundle has neither loadable full model nor model_state_dict_cpu.")
            missing, unexpected = model.load_state_dict(sd_cpu, strict=False)
            print(Fore.CYAN + f"[Resume] Loaded state_dict into base model (missing={len(missing)} unexpected={len(unexpected)})." + Style.RESET_ALL)
    else:
        model = base_model

    model = model.float()
    _assert_vit_visual_or_abort(model, cfg)

    model_dtype = _get_model_dtype(model)
    image_dtype = _get_image_dtype(model)
    print(f"Precision (param dtype): {model_dtype}")
    print(f"Image dtype (conv1):     {image_dtype}")

    # ------------------------
    # Logs init
    # ------------------------
    
    teacher_specs: Dict[int, Dict[str, Any]] = {}
    teacher_cache_paths: Dict[int, str] = {}

    if cfg.use_regression_teachers:
        teacher_specs = resolve_regression_teacher_specs(cfg, model)
        teacher_cache_paths = make_teacher_cache_paths(cfg, teacher_specs)

        _init_log_file(
            teacher_log_path,
            header=(
                "epoch\tlayer\trebuilt\treason\t"
                "tcos\tbest\tfit_cos\tfit_mse\tn_pairs\t"
                "reg_threshold\tlam\tcls_mix\t"
                "jl_dim\tjl_num_proj\tjl_seed\tjl_seed_stride\n"
            ),
            continue_run=continue_run
        )

    bench_log_path = os.path.join(cfg.text_logs_folder, "tiny_benchmark_log.txt")
    _init_log_file(
        bench_log_path,
        header="epoch\tfolder\tn\tacc\tmean_margin\tmean_logit_correct\tmean_logit_othermax\n",
        continue_run=continue_run
    )

    gap_log_path = os.path.join(cfg.text_logs_folder, "gapness_log.txt")
    _init_log_file(
        gap_log_path,
        header="epoch\tn\tcentroid_gap_l2\tlinsep_acc\n",
        continue_run=continue_run
    )

    quick_probe_log_path = os.path.join(cfg.text_logs_folder, "quick_probe_log.txt")
    if cfg.run_quick_probe:
        _init_log_file(
            quick_probe_log_path,
            header="epoch\tlinear_probe_acc\tzero_shot_acc\n",
            continue_run=continue_run
        )

    ko_log_path = os.path.join(cfg.text_logs_folder, cfg.ko_log_name)
    if cfg.use_ko_config:
        _init_log_file(
            ko_log_path,
            header="epoch\tmean_decatt\tmean_kproj\n",
            continue_run=continue_run
        )

    adv_log_path = os.path.join(cfg.text_logs_folder, cfg.adversarial_log_name)
    if cfg.use_adv_triplet:
        _init_log_file(
            adv_log_path,
            header="epoch\tglobal_batch_step\tbatch_idx\tadv_loss\tadv_lambda\tsim_pos0\tsim_pos1\tsim_neg\n",
            continue_run=continue_run
        )

    # ------------------------
    # Adversarial augs (GPU)
    # ------------------------
    adversarial_augs = None
    if cfg.use_adv_triplet:
        try:
            import kornia.augmentation as K
            adversarial_augs = torch.nn.Sequential(
                K.RandomAffine(degrees=12, translate=0.12, scale=(0.94, 1.06), p=0.8),
                K.RandomResizedCrop(size=(224, 224), scale=(0.92, 1.0), ratio=(0.9, 1.1), p=0.7),
                K.ColorJitter(brightness=0.15, contrast=0.15, p=0.5),
            ).to(device)
            adversarial_augs.eval()
            print(Fore.CYAN + "[AdvTriplet] kornia adversarial_augs enabled on device." + Style.RESET_ALL)
        except Exception as e:
            raise SystemExit(f"[Abort] cfg.use_adv_triplet=True but kornia aug init failed: {e}")

    # ------------------------
    # Datasets / loaders
    # ------------------------
    print(Fore.YELLOW + "NOTE! Initially, worker init can take 1-2 minutes, depending on OS (process 'spawn' vs. 'fork')" + Style.RESET_ALL)
    print(Fore.YELLOW + "Please be patient. Subsequent (second time) Dataloader execution will be super fast." + Style.RESET_ALL)

    # Build datasets
    if cfg.use_custom_dataset:
        print(Fore.CYAN + "[Data] using heuristic dataset loader (cfg.use_custom_dataset=True)" + Style.RESET_ALL)
        print(Fore.MAGENTA + Style.BRIGHT + "When loading an Image-as-URL (pointers) dataset from HuggingFace, the dataloader will take as long as the download takes." + Style.RESET_ALL)
        print(Fore.MAGENTA + Style.BRIGHT + "-> If you see the bar slowly dragging along, consider separately downloading the dataset to local first." + Style.RESET_ALL)
        train_spec = _build_heuristic_spec_from_cfg_dict(cfg.custom_dataset_train, seed=cfg.teacher_seed)
        train_dataset = build_heuristic_dataset(
            clip,
            train_spec,
            transform=preprocess,
            tokenize_fn=clip.tokenize,
            verbose=True,
        )

        # val: if not specified, fall back to train spec
        val_cfg = cfg.custom_dataset_val or {}
        val_porh = (val_cfg.get("path_or_hf_dataset", "") or "").strip()
        val_tlp = val_cfg.get("text_labels_path", None)

        if val_porh == "" and (val_tlp is None or str(val_tlp).strip() == ""):
            print(Fore.YELLOW + "[Data] custom_dataset_val not set; using train dataset spec for validation." + Style.RESET_ALL)
            val_spec = train_spec
            val_spec.deterministic_text = True
        else:
            val_spec = _build_heuristic_spec_from_cfg_dict(cfg.custom_dataset_val, seed=cfg.teacher_seed)

        val_dataset = build_heuristic_dataset(
            clip,
            val_spec,
            transform=preprocess,
            tokenize_fn=clip.tokenize,
            verbose=True,
        )

    else:
        print(Fore.CYAN + "[Data] using COCO SPRIGHT dataset loader (cfg.use_custom_dataset=False)" + Style.RESET_ALL)
        train_dataset = ImageTextDataset(cfg.coco_root, cfg.coco_train_json, transform=preprocess)
        val_dataset   = ImageTextDataset(cfg.coco_root, cfg.coco_val_json,   transform=preprocess)

    # Single, canonical dl_kwargs (DO NOT redefine later)
    dl_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        worker_init_fn=(_seed_worker if cfg.num_workers > 0 else None),
        generator=dl_generator,
    )
    if dl_kwargs["prefetch_factor"] is None:
        del dl_kwargs["prefetch_factor"]
    if dl_kwargs["worker_init_fn"] is None:
        del dl_kwargs["worker_init_fn"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        **dl_kwargs,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    print(Fore.CYAN + f"Total batches: {len(train_dataloader)} @ Batch Size: {cfg.batch_size}" + Style.RESET_ALL)

    adversarial_triplet_loader = None
    if cfg.use_adv_triplet:
        adversarial_triplet_dataset = AdversarialTripletTextDataset(
            cfg.adv_annotations_file,
            transform=preprocess,
            tokenize_fn=clip.tokenize,
        )
        adversarial_triplet_loader = DataLoader(
            adversarial_triplet_dataset,
            batch_size=cfg.adv_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
    if cfg.use_adv_triplet:
        print(Fore.CYAN + f"[AdvTriplet] enabled=True inject_every={cfg.adv_inject_every} lambda={cfg.adv_lambda} bs={cfg.adv_batch_size}" + Style.RESET_ALL)
        print(Fore.CYAN + f"[AdvTriplet] dataset_len={len(adversarial_triplet_dataset)} loader_len={len(adversarial_triplet_loader)}" + Style.RESET_ALL)

    teacher_imagenet_train_loader = None
    teacher_imagenet_val_loader = None

    if cfg.use_regression_teachers:
        if cfg.use_imagenet:
            teacher_imagenet_train = BalancedImageFolderTeacher(
                root=cfg.imagenet_train_root,
                transform=preprocess,
                max_per_class=cfg.imagenet_max_per_class,
                seed=cfg.teacher_seed,
            )
            teacher_imagenet_val = BalancedImageFolderTeacher(
                root=cfg.imagenet_val_root,
                transform=preprocess,
                max_per_class=cfg.imagenet_max_per_class,
                seed=cfg.teacher_seed,
            )

            teacher_imagenet_train_loader = DataLoader(
                teacher_imagenet_train,
                batch_size=cfg.batch_size,
                shuffle=False,
                **dl_kwargs,
            )
            teacher_imagenet_val_loader = DataLoader(
                teacher_imagenet_val,
                batch_size=cfg.batch_size,
                shuffle=False,
                **dl_kwargs,
            )
            print(Fore.CYAN + "[Teacher] ImageNet extra enabled for teacher loaders." + Style.RESET_ALL)
        else:
            print(Fore.CYAN + "[Teacher] ImageNet extra disabled. Teachers will train only on val." + Style.RESET_ALL)


    # ------------------------
    # Linear Probe loader
    # ------------------------
    probe_loader = None
    classnames = None
    probe_label_indices = None
    if cfg.run_quick_probe:
        df = pd.read_csv(cfg.probe_csv)
        classnames = sorted(list(set(df["label"])))
        class2idx = {c: i for i, c in enumerate(classnames)}
        df["label_idx"] = df["label"].map(class2idx)
        probe_label_indices = torch.tensor(df["label_idx"].values, dtype=torch.long)

        probe_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        probe_ds = CroppedImageCSVFileDataset(df, cfg.probe_image_dir, transform=probe_tf)
        probe_dl_kwargs = dict(
            num_workers=cfg.probe_num_workers,
            pin_memory=True,
            persistent_workers=(cfg.persistent_workers and cfg.probe_num_workers > 0),
        )
        if cfg.probe_num_workers > 0:
            probe_dl_kwargs["prefetch_factor"] = cfg.prefetch_factor

        probe_loader = DataLoader(
            probe_ds,
            batch_size=cfg.probe_batch_size,
            shuffle=False,
            **probe_dl_kwargs,
        )

    # ------------------------
    # Warmup workers
    # ------------------------
    if cfg.num_workers > 0:
        _ = next(iter(train_dataloader))
    if cfg.probe_num_workers > 0 and probe_loader is not None:
        _ = next(iter(probe_loader))

    # ============================================================
    # Histories (for plotting)
    # ============================================================
    epoch_ids: List[int] = []
    training_losses: List[float] = []
    validation_losses: List[float] = []

    logits_diag_train_hist: List[float] = []
    logits_off_train_hist: List[float] = []
    logits_diag_val_hist: List[float] = []
    logits_off_val_hist: List[float] = []

    # load last logged epoch per layer (fast map)
    teacher_last_logged_epoch: Dict[int, int] = {}
    if cfg.use_regression_teachers and os.path.exists(teacher_log_path):
        from utils_train.config_utils import load_last_logged_teacher_epoch_by_layer
        teacher_last_logged_epoch = load_last_logged_teacher_epoch_by_layer(teacher_log_path)

    teacher_hist: Dict[int, Dict[str, List[float]]] = {}
    if cfg.use_regression_teachers:
        for layer in teacher_specs.keys():
            layer = int(layer)
            teacher_hist[layer] = {"epochs": [], "tcos": [], "fit_cos": [], "fit_mse": []}

    # resume teacher history from log (for unbroken plots)
    if cfg.use_regression_teachers and continue_run and os.path.exists(teacher_log_path):
        from utils_train.config_utils import load_teacher_history_from_log
        loaded_hist = load_teacher_history_from_log(teacher_log_path)
        for layer in teacher_specs.keys():
            layer = int(layer)
            if layer in loaded_hist:
                teacher_hist[layer] = loaded_hist[layer]  # overwrite with log-derived history

    probe_epochs: List[int] = []
    lin_probe_hist: List[float] = []
    zs_hist: List[float] = []

    tiny_history: Dict[str, Dict[str, List[float]]] = {}
    if cfg.run_tiny_benchmark:
        folders = [p.format(dataset=cfg.tiny_dataset) for p in cfg.tiny_folders]
        for fol in folders:
            tiny_history[fol] = {"epochs": [], "acc": [], "margin": []}

    # ============================================================
    # load plot histories from logs on resume
    # ============================================================
    if continue_run:
        from utils_train.config_utils import (
            load_quick_probe_history_from_log,
            load_tiny_benchmark_history_from_log,
            load_training_history_from_log,
        )

        # training_log -> combined_training_plots
        training_log_path = os.path.join(cfg.text_logs_folder, "training_log.txt")
        th = load_training_history_from_log(training_log_path)
        epoch_ids = th["epoch_ids"]
        training_losses = th["training_losses"]
        validation_losses = th["validation_losses"]
        logits_diag_train_hist = th["logits_diag_train_hist"]
        logits_off_train_hist  = th["logits_off_train_hist"]
        logits_diag_val_hist   = th["logits_diag_val_hist"]
        logits_off_val_hist    = th["logits_off_val_hist"]

        # regenerate immediately so the plot is "unbroken" before epoch loop.
        plot_training_info(
            epoch_ids=epoch_ids,
            training_losses=training_losses,
            validation_losses=validation_losses,
            logits_diag_train=logits_diag_train_hist,
            logits_off_train=logits_off_train_hist,
            logits_diag_val=logits_diag_val_hist,
            logits_off_val=logits_off_val_hist,
            plots_folder=cfg.plots_folder,
        )

        # quick_probe_log
        if cfg.run_quick_probe and os.path.exists(quick_probe_log_path):
            pe, lp, zs = load_quick_probe_history_from_log(quick_probe_log_path)
            probe_epochs = pe
            lin_probe_hist = lp
            zs_hist = zs
            plot_probe_info(probe_epochs, lin_probe_hist, zs_hist, cfg.plots_folder)

        # tiny_benchmark_log
        if cfg.run_tiny_benchmark and os.path.exists(bench_log_path):
            loaded_tiny = load_tiny_benchmark_history_from_log(bench_log_path)
            for fol in list(tiny_history.keys()):
                if fol in loaded_tiny:
                    tiny_history[fol] = loaded_tiny[fol]
            plot_tiny_benchmark(tiny_history, cfg.plots_folder)


    # ============================================================
    # Quick probe post-eval wrapper
    # ============================================================
    def _normalize_quick_probe_output(out):
        lp_acc = float("nan")
        zs_acc = float("nan")

        if isinstance(out, dict):
            for k in ["linear_probe_acc", "lp_acc", "lp", "probe_acc"]:
                if k in out:
                    lp_acc = float(out[k])
                    break
            for k in ["zero_shot_acc", "zs_acc", "zs", "zeroshot_acc"]:
                if k in out:
                    zs_acc = float(out[k])
                    break
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            lp_acc = float(out[0])
            zs_acc = float(out[1])
        elif isinstance(out, (int, float)):
            lp_acc = float(out)

        return lp_acc, zs_acc

    def _run_quick_probe_epoch(epoch_i: int):
        if not cfg.run_quick_probe:
            return
        if probe_loader is None or classnames is None or probe_label_indices is None:
            print(Fore.YELLOW + "[QuickProbe] skipped (probe_loader/classnames/labels missing)." + Style.RESET_ALL)
            return

        out = _call_with_signature(
            run_quick_probe,
            model=model,
            clip=clip,
            probe_loader=probe_loader,
            dataloader=probe_loader,
            loader=probe_loader,
            classnames=classnames,
            class_names=classnames,
            label_indices=probe_label_indices,
            probe_label_indices=probe_label_indices,
            device=device,
            max_iter=cfg.probe_max_iter,
            max_steps=cfg.probe_max_iter,
        )

        lp_acc, zs_acc = _normalize_quick_probe_output(out)

        print(Fore.CYAN + f"[QuickProbe] epoch={epoch_i}  LP={lp_acc:.4f}  ZS={zs_acc:.4f}" + Style.RESET_ALL)
        with open(quick_probe_log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch_i}\t{lp_acc:.6f}\t{zs_acc:.6f}\n")

        probe_epochs.append(int(epoch_i))
        lin_probe_hist.append(float(lp_acc))
        zs_hist.append(float(zs_acc))
        plot_probe_info(probe_epochs, lin_probe_hist, zs_hist, cfg.plots_folder)

    # ============================================================
    # Teacher init/build  (RESUME-SAFE)
    # ============================================================
    if cfg.use_regression_teachers:
        cls_patch_teachers: Dict[int, Optional[dict]] = {int(layer): None for layer in teacher_specs.keys()}
        jl_projectors: Dict[int, Optional[nn.Module]] = {int(layer): None for layer in teacher_specs.keys()}

        _embed_dim = int(model.visual.proj.shape[1])

        # Build JL projectors per teacher spec
        for layer, spec in teacher_specs.items():
            layer = int(layer)
            jl_projectors[layer] = make_jl_projector_for_teacher(_embed_dim, spec, device)
            if jl_projectors[layer] is not None:
                jl = spec["jl"]
                print(
                    Fore.CYAN
                    + f"[Teacher{layer}-JL] enabled: d_out={jl['dim']} n_proj={jl['num_proj']} seed={jl['seed']} stride={jl['seed_stride']}"
                    + Style.RESET_ALL
                )

        # Only include ImageNet loaders when present
        extra_loaders = []
        if teacher_imagenet_train_loader is not None:
            extra_loaders.append(teacher_imagenet_train_loader)
        if teacher_imagenet_val_loader is not None:
            extra_loaders.append(teacher_imagenet_val_loader)

        # restore per-layer state from teacher_log on resume (best cos + last rebuild epoch)
        teacher_state: Dict[int, Dict[str, Any]] = {}
        if continue_run and os.path.exists(teacher_log_path):
            from utils_train.config_utils import load_teacher_state_from_log
            teacher_state = load_teacher_state_from_log(teacher_log_path)

        # use distinct "init epoch tags" to avoid collisions
        # fresh run init rows:  epoch = -1
        # resume-startup rows:  epoch = -2
        init_teacher_epoch = (-1 if not continue_run else -2)

        last_teacher_update_epoch: Dict[int, int] = {}
        best_teacher_cos: Dict[int, Optional[float]] = {}

        for layer in teacher_specs.keys():
            layer = int(layer)
            st = teacher_state.get(layer, {}) or {}
            best_teacher_cos[layer] = st.get("best_tcos", None)

            lre = st.get("last_rebuild_epoch", None)
            if lre is None:
                # start cooldown baseline at "init tag"
                last_teacher_update_epoch[layer] = int(init_teacher_epoch)
            else:
                last_teacher_update_epoch[layer] = int(lre)

        def _tstats(t: Optional[dict]):
            if t is None:
                return float("nan"), float("nan"), 0
            return (
                float(t.get("fit_cos_val", float("nan"))),
                float(t.get("fit_mse_val", float("nan"))),
                int(t.get("n_pairs", 0)),
            )

        rebuilt_by_layer: Dict[int, int] = {int(layer): 0 for layer in teacher_specs.keys()}
        reason_by_layer: Dict[int, str] = {int(layer): "keep" for layer in teacher_specs.keys()}
        payload_by_layer: Dict[int, Dict[str, Any]] = {}

        # resume loads cache first; rebuild only if needed
        for layer, spec in teacher_specs.items():
            layer = int(layer)

            # Fresh run: always rebuild to match fresh model weights
            force_rebuild = (not continue_run)

            cls_patch_teachers[layer] = build_cls_patch_regression_teacher(
                model=model,
                visual=model.visual,
                dataloader=val_dataloader,
                device=device,
                max_samples=cfg.cls_patch_max_samples,
                val_frac=cfg.cls_patch_val_frac,
                cache_path=teacher_cache_paths[layer],
                reg_threshold=spec["reg_threshold"],
                teacher_layer=int(layer),
                teacher_seed=cfg.teacher_seed,
                extra_image_loaders=extra_loaders,
                force_rebuild=bool(force_rebuild),
                cls_mix=float(spec.get("cls_mix", 0.0)),
                cls_mix_use_reg=bool(spec.get("cls_mix_use_reg", False)),
                clipmodel_id=str(cfg.clipmodel),
                is_reg_teacher=bool(spec.get("is_reg_teacher", False)),
                use_reg_whitening=bool(spec.get("use_reg_whitening", False)), 
            )

            t = cls_patch_teachers[layer]

            if not continue_run:
                rebuilt_by_layer[layer] = 1
                reason_by_layer[layer] = "init_build"
                last_teacher_update_epoch[layer] = int(init_teacher_epoch)
            else:
                if t is None:
                    # resume but cache invalid/missing -> must rebuild now
                    cls_patch_teachers[layer] = build_cls_patch_regression_teacher(
                        model=model,
                        visual=model.visual,
                        dataloader=val_dataloader,
                        device=device,
                        max_samples=cfg.cls_patch_max_samples,
                        val_frac=cfg.cls_patch_val_frac,
                        cache_path=teacher_cache_paths[layer],
                        reg_threshold=spec["reg_threshold"],
                        teacher_layer=int(layer),
                        teacher_seed=cfg.teacher_seed,
                        extra_image_loaders=extra_loaders,
                        force_rebuild=True,
                        cls_mix=float(spec.get("cls_mix", 0.0)),
                        cls_mix_use_reg=bool(spec.get("cls_mix_use_reg", False)),
                        clipmodel_id=str(cfg.clipmodel),
                        is_reg_teacher=bool(spec.get("is_reg_teacher", False)),
                        use_reg_whitening=bool(spec.get("use_reg_whitening", False)), 
                    )
                    t = cls_patch_teachers[layer]
                    rebuilt_by_layer[layer] = 1
                    reason_by_layer[layer] = "resume_missing_or_invalid_cache_rebuild"
                    last_teacher_update_epoch[layer] = int(init_teacher_epoch)
                else:
                    if (t is not None) and (not bool(t.get("loaded_from_cache", False))):
                        rebuilt_by_layer[layer] = 1
                        reason_by_layer[layer] = "resume_cache_mismatch_rebuild"
                        last_teacher_update_epoch[layer] = int(init_teacher_epoch)
                    else:
                        rebuilt_by_layer[layer] = 0
                        reason_by_layer[layer] = "resume_load_cache"

            # Evaluate teacher cosine once so payload is well-defined
            if t is None:
                payload_by_layer[layer] = {
                    "tcos": None, "best": best_teacher_cos[layer], "fit_cos": None, "fit_mse": None, "n": 0,
                    "reg_threshold": str(spec["reg_threshold"]),
                    "lam": float(spec["lam"]),
                    "jl": spec.get("jl", {}),
                    "cls_mix": float(spec.get("cls_mix", 0.0)),
                }
                continue

            tcos = evaluate_teacher_cosine(
                model=model,
                visual=model.visual,
                dataloader=val_dataloader,
                teacher=t,
                device=device,
                reg_threshold=spec["reg_threshold"],
                teacher_layer=int(layer),
                max_batches=30,
                projector=jl_projectors.get(layer, None),
                cls_mix=float(spec.get("cls_mix", 0.0)),
                cls_mix_use_reg=bool(spec.get("cls_mix_use_reg", False)),
            )

            if (tcos is not None) and ((best_teacher_cos[layer] is None) or (tcos > best_teacher_cos[layer])):
                best_teacher_cos[layer] = float(tcos)

            fitcos, fitmse, n_pairs = _tstats(t)
            payload_by_layer[layer] = {
                "tcos": (float(tcos) if tcos is not None else None),
                "best": (float(best_teacher_cos[layer]) if best_teacher_cos[layer] is not None else None),
                "fit_cos": float(fitcos),
                "fit_mse": float(fitmse),
                "n": int(n_pairs),
                "reg_threshold": str(spec["reg_threshold"]),
                "lam": float(spec["lam"]),
                "jl": spec.get("jl", {}),
                "cls_mix": float(spec.get("cls_mix", 0.0)),
            }

        # startup logging policy
        # - fresh run: write ALL layers at epoch=-1
        # - resume: write ONLY rebuilt layers at epoch=-2 (so it never collides / gets skipped)
        if not continue_run:
            _append_teacher_log_rows(
                teacher_log_path=teacher_log_path,
                epoch_i=int(init_teacher_epoch),
                rebuilt_by_layer=rebuilt_by_layer,
                reason_by_layer=reason_by_layer,
                payload_by_layer=payload_by_layer,
                last_logged_epoch_by_layer={},
                nd=5,
            )
            for layer in teacher_specs.keys():
                layer = int(layer)
                payload = payload_by_layer.get(layer, {}) or {}
                teacher_hist[layer]["epochs"].append(int(init_teacher_epoch))
                teacher_hist[layer]["tcos"].append(float(payload.get("tcos")) if payload.get("tcos") is not None else float("nan"))
                teacher_hist[layer]["fit_cos"].append(float(payload.get("fit_cos")) if payload.get("fit_cos") is not None else float("nan"))
                teacher_hist[layer]["fit_mse"].append(float(payload.get("fit_mse")) if payload.get("fit_mse") is not None else float("nan"))

                plot_teacher_info(
                    teacher_hist[layer]["epochs"],
                    teacher_hist[layer]["tcos"],
                    teacher_hist[layer]["fit_cos"],
                    teacher_hist[layer]["fit_mse"],
                    cfg.plots_folder,
                    out_name=f"teacher_stats_b{layer}.png",
                )

            # update map in-memory (prevents duplicates if called twice somehow)
            for layer in teacher_specs.keys():
                teacher_last_logged_epoch[int(layer)] = int(init_teacher_epoch)

        else:
            rebuilt_layers = [int(l) for l, r in rebuilt_by_layer.items() if int(r) == 1]
            if len(rebuilt_layers) > 0:
                payload_subset = {int(l): payload_by_layer[int(l)] for l in rebuilt_layers}
                rebuilt_subset = {int(l): 1 for l in rebuilt_layers}
                reason_subset = {int(l): reason_by_layer[int(l)] for l in rebuilt_layers}

                _append_teacher_log_rows(
                    teacher_log_path=teacher_log_path,
                    epoch_i=int(init_teacher_epoch),
                    rebuilt_by_layer=rebuilt_subset,
                    reason_by_layer=reason_subset,
                    payload_by_layer=payload_subset,
                    last_logged_epoch_by_layer={},
                    nd=5,
                )

                for layer in rebuilt_layers:
                    payload = payload_by_layer.get(int(layer), {}) or {}
                    teacher_hist[int(layer)]["epochs"].append(int(init_teacher_epoch))
                    teacher_hist[int(layer)]["tcos"].append(float(payload.get("tcos")) if payload.get("tcos") is not None else float("nan"))
                    teacher_hist[int(layer)]["fit_cos"].append(float(payload.get("fit_cos")) if payload.get("fit_cos") is not None else float("nan"))
                    teacher_hist[int(layer)]["fit_mse"].append(float(payload.get("fit_mse")) if payload.get("fit_mse") is not None else float("nan"))

                    plot_teacher_info(
                        teacher_hist[int(layer)]["epochs"],
                        teacher_hist[int(layer)]["tcos"],
                        teacher_hist[int(layer)]["fit_cos"],
                        teacher_hist[int(layer)]["fit_mse"],
                        cfg.plots_folder,
                        out_name=f"teacher_stats_b{int(layer)}.png",
                    )

                for layer in rebuilt_layers:
                    teacher_last_logged_epoch[int(layer)] = int(init_teacher_epoch)

            else:
                print(Fore.CYAN + "[Teacher] Resume: all teachers loaded from cache; no startup log rows written." + Style.RESET_ALL)

    
    # ============================================================
    # Optimizer, Scheduler, param groups
    # ============================================================
    from utils_train.grad_groups_utils import (
        apply_param_groups_trainability,
        build_param_groups_automatic,
    )

    # build param_groups from manual or automatic presets
    if cfg.grad_set_manual:
        preset_name = "grad_set_manual"
        param_groups = set_manual_param_groups(model, cfg)
    else:
        try:
            preset_name, param_groups = build_param_groups_automatic(model, cfg)
        except Exception as e:
            print(Fore.RED + Style.BRIGHT + f"[GradPreset] INVALID CONFIG: {e}" + Style.RESET_ALL)
            raise SystemExit(1)

    print("[Optim] built param_groups:", len(param_groups))
    print("[Optim] group lrs (pre-opt):", [g.get("lr", None) for g in param_groups])
    print("[Optim] group sizes:", [len(g.get("params", [])) for g in param_groups])


    # unfreeze + detailed print: is grad hot or not?
    print(Fore.CYAN + f"[GradPreset] {preset_name}" + Style.RESET_ALL)
    unfreeze_layers(model, param_groups, verbose=cfg.grad_debug_print)

    # build optimizer/scheduler from these groups
    optimizer = build_optimizer_from_cfg(cfg, param_groups)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    print("[Optim] optimizer group lrs (post-opt):", [g.get("lr", None) for g in optimizer.param_groups])

    if scheduler is None:
        print(Fore.CYAN + "[Sched] disabled (scheduler_name='none')." + Style.RESET_ALL)
    else:
        print(Fore.CYAN + f"[Sched] {cfg.scheduler_name}  kwargs={cfg.scheduler_kwargs}  step_mode={cfg.scheduler_step_mode}" + Style.RESET_ALL)


    # ============================================================
    # Contrastive loss
    # ============================================================
    if cfg.use_gap_schedule:
        contrastive_loss = GapContrastiveLoss(
            temperature=0.07,
            smoothing=cfg.contrastive_smoothing,
        ).to(device)
        print(Fore.CYAN + "[Loss] GapContrastiveLoss enabled." + Style.RESET_ALL)
    else:
        contrastive_loss = BaseContrastiveLoss(
            temperature=0.07,
            smoothing=cfg.contrastive_smoothing,
        ).to(device)
        print(Fore.CYAN + "[Loss] BaseContrastiveLoss enabled (no gap extras)." + Style.RESET_ALL)

    scaler = GradScaler()

    # ------------------------
    # Optional EMA model
    # ------------------------
    ema = None
    if cfg.use_ema:
        ema = EMAState(
            model=model,
            decay_step=cfg.ema_decay_step,
            store_fp16=cfg.ema_store_dtype_fp16,
        )
        print(Fore.CYAN + f"[EMA] enabled (store_dtype={ema.store_dtype}, update_every_n_optim_steps={cfg.ema_update_every_n_optim_steps})" + Style.RESET_ALL)

    # ------------------------------------------------
    # Resume: load optimizer/scheduler/scaler/RNG/EMA
    # ------------------------------------------------
    global_optim_step = 0
    if continue_run and loaded_bundle is not None:
        try:
            optimizer.load_state_dict(loaded_bundle["optimizer"])
            if loaded_bundle.get("scheduler", None) is not None:
                scheduler.load_state_dict(loaded_bundle["scheduler"])
            if loaded_bundle.get("scaler", None) is not None:
                scaler.load_state_dict(loaded_bundle["scaler"])

            global_optim_step = int(loaded_bundle.get("global_optim_step", 0))

            try:
                tstate = _coerce_rng_state_to_uint8(loaded_bundle.get("torch_rng_state", None))
                if tstate is not None:
                    torch.set_rng_state(tstate)

                if torch.cuda.is_available():
                    cstates = loaded_bundle.get("cuda_rng_state_all", None)
                    if cstates is not None:
                        if isinstance(cstates, (list, tuple)):
                            cstates_u8 = [_coerce_rng_state_to_uint8(s) for s in cstates]
                            torch.cuda.set_rng_state_all(cstates_u8)
                        else:
                            torch.cuda.set_rng_state(_coerce_rng_state_to_uint8(cstates))

                random.setstate(loaded_bundle["py_random_state"])
            except Exception as e:
                print(Fore.YELLOW + f"[Resume] RNG restore failed: {e}" + Style.RESET_ALL)

            if (ema is not None) and (loaded_bundle.get("ema", None) is not None):
                try:
                    ema.load_state_dict_cpu(loaded_bundle["ema"])
                    print(Fore.CYAN + "[Resume] EMA restored." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.YELLOW + f"[Resume] EMA restore failed: {e}" + Style.RESET_ALL)

            print(Fore.CYAN + "[Resume] optimizer/scheduler/scaler restored (loaded-state wins)." + Style.RESET_ALL)
        except Exception as e:
            raise SystemExit(f"[Abort] Failed to restore optimizer bundle: {e}")

    # ------------------------------------------------------------------------
    # Quick PRE-EVAL: Typographic Attack, Linear Probe, Zero-Shot
    # ------------------------------------------------------------------------    
    pre_epoch = -1
    if (not continue_run) and cfg.run_tiny_benchmark:
        folders = [p.format(dataset=cfg.tiny_dataset) for p in cfg.tiny_folders]
        bench = run_tiny_benchmark(
            model=model,
            clip=clip,
            preprocess=preprocess,
            device=device,
            folders=folders,
            choices=list(cfg.tiny_choices),
            correct_choice_idx=0,
            batch_size=cfg.tiny_batch_size,
        )
        for fol, stats in bench.items():
            with open(bench_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{pre_epoch}\t{fol}\t{stats.get('n',0)}\t{stats.get('acc',float('nan')):.6f}\t"
                    f"{stats.get('mean_margin',float('nan')):.6f}\t{stats.get('mean_logit_correct',float('nan')):.6f}\t"
                    f"{stats.get('mean_logit_othermax',float('nan')):.6f}\n"
                )
            if fol in tiny_history:
                tiny_history[fol]["epochs"].append(int(pre_epoch))
                tiny_history[fol]["acc"].append(float(stats.get("acc", float("nan"))))
                tiny_history[fol]["margin"].append(float(stats.get("mean_margin", float("nan"))))
        plot_tiny_benchmark(tiny_history, cfg.plots_folder)

    # Run quick probe PRE only when probe_every_n_epochs==1, and only for fresh runs.
    if (not continue_run) and cfg.run_quick_probe and (probe_loader is not None) and (cfg.probe_every_n_epochs == 1):
        print(Fore.MAGENTA + Style.BRIGHT + "Running QuickProbe PRE (linear probe + zero-shot)..." + Style.RESET_ALL)
        _run_quick_probe_epoch(pre_epoch)

    # Adversarial Triplet Injection
    global_batch_step = 0
    adv_iter = None
    if cfg.use_adv_triplet and adversarial_triplet_loader is not None:
        adv_iter = infinite_dataloader(adversarial_triplet_loader)

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(start_epoch, cfg.epochs):
        epoch_idx = epoch

        apply_gap_loss_schedule(contrastive_loss, epoch_idx, cfg)
        if cfg.use_gap_schedule:
            print(Fore.CYAN + f"[GapLoss] e{epoch_idx}: wU={contrastive_loss.w_uniform} wA={contrastive_loss.w_align} wXU={contrastive_loss.w_xuniform}" + Style.RESET_ALL)
        else:
            print(f"[GapLoss] e{epoch_idx}: disabled (BaseContrastiveLoss)")

        gradient_norms_raw = {}
        gradient_norms_unscaled = {}
        gradient_rms_unscaled = {}

        if cfg.log_cuda_memory and device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        model.train()

        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        total_decatt_loss = 0.0
        total_kproj_loss = 0.0

        last_adv_loss_value: float = 0.0
        last_adv_sims = (float("nan"), float("nan"), float("nan"))

        train_accs, train_f1s = [], []
        batch_logits_diag = []
        batch_logits_off = []

        # --- for postfix ---
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch_idx}/{cfg.epochs-1}",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.1,
        )

        detail_every = int(getattr(cfg, "show_losses_every", 25) or 25)  # update details every N batches
        detail_every = max(1, detail_every)  # safety
        _last_line2 = ""  # persists between batches


        # ------------------------------------------------------------------------
        # BATCH
        # ------------------------------------------------------------------------
        for batch_idx, (images, texts) in progress_bar:
            images = images.to(device, non_blocking=True, dtype=image_dtype)
            texts = texts.to(device, non_blocking=True)

            with autocast():
                # ----------------------- CLIP LOSS ------------------------------
                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(texts)

                img_f = img_feats.float()
                txt_f = txt_feats.float()

                loss_clip, loss_parts = contrastive_loss(img_f, txt_f, return_parts=True)

                loss_reg_by_layer: Dict[int, torch.Tensor] = {}
                reg_term = torch.tensor(0.0, device=device)
                adv_loss = torch.tensor(0.0, device=device)
                
                # ----------------------- TEACHER LOSS ----------------------------
                if cfg.use_regression_teachers:
                    reps_by_layer: Dict[int, Dict[str, torch.Tensor]] = {}

                    # Compute CLS+PATCH once per layer (cache)
                    for layer, spec in teacher_specs.items():
                        lam = float(spec["lam"])
                        if lam == 0.0:
                            continue
                        t = cls_patch_teachers.get(layer, None)
                        if t is None:
                            continue

                        want_reg = bool(spec.get("is_reg_teacher", False) or spec.get("cls_mix_use_reg", False))

                        reps_by_layer[layer] = compute_cls_patch_embeddings(
                            images=images,
                            visual=model.visual,
                            reg_threshold=spec["reg_threshold"],
                            teacher_layer=int(layer),
                            detach=False,
                            return_reg=want_reg,
                        )

                    # Per-layer teacher losses: local CLS target / final mixing
                    for layer, spec in teacher_specs.items():
                        lam = float(spec["lam"])
                        if lam == 0.0:
                            continue
                        t = cls_patch_teachers.get(layer, None)
                        if t is None:
                            continue
                        reps = reps_by_layer.get(layer, None)
                        if reps is None:
                            continue

                        cls_local = reps["cls"]          # [B, embed_dim]
                        patch_emb = reps["patch"]
                        reg_emb = reps.get("reg", None)  # may exist even if is_reg_teacher=False (if cls_mix_use_reg=True)

                        m = float(spec.get("cls_mix", 0.0))
                        use_reg_mix = bool(spec.get("cls_mix_use_reg", False))

                        if m > 0.0:
                            if use_reg_mix:
                                if reg_emb is None:
                                    # Shouldn't happen because return_reg=True above, but keep a hard fallback.
                                    cls_target = cls_local.float()
                                else:
                                    # REG-based target mixing (stable)
                                    # Center REG to avoid "mean REG direction" dominating, then normalize both sides.
                                    r = reg_emb.float()
                                    r = r - r.mean(dim=0, keepdim=True)

                                    cls_a = F.normalize(cls_local.float(), dim=-1)
                                    cls_b = F.normalize(r, dim=-1)
                                    cls_target = F.normalize((1.0 - m) * cls_a + m * cls_b, dim=-1)
                            else:
                                # Original behavior: blend toward final image embedding
                                cls_a = F.normalize(cls_local.float(), dim=-1)   # normalize for stability
                                cls_b = F.normalize(img_f, dim=-1)
                                cls_target = F.normalize((1.0 - m) * cls_a + m * cls_b, dim=-1)
                        else:
                            cls_target = cls_local.float()


                        reg_emb_for_loss = reps.get("reg", None) if bool(spec.get("is_reg_teacher", False)) else None

                        loss_reg = regression_consistency_loss(
                            cls_target,
                            patch_emb.float(),
                            t,
                            normalize=True,
                            projector=jl_projectors.get(layer, None),
                            reg_embed=(reg_emb_for_loss.float() if reg_emb_for_loss is not None else None),
                        )


                        loss_reg_by_layer[layer] = loss_reg
                        reg_term = reg_term + (lam * loss_reg)

                # ============= LOSS, SO FAR =============
                total_loss = loss_clip + reg_term


                # ----------------------- GEOMETRY-PRESERVING LOSS ------------------------
                geom_term = torch.tensor(0.0, device=device)
                if float(getattr(cfg, "geometry_preserving_lam", 0.0)) > 0.0:
                    mode = str(getattr(cfg, "geometry_preserving_on", "img")).lower()
                    if mode == "both":
                        geom_term = (
                            _call_with_signature(geometry_preserving_loss, x=img_f, y=txt_f, img=img_f, txt=txt_f, img_f=img_f, txt_f=txt_f, device=device)
                        )
                    elif mode == "txt":
                        geom_term = (
                            _call_with_signature(geometry_preserving_loss, x=txt_f, y=None, txt=txt_f, txt_f=txt_f, device=device)
                        )
                    else:  # default "img"
                        geom_term = (
                            _call_with_signature(geometry_preserving_loss, x=img_f, y=None, img=img_f, img_f=img_f, device=device)
                        )

                    total_loss = total_loss + (float(cfg.geometry_preserving_lam) * geom_term)

                # ----------------------- KO LOSS ---------------------------------
                decatt = torch.tensor(0.0, device=device)
                kproj_orth = torch.tensor(0.0, device=device)
                if cfg.use_ko_config:
                    decatt = decatt_loss(model, selected_layers=list(cfg.ko_layers_decatt), lam=cfg.ko_decatt_lambda)
                    kproj_orth = k_proj_orthogonality_loss(model, selected_layers=list(cfg.ko_layers_kproj_ortho), lam=cfg.ko_kproj_lambda)
                    # ============= LOSS, ADD =======
                    total_loss = total_loss + decatt + kproj_orth

                # ----------------------- ADVERSARIAL LOSS ------------------------
                adv_loss = torch.tensor(0.0, device=device)
                if (
                    cfg.use_adv_triplet
                    and adv_iter is not None
                    and (cfg.adv_inject_every > 0)
                    and ((global_batch_step + 1) % cfg.adv_inject_every == 0)
                ):
                    adv_images, adv_texts = next(adv_iter)

                    adv_images = adv_images.to(device, non_blocking=True, dtype=image_dtype)
                    adv_texts = adv_texts.to(device, non_blocking=True)

                    if adversarial_augs is not None:
                        adv_images = adversarial_augs(adv_images)

                    B, C, H, W = adv_images.shape
                    seq_len = adv_texts.shape[-1]

                    adv_images_expanded = adv_images.unsqueeze(1).repeat(1, 3, 1, 1, 1)
                    #adv_images_flat = adv_images_expanded.view(B * 3, C, H, W)     # potentially less memory; however, reshape safter than view #overinterpredation
                    adv_images_flat = adv_images_expanded.reshape(B * 3, C, H, W)   # also, using this will avoid perpetual nitpicking wrath by the GPT-5.2
                    adv_texts_flat = adv_texts.view(B * 3, seq_len)

                    adv_image_features = model.encode_image(adv_images_flat)
                    adv_text_features = model.encode_text(adv_texts_flat)

                    sims = torch.cosine_similarity(adv_image_features.float(), adv_text_features.float(), dim=-1).view(B, 3)
                    adv_loss = (sims[:, 0] + sims[:, 1] + (1.0 - sims[:, 2])).mean()
                    
                    last_adv_loss_value = float(adv_loss.item())
                    s0 = float(sims[:, 0].mean().item())
                    s1 = float(sims[:, 1].mean().item())
                    sn = float(sims[:, 2].mean().item())
                    last_adv_sims = (s0, s1, sn)
                    
                    # ============= LOSS, ADD =============
                    total_loss = total_loss + (cfg.adv_lambda * adv_loss)

                # ---- Getting stats ----
                img_norm = F.normalize(img_f, dim=-1)
                txt_norm = F.normalize(txt_f, dim=-1)
                logits_matrix = (img_norm @ txt_norm.t()) / getattr(contrastive_loss, "temperature", 0.07)

                diag = logits_matrix.diag()
                off = logits_matrix[~torch.eye(logits_matrix.size(0), dtype=torch.bool, device=logits_matrix.device)]
                batch_logits_diag.append(float(diag.mean().item()))
                batch_logits_off.append(float(off.mean().item()) if off.numel() > 0 else float("nan"))

                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                acc, f1 = calculate_metrics(logits_matrix, ground_truth)
                train_accs.append(acc)
                train_f1s.append(f1)

            # ============= SCALER ============
            scaler.scale(total_loss).backward()
            
            # Gradient norm capture for plotting
            if (getattr(cfg, "grad_log_every_n_steps", 0) > 0) and ((batch_idx % cfg.grad_log_every_n_steps) == 0):
                scale = float(scaler.get_scale())
                for name, parameter in model.named_parameters():
                    g = parameter.grad
                    if g is None:
                        continue
                    raw_norm = float(g.norm().item())
                    unscaled_norm = raw_norm / max(scale, 1.0)
                    rms_unscaled = unscaled_norm / (g.numel() ** 0.5)

                    gradient_norms_raw.setdefault(name, []).append(raw_norm)
                    gradient_norms_unscaled.setdefault(name, []).append(unscaled_norm)
                    gradient_rms_unscaled.setdefault(name, []).append(rms_unscaled)

                if (getattr(cfg, "monitor_grad_every_n_steps", 0) > 0) and ((batch_idx % cfg.monitor_grad_every_n_steps) == 0):
                    monitor_gradient_norms(
                        gradient_norms_raw=gradient_norms_raw,
                        gradient_norms_unscaled=gradient_norms_unscaled,
                        gradient_rms_unscaled=gradient_rms_unscaled,
                        scale=float(scaler.get_scale()),
                    )

            # ===================== Gradient accumulation, Optimizer step =====================
            do_optim = ((batch_idx + 1) % cfg.accumulation_steps == 0) or ((batch_idx + 1) == len(train_dataloader))
            if do_optim:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_optim_step += 1
                
                # scheduler step mode
                if scheduler is not None:
                    if cfg.scheduler_step_mode == "fractional_epoch":
                        frac_epoch = epoch + float(batch_idx + 1) / float(len(train_dataloader))
                        scheduler.step(frac_epoch)
                    elif cfg.scheduler_step_mode == "per_optim":
                        scheduler.step()

                if ema is not None:
                    ema.step_counter_inc(1)
                    if cfg.ema_update_every_n_optim_steps > 0 and (global_optim_step % cfg.ema_update_every_n_optim_steps) == 0:
                        ema.update(model, force=True)

            total_train_loss += float(total_loss.item())
            global_batch_step += 1

            # ----------------- postfix -----------------
            postfix_main = {
                "loss": f"{total_train_loss / (batch_idx + 1):.5f}",
                "clip": f"{float(loss_clip.item()):.4f}",
            }

            base_loss_v = loss_parts.get("base_loss", torch.tensor(0.0, device=device))  # keep this
            postfix_main["base"] = f"{float(base_loss_v.item()):.3f}"

            if cfg.use_regression_teachers:
                if loss_reg_by_layer:
                    for _i, layer in enumerate(list(loss_reg_by_layer.keys())[:4]):
                        lam = float(teacher_specs[layer]["lam"])
                        postfix_main[f"r{layer}"] = f"{float((lam * loss_reg_by_layer[layer]).item()):.4f}"
                else:
                    postfix_main["rΣ"] = f"{float(reg_term.item()):.3f}"

            # Build line2 every batch (cheap), persist last non-empty
            line2_parts = []

            if cfg.use_adv_triplet:
                line2_parts.append(f"adv={last_adv_loss_value:.3f}")
                line2_parts.append(
                    f"a0,a1,an={last_adv_sims[0]:.3f},{last_adv_sims[1]:.3f},{last_adv_sims[2]:.3f}"
                )

            if cfg.use_ko_config:
                line2_parts.append(f"decatt={(total_decatt_loss / (batch_idx + 1)):3e}")
                line2_parts.append(f"kproj={(total_kproj_loss / (batch_idx + 1)):.5f}")

            if cfg.use_gap_schedule:
                u_v  = loss_parts.get("LUniform",  torch.tensor(0.0, device=device))
                a_v  = loss_parts.get("LAlign",    torch.tensor(0.0, device=device))
                xu_v = loss_parts.get("LXUniform", torch.tensor(0.0, device=device))
                line2_parts.append(
                    f"U,wU={float(u_v.item()):.3f},{float(getattr(contrastive_loss,'w_uniform',0.0)):.4f}"
                )
                line2_parts.append(
                    f"A,wA={float(a_v.item()):.3f},{float(getattr(contrastive_loss,'w_align',0.0)):.4f}"
                )
                line2_parts.append(
                    f"XU,wXU={float(xu_v.item()):.3f},{float(getattr(contrastive_loss,'w_xuniform',0.0)):.4f}"
                )

            if cfg.use_ema and (ema is not None):
                d = ema.debug_delta(model)
                line2_parts.append(f"ema=u{ema.num_updates} d{d:.2e}")

            new_line2 = "  ".join([p for p in line2_parts if p])
            if new_line2:
                _last_line2 = new_line2

            # ONE postfix update (keep it small-ish)
            progress_bar.set_postfix(postfix_main)

            # Occasionally print verbose stuff WITHOUT touching the bar line
            if _last_line2 and ((batch_idx + 1) % detail_every == 0):
                tqdm.write(
                    f"[Other Losses: Epoch {epoch_idx} Batch {batch_idx+1}/{len(train_dataloader)}] {_last_line2}"
                )
            # -----------------

        #progress_bar.close()
        
        avg_train_loss = total_train_loss / max(1, len(train_dataloader))
        epoch_train_acc = float(sum(train_accs) / max(1, len(train_accs)))
        epoch_train_f1  = float(sum(train_f1s)  / max(1, len(train_f1s)))

        if cfg.use_ko_config:
            mean_decatt = total_decatt_loss / max(1, len(train_dataloader))
            mean_kproj = total_kproj_loss / max(1, len(train_dataloader))
            with open(ko_log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch_idx}\t{mean_decatt:.8e}\t{mean_kproj:.8e}\n")

        if cfg.use_adv_triplet:
            s0, s1, sn = last_adv_sims
            with open(adv_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch_idx}\t{global_batch_step}\t{batch_idx}\t{last_adv_loss_value:.6f}\t{cfg.adv_lambda:.6f}\t"
                    f"{s0:.6f}\t{s1:.6f}\t{sn:.6f}\n"
                )  

        # per-epoch scheduler stepping (test -- NOT recommended!)
        if scheduler is not None and cfg.scheduler_step_mode == "per_epoch":
            scheduler.step()
        
        # END TRAIN --------------------------------------------------------------        

        # ============================================================
        # Validation (raw model only, not optional EMA)
        # ============================================================
        model.eval()
        total_val_loss = 0.0
        val_accs, val_f1s = [], []
        val_logits_diag = []
        val_logits_off = []

        print("Running Validation...")
        with torch.no_grad():
            for images, texts in val_dataloader:
                images = images.to(device, non_blocking=True, dtype=image_dtype)
                texts = texts.to(device, non_blocking=True)

                with autocast():
                    img_feats = model.encode_image(images).float()
                    txt_feats = model.encode_text(texts).float()
                    loss_clip_val, _vp = contrastive_loss(img_feats, txt_feats, return_parts=True)

                    img_norm = F.normalize(img_feats, dim=-1)
                    txt_norm = F.normalize(txt_feats, dim=-1)
                    logits_matrix = (img_norm @ txt_norm.t()) / getattr(contrastive_loss, "temperature", 0.07)

                    diag = logits_matrix.diag()
                    off = logits_matrix[~torch.eye(logits_matrix.size(0), dtype=torch.bool, device=logits_matrix.device)]
                    val_logits_diag.append(float(diag.mean().item()))
                    val_logits_off.append(float(off.mean().item()) if off.numel() > 0 else float("nan"))

                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                val_acc, val_f1 = calculate_metrics(logits_matrix, ground_truth)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)

                total_val_loss += float(loss_clip_val.item())

        # ============================================================
        # Stats & Logging
        # ============================================================
        avg_val_loss = total_val_loss / max(1, len(val_dataloader))
        epoch_val_acc = float(sum(val_accs) / max(1, len(val_accs)))
        epoch_val_f1  = float(sum(val_f1s)  / max(1, len(val_f1s)))

        epoch_logits_diag_train = float(sum(batch_logits_diag) / max(1, len(batch_logits_diag)))
        epoch_logits_off_train  = float(sum(batch_logits_off) / max(1, len(batch_logits_off)))
        epoch_logits_diag_val   = float(sum(val_logits_diag) / max(1, len(val_logits_diag)))
        epoch_logits_off_val    = float(sum(val_logits_off) / max(1, len(val_logits_off)))

        gapm = compute_gapness_metrics(model, val_dataloader, device=device, max_batches=30)
        print(Fore.CYAN + f"[Gapness] e{epoch_idx}: centroid_gap_l2={gapm['centroid_gap_l2']:.4f} linsep_acc={gapm['linsep_acc']:.4f} n={gapm['n']}" + Style.RESET_ALL)
        with open(gap_log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch_idx}\t{gapm['n']}\t{gapm['centroid_gap_l2']:.6f}\t{gapm['linsep_acc']:.6f}\n")

        print(Fore.YELLOW + "======================== STATS =============================")
        print(Fore.YELLOW + f"Epoch {epoch_idx}/{cfg.epochs-1} - Val Acc: {epoch_val_acc:.4f}, Val F1: {epoch_val_f1:.4f}")
        print(Fore.YELLOW + f"Epoch {epoch_idx}/{cfg.epochs-1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(Fore.YELLOW + f"Logits (train) diag={epoch_logits_diag_train:.4f} off={epoch_logits_off_train:.4f} gap={epoch_logits_diag_train-epoch_logits_off_train:.4f}")
        print(Fore.YELLOW + f"Logits (val)   diag={epoch_logits_diag_val:.4f} off={epoch_logits_off_val:.4f} gap={epoch_logits_diag_val-epoch_logits_off_val:.4f}")
        if ema is not None:
            print(Fore.YELLOW + f"EMA updates={ema.num_updates} last_decay_eff={ema.last_decay_eff:.6f} debug_delta={ema.debug_delta(model):.2e}")
        print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)

        with open(os.path.join(cfg.text_logs_folder, "training_log.txt"), "a", encoding="utf-8") as f:
            f.write("======================== STATS =============================\n")
            f.write(f"epoch={epoch_idx}\n")
            f.write(f"val_acc={epoch_val_acc:.6f}\tval_f1={epoch_val_f1:.6f}\n")
            f.write(f"train_acc={epoch_train_acc:.6f}\ttrain_f1={epoch_train_f1:.6f}\n")
            f.write(f"train_loss={avg_train_loss:.6f}\tval_loss={avg_val_loss:.6f}\n")
            f.write(f"logits_train_diag={epoch_logits_diag_train:.6f}\tlogits_train_off={epoch_logits_off_train:.6f}\n")
            f.write(f"logits_val_diag={epoch_logits_diag_val:.6f}\tlogits_val_off={epoch_logits_off_val:.6f}\n")
            if ema is not None:
                f.write(f"ema_updates={ema.num_updates}\tema_last_decay_eff={ema.last_decay_eff:.6f}\tema_debug_delta={ema.debug_delta(model):.6e}\n")
            f.write("============================================================\n")

        # ============================================================
        # Save epoch-level histories + plots
        # ============================================================
        epoch_ids.append(int(epoch_idx))
        training_losses.append(float(avg_train_loss))
        validation_losses.append(float(avg_val_loss))

        logits_diag_train_hist.append(float(epoch_logits_diag_train))
        logits_off_train_hist.append(float(epoch_logits_off_train))
        logits_diag_val_hist.append(float(epoch_logits_diag_val))
        logits_off_val_hist.append(float(epoch_logits_off_val))

        plot_training_info(
            epoch_ids=epoch_ids,
            training_losses=training_losses,
            validation_losses=validation_losses,
            logits_diag_train=logits_diag_train_hist,
            logits_off_train=logits_off_train_hist,
            logits_diag_val=logits_diag_val_hist,
            logits_off_val=logits_off_val_hist,
            plots_folder=cfg.plots_folder,
        )

        plot_gradient_norms(
            gradient_norms_unscaled=gradient_norms_unscaled,
            gradient_rms_unscaled=gradient_rms_unscaled,
            epoch=epoch_idx,
            plots_folder=cfg.plots_folder,
            topk=cfg.grad_plot_topk,
        )

        # ------------------------
        # VRAM logging + cleanup
        # ------------------------
        if cfg.log_cuda_memory and device.startswith("cuda"):
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(Fore.CYAN + f"[CUDA] alloc={alloc:.1f} MiB  reserved={reserv:.1f} MiB  peak={peak:.1f} MiB" + Style.RESET_ALL)
            with open(os.path.join(cfg.text_logs_folder, "cuda_mem_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"epoch={epoch_idx}\talloc={alloc:.1f}MiB\treserved={reserv:.1f}MiB\tpeak={peak:.1f}MiB\n")

        if cfg.cuda_empty_cache_each_epoch and device.startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()

        # ------------------------
        # Save resume state
        # ------------------------
        if cfg.save_resume_state:
            save_resume_state(
                cfg=cfg,
                epoch_idx=epoch_idx,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                global_optim_step=global_optim_step,
            )
            bp, mp = _resume_paths(cfg)
            print(Fore.CYAN + f"[ResumeSave] bundle={bp} model={mp}" + Style.RESET_ALL)

        # ------------------------
        # Save checkpoints (RAW + EMA)
        # ------------------------
        print(Fore.CYAN + "Saving checkpoints..." + Style.RESET_ALL)

        ModelSaver(
            model=model,
            epoch=epoch_idx,
            device=device,
            ft_checkpoints_folder=cfg.ft_checkpoints_folder,
            state_dict_cpu=None,
            tag="raw",
            save_fp16=cfg.save_checkpoints_fp16,
            save_full=cfg.model_save_full,
            save_dict=cfg.model_save_dict,
        )

        if cfg.save_ema_checkpoint and ema is not None:
            ema_sd = ema.state_dict_cpu()
            ModelSaver(
                model=model,
                epoch=epoch_idx,
                device=device,
                ft_checkpoints_folder=cfg.ft_checkpoints_folder,
                state_dict_cpu=ema_sd,
                tag="ema",
                save_fp16=cfg.save_checkpoints_fp16,
                save_full=cfg.model_save_full,
                save_dict=cfg.model_save_dict,
            )

        print(Fore.GREEN + f"Model saved to {cfg.ft_checkpoints_folder}" + Style.RESET_ALL)
        
        # ============================================================
        # POST-EPOCH evals: Typographic Attack, Linear Probe, Zero-Shot
        # ============================================================
        if cfg.run_tiny_benchmark and (cfg.probe_every_n_epochs > 0) and ((epoch_idx % cfg.probe_every_n_epochs) == 0):
            folders = [p.format(dataset=cfg.tiny_dataset) for p in cfg.tiny_folders]
            bench = run_tiny_benchmark(
                model=model,
                clip=clip,
                preprocess=preprocess,
                device=device,
                folders=folders,
                choices=list(cfg.tiny_choices),
                correct_choice_idx=0,
                batch_size=cfg.tiny_batch_size,
            )
            for fol, stats in bench.items():
                with open(bench_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch_idx}\t{fol}\t{stats.get('n',0)}\t{stats.get('acc',float('nan')):.6f}\t"
                        f"{stats.get('mean_margin',float('nan')):.6f}\t{stats.get('mean_logit_correct',float('nan')):.6f}\t"
                        f"{stats.get('mean_logit_othermax',float('nan')):.6f}\n"
                    )
                if fol in tiny_history:
                    tiny_history[fol]["epochs"].append(int(epoch_idx))
                    tiny_history[fol]["acc"].append(float(stats.get("acc", float("nan"))))
                    tiny_history[fol]["margin"].append(float(stats.get("mean_margin", float("nan"))))
            plot_tiny_benchmark(tiny_history, cfg.plots_folder)

        if cfg.run_quick_probe and (cfg.probe_every_n_epochs > 0) and ((epoch_idx % cfg.probe_every_n_epochs) == 0):
            _run_quick_probe_epoch(epoch_idx)

        # ============================================================
        # Teacher rebuild + logging + plots
        # ============================================================
        rebuilt_by_layer: Dict[int, int] = {int(layer): 0 for layer in teacher_specs.keys()}
        reason_by_layer: Dict[int, str] = {int(layer): "keep" for layer in teacher_specs.keys()}

        model.eval()

        if cfg.use_regression_teachers:
            # skip disabled teachers (lam == 0) everywhere in maintenance
            active_layers = [
                int(layer) for layer, spec in teacher_specs.items()
                if float(spec.get("lam", 0.0)) > 0.0
            ]

            # cache tcos from the "decide" phase so we don't eval twice
            tcos_pre_by_layer: Dict[int, Optional[float]] = {int(layer): None for layer in active_layers}


            # manual rebuild schedule (epoch allowlist)
            manual_epochs_set = set(int(e) for e in (getattr(cfg, "teacher_rebuild_epochs", []) or []))
            manual_mode = (len(manual_epochs_set) > 0)
            manual_mode_kind = str(getattr(cfg, "teacher_rebuild_epochs_mode", "only")).lower().strip()
            if manual_mode_kind not in ("only", "extra"):
                manual_mode_kind = "only"

            if manual_mode:
                print(
                    Fore.CYAN
                    + f"[TeacherPolicy] manual_epochs={sorted(list(manual_epochs_set))} "
                      f"mode={manual_mode_kind} auto_every_epoch={bool(cfg.teacher_retrain_every_epoch)}"
                    + Style.RESET_ALL
                )

            # -------------------------
            # Decide + rebuild per layer
            # -------------------------
            with torch.no_grad():
                for layer in active_layers:
                    spec = teacher_specs[int(layer)]
                    t = cls_patch_teachers.get(int(layer), None)

                    # no teacher -> must rebuild
                    if t is None:
                        do_rebuild, reason = True, "teacher_none"
                        tcos_cur = None

                    else:
                        # manual schedule takes priority
                        if manual_mode and (epoch_idx in manual_epochs_set):
                            do_rebuild, reason = True, f"manual_epoch={epoch_idx}"
                            tcos_cur = None

                        elif cfg.teacher_retrain_every_epoch:
                            # if manual_mode_kind=="only", we suppress this below.
                            do_rebuild, reason = True, "every_epoch=True"
                            tcos_cur = None

                        else:
                            # Evaluate cosine for monitoring (cheap; keep signal even if we don't rebuild)
                            tcos_cur = evaluate_teacher_cosine(
                                model=model,
                                visual=model.visual,
                                dataloader=val_dataloader,
                                teacher=t,
                                device=device,
                                reg_threshold=spec["reg_threshold"],
                                teacher_layer=int(layer),
                                max_batches=50,
                                projector=jl_projectors.get(int(layer), None),
                                cls_mix=float(spec.get("cls_mix", 0.0)),
                                cls_mix_use_reg=bool(spec.get("cls_mix_use_reg", False)),
                            )

                            if tcos_cur is not None:
                                if (best_teacher_cos[int(layer)] is None) or (float(tcos_cur) > float(best_teacher_cos[int(layer)])):
                                    best_teacher_cos[int(layer)] = float(tcos_cur)

                            do_rebuild, reason = should_rebuild_teacher(
                                epoch_idx=epoch_idx,
                                cfg=cfg,
                                last_teacher_update_epoch=last_teacher_update_epoch[int(layer)],
                                current_teacher_cos=tcos_cur,                  # should_rebuild_teacher handles None
                                best_teacher_cos=best_teacher_cos[int(layer)], # may also be None early
                            )

                        # if manual mode is "only", suppress ALL auto rebuilds on non-manual epochs
                        if manual_mode and (epoch_idx not in manual_epochs_set) and (manual_mode_kind == "only"):
                            # keep tcos_cur for logging, but do not rebuild
                            if do_rebuild:
                                reason = f"manual_hold|suppressed:{reason}"
                            do_rebuild = False

                        # if manual mode is "extra", allow auto rebuilds too

                    reason_by_layer[int(layer)] = str(reason)
                    tcos_pre_by_layer[int(layer)] = (float(tcos_cur) if tcos_cur is not None else None)

                    if do_rebuild:
                        t_new = build_cls_patch_regression_teacher(
                            model=model,
                            visual=model.visual,
                            dataloader=val_dataloader,
                            device=device,
                            max_samples=cfg.cls_patch_max_samples,
                            val_frac=cfg.cls_patch_val_frac,
                            cache_path=teacher_cache_paths[int(layer)],
                            reg_threshold=spec["reg_threshold"],
                            teacher_layer=int(layer),
                            teacher_seed=cfg.teacher_seed,
                            extra_image_loaders=extra_loaders,
                            force_rebuild=True,
                            cls_mix=float(spec.get("cls_mix", 0.0)),
                            cls_mix_use_reg=bool(spec.get("cls_mix_use_reg", False)),
                            is_reg_teacher=bool(spec.get("is_reg_teacher", False)),
                            use_reg_whitening=bool(spec.get("use_reg_whitening", False)),
                        )
                        cls_patch_teachers[int(layer)] = t_new

                        # only mark rebuild success / advance cooldown if teacher is usable
                        if t_new is not None:
                            last_teacher_update_epoch[int(layer)] = int(epoch_idx)
                            rebuilt_by_layer[int(layer)] = 1
                        else:
                            rebuilt_by_layer[int(layer)] = 0
                            reason_by_layer[int(layer)] = f"{reason_by_layer[int(layer)]}|rebuild_failed_none"


            # ---------------------------------
            # Post-(rebuild) evaluate + LOG rows
            # ---------------------------------
            payload_by_layer: Dict[int, Dict[str, Any]] = {}

            with torch.no_grad():
                for layer in active_layers:
                    spec = teacher_specs[int(layer)]
                    t = cls_patch_teachers.get(int(layer), None)

                    if t is None:
                        payload_by_layer[int(layer)] = {
                            "tcos": None,
                            "best": (float(best_teacher_cos[int(layer)]) if best_teacher_cos[int(layer)] is not None else None),
                            "fit_cos": None,
                            "fit_mse": None,
                            "n": 0,
                            "reg_threshold": str(spec["reg_threshold"]),
                            "lam": float(spec["lam"]),
                            "jl": spec.get("jl", {}),
                            "cls_mix": float(spec.get("cls_mix", 0.0)),
                        }
                        continue

                    # avoid double eval if we didn't rebuild; reuse pre-eval
                    if rebuilt_by_layer[int(layer)] == 0 and tcos_pre_by_layer[int(layer)] is not None:
                        tcos_post = tcos_pre_by_layer[int(layer)]
                    else:
                        tcos_post = evaluate_teacher_cosine(
                            model=model,
                            visual=model.visual,
                            dataloader=val_dataloader,
                            teacher=t,
                            device=device,
                            reg_threshold=spec["reg_threshold"],
                            teacher_layer=int(layer),
                            max_batches=50,
                            projector=jl_projectors.get(int(layer), None),
                            cls_mix=float(spec.get("cls_mix", 0.0)),
                            cls_mix_use_reg=bool(spec.get("cls_mix_use_reg", False)),
                        )
                        tcos_post = (float(tcos_post) if tcos_post is not None else None)

                    # best_cos update guarded
                    if tcos_post is not None:
                        if (best_teacher_cos[int(layer)] is None) or (tcos_post > float(best_teacher_cos[int(layer)])):
                            best_teacher_cos[int(layer)] = float(tcos_post)

                    fitcos, fitmse, n_pairs = _tstats(t)
                    payload_by_layer[int(layer)] = {
                        "tcos": tcos_post,
                        "best": (float(best_teacher_cos[int(layer)]) if best_teacher_cos[int(layer)] is not None else None),
                        "fit_cos": float(fitcos) if fitcos == fitcos else None,  # keep NaN out of JSON
                        "fit_mse": float(fitmse) if fitmse == fitmse else None,
                        "n": int(n_pairs),
                        "reg_threshold": str(spec["reg_threshold"]),
                        "lam": float(spec["lam"]),
                        "jl": spec.get("jl", {}),
                        "cls_mix": float(spec.get("cls_mix", 0.0)),
                    }

            # append only active layers (keeps log clean)
            _append_teacher_log_rows(
                teacher_log_path=teacher_log_path,
                epoch_i=int(epoch_idx),
                rebuilt_by_layer={int(l): int(rebuilt_by_layer[int(l)]) for l in active_layers},
                reason_by_layer={int(l): str(reason_by_layer[int(l)]) for l in active_layers},
                payload_by_layer=payload_by_layer,
                last_logged_epoch_by_layer=teacher_last_logged_epoch,
                nd=5,
            )

            # keep the in-memory "last logged" map in sync
            for l in active_layers:
                teacher_last_logged_epoch[int(l)] = int(epoch_idx)

            # -------------------------
            # Teacher histories + plots
            # -------------------------
            for layer in active_layers:
                payload = payload_by_layer.get(int(layer), {}) or {}
                tcos = payload.get("tcos", None)
                fitcos = payload.get("fit_cos", None)
                fitmse = payload.get("fit_mse", None)

                teacher_hist[int(layer)]["epochs"].append(int(epoch_idx))
                teacher_hist[int(layer)]["tcos"].append(float(tcos) if tcos is not None else float("nan"))
                teacher_hist[int(layer)]["fit_cos"].append(float(fitcos) if fitcos is not None else float("nan"))
                teacher_hist[int(layer)]["fit_mse"].append(float(fitmse) if fitmse is not None else float("nan"))

                plot_teacher_info(
                    teacher_hist[int(layer)]["epochs"],
                    teacher_hist[int(layer)]["tcos"],
                    teacher_hist[int(layer)]["fit_cos"],
                    teacher_hist[int(layer)]["fit_mse"],
                    cfg.plots_folder,
                    out_name=f"teacher_stats_b{int(layer)}.png",
                )

        model.train()

        
        print(Fore.GREEN + f"EPOCH {epoch_idx} COMPLETE." + Style.RESET_ALL)
        print("------------------------------------")
        # END VAL, END EPOCH (phew...!)
        # ------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainConfig()
    cfg = maybe_load_cfg_from_json(load_json_config, cfg)
    main(cfg)