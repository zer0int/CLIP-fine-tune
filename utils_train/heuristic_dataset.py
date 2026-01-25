# utils_train/heuristic_dataset.py
from __future__ import annotations

import os
import re
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
from colorama import Fore, Style


_JUNK_DIR_NAMES = {
    "train", "training", "val", "valid", "validation", "test", "tests",
    "tmp", "temp", "temporary", "meta",
    "new folder", "newfolder", "new_folder",
    "asdf", "asdfghj", "qwerty",
    "misc", "stuff", "unsorted", "unknown",
    "__pycache__", ".git", ".svn",
}

_WNID_RE = re.compile(r"^n\d{8}$")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_HF_TEXTCOL_DECISIONS: Dict[str, str] = {}

_BASENAME_INDEX_CACHE: Dict[str, Tuple[Dict[str, str], int]] = {}
_BASENAME_INDEX_EMPTY: set[str] = set()

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)



# ----------------------------
# API
# ----------------------------
@dataclass
class HeuristicDatasetSpec:
    path_or_hf_dataset: str = ""
    text_labels_path: Optional[str] = None

    split: Optional[str] = None
    image_root: Optional[str] = None
    seed: int = 42

    pretok_batch_size: int = 4096
    max_unique_texts_for_table: int = 500_000

    deterministic_text: bool = False


def build_heuristic_dataset(
    clip,
    spec: HeuristicDatasetSpec,
    transform=None,
    tokenize_fn=None,
    verbose: bool = True,
) -> Dataset:
    """
    Build a PyTorch Dataset that returns (image_tensor, token_ids[77]).
    Attempts local-dir heuristics first if path exists, otherwise HF dataset.
    """
    if tokenize_fn is None:
        try:
            tokenize_fn = clip.tokenize
        except Exception as e:
            raise RuntimeError(
                "tokenize_fn was None and importing gmpclipregression.clip.tokenize failed. "
                f"Pass tokenize_fn explicitly. Underlying error: {e}"
            )

    log = _make_logger(verbose)
    
    porh = (spec.path_or_hf_dataset or "").strip()
    cache_ns = f"hf::{porh}"
    tlp = spec.text_labels_path
    if tlp is not None and str(tlp).strip() == "":
        tlp = None

    log.cyan(f"[HeuristicDS] starting build")
    log.cyan(f"[HeuristicDS] path_or_hf_dataset={repr(porh)}")
    log.cyan(f"[HeuristicDS] text_labels_path={repr(tlp)}")

    # If labels path exists (or provided), treat as metadata-driven.
    if tlp is not None:
        meta_path = tlp
        if not os.path.exists(meta_path):
            log.red(f"[HeuristicDS] metadata file does not exist: {meta_path}")
            raise FileNotFoundError(meta_path)

        image_root = _resolve_image_root(spec, porh, meta_path, log)
        samples = _load_metadata_pairs(meta_path, image_root=image_root, log=log)

        ds = HeuristicImageTextDataset(
            samples=samples,
            transform=transform,
            tokenize_fn=tokenize_fn,
            pretok_batch_size=spec.pretok_batch_size,
            max_unique_texts_for_table=spec.max_unique_texts_for_table,
            deterministic_text=spec.deterministic_text,
            log=log,
        )
        log.green(f"[HeuristicDS] success via metadata file ({os.path.basename(meta_path)}) -> n={len(ds)}")
        return ds

    # No labels path provided: decide local vs HF
    if porh != "" and _looks_like_existing_path(porh):
        local_root = os.path.abspath(porh)
        log.cyan(f"[HeuristicDS] detected local path -> {local_root}")

        # try special binary dataset formats first (e.g., CIFAR python pickles)
        special_ds = _try_build_special_local_dataset(
            local_root=local_root,
            spec=spec,
            transform=transform,
            tokenize_fn=tokenize_fn,
            log=log,
        )
        if special_ds is not None:
            log.green(f"[HeuristicDS] success via special local loader -> n={len(special_ds)}")
            return special_ds

        # Fall back to the usual “image files on disk” heuristics
        samples = _infer_local_samples(local_root, log=log)
        ds = HeuristicImageTextDataset(
            samples=samples,
            transform=transform,
            tokenize_fn=tokenize_fn,
            pretok_batch_size=spec.pretok_batch_size,
            max_unique_texts_for_table=spec.max_unique_texts_for_table,
            deterministic_text=spec.deterministic_text,
            log=log,
        )
        log.green(f"[HeuristicDS] success via local inference -> n={len(ds)}")
        return ds


    # Otherwise treat as HF dataset id (or error if empty)
    if porh == "":
        log.red("[HeuristicDS] path_or_hf_dataset is empty AND text_labels_path is empty -> nothing to load.")
        raise ValueError("No dataset source provided (both path_or_hf_dataset and text_labels_path are empty).")

    # HF path now returns a HF-wrapped dataset (no disk export)
    log.cyan(f"[HeuristicDS] attempting Hugging Face datasets.load_dataset({repr(porh)})")

    dset, image_col, text_col, label_col, class_names = _infer_hf_dataset_info(porh, split=spec.split, log=log)

    ds = HeuristicHFImageTextDataset(
        hf_dataset=dset,
        image_col=image_col,
        text_col=text_col,
        label_col=label_col,
        class_names=class_names,
        transform=transform,
        tokenize_fn=tokenize_fn,
        pretok_batch_size=spec.pretok_batch_size,
        max_unique_texts_for_table=spec.max_unique_texts_for_table,
        deterministic_text=spec.deterministic_text,
        log=log,
        cache_namespace=cache_ns,
    )
    log.green(f"[HeuristicDS] success via Hugging Face inference -> n={len(ds)}")
    return ds


# ----------------------------
# Dataset implementation
# ----------------------------
class HeuristicCIFARPickleDataset(Dataset):
    """
    CIFAR-10/100 'python' pickles (no image files on disk).
    Produces (image_tensor, text_token_ids[77]) using the same token-table trick.

    - CIFAR-100: uses fine_label_names as text
    - CIFAR-10:  uses label_names as text
    """
    def __init__(
        self,
        root: str,
        kind: str,                 # 'cifar10' | 'cifar100'
        split: str,                # 'train' | 'test'
        transform=None,
        tokenize_fn=None,
        pretok_batch_size: int = 4096,
        log=None,
    ):
        assert kind in {"cifar10", "cifar100"}
        assert split in {"train", "test"}

        self.root = os.path.abspath(root)
        self.kind = kind
        self.split = split
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self.pretok_batch_size = int(pretok_batch_size)
        self.log = log or _make_logger(False)

        if self.tokenize_fn is None:
            raise RuntimeError("HeuristicCIFARPickleDataset requires tokenize_fn (CLIP tokenizer).")

        # python2 pickle compatibility
        import pickle

        def _load_pickle(path: str):
            with open(path, "rb") as f:
                # CIFAR pickles are python2; 'latin1' is the standard py3 bridge
                return pickle.load(f, encoding="latin1")

        if self.kind == "cifar100":
            meta = _load_pickle(os.path.join(self.root, "meta"))
            fine_names = list(meta.get("fine_label_names", []))
            if not fine_names:
                raise RuntimeError("CIFAR-100 meta missing fine_label_names.")

            blob = _load_pickle(os.path.join(self.root, self.split))
            data = blob.get("data", None)
            labels = blob.get("fine_labels", None)
            if data is None or labels is None:
                raise RuntimeError("CIFAR-100 split missing 'data' or 'fine_labels'.")

            self.class_names = fine_names
            self.labels = labels

        else:  # cifar10
            meta = _load_pickle(os.path.join(self.root, "batches.meta"))
            label_names = list(meta.get("label_names", []))
            if not label_names:
                raise RuntimeError("CIFAR-10 batches.meta missing label_names.")

            if self.split == "test":
                blob = _load_pickle(os.path.join(self.root, "test_batch"))
                parts = [blob]
            else:
                parts = []
                for i in range(1, 6):
                    p = os.path.join(self.root, f"data_batch_{i}")
                    if os.path.isfile(p):
                        parts.append(_load_pickle(p))
                if len(parts) == 0:
                    raise RuntimeError("CIFAR-10: no data_batch_* files found.")

            # concat
            data_list = []
            label_list = []
            for b in parts:
                d = b.get("data", None)
                l = b.get("labels", None)
                if d is None or l is None:
                    raise RuntimeError("CIFAR-10 batch missing 'data' or 'labels'.")
                data_list.append(d)
                label_list.extend(l)

            import numpy as np
            data = np.concatenate(data_list, axis=0)

            self.class_names = label_names
            self.labels = label_list

        # data shape: [N, 3072] uint8-like
        import numpy as np
        arr = np.asarray(data)
        if arr.ndim != 2 or arr.shape[1] != 3072:
            raise RuntimeError(f"CIFAR pickle data has unexpected shape: {arr.shape}")

        # store as uint8 for PIL
        self.data = arr.astype(np.uint8, copy=False)
        n = self.data.shape[0]
        if len(self.labels) != n:
            raise RuntimeError(f"CIFAR labels length mismatch: labels={len(self.labels)} data={n}")

        self.log.cyan(f"[HeuristicDS] CIFAR: kind={self.kind} split={self.split} n={n} classes={len(self.class_names)}")

        # --- token table (unique class names only) ---
        unique_labels = [str(s).strip() for s in self.class_names]
        unique_labels = [s if s else "" for s in unique_labels]
        if "" not in unique_labels:
            unique_labels.append("")

        tok_chunks: List[torch.Tensor] = []
        for start in range(0, len(unique_labels), self.pretok_batch_size):
            chunk = unique_labels[start:start + self.pretok_batch_size]
            tok = self.tokenize_fn(chunk, truncate=True)  # [B,77] CPU
            tok_chunks.append(tok)

        self._token_table = torch.cat(tok_chunks, dim=0).contiguous()
        self._label_to_tokidx = {i: i for i in range(len(self.class_names))}
        self._unk_tokidx = unique_labels.index("")

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx: int):
        import numpy as np
        from PIL import Image

        flat = self.data[idx]  # [3072]
        # CIFAR layout: 1024 R, 1024 G, 1024 B
        img = flat.reshape(3, 32, 32).transpose(1, 2, 0)  # HWC
        pil = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            pil = self.transform(pil)

        lab = self.labels[idx]
        try:
            li = int(lab)
            tok_i = self._label_to_tokidx.get(li, self._unk_tokidx)
        except Exception:
            tok_i = self._unk_tokidx

        text_tok = self._token_table[tok_i]
        return pil, text_tok


class HeuristicHFImageTextDataset(Dataset):
    """
    Wraps a Hugging Face datasets.Dataset directly (no disk export),
    while preserving your fast "unique text -> token table" trick.

    Returns:
        image: transformed image tensor
        text:  token ids [77] LongTensor (CPU)
    """
    def __init__(
        self,
        hf_dataset,
        image_col: str,
        text_col: Optional[str],
        label_col: Optional[str],
        class_names: Optional[List[str]],
        transform=None,
        tokenize_fn=None,
        pretok_batch_size: int = 4096,
        max_unique_texts_for_table: int = 500_000,
        deterministic_text: bool = False,
        log=None,
        cache_namespace: Optional[str] = None,
    ):
        self.hf_dataset = hf_dataset
        self.image_col = image_col
        self.text_col = text_col
        self.label_col = label_col
        self.class_names = class_names
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self.pretok_batch_size = int(pretok_batch_size)
        self.max_unique_texts_for_table = int(max_unique_texts_for_table)
        self.deterministic_text = bool(deterministic_text)
        self.log = log or _make_logger(False)
        self.cache_namespace = cache_namespace



        if self.tokenize_fn is None:
            raise RuntimeError("HeuristicHFImageTextDataset requires tokenize_fn (CLIP tokenizer).")

        n = len(self.hf_dataset)
        if n == 0:
            raise RuntimeError("HF dataset split is empty.")

        # --- Build per-row candidate strings and the unique token table ---
        labels_per_row: List[List[str]] = []
        all_labels: List[str] = []

        self.log.cyan(f"[HeuristicDS] HF wrapper: scanning text/labels for {n} rows (for token table)")

        for i in range(n):
            row = self.hf_dataset[i]

            if self.text_col is not None:
                cands = _coerce_text_candidates(row.get(self.text_col, ""))
            else:
                lv = row.get(self.label_col, "")
                if self.class_names is not None:
                    try:
                        lv_i = int(lv)
                        cands = [self.class_names[lv_i]]
                    except Exception:
                        cands = [str(lv)]
                else:
                    cands = [str(lv)]

            # normalize like in your COCO loader
            norm_cands = []
            for s in (cands or [""]):
                if s is None:
                    continue
                ss = str(s).strip()
                if ss:
                    norm_cands.append(ss)
            if len(norm_cands) == 0:
                norm_cands = [""]

            labels_per_row.append(norm_cands)
            all_labels.extend(norm_cands)

            if (i + 1) % 5000 == 0:
                self.log.cyan(f"[HeuristicDS] HF wrapper: scanned {i+1}/{n} rows...")

        unique_labels = sorted(set(all_labels))
        if "" not in unique_labels:
            unique_labels.append("")

        self.log.cyan(f"[HeuristicDS] HF wrapper: unique_texts={len(unique_labels)}")

        if len(unique_labels) > self.max_unique_texts_for_table:
            self.log.yellow(
                "[HeuristicDS] WARNING: HF unique_texts is huge; falling back to on-the-fly tokenization cache."
            )
            self._use_token_table = False
            self._tok_cache: Dict[str, torch.Tensor] = {}
            self._tok_cache_max = 50_000
            self._labels_per_row = labels_per_row
            return

        self._use_token_table = True
        self._label_to_idx = {s: i for i, s in enumerate(unique_labels)}

        tok_chunks: List[torch.Tensor] = []
        for start in range(0, len(unique_labels), self.pretok_batch_size):
            chunk = unique_labels[start:start + self.pretok_batch_size]
            tok = self.tokenize_fn(chunk, truncate=True)  # [B,77] CPU
            tok_chunks.append(tok)

        self._token_table = torch.cat(tok_chunks, dim=0).contiguous()

        self._cand_token_idxs: List[List[int]] = []
        unk = self._label_to_idx[""]
        for cands in labels_per_row:
            self._cand_token_idxs.append([self._label_to_idx.get(s, unk) for s in cands])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        # bounded resample on decode failure (dead URLs etc.)
        max_decode_tries = 8

        def _decode_cell(x: Any) -> Optional[Image.Image]:
            # datasets.Image -> PIL
            if hasattr(x, "convert"):
                try:
                    return x.convert("RGB")
                except Exception:
                    return None

            # dict: {path, bytes} or url-ish keys
            if isinstance(x, dict):
                if "path" in x and x["path"] and os.path.exists(x["path"]):
                    try:
                        return _safe_open_image(x["path"])
                    except Exception:
                        return None
                if "bytes" in x and x["bytes"]:
                    try:
                        return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
                    except Exception:
                        return None

                # URL-ish dict keys
                for k in ["url", "image_url", "uri", "link", "href"]:
                    if k in x and x[k] and _looks_like_url(x[k]):
                        try:
                            return _safe_open_image_url(
                                x[k],
                                log=self.log,
                                cache_namespace=self.cache_namespace,
                            )
                        except Exception:
                            return None

            # string: local path OR URL
            if isinstance(x, str):
                s = x.strip()
                if _looks_like_url(s):
                    try:
                        return _safe_open_image_url(
                            s,
                            log=self.log,
                            cache_namespace=self.cache_namespace,
                        )
                    except Exception:
                        return None
                try:
                    if os.path.exists(s):
                        return _safe_open_image(s)
                except Exception:
                    return None

            # last resort: stringify -> local path
            try:
                s = str(x).strip()
                if os.path.exists(s):
                    return _safe_open_image(s)
            except Exception:
                pass

            return None

        # Try idx; if fail, resample a few indices.
        cur_idx = idx
        row = self.hf_dataset[cur_idx]
        pil = _decode_cell(row.get(self.image_col, None))

        if pil is None:
            for _ in range(max_decode_tries):
                ridx = random.randrange(0, len(self.hf_dataset))
                rrow = self.hf_dataset[ridx]
                rpil = _decode_cell(rrow.get(self.image_col, None))
                if rpil is not None:
                    cur_idx = ridx
                    row = rrow
                    pil = rpil
                    break

        if pil is None:
            raise RuntimeError(
                f"HF row {idx}: could not decode image from column {self.image_col} "
                f"(tried {1 + max_decode_tries} indices; supports PIL/dict(path|bytes)/local-path/URL)."
            )

        if self.transform is not None:
            pil = self.transform(pil)

        # if we resampled, use the *matching* text for that resampled row index
        if self._use_token_table:
            cand_idxs = self._cand_token_idxs[cur_idx]
            tok_i = cand_idxs[0] if self.deterministic_text else (
                random.choice(cand_idxs) if len(cand_idxs) > 1 else cand_idxs[0]
            )
            text_tok = self._token_table[tok_i]
            return pil, text_tok

        cand_list = self._labels_per_row[cur_idx]
        s = cand_list[0] if self.deterministic_text else (
            random.choice(cand_list) if len(cand_list) > 1 else cand_list[0]
        )
        if s not in self._tok_cache:
            tok = self.tokenize_fn([s])[0].contiguous()
            if len(self._tok_cache) >= self._tok_cache_max:
                self._tok_cache.clear()
            self._tok_cache[s] = tok
        return pil, self._tok_cache[s]



        # Try idx; if fail, resample a few indices.
        cur_idx = idx
        row = self.hf_dataset[cur_idx]
        pil = _decode_cell(row.get(self.image_col, None))

        if pil is None:
            for _ in range(max_decode_tries):
                ridx = random.randrange(0, len(self.hf_dataset))
                rrow = self.hf_dataset[ridx]
                rpil = _decode_cell(rrow.get(self.image_col, None))
                if rpil is not None:
                    cur_idx = ridx
                    row = rrow
                    pil = rpil
                    break

        if pil is None:
            raise RuntimeError(
                f"HF row {idx}: could not decode image from column {self.image_col} "
                f"(tried {1 + max_decode_tries} indices; supports PIL/dict(path|bytes)/local-path/URL)."
            )

        if self.transform is not None:
            pil = self.transform(pil)

        # if we resampled, use the *matching* text for that resampled row index
        if self._use_token_table:
            cand_idxs = self._cand_token_idxs[cur_idx]
            tok_i = cand_idxs[0] if self.deterministic_text else (
                random.choice(cand_idxs) if len(cand_idxs) > 1 else cand_idxs[0]
            )
            text_tok = self._token_table[tok_i]
            return pil, text_tok

        cand_list = self._labels_per_row[cur_idx]
        s = cand_list[0] if self.deterministic_text else (
            random.choice(cand_list) if len(cand_list) > 1 else cand_list[0]
        )
        if s not in self._tok_cache:
            tok = self.tokenize_fn([s])[0].contiguous()
            if len(self._tok_cache) >= self._tok_cache_max:
                self._tok_cache.clear()
            self._tok_cache[s] = tok
        return pil, self._tok_cache[s]


class HeuristicImageTextDataset(Dataset):
    """
    Generic (image_path, [text candidates]) dataset, with your fast pretoken table.

    Returns:
        image: transformed image tensor
        text:  token ids [77] LongTensor (CPU) (caller moves to device)
    """
    def __init__(
        self,
        samples: Sequence[Tuple[str, List[str]]],
        transform=None,
        tokenize_fn=None,
        pretok_batch_size: int = 4096,
        max_unique_texts_for_table: int = 500_000,
        deterministic_text: bool = False,
        log=None,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self.pretok_batch_size = int(pretok_batch_size)
        self.max_unique_texts_for_table = int(max_unique_texts_for_table)
        self.deterministic_text = bool(deterministic_text)
        self.log = log or _make_logger(False)

        if len(self.samples) == 0:
            raise RuntimeError("HeuristicImageTextDataset got 0 samples. Nothing usable was found.")

        # Normalize candidate strings, gather all
        labels_per_image: List[List[str]] = []
        all_labels: List[str] = []

        for img_path, cands in self.samples:
            norm_cands = []
            for s in (cands or [""]):
                if s is None:
                    continue
                ss = str(s).strip()
                if ss == "":
                    continue
                norm_cands.append(ss)
            if len(norm_cands) == 0:
                norm_cands = [""]  # defined behavior

            labels_per_image.append(norm_cands)
            all_labels.extend(norm_cands)

        unique_labels = sorted(set(all_labels))
        if "" not in unique_labels:
            unique_labels.append("")

        self.log.cyan(f"[HeuristicDS] n_samples={len(self.samples)}")
        self.log.cyan(f"[HeuristicDS] unique_texts={len(unique_labels)} (pretok_batch_size={self.pretok_batch_size})")

        if len(unique_labels) > self.max_unique_texts_for_table:
            self.log.yellow(
                "[HeuristicDS] WARNING: unique_texts is huge; building a full token table may be RAM-heavy.\n"
                f"             unique_texts={len(unique_labels)} > max_unique_texts_for_table={self.max_unique_texts_for_table}\n"
                "             Falling back to on-the-fly tokenization with a small cache."
            )
            self._use_token_table = False
            self._tok_cache: Dict[str, torch.Tensor] = {}
            self._tok_cache_max = 50_000  # cap
            self._labels_per_image = labels_per_image
            return

        self._use_token_table = True
        self._label_to_idx: Dict[str, int] = {s: i for i, s in enumerate(unique_labels)}

        # Tokenize unique labels in batches (fast)
        tok_chunks: List[torch.Tensor] = []
        for start in range(0, len(unique_labels), self.pretok_batch_size):
            chunk = unique_labels[start:start + self.pretok_batch_size]
            try:
                tok = self.tokenize_fn(chunk, truncate=True)  # [B,77] CPU LongTensor
            except Exception as e:
                raise RuntimeError(
                    "Tokenization failed while building token table. "
                    "This usually means tokenize_fn isn't the CLIP tokenizer you think it is.\n"
                    f"Underlying error: {e}"
                )
            tok_chunks.append(tok)

        self._token_table = torch.cat(tok_chunks, dim=0).contiguous()  # [n_unique,77] on CPU

        # Store per-sample candidate token indices
        self._cand_token_idxs: List[List[int]] = []
        unk = self._label_to_idx[""]
        for cands in labels_per_image:
            self._cand_token_idxs.append([self._label_to_idx.get(s, unk) for s in cands])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, cands = self.samples[idx]

        img = _safe_open_image(image_path)
        if self.transform is not None:
            img = self.transform(img)

        if self._use_token_table:
            cand_idxs = self._cand_token_idxs[idx]
            tok_i = cand_idxs[0] if self.deterministic_text else (
                random.choice(cand_idxs) if len(cand_idxs) > 1 else cand_idxs[0]
            )
            text_tok = self._token_table[tok_i]
            return img, text_tok

        # Fallback: tokenize on-the-fly with caching
        cand_list = self._labels_per_image[idx]
        s = cand_list[0] if self.deterministic_text else (
            random.choice(cand_list) if len(cand_list) > 1 else cand_list[0]
        )
        if s not in self._tok_cache:
            tok = self.tokenize_fn([s])[0].contiguous()
            if len(self._tok_cache) >= self._tok_cache_max:
                self._tok_cache.clear()
            self._tok_cache[s] = tok
        return img, self._tok_cache[s]

# ----------------------------
# ImageNet implementation
# ----------------------------
def _try_load_imagenet_label_map(root_dir: str, log) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Searches for common ImageNet label map artifacts and returns wnid -> human_name.
    Supports:
      - JSON {"0": ["n########", "name"], ...}
      - JSON {"n########": "name"} or {"n########": ["name", ...]}
      - TXT lines like: n########\\twords...  OR  n######## words...
      - meta.mat (ILSVRC devkit) if scipy is installed
    """
    candidates = _find_imagenet_map_candidates(root_dir)
    if len(candidates) == 0:
        log.yellow("[HeuristicDS] ImageNet WNIDs detected but no label-map files found.")
        return None, None

    # Try up to 10 candidates, pick the one with the most wnids
    best_map = None
    best_src = None
    best_n = 0

    to_try = candidates[:10]
    log.cyan(f"[HeuristicDS] ImageNet map: trying {len(to_try)} candidate files")
    for p in to_try:
        try:
            m = _load_imagenet_map_file(p, log=log)
            n = len(m) if m is not None else 0
            if n > best_n:
                best_n = n
                best_map = m
                best_src = p
        except Exception as e:
            log.yellow(f"[HeuristicDS] ImageNet map parse failed: {p} ({e})")

    if best_map is None or best_n < 10:
        log.yellow("[HeuristicDS] ImageNet map: no usable mapping parsed (or too few entries).")
        return None, None

    log.cyan(f"[HeuristicDS] ImageNet map: using {best_src} (entries={best_n})")
    return best_map, os.path.abspath(best_src)


def _find_imagenet_map_candidates(root_dir: str) -> List[str]:
    """
    Find likely label-map files (shallow-ish scan).
    """
    hits: List[str] = []
    want_substr = ["wnid", "synset", "class", "classes", "label", "labels", "meta", "imagenet", "index"]

    # Scan root and one level deep (fast); if your datasets stash these deeper, we can widen later.
    roots = [root_dir]
    try:
        for d in os.listdir(root_dir):
            sd = os.path.join(root_dir, d)
            if os.path.isdir(sd):
                roots.append(sd)
    except Exception:
        pass

    for rd in roots:
        try:
            for fn in os.listdir(rd):
                p = os.path.join(rd, fn)
                if not os.path.isfile(p):
                    continue
                lfn = fn.lower()
                if any(w in lfn for w in want_substr):
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in {".json", ".txt", ".mat"} or ext == "":
                        hits.append(p)
        except Exception:
            pass

    # De-dupe
    seen = set()
    out = []
    for p in hits:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def _load_imagenet_map_file(path: str, log) -> Optional[Dict[str, str]]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json" or ext == "":
        # Some people name them without extension; try JSON first.
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            m = _map_from_json_obj(obj)
            if m is not None and len(m) > 0:
                return m
        except Exception:
            # not json, fall through
            pass

    if ext in {".txt", ""}:
        try:
            m = _map_from_synset_txt(path)
            if m is not None and len(m) > 0:
                return m
        except Exception:
            pass

    if ext == ".mat":
        m = _map_from_meta_mat(path, log=log)
        if m is not None and len(m) > 0:
            return m

    return None


def _map_from_json_obj(obj: Any) -> Optional[Dict[str, str]]:
    # Case 1: {"0": ["n01440764", "tench"], ...}
    if isinstance(obj, dict):
        # numeric-key dict with [wnid, human]
        ok = True
        wnid_to_human: Dict[str, str] = {}
        for k, v in obj.items():
            ks = str(k).strip()
            if not ks.isdigit():
                ok = False
                break
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                wnid = str(v[0]).strip()
                human = str(v[1]).strip()
                if _WNID_RE.match(wnid) and human:
                    wnid_to_human[wnid] = human
        if ok and len(wnid_to_human) > 0:
            return wnid_to_human

        # Case 2: {"n01440764": "tench", ...} or {"n014...": ["tench", ...]}
        wnid_to_human = {}
        for k, v in obj.items():
            wnid = str(k).strip()
            if _WNID_RE.match(wnid) is None:
                continue
            if isinstance(v, str):
                human = v.strip()
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                human = str(v[0]).strip()
            else:
                human = str(v).strip()
            if human:
                wnid_to_human[wnid] = human
        if len(wnid_to_human) > 0:
            return wnid_to_human

    return None


def _map_from_synset_txt(path: str) -> Dict[str, str]:
    """
    Parses lines like:
      n01440764 tench, Tinca tinca
      n01440764\ttench
    """
    m: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t") if "\t" in ln else ln.split(" ", 1)
            if len(parts) < 2:
                continue
            wnid = parts[0].strip()
            rest = parts[1].strip()
            if _WNID_RE.match(wnid) is None:
                continue
            # take first comma chunk as simple label
            human = rest.split(",")[0].strip()
            if human:
                m[wnid] = human
    return m


def _map_from_meta_mat(path: str, log) -> Optional[Dict[str, str]]:
    """
    Attempts to parse ILSVRC devkit 'meta.mat' (requires scipy).
    If scipy isn't installed, just returns None.
    """
    try:
        from scipy.io import loadmat
    except Exception:
        log.yellow("[HeuristicDS] meta.mat detected but scipy not installed; skipping .mat parsing.")
        return None

    try:
        mat = loadmat(path)
    except Exception as e:
        log.yellow(f"[HeuristicDS] meta.mat load failed: {e}")
        return None

    # Devkit often contains 'synsets' array with fields 'WNID' and 'words' (varies).
    syn = mat.get("synsets", None)
    if syn is None:
        return None

    out: Dict[str, str] = {}
    try:
        # Very defensive parsing; .mat structures vary.
        for row in syn.squeeze():
            # row might be a numpy void with named fields or an object array
            wnid = None
            words = None

            # Named fields
            if hasattr(row, "dtype") and row.dtype.names:
                names = set(row.dtype.names)
                for k in ["WNID", "wnid"]:
                    if k in names:
                        wnid = str(row[k].squeeze()).strip()
                        break
                for k in ["words", "Words", "label", "labels"]:
                    if k in names:
                        words = str(row[k].squeeze()).strip()
                        break
            else:
                # Fallback: positional guessing (common in some variants)
                try:
                    wnid = str(row[1].squeeze()).strip()
                    words = str(row[2].squeeze()).strip()
                except Exception:
                    pass

            if wnid and _WNID_RE.match(wnid) and words:
                human = words.split(",")[0].strip()
                if human:
                    out[wnid] = human
    except Exception:
        return None

    return out if len(out) > 0 else None
    
    
# ----------------------------
# Special local loaders: CIFAR-10 / CIFAR-100 python pickles
# ----------------------------
def _try_build_special_local_dataset(local_root: str, spec: HeuristicDatasetSpec, transform, tokenize_fn, log):
    """
    Return a Dataset if local_root looks like a known special format.
    Otherwise return None.
    """
    kind = _detect_cifar_python_dir(local_root)
    if kind is None:
        return None

    split = (spec.split or "train").strip().lower()
    # CIFAR doesn't have val; common convention: val -> test
    if split in {"val", "valid", "validation"}:
        split = "test"
    if split not in {"train", "test"}:
        log.yellow(f"[HeuristicDS] CIFAR: unknown split={repr(spec.split)}; defaulting to 'train'")
        split = "train"

    if kind == "cifar100":
        log.cyan("[HeuristicDS] detected CIFAR-100 (python pickles)")
        return HeuristicCIFARPickleDataset(
            root=local_root,
            kind="cifar100",
            split=split,
            transform=transform,
            tokenize_fn=tokenize_fn,
            pretok_batch_size=spec.pretok_batch_size,
            log=log,
        )

    if kind == "cifar10":
        log.cyan("[HeuristicDS] detected CIFAR-10 (python pickles)")
        return HeuristicCIFARPickleDataset(
            root=local_root,
            kind="cifar10",
            split=split,
            transform=transform,
            tokenize_fn=tokenize_fn,
            pretok_batch_size=spec.pretok_batch_size,
            log=log,
        )

    return None

def _detect_cifar_python_dir(root: str) -> Optional[str]:
    """
    Returns:
      'cifar100' if root looks like cifar-100-python
      'cifar10'  if root looks like cifar-10-batches-py
      None otherwise
    """
    # CIFAR-100 python: files named exactly 'train', 'test', 'meta'
    p100_train = os.path.join(root, "train")
    p100_test  = os.path.join(root, "test")
    p100_meta  = os.path.join(root, "meta")
    if os.path.isfile(p100_train) and os.path.isfile(p100_test) and os.path.isfile(p100_meta):
        return "cifar100"

    # CIFAR-10 python: data_batch_1..5, test_batch, batches.meta
    p10_meta = os.path.join(root, "batches.meta")
    p10_test = os.path.join(root, "test_batch")
    if os.path.isfile(p10_meta) and os.path.isfile(p10_test):
        # require at least one train batch
        for i in range(1, 6):
            if os.path.isfile(os.path.join(root, f"data_batch_{i}")):
                return "cifar10"

    return None

# ----------------------------
# Heuristic inference: local
# ----------------------------
def _resolve_image_paths_with_fallback(
    raw_paths: Sequence[Any],
    image_root: str,
    log=None
) -> List[Optional[Any]]:
    if log is None:
        log = _make_logger(False)

    abs_paths: List[Optional[Any]] = []
    exists_flags: List[bool] = []

    for rp in raw_paths:
        ip = _resolve_image_path(rp, image_root=image_root)  # may be str OR bytes OR dict
        abs_paths.append(ip)

        # embedded refs count as "exists"
        if ip is None:
            exists_flags.append(False)
        elif isinstance(ip, str):
            exists_flags.append(os.path.exists(ip))
        else:
            exists_flags.append(True)

    n = len(abs_paths)
    n_ok = sum(exists_flags)
    if n == 0:
        return abs_paths

    ok_ratio = n_ok / max(1, n)
    if ok_ratio >= 0.50:
        return abs_paths

    # only basename-fallback applies to missing *string paths*
    root_abs = os.path.abspath(image_root)
    if root_abs in _BASENAME_INDEX_EMPTY:
        return abs_paths

    first_time = root_abs not in _BASENAME_INDEX_CACHE
    if first_time:
        log.yellow(
            f"[HeuristicDS] image resolve hit-rate low ({n_ok}/{n} = {ok_ratio:.2%}). "
            f"Scanning subfolders to match basenames..."
        )

    idx, dupes = _get_basename_index_cached(image_root=image_root, log=log)

    if len(idx) == 0:
        _BASENAME_INDEX_EMPTY.add(root_abs)
        if first_time:
            log.yellow(f"[HeuristicDS] basename index is empty under {root_abs}; disabling basename fallback for this root.")
        return abs_paths

    fixed: List[Optional[Any]] = []
    fixed_ok = 0
    for ip in abs_paths:
        if ip is None:
            fixed.append(None)
            continue
        if not isinstance(ip, str):
            fixed.append(ip)
            fixed_ok += 1
            continue
        if os.path.exists(ip):
            fixed.append(ip)
            fixed_ok += 1
            continue

        base = os.path.basename(ip)
        cand = idx.get(base, None)
        if cand is not None:
            fixed.append(cand)
            fixed_ok += 1
        else:
            fixed.append(ip)

    if dupes > 0 and first_time:
        log.yellow(f"[HeuristicDS] basename index warning: {dupes} duplicate basenames encountered (kept first).")

    if first_time:
        log.green(f"[HeuristicDS] basename fallback improved hits: {fixed_ok}/{n} = {fixed_ok/max(1,n):.2%}")
    return fixed


def _pairs_from_caption_lines(lines: Sequence[str], image_root: str, log) -> List[Tuple[str, List[str]]]:
    """
    Flickr8k-style captions:
      image.jpg,caption text...
    often multiple lines per image.

    Also supports:
      image.jpg<TAB>caption
      image.jpg caption...
    """
    raw_imgs: List[str] = []
    raw_caps: List[str] = []

    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue

        # split at first comma (most common)
        if "," in s:
            a, b = s.split(",", 1)
        elif "\t" in s:
            a, b = s.split("\t", 1)
        else:
            parts = s.split(maxsplit=1)
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]

        img_tok = a.strip().strip('"').strip("'")
        cap = b.strip().strip('"').strip("'")
        if not img_tok or not cap:
            continue

        # common variants: "xxx.jpg#0"
        if "#" in img_tok:
            img_tok = img_tok.split("#", 1)[0].strip()

        raw_imgs.append(img_tok)
        raw_caps.append(cap)

    if len(raw_imgs) == 0:
        log.yellow("[HeuristicDS] .txt captions parse produced 0 lines with (image, caption).")
        return []

    resolved = _resolve_image_paths_with_fallback(raw_imgs, image_root=image_root, log=log)
    img_to_caps: Dict[str, List[str]] = {}
    n_bad = 0

    for ip, cap in zip(resolved, raw_caps):
        if ip is None or not os.path.exists(ip):
            n_bad += 1
            continue
        img_to_caps.setdefault(ip, []).append(cap)

    samples = [(ip, caps) for ip, caps in img_to_caps.items()]
    log.cyan(f"[HeuristicDS] txt captions: images_with_caps={len(samples)} lines={len(raw_imgs)} bad_images={n_bad}")
    return samples


def _build_basename_index(image_root: str, log, max_n: int = 2_000_000) -> Tuple[Dict[str, str], int]:
    """
    Build basename -> fullpath map by scanning for images under image_root.
    Returns (index, num_dupe_basenames).
    """
    idx: Dict[str, str] = {}
    dupes = 0
    n_seen = 0

    for r, _ds, fs in os.walk(image_root):
        for fn in fs:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _IMAGE_EXTS:
                continue
            n_seen += 1
            if n_seen > max_n:
                log.yellow(f"[HeuristicDS] basename index hit max_n={max_n}; stopping scan early.")
                return idx, dupes
            if fn in idx:
                dupes += 1
                continue
            idx[fn] = os.path.join(r, fn)

    log.cyan(f"[HeuristicDS] basename index built: {len(idx)} images indexed under {image_root}")
    return idx, dupes


def _infer_local_samples(root_dir: str, log) -> List[Tuple[str, List[str]]]:
    """
    Tries:
      1) Known metadata files in root (or subdirs, shallow)
      2) COCO-style annotations JSON
      3) Sidecar .txt per image
      4) ImageFolder-style class labels (subdir name)
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(root_dir)

    log.cyan(f"[HeuristicDS] local inference: scanning -> {root_dir}")

    # 1) metadata files (common names + fuzzy keyword match)
    meta_candidates = _find_metadata_files(root_dir, log=log)

    successes: List[Tuple[str, List[Tuple[str, List[str]]]]] = []  # (meta_path, samples)
    for mp in meta_candidates[:5]:  # cap attempts
        try:
            log.cyan(f"[HeuristicDS] trying metadata file -> {mp}")
            samples = _load_metadata_pairs(mp, image_root=root_dir, log=log)
            if len(samples) > 0:
                log.green(f"[HeuristicDS] metadata success -> {os.path.basename(mp)} produced n={len(samples)}")
                successes.append((mp, samples))
        except Exception as e:
            log.yellow(f"[HeuristicDS] metadata failed for {mp}: {e}")

    if len(successes) == 1:
        return successes[0][1]

    if len(successes) > 1:
        # auto-merge parquet shards (train-00000-of-000xx.parquet etc.)
        parquet_succ = [(mp, smp) for (mp, smp) in successes if mp.lower().endswith(".parquet")]
        if len(parquet_succ) == len(successes):
            def _shard_split_name(p: str) -> str:
                bn = os.path.basename(p).lower()
                for s in ["train", "test", "val", "valid", "validation"]:
                    if bn.startswith(s + "-") or bn.startswith(s + "_"):
                        return "val" if s in {"val", "valid", "validation"} else s
                return "unknown"

            groups: Dict[str, List[Tuple[str, List[Tuple[Any, List[str]]]]]] = {}
            for mp, smp in parquet_succ:
                groups.setdefault(_shard_split_name(mp), []).append((mp, smp))

            # prefer train > val > test > unknown
            for pref in ["train", "val", "test", "unknown"]:
                if pref in groups:
                    chosen = groups[pref]
                    merged: List[Tuple[Any, List[str]]] = []
                    for _mp, _smp in chosen:
                        merged.extend(_smp)

                    log.yellow(f"[HeuristicDS] parquet shards detected: auto-merging split='{pref}' shards={len(chosen)}")
                    for mp, _ in chosen[:5]:
                        log.yellow(f"  - {mp}")
                    if len(chosen) > 5:
                        log.yellow(f"  ... (+{len(chosen)-5} more)")

                    log.green(f"[HeuristicDS] parquet merged -> n={len(merged)}")
                    return merged
        # Sort by sample count (best first)
        successes.sort(key=lambda x: len(x[1]), reverse=True)
        log.red(f"[HeuristicDS] AMBIGUOUS: found >={len(successes)} metadata files that look valid.")
        show = successes[:5]
        for mp, smp in show:
            log.red(f"  - {mp}  (pairs={len(smp)})")
        if len(successes) > 5:
            log.red(f"  ... (+{len(successes)-5} more)")

        raise RuntimeError(
            "Heuristic dataset inference is ambiguous: multiple caption/label files look usable.\n"
            "Please pick ONE and set it explicitly as cfg.custom_dataset_{train/val}['text_labels_path'].\n"
            "Shown above: up to 5 candidates."
        )
    # 2) sidecar .txt captions
    try:
        log.cyan("[HeuristicDS] trying sidecar captions: image.ext + image.txt (same stem)")
        samples = _scan_sidecar_txt(root_dir, log=log)
        if len(samples) > 0:
            log.green(f"[HeuristicDS] sidecar success -> n={len(samples)}")
            return samples
    except Exception as e:
        log.yellow(f"[HeuristicDS] sidecar scan failed: {e}")

    # 3) ImageFolder class labels
    try:
        log.cyan("[HeuristicDS] trying ImageFolder-style labels (subfolder name as label)")
        samples = _scan_imagefolder_labels(root_dir, log=log)
        if len(samples) > 0:
            log.green(f"[HeuristicDS] ImageFolder-label success -> n={len(samples)}")
            return samples
    except Exception as e:
        log.yellow(f"[HeuristicDS] ImageFolder-label scan failed: {e}")

    # 4) last resort: images exist but no text at all -> error
    imgs = _list_images_recursive(root_dir, max_n=50_000)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found under: {root_dir}")
    raise RuntimeError(
        f"Found images (e.g. {os.path.basename(imgs[0])}) but could not infer any captions/labels.\n"
        "Provide text_labels_path (json/csv/parquet) or add sidecar .txt files."
    )

def _find_metadata_files(root_dir: str, log) -> List[str]:
    """
    Find metadata/caption files in root or one-level-deep subdirs.

    Two tiers:
      A) exact common filenames (fast + precise)
      B) fuzzy keyword match (caption/train/val/test/karpathy/etc.)
         supports .txt for Flickr8k-style caption lists.
    """
    common_names = [
        "metadata.csv", "metadata.tsv", "metadata.parquet",
        "captions.csv", "captions.tsv", "captions.parquet",
        "labels.csv", "labels.tsv", "labels.parquet",
        "annotations.json", "captions.json", "labels.json", "dataset.json",
        "annotations.jsonl", "captions.jsonl", "labels.jsonl", "dataset.jsonl",
        "train.json", "val.json", "test.json", "meta.mat"
    ]

    allowed_exts = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".txt", ".mat"}

    # include "noisy"/"imagenette" so we pick up noisy_imagenette.csv
    keywords = (
        "caption", "captions", "label", "labels", "annotation", "annotations",
        "train", "val", "valid", "validation", "test", "metadata", "meta",
        "noisy", "imagenette", "imagenet", "anno", "annos",
    )

    def scan_dir(d: str) -> List[str]:
        out = []
        try:
            for fn in os.listdir(d):
                p = os.path.join(d, fn)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in allowed_exts:
                    continue
                lfn = fn.lower()

                # tier A: exact names
                if fn in common_names:
                    out.append(p)
                    continue

                # tier B: fuzzy match
                if any(k in lfn for k in keywords):
                    out.append(p)
        except Exception as e:
            log.yellow(f"[HeuristicDS] metadata scan warning: {e}")
        return out

    hits: List[str] = []
    hits.extend(scan_dir(root_dir))

    # shallow subdirs
    try:
        for d in os.listdir(root_dir):
            sd = os.path.join(root_dir, d)
            if os.path.isdir(sd):
                hits.extend(scan_dir(sd))
    except Exception as e:
        log.yellow(f"[HeuristicDS] metadata scan warning: {e}")

    # also scan parent directory (common: CSV sits next to data/)
    try:
        parent = os.path.dirname(os.path.abspath(root_dir))
        if parent and parent != os.path.abspath(root_dir) and os.path.isdir(parent):
            hits.extend(scan_dir(parent))
    except Exception as e:
        log.yellow(f"[HeuristicDS] metadata parent-scan warning: {e}")

    # de-dupe while preserving order
    seen = set()
    out = []
    for p in hits:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)

    if len(out) > 0:
        log.cyan(f"[HeuristicDS] found metadata candidates: {len(out)}")
        for p in out[:3]:
            log.cyan(f"  - {p}")
        if len(out) > 3:
            log.cyan(f"  ... (+{len(out)-3} more)")
    else:
        log.cyan("[HeuristicDS] no obvious metadata files found")
    return out


def _scan_sidecar_txt(root_dir: str, log) -> List[Tuple[str, List[str]]]:
    imgs = _list_images_recursive(root_dir, max_n=1_000_000)
    if len(imgs) == 0:
        return []

    samples: List[Tuple[str, List[str]]] = []
    n_txt = 0
    # only require a modest hit-rate to accept this heuristic
    for ip in imgs:
        stem, _ = os.path.splitext(ip)
        tp = stem + ".txt"
        if os.path.exists(tp) and os.path.isfile(tp):
            n_txt += 1
            try:
                with open(tp, "r", encoding="utf-8") as f:
                    cap = f.read().strip()
                if cap != "":
                    samples.append((ip, [cap]))
            except Exception:
                pass

    log.cyan(f"[HeuristicDS] sidecar: images={len(imgs)} txt_matches={n_txt} usable_pairs={len(samples)}")
    # Accept if at least 5% of images have captions, or if >= 100 captions exist
    if len(samples) >= 100 or (len(imgs) > 0 and (n_txt / max(1, len(imgs))) >= 0.05):
        return samples
    return []


# ----------------------------
# ImageFolder heuristic (SAFER)
# ----------------------------
def _norm_dirname(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def _looks_junk_dirname(name: str) -> bool:
    n = _norm_dirname(name)
    if n in _JUNK_DIR_NAMES:
        return True
    # filter trash :)
    if n.startswith("new folder"):
        return True
    return False

def _foldername_to_prompt(name: str) -> str:
    """
    Turn folder name into a decent text prompt candidate.
    Examples:
      "a_photo_of_a_dog" -> "a photo of a dog"
      "golden-retriever" -> "golden retriever"
    """
    s = str(name).strip()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s

def _dir_has_any_images_recursive(root: str, max_scan: int = 200_000) -> bool:
    """
    Returns True if there exists at least one image file anywhere under root.
    Stops early after max_scan image candidates encountered.
    """
    seen = 0
    for r, _ds, fs in os.walk(root):
        for fn in fs:
            ext = os.path.splitext(fn)[1].lower()
            if ext in _IMAGE_EXTS:
                return True
            seen += 1
            if seen >= max_scan:
                # if we scanned tons of files and never saw an image extension, call it False
                return False
    return False

def _collect_image_leaf_dirs(root_dir: str, log, max_dirs: int = 50_000) -> List[str]:
    """
    Collect "leaf image folders":
      - directories that contain images somewhere under them,
      - but that do NOT have a subdirectory that also contains images.

    This avoids selecting wrapper directories like "New Folder" when the real image dirs are deeper.

    If the structure is *mixed* (a dir contains images and also a child dir contains images),
    we treat that as ambiguous and return [] (force user to provide captions).
    """
    # Step 1: find all dirs that contain images (recursively)
    img_dirs: List[str] = []
    for r, ds, _fs in os.walk(root_dir):
        # early stop if this explodes
        if len(img_dirs) >= max_dirs:
            log.yellow(f"[HeuristicDS] ImageFolder scan hit max_dirs={max_dirs}; aborting folder-label inference.")
            return []
        # quick check: any images under this dir?
        if _dir_has_any_images_recursive(r):
            img_dirs.append(os.path.abspath(r))

    if not img_dirs:
        return []

    img_dirs = sorted(set(img_dirs), key=lambda x: (len(x), x))

    # Step 2: keep only those that are not parents of another image-dir
    img_dirs_set = set(img_dirs)
    leaf_dirs: List[str] = []
    mixed_ambiguous = False

    # Precompute for speed: sort by length ascending so parents come first
    for d in img_dirs:
        # if any strict subdir of d is also in img_dirs -> d is not a leaf
        is_parent = False
        prefix = d.rstrip(os.sep) + os.sep
        for other in img_dirs:
            if other != d and other.startswith(prefix):
                is_parent = True
                break
        if not is_parent:
            leaf_dirs.append(d)

    # Ambiguity check: directories that contain images AND have image-containing children
    # We detect it by checking if any non-leaf dir *also directly contains images*.
    # That pattern is often accidental messy downloads; better to abort than label-poison.
    for d in img_dirs:
        if d in leaf_dirs:
            continue
        # does d contain images directly (not just in descendants)?
        try:
            for fn in os.listdir(d):
                p = os.path.join(d, fn)
                if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in _IMAGE_EXTS:
                    mixed_ambiguous = True
                    break
        except Exception:
            pass
        if mixed_ambiguous:
            break

    if mixed_ambiguous:
        log.yellow("[HeuristicDS] ImageFolder structure is mixed (images in parent AND child dirs).")
        log.yellow("[HeuristicDS] Refusing folder-name caption inference (too ambiguous). Provide text_labels_path.")
        return []

    return leaf_dirs

def _humanize_imagenet_label(s: str) -> str:
    """
    ImageNet labels often come as 'great_white_shark' or 'golden retriever, ...'.
    We keep it simple: underscores -> spaces, and only use the first comma chunk.
    """
    t = str(s).strip()
    if "," in t:
        t = t.split(",", 1)[0].strip()
    t = t.replace("_", " ")
    t = " ".join(t.split())
    return t


def _scan_imagefolder_labels(root_dir: str, log) -> List[Tuple[str, List[str]]]:
    """
    Safer ImageFolder heuristic:
      - only uses *leaf image folders* (folders that actually contain images, and aren't wrappers)
      - requires diversity: >= 4 meaningful leaf image folders
      - refuses junk folder names (train/val/test/temp/New Folder/etc.)
      - refuses if normalized prompts collapse to 1 unique label
      - warns (yellow) by printing the first 10 inferred prompts
      - If leaf folder names look like ImageNet WNIDs (n########), try to map WNID->human label
        by searching for common ImageNet label-map files (json/txt/meta.mat).
    """
    leaf_dirs = _collect_image_leaf_dirs(root_dir, log=log)
    if len(leaf_dirs) == 0:
        return []

    # derive candidate labels from leaf folder basenames
    base_names = [os.path.basename(d) for d in leaf_dirs]
    bad = [bn for bn in base_names if _looks_junk_dirname(bn)]
    if bad:
        bad_uniq = sorted(set(bad))[:20]
        log.yellow(f"[HeuristicDS] ImageFolder-label: found image-containing folders with junk names: {bad_uniq}")
        log.yellow("[HeuristicDS] Refusing folder-name caption inference. Provide text_labels_path.")
        return []

    # require diversity (your rule: “More than three”)
    if len(leaf_dirs) <= 3:
        log.yellow(f"[HeuristicDS] ImageFolder-label: only {len(leaf_dirs)} image folders found (need >=4).")
        log.yellow("[HeuristicDS] Refusing folder-name caption inference. Provide captions/labels.")
        return []

    # detect ImageNet-style WNID folders and try to map to human labels ---
    wnid_count = sum(1 for bn in base_names if _WNID_RE.match(str(bn).strip()) is not None)
    wnid_ratio = wnid_count / max(1, len(base_names))
    wnid_to_human: Optional[Dict[str, str]] = None
    map_src: Optional[str] = None

    # Heuristic: looks like ImageNet if most folder names are WNIDs and there are "many" of them
    if wnid_ratio >= 0.60 and wnid_count >= 20:
        wnid_to_human, map_src = _try_load_imagenet_label_map(root_dir, log=log)

    # Build prompts (foldername -> prompt), then optionally map WNID->human
    prompts: List[str] = []
    for bn in base_names:
        raw = str(bn).strip()
        if wnid_to_human is not None and _WNID_RE.match(raw) is not None:
            human = wnid_to_human.get(raw, raw)
            prompts.append(_humanize_imagenet_label(human))
        else:
            prompts.append(_foldername_to_prompt(raw))

    # Check uniqueness collapse (do this on the *final* prompts)
    uniq_prompts = sorted(set(p.lower() for p in prompts if p.strip() != ""))
    if len(uniq_prompts) <= 1:
        log.yellow(f"[HeuristicDS] ImageFolder-label: prompts collapse to {len(uniq_prompts)} unique label(s).")
        log.yellow("[HeuristicDS] This is usually a wrapper-folder trap. Provide text_labels_path.")
        return []

    # At this point: we accept. Build samples.
    # Warn user with first 10 prompts
    log.yellow("[HeuristicDS] WARNING: inferring captions from folder names (ImageFolder heuristic).")
    if map_src is not None:
        log.yellow(f"[HeuristicDS] ImageNet label-map detected -> {map_src}")
    log.yellow("[HeuristicDS] First 10 inferred prompts:")
    for p in uniq_prompts[:10]:
        log.yellow(f"  - {p}")
    if len(uniq_prompts) > 10:
        log.yellow(f"  ... (+{len(uniq_prompts)-10} more)")

    samples: List[Tuple[str, List[str]]] = []
    for d, prompt in zip(leaf_dirs, prompts):
        imgs = _list_images_recursive(d, max_n=2_000_000)
        for ip in imgs:
            samples.append((ip, [prompt]))

    log.cyan(
        f"[HeuristicDS] ImageFolder-label success -> leaf_dirs={len(leaf_dirs)} "
        f"samples={len(samples)} unique_prompts={len(uniq_prompts)}"
    )
    return samples

def _list_images_recursive(root_dir: str, max_n: int = 200_000) -> List[str]:
    out: List[str] = []
    for r, _ds, fs in os.walk(root_dir):
        for fn in fs:
            ext = os.path.splitext(fn)[1].lower()
            if ext in _IMAGE_EXTS:
                out.append(os.path.join(r, fn))
                if len(out) >= max_n:
                    return out
    return out


# ----------------------------
# Heuristic inference: metadata files
# ----------------------------
def _load_metadata_pairs(meta_path: str, image_root: str, log) -> List[Tuple[str, List[str]]]:
    """
    Supports:
      - JSON dict mapping
      - JSON list of row dicts
      - COCO captions JSON
      - JSONL
      - CSV/TSV
      - Parquet
      - TXT (Flickr8k-style): "image.jpg,caption..." (many lines per image)
        also supports tab-separated or "image.jpg caption..." lines.
    """
    meta_path = os.path.abspath(meta_path)
    ext = os.path.splitext(meta_path)[1].lower()
    log.cyan(f"[HeuristicDS] loading metadata: {meta_path} (ext={ext})")

    if ext == ".json":
        with open(meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _pairs_from_json_obj(obj, image_root=image_root, log=log)

    if ext == ".jsonl":
        rows = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rows.append(json.loads(ln))
        return _pairs_from_row_dicts(rows, image_root=image_root, log=log)

    if ext in {".csv", ".tsv"}:
        import pandas as pd
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(meta_path, sep=sep)
        return _pairs_from_dataframe(df, image_root=image_root, log=log)

    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(meta_path)
        return _pairs_from_dataframe(df, image_root=image_root, log=log)

    if ext == ".txt":
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]
        return _pairs_from_caption_lines(lines, image_root=image_root, log=log)

    raise RuntimeError(f"Unsupported metadata extension: {ext} ({meta_path})")

def _pairs_from_json_obj(obj: Any, image_root: str, log) -> List[Tuple[str, List[str]]]:
    # Case A: dict mapping
    if isinstance(obj, dict):
        # COCO style?
        if "images" in obj and "annotations" in obj and isinstance(obj["images"], list) and isinstance(obj["annotations"], list):
            return _pairs_from_coco_captions(obj, image_root=image_root, log=log)

        # Otherwise assume mapping image_path -> caption(s)
        samples: List[Tuple[str, List[str]]] = []
        n_bad = 0
        for k, v in obj.items():
            img_path = _resolve_image_path(k, image_root=image_root)
            cands = _coerce_text_candidates(v)
            if img_path is None or not os.path.exists(img_path):
                n_bad += 1
                continue
            samples.append((img_path, cands))
        log.cyan(f"[HeuristicDS] json dict mapping: pairs={len(samples)} bad_images={n_bad}")
        return samples

    # Case B: list of dict rows
    if isinstance(obj, list):
        return _pairs_from_row_dicts(obj, image_root=image_root, log=log)

    raise RuntimeError(f"JSON top-level must be dict or list; got {type(obj)}")

def _pairs_from_coco_captions(coco_obj: Dict[str, Any], image_root: str, log) -> List[Tuple[str, List[str]]]:
    """
    COCO captions:
      images: [{id, file_name, ...}, ...]
      annotations: [{image_id, caption, ...}, ...]
    """
    id_to_file: Dict[int, str] = {}
    for im in coco_obj.get("images", []):
        try:
            iid = int(im["id"])
            fn = str(im.get("file_name", "")).strip()
            if fn:
                id_to_file[iid] = fn
        except Exception:
            continue

    img_to_caps: Dict[str, List[str]] = {}
    n_ann = 0
    for ann in coco_obj.get("annotations", []):
        try:
            iid = int(ann["image_id"])
            cap = str(ann.get("caption", "")).strip()
            if not cap:
                continue
            fn = id_to_file.get(iid, "")
            if not fn:
                continue
            img_to_caps.setdefault(fn, []).append(cap)
            n_ann += 1
        except Exception:
            continue

    samples: List[Tuple[str, List[str]]] = []
    n_bad = 0
    for fn, caps in img_to_caps.items():
        ip = _resolve_image_path(fn, image_root=image_root)
        if ip is None or not os.path.exists(ip):
            n_bad += 1
            continue
        samples.append((ip, caps))

    log.cyan(f"[HeuristicDS] COCO captions: images_with_caps={len(samples)} annotations_used={n_ann} bad_images={n_bad}")
    return samples

def _pairs_from_row_dicts(rows: Sequence[Any], image_root: str, log) -> List[Tuple[str, List[str]]]:
    """
    Accepts list of dict-like items; tries common key names.
    """
    if len(rows) == 0:
        return []

    image_keys = ["image", "img", "image_path", "filepath", "file", "path", "file_name", "filename"]
    text_keys = ["text", "caption", "captions", "prompt", "label", "labels", "description", "sentence"]

    samples: List[Tuple[str, List[str]]] = []
    n_bad = 0
    for r in rows:
        if not isinstance(r, dict):
            continue

        ik = _first_present_key(r, image_keys)
        tk = _first_present_key(r, text_keys)

        if ik is None or tk is None:
            n_bad += 1
            continue

        ip_raw = r.get(ik, None)
        txt_raw = r.get(tk, None)

        # resolve exactly one ref with fallback (and pass log) ---
        ip = _resolve_image_paths_with_fallback([ip_raw], image_root=image_root, log=log)[0]

        if ip is None or not os.path.exists(ip):
            n_bad += 1
            continue

        cands = _coerce_text_candidates(txt_raw)
        samples.append((ip, cands))

    log.cyan(f"[HeuristicDS] row-dicts: pairs={len(samples)} skipped={n_bad}")
    if len(samples) == 0:
        log.yellow("[HeuristicDS] row-dicts produced 0 usable pairs; keys may be nonstandard.")
        log.yellow(f"[HeuristicDS] tried image_keys={image_keys}")
        log.yellow(f"[HeuristicDS] tried text_keys={text_keys}")
    return samples


def _pairs_from_dataframe(df, image_root: str, log) -> List[Tuple[Any, List[str]]]:
    cols = [str(c) for c in df.columns]
    log.cyan(f"[HeuristicDS] dataframe columns={cols}")

    image_cols = ["image", "img", "image_path", "filepath", "file", "path", "file_name", "filename"]
    text_cols = ["text", "caption", "captions", "prompt", "label", "labels", "description", "sentence"]

    ic = _first_present_key({c: True for c in cols}, image_cols)
    tc = _first_present_key({c: True for c in cols}, text_cols)

    if ic is None or tc is None:
        raise RuntimeError(
            "Could not infer required columns from metadata table.\n"
            f"Columns={cols}\n"
            f"Tried image columns={image_cols}\n"
            f"Tried text columns={text_cols}"
        )

    samples: List[Tuple[Any, List[str]]] = []
    n_bad = 0

    for _, row in df.iterrows():
        ip_raw = row[ic]
        txt_raw = row[tc]

        # resolve exactly one ref with fallback (and pass log)
        ip = _resolve_image_paths_with_fallback([ip_raw], image_root=image_root, log=log)[0]

        # embedded refs are valid
        if ip is None:
            n_bad += 1
            continue
        if isinstance(ip, str) and (not os.path.exists(ip)):
            n_bad += 1
            continue

        cands = _coerce_text_candidates(txt_raw)
        samples.append((ip, cands))

    log.cyan(f"[HeuristicDS] dataframe pairs={len(samples)} bad_images={n_bad}")
    return samples



def _resolve_image_root(spec: HeuristicDatasetSpec, porh: str, meta_path: str, log) -> str:
    """
    Priority:
      1) spec.image_root if set
      2) porh if it exists and is a dir
      3) directory containing metadata file
      4) CWD
    """
    if spec.image_root is not None and str(spec.image_root).strip() != "":
        root = os.path.abspath(spec.image_root)
        log.cyan(f"[HeuristicDS] image_root override -> {root}")
        return root

    if porh.strip() != "" and _looks_like_existing_path(porh) and os.path.isdir(porh):
        root = os.path.abspath(porh)
        log.cyan(f"[HeuristicDS] image_root inferred from path_or_hf_dataset dir -> {root}")
        return root

    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    if meta_dir and os.path.isdir(meta_dir):
        log.cyan(f"[HeuristicDS] image_root inferred from metadata dir -> {meta_dir}")
        return meta_dir

    cwd = os.getcwd()
    log.cyan(f"[HeuristicDS] image_root fallback -> CWD {cwd}")
    return cwd

def _resolve_image_path(path_like: Any, image_root: str) -> Optional[Any]:
    """
    Resolve absolute or relative image paths.
    Also supports embedded images from parquet/HF exports:
      - bytes / bytearray / memoryview
      - dict containing {"bytes": ...} or {"data": ...} or {"array": ...}
    """
    if path_like is None:
        return None

    # embedded image refs (parquet/HF style)
    if isinstance(path_like, (bytes, bytearray, memoryview)):
        return bytes(path_like)

    if isinstance(path_like, dict):
        # embedded bytes?
        for k in ["bytes", "data"]:
            if k in path_like and path_like[k]:
                v = path_like[k]
                if isinstance(v, (bytes, bytearray, memoryview)):
                    return bytes(v)
        # embedded array?
        if "array" in path_like and path_like["array"] is not None:
            return path_like  # keep as-is; decoded later

        # path-like keys
        for k in ["path", "image", "image_path", "file", "file_name", "filename"]:
            if k in path_like and path_like[k]:
                path_like = path_like[k]
                break

    p = str(path_like).strip()
    if p == "":
        return None

    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(image_root, p))


def _coerce_text_candidates(x: Any) -> List[str]:
    """
    Convert various forms into a list[str].
    """
    if x is None:
        return [""]

    # pandas NaN
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return [""]
    except Exception:
        pass

    if isinstance(x, str):
        s = x.strip()
        return [s] if s != "" else [""]

    if isinstance(x, (list, tuple)):
        out = []
        for t in x:
            if t is None:
                continue
            ss = str(t).strip()
            if ss != "":
                out.append(ss)
        return out if len(out) > 0 else [""]

    # Some datasets store {"text": "..."} etc.
    if isinstance(x, dict):
        for k in ["text", "caption", "prompt", "label", "description", "sentence"]:
            if k in x:
                return _coerce_text_candidates(x[k])
        return [""]

    return [str(x).strip() or ""]

def _first_present_key(d: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    """
    Returns the first candidate key that exists in dict d (case-insensitive match too).
    """
    keys = set(d.keys())
    lkeys = {str(k).lower(): k for k in d.keys()}
    for c in candidates:
        if c in keys:
            return c
        lc = c.lower()
        if lc in lkeys:
            return lkeys[lc]
    return None


# ----------------------------
# Heuristic inference: Hugging Face
# ----------------------------
def _infer_hf_string_text_column(
    ds_id: str,
    dset,
    colnames: List[str],
    log,
    sample_n: int = 256,
    interactive: bool = False,
) -> Optional[str]:
    """
    Last-resort HF text column inference.

    Behavior:
      - If exactly 1 strong candidate: auto-pick (with examples).
      - If multiple strong candidates:
          * if interactive=True: ask y/n in ranked order, cache choice per ds_id.
          * else: raise (ambiguous).
      - If none: return None.
    """
    import random as _rnd
    from collections import Counter

    # If user already chose for this HF id, reuse it.
    if ds_id in _HF_TEXTCOL_DECISIONS:
        chosen = _HF_TEXTCOL_DECISIONS[ds_id]
        if chosen in colnames:
            log.yellow(f"[HeuristicDS] HF: reusing cached text column choice -> {chosen}")
            return chosen
        # if schema changed, ignore cache
        log.yellow(f"[HeuristicDS] HF: cached text col {chosen} not in current columns; ignoring cache.")
        _HF_TEXTCOL_DECISIONS.pop(ds_id, None)

    n = len(dset)
    if n <= 0:
        return None

    take = min(sample_n, n)
    idxs = list(range(n))
    _rnd.shuffle(idxs)
    idxs = idxs[:take]

    def _is_good_str(x) -> bool:
        return isinstance(x, str) and (x.strip() != "")

    # Don’t ever treat image column as text
    skip_cols = {"image", "img", "images", "pixel_values"}

    # Collect candidate stats
    candidates = []
    for c in colnames:
        if str(c).lower() in skip_cols:
            continue

        good = 0
        lens: List[int] = []
        uniq = set()

        for i in idxs:
            try:
                v = dset[i].get(c, None)
            except Exception:
                v = None

            if _is_good_str(v):
                s = v.strip()
                good += 1
                lens.append(len(s))
                uniq.add(s)

        if good == 0:
            continue

        good_ratio = good / max(1, take)
        uniq_ratio = len(uniq) / max(1, good)

        cnt = Counter(lens)
        mode_len, mode_ct = cnt.most_common(1)[0]
        mode_frac = mode_ct / max(1, len(lens))
        lens_sorted = sorted(lens)
        med_len = lens_sorted[len(lens_sorted) // 2]

        # Filters (caption-ish vs ID-ish)
        if good_ratio < 0.80:
            continue
        # reject fixed-length short IDs
        if mode_frac > 0.90 and mode_len <= 12:
            continue
        # require "caption-like" length
        if med_len <= 10:
            continue
        # require some variability / uniqueness
        if uniq_ratio < 0.30:
            continue

        # score: prefer longer, then more unique, then higher coverage
        score = (med_len, uniq_ratio, good_ratio)
        candidates.append((c, score, good_ratio, uniq_ratio, med_len, mode_len, mode_frac))

    if len(candidates) == 0:
        log.yellow("[HeuristicDS] HF: no usable string columns found for text prompts.")
        return None

    # Sort best-first
    candidates.sort(key=lambda t: t[1], reverse=True)

    def _examples_for_col(col: str, k: int = 3) -> List[str]:
        ex = []
        for i in idxs:
            if len(ex) >= k:
                break
            try:
                v = dset[i].get(col, None)
            except Exception:
                v = None
            if isinstance(v, str) and v.strip():
                ex.append(v.strip().replace("\n", " ")[:160])
        return ex

    # If only one candidate, auto-pick
    if len(candidates) == 1:
        c = candidates[0][0]
        log.yellow(f"[HeuristicDS] HF: inferred text column via heuristic -> {c}")
        for j, s in enumerate(_examples_for_col(c, k=3)):
            log.yellow(f"  example[{j}]: {s}")
        _HF_TEXTCOL_DECISIONS[ds_id] = c  # cache even for auto-pick
        return c

    # Multiple candidates: either interactive selection or fail hard.
    header_row = " | ".join(colnames)
    log.red("[HeuristicDS] HF: Multiple potential text columns found.")
    log.red(f"[HeuristicDS] HF columns: {header_row}")
    log.red("[HeuristicDS] Please choose which column to use as the text prompt.")

    if not interactive:
        for c, _score, goodr, uniqr, medl, model, modef in candidates[:10]:
            log.red(f"  - {c}: good={goodr:.2f} uniq={uniqr:.2f} med_len={medl} mode_len={model} mode_frac={modef:.2f}")
        raise RuntimeError(
            "HF dataset text-column inference is ambiguous.\n"
            "Enable interactive selection or provide an explicit choice in code."
        )

    # Interactive: ask y/n in ranked order; stop at first 'y'
    for c, _score, goodr, uniqr, medl, model, modef in candidates:
        log.yellow(f"\nCandidate: {c}")
        log.yellow(f"  stats: good={goodr:.2f} uniq={uniqr:.2f} med_len={medl} mode_len={model} mode_frac={modef:.2f}")
        ex = _examples_for_col(c, k=2)
        for j, s in enumerate(ex):
            log.yellow(f"  example[{j}]: {s}")

        try:
            ans = input(f"Use this label column '{c}' as text prompt for ALL rows? [y/n]: ").strip().lower()
        except EOFError:
            raise RuntimeError(
                "Interactive HF label selection needed, but stdin is not available (EOF). "
                "Run in an interactive console or hardcode a choice."
            )

        if ans in {"y", "yes"}:
            log.green(f"[HeuristicDS] HF: selected text column -> {c}")
            _HF_TEXTCOL_DECISIONS[ds_id] = c
            return c
        if ans in {"n", "no"}:
            continue

        # any weird answer: treat as "no" but warn
        log.yellow("Please answer 'y' or 'n'. Treating as 'n'.")
        continue

    # user said no to everything
    raise RuntimeError(
        "No text column selected. Refusing to guess.\n"
        "Pick a column next run, or use a dataset that includes captions/labels."
    )



def _infer_hf_dataset_info(ds_id: str, split: Optional[str], log):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Hugging Face 'datasets' is not installed or failed to import. "
            "Install it, or provide a local dataset path.\n"
            f"Underlying error: {e}"
        )

    def _pick_fallback_split(pack: Dict[str, Any]) -> str:
        # prefer canonical order
        for cand in ["train", "validation", "val", "test"]:
            if cand in pack:
                return cand
        return list(pack.keys())[0]

    # robust split loading with fallback
    try:
        if split is not None:
            req = str(split).strip()
            log.cyan(f"[HeuristicDS] HF: load_dataset(id={ds_id}, split={req})")
            try:
                dset = load_dataset(ds_id, split=req)
            except Exception as e:
                emsg = str(e)
                # datasets raises ValueError: Unknown split "X". Should be one of [...]
                if "Unknown split" in emsg and "Should be one of" in emsg:
                    log.yellow(f"[HeuristicDS] HF: requested split={req} not available; falling back to an existing split.")
                    pack = load_dataset(ds_id)
                    if isinstance(pack, dict):
                        fb = _pick_fallback_split(pack)
                        log.yellow(f"[HeuristicDS] HF: using split -> {fb}")
                        dset = pack[fb]
                    else:
                        log.yellow("[HeuristicDS] HF: dataset returned non-dict pack; using it as-is.")
                        dset = pack
                else:
                    raise
        else:
            log.cyan(f"[HeuristicDS] HF: load_dataset(id={ds_id}) (no split specified)")
            pack = load_dataset(ds_id)
            if isinstance(pack, dict):
                fb = _pick_fallback_split(pack)
                dset = pack[fb]
                log.cyan(f"[HeuristicDS] HF: selected split -> {fb}")
            else:
                dset = pack
    except Exception as e:
        raise RuntimeError(f"Hugging Face load_dataset failed for {ds_id}: {e}")

    feats = getattr(dset, "features", None)
    colnames = list(getattr(dset, "column_names", []))
    log.cyan(f"[HeuristicDS] HF: columns={colnames}")

    # infer image column
    image_col = None
    if feats is not None:
        for c in colnames:
            try:
                f = feats[c]
                if getattr(f, "__class__", None).__name__ == "Image" or "Image" in str(type(f)):
                    image_col = c
                    break
            except Exception:
                pass
    if image_col is None:
        for c in ["image", "img", "images", "pixel_values", "jpg", "png"]:
            if c in colnames:
                image_col = c
                break

    # last-resort infer URL-based image column (e.g. 'image_url')
    if image_col is None:
        image_col = _infer_hf_image_url_column(dset, colnames, log=log)

    if image_col is None:
        raise RuntimeError(f"[HeuristicDS] HF: could not infer an image column. columns={colnames}")

    # infer text column (common names)
    text_col = None
    for c in ["text", "caption", "captions", "prompt", "description", "sentence"]:
        if c in colnames:
            text_col = c
            break

    # label fallback
    label_col = None
    for c in ["label", "labels", "class", "category", "category_name"]:
        if c in colnames:
            label_col = c
            break

    if image_col is None:
        raise RuntimeError(f"[HeuristicDS] HF: could not infer an image column. columns={colnames}")

    class_names = None
    try:
        if feats is not None and label_col is not None:
            f = feats[label_col]
            if getattr(f, "__class__", None).__name__ == "ClassLabel":
                class_names = list(getattr(f, "names", []))
                if class_names:
                    log.cyan(f"[HeuristicDS] HF: detected ClassLabel with {len(class_names)} names")
    except Exception:
        pass

    # if neither text_col nor label_col found, do string heuristic, then interactive choice if ambiguous
    if text_col is None and label_col is None:
        text_col = _infer_hf_string_text_column(
            ds_id=ds_id,
            dset=dset,
            colnames=colnames,
            log=log,
            interactive=True,   # only used in this "all else failed" path
        )
        if text_col is None:
            raise RuntimeError(
                "[HeuristicDS] HF: could not infer a text/caption column nor a label column.\n"
                f"columns={colnames}"
            )

    return dset, image_col, text_col, label_col, class_names


# ----------------------------
# Utilities
# ----------------------------

# cache size helper
def _dir_size_bytes(root: str, max_files: int = 2_000_000) -> int:
    total = 0
    n = 0
    for r, _ds, fs in os.walk(root):
        for fn in fs:
            n += 1
            if n > max_files:
                return total
            try:
                total += os.path.getsize(os.path.join(r, fn))
            except Exception:
                pass
    return total

def _fmt_bytes(n: int) -> str:
    f = float(max(0, n))
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if f < 1024.0:
            return f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} PB"


def _looks_like_url(x: Any) -> bool:
    try:
        return isinstance(x, str) and bool(_URL_RE.match(x.strip()))
    except Exception:
        return False

def _infer_hf_image_url_column(dset, colnames: List[str], log, sample_n: int = 64) -> Optional[str]:
    """
    Pick a column that *looks* like an image URL by sampling rows.
    Preference order: names containing 'image' + 'url', then any 'url'ish col.
    """
    n = len(dset)
    if n <= 0:
        return None

    take = min(sample_n, n)
    idxs = list(range(take))

    # rank columns by name first (cheap)
    def _name_score(c: str) -> int:
        lc = c.lower()
        s = 0
        if "image" in lc: s += 3
        if "img" in lc: s += 2
        if "url" in lc: s += 4
        if "uri" in lc: s += 2
        return s

    cols_ranked = sorted(colnames, key=_name_score, reverse=True)

    best = None
    best_hits = 0

    for c in cols_ranked:
        hits = 0
        for i in idxs:
            try:
                v = dset[i].get(c, None)
            except Exception:
                v = None
            if _looks_like_url(v):
                hits += 1

        if hits > best_hits:
            best_hits = hits
            best = c

    if best is not None and best_hits >= max(3, take // 4):
        log.yellow(f"[HeuristicDS] HF: inferred image URL column -> {best} (hits={best_hits}/{take})")
        return best

    return None

def _get_basename_index_cached(image_root: str, log) -> Tuple[Dict[str, str], int]:
    root = os.path.abspath(image_root)

    if root in _BASENAME_INDEX_CACHE:
        return _BASENAME_INDEX_CACHE[root]

    idx, dupes = _build_basename_index(image_root=root, log=log)
    _BASENAME_INDEX_CACHE[root] = (idx, dupes)
    if len(idx) == 0:
        _BASENAME_INDEX_EMPTY.add(root)
    return idx, dupes

_URL_CACHE_NOTICE_PRINTED = False

def _url_cache_dir(cache_namespace: Optional[str] = None, log=None) -> str:
    global _URL_CACHE_NOTICE_PRINTED

    base_root = os.path.join(os.path.expanduser("~"), ".cache", "heuristic_ds_url_cache")

    base = base_root
    if cache_namespace and str(cache_namespace).strip():
        import hashlib
        ns = str(cache_namespace).strip().encode("utf-8")
        h = hashlib.sha256(ns).hexdigest()[:24]
        base = os.path.join(base_root, h)

    try:
        os.makedirs(base, exist_ok=True)
        os.makedirs(base_root, exist_ok=True)
    except Exception:
        pass

    # don't print from DataLoader workers (Windows = separate processes)
    try:
        from torch.utils.data import get_worker_info
        in_worker = (get_worker_info() is not None)
    except Exception:
        in_worker = False

    if (log is not None) and (not in_worker) and (not _URL_CACHE_NOTICE_PRINTED):
        _URL_CACHE_NOTICE_PRINTED = True
        try:
            sz = _dir_size_bytes(base_root)
            log.yellow(f"[HeuristicDS] URL image cache root: {os.path.abspath(base_root)}  (size={_fmt_bytes(sz)})")
            log.yellow("[HeuristicDS] Tip: delete unused subfolders here if disk grows over time.")
        except Exception:
            pass

    return base

def _safe_slug(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120]

def _safe_open_image_url(
    url: str,
    log,
    timeout: float = 10.0,
    cache_namespace: Optional[str] = None,
) -> Image.Image:
    """
    Downloads an image URL (if not cached) and returns PIL RGB.
    Uses on-disk cache under ~/.cache/heuristic_ds_url_cache/<hash-of-namespace>/.
    Raises on failure.
    """
    # define url, h, ext BEFORE using them
    url = str(url).strip()
    if not _looks_like_url(url):
        raise RuntimeError(f"Not a URL: {url}")

    import hashlib
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]

    ext = os.path.splitext(url.split("?", 1)[0])[1].lower()
    if ext not in _IMAGE_EXTS:
        ext = ".img"

    # use namespaced cache dir (not the global root)
    cache_dir = _url_cache_dir(cache_namespace=cache_namespace, log=log)
    cache_path = os.path.join(cache_dir, f"{h}{ext}")

    # cache hit
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            return _safe_open_image(cache_path)
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass

    # add UA header (many hosts block default urllib/requests)
    headers = {"User-Agent": "Mozilla/5.0"}

    data = None
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        data = r.content
    except Exception:
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
        except Exception as e:
            raise RuntimeError(f"URL fetch failed: {e}")

    if not data:
        raise RuntimeError("Empty response body")

    # write cache (best effort), else decode from memory
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        # atomic-ish replace (Windows-friendly)
        try:
            os.replace(tmp_path, cache_path)
        except Exception:
            # fallback: try direct write
            with open(cache_path, "wb") as f:
                f.write(data)
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        return _safe_open_image(cache_path)
    except Exception:
        return Image.open(io.BytesIO(data)).convert("RGB")


def _safe_open_image(x: Any) -> Image.Image:
    """
    Robust-ish image open:
      - path string
      - raw bytes / bytearray / memoryview
      - dict with {"bytes": ...} or {"data": ...} or {"array": ...}
      - PIL.Image.Image
    """
    if hasattr(x, "convert"):
        return x.convert("RGB")

    if isinstance(x, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(x)))
        return img.convert("RGB")

    if isinstance(x, dict):
        for k in ["bytes", "data"]:
            if k in x and x[k]:
                b = x[k]
                if isinstance(b, (bytes, bytearray, memoryview)):
                    img = Image.open(io.BytesIO(bytes(b)))
                    return img.convert("RGB")
        if "array" in x and x["array"] is not None:
            import numpy as np
            arr = np.asarray(x["array"])
            if arr.ndim == 3:
                img = Image.fromarray(arr.astype("uint8"), mode="RGB")
                return img.convert("RGB")
        # fallback to path-like keys
        for k in ["path", "image", "image_path", "file", "file_name", "filename"]:
            if k in x and x[k]:
                x = x[k]
                break

    # default: treat as path string
    path = str(x)
    img = Image.open(path)
    try:
        return img.convert("RGB")
    except Exception:
        img.load()
        return img.convert("RGB")

def _looks_like_existing_path(p: str) -> bool:
    try:
        return os.path.exists(os.path.expandvars(os.path.expanduser(p)))
    except Exception:
        return False

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
    except Exception:
        pass


class _Logger:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _p(self, msg: str):
        if self.enabled:
            print(msg)

    def cyan(self, msg: str):
        self._p(Fore.CYAN + msg + Style.RESET_ALL)

    def green(self, msg: str):
        self._p(Fore.GREEN + msg + Style.RESET_ALL)

    def yellow(self, msg: str):
        self._p(Fore.YELLOW + msg + Style.RESET_ALL)

    def red(self, msg: str):
        self._p(Fore.RED + msg + Style.RESET_ALL)


def _make_logger(enabled: bool) -> _Logger:
    return _Logger(enabled)