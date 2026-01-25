from __future__ import annotations

import os
import torch
import random
import json
import pandas as pd
from PIL import Image
from typing import List, Dict
from torch.utils.data import Dataset
from torchvision import datasets


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def infinite_dataloader(dl):
    """Yield batches from a DataLoader forever (cycles)."""
    while True:
        for batch in dl:
            yield batch

class AdversarialTripletTextDataset(Dataset):
    """
    JSON maps image_path -> [label0, label1, label2]
    - label0,label1: adversarial / should be minimized (negative)
    - label2: positive / should be maximized
    NOTE: image paths are resolved relative to the annotations_file directory if not absolute.
    """
    def __init__(self, annotations_file: str, transform=None, tokenize_fn=None):
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self.annotations_file = annotations_file
        self.base_dir = os.path.dirname(os.path.abspath(annotations_file))

        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_or_abs = self.image_paths[idx]
        image_path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(self.base_dir, rel_or_abs)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # expected [3,H,W] float tensor

        assert image.shape[0] == 3, f"Image shape before adv aug: {image.shape}"

        labels = self.annotations[rel_or_abs]
        while len(labels) < 3:
            labels.append("")

        if self.tokenize_fn is None:
            raise RuntimeError("AdversarialTripletTextDataset requires tokenize_fn.")
        texts = self.tokenize_fn(labels, truncate=True)  # (3, seq_len)

        return image, texts  # image: [3,224,224]; texts: [3, seq_len]


class ImageTextDataset(Dataset):
    """
    Main dataset: COCO-SPRIGHT loader.
    """
    def __init__(self, image_folder, annotations_file, transform=None, pretok_batch_size: int = 4096, tokenize_fn=None):
        self.image_folder = image_folder
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        if self.tokenize_fn is None:
            import gmpclipregression as clip
            self.tokenize_fn = clip.tokenize
        
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_paths = list(self.annotations.keys())

        # Build per-image candidate label strings
        labels_per_image: List[List[str]] = []
        all_labels: List[str] = []

        for rel_path in self.image_paths:
            labels = self.annotations.get(rel_path, [])
            if len(labels) >= 2:
                cands = [labels[0], labels[1]]
            elif len(labels) == 1:
                cands = [labels[0]]
            else:
                cands = [""]  # keep behavior defined

            labels_per_image.append(cands)
            all_labels.extend(cands)

        # Deduplicate labels -> token table
        unique_labels = sorted(set(all_labels))
        if "" not in unique_labels:
            unique_labels.append("")

        self._label_to_idx: Dict[str, int] = {s: i for i, s in enumerate(unique_labels)}

        # Tokenize unique labels in batches (faster)
        tok_chunks = []
        for start in range(0, len(unique_labels), int(pretok_batch_size)):
            chunk = unique_labels[start:start + int(pretok_batch_size)]
            tok = self.tokenize_fn(chunk, truncate=True) # [B, 77] (CPU)
            tok_chunks.append(tok)

        self._token_table = torch.cat(tok_chunks, dim=0).contiguous()  # [n_unique, 77] on CPU

        # Store per-image candidate indices (NOT tensors) to keep it compact
        self._cand_token_idxs: List[List[int]] = []
        for cands in labels_per_image:
            self._cand_token_idxs.append([self._label_to_idx.get(s, self._label_to_idx[""]) for s in cands])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_rel_path = self.image_paths[idx]
        image_path = os.path.join(self.image_folder, image_rel_path)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        cand_idxs = self._cand_token_idxs[idx]
        tok_i = random.choice(cand_idxs) if len(cand_idxs) > 1 else cand_idxs[0]
        text = self._token_table[tok_i]  # [77] LongTensor

        return image, text


class BalancedImageFolderTeacher(Dataset):
    """
    ImageNet for Teacher.
    """
    def __init__(self, root: str, transform, max_per_class: int = 10, seed: int = 42):
        self.base = datasets.ImageFolder(root=root, transform=transform)
        rng = random.Random(seed)

        class_to_indices = {}
        for idx, (_, label) in enumerate(self.base.samples):
            class_to_indices.setdefault(label, []).append(idx)

        selected_indices = []
        for label, idxs in class_to_indices.items():
            if len(idxs) <= max_per_class:
                chosen = idxs
            else:
                chosen = rng.sample(idxs, max_per_class)
            selected_indices.extend(chosen)

        self.indices = sorted(selected_indices)
        print(f"[Teacher] {root}: using {len(self.indices)} images ({max_per_class} per class cap).")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, _ = self.base[idx]
        return img

class CroppedImageCSVFileDataset(Dataset):
    """
    ImageNet/ObjectNet MVT for quick ZS and LP.
    """    
    def __init__(self, df: pd.DataFrame, image_folder: str, transform=None):
        self.data = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image']
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = int(self.data.iloc[idx]['label_idx'])
        return image, label_idx

class TinyImageFolderDataset(Dataset):
    """
    Typographic attack val dataset.
    """    
    def __init__(self, folder: str, preprocess):
        self.folder = folder
        self.preprocess = preprocess
        files = []
        if os.path.isdir(folder):
            for root, _, fnames in os.walk(folder):
                for fn in fnames:
                    if fn.lower().endswith(_IMG_EXTS):
                        files.append(os.path.join(root, fn))
        self.files = sorted(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = self.preprocess(img)
        return img, p