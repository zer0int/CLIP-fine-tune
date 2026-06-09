#!/usr/bin/env python3
"""
clip_ocr_rewrites_to_labels.py

Convert GPT-OSS OCR-caption rewrite records back into a plain dataset-label JSON.

Input can be either:
  1. *_gptoss_rewrites.json
     {
       "data/9/0.jpg": {
         "labels_rewritten": [...]
       }
     }

  2. *_leftovers_filled.json
     [
       {
         "image_key": "data/9/0.jpg",
         "labels_rewritten": [...]
       }
     ]

  3. compact manual rewrites:
     {
       "data/9/0.jpg": ["label 0", "label 1"]
     }

By default, this script writes:
  {input_stem}_labels_gpt_oss.json

If --base-labels is provided, it starts from that full dataset-label JSON and overlays
the rewrites. This is usually what you want for partial rewrite files.

Examples:

  # Convert a full GPT-OSS rewrites dict, using original labels as base:
  python clip_ocr_rewrites_to_labels.py short-coco-spright-train-10_11_dump_gpt_oss_gptoss_rewrites.json --base-labels short-coco-spright-train-10_11.json

  # Convert already-merged rewrite output directly:
  python clip_ocr_rewrites_to_labels.py short-coco-spright-train-10_11_dump_gpt_oss_rewritten_labels.json

  # Convert manual leftovers into a labels json overlaid onto base:
  python clip_ocr_rewrites_to_labels.py short-coco-spright-train-10_11_manual_rewrites.json --base-labels short-coco-spright-train-10_11.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_label_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def normalize_rewrites(data: Any) -> dict[str, list[str]]:
    """
    Normalize multiple accepted rewrite formats to:
      image_key -> labels list
    """
    out: dict[str, list[str]] = {}

    if isinstance(data, dict):
        # Already plain dataset labels or compact manual rewrites:
        # {"image": ["label0", "label1"]}
        if all(is_label_list(v) for v in data.values()):
            return {str(k): [clean_text(x) for x in v] for k, v in data.items()}

        # GPT-OSS rewrites:
        # {"image": {"labels_rewritten": [...]}}
        for image_key, record in data.items():
            if not isinstance(record, dict):
                continue

            labels = (
                record.get("labels_rewritten")
                or record.get("labels")
                or record.get("rewritten_labels")
            )
            if is_label_list(labels):
                out[str(image_key)] = [clean_text(x) for x in labels]

        return out

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            image_key = str(item.get("image_key") or item.get("image_path") or "")
            labels = (
                item.get("labels_rewritten")
                or item.get("labels")
                or item.get("rewritten_labels")
            )
            if image_key and is_label_list(labels):
                out[image_key] = [clean_text(x) for x in labels]

        return out

    raise ValueError("Input JSON must be a dict or list.")


def output_path_for(input_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return input_path.with_name(f"{input_path.stem}_labels_gpt_oss.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GPT-OSS OCR rewrite records into plain dataset labels JSON.")
    parser.add_argument("input_json", help="Rewrite JSON, merged rewrite JSON, manual compact JSON, or filled leftovers JSON.")
    parser.add_argument("--base-labels", default=None, help="Optional original full dataset-label JSON to overlay rewrites onto.")
    parser.add_argument("--output", default=None, help="Output path. Default: {input_stem}_labels_gpt_oss.json")
    parser.add_argument("--require-base-keys", action="store_true", help="Fail if a rewrite key is not present in --base-labels.")
    parser.add_argument("--sort-keys", action="store_true", help="Sort JSON keys in output.")
    args = parser.parse_args()

    input_path = Path(args.input_json).resolve()
    output_path = output_path_for(input_path, args.output)

    input_data = load_json(input_path)

    # Special case: input is already a plain dataset labels JSON.
    # If no base is given and all values are label lists, just clean/write it.
    if args.base_labels is None and isinstance(input_data, dict) and all(is_label_list(v) for v in input_data.values()):
        labels_out = {str(k): [clean_text(x) for x in v] for k, v in input_data.items()}
        save_json(output_path, dict(sorted(labels_out.items())) if args.sort_keys else labels_out)
        print(f"[done] input already looked like dataset labels")
        print(f"[done] wrote {output_path} with {len(labels_out)} entries")
        return

    rewrites = normalize_rewrites(input_data)
    if not rewrites:
        raise ValueError("No rewritten labels found in input.")

    if args.base_labels:
        base_path = Path(args.base_labels).resolve()
        base = load_json(base_path)
        if not isinstance(base, dict) or not all(is_label_list(v) for v in base.values()):
            raise ValueError("--base-labels must be a dict mapping image paths to label lists.")

        labels_out = {str(k): [clean_text(x) for x in v] for k, v in base.items()}

        missing = [k for k in rewrites if k not in labels_out]
        if missing and args.require_base_keys:
            raise KeyError(f"{len(missing)} rewrite keys not found in base labels. First few: {missing[:10]}")

        applied = 0
        for image_key, labels in rewrites.items():
            if image_key in labels_out or not args.require_base_keys:
                labels_out[image_key] = labels
                applied += 1

        if args.sort_keys:
            labels_out = dict(sorted(labels_out.items()))

        save_json(output_path, labels_out)
        print(f"[done] base entries={len(base)} rewrites={len(rewrites)} applied={applied} missing_in_base={len(missing)}")
        print(f"[done] wrote {output_path} with {len(labels_out)} entries")
        return

    # No base: output only rewritten entries as plain labels.
    labels_out = rewrites
    if args.sort_keys:
        labels_out = dict(sorted(labels_out.items()))

    save_json(output_path, labels_out)
    print(f"[done] wrote rewritten-only labels {output_path} with {len(labels_out)} entries")
    print("[note] pass --base-labels ORIGINAL.json if you want the full dataset with rewrites overlaid.")


if __name__ == "__main__":
    main()
