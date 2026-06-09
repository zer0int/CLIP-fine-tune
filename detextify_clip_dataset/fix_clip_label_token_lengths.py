#!/usr/bin/env python3
"""
fix_clip_label_token_lengths.py

Tokenize all labels in a JSON mapping image_key -> list[str] using oaiclip/OpenAI-CLIP syntax:

    import oaiclip as clip
    clip.tokenize(text, context_length=..., truncate=False)

Any label that exceeds CLIP's max token length is handled without crashing.

Input JSON format:
{
  "data/0/0.jpg": [
    "A long label ...",
    "Another label ..."
  ],
  ...
}

Modes:

1) Auto mode:
   Repeatedly strip from the back until the previous comma or period.
   If stripping at comma, replace it with ".".
   Retry until clip.tokenize(..., truncate=False) succeeds.
   Save to "{original_json_filename}_truncated.json".

2) Interactive mode:
   For each too-long label, show current token count, max tokens, excess tokens,
   image key, label index, and the original label.
   Opens an editable prompt prefilled with the current label.
   Enter submits; if still too long, it asks again.

Examples:

  python fix_clip_label_token_lengths.py labels.json --auto

  python fix_clip_label_token_lengths.py labels.json

  python fix_clip_label_token_lengths.py labels.json --context-length 77 --output labels_fixed.json
  python fix_clip_label_token_lengths.py labels.json --context-length 248 --output longclip_labels_fixed.json

Interactive editing notes:
  - On Windows, editable prefilled input uses pyreadline3 if installed.
  - Otherwise it falls back to plain input after printing the current label.
  - Install optional helper:
      pip install pyreadline3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

import oaiclip as clip


def load_json(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict, got {type(data)}")

    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise ValueError(f"Expected string key, got {type(k)}")
        if not isinstance(v, list):
            raise ValueError(f"Expected list labels at key {k!r}, got {type(v)}")
        labels = []
        for i, label in enumerate(v):
            if not isinstance(label, str):
                raise ValueError(f"Expected string label at {k!r}[{i}], got {type(label)}")
            labels.append(label)
        out[k] = labels
    return out


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_truncated{input_path.suffix}")


def write_json(path: Path, data: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def canonicalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_ok(text: str, context_length: int) -> tuple[bool, str | None]:
    try:
        _ = clip.tokenize(text, context_length=context_length, truncate=False)
        return True, None
    except Exception as exc:
        return False, str(exc)


def token_count_truncate_true(text: str, context_length: int) -> int:
    """
    Estimate number of non-zero token slots under truncation. This is useful for display only.

    OpenAI CLIP tokenize pads with 0. Usually EOT is the highest token id and present before padding.
    This count includes SOT/EOT if they are non-zero. It is not used for correctness.
    """
    try:
        toks = clip.tokenize(text, context_length=context_length, truncate=True)
        row = toks[0] if getattr(toks, "ndim", 0) == 2 else toks
        return int((row != 0).sum().item())
    except Exception:
        return -1


def infer_overflow_count(error_msg: str | None, context_length: int, text: str) -> tuple[int | None, int | None]:
    """
    Try to infer actual token length from common OpenAI CLIP error messages.

    openai/clip often raises:
      RuntimeError: Input ... is too long for context length 77

    Some forks include lengths; many do not. Fallback returns (None, None).
    """
    if not error_msg:
        return None, None

    # Common variants:
    # "Token indices sequence length is longer than the specified maximum sequence length for this model (84 > 77)"
    m = re.search(r"(\d+)\s*>\s*(\d+)", error_msg)
    if m:
        got = int(m.group(1))
        max_len = int(m.group(2))
        return got, max(0, got - max_len)

    # "too long for context length 77"
    m = re.search(r"context length\s+(\d+)", error_msg, flags=re.IGNORECASE)
    if m:
        return None, None

    return None, None


def display_token_status(text: str, context_length: int, error_msg: str | None) -> dict[str, Any]:
    truncated_count = token_count_truncate_true(text, context_length)
    actual_count, excess = infer_overflow_count(error_msg, context_length, text)
    return {
        "context_length": context_length,
        "truncated_nonpad_count": truncated_count,
        "actual_count_if_known": actual_count,
        "excess_if_known": excess,
        "error": error_msg,
    }


def strip_back_to_previous_boundary(text: str) -> tuple[str, str]:
    """
    Strip from the back until previous comma or period.

    If comma is selected:
      "foo, bar baz" -> "foo."
    If period is selected:
      "foo. bar baz" -> "foo."
    If no comma/period exists:
      fallback to dropping last word.
    """
    s = canonicalize_spaces(text)
    if not s:
        return s, "empty"

    # Ignore a trailing punctuation mark as a boundary; otherwise no progress for "... ."
    scan = s.rstrip()
    while scan and scan[-1] in ",.":
        scan = scan[:-1].rstrip()

    last_comma = scan.rfind(",")
    last_period = scan.rfind(".")
    idx = max(last_comma, last_period)

    if idx >= 0:
        punct = scan[idx]
        prefix = scan[:idx].rstrip()
        if not prefix:
            return "", "boundary_to_empty"
        if punct == ",":
            return prefix + ".", "comma"
        return prefix + ".", "period"

    # Fallback: drop one trailing word.
    parts = scan.split()
    if len(parts) <= 1:
        return "", "word_to_empty"
    return " ".join(parts[:-1]).rstrip(" ,") + ".", "word"


def auto_truncate_label(text: str, context_length: int, max_steps: int = 1000) -> tuple[str, int, list[str]]:
    current = canonicalize_spaces(text)
    reasons: list[str] = []

    ok, err = tokenize_ok(current, context_length)
    if ok:
        return current, 0, reasons

    for step in range(1, max_steps + 1):
        new_text, reason = strip_back_to_previous_boundary(current)
        reasons.append(reason)

        if new_text == current:
            # Last-resort hard word drop, should almost never trigger.
            words = current.split()
            new_text = " ".join(words[:-1]).strip() if len(words) > 1 else ""

        current = canonicalize_spaces(new_text)

        if not current:
            raise RuntimeError(
                "Auto truncation stripped label to empty while trying to satisfy token limit. "
                f"Original label was: {text!r}"
            )

        ok, err = tokenize_ok(current, context_length)
        if ok:
            return current, step, reasons

    raise RuntimeError(f"Auto truncation exceeded {max_steps} steps for label: {text!r}")


def editable_input(prompt: str, prefill: str) -> str:
    """
    Editable prefilled input where possible.

    Linux/macOS usually use readline.
    Windows can use pyreadline3 if installed.
    Fallback prints prefill and asks for replacement text.
    """
    try:
        import readline  # type: ignore

        def hook():
            readline.insert_text(prefill)
            readline.redisplay()

        readline.set_startup_hook(hook)
        try:
            return input(prompt)
        finally:
            readline.set_startup_hook(None)
    except Exception:
        pass

    # Windows optional helper.
    try:
        import pyreadline3  # noqa: F401  # type: ignore
        import readline  # type: ignore

        def hook():
            readline.insert_text(prefill)
            readline.redisplay()

        readline.set_startup_hook(hook)
        try:
            return input(prompt)
        finally:
            readline.set_startup_hook(None)
    except Exception:
        pass

    print("\n[editable prefill unavailable]")
    print("Current label:")
    print(prefill)
    print()
    return input(prompt)


def interactive_fix_label(
    image_key: str,
    label_index: int,
    original_text: str,
    context_length: int,
) -> tuple[str, int]:
    current = original_text
    attempts = 0

    while True:
        attempts += 1
        ok, err = tokenize_ok(current, context_length)
        if ok:
            return current, attempts - 1

        status = display_token_status(current, context_length, err)
        actual_count = status["actual_count_if_known"]
        excess = status["excess_if_known"]
        truncated_count = status["truncated_nonpad_count"]

        print("\n" + "=" * 100)
        print(f"[TOO LONG] image_key={image_key} label_index={label_index}")
        print(f"max/context_length: {context_length}")
        if actual_count is not None:
            print(f"tokens: {actual_count}")
            print(f"excess: {excess}")
        else:
            print(f"tokens: unknown exact count from tokenizer error")
            print(f"nonpad tokens under truncate=True: {truncated_count}")
            print(f"excess: unknown")
        print(f"tokenize error: {err}")
        print("-" * 100)

        edited = editable_input("Edit label, then Enter to retry:\n> ", current)
        edited = canonicalize_spaces(edited)

        if not edited:
            print("[warn] Empty label rejected; keep editing.")
            continue

        current = edited


def process_labels(
    data: dict[str, list[str]],
    context_length: int,
    auto: bool,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    out = deepcopy(data)

    total_labels = 0
    too_long_labels = 0
    truncated_labels = 0
    unchanged_labels = 0
    failures: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []

    for image_key, labels in out.items():
        for i, label in enumerate(labels):
            total_labels += 1
            ok, err = tokenize_ok(label, context_length)
            if ok:
                unchanged_labels += 1
                continue

            too_long_labels += 1

            if auto:
                try:
                    new_label, steps, reasons = auto_truncate_label(label, context_length)
                except Exception as exc:
                    failures.append({
                        "image_key": image_key,
                        "label_index": i,
                        "label": label,
                        "error": str(exc),
                    })
                    continue
            else:
                new_label, steps = interactive_fix_label(image_key, i, label, context_length)
                reasons = ["interactive"]

            ok2, err2 = tokenize_ok(new_label, context_length)
            if not ok2:
                failures.append({
                    "image_key": image_key,
                    "label_index": i,
                    "label": label,
                    "attempted_label": new_label,
                    "error": err2,
                })
                continue

            out[image_key][i] = new_label
            truncated_labels += 1

            changes.append({
                "image_key": image_key,
                "label_index": i,
                "old": label,
                "new": new_label,
                "steps": steps,
                "reasons": reasons,
                "old_status": display_token_status(label, context_length, err),
            })

            print(
                f"[fixed] {image_key}[{i}] "
                f"steps={steps} old_chars={len(label)} new_chars={len(new_label)}"
            )

    summary = {
        "context_length": context_length,
        "total_images": len(data),
        "total_labels": total_labels,
        "unchanged_labels": unchanged_labels,
        "too_long_labels": too_long_labels,
        "truncated_or_edited_labels": truncated_labels,
        "failures": failures,
        "num_failures": len(failures),
        "changes": changes,
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix CLIP token-length failures in label JSON using oaiclip tokenization.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--context-length", type=int, default=77)
    parser.add_argument("--auto", action="store_true", help="Automatically strip from back to previous comma/period until tokenization succeeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input_json.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(input_path)
    summary_path = args.summary_json.resolve() if args.summary_json else output_path.with_name(output_path.stem + "_summary.json")

    print(f"[load] {input_path}")
    data = load_json(input_path)

    print(f"[plan] images={len(data)} context_length={args.context_length} auto={args.auto}")
    print(f"[plan] output={output_path}")
    print(f"[plan] summary={summary_path}")

    fixed, summary = process_labels(
        data=data,
        context_length=args.context_length,
        auto=args.auto,
    )

    if summary["num_failures"] > 0:
        print("\n[WARN] failures occurred; output still written with successful fixes only.")
        for f in summary["failures"][:10]:
            print(f"  - {f.get('image_key')}[{f.get('label_index')}]: {f.get('error')}")
        if summary["num_failures"] > 10:
            print(f"  ... and {summary['num_failures'] - 10} more")

    write_json(output_path, fixed)
    write_json(summary_path, summary)

    print("\n[done]")
    print(f"  total_images: {summary['total_images']}")
    print(f"  total_labels: {summary['total_labels']}")
    print(f"  unchanged_labels: {summary['unchanged_labels']}")
    print(f"  too_long_labels: {summary['too_long_labels']}")
    print(f"  truncated_or_edited_labels: {summary['truncated_or_edited_labels']}")
    print(f"  failures: {summary['num_failures']}")
    print(f"  output: {output_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
