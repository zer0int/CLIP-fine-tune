#!/usr/bin/env python3
"""
clip_ocr_rewrite_leftovers.py

Extract GPT-OSS OCR-caption rewrite leftovers and merge manual fixes back.

Where 'manual' can also imply 'give to SOTA LLM and let the big one do the rest' (recommended).

Usage:
  python clip_ocr_rewrite_leftovers.py extract train_dump_gpt_oss.json
  python clip_ocr_rewrite_leftovers.py extract train_dump_gpt_oss.json --errors-only
  python clip_ocr_rewrite_leftovers.py merge train_dump_gpt_oss.json --manual manual_rewrites.json

Manual formats:

1) Compact:
{
  "data/9/940.jpg": ["rewritten label 0", "rewritten label 1"]
}

2) List:
[
  {
    "image_key": "data/9/940.jpg",
    "labels_rewritten": ["rewritten label 0", "rewritten label 1"],
    "notes": "manual fix"
  }
]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[\wÄÖÜäöüß]+(?:[-'’][\wÄÖÜäöüß]+)*", re.UNICODE)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "there", "their", "his", "her", "they",
    "them", "you", "your", "we", "our", "he", "she", "i", "me", "my",
    "into", "over", "under", "above", "below", "near", "next", "front",
    "back", "left", "right", "large", "small", "white", "black", "red",
    "blue", "green", "yellow", "orange", "brown", "gray", "grey",
}


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_token(text: str) -> str:
    text = str(text or "").lower().replace("ß", "ss")
    return re.sub(r"[^0-9a-zäöü]+", "", text)


def norm_tokens(text: str, min_len: int = 3) -> list[str]:
    toks = []
    for match in TOKEN_RE.finditer(str(text or "")):
        tok = normalize_token(match.group(0))
        if len(tok) >= min_len and tok not in STOPWORDS:
            toks.append(tok)
    return toks


def norm_compact(text: str) -> str:
    return "".join(norm_tokens(text, min_len=1))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def paths_for(dump_path: Path) -> dict[str, Path]:
    stem = dump_path.stem
    return {
        "rewrites": dump_path.with_name(f"{stem}_gptoss_rewrites.json"),
        "errors": dump_path.with_name(f"{stem}_gptoss_errors.json"),
        "leftovers": dump_path.with_name(f"{stem}_leftovers_for_human.json"),
        "merged": dump_path.with_name(f"{stem}_rewritten_labels.json"),
    }


def get_entry_id(entry: dict[str, Any]) -> str:
    return str(entry.get("image_key") or entry.get("image_path") or "")


def build_forbidden_terms(entry: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    raw_terms = []
    raw_terms.extend(entry.get("forbidden_caption_spans") or [])
    raw_terms.extend(entry.get("forbidden_ocr_text") or [])

    for m in entry.get("flagged_spans") or []:
        raw_terms.append(m.get("span_text", ""))
        raw_terms.append(m.get("ocr_text", ""))

    for o in entry.get("ocr") or []:
        raw_terms.append(o.get("text", ""))

    phrases = []
    tokens = set()
    compact_phrases = set()

    for term in raw_terms:
        term = clean_text(term)
        toks = norm_tokens(term)
        if not toks:
            continue
        phrases.append(term)
        tokens.update(toks)

        # strict=True is CLIP-minded: "build" also catches "building" via compact match.
        # strict=False only compact-matches multi-token phrases.
        if strict or len(toks) >= 2:
            compact_phrases.add("".join(toks))

    return {
        "raw_terms": sorted(set(phrases), key=lambda x: (x.lower(), x)),
        "tokens": sorted(tokens),
        "compact_phrases": sorted(compact_phrases),
    }


def contains_forbidden(text: str, forbidden: dict[str, Any]) -> list[str]:
    text_tokens = set(norm_tokens(text))
    text_compact = norm_compact(text)
    hits = []

    for tok in forbidden.get("tokens", []):
        if tok in text_tokens:
            hits.append(tok)

    for phrase in forbidden.get("compact_phrases", []):
        if len(phrase) >= 4 and phrase in text_compact:
            hits.append(phrase)

    return sorted(set(hits))


def effective_flagged_indices(entry: dict[str, Any], strict: bool = True) -> list[int]:
    labels = [clean_text(x) for x in entry.get("labels", [])]
    forbidden = build_forbidden_terms(entry, strict=strict)
    flagged = set(int(x) for x in (entry.get("flagged_caption_indices") or []))

    for i, label in enumerate(labels):
        if contains_forbidden(label, forbidden):
            flagged.add(i)

    return sorted(i for i in flagged if 0 <= i < len(labels))


def validate_labels(entry: dict[str, Any], labels: list[str], strict: bool = True) -> tuple[bool, dict[str, list[str]]]:
    original = [clean_text(x) for x in entry.get("labels", [])]
    labels = [clean_text(x) for x in labels]
    if len(labels) != len(original):
        return False, {"__length__": [f"expected {len(original)}, got {len(labels)}"]}

    forbidden = build_forbidden_terms(entry, strict=strict)
    hits_by_label = {}
    for i, label in enumerate(labels):
        hits = contains_forbidden(label, forbidden)
        if hits:
            hits_by_label[str(i)] = hits

    return not hits_by_label, hits_by_label


def make_leftover(entry: dict[str, Any], error: Any, strict: bool = True) -> dict[str, Any]:
    forbidden = build_forbidden_terms(entry, strict=strict)
    return {
        "image_key": entry.get("image_key"),
        "image_path": entry.get("image_path"),
        "labels": entry.get("labels", []),
        "labels_rewritten": [],
        "notes": "",
        "instruction": "Fill labels_rewritten with the full label list, same length/order, with forbidden OCR/text terms removed.",
        "flagged_caption_indices": entry.get("flagged_caption_indices", []),
        "effective_flagged_indices": effective_flagged_indices(entry, strict=strict),
        "forbidden_terms": forbidden["raw_terms"],
        "forbidden_tokens": forbidden["tokens"],
        "flagged_spans": entry.get("flagged_spans", []),
        "ocr": entry.get("ocr", []),
        "current_error": error,
    }


def normalize_manual(data: Any) -> dict[str, dict[str, Any]]:
    out = {}

    if isinstance(data, dict):
        for image_key, value in data.items():
            if isinstance(value, list):
                out[str(image_key)] = {
                    "labels_rewritten": [clean_text(x) for x in value],
                    "notes": "manual",
                }
            elif isinstance(value, dict):
                labels = value.get("labels_rewritten", value.get("labels", []))
                out[str(image_key)] = {
                    "labels_rewritten": [clean_text(x) for x in labels],
                    "notes": clean_text(value.get("notes", "manual")),
                }
        return out

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            image_key = str(item.get("image_key") or item.get("image_path") or "")
            labels = item.get("labels_rewritten", item.get("labels", []))
            if image_key:
                out[image_key] = {
                    "labels_rewritten": [clean_text(x) for x in labels],
                    "notes": clean_text(item.get("notes", "manual")),
                }
        return out

    raise ValueError("manual JSON must be a dict or list")


def extract_leftovers(args: argparse.Namespace) -> None:
    dump_path = Path(args.dump_json).resolve()
    paths = paths_for(dump_path)
    strict = not args.literal_validator

    dump = load_json(dump_path, {})
    entries = dump.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("dump must contain an entries list")

    rewrites = load_json(Path(args.rewrites) if args.rewrites else paths["rewrites"], {})
    errors = load_json(Path(args.errors) if args.errors else paths["errors"], {})

    leftovers = []

    for entry in entries:
        image_key = get_entry_id(entry)
        if not image_key:
            continue

        record = rewrites.get(image_key)
        error = errors.get(image_key)

        needs = False
        reason = None

        if args.errors_only:
            needs = image_key in errors
            reason = error
        elif not record:
            needs = True
            reason = error or "missing rewrite"
        else:
            ok, hits = validate_labels(entry, record.get("labels_rewritten", []), strict=strict)
            if not ok:
                needs = True
                reason = {"invalid_existing_rewrite": hits}

        if needs:
            leftovers.append(make_leftover(entry, reason, strict=strict))
            if args.limit is not None and len(leftovers) >= args.limit:
                break

    output = Path(args.output) if args.output else paths["leftovers"]
    save_json(output, leftovers)

    compact = {item["image_key"]: item["labels_rewritten"] for item in leftovers}
    compact_path = output.with_name(output.stem + "_compact_template.json")
    save_json(compact_path, compact)

    print(f"[done] wrote {output} with {len(leftovers)} leftovers")
    print(f"[done] wrote {compact_path}")
    for item in leftovers:
        print(" ", item["image_key"])


def merge_manual(args: argparse.Namespace) -> None:
    dump_path = Path(args.dump_json).resolve()
    paths = paths_for(dump_path)
    strict = not args.literal_validator

    dump = load_json(dump_path, {})
    entries = dump.get("entries", [])
    entries_by_id = {get_entry_id(e): e for e in entries if get_entry_id(e)}

    rewrites_path = Path(args.rewrites) if args.rewrites else paths["rewrites"]
    errors_path = Path(args.errors) if args.errors else paths["errors"]
    merged_path = Path(args.merged) if args.merged else paths["merged"]

    rewrites = load_json(rewrites_path, {})
    errors = load_json(errors_path, {})
    manual = normalize_manual(load_json(Path(args.manual), {}))

    applied = 0
    skipped = 0

    for image_key, fix in manual.items():
        entry = entries_by_id.get(image_key)
        if entry is None:
            print(f"[skip] unknown image_key: {image_key}")
            skipped += 1
            continue

        labels = fix.get("labels_rewritten", [])
        ok, hits = validate_labels(entry, labels, strict=strict)
        if not ok and not args.allow_invalid:
            print(f"[skip] invalid labels for {image_key}: {hits}")
            skipped += 1
            continue

        original = [clean_text(x) for x in entry.get("labels", [])]
        rewritten = [clean_text(x) for x in labels]
        changed = [i for i, (a, b) in enumerate(zip(original, rewritten)) if a != b]
        forbidden = build_forbidden_terms(entry, strict=strict)

        rewrites[image_key] = {
            "image_key": image_key,
            "labels_original": original,
            "labels_rewritten": rewritten,
            "changed_indices": changed,
            "notes": clean_text(fix.get("notes", "manual")),
            "forbidden_terms": forbidden["raw_terms"],
            "forbidden_tokens": forbidden["tokens"],
            "flagged_caption_indices": entry.get("flagged_caption_indices", []),
            "effective_flagged_indices": effective_flagged_indices(entry, strict=strict),
            "flagged_spans": entry.get("flagged_spans", []),
            "ocr": entry.get("ocr", []),
            "manual": True,
            "validator_hits_allowed": hits if not ok else {},
        }
        errors.pop(image_key, None)
        applied += 1

    save_json(rewrites_path, rewrites)
    save_json(errors_path, errors)

    source_json = Path(dump.get("source_json", ""))
    if source_json.exists():
        source = load_json(source_json, {})
        merged = dict(source)
        for image_key, record in rewrites.items():
            if image_key in merged and record.get("labels_rewritten"):
                merged[image_key] = record["labels_rewritten"]
        save_json(merged_path, merged)
        print(f"[done] wrote merged labels: {merged_path}")
    else:
        print(f"[warn] source_json not found: {source_json}")

    print(f"[done] applied={applied} skipped={skipped}")
    print(f"[done] wrote rewrites: {rewrites_path}")
    print(f"[done] wrote errors: {errors_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract/merge CLIP OCR rewrite leftovers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("dump_json")
    p.add_argument("--rewrites", default=None)
    p.add_argument("--errors", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--errors-only", action="store_true")
    p.add_argument("--literal-validator", action="store_true", help="Use exact-token/multi-token phrase validation instead of strict substring validation.")
    p.set_defaults(func=extract_leftovers)

    p = sub.add_parser("merge")
    p.add_argument("dump_json")
    p.add_argument("--manual", required=True)
    p.add_argument("--rewrites", default=None)
    p.add_argument("--errors", default=None)
    p.add_argument("--merged", default=None)
    p.add_argument("--allow-invalid", action="store_true")
    p.add_argument("--literal-validator", action="store_true", help="Use exact-token/multi-token phrase validation instead of strict substring validation.")
    p.set_defaults(func=merge_manual)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
