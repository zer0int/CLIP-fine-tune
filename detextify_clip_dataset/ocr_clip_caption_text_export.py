#!/usr/bin/env python3
"""
ocr_clip_caption_text_export.py

-> Example code as used with utils_datasets/coco_spright labels .json

First-stage CLIP caption text-leak detector.

Given a JSON mapping image relative paths to caption lists, this script:
  1. Runs PaddleOCR on each image at 0/90/180/270 degrees.
  2. Maps OCR boxes back to original image coordinates.
  3. Finds exact / substring / fuzzy / compound overlaps between OCR text and
     the dataset captions.
  4. Exports only entries where at least one caption contains OCR-matched words.
  5. Dumps up to N debug images with red OCR boxes and a text panel containing
     captions + match metadata.

Output defaults:
  {input_stem}_dump_gpt_oss.json
  {input_stem}_debug_images/*.png
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from paddleocr import PaddleOCR
except Exception as exc:
    raise SystemExit(
        "PaddleOCR import failed. Install/use your PaddleOCR environment first.\n"
        "Example: pip install paddleocr\n"
        f"Original error: {exc}"
    ) from exc


try:
    from tqdm import tqdm
except Exception:
    tqdm = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Conservative: content words like floor/stop/sign/watch still pass.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "there", "their", "his", "her", "they",
    "them", "you", "your", "we", "our", "he", "she", "i", "me", "my",
    "und", "oder", "der", "die", "das", "ein", "eine", "einer", "eines",
    "im", "am", "an", "auf", "mit", "für", "von", "zu", "den", "dem", "des",
    "ist", "sind",
}

TOKEN_RE = re.compile(r"[\wÄÖÜäöüß]+(?:[-'’][\wÄÖÜäöüß]+)*", re.UNICODE)


@dataclass
class CaptionToken:
    text: str
    norm: str
    start: int
    end: int


@dataclass
class OCRItem:
    text: str
    norm_text: str
    conf: float
    rotation: int
    box: list[list[float]]
    bbox: list[float]


@dataclass
class MatchItem:
    caption_index: int
    span_text: str
    span_norm: str
    start: int
    end: int
    match_type: str
    ocr_text: str
    ocr_norm: str
    ocr_conf: float
    ocr_rotation: int
    ocr_box: list[list[float]]


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ß", "ss")
    return re.sub(r"[^0-9a-zäöü]+", "", text)


def normalize_phrase(text: str) -> str:
    toks = [normalize_token(t.group(0)) for t in TOKEN_RE.finditer(text)]
    return " ".join(t for t in toks if t)


def is_content_norm(norm: str, min_len: int = 3) -> bool:
    if len(norm) < min_len:
        return False
    if norm in STOPWORDS:
        return False
    if norm.isdigit() and len(norm) < 3:
        return False
    return True


def content_tokens_from_text(text: str, min_len: int) -> list[str]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        n = normalize_token(m.group(0))
        if is_content_norm(n, min_len=min_len):
            toks.append(n)
    return toks


def tokenize_caption(caption: str, min_len: int) -> list[CaptionToken]:
    tokens = []
    for match in TOKEN_RE.finditer(caption):
        text = match.group(0)
        norm = normalize_token(text)
        if is_content_norm(norm, min_len=min_len):
            tokens.append(CaptionToken(text=text, norm=norm, start=match.start(), end=match.end()))
    return tokens


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def bbox_from_box(box: list[list[float]]) -> list[float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def rotate_image(image: Image.Image, rotation: int) -> Image.Image:
    return image if rotation == 0 else image.rotate(rotation, expand=True)


def inverse_rotate_point(x: float, y: float, original_w: int, original_h: int, rotation: int) -> tuple[float, float]:
    # rotation is PIL CCW degrees applied before OCR.
    if rotation == 0:
        return x, y
    if rotation == 90:
        return original_w - y, x
    if rotation == 180:
        return original_w - x, original_h - y
    if rotation == 270:
        return y, original_h - x
    raise ValueError(f"Unsupported rotation: {rotation}")


def inverse_rotate_box(box: list[list[float]], original_w: int, original_h: int, rotation: int) -> list[list[float]]:
    return [[float(a), float(b)] for a, b in (inverse_rotate_point(float(x), float(y), original_w, original_h, rotation) for x, y in box)]


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not-installed"


def make_ocr_engine(args: argparse.Namespace) -> PaddleOCR:
    """
    PaddleOCR has incompatible 2.x/3.x constructor APIs. Try 3.x first, then
    legacy 2.x. If all constructors fail with set_optimization_level, the
    installed paddlepaddle runtime is too old for PaddleOCR/PaddleX 3.x.
    """
    print(
        "[paddle versions] "
        f"paddleocr={package_version('paddleocr')} "
        f"paddlex={package_version('paddlex')} "
        f"paddlepaddle={package_version('paddlepaddle')} "
        f"paddlepaddle-gpu={package_version('paddlepaddle-gpu')}"
    )

    kwargs_candidates = []

    if args.paddle_api in {"auto", "v3"}:
        kwargs_candidates.extend([
            dict(
                lang=args.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            ),
            dict(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            ),
            dict(lang=args.lang),
            dict(),
        ])

    if args.paddle_api in {"auto", "v2"}:
        kwargs_candidates.extend([
            dict(lang=args.lang, use_angle_cls=True, show_log=False),
            dict(lang=args.lang, use_angle_cls=True),
            dict(lang=args.lang),
        ])

    errors = []
    for kwargs in kwargs_candidates:
        try:
            print(f"[paddle init] trying PaddleOCR({kwargs})")
            return PaddleOCR(**kwargs)
        except Exception as exc:
            errors.append((kwargs, repr(exc)))

    msg = "\n".join(f"  kwargs={kw} -> {err}" for kw, err in errors[-8:])
    if any("set_optimization_level" in err for _, err in errors):
        raise RuntimeError(
            "Could not initialize PaddleOCR because PaddleOCR/PaddleX is calling "
            "AnalysisConfig.set_optimization_level, but your installed PaddlePaddle "
            "runtime does not provide that method. This is a version mismatch.\n\n"
            "Check:\n"
            "  python -m pip show paddlepaddle paddlepaddle-gpu paddleocr paddlex\n\n"
            "CPU sanity-fix option:\n"
            "  python -m pip uninstall -y paddlepaddle paddlepaddle-gpu paddleocr paddlex\n"
            "  python -m pip install paddlepaddle==3.1.1 paddleocr==3.5.0 paddlex==3.5.2\n\n"
            "GPU option: install the PaddlePaddle build matching your CUDA from the official "
            "Paddle install selector, then install paddleocr==3.5.0 paddlex==3.5.2.\n\n"
            f"Constructor attempts:\n{msg}"
        )

    raise RuntimeError(f"Could not initialize PaddleOCR. Constructor attempts:\n{msg}")


def as_plain_result(obj: Any) -> Any:
    """Convert PaddleOCR 3.x OCRResult-ish objects into dict/list when possible."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple, dict, str, int, float)):
        return obj
    for attr in ("json", "res", "data"):
        try:
            val = getattr(obj, attr)
            if callable(val):
                val = val()
            if val is not None:
                return val
        except Exception:
            pass
    try:
        return dict(obj)
    except Exception:
        return obj


def parse_paddle_result(result: Any) -> list[tuple[list[list[float]], str, float]]:
    """
    Supports:
      - PaddleOCR 2.x: [[box, (text, conf)], ...] or wrapped once.
      - PaddleOCR 3.x: OCRResult objects / dicts with rec_texts, rec_scores,
        rec_polys / dt_polys / rec_boxes, or nested JSON-like structures.
    """
    parsed: list[tuple[list[list[float]], str, float]] = []

    if result is None:
        return parsed

    result = as_plain_result(result)

    # PaddleOCR 3.x predict() often returns a list of per-image result objects.
    if isinstance(result, list) and result:
        # Avoid treating legacy [box, (text, conf)] as a list of result objects.
        is_legacy_det = (
            len(result) == 2
            and isinstance(result[1], (list, tuple))
            and len(result[1]) >= 2
            and isinstance(result[1][0], str)
        )
        if not is_legacy_det:
            for sub in result:
                sub_plain = as_plain_result(sub)
                # Legacy detections will be handled below if recursive parse fails.
                sub_parsed = parse_paddle_result(sub_plain) if sub_plain is not sub or isinstance(sub_plain, dict) else []
                parsed.extend(sub_parsed)
            if parsed:
                return parsed

    # Common wrapper for one image in legacy API.
    if (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], list)
        and (not result[0] or isinstance(result[0][0], (list, tuple, dict)))
    ):
        first = result[0]
        if not (len(first) == 2 and isinstance(first[1], (tuple, list)) and first[1] and isinstance(first[1][0], str)):
            result = first

    if isinstance(result, dict):
        if "res" in result and isinstance(result["res"], dict):
            result = result["res"]

        rec_texts = result.get("rec_texts") or result.get("texts") or result.get("text")
        rec_scores = result.get("rec_scores") or result.get("scores") or result.get("confidence")
        polys = (
            result.get("rec_polys")
            or result.get("dt_polys")
            or result.get("polys")
            or result.get("boxes")
            or result.get("rec_boxes")
            or result.get("dt_boxes")
        )

        if isinstance(rec_texts, str):
            rec_texts = [rec_texts]
        if isinstance(rec_scores, (int, float)):
            rec_scores = [float(rec_scores)]

        if rec_texts is not None and polys is not None:
            for i, text in enumerate(rec_texts):
                conf = 1.0
                if isinstance(rec_scores, (list, tuple)) and i < len(rec_scores):
                    conf = float(rec_scores[i])

                box = polys[i] if isinstance(polys, (list, tuple)) and i < len(polys) else None
                if box is None:
                    continue

                box_arr = np.asarray(box, dtype=float)
                if box_arr.ndim == 1 and box_arr.size == 4:
                    x1, y1, x2, y2 = box_arr.tolist()
                    box_arr = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
                if box_arr.ndim == 2 and box_arr.shape[1] == 2:
                    parsed.append((box_arr.tolist(), str(text), conf))
            if parsed:
                return parsed

        for val in result.values():
            parsed.extend(parse_paddle_result(val))
        if parsed:
            return parsed

    if isinstance(result, list):
        for item in result:
            item = as_plain_result(item)
            if not item:
                continue

            # Old format: [box, (text, conf)]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                box = item[0]
                rec = item[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2 and isinstance(rec[0], str):
                    text = rec[0]
                    conf = float(rec[1])
                    parsed.append(([[float(x), float(y)] for x, y in box], text, conf))
                    continue

            if isinstance(item, dict):
                parsed.extend(parse_paddle_result(item))
                continue

    return parsed


def run_ocr_on_image(ocr: PaddleOCR, image: Image.Image, args: argparse.Namespace) -> list[OCRItem]:
    original_w, original_h = image.size
    candidates = []

    for rotation in args.rotations:
        rot_img = rotate_image(image, rotation).convert("RGB")
        arr = np.array(rot_img)

        result = None

        if args.paddle_api in {"auto", "v3"} and hasattr(ocr, "predict"):
            try:
                result = ocr.predict(input=arr)
            except Exception:
                try:
                    result = ocr.predict(arr)
                except Exception:
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp_path = Path(tmp.name)
                        rot_img.save(tmp_path)
                        result = ocr.predict(input=str(tmp_path))
                    finally:
                        if tmp_path is not None:
                            try:
                                tmp_path.unlink(missing_ok=True)
                            except Exception:
                                pass

        if result is None and args.paddle_api in {"auto", "v2"} and hasattr(ocr, "ocr"):
            try:
                result = ocr.ocr(arr, cls=True)
            except TypeError:
                result = ocr.ocr(arr)

        for box_rot, text, conf in parse_paddle_result(result):
            text = clean_text(text)
            norm_text = normalize_phrase(text)
            if not text or not norm_text or conf < args.min_conf:
                continue
            box_orig = inverse_rotate_box(box_rot, original_w, original_h, rotation)
            bbox = bbox_from_box(box_orig)
            candidates.append(OCRItem(text=text, norm_text=norm_text, conf=float(conf), rotation=rotation, box=box_orig, bbox=bbox))

    return dedupe_ocr_items(candidates, iou_threshold=args.dedupe_iou)


def dedupe_ocr_items(items: list[OCRItem], iou_threshold: float) -> list[OCRItem]:
    items = sorted(items, key=lambda x: x.conf, reverse=True)
    kept = []
    for item in items:
        duplicate = any(item.norm_text == prev.norm_text and bbox_iou(item.bbox, prev.bbox) >= iou_threshold for prev in kept)
        if not duplicate:
            kept.append(item)
    kept.sort(key=lambda x: (x.bbox[1], x.bbox[0], -x.conf))
    return kept


def single_token_match_type(caption_norm: str, ocr_norm: str, args: argparse.Namespace) -> str | None:
    if caption_norm == ocr_norm:
        return "exact"
    if len(caption_norm) >= args.min_substring_len and len(ocr_norm) >= args.min_substring_len:
        if caption_norm in ocr_norm or ocr_norm in caption_norm:
            return "substring"
    if min(len(caption_norm), len(ocr_norm)) >= args.min_fuzzy_len:
        if fuzzy_ratio(caption_norm, ocr_norm) >= args.fuzzy_threshold:
            return "fuzzy"
    return None


def add_match(matches: list[MatchItem], caption_index: int, caption: str, start: int, end: int, match_type: str, ocr: OCRItem, span_norm: str) -> None:
    matches.append(MatchItem(
        caption_index=caption_index,
        span_text=caption[start:end],
        span_norm=span_norm,
        start=int(start),
        end=int(end),
        match_type=match_type,
        ocr_text=ocr.text,
        ocr_norm=ocr.norm_text,
        ocr_conf=ocr.conf,
        ocr_rotation=ocr.rotation,
        ocr_box=ocr.box,
    ))


def find_caption_matches(captions: list[str], ocr_items: list[OCRItem], args: argparse.Namespace) -> list[MatchItem]:
    matches = []

    for caption_index, caption in enumerate(captions):
        tokens = tokenize_caption(caption, min_len=args.min_token_len)
        for token in tokens:
            for ocr in ocr_items:
                for ocr_tok in content_tokens_from_text(ocr.text, min_len=args.min_token_len):
                    match_type = single_token_match_type(token.norm, ocr_tok, args)
                    if match_type:
                        add_match(matches, caption_index, caption, token.start, token.end, match_type, ocr, token.norm)

    for caption_index, caption in enumerate(captions):
        all_tokens = []
        for m in TOKEN_RE.finditer(caption):
            n = normalize_token(m.group(0))
            if n:
                all_tokens.append(CaptionToken(text=m.group(0), norm=n, start=m.start(), end=m.end()))

        for ocr in ocr_items:
            ocr_content = content_tokens_from_text(ocr.text, min_len=args.min_token_len)
            if len(ocr_content) < 2:
                continue
            ocr_join = "".join(ocr_content)
            for ngram_len in range(2, min(args.max_compound_ngram, len(all_tokens)) + 1):
                for i in range(0, len(all_tokens) - ngram_len + 1):
                    ngram = all_tokens[i:i + ngram_len]
                    content = [t.norm for t in ngram if is_content_norm(t.norm, min_len=args.min_token_len)]
                    if len(content) < 2:
                        continue
                    cap_join = "".join(content)
                    match_type = None
                    if cap_join == ocr_join:
                        match_type = "compound_exact"
                    elif len(cap_join) >= args.min_substring_len and len(ocr_join) >= args.min_substring_len and (cap_join in ocr_join or ocr_join in cap_join):
                        match_type = "compound_substring"
                    elif min(len(cap_join), len(ocr_join)) >= args.min_fuzzy_len and fuzzy_ratio(cap_join, ocr_join) >= args.compound_fuzzy_threshold:
                        match_type = "compound_fuzzy"
                    if match_type:
                        add_match(matches, caption_index, caption, ngram[0].start, ngram[-1].end, match_type, ocr, " ".join(content))

    return dedupe_matches(matches)


def match_priority(match_type: str) -> int:
    return {
        "compound_exact": 0,
        "compound_substring": 1,
        "compound_fuzzy": 2,
        "exact": 3,
        "substring": 4,
        "fuzzy": 5,
    }.get(match_type, 99)


def dedupe_matches(matches: list[MatchItem]) -> list[MatchItem]:
    best = {}
    for m in matches:
        key = (m.caption_index, m.start, m.end, m.ocr_norm)
        prev = best.get(key)
        if prev is None or (match_priority(m.match_type), -m.ocr_conf) < (match_priority(prev.match_type), -prev.ocr_conf):
            best[key] = m
    deduped = list(best.values())

    final = []
    for m in sorted(deduped, key=lambda x: (x.caption_index, x.start, x.end, match_priority(x.match_type))):
        if not m.match_type.startswith("compound"):
            if any(
                other is not m and other.match_type.startswith("compound") and other.caption_index == m.caption_index
                and other.ocr_norm == m.ocr_norm and other.start <= m.start and other.end >= m.end
                for other in deduped
            ):
                continue
        final.append(m)
    return sorted(final, key=lambda x: (x.caption_index, x.start, match_priority(x.match_type), -x.ocr_conf))


def match_to_json(m: MatchItem) -> dict[str, Any]:
    return {
        "caption_index": m.caption_index,
        "span_text": m.span_text,
        "span_norm": m.span_norm,
        "start": m.start,
        "end": m.end,
        "match_type": m.match_type,
        "ocr_text": m.ocr_text,
        "ocr_norm": m.ocr_norm,
        "ocr_conf": round(float(m.ocr_conf), 4),
        "ocr_rotation": m.ocr_rotation,
        "ocr_box": [[round(float(x), 2), round(float(y), 2)] for x, y in m.ocr_box],
    }


def ocr_to_json(o: OCRItem) -> dict[str, Any]:
    return {
        "text": o.text,
        "norm_text": o.norm_text,
        "conf": round(float(o.conf), 4),
        "rotation": o.rotation,
        "box": [[round(float(x), 2), round(float(y), 2)] for x, y in o.box],
        "bbox": [round(float(v), 2) for v in o.bbox],
    }


def safe_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("._")
    return text[:max_len] or "image"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = ""
    for word in words:
        trial = (cur + " " + word).strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_debug_image(image: Image.Image, image_key: str, captions: list[str], ocr_items: list[OCRItem], matches: list[MatchItem], out_path: Path) -> None:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, 1400 / max(w, h))
    draw_img = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    font = load_font(16)
    small_font = load_font(13)
    panel_w = draw_img.size[0]
    dummy = Image.new("RGB", (panel_w, 10), "white")
    dummy_draw = ImageDraw.Draw(dummy)

    lines = [(f"IMAGE: {image_key}", font, (0, 0, 0))]
    for idx, caption in enumerate(captions):
        for line in wrap_text(dummy_draw, f"LABEL[{idx}]: {caption}", small_font, panel_w - 20):
            lines.append((line, small_font, (0, 0, 0)))
    if matches:
        summary = []
        for m in matches:
            summary.append(f"L{m.caption_index} '{m.span_text}' ⇐ OCR '{m.ocr_text}' conf={m.ocr_conf:.2f} rot={m.ocr_rotation} type={m.match_type}")
        for line in wrap_text(dummy_draw, "MATCHES: " + " | ".join(summary), small_font, panel_w - 20):
            lines.append((line, small_font, (180, 0, 0)))
    ocr_summary = [f"'{o.text}'({o.conf:.2f},r{o.rotation})" for o in ocr_items]
    for line in wrap_text(dummy_draw, "OCR: " + " | ".join(ocr_summary), small_font, panel_w - 20):
        lines.append((line, small_font, (80, 80, 80)))

    heights = []
    for text, fnt, _ in lines:
        bbox = dummy_draw.textbbox((0, 0), text, font=fnt)
        heights.append(max(18, bbox[3] - bbox[1] + 4))
    panel_h = sum(heights) + 16

    canvas = Image.new("RGB", (panel_w, panel_h + draw_img.size[1]), "white")
    cd = ImageDraw.Draw(canvas)
    y = 8
    for (text, fnt, color), lh in zip(lines, heights):
        cd.text((10, y), text, fill=color, font=fnt)
        y += lh
    canvas.paste(draw_img, (0, panel_h))
    cd = ImageDraw.Draw(canvas)

    for ocr in ocr_items:
        pts = [(x * scale, y * scale + panel_h) for x, y in ocr.box]
        if len(pts) >= 2:
            cd.line(pts + [pts[0]], fill=(255, 0, 0), width=1)
            cd.text((pts[0][0] + 2, pts[0][1] + 2), f"{ocr.conf:.2f}", fill=(255, 0, 0), font=small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def resolve_image_path(json_path: Path, image_key: str) -> Path:
    image_path = Path(image_key)
    return image_path if image_path.is_absolute() else (json_path.parent / image_path).resolve()


def process_dataset(args: argparse.Namespace) -> None:
    json_path = Path(args.path_to_json).resolve()
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected input JSON to be a dict mapping image paths to caption lists.")

    output_json = Path(args.output) if args.output else json_path.with_name(f"{json_path.stem}_dump_gpt_oss.json")
    debug_dir = Path(args.debug_dir) if args.debug_dir else json_path.with_name(f"{json_path.stem}_debug_images")
    debug_dir.mkdir(parents=True, exist_ok=True)

    rotations = []
    for r in args.rotations:
        r = int(r) % 360
        if r not in {0, 90, 180, 270}:
            raise ValueError("--rotations supports only 0 90 180 270")
        rotations.append(r)
    args.rotations = rotations

    print(f"[load] PaddleOCR lang={args.lang} rotations={args.rotations}")
    ocr_engine = make_ocr_engine(args)

    export_entries = []
    errors = []
    debug_written = 0
    items = list(data.items())
    if args.limit is not None:
        items = items[:args.limit]

    iterator = enumerate(items, start=1)
    if tqdm is not None and not args.no_tqdm:
        iterator = tqdm(
            iterator,
            total=len(items),
            desc="OCR+match",
            unit="img",
            dynamic_ncols=True,
            smoothing=0.05,
        )

    for idx, (image_key, captions_raw) in iterator:
        captions = [clean_text(c) for c in (captions_raw or []) if clean_text(c)]
        if not captions:
            continue
        image_path = resolve_image_path(json_path, image_key)
        if not image_path.exists():
            errors.append({"image_key": image_key, "error": f"image not found: {image_path}"})
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            errors.append({"image_key": image_key, "error": f"not an image extension: {image_path.suffix}"})
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            ocr_items = run_ocr_on_image(ocr_engine, image, args)
            if not ocr_items:
                continue
            matches = find_caption_matches(captions, ocr_items, args)
            if not matches:
                continue

            flagged_caption_indices = sorted({m.caption_index for m in matches})
            forbidden_caption_spans = sorted({m.span_text for m in matches}, key=lambda s: (s.lower(), s))
            forbidden_ocr_text = sorted({o.text for o in ocr_items}, key=lambda s: (s.lower(), s))

            entry = {
                "image_key": image_key,
                "image_path": str(image_path),
                "labels": captions,
                "flagged_caption_indices": flagged_caption_indices,
                "forbidden_caption_spans": forbidden_caption_spans,
                "forbidden_ocr_text": forbidden_ocr_text,
                "flagged_spans": [match_to_json(m) for m in matches],
                "ocr": [ocr_to_json(o) for o in ocr_items],
                "gpt_oss_instruction_payload": {
                    "task": (
                        "Rewrite only the labels whose indices are in flagged_caption_indices. "
                        "Remove/paraphrase all flagged caption spans and avoid including any visible OCR text. "
                        "Never include exact written words, brands, logos, quoted text, or OCR strings. "
                        "Preserve visual object/scene meaning, spatial relations, and caption style. "
                        "Keep captions concise enough for CLIP tokenization."
                    ),
                    "must_not_include": sorted(set(forbidden_caption_spans + forbidden_ocr_text), key=lambda s: (s.lower(), s)),
                },
            }
            export_entries.append(entry)

            if debug_written < args.max_debug_images:
                base = safe_filename(Path(image_key).stem)
                draw_debug_image(image, image_key, captions, ocr_items, matches, debug_dir / f"{len(export_entries):05d}_{base}.png")
                debug_written += 1
            if len(export_entries) % args.log_every == 0:
                msg = f"[progress] scanned={idx}/{len(items)} exported={len(export_entries)} debug={debug_written}"
                if tqdm is not None and not args.no_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)
        except Exception as exc:
            errors.append({"image_key": image_key, "error": repr(exc)})
            continue

    output = {
        "source_json": str(json_path),
        "num_input_items": len(data),
        "num_scanned_items": len(items),
        "num_exported_items": len(export_entries),
        "ocr_params": {
            "lang": args.lang,
            "rotations": args.rotations,
            "min_conf": args.min_conf,
            "min_token_len": args.min_token_len,
            "min_substring_len": args.min_substring_len,
            "min_fuzzy_len": args.min_fuzzy_len,
            "fuzzy_threshold": args.fuzzy_threshold,
            "compound_fuzzy_threshold": args.compound_fuzzy_threshold,
            "dedupe_iou": args.dedupe_iou,
        },
        "entries": export_entries,
        "errors": errors,
    }
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        err_path = output_json.with_name(output_json.stem + "_errors.json")
        err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[warn] wrote {err_path} with {len(errors)} errors")
    print(f"[done] wrote {output_json} with {len(export_entries)} entries")
    print(f"[done] debug images: {debug_dir} ({debug_written})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR + caption text-leak export for GPT-OSS rewriting.")
    parser.add_argument("path_to_json", help="Input JSON mapping image relative path -> caption list.")
    parser.add_argument("--output", default=None, help="Output JSON. Default: {input_stem}_dump_gpt_oss.json")
    parser.add_argument("--debug-dir", default=None, help="Debug image dir. Default: {input_stem}_debug_images")
    parser.add_argument("--max-debug-images", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N JSON entries.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--lang", default="en", help="PaddleOCR language, e.g. en, german, ch.")
    parser.add_argument("--paddle-api", choices=["auto", "v3", "v2"], default="auto", help="PaddleOCR API style to try.")
    parser.add_argument("--rotations", type=int, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--min-conf", type=float, default=0.50)
    parser.add_argument("--dedupe-iou", type=float, default=0.50)
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--min-substring-len", type=int, default=3)
    parser.add_argument("--min-fuzzy-len", type=int, default=5)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.86)
    parser.add_argument("--compound-fuzzy-threshold", type=float, default=0.90)
    parser.add_argument("--max-compound-ngram", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    process_dataset(parse_args())


if __name__ == "__main__":
    main()