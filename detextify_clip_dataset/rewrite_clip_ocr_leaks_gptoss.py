#!/usr/bin/env python3
"""
rewrite_clip_ocr_leaks_gptoss.py

Second-stage GPT-OSS caption rewrite for CLIP OCR/text-leak cleanup.

Input:
  *_dump_gpt_oss.json from ocr_clip_caption_text_export.py

Outputs next to input:
  {input_stem}_gptoss_rewrites.json       per-image validated rewrite records
  {input_stem}_gptoss_finals.json         raw final-channel strings
  {input_stem}_gptoss_errors.json         failed items, rerun retries them
  {input_stem}_rewritten_labels.json      full original labels JSON with rewrites merged
  {input_stem}_harmony_traces/*.txt       optional raw Harmony streams

Rerun behavior:
  Existing valid rewrites are skipped unless --force is passed.
  If a model gets stuck in analysis / CoT and never emits valid final JSON,
  the item is logged as an error and remains flagged for scheduling on rerun.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import re
import sys
from pathlib import Path
from threading import Thread
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, TextIteratorStreamer

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
except Exception:
    class _Dummy:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
        RESET_ALL = BRIGHT = DIM = NORMAL = ""
    Fore = _Dummy()
    Style = _Dummy()


DEFAULT_MODEL_NAME = "openai/gpt-oss-20b" # or local copy
DEFAULT_MAX_NEW_TOKENS = 1152
TRACE_DIR_SUFFIX = "_harmony_traces"

os.environ["CL"] = "/nologo"

TOKEN_RE = re.compile(r"[\wÄÖÜäöüß]+(?:[-'’][\wÄÖÜäöüß]+)*", re.UNICODE)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "there", "their", "his", "her", "they",
    "them", "you", "your", "we", "our", "he", "she", "i", "me", "my",
    "into", "over", "under", "above", "below", "near", "next", "front",
    "back", "left", "right", "large", "small", "white", "black", "red",
    "blue", "green", "yellow", "orange", "brown", "gray", "grey",
    "und", "oder", "der", "die", "das", "ein", "eine", "einer", "eines",
    "im", "am", "an", "auf", "mit", "für", "von", "zu", "den", "dem",
    "des", "ist", "sind",
}


@contextlib.contextmanager
def suppress_native_output(enabled: bool = True):
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_token(text: str) -> str:
    text = str(text or "").lower().replace("ß", "ss")
    return re.sub(r"[^0-9a-zäöü]+", "", text)


def norm_tokens(text: str, min_len: int = 3) -> list[str]:
    out = []
    for m in TOKEN_RE.finditer(str(text or "")):
        tok = normalize_token(m.group(0))
        if len(tok) >= min_len and tok not in STOPWORDS:
            out.append(tok)
    return out


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


def cleanup_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()


def get_entry_id(entry: dict[str, Any]) -> str:
    return str(entry.get("image_key") or entry.get("image_path") or "")


def safe_trace_name(entry_id: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", entry_id).strip("._")
    return (name[:180] or "entry") + ".txt"


def build_forbidden_terms(entry: dict[str, Any]) -> dict[str, Any]:
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
        if not term:
            continue
        toks = norm_tokens(term)
        if toks:
            phrases.append(term)
            for tok in toks:
                tokens.add(tok)

            if len(toks) >= 2:
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


def get_effective_flagged_indices(entry: dict[str, Any]) -> list[int]:
    """Union OCR-stage flagged indices with any label that still contains forbidden tokens."""
    labels = [clean_text(x) for x in entry.get("labels", [])]
    forbidden = build_forbidden_terms(entry)
    flagged = set(int(x) for x in (entry.get("flagged_caption_indices") or []))
    for i, label in enumerate(labels):
        if contains_forbidden(label, forbidden):
            flagged.add(i)
    return sorted(i for i in flagged if 0 <= i < len(labels))


def build_system_prompt(reasoning: str) -> str:
    return f"""Reasoning: {reasoning}

# Valid channels: analysis, final. Channel must be included for every message.

You are a strict caption detox rewriter for CLIP fine-tuning.

Goal:
Rewrite image captions so they do NOT reward CLIP for reading visible glyphs,
brands, logos, watermarks, signs, shirt text, labels, or OCR-detected words.

Core rules:
- You receive an image key, labels, flagged label indices, OCR text, and flagged spans.
- Return the full list of labels, same length and same order.
- Rewrite flagged labels. Unflagged labels should normally be copied exactly.
- NEVER include any forbidden word, OCR word, brand name, logo text, quoted text, visible sign text, or obvious inflection/compound of such words.
- This ban still applies if the forbidden word is a correct object name.
- If OCR sees "CLOCK", do not write "clock"; use "timepiece" or another visual description.
- If OCR sees "PLAY", do not write "play" or "playing"; use "having fun", "interacting", or another description.
- If OCR sees "FLOOR", do not write "floor"; use "ground", "surface", or another description.
- If OCR sees a brand/company name, use generic visual wording: "advertising board", "soda can", "delivery truck", "mobile device", etc.
- Preserve visual object/scene meaning, spatial relations, colors, actions, and style as much as possible.
- Prefer concise, CLIP-useful captions. Do not become excessively elaborate.
- If several rewrites are possible, choose the simplest good one and stop.

Examples:
A:
Original: "a couple of kids that are playing on the ground"
Forbidden: PLAY, playing
Rewrite: "a couple of kids having fun on the ground"

B:
Original: "This photograph appears to be looking truly wonderful."
Forbidden: photography, photograph
Rewrite: "A close-up view shows a white computer mouse resting on a pale keyboard."

C:
Original: "The player is in front of a large advertisement for Sony Ericsson, which is on the wall behind him."
Forbidden: Sony, Ericsson
Rewrite: "The player is in front of a large green advertising board on the wall behind him."

D:
Original: "A grandfather clock is in the middle of the sidewalk."
Forbidden: STEAM CLOCK, clock
Rewrite: "A tall ornate public timepiece stands in the middle of the sidewalk."

E:
Original: "a stop sign that says hammer time underneath stop"
Forbidden: STOP, hammer time
Rewrite: "a red octagonal traffic marker with extra writing underneath"

F:
Original: "The boy is wearing a shirt that says I Portland."
Forbidden: Portland
Rewrite: "The boy is wearing a shirt that references a city in Oregon."

Use the analysis channel briefly.
In the final channel, return exactly this JSON object:
{{
  "image_key": "...",
  "labels": ["full rewritten label 0", "full rewritten label 1"],
  "changed_indices": [0],
  "notes": "very brief"
}}

The final channel must start with {{ and end with }}.
No Markdown.
No extra text in final."""


def build_messages(entry: dict[str, Any], reasoning: str, max_label_chars: int) -> list[dict[str, str]]:
    image_key = entry.get("image_key", "")
    labels = [clean_text(x) for x in entry.get("labels", [])]
    trimmed_labels = []
    for x in labels:
        if len(x) > max_label_chars:
            x = x[:max_label_chars].rsplit(" ", 1)[0] + " ..."
        trimmed_labels.append(x)

    flagged_indices = get_effective_flagged_indices(entry)
    forbidden = build_forbidden_terms(entry)

    flagged_spans = []
    for m in entry.get("flagged_spans") or []:
        flagged_spans.append({
            "caption_index": m.get("caption_index"),
            "span_text": m.get("span_text"),
            "match_type": m.get("match_type"),
            "ocr_text": m.get("ocr_text"),
            "ocr_conf": m.get("ocr_conf"),
        })

    ocr_text = []
    for o in entry.get("ocr") or []:
        txt = clean_text(o.get("text"))
        if txt:
            ocr_text.append({"text": txt, "conf": o.get("conf"), "rotation": o.get("rotation")})

    user_payload = {
        "image_key": image_key,
        "labels": trimmed_labels,
        "flagged_caption_indices": flagged_indices,
        "forbidden_terms": forbidden["raw_terms"],
        "forbidden_tokens": forbidden["tokens"],
        "ocr_text": ocr_text,
        "flagged_spans": flagged_spans,
        "instruction": (
            "Rewrite every label index listed in flagged_caption_indices. "
            "These indices already include any label that contains forbidden text. "
            "Copy all other labels exactly. "
            "Return full labels list, same length and order."
        ),
    }

    return [
        {"role": "system", "content": build_system_prompt(reasoning)},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def render_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System:\n{messages[0]['content']}\n\nUser:\n{messages[1]['content']}\n\nAssistant:\n"


def parse_harmony_channels(raw_text: str) -> dict[str, str]:
    channels: dict[str, str] = {}
    pattern = re.compile(
        r"<\|start\|>assistant"
        r"(?:<\|channel\|>(?P<channel>analysis|commentary|final))?"
        r"<\|message\|>(?P<message>.*?)"
        r"<\|end\|>",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(raw_text):
        channel = match.group("channel") or "final"
        message = match.group("message").strip()
        channels[channel] = (channels.get(channel, "") + "\n" + message).strip()
    if channels:
        return channels

    final_marker = "<|channel|>final<|message|>"
    analysis_marker = "<|channel|>analysis<|message|>"
    if final_marker in raw_text:
        before, final = raw_text.split(final_marker, 1)
        channels["final"] = final.replace("<|end|>", "").replace("<|return|>", "").strip()
        if analysis_marker in before:
            channels["analysis"] = before.split(analysis_marker, 1)[1].replace("<|end|>", "").strip()
        return channels

    channels["raw"] = raw_text.strip()
    return channels


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip().replace("<|return|>", "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No '{{' found in final output: {text!r}")
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed JSON is not an object: {parsed!r}")
    return parsed


def validate_rewrite(entry: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    image_key = str(entry.get("image_key", ""))
    original_labels = [clean_text(x) for x in entry.get("labels", [])]
    flagged_indices = set(get_effective_flagged_indices(entry))

    if str(parsed.get("image_key", "")) != image_key:
        raise ValueError(f"image_key mismatch: got {parsed.get('image_key')!r}, expected {image_key!r}")

    labels = parsed.get("labels")
    if not isinstance(labels, list) or len(labels) != len(original_labels):
        raise ValueError(f"Expected labels list of length {len(original_labels)}, got {labels!r}")

    labels = [clean_text(x) for x in labels]
    if any(not x for x in labels):
        raise ValueError(f"Empty rewritten label in {labels!r}")

    changed_indices = parsed.get("changed_indices", [])
    if not isinstance(changed_indices, list):
        raise ValueError("changed_indices must be a list")
    changed_indices = sorted(set(int(x) for x in changed_indices))

    actual_changed = [i for i, (a, b) in enumerate(zip(original_labels, labels)) if a != b]
    if sorted(actual_changed) != changed_indices:
        changed_indices = actual_changed

    forbidden = build_forbidden_terms(entry)

    for i, (old, new) in enumerate(zip(original_labels, labels)):
        if i not in flagged_indices and old != new:
            old_hits = contains_forbidden(old, forbidden)
            if not old_hits:
                raise ValueError(f"Unflagged label {i} changed without forbidden hit. old={old!r} new={new!r}")

    forbidden_hits_by_label = {}
    for i, label in enumerate(labels):
        hits = contains_forbidden(label, forbidden)
        if hits:
            forbidden_hits_by_label[str(i)] = hits
    if forbidden_hits_by_label:
        raise ValueError(f"Forbidden terms still present after rewrite: {forbidden_hits_by_label}")

    return {
        "image_key": image_key,
        "labels_original": original_labels,
        "labels_rewritten": labels,
        "changed_indices": changed_indices,
        "notes": clean_text(parsed.get("notes", ""))[:300],
        "forbidden_terms": forbidden["raw_terms"],
        "forbidden_tokens": forbidden["tokens"],
        "flagged_caption_indices": sorted(flagged_indices),
        "flagged_spans": entry.get("flagged_spans") or [],
        "ocr": entry.get("ocr") or [],
    }


class HarmonyStreamPrinter:
    SPECIAL_RE = re.compile(
        r"(<\|start\|>assistant|<\|channel\|>analysis|<\|channel\|>final|"
        r"<\|channel\|>commentary|<\|message\|>|<\|end\|>|<\|return\|>)"
    )
    MAX_MARKER_LEN = 32

    def __init__(self) -> None:
        self.buffer = ""
        self.channel = "raw"

    def color_for_channel(self) -> str:
        if self.channel == "analysis":
            return Fore.CYAN + Style.DIM
        if self.channel == "final":
            return Fore.GREEN + Style.BRIGHT
        if self.channel == "commentary":
            return Fore.MAGENTA
        return Style.DIM

    def label_for_channel(self, channel: str) -> str:
        if channel == "analysis":
            return Fore.CYAN + Style.BRIGHT + "\n\n[analysis]\n" + Style.RESET_ALL
        if channel == "final":
            return Fore.GREEN + Style.BRIGHT + "\n\n[final]\n" + Style.RESET_ALL
        if channel == "commentary":
            return Fore.MAGENTA + Style.BRIGHT + "\n\n[commentary]\n" + Style.RESET_ALL
        return ""

    def emit_text(self, text: str) -> None:
        if text:
            print(self.color_for_channel() + text + Style.RESET_ALL, end="", flush=True)

    def handle_marker(self, marker: str) -> None:
        if marker == "<|channel|>analysis":
            self.channel = "analysis"
            print(self.label_for_channel("analysis"), end="", flush=True)
        elif marker == "<|channel|>final":
            self.channel = "final"
            print(self.label_for_channel("final"), end="", flush=True)
        elif marker == "<|channel|>commentary":
            self.channel = "commentary"
            print(self.label_for_channel("commentary"), end="", flush=True)
        elif marker == "<|end|>":
            print(Style.DIM + "\n[end]\n" + Style.RESET_ALL, end="", flush=True)
            self.channel = "raw"
        elif marker == "<|return|>":
            print(Style.DIM + "<|return|>" + Style.RESET_ALL, end="", flush=True)
        elif marker in {"<|start|>assistant", "<|message|>"}:
            return
        else:
            self.emit_text(marker)

    def feed(self, chunk: str, flush: bool = False) -> None:
        self.buffer += chunk
        while True:
            search_region = self.buffer if flush else self.buffer[:-self.MAX_MARKER_LEN]
            if not search_region:
                break
            match = self.SPECIAL_RE.search(search_region)
            if not match:
                self.emit_text(search_region)
                self.buffer = self.buffer[len(search_region):]
                break
            if match.start() > 0:
                self.emit_text(self.buffer[:match.start()])
            marker = match.group(1)
            self.handle_marker(marker)
            self.buffer = self.buffer[match.end():]
        if flush and self.buffer:
            self.emit_text(self.buffer)
            self.buffer = ""


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def rewrite_entry(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    entry: dict[str, Any],
    reasoning: str,
    max_new_tokens: int,
    repetition_penalty: float,
    max_label_chars: int,
    use_kv_cache: bool,
    stream: bool = True,
) -> tuple[dict[str, Any], str, str, dict[str, str]]:
    messages = build_messages(entry, reasoning=reasoning, max_label_chars=max_label_chars)
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = get_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=use_kv_cache,
        return_dict_in_generate=False,
    )

    raw_text = ""
    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        generation_kwargs["streamer"] = streamer
        printer = HarmonyStreamPrinter()
        thread_error: list[BaseException] = []

        def _generate_worker() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**generation_kwargs)
            except BaseException as exc:
                thread_error.append(exc)
                raise

        thread = Thread(target=_generate_worker)
        thread.start()
        for piece in streamer:
            raw_text += piece
            printer.feed(piece, flush=False)
        thread.join()
        if thread_error:
            raise thread_error[0]
        printer.feed("", flush=True)
        print()
    else:
        with torch.inference_mode():
            output = model.generate(**generation_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    channels = parse_harmony_channels(raw_text)
    final_text = channels.get("final", channels.get("raw", raw_text)).strip()
    parsed = extract_json_object(final_text)
    record = validate_rewrite(entry, parsed)

    # drop generation objects and release cached allocator blocks.
    del inputs
    del generation_kwargs
    if "streamer" in locals():
        del streamer
    cleanup_torch_memory()

    return record, final_text, raw_text, channels


def merge_rewrites_into_source(dump: dict[str, Any], rewrites_by_id: dict[str, dict[str, Any]], output_path: Path) -> None:
    source_json = Path(dump.get("source_json", ""))
    if not source_json.exists():
        print(Fore.YELLOW + f"[warn] source_json not found; not writing merged labels: {source_json}" + Style.RESET_ALL)
        return
    source = load_json(source_json, {})
    if not isinstance(source, dict):
        print(Fore.YELLOW + f"[warn] source_json is not a dict; not writing merged labels: {source_json}" + Style.RESET_ALL)
        return
    merged = dict(source)
    for image_key, record in rewrites_by_id.items():
        if image_key in merged:
            merged[image_key] = record["labels_rewritten"]
    save_json(output_path, merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite CLIP OCR/text-leak captions with local GPT-OSS.")
    parser.add_argument("dump_json", help="Input *_dump_gpt_oss.json from OCR stage.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model path/name.")
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--max-label-chars", type=int, default=1800)
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable generation KV cache.")
    parser.add_argument("--cleanup-every", type=int, default=1, help="Run gc + torch.cuda.empty_cache every N processed entries.")
    parser.add_argument("--merge-every", type=int, default=25, help="Write merged full labels JSON every N successful rewrites, plus once at end. Rewrites/errors are still saved every item.")
    parser.add_argument("--limit", type=int, default=None, help="Rewrite only N remaining entries.")
    parser.add_argument("--start-index", type=int, default=None, help="Skip entries before this 0-based index in dump entries.")
    parser.add_argument("--force", action="store_true", help="Reclassify even if already rewritten.")
    parser.add_argument("--no-stream", action="store_true", help="Don't stream GPT-OSS musings to CLI")
    parser.add_argument("--save-harmony-traces", action="store_true", help="Enable debug outputs")
    parser.add_argument("--dequantize", action="store_true", help="Use Mxfp4Config(dequantize=True).")
    parser.add_argument("--suppress-native-output", action="store_true", help="Suppress native compiler/Triton output during load.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_path = Path(args.dump_json).resolve()
    if not dump_path.exists():
        raise FileNotFoundError(dump_path)

    stem = dump_path.stem
    out_rewrites = dump_path.with_name(f"{stem}_gptoss_rewrites.json")
    out_finals = dump_path.with_name(f"{stem}_gptoss_finals.json")
    out_errors = dump_path.with_name(f"{stem}_gptoss_errors.json")
    out_merged = dump_path.with_name(f"{stem}_rewritten_labels.json")
    trace_dir = dump_path.with_name(f"{stem}{TRACE_DIR_SUFFIX}")

    dump = load_json(dump_path, {})
    entries = dump.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Input dump JSON must contain an 'entries' list.")

    print(Fore.YELLOW + "[load] tokenizer/model" + Style.RESET_ALL)
    model_path = Path(args.model).resolve()
    print(f"[model path] {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    quant_config = Mxfp4Config(dequantize=args.dequantize)
    with suppress_native_output(args.suppress_native_output):
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            quantization_config=quant_config,
            local_files_only=True,
            trust_remote_code=True,
        ).eval()

    rewrites_by_id: dict[str, dict[str, Any]] = load_json(out_rewrites, {})
    finals_by_id: dict[str, str] = load_json(out_finals, {})
    errors_by_id: dict[str, Any] = load_json(out_errors, {})

    if args.force:
        rewrites_by_id = {}
        finals_by_id = {}
        errors_by_id = {}

    if args.save_harmony_traces:
        trace_dir.mkdir(exist_ok=True)

    print(
        Fore.YELLOW
        + f"[load] entries={len(entries)} existing_rewrites={len(rewrites_by_id)} "
          f"errors={len(errors_by_id)} reasoning={args.reasoning} "
          f"max_new_tokens={args.max_new_tokens} repetition_penalty={args.repetition_penalty} "
          f"use_kv_cache={not args.no_kv_cache}"
        + Style.RESET_ALL
    )

    processed_this_run = 0

    for index, entry in enumerate(entries):
        if args.start_index is not None and index < args.start_index:
            continue

        entry_id = get_entry_id(entry)
        if not entry_id:
            continue

        if entry_id in rewrites_by_id and not args.force:
            continue

        if args.limit is not None and processed_this_run >= args.limit:
            break

        labels = entry.get("labels") or []
        print(Fore.YELLOW + Style.BRIGHT + f"\n\n=== [{index + 1}/{len(entries)}] {entry_id} ===" + Style.RESET_ALL)
        for i, label in enumerate(labels):
            print(Fore.YELLOW + f"LABEL[{i}]: {clean_text(label)[:260]}" + Style.RESET_ALL)

        try:
            record, final_text, raw_text, channels = rewrite_entry(
                model=model,
                tokenizer=tokenizer,
                entry=entry,
                reasoning=args.reasoning,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                max_label_chars=args.max_label_chars,
                use_kv_cache=not args.no_kv_cache,
                stream=not args.no_stream,
            )

            rewrites_by_id[entry_id] = record
            finals_by_id[entry_id] = final_text
            errors_by_id.pop(entry_id, None)

            if args.save_harmony_traces:
                (trace_dir / safe_trace_name(entry_id)).write_text(raw_text, encoding="utf-8")

            print(Fore.GREEN + Style.BRIGHT + f"\n[SAVED rewrite] {entry_id} changed={record['changed_indices']}" + Style.RESET_ALL)

        except Exception as exc:
            errors_by_id[entry_id] = {
                "image_key": entry.get("image_key"),
                "error": str(exc),
                "labels": entry.get("labels"),
                "flagged_caption_indices": entry.get("flagged_caption_indices"),
                "effective_flagged_indices": get_effective_flagged_indices(entry),
                "forbidden_terms": build_forbidden_terms(entry).get("raw_terms"),
            }
            print(Fore.RED + Style.BRIGHT + f"\n[ERROR] {entry_id}: {exc}" + Style.RESET_ALL)

        processed_this_run += 1

        save_json(out_rewrites, rewrites_by_id)
        save_json(out_finals, finals_by_id)
        save_json(out_errors, errors_by_id)

        if processed_this_run % max(1, args.merge_every) == 0:
            merge_rewrites_into_source(dump, rewrites_by_id, out_merged)

        if processed_this_run % max(1, args.cleanup_every) == 0:
            cleanup_torch_memory()

    merge_rewrites_into_source(dump, rewrites_by_id, out_merged)

    print(Fore.YELLOW + "\n[done]" + Style.RESET_ALL)
    print(f"  wrote {out_rewrites} ({len(rewrites_by_id)} rewritten)")
    print(f"  wrote {out_finals}")
    print(f"  wrote {out_errors} ({len(errors_by_id)} errors)")
    print(f"  wrote {out_merged}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "\n[interrupted] partial JSON files are preserved; rerun to resume." + Style.RESET_ALL)
        sys.exit(130)