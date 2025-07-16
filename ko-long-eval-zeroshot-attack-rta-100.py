import os
import re
import torch
from PIL import Image
import longclip as clip
import argparse
from tqdm import tqdm
import pandas as pd
from collections import Counter
from safetensors.torch import load_file

from cliptools import fix_random_seed
fix_random_seed()

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# ------------------------------------------------------------------
# Eval on another, diverse typographic attack dataset.
# 1000 photos with post-it notes stuck to objects.
# Download:
# https://github.com/azuma164/Defense-Prefix/blob/main/rta100.zip
# ------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate CLIP Zero-Shot on RTA-100')
    parser.add_argument('--data_dir', default="path/to/images/of/rta100", help="Dataset folder")
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-KO-LITE-FULL-OpenAI-format.safetensors", help="CLIP model path")
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

args = parse_arguments()
device = args.device
model_name_or_path = args.use_model


model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float()

pattern = re.compile(r'label=(.+?)_text=(.+?)\.(jpg|jpeg|png)$', re.IGNORECASE)
data = []
for fname in os.listdir(args.data_dir):
    m = pattern.match(fname)
    if not m:
        continue
    correct_label = m.group(1)
    distractor_label = m.group(2)
    data.append({'filename': fname, 'correct_label': correct_label, 'distractor_label': distractor_label})

print(f"Found {len(data)} images.")


results = []
with torch.no_grad():
    for entry in tqdm(data, desc="Evaluating images"):
        image_path = os.path.join(args.data_dir, entry['filename'])
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        # Prompt templates
        texts = [
            f"a photo of a {entry['correct_label']}",
            f"a photo of a {entry['distractor_label']}"
        ]
        text_tokens = clip.tokenize(texts).to(device)

        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits_per_image = image_features @ text_features.T  # (1, 2)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

        pred_idx = int(probs.argmax())
        is_correct = (pred_idx == 0)
        results.append({
            'filename': entry['filename'],
            'correct_label': entry['correct_label'],
            'distractor_label': entry['distractor_label'],
            'pred_label': texts[pred_idx].replace("a photo of a ", ""),
            'is_correct': is_correct,
            'confidence_correct': float(probs[0]),
            'confidence_distractor': float(probs[1])
        })

# --- SUMMARY ---
n_total = len(results)
n_correct = sum(r['is_correct'] for r in results)
print(f"\nZero-shot accuracy: {n_correct}/{n_total} = {n_correct/n_total:.4f}")

# --- OPTIONAL: SAVE TO CSV ---
out_dir = "results_oai/rta100"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rta100_results.csv")
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"Saved per-image results to {out_path}")

# Collect wrong predictions: which distractor label was picked?
incorrect_preds = [
    r['distractor_label'] for r in results if not r['is_correct'] and r['pred_label'] == r['distractor_label']
]
counter = Counter(incorrect_preds)
worst_20 = counter.most_common(20)

print("\nTop-20 most problematic distractors:")
for label, count in worst_20:
    print(f"{label:20s}: {count}")

with open(os.path.join(out_dir, "rta100_top100_wrong.txt"), "w", encoding="utf8") as f:
    for label, count in worst_20:
        f.write(f"{label:100s}: {count}\n")

