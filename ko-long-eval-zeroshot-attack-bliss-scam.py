import os
from datasets import load_dataset
import longclip as clip
import torch
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file
import pandas as pd
import argparse
from cliptools import fix_random_seed
fix_random_seed()

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------
# BLISS / SCAM Typographic Attack Dataset (auto-load)
# https://huggingface.co/datasets/BLISS-e-V/SCAM
# -----------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate CLIP Zero-Shot on BLISS-SCAM')
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-KO-LITE-FULL-OpenAI-format.safetensors", help="CLIP model path")
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

args = parse_arguments()
device = args.device
model_name_or_path = args.use_model

ds = load_dataset("BLISS-e-V/SCAM", split="train")

model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float()

out_dir = "results_oai/bliss_scam"
os.makedirs(out_dir, exist_ok=True)

for variant in ["NoSCAM", "SCAM", "SynthSCAM"]:
    print(f"\n=== Evaluating variant: {variant} ===")
    # Filter dataset for current variant
    idxs = [i for i, v in enumerate(ds['id']) if v.startswith(variant)]
    if not idxs:
        print(f"  No samples for {variant}")
        continue
    # Build subset
    subset = [ds[i] for i in idxs]

    results = []
    for entry in tqdm(subset, desc=f"Evaluating {variant}", ncols=75):
        img = entry['image']
        object_label = entry['object_label']
        attack_word = entry['attack_word']

        texts = [f"a photo of a {object_label}", f"a photo of a {attack_word}"]
        text_tokens = clip.tokenize(texts).to(device)
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_features = model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits = img_features @ text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy().flatten()
            pred_idx = probs.argmax()
            pred_label = [object_label, attack_word][pred_idx]
            is_correct = (pred_label == object_label)

        results.append({
            "id": entry['id'],
            "object_label": object_label,
            "attack_word": attack_word,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "confidence_object": float(probs[0]),
            "confidence_attack": float(probs[1]),
            "postit_area_pct": entry['postit_area_pct'],
            "type": entry['type']
        })

    n_total = len(results)
    n_correct = sum(r['is_correct'] for r in results)
    acc = n_correct / n_total if n_total else float('nan')
    print(f"Zero-shot accuracy for {variant}: {n_correct}/{n_total} = {acc:.4f}")

    out_path = os.path.join(out_dir, f"bliss_scam_{variant}_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    # Margin/statistics
    df = pd.DataFrame(results)
    df['margin'] = df['confidence_object'] - df['confidence_attack']

    print("Cosine margin statistics:")
    print(df['margin'].describe())

    # Error cases: which attack words are most successful?
    wrong = df[~df['is_correct']]
    print("\nTop 20 attack words that fooled CLIP:")
    print(wrong['attack_word'].value_counts().head(20))

print("\nAll variants processed.")
