import os
import torch
from PIL import Image
import csv
import numpy as np
import clip
import pandas as pd
import re
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from colorama import Fore, Style, init as colorama_init
from cliptools import fix_random_seed
import argparse

colorama_init(autoreset=True)
fix_random_seed()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# A 'quick scan' test for all your fine-tune checkpoints 
# Runs the following:
# 1. Typographic attack vulnerability 
# 2. Linear Probe
# 3. Zero-Shot

# CONVERT WITH >>  ko-2-convert-back-to-weight.py  << FIRST!
# Checkpoint names must be e.g. "clip_ft_1_backtoweight.pt"

# ----------- DATASET -----------
# Download here:
# https://objectnet.dev/mvt/
#
# Then use: --data_dir /path/to/all

# ---------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Quick eval on ObjectNet MVT')
    parser.add_argument('--model_dir', default="ft-checkpoints", help="Path to directory of fine-tuned .pt checkpoints")
    parser.add_argument('--data_dir', default='path/to/dataset-difficulty-CLIP/data_release_2023/all/', help="Path to image folder")
    parser.add_argument('--data_csv', default='image_sets/human_responses-mini.csv', help="Path to labels file (provided with repo!)")
    parser.add_argument("--use_cats", action='store_true', help="Use 'dog' on cat attack (default: 'bumblebee' on goldfinch)")
    return parser.parse_args()

args = parse_arguments()
csv_file = args.data_csv
image_folder = args.data_dir
ft_dir = args.model_dir
out_dir = os.path.join(ft_dir, "results")
os.makedirs(out_dir, exist_ok=True)
adversarial_summary_file = os.path.join(out_dir, "adversarial.txt")

if args.use_cats:
    dataset = "n02123159"
    CHOICES = ["a photo of a cat", "a photo of a dog", "a photo of a text"]
else:
    dataset = "n01531178"
    CHOICES = ["a photo of a bird", "a photo of a bumblebee", "a photo of a text"]

FOLDERS = [f"image_sets/{dataset}", f"image_sets/{dataset}_adv"]

def get_images_in_folder(folder):
    allowed_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(allowed_ext)]

def compute_results(model, image, text_embeddings):
    with torch.no_grad():
        image_embeddings = model.encode_image(image)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        logits = (image_embeddings @ text_embeddings.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1).squeeze().cpu().numpy()
        return probs

def color_text(text, color, bright=True):
    style = Style.BRIGHT if bright else ""
    return f"{color}{style}{text}{Style.RESET_ALL}"

def stat_str(arr):
    if not arr: return "n/a"
    return f"min={min(arr):.4f}, max={max(arr):.4f}, mean={np.mean(arr):.4f}"

def process_folder(model, model_name, folder, text_embeds, out_csv, print_summary=True):
    image_paths = get_images_in_folder(folder)
    correct, incorrect, margins_correct, margins_incorrect = 0, 0, [], []
    rows = []
    for path in image_paths:
        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            probs = compute_results(model, image, text_embeds)
            idx_cat, idx_dog = 0, 1
            margin = float(np.abs(probs[idx_cat] - probs[idx_dog]))
            predicted = np.argmax(probs)
            gt_label = 0 if f"{dataset}" in folder else 1
            correct_pred = (predicted == gt_label)
            if correct_pred:
                correct += 1
                margins_correct.append(margin)
            else:
                incorrect += 1
                margins_incorrect.append(margin)
            rows.append([
                os.path.basename(path),
                probs[0], probs[1], probs[2],
                "cat" if predicted == 0 else ("dog" if predicted == 1 else "text"),
                "correct" if correct_pred else "misclassified",
                margin
            ])
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Save to CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "p_cat", "p_dog", "p_text", "predicted", "result", "margin"])
        for row in rows:
            writer.writerow(row)

    summary = (
        f"{model_name}, {folder}, \n"
        f"correct: {correct} [{stat_str(margins_correct)}], "
        f"\nmisclassified: {incorrect} [{stat_str(margins_incorrect)}],\n"
    )

    # Colorized summary for print
    summary_for_print = (
        f"{color_text(model_name, Fore.MAGENTA)}, {folder}, "
        f"\n{color_text('correct', Fore.GREEN)}: {correct} [{stat_str(margins_correct)}], "
        f"\n{color_text('misclassified', Fore.RED)}: {incorrect} [{stat_str(margins_incorrect)}],\n"
    )

    if print_summary:
        print(summary_for_print)

    with open(out_csv, "a", newline="") as f:
        f.write("# " + summary + "\n")

    return summary

def sorted_checkpoints(directory):
    pts = [f for f in os.listdir(directory) if f.endswith('backtoweight.pt') and 'clip_ft_' in f]
    def extract_epoch(s):
        parts = s.split('_')
        if len(parts) < 3:
            return -1
        epoch_str = os.path.splitext(parts[2])[0]
        try:
            return int(epoch_str)
        except ValueError:
            return -1
    pts_sorted = sorted(pts, key=extract_epoch)
    return pts_sorted

class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_file_or_df, image_folder, transform=None):
        if isinstance(csv_file_or_df, str):
            self.data = pd.read_csv(csv_file_or_df)
        else:
            self.data = csv_file_or_df.reset_index(drop=True)
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
        label = self.data.iloc[idx]['label']
        return image, label

def save_embeddings(embs, fname):
    torch.save(embs, fname)

def load_embeddings(fname):
    return torch.load(fname)

def compute_linear_probe(image_embs, label_indices):
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    clf.fit(image_embs.cpu().numpy(), label_indices)
    preds = clf.predict(image_embs.cpu().numpy())
    acc = accuracy_score(label_indices, preds)
    return acc

def compute_zero_shot(image_embs, class_prompts, label_indices, model):
    with torch.no_grad():
        text_tokens = clip.tokenize(class_prompts).to(device)
        text_embs = model.encode_text(text_tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        sims = image_embs @ text_embs.t()
        preds = torch.argmax(sims, dim=1).cpu().numpy()
    return accuracy_score(label_indices, preds)

def sorted_checkpoints(directory):
    pts = [f for f in os.listdir(directory) if f.endswith('backtoweight.pt') and 'clip_ft_' in f]
    def extract_epoch(s):
        parts = s.split('_')
        if len(parts) < 3:
            return -1  # malformed
        epoch_str = os.path.splitext(parts[2])[0]
        try:
            return int(epoch_str)
        except ValueError:
            return -1
    pts_sorted = sorted(pts, key=extract_epoch)
    return pts_sorted

if __name__ == "__main__":
    text_tokens = clip.tokenize(CHOICES).to(device)

    checkpoints = sorted_checkpoints(ft_dir)
    print(Fore.MAGENTA + Style.BRIGHT + "\nRunning Typographic Attack..." + Fore.RESET)
    print(f"Found {len(checkpoints)} checkpoints.\n")

    with open(adversarial_summary_file, "w", encoding="utf-8") as adv_out:
        for checkpoint in checkpoints:
            checkpoint_num = checkpoint.split('_')[2]
            print(f"{'-'*25}\ncheckpoint {checkpoint_num}\n{'-'*25}")

            model_path = os.path.join(ft_dir, checkpoint)
            print("Loading:", model_path)
            finetuned_model, preprocess = clip.load(model_path, device=device, jit=False)
            finetuned_model = finetuned_model.float()

            # Precompute text embedding
            with torch.no_grad():
                emb = finetuned_model.encode_text(text_tokens)
                emb /= emb.norm(dim=-1, keepdim=True)

            # Run both folders, but only store summary for _adv
            for folder in FOLDERS:
                out_csv = os.path.join(out_dir, f"ft_model_{checkpoint_num}_{'adv' if folder.endswith('_adv') else 'normal'}.csv")
                summary = process_folder(
                    finetuned_model, f"Finetuned CLIP", folder, emb, out_csv, print_summary=True
                )

                if folder.endswith('_adv'):
                    adv_out.write(f"checkpoint {checkpoint_num}\n")
                    adv_out.write(summary)
                    adv_out.write("\n")

    print(Fore.YELLOW + Style.BRIGHT + f"Wrote adversarial summaries to {adversarial_summary_file}\n\n" + Fore.RESET)
    print(Fore.MAGENTA + Style.BRIGHT + "Running Linear Probe and Zero-Shot..." + Fore.RESET)
    temp_dir = os.path.join(ft_dir, "temp")
    results_dir = os.path.join(ft_dir, "results")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    many_ckpt = len(checkpoints)
    print(f"Found {len(checkpoints)} checkpoints.")

    df = pd.read_csv(csv_file)
    classnames = sorted(list(set(df['label'])))
    class2idx = {c: i for i, c in enumerate(classnames)}
    df['label_idx'] = df['label'].map(class2idx)
    label_indices = df['label_idx'].values

    dataset = CroppedImageCSVFileDataset(df, image_folder, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ]))
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=6)

    metrics = []
    for ckpt_num in range(1, many_ckpt):
        ckpt_path = os.path.join(ft_dir, f"clip_ft_{ckpt_num}_backtoweight.pt")
        print(f"\n=== Checkpoint {ckpt_num} ===")
        temp_emb_file = os.path.join(temp_dir, f"embeddings_{ckpt_num}.pt")

        model, preprocess = clip.load(ckpt_path, device=device, jit=False)
        model.eval()

        if not os.path.exists(temp_emb_file):
            image_embs = []
            with torch.no_grad():
                for imgs, _ in tqdm(dataloader, desc=f"Encoding images (ckpt {ckpt_num})"):
                    imgs = imgs.to(device)
                    emb = model.encode_image(imgs)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    image_embs.append(emb.cpu())
            image_embs = torch.cat(image_embs, dim=0)
            save_embeddings(image_embs, temp_emb_file)
        else:
            image_embs = load_embeddings(temp_emb_file)

        # Linear probe
        lin_probe_acc = compute_linear_probe(image_embs, label_indices)
        # Zero-shot accuracy (bare class label)
        zs_acc = compute_zero_shot(image_embs.to(device), classnames, label_indices, model)

        print(f"Linear probe acc: {lin_probe_acc:.4f}")
        print(f"Zero-shot acc:    {zs_acc:.4f}")

        metrics.append({
            "checkpoint": ckpt_num,
            "linear_probe_acc": lin_probe_acc,
            "zero_shot_acc": zs_acc,
        })

    results_df = pd.DataFrame(metrics)
    results_df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print(Fore.YELLOW + Style.BRIGHT + f"\nSaved results to {os.path.join(results_dir, 'metrics.csv')}\n" + Fore.RESET)

    metrics_csv = os.path.join(results_dir, "metrics.csv")
    adversarial_txt = os.path.join(results_dir, "adversarial.txt")
    output_img = os.path.join(results_dir, "all.png")

    adversarial_stats = []
    with open(adversarial_txt, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        # Check for line like 'checkpoint X'
        m_ckpt = re.match(r'checkpoint\s+(\d+)', lines[i], re.IGNORECASE)
        if m_ckpt:
            checkpoint = int(m_ckpt.group(1))
            # The next two lines are irrelevant text, then data
            correct_line = lines[i+2] if 'correct:' in lines[i+2] else lines[i+3]
            misclassified_line = lines[i+3] if 'misclassified:' in lines[i+3] else lines[i+4]

            # Parse correct line
            mc = re.search(r'correct:\s*(\d+)\s*\[min=([0-9.eE+-]+),\s*max=([0-9.eE+-]+),\s*mean=([0-9.eE+-]+)\]', correct_line)
            mm = re.search(r'misclassified:\s*(\d+)\s*\[min=([0-9.eE+-]+),\s*max=([0-9.eE+-]+),\s*mean=([0-9.eE+-]+)\]', misclassified_line)
            if mc and mm:
                adversarial_stats.append({
                    "checkpoint": checkpoint,
                    "correct_n": int(mc.group(1)),
                    "correct_min": float(mc.group(2)),
                    "correct_max": float(mc.group(3)),
                    "correct_mean": float(mc.group(4)),
                    "misclassified_n": int(mm.group(1)),
                    "misclassified_min": float(mm.group(2)),
                    "misclassified_max": float(mm.group(3)),
                    "misclassified_mean": float(mm.group(4)),
                })
            else:
                print(f"Regex failed on block starting at line {i+1}")
            # Advance to next block (each block is 5 lines + 1 blank, robust to minor changes)
            i += 6
        else:
            i += 1

    adversarial_df = pd.DataFrame(adversarial_stats)
    print(adversarial_df.head())

    metrics_df = pd.read_csv(metrics_csv)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    x = adversarial_df["checkpoint"]

    # Correct
    axs[0].plot(x, adversarial_df["correct_min"], label="Correct min", color="darkgreen", linestyle='--')
    axs[0].plot(x, adversarial_df["correct_mean"], label="Correct mean", color="darkgreen", linestyle='-')
    axs[0].plot(x, adversarial_df["correct_max"], label="Correct max", color="darkgreen", linestyle=':')

    # Misclassified
    axs[0].plot(x, adversarial_df["misclassified_min"], label="Misclassified min", color="red", linestyle='--')
    axs[0].plot(x, adversarial_df["misclassified_mean"], label="Misclassified mean", color="red", linestyle='-')
    axs[0].plot(x, adversarial_df["misclassified_max"], label="Misclassified max", color="red", linestyle=':')

    axs[0].set_ylabel("Cosine similarity")
    axs[0].set_title("Adversarial dataset stats (top: correct in green, misclassified in red)")
    axs[0].legend(loc="upper left", ncol=2)
    axs[0].grid(True)

    # --- Bottom: Metrics ---
    x2 = metrics_df["checkpoint"]
    axs[1].plot(x2, metrics_df["linear_probe_acc"], label="Linear probe acc", color="royalblue", linewidth=2)
    axs[1].plot(x2, metrics_df["zero_shot_acc"], label="Zero-shot acc", color="orange", linewidth=2)
    axs[1].set_xlabel("Checkpoint")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Scan metrics (bottom: linear probe and zero-shot accuracy)")
    axs[1].legend(loc="lower right")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_img)
    print(Fore.YELLOW + Style.BRIGHT + f"\nSaved plot to {output_img}" + Fore.RESET)
    print(Fore.GREEN + Style.BRIGHT + "DONE!" + Fore.RESET)