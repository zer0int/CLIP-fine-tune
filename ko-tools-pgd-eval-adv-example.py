import os
import torch
from PIL import Image
import csv
import numpy as np
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)
from cliptools import fix_random_seed
fix_random_seed()
import clip
from safetensors.torch import load_file
import argparse
import matplotlib.pyplot as plt
import re
from glob import glob

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
alias = "None"

# ------------------------------------------
# Use to eval results after first running:
# ko-tools-pgd-adversarial-perturb.py
# ------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='ImageNet zero-shot attack, Pre-Trained vs. Fine-Tuned CLIP')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--path', default="image_sets/n01531178_pgd/custom", help="Path to adversarial dataset base dir")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model


if os.path.isfile(args.use_model):
    alias = "custom"  # It's a path to a file
else:
    alias = "pretrained"  # It's a model name like "ViT-L/14"

out_dir = f"results_oai/zeroshot-attack-adv-perturb/{alias}"
os.makedirs(out_dir, exist_ok=True)

# ---------------- ADVANCED SETTINGS --------------------------
# Set an attention head to ablate & compare results here
target_head = 10 # 0-15 for ViT-L/14
ablate_head = False # Set True to ablate
# -------------------------------------------------------------

if model_name_or_path.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    
    # Load ViT-L/14 explicitly
    finetuned_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # Load the safetensors state_dict and apply
    state_dict = load_file(model_name_or_path)
    finetuned_model.load_state_dict(state_dict)

else:
    print("Detected non-.safetensors file or name. Attempting to load model...")
    
    # Load normally as per the existing logic
    finetuned_model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

finetuned_model = finetuned_model.float() # full precision

original_model, _ = clip.load("ViT-L/14", device=device, jit=False)
original_model = original_model.float()

dataset = args.path
CHOICES = ["a photo of a bird", "a photo of a bumblebee", "a photo of a text"]

MODELS = {
    "Original CLIP": original_model,
    "Finetuned CLIP": finetuned_model,
}

OUT_CSVS = {
    ("Original CLIP", f"image_sets/{dataset}"): f"{out_dir}/original_model_normal_batch.csv",
    ("Finetuned CLIP", f"image_sets/{dataset}"): f"{out_dir}/finetuned_model_normal_batch.csv",
}

FOLDERS = [f"image_sets/{dataset}"]


def get_integer_subfolders(path):
    """Return sorted list of integer-named subfolders in the given path."""
    if not os.path.isdir(path):
        return []
    subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.isdigit()]
    return sorted(subfolders, key=lambda x: int(x))


def get_next_subfolder(base_dir):
    """Finds the next unused integer-named subfolder in base_dir."""
    i = 0
    while True:
        candidate = os.path.join(base_dir, str(i))
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        i += 1


def get_step_from_filename(fname):
    # Expect: ..._stepXXXX_adv.png, where XXXX is the step
    match = re.search(r'_step(\d+)', fname)
    return int(match.group(1)) if match else -1

def plot_per_image_adv_trajectory(
    adv_folder,
    models_dict,
    prompts,
    preprocess,
    out_dir,
    device,
    folder_suffix=None   # <-- new
):
    # Find all image files with _stepXXX_adv.png
    img_files = glob(os.path.join(adv_folder, "*_step*_adv.png"))
    if not img_files:
        print(f"No perturbed images found in {adv_folder}")
        return

    # Group by base image prefix
    img_groups = {}
    for f in img_files:
        base = os.path.basename(f)
        img_prefix = base.split("_step")[0]
        if img_prefix not in img_groups:
            img_groups[img_prefix] = []
        img_groups[img_prefix].append(f)

    # Precompute text embeddings for both models
    with torch.no_grad():
        text_embeds = {
            name: model.encode_text(clip.tokenize(prompts).to(device)).detach()
            for name, model in models_dict.items()
        }

    # Suffix for filenames
    suffix = f"_{folder_suffix}" if folder_suffix is not None else ""

    for img_prefix, files in img_groups.items():
        # Use subfolder per image for clarity (keeps things robust for many images)
        img_out_dir = get_next_subfolder(out_dir)
        files = sorted(files, key=get_step_from_filename)
        steps = [get_step_from_filename(f) for f in files]
        results = {name: {i: [] for i in range(len(prompts))} for name in models_dict.keys()}

        for f, step in zip(files, steps):
            img_pil = Image.open(f).convert("RGB")
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            for name, model in models_dict.items():
                with torch.no_grad():
                    img_emb = model.encode_image(img_tensor)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    for idx in range(len(prompts)):
                        text_emb = text_embeds[name][idx].unsqueeze(0)
                        sim = (img_emb @ text_emb.T).item()
                        results[name][idx].append(sim)

        # Plot for each prompt
        for idx, prompt in enumerate(prompts):
            plt.figure(figsize=(8,4))
            for name, color in zip(models_dict.keys(), ['red', 'blue']):
                sims = results[name][idx]
                plt.plot(steps, sims, marker='o', label=name, color=color)
            plt.title(f"{img_prefix} – Prompt: {prompt}")
            plt.xlabel("Step")
            plt.ylabel("Cosine Similarity")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_filename = os.path.join(img_out_dir, f"{img_prefix}_prompt{idx}{suffix}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved plot: {plot_filename}")

        # All prompts together
        plt.figure(figsize=(8,6))
        for idx, prompt in enumerate(prompts):
            for name, color, style in zip(models_dict.keys(), ['red', 'blue'], ['-', '--']):
                sims = results[name][idx]
                plt.plot(steps, sims, marker='o', label=f"{name}: {prompt}", color=color, linestyle=style)
        plt.title(f"{img_prefix}: Adversarial Trajectory")
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        allprompts_plot = os.path.join(img_out_dir, f"{img_prefix}_allprompts{suffix}.png")
        plt.savefig(allprompts_plot)
        plt.close()
        print(f"Saved: {allprompts_plot}")

# ---- 1. Zeroing Head 10 Attention Output FOR ALL LAYERS ----
def ablate_head_output_all_layers(model, head_idx):
    handles = []
    for block in model.visual.transformer.resblocks:
        def make_hook(head_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_out = output[0]
                    rest = output[1:]
                else:
                    attn_out = output
                    rest = ()
                B, N, D = attn_out.shape
                output_reshaped = attn_out.view(B, N, module.num_heads, module.head_dim)
                output_reshaped[:, :, head_idx, :] = 0.0
                attn_out_new = output_reshaped.view(B, N, D)
                if rest:
                    return (attn_out_new, *rest)
                else:
                    return attn_out_new
            return hook_fn
        handles.append(block.attn.register_forward_hook(make_hook(head_idx)))
    return handles

# ---- 2. Zeroing Q and V weights for Head 10 IN ALL LAYERS ----
def zero_qv_weights_all_layers(model, head_idx):
    for block in model.visual.transformer.resblocks:
        attn = block.attn
        head_dim = attn.head_dim
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        with torch.no_grad():
            attn.q_proj.weight[start:end, :] = 0.0
            if attn.q_proj.bias is not None:
                attn.q_proj.bias[start:end] = 0.0
            attn.v_proj.weight[start:end, :] = 0.0
            if attn.v_proj.bias is not None:
                attn.v_proj.bias[start:end] = 0.0

if ablate_head:
    print("---------------------------------------")
    print(f"WARNING!! Ablating Attention Head {target_head}")
    print("Ensure this is what you want!")
    print("---------------------------------------")
    hook_handles = ablate_head_output_all_layers(original_model, target_head)
    hook_handles_ft = ablate_head_output_all_layers(finetuned_model, target_head)


# Tokenize text once (shared between all models and images)
text_tokens = clip.tokenize(CHOICES).to(device)

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
            # Save row
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

    # Summary stats
    def stat_str(arr):
        if not arr: return "n/a"
        return f"min={min(arr):.4f}, max={max(arr):.4f}, mean={np.mean(arr):.4f}"

    summary = (
        f"Choices: {CHOICES}\n"
        f"{model_name}, {folder}, \ncorrect: {correct} [{stat_str(margins_correct)}], "
        f"\nmisclassified: {incorrect} [{stat_str(margins_incorrect)}]\n\n"
    )

    # Colorized summary for print
    summary_for_print = (
        f"Choices: {CHOICES}\n"
        f"{color_text(model_name, Fore.MAGENTA)}"
        f", {folder}, "
        f"\n{color_text('correct', Fore.GREEN)}: {correct} [{stat_str(margins_correct)}], "
        f"\n{color_text('misclassified', Fore.RED)}: {incorrect} [{stat_str(margins_incorrect)}]\n\n"
    )

    if print_summary:
        print(summary_for_print)

    with open(f"{out_dir}/zeroshot-summary.txt", "a", encoding='utf-8') as f:
        f.write(summary + "\n")

    with open(out_csv, "a", newline="") as f:
        f.write("# " + summary + "\n")
    return summary

if __name__ == "__main__":
    # Precompute text embeddings
    with torch.no_grad():
        text_embeds = {}
        for name, model in MODELS.items():
            emb = model.encode_text(text_tokens)
            emb /= emb.norm(dim=-1, keepdim=True)
            text_embeds[name] = emb

    eval_base_dir = args.path
    integer_subfolders = get_integer_subfolders(eval_base_dir)

    # Case 1: Subfolders exist (i.e., "0", "1", "2", ...)
    if integer_subfolders:
        print(f"Detected {len(integer_subfolders)} subfolders: {integer_subfolders}")
        for subdir in integer_subfolders:
            folder = os.path.join(eval_base_dir, subdir)
            print(f"\n=== Evaluating subfolder: {folder} ===")
            plot_out_dir = os.path.join(out_dir, subdir)
            os.makedirs(plot_out_dir, exist_ok=True)

            # For each model, run evaluation and save results, using subdir in filenames
            for model_name, model in MODELS.items():
                out_csv = os.path.join(out_dir, f"{model_name.replace(' ', '_')}_{subdir}.csv")
                summary_file = os.path.join(out_dir, f"zeroshot-summary_{subdir}.txt")
                process_folder(
                    model, model_name, folder, text_embeds[model_name], out_csv, print_summary=True
                )

            # Only call plot_per_image_adv_trajectory once per subfolder
            plot_per_image_adv_trajectory(
                adv_folder=folder,
                models_dict=MODELS,
                prompts=CHOICES,
                preprocess=preprocess,
                out_dir=plot_out_dir,  # Plots go in out_dir/{subdir}
                device=device,
                folder_suffix=subdir
            )
    else:
        # Case 2: Evaluate a single flat folder of images (old format)
        print(f"Evaluating a single folder: {eval_base_dir}")
        for model_name, model in MODELS.items():
            out_csv = OUT_CSVS.get((model_name, eval_base_dir), os.path.join(out_dir, f"{model_name.replace(' ', '_')}_single.csv"))
            process_folder(
                model, model_name, eval_base_dir, text_embeds[model_name], out_csv, print_summary=True
            )

        plot_per_image_adv_trajectory(
            adv_folder=eval_base_dir,
            models_dict=MODELS,
            prompts=CHOICES,
            preprocess=preprocess,
            out_dir=out_dir,
            device=device
        )
