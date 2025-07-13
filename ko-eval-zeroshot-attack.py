import os
import torch
from PIL import Image
import csv
import numpy as np
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)
import clip
from safetensors.torch import load_file
import argparse
from cliptools import fix_random_seed
fix_random_seed()

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# Quick typographic attack test on CLIP
# Measures classification of images
# -> as the text in the image (misclassification)
# -> as the actual object in image (correct label)
# ---------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='ImageNet zero-shot attack, Pre-Trained vs. Fine-Tuned CLIP')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument("--use_cats", action='store_true', help="Use cats with 'dog' written on them. Default: Goldfinch with 'bumblebee' text.")
    parser.add_argument('--head', default=10, type=int, help="Attention Head to ablate (0-15). Default: 10. Only active when also passing --ablate_head")
    parser.add_argument("--ablate_head", action='store_true', help="Ablate Attention Head (to benchmark influence thereof)")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

out_dir = "results_oai/zeroshot-attack"
os.makedirs(out_dir, exist_ok=True)

# ---------------- ADVANCED SETTINGS --------------------------
# Set an attention head to ablate & compare results here
target_head = args.head # 0-15 for ViT-L/14
ablate_head = False
if args.ablate_head:
    ablate_head = True
# -------------------------------------------------------------

use_cats = False # If false, use bumblebee
if args.use_cats:
    use_cats = True


if use_cats:
    dataset = "n02123159" # cats, or "dog" written on cats
    CHOICES = ["a photo of a cat", "a photo of a dog", "a photo of a text"]
else:
    dataset = "n01531178"  # goldfinch, or "bumblebee" written on goldfinch
    CHOICES = ["a photo of a bird", "a photo of a bumblebee", "a photo of a text"]


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


MODELS = {
    "Original CLIP": original_model,
    "Finetuned CLIP": finetuned_model,
}

OUT_CSVS = {
    ("Original CLIP", f"image_sets/{dataset}"): f"{out_dir}/original_model_normal_batch.csv",
    ("Original CLIP", f"image_sets/{dataset}_adv"): f"{out_dir}/original_model_adv_batch.csv",
    ("Finetuned CLIP", f"image_sets/{dataset}"): f"{out_dir}/finetuned_model_normal_batch.csv",
    ("Finetuned CLIP", f"image_sets/{dataset}_adv"): f"{out_dir}/finetuned_model_adv_batch.csv",
}

FOLDERS = [f"image_sets/{dataset}", f"image_sets/{dataset}_adv"]

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

    for folder in FOLDERS:
        for model_name, model in MODELS.items():
            out_csv = OUT_CSVS[(model_name, folder)]
            process_folder(
                model, model_name, folder, text_embeds[model_name], out_csv, print_summary=True
            )
