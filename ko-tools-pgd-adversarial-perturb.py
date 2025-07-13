import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import cv2
from skimage import exposure
from safetensors.torch import load_file
import argparse

import clip

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
alias = "None"

# --------------------------------------------------------
# Uses PGD (Projected Gradient Descent) to perturb images
# -> 'Shuffle' pixels so they 'look like' something else
# Goldfinch -> Bumblebee (to CLIP! Humans can't see it!)
# Generated images act as adversarial examples.
#
# Run ko-tools-pgd-eval-adv-example.py to eval these.
# --------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Adversarial Perturbation Attack Generator')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

# === CONFIG ===
IN_DIR = "image_sets/n01531178"
OUT_DIR = "image_sets/n01531178_pgd"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 4/255       # Max L∞ perturbation
ALPHA = 0.05/255     # Step size
MAX_STEPS = 513  # Max steps

prompts = [
    "a photo of a bumblebee",
    "a photo of a goldfinch", 
]
TARGET_IDX = 1

if os.path.isfile(args.use_model):
    alias = "custom"  # It's a path to a file
else:
    alias = "pretrained"  # It's a model name like "ViT-L/14"

FINAL_OUT_BASE = f"image_sets/n01531178_pgd/{alias}"
os.makedirs(FINAL_OUT_BASE, exist_ok=True)

# ---- CLIP Normalization constants ----
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def get_next_subfolder(base_dir):
    # Finds the next unused integer-named subfolder in base_dir.
    i = 0
    while True:
        candidate = os.path.join(base_dir, str(i))
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        i += 1

def unnormalize(img_tensor):
    mean = torch.tensor(CLIP_MEAN).view(-1,1,1).to(img_tensor.device)
    std = torch.tensor(CLIP_STD).view(-1,1,1).to(img_tensor.device)
    return img_tensor * std + mean

def adjust_black_white_points(image_np, black_points, white_points):
    out = np.zeros_like(image_np)
    for c in range(3):
        out[..., c] = exposure.rescale_intensity(
            image_np[..., c],
            in_range=(black_points[c], white_points[c]),
            out_range=(0,255)
        )
    return out.astype(np.uint8)

def save_img_unnormalized(tensor, filename, apply_clahe=True, apply_stretch=True, upscale_factor=2):
    arr = unnormalize(tensor.cpu())
    arr = np.clip(arr.permute(1,2,0).detach().numpy(), 0, 1)

    # Optionally stretch contrast for adversarial images
    if apply_stretch:
        arr = exposure.rescale_intensity(arr, in_range=(arr.min(), arr.max()), out_range=(0, 1))

    arr = (arr * 255).astype(np.uint8)
    if apply_clahe:
        # Convert to LAB and apply CLAHE to L-channel
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        limg = cv2.merge((l, a, b))
        arr = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(arr)
    if upscale_factor > 1:
        img = img.resize((img.width*upscale_factor, img.height*upscale_factor), Image.LANCZOS)
    img.save(filename)

  
def tv_loss(img):
    # Total variation loss for smoothness (regularizes spatial gradients)
    return (
        torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
        torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    )

def cosine_lr_schedule(step, total_steps, min_lr=1e-3, max_lr=2e-2):
    t = step / total_steps
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * t))

def print_all_cosines(model, img, text_features, prompts):
    with torch.no_grad():
        feats = model.encode_image(img)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for idx, p in enumerate(prompts):
            score = (feats @ text_features[idx].unsqueeze(0).T)[0,0].item()
            print(f"{p}: {score:.4f}", end="  ")
        print()

def pgd_attack(
    model, image, target_embedding,
    epsilon, alpha, iters,
    save_every=False, save_steps=25, image_path='',
    use_momentum=True, lr_schedule=None, reg_factor=1e-4, tv_factor=1e-7,
    swa_start=0.75, swa_frequency=5,
    print_cosines_fn=None, text_features=None, prompts=None,
    device="cuda"
):
    mean = torch.tensor(CLIP_MEAN).view(-1,1,1).to(device)
    std = torch.tensor(CLIP_STD).view(-1,1,1).to(device)

    perturbed_image = image.clone().detach().requires_grad_(True)
    swa_image = perturbed_image.clone().detach()
    momentum = torch.zeros_like(perturbed_image).to(device)
    best_loss = None

    for i in range(iters):
        cur_alpha = alpha if not lr_schedule else lr_schedule(i, iters)
        output = model.encode_image(perturbed_image)
        output = output / output.norm(dim=-1, keepdim=True)
        target_embedding_norm = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        cos_sim = torch.nn.functional.cosine_similarity(output, target_embedding_norm).mean()

        # Want to maximize similarity to target (minimize negative sim)
        logits = (output @ text_features.T)[0]
        loss = -logits[TARGET_IDX] + logits[0]  # 0: correct, 1: adversarial


        # L2 regularization (keep perturbation small)
        l2_reg = reg_factor * torch.norm(perturbed_image - image)
        loss += l2_reg

        # TV loss (smoothness)
        if tv_factor > 0:
            loss += tv_factor * tv_loss(perturbed_image)

        model.zero_grad()
        loss.backward(retain_graph=False)

        # Print all prompt cosines, if provided
        if print_cosines_fn and text_features is not None and prompts is not None and (i & (i - 1) == 0):
            print_all_cosines(model, perturbed_image, text_features, prompts)

        grad = perturbed_image.grad
        if grad is None:
            raise RuntimeError("Gradient is None at step {}".format(i))
        grad = grad / (grad.norm() + 1e-8)

        if use_momentum:
            momentum = 0.3 * momentum + grad
            perturbed_image = perturbed_image + cur_alpha * momentum.sign()
        else:
            perturbed_image = perturbed_image + cur_alpha * grad.sign()

        # Project back to epsilon L_inf ball, clamp to valid
        perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
        perturbed_image = perturbed_image.detach().requires_grad_(True)

        # SWA averaging in late phase
        if i > int(swa_start * iters) and i % swa_frequency == 0:
            swa_image = (swa_image + perturbed_image) / 2
            perturbed_image = swa_image.detach().requires_grad_(True)

        # Save intermediates
        #if i % 25 == 0:
        if (i & (i - 1) == 0) or (i == iters - 1):
            save_img_unnormalized(
                perturbed_image[0],
                os.path.join(OUT_DIR, f"{fname}_step{i:04d}_adv.png"),
                apply_clahe=True,
                apply_stretch=True,
                upscale_factor=2
            )
            print(f"[step {i+1}] loss={loss.item():.5f}")

    return perturbed_image.detach()

# -- CLIP Loading --
if model_name_or_path.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    
    # Load ViT-L/14 explicitly
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # Load the safetensors state_dict and apply
    state_dict = load_file(model_name_or_path)
    model.load_state_dict(state_dict)

else:
    print("Detected non-.safetensors file or name. Attempting to load model...")
    
    # Load normally as per the existing logic
    model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model.eval().float()

# -- Get text embeddings (do only once)
with torch.no_grad():
    texts = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# -- PGD for each image --
img_files = [f for f in os.listdir(IN_DIR) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]   
for idx, fname in enumerate(tqdm(img_files)):
    # Unique output dir for this image (incremental 0, 1, ...)
    OUT_DIR = get_next_subfolder(FINAL_OUT_BASE)
    out_base = os.path.join(OUT_DIR, os.path.splitext(fname)[0])

    img_path = os.path.join(IN_DIR, fname)
    orig_pil = Image.open(img_path).convert('RGB')
    img = preprocess(orig_pil).unsqueeze(0).to(device).clone()



    # Save original
    with torch.no_grad():
        img_features = model.encode_image(img)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        cos_correct = (img_features @ text_features[0].unsqueeze(0).T)[0,0].item()
        cos_adv     = (img_features @ text_features[TARGET_IDX].unsqueeze(0).T)[0,0].item()
        print(f"\n{fname}: Goldfinch={cos_correct:.4f}  Bumblebee={cos_adv:.4f}")

    save_img_unnormalized(
        img[0],
        os.path.join(f"{out_base}_orig_C{cos_correct:.4f}-X{cos_adv:.4f}.png"),
        apply_clahe=True,
        apply_stretch=True,
        upscale_factor=2
    )

    # --- Run PGD attack ---
    adv_img = pgd_attack(
        model, img, text_features[TARGET_IDX], epsilon=EPS, alpha=ALPHA, iters=MAX_STEPS,
        save_every=True, save_steps=20, image_path=out_base,
        use_momentum=True,
        lr_schedule=cosine_lr_schedule,
        reg_factor=1e-4, tv_factor=1e-7,
        print_cosines_fn=print_all_cosines,
        text_features=text_features, prompts=prompts,
        device=device
    )

    # Final cosine scores and save
    with torch.no_grad():
        img_features = model.encode_image(adv_img)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        cos_correct_new = (img_features @ text_features[0].unsqueeze(0).T)[0,0].item()
        cos_adv_new     = (img_features @ text_features[TARGET_IDX].unsqueeze(0).T)[0,0].item()
        print(f"ADV: Goldfinch={cos_correct_new:.4f}  Bumblebee={cos_adv_new:.4f}")

    save_img_unnormalized(
        adv_img[0],
        os.path.join(f"{out_base}_adv_C{cos_correct_new:.4f}-X{cos_adv_new:.4f}.png"),
        apply_clahe=True,
        apply_stretch=True,
        upscale_factor=2
    )

print("Done!")
