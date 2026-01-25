"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

CLIP gradient ascent on text to sample some 'CLIP opinion' tokens about an image.

Uses a heavily modified version of Original CLIP Gradient Ascent Script: by Twitter / X: @advadnoun
"""
import argparse
import os
import kornia.augmentation as kaugs
import kornia
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from safetensors.torch import load_file
from PIL import Image
from colorama import Fore, Style
import copy
import random

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

scaler = GradScaler()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Get a CLIP gradient ascent opinion about an image!')
    parser.add_argument('--use_model', default="zer0int/CLIP-Regression-ViT-L-14", help="OpenAI model, local path .pt or .safetensors, HuggingFace Hub")
    parser.add_argument('--use_image', type=str, default="image_sets/n01531178_adv/n01531178_877.JPEG", help="Path to image")
    parser.add_argument('--img_folder', type=str, default="None", help="Path to image folder (batch). If None, uses single --use_image.")
    parser.add_argument('--batch_size', default=24, type=int, help="Reduce batch_size for lower VRAM consumption")
    parser.add_argument('--num_tokens', default=12, type=int, help="If lower batch_size and gibberish shows up, try reducing to 4 or 8") 
    parser.add_argument("--dump_embeds", action='store_true', help="Dump optimized text embeddings as .pt files")
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (CUDA backends, torch, numpy)")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

def fix_random_seed(seed: int = 6247423):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Expect mean and std as lists of 3 elements.
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

def augment(into, augs):
    return augs(into)

# Text encoder forward
def clip_encode_text(model, text, many_tokens, prompt):
    x = torch.matmul(text, model.token_embedding.weight)
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ model.text_projection
    return x

# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, bests, imagename):
    unique_tokens = set()
    os.makedirs("out_ga_texts", exist_ok=True)
    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('<|startoftext|>', '').replace('<|endoftext|>', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
                with open(f"out_ga_texts/tokens_{imagename}_all.txt", "a", encoding='utf-8') as f:
                    f.write("".join(decoded_tokens))
                    f.write("\n")
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:12], list(bests.keys())[:12]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        tokens = j.split()
        unique_tokens.update(tokens)
    
    with open(f"out_ga_texts/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, batch_size, text_pos_shape, many_tokens, prompt):
        super(Pars, self).__init__()
        self.batch_size = batch_size
        self.many_tokens = many_tokens
        self.prompt = prompt
        self.gumbel_temp = 1000
        self.text_pos_shape = text_pos_shape
        
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_embeddings = torch.zeros(batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt_embeddings[:, jk, pt] = 1 

        pad_length = text_pos_shape - (self.many_tokens + len(self.prompt) + 1)
        self.pad = torch.zeros(self.batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        soft = F.gumbel_softmax(self.normu, tau=self.gumbel_temp, dim=-1, hard=True)

        return torch.cat([self.start, self.prompt_embeddings, soft, self.pad], 1)


# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment, j):
    B = lats.normu.shape[0]
    R = 4
    LSE_TAU = 0.07

    # make B*R augmented views
    img_rep = image[:, :3, :, :].expand(B * R, -1, -1, -1)   # [B*R,3,H,W]
    iii_in = nom(augment(img_rep))                           # [B*R,3,H,W]

    # encode image repeats once (no grads)
    with torch.no_grad(), autocast():
        iii = model.encode_image(iii_in).detach()            # [B*R,D]

    # reshape to repeats
    iii_rep = iii.view(R, B, -1)                              # [R,B,D]

    # Text side (one per candidate)
    lll = lats()                                              # [B,77,V]
    tx = clip_encode_text(model, lll, many_tokens, prompt)     # [B,D]

    # cosine similarity per repeat - sim_rb: [R,B]
    sim_rb = F.cosine_similarity(
        iii_rep.float(),                                       # [R,B,D]
        tx.unsqueeze(0).float().expand(R, -1, -1),             # [R,B,D]
        dim=-1
    )

    # logsumexp aggregation over repeats (soft-max over augmentations)
    tau = LSE_TAU
    sim_lse = tau * torch.logsumexp(sim_rb / tau, dim=0) - tau * np.log(R)   # [B]

    if j % 50 == 0:
        sim_mean_print = sim_rb.mean(dim=0).mean().item()
        sim_max_print  = sim_rb.max(dim=0).values.mean().item()
        sim_lse_print  = sim_lse.mean().item()
        print(f"sim_mean={sim_mean_print:.4f} sim_max={sim_max_print:.4f} sim_lse={sim_lse_print:.4f}")

    loss = -100 * sim_lse                                      # [B]

    return loss, tx, lll


# Loop with AMP
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment, j):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment, j)
    loss = loss1.mean()
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll


def generate_target_text_embeddings(img_path, model, lats, optimizer, training_iterations, checkin_step, many_tokens, prompt, nom, augment, tok, bests, args):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)

    scaler = GradScaler()

    best_loss = float('inf')
    best_text_embeddings = None

    for j in range(training_iterations):
                
        # Training step
        loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment, j)
        current_loss = loss.mean().item()
        
        # Update best embeddings if current loss is better
        if current_loss < best_loss:
            best_loss = current_loss
            best_text_embeddings = copy.deepcopy(tx.detach())
            print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)
            print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)

        # Print for monitoring
        if j % 50 == 0:
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)

    if args.dump_embeds:
        os.makedirs("out_ga_txtembeds", exist_ok=True)
        torch.save(best_text_embeddings, f"out_ga_txtembeds/{img_name}_emb.pt")
        print(Fore.MAGENTA + Style.BRIGHT + "\nBest text embedding saved to 'out_ga_txtembeds'." + Fore.RESET)
    print(Fore.MAGENTA + Style.BRIGHT + "Tokens (CLIP 'opinion') saved to 'out_ga_texts' folder.\n" + Fore.RESET)
    return img, best_text_embeddings, img_path


def main():
    args = parse_arguments()
    if args.deterministic:
        fix_random_seed()
    
    device="cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess, _ = load_openai_clip_anything(clip, model_name_or_path, device=device, jit=False, strict=True)    
    model = model.float()
    text_pos_shape = model.positional_embedding.shape[0]

    normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).cuda()
    tok = clip.simple_tokenizer.SimpleTokenizer()

    if args.img_folder != "None":
        image_folder = args.img_folder
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                       if f.lower().endswith(valid_extensions)]
        
        for img_path in image_files:
            augs = torch.nn.Sequential(kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda()).cuda()
            checkin_step = 10  
            iterations = 301
            tokinit = args.num_tokens

            bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None', 1006: 'None', 1007: 'None', 1008: 'None', 1009: 'None', 1010: 'None', 1011: 'None'}
            prompt = clip.tokenize('''''').numpy().tolist()[0]
            prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

            lats = Pars(args.batch_size, text_pos_shape, tokinit, prompt).cuda()    
            optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
            
          
            img, target_text_embedding, img_path = generate_target_text_embeddings(img_path, model, lats, optimizer, iterations, checkin_step, tokinit, prompt, normalizer, augs, tok, bests, args)
            print(f"Done processing image: {img_path}")

    else:
        augs = torch.nn.Sequential(kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda()).cuda()
        checkin_step = 10  
        iterations = 301
        tokinit = args.num_tokens

        bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None', 1006: 'None', 1007: 'None', 1008: 'None', 1009: 'None', 1010: 'None', 1011: 'None'}
        prompt = clip.tokenize('''''').numpy().tolist()[0]
        prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

        lats = Pars(args.batch_size, text_pos_shape, tokinit, prompt).cuda()    
        optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
       
        img, target_text_embedding, img_path = generate_target_text_embeddings(args.use_image, model, lats, optimizer, iterations, checkin_step, tokinit, prompt, normalizer, augs, tok, bests, args)
        print(f"Done processing image: {img_path}")


if __name__ == "__main__":
    main()