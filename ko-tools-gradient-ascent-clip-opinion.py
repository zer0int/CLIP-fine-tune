import argparse
import os
import kornia.augmentation as kaugs
import kornia
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from safetensors.torch import load_file
from PIL import Image
from colorama import Fore, Style
import copy
import random
from cliptools import fix_random_seed
scaler = GradScaler()

import clip

# -----------------------------------------------
# Gradient Ascent on the Text Embeddings for 
# Cosine Similarity with a given Image Embedding
# -> Gets a "CLIP opinion" about an image.
#
# Uses a heavily modified version of 
# Original CLIP Gradient Ascent Script: 
# by Twitter / X: @advadnoun
# -----------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP gradient ascent')
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Path to a ViT-L/14 model, pickle (.pt) or .safetensors")
    parser.add_argument('--use_image', type=str, default="image_sets/special_attn_img/readtxt.png", help="Path to image")
    parser.add_argument('--img_folder', type=str, default="None", help="Path to image folder (batch process)")
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (CUDA backends, torch, numpy)")
    parser.add_argument("--min_cos_sim", action='store_true', help="Minimize Cosine Similarity of Text with Image, instead of Max Cos Sim")
    parser.add_argument("--ablate_neurons", action='store_true', help="Ablate the 13 register neurons in Layer 11 and Layer 12")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model
args.model_name = args.use_model

img_path = args.use_image
img_name = os.path.splitext(os.path.basename(img_path))[0]


# Custom hook to scale the feature activation
class FeatureScalerHook:
    def __init__(self, model, layer_idx, feature_idx, scale_factor, transformer_type='visual'):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_idx = feature_idx
        self.scale_factor = scale_factor
        self.transformer_type = transformer_type
        self.handle = None
        self.register_hook()

    def register_hook(self):
        def hook(module, input, output):
            output[:, :, self.feature_idx] *= self.scale_factor
            return output

        if self.transformer_type == 'visual':
            layer = self.model.visual.transformer.resblocks[self.layer_idx].mlp.c_fc
        else:
            layer = self.model.transformer.resblocks[self.layer_idx].mlp.c_fc
        self.handle = layer.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

# Image Loader
def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255   
    im = F.interpolate(im, (sideX, sideY))
    return im

# Augmentation Pipeline
def augment(into, augs):
    return augs(into)

# Gradient Ascent / text encoder forward
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
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')
        tokens = j.split()
        unique_tokens.update(tokens)
    os.makedirs("txtopinion", exist_ok=True)
    with open(f"txtopinion/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, batch_size, many_tokens, prompt):
        super(Pars, self).__init__()
        self.batch_size = batch_size
        self.many_tokens = many_tokens
        self.prompt = prompt
        self.gumbel_temp = 1000

        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_embeddings = torch.zeros(batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt_embeddings[:, jk, pt] = 1 

        pad_length = 77 - (self.many_tokens + len(self.prompt) + 1)
        self.pad = torch.zeros(self.batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        soft = F.gumbel_softmax(self.normu, tau=self.gumbel_temp, dim=-1, hard=True)

        return torch.cat([self.start, self.prompt_embeddings, soft, self.pad], 1)

def clip_encode_image(model, img_tensor, img_name):
    visual = model.visual
    x = visual.conv1(img_tensor)                      # [B, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)         # [B, width, grid**2]
    x = x.permute(0, 2, 1)                            # [B, grid**2, width]
    # Add class embedding token
    class_embed = visual.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
    x = torch.cat([class_embed, x], dim=1)            # [B, grid**2 + 1, width]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # Go through transformer blocks
    x = x.permute(1, 0, 2)                            # [L, B, width]
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                            # [B, L, width]

    x_cls = x[:, 0, :]                                # [B, width]
    x_ln = visual.ln_post(x_cls)

    if hasattr(visual, 'proj') and visual.proj is not None:
        x_proj = x_ln @ visual.proj
    else:
        x_proj = x_ln

    return x_proj


# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    #iii = model.encode_image(iii).detach()
    iii = clip_encode_image(model, iii, img_name).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    
    if args.min_cos_sim:
        loss = 100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1) # min cos sim
    else:
        loss = -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1) # max cos sim
    
    return loss, tx, lll


# Loop with AMP
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll


def generate_target_text_embeddings(img_path, model, lats, optimizer, training_iterations, checkin_step, many_tokens, prompt, nom, augment, tok, bests, args):

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    input_dims = model.visual.input_resolution
    img = load_image(img_path, input_dims, input_dims)
    
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)

    scaler = GradScaler()

    best_loss = float('inf')  # Initialize the best loss as infinity
    best_text_embeddings = None  # Placeholder for the best text embeddings

    for j in range(training_iterations):
        loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
        current_loss = loss.mean().item()

        # Update best embeddings if current loss is better
        if current_loss < best_loss:
            best_loss = current_loss
            best_text_embeddings = copy.deepcopy(tx.detach())
            print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)
            print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)

        if j % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)

    # Save the best embeddings to disk
    os.makedirs("txtembeds", exist_ok=True)
    torch.save(best_text_embeddings, f"txtembeds/{img_name}_emb.pt")
    print(Fore.MAGENTA + Style.BRIGHT + "\nBest text embedding saved to 'txtembeds'.\nTokens (CLIP 'opinion') saved to 'txtopinion' folder.\n" + Fore.RESET)
    del optimizer, lats, scaler, prompt
    return img, best_text_embeddings, img_path


# Main loop
def main():
    args = parse_arguments()
    
    if args.deterministic:
        fix_random_seed()
    
    device="cuda" if torch.cuda.is_available() else "cpu"

    if model_name_or_path.endswith(".safetensors"):
        print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")       
        model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
        state_dict = load_file(model_name_or_path)
        model.load_state_dict(state_dict)

    else:
        print("Detected non-.safetensors file. Attempting to load as a pickle...")
        model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

    normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).cuda()
    model = model.float()

    # -------------------- REGISTER NEURON ABLATION --------------------
    # You can set scale_factor to non-zero (e.g. 10) and see what happens!
    # If you see NaN in the loss, your scale_factor was too extreme.
   
    if args.ablate_neurons:
        top_activations_layer_11 = [9, 987, 1967, 2555, 3661, 3784]        # idx: Register neurons Layer 11
        top_activations_layer_12 = [42, 983, 1571, 2687, 3002, 3008, 3868] # idx: Register neurons Layer 12

        hooks_layer_11 = []
        for feature_idx in top_activations_layer_11:
            hook = FeatureScalerHook(model, layer_idx=11, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
            hooks_layer_11.append(hook)

        hooks_layer_12 = []
        for feature_idx in top_activations_layer_12:
            hook = FeatureScalerHook(model, layer_idx=12, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
            hooks_layer_12.append(hook)

    # ------------------------------------------------------------------

    tok = clip.simple_tokenizer.SimpleTokenizer()

    augs = torch.nn.Sequential(
        kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
    ).cuda()
    
    bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
    prompt = clip.tokenize('''''').numpy().tolist()[0]
    prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]


    checkin_step = 10  
    iterations=300
    tokinit = 4
    lats = Pars(args.batch_size, tokinit, prompt).cuda()
    
    optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])

    if args.img_folder != "None":
        image_folder = args.img_folder
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                       if f.lower().endswith(valid_extensions)]
        
        for img_path in image_files:
            tokinit = tokinit
            iterations = iterations
            lats = Pars(args.batch_size, tokinit, prompt).cuda()            
            optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
            
            bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
            prompt = clip.tokenize('''''').numpy().tolist()[0]
            prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]
            img, target_text_embedding, img_path = generate_target_text_embeddings(img_path, model, lats, optimizer, iterations, checkin_step, tokinit, prompt, normalizer, augs, tok, bests, args)
            print(f"Done processing image: {img_path}")

    else:
        img, target_text_embedding, img_path = generate_target_text_embeddings(args.use_image, model, lats, optimizer, iterations, checkin_step, tokinit, prompt, normalizer, augs, tok, bests, args)
        print(f"Done processing image: {img_path}")


if __name__ == "__main__":
    main()