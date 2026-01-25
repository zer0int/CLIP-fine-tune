"""
zer0int 2026 ~ github.com/zer0int/CLIP-fine-tune
______________________________________________________________

Simple Feature Activation Max Visualization.

Adapted from this original CLIP feature activation max visualization code:
https://github.com/hamidkazemi22/vit-visualization

Uses 'Sophia' stochastic second-order optimizer:
https://github.com/Liuhong99/Sophia

"""
import os
import random
import collections
import argparse
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
from opt_sophia.sophia import SophiaG
from torchvision.transforms import Resize
from safetensors.torch import load_file
from colorama import Fore, Style

import oaiclip as clip
from oaiclip.model import QuickGELU
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

from utils_clip_loader import cliptools
cliptools.configure(clip)

# Custom imports for Visualization
from utils_clip_loader.cliptools import LossArray, TotalVariation, TruePatchCorrelationLoss
from utils_clip_loader.cliptools import ViTEnsFeatHook
from utils_clip_loader.cliptools import new_init
from utils_clip_loader.cliptools import GaussianNoise
from utils_clip_loader.cliptools import TileGPT as Tile
from utils_clip_loader.cliptools import ColorJitter
from utils_clip_loader.cliptools import ClipGeLUHook
from utils_clip_loader.cliptools import Clip, Jitter, RepeatBatch
from utils_clip_loader.cliptools import ClipViTWrapper as ClipWrapper
from utils_clip_loader.cliptools import save_intermediate_step, save_image, fix_random_seed

# Suppress warnings spam from torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)




def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP MLP feature activation max visualization')
    parser.add_argument('--use_model', default="zer0int/CLIP-Regression-ViT-L-14", help="OpenAI model, local path .pt or .safetensors, HuggingFace Hub")
    parser.add_argument('--clipname', default="regression", help="Custom name, use for filename tagging")
    parser.add_argument('--layer_range', default="6, 10, 12, 19, 23", type=str, help="Which layers to visualize; continuous range ('5-10') or discrete values ('5,6,8').")
    parser.add_argument('--feature_range', default="123, 4065", type=str, help="Which MLP features to visualize.")
    parser.add_argument("--output_folder", default='out_vis_features', help="Folder to save output image; default: FeatureViz/NORMAL-MLP")
    parser.add_argument('--steps', default=400, type=int, help="Number of image optimization steps; default: 400")
    parser.add_argument('--lr', default=1.0, type=float, help="Learning Rate; default: 1.0")
    # Note: Below is applied twice, second time: patch_stride *2; CTRL+F for: loss += TruePatchCorrelationLoss 
    parser.add_argument('--patch_kernel', default=14, type=int, help="Patch Correlation Loss Kernel")
    parser.add_argument('--patch_stride', default=14, type=int, help="Patch Correlation Stride")
    parser.add_argument('--patch_coeff', default=0.02, type=float, help="Patch Correlation Coefficient (strength)")
    parser.add_argument('--coeff', default=0.0000005, type=float, help="Coefficient for Total Variation Loss")
    parser.add_argument("--repeat_batch", default=8, type=int, help="Quality for Color Jitter etc. (Repeated Batches). Less = Lower VRAM usage. Default: 8")
    parser.add_argument("--save_intermediate", action='store_true', help="Save intermediate steps, too, for a quick look. Saves to folder: 'steps'")
    parser.add_argument("--fast", action='store_true', help="FAST: torch.set_float32_matmul_precision('medium') - NVIDIA Ampere or newer GPUs only") 
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior")
    return parser.parse_args()

args = parse_arguments()
steps_folder = args.output_folder
os.makedirs(steps_folder, exist_ok=True)
repeats = args.repeat_batch
iterations = args.steps

clipmodel = args.use_model
clipname = args.clipname


if args.fast:
    torch.set_float32_matmul_precision("medium")

if args.deterministic:
    fix_random_seed()

class ImageNetVisualizer:
    def __init__(self, loss_array: LossArray, pre_aug: nn.Module = None,
                 post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, save_every: int = 200, saver: bool = True,
                 print_every: int = 5, **_):
        self.loss = loss_array
        self.saver = saver
        print(self.saver)

        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None, layer: int = None, feature: int = None, clipname: str = None):
        if not img.is_cuda or img.device != torch.device('cuda:0'):
            img = img.to('cuda:0')
        if not img.requires_grad:
            img.requires_grad_()
            
        #optimizer = optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8) # prefer SophiaG
        optimizer = SophiaG([img], lr=self.lr, betas=(0.98, 0.99), rho=0.0, weight_decay=0.0)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{self.loss.header()}', flush=True)

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f'{i}\t{self.loss}', flush=True)
            if i % self.save_every == 0 and self.saver is True:
                save_intermediate_step(img, i, layer, feature, clipname)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.update_hessian() 

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()

        optimizer.state = collections.defaultdict(dict)
        return img


def get_clip_dimensions(model, preprocess):
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            break
    num_layers = None
    num_features = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features

    return input_dims, num_layers, num_features


def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    model, preprocess, _ = load_openai_clip_anything(clip, clipmodel, device=device, jit=False, strict=True)
    premodel = model
    model = ClipWrapper(model).to(device).float()
    return model, premodel, preprocess


def parse_range(range_str):
    out = []
    seen = set()

    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        it = range(start, end + 1)
    else:
        it = map(int, range_str.split(','))

    for x in it:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def generate_visualizations(model, premodel, clipname, layer_range_str, feature_range_str, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, args):
    layer_range = parse_range(layer_range_str)
    feature_range = parse_range(feature_range_str)
   
    for layer in layer_range:
        for feature in feature_range:
            print(Fore.MAGENTA + Style.BRIGHT + f"Generating visualization for Layer {layer}, Feature {feature}..." + Fore.RESET)
            loss = LossArray()
            loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
            if args.coeff != 0.0:
                loss += TotalVariation(2, image_size, coefficient * tv)
            if args.patch_coeff != 0.0:
                loss += TruePatchCorrelationLoss(args.patch_kernel, args.patch_stride, args.patch_coeff, 256)
                loss += TruePatchCorrelationLoss(args.patch_kernel, args.patch_stride*2, args.patch_coeff, 256)

                
            pre, post = torch.nn.Sequential(RepeatBatch(repeats), ColorJitter(repeats, shuffle_every=True),
                                            GaussianNoise(repeats, True, 0.5, iterations), Tile(image_size // image_size), Jitter()), Clip()
            image = new_init(image_size, 1)
            
            visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient)
            image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

            save_image(image, f'{steps_folder}/{clipname}_L{layer}_F{feature}.png')

def main():
    args = parse_arguments()

    model, premodel, preprocess = load_clip_model()
    input_dims, num_layers, num_features = get_clip_dimensions(premodel, preprocess)
    image_size = input_dims
    print(f"\nSelected input dimension for {clipmodel}:" + Fore.GREEN + Style.BRIGHT + f" {input_dims}" + Fore.RESET)
    print(f"Layers:" + Fore.GREEN + Style.BRIGHT + f"0-{num_layers-1} with 0-{num_features-1} Features / Layer" + Fore.RESET)

    layer_range_str = args.layer_range
    feature_range_str = args.feature_range
    
    tv = 1.0
    lr = args.lr
    coefficient=args.coeff

    steps = args.steps
    print_every = 10
    save_every = 10

    saver = False
    if args.save_intermediate:
        saver = True

    generate_visualizations(model, premodel, clipname, layer_range_str, feature_range_str, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, args)

if __name__ == '__main__':
    main()