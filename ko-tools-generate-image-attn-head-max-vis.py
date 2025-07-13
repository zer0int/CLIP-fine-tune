import os
import random
import collections
import argparse
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from colorama import Fore, Style
from safetensors.torch import load_file

import attnclip as clip
from attnclip.model import convert_state_dict_inproj_to_qkv
from attnclip.model import QuickGELU

# Custom imports
from cliptools import LossArray, TotalVariation
from cliptools import ViTFeatHook, ViTEnsFeatHook
from cliptools import ClipGeLUHook
from cliptools import Clip, Jitter, RepeatBatch
from cliptools import TileGPT as Tile
from cliptools import ColorJitterGPT as ColorJitter
from cliptools import ClipViTWrapper as ClipWrapper
from cliptools import save_intermediate_step, save_image, fix_random_seed

# Suppress warnings spam from torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------------------------------------------------------
# Generates images that maximize activations of specific attention heads 
# in a CLIP ViT model, layer-by-layer, using gradient ascent.
#
# Basically like Feature Activation Max Visualization,
# but instead we slice activations per attention head and explicitly optimize
# *for* a single head while pushing *against* all other heads.
# -> Resulting images reveal unique max salience visual features for a head.
#
# For example, Head idx 7 is a 'text salience head' (seen from Layer 8++),
# Head 4 is a 'portraits of people head' (best seen from layer 15 onward).
# -----------------------------------------------------------------------------


# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Activation Max Visualization, optim towards individual Attention Heads')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--head_range', default="0-15", type=str, help="Which layers to visualize; continuous range ('0-2') or discrete values ('5,6,8'). Default: 0-15")
    parser.add_argument('--layer_range', default="0-23", type=str, help="Which layers to visualize; continuous range ('5-10') or discrete values ('5,6,8'). Default: 14-23")
    parser.add_argument('--feature_range', default="0", type=str, help="Which features to visualize; continuous range ('50-90') or discrete values ('500,1000)'. Default: 0-10") # Irrelevant when max. for attn head, leave at 0
    parser.add_argument('--steps', default=400, type=int, help="Number of image optimization steps; default: 400")
    parser.add_argument('--lr', default=1.0, type=float, help="Learning Rate; default: 1.0")
    parser.add_argument('--tv', default=1.0, type=float, help="Total Variation Loss; default: 1.0")
    parser.add_argument('--coeff', default=0.00005, type=float, help="For tv*coeff. 0.00005 -> sharp and noisy image; 0.05 -> soft, blurry; default: 0.00005")
    parser.add_argument("--output_folder", default='results_oai/FeatureViz/SingleHead', help="Folder to save output image; default: FeatureViz/Single_Head")
    parser.add_argument("--save_intermediate", action='store_true', help="Save intermediate steps, too, for a quick look. Saves to folder: 'steps'")
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior")
    return parser.parse_args()

args = parse_arguments()
steps_folder = args.output_folder
os.makedirs(steps_folder, exist_ok=True)

clipmodel = args.use_model
clipname = os.path.basename(clipmodel)
clipname = os.path.splitext(clipname)[0]
clipname = clipname.replace("@", "-").replace("/", "").replace("_backtoweight", "")

if args.deterministic:
    fix_random_seed()
    from cliptools import new_init_rnd as new_init # randomness ensues!
    from cliptools import GaussianNoiseGPTRnd as GaussianNoise # randomness ensues!
else:
    from cliptools import new_init # not random
    from cliptools import GaussianNoiseGPT as GaussianNoise # not random

class HeadCaptureHook:
    def __init__(self, block, head_idx):
        self.activations = None
        self.head_idx = head_idx
        self.hook_handle = block.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # output: [seq_len, batch, embed_dim]
        seq_len, batch, embed_dim = output.shape
        num_heads = module.attn.num_heads
        head_dim = embed_dim // num_heads
        x = output.permute(1, 0, 2).contiguous()  # [batch, seq, embed_dim]
        x = x.view(batch, seq_len, num_heads, head_dim)  # [batch, seq, heads, head_dim]
        self.activations = x[:, :, self.head_idx, :]  # [batch, seq, head_dim]

    def clear(self):
        self.activations = None

class AllHeadsCaptureHook:
    def __init__(self, block):
        self.activations = None
        self.hook_handle = block.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # output: [seq_len, batch, embed_dim]
        seq_len, batch, embed_dim = output.shape
        num_heads = module.attn.num_heads
        head_dim = embed_dim // num_heads
        x = output.permute(1, 0, 2).contiguous()  # [batch, seq, embed_dim]
        x = x.view(batch, seq_len, num_heads, head_dim)  # [batch, seq, heads, head_dim]
        self.activations = x  # [batch, seq, num_heads, head_dim]

    def clear(self):
        self.activations = None


class ImageNetVisualizer:
    def __init__(self, model, loss_array: LossArray, target_layer, target_head, head_hook, all_heads_hook,
                 pre_aug: nn.Module = None, post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1,
                 save_every: int = 200, saver: bool = True, print_every: int = 5, **_):
        self.loss = loss_array
        self.model = model  # <--- Save model!
        self.saver = saver
        self.target_layer = target_layer
        self.target_head = target_head
        self.head_hook = head_hook
        self.all_heads_hook = all_heads_hook
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

        # ---- Other optimizers to consider using instead of Adamax ----
        #optim.Adam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        #optim.AdamW([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        #optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        #optim.RMSprop([img], lr=self.lr)
        #optim.Adagrad([img], lr=self.lr)
        #optim.RAdam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        #optim.NAdam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        # ---------------------------------------------------------------


        optimizer = optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{"Head_Loss, TV, Total"}', flush=True)

    
        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            _ = self.model(augmented)  # forward triggers hook

            acts_all = self.all_heads_hook.activations  # [batch, seq, num_heads, head_dim]
            if acts_all is None:
                raise RuntimeError("No activations from all heads hook")

            target_head_acts = acts_all[:, :, self.target_head, :]      # [batch, seq, head_dim]
            other_heads_mask = [i for i in range(acts_all.shape[2]) if i != self.target_head]
            other_acts = acts_all[:, :, other_heads_mask, :]  # [batch, seq, num_heads-1, head_dim]

            # Compute mean L2 norm for head 10 and for all others
            target_head_norm = target_head_acts.norm(dim=-1).mean()
            other_heads_norm = other_acts.norm(dim=-1).mean()

            # **The key difference loss:**
            attn_loss = -(target_head_norm - other_heads_norm)  # minus for gradient ascent
            # TV regularization
            tv_loss = 0
            if hasattr(self.loss, "__call__"):
                try:
                    tv_loss = self.loss(augmented)
                except Exception:
                    tv_loss = 0

            total_loss = attn_loss + tv_loss

            if i % self.print_every == 0:
                print(f'{i}\t{attn_loss:.3f}\t{tv_loss:.3f}\t{total_loss:.3f}', flush=True)
            if i % self.save_every == 0 and self.saver is True:
                save_intermediate_step(img, i, layer, feature, clipname)


            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

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
    num_heads = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features
        if hasattr(last_block, 'attn'):  # <-- Add for num_heads
            num_heads = last_block.attn.num_heads

    return input_dims, num_layers, num_features, num_heads

def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    if clipmodel.endswith(".safetensors"):
        print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
        model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
        state_dict = load_file(model_name_or_path)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            msg = str(e)
            if (
                "Missing key(s) in state_dict" in msg or
                "Unexpected key(s) in state_dict" in msg
            ):
                print("State dict format mismatch, attempting QKV conversion...")
                state_dict = convert_state_dict_inproj_to_qkv(state_dict)
                try:
                    model.load_state_dict(state_dict)
                    print("OK!")
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load state_dict after QKV conversion: {e2}\n"
                        f"Original error: {e}"
                    )
            else:
                raise

    else:
        print("Detected non-.safetensors file or name. Attempting to load model...")
        model, preprocess = clip.load(clipmodel, device=device, jit=False)
     
    premodel = model
    model = ClipWrapper(model).to(device).float()
    return model, premodel, preprocess

def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return list(map(int, range_str.split(',')))

def clamp_range(vals, minval, maxval):
    # vals: list[int]
    return [max(min(val, maxval), minval) for val in vals]

def generate_visualizations(model, premodel, clipname, layer_range, feature_range, head_range, image_size, tv, lr, steps, print_every, save_every, saver, coefficient):
    for layer in layer_range:
        for head in head_range:  # <-- Added
            block = premodel.visual.transformer.resblocks[layer]
            head_hook = HeadCaptureHook(block, head)
            all_heads_hook = AllHeadsCaptureHook(block)
            for feature in feature_range:
                print(Fore.MAGENTA + Style.BRIGHT + f"Generating visualization for Layer {layer}, Head {head}, Feature {feature}..." + Fore.RESET)
                loss = LossArray()
                loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
                loss += TotalVariation(2, image_size, coefficient * tv)

                pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                                GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
                image = new_init(image_size, 1)

                visualizer = ImageNetVisualizer(model, loss_array=loss, target_layer=layer, target_head=head,
                                                head_hook=head_hook, all_heads_hook=all_heads_hook,
                                                pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps,
                                                save_every=save_every, saver=saver, coefficient=coefficient)
                image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

                save_image(image, f'{steps_folder}/{clipname}_H{head}_L{layer}_F{feature}.png')


def main():
    args = parse_arguments()

    model, premodel, preprocess = load_clip_model()
    input_dims, num_layers, num_features, num_heads = get_clip_dimensions(premodel, preprocess)
    image_size = input_dims
    print(f"\nSelected input dimension for {clipmodel}:" + Fore.GREEN + Style.BRIGHT + f" {input_dims}" + Fore.RESET)
    print(f"Layers:" + Fore.GREEN + Style.BRIGHT + f"0-{num_layers-1} with 0-{num_features-1} Features / Layer, 0-{num_heads-1} Attn Heads" + Fore.RESET)

    # Parse ranges
    layer_range = parse_range(args.layer_range)
    feature_range = parse_range(args.feature_range)
    head_range = parse_range(args.head_range)
    # Clamp
    layer_range = clamp_range(layer_range, 0, num_layers-1)
    feature_range = clamp_range(feature_range, 0, num_features-1)
    head_range = clamp_range(head_range, 0, num_heads-1)
    
    tv = args.tv
    lr = args.lr
    coefficient=args.coeff

    steps = args.steps
    print_every = 10
    save_every = 10

    saver = False
    if args.save_intermediate:
        saver = True

    generate_visualizations(model, premodel, clipname, layer_range, feature_range, head_range, image_size, tv, lr, steps, print_every, save_every, saver, coefficient)
    print(f"All done! Check out '{steps_folder}' !")

if __name__ == '__main__':
    main()