"""
Credits to the original author for most of these: 
https://github.com/hamidkazemi22/vit-visualization

A few additions taken from 
https://github.com/stanislavfort/Direct_Ascent_Synthesis

Additional changes and further tools by zer0int:
https://github.com/zer0int

Iteration V4 - 24/August/2025

Update 01/2026: Add configure
"""
from __future__ import annotations
import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data
from torch.nn import ReLU, GELU
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from typing import Optional, List, Any, Callable
import torchvision.utils
import torchvision
import datetime
import contextlib
import math

random_seed=6247423
_nums = '0123456789'

clip: Optional[Any] = None
QuickGELU: Optional[Callable[..., Any]] = None

def configure(clip_module: Any) -> None:
    """
    Call this once after you import the 'active' CLIP module (attnclipindiv, clip, etc).
    """
    global clip, QuickGELU
    clip = clip_module
    QuickGELU = clip_module.model.QuickGELU
    
def _require_configured() -> None:
    if clip is None or QuickGELU is None:
        raise RuntimeError(
            "cliptools is not configured. Call cliptools.configure(clip_module) "
            "before using hooks that need clip/tokenize/QuickGELU."
        )

# Model
# ------------------------
def save_model_dtypes(model):
    """Stores original dtypes for all parameters in a dictionary."""
    return {k: v.dtype for k, v in model.state_dict().items()}

def convert_model_to_full_precision(model):
    """Converts all model parameters to float32 for stable computation."""
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    return model

def restore_model_dtypes(model, original_dtypes):
    """Restores original dtypes for all parameters from saved state."""
    for k, v in model.state_dict().items():
        expected_dtype = original_dtypes[k]
        model.state_dict()[k].data = v.to(expected_dtype)

def _abbreviation(name: str) -> str:
    if len(name) <= 3:
        return name
    abr = ''.join(x for x in name if x.isupper() or x in _nums)
    return abr[:3]

def freeze_module(module: nn.Module, reverse=False):
    for param in module.parameters():
        param.requires_grad = reverse

def get_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

def zero_grad(image):
    if image.grad is not None:
        if image.grad.grad_fn is not None:
            image.grad.detach_()
        else:
            image.grad.requires_grad_(False)
        image.grad.data.zero_()

def normalize_for_clip(x, mean, std):
    return (x - torch.Tensor(mean).reshape([1, 3, 1, 1]).to("cuda")) / torch.Tensor(std).reshape([1, 3, 1, 1]).to("cuda")

def _round(num: float) -> str:
    if num > 100:
        return str(int(round(num, 0)))
    if num > 10:
        return str(round(num, 1))
    return str(round(num, 2))

def fix_random_seed(seed: int = random_seed):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def params_num(module: nn.Module):
    return len(list(get_trainable_params(module)))

def get_trainable_params(module: nn.Module):
    trainable_params = filter(lambda p: p.requires_grad, module.parameters())
    return trainable_params

def get_clip_vit_dimensions(model, preprocess):
    model = model.eval()
    input_dims = None
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            break

    num_layers, num_features = None, None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features

    return input_dims, num_layers, num_features

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

# Image Tensor
# ------------------------

# === Color utilities (sRGB -> Lab) ===
def _srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _rgb_to_xyz(x):
    # x in [0,1], shape [B,3,H,W], sRGB, D65
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    return torch.cat([X, Y, Z], dim=1)

def _f_lab(t):
    eps, kappa = 216/24389, 24389/27
    return torch.where(t > eps, t.pow(1/3), (kappa*t + 16)/116)

def rgb_to_lab(x):
    # clamp to [0,1] for stability
    x = x.clamp(0, 1)
    xyz = _rgb_to_xyz(x)
    # White point D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x_, y_, z_ = xyz[:,0:1]/Xn, xyz[:,1:2]/Yn, xyz[:,2:3]/Zn
    fx, fy, fz = _f_lab(x_), _f_lab(y_), _f_lab(z_)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.cat([L, a, b], dim=1)  # [B,3,H,W]

def gaussian_blur_2d(x: torch.Tensor, sigma: float = 3.0, ksize: int | None = None) -> torch.Tensor:
    if ksize is None:
        ksize = int(max(3, 2*round(3*sigma)+1))
    ax = torch.arange(ksize, device=x.device, dtype=x.dtype) - (ksize - 1) / 2
    g1d = torch.exp(-(ax**2) / (2 * sigma * sigma))
    g1d = g1d / (g1d.sum() + 1e-12)

    C = x.size(1)
    g_h = g1d.view(1,1,1,ksize).expand(C,1,1,ksize)
    g_v = g1d.view(1,1,ksize,1).expand(C,1,ksize,1)

    # reflect-pad instead of zero-pad
    pad = ksize // 2
    x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, g_h, padding=0, groups=C)
    x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="reflect")
    x = torch.nn.functional.conv2d(x, g_v, padding=0, groups=C)
    return x

def edge_gate_from_luminance(x, sigma=1.0, ksize=None):
    # x: [B,3,H,W] in [0,1] or [-1,1]
    L = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)[:,0:1]
    # gradient magnitude (Sobel)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = torch.nn.functional.conv2d(L, kx, padding=1)
    gy = torch.nn.functional.conv2d(L, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
    mag = gaussian_blur_2d(mag, sigma=sigma, ksize=ksize)
    gate = torch.exp(-5.0 * mag)  # ∈ (0,1], small near edges
    return gate

# === Raw to Real / Real to Raw 

def raw_to_real_image(raw_image):
    return (torch.tanh(raw_image) + 1.0) / 2.0

def real_to_raw_image(real_image, eps=1e-5):
    return torch.arctanh(torch.clip(real_image, eps, 1 - eps) * 2.0 - 1.0)

def gray_scale(image):
    return torch.mean(image, dim=1, keepdim=True)

def new_init(size: int, batch_size: int = 1, last: torch.nn = None, padding: int = -1, zero: bool = False) -> torch.nn:
    # OPT: Use pinned memory and non-blocking transfer for faster GPU upload.
    output = torch.rand(size=(batch_size, 3, size, size), pin_memory=True) if not zero else torch.zeros(size=(batch_size, 3, size, size), pin_memory=True)
    output = output.to('cuda:0', non_blocking=True)
    if last is not None:
        big_size = size if padding == -1 else size - padding
        up = torch.nn.Upsample(size=(big_size, big_size), mode='bilinear', align_corners=False).cuda()
        scaled = up(last)
        cx = (output.size(-1) - big_size) // 2
        output[:, :, cx:cx + big_size, cx:cx + big_size] = scaled
    output = output.detach().clone()
    output.requires_grad_()
    return output

def new_init_rnd(
    size: int,
    batch_size: int = 1,
    last: torch.Tensor = None,
    padding: int = -1,
    zero: bool = False,
    seed: int = None,
    device: str = 'cuda:0'
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    dtype = torch.float32
    shape = (batch_size, 3, size, size)
    output = (
        torch.rand(shape, dtype=dtype, pin_memory=True)
        if not zero else
        torch.zeros(shape, dtype=dtype, pin_memory=True)
    )
    output = output.to(device, non_blocking=True)
    if last is not None:
        big_size = size if padding == -1 else size - padding
        up = torch.nn.Upsample(size=(big_size, big_size), mode='bilinear', align_corners=False).to(device)
        scaled = up(last)
        cx = (output.size(-1) - big_size) // 2
        output[:, :, cx:cx + big_size, cx:cx + big_size] = scaled
    output = output.detach().clone()
    output.requires_grad_()
    return output

def save_intermediate_step(tensor: torch.Tensor, step: int, layer: int, feature: int, clipname: str, base_path: str):
    """
    Saves an intermediate step image during visualization.

    Parameters:
    - tensor: A torch.Tensor object. Expected shape [1, C, H, W].
    - step: An integer, the current optimization step.
    - layer: An integer, the current layer being visualized.
    - feature: An integer, the specific feature within the layer being targeted.
    - base_path: A string, the base directory to save the images.
    """
    import os
    import torchvision.utils

    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)

    # Construct the filename
    base_path = f'{base_path}/{clipname}_L{layer}-F{feature}/'
    os.makedirs(base_path, exist_ok=True)
    filename = f'step{step}.png'
    filepath = os.path.join(base_path, filename)

    # If the tensor has a batch dimension, remove it
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Normalize the tensor to [0, 1] if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Save the image
    torchvision.utils.save_image(tensor, filepath)

def save_image(tensor: torch.Tensor, path: str):
    """
    Saves a tensor as an image.

    Parameters:
    - tensor: A torch.Tensor object. Expected shape [C, H, W] or [1, C, H, W].
    - path: A string, the path where the image will be saved.
    """
    # If the tensor has a batch dimension, remove it
    #os.makedirs(save_path, exist_ok=True)
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Normalize the tensor to [0, 1] if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Save the image
    torchvision.utils.save_image(tensor, path)

def make_image(image: torch.Tensor):
    batch_size, c, h, w = image.shape
    flattened = image.view(batch_size, -1)
    batch_min, batch_max = torch.min(flattened, 1, keepdim=True)[0], torch.max(flattened, 1, keepdim=True)[0]
    flattened -= batch_min
    flattened /= torch.clamp(batch_max - batch_min, min=1e-5)
    return flattened.view(batch_size, c, h, w)

def _get_gelu_module(premodel, layer_idx):
    mlp = premodel.visual.transformer.resblocks[layer_idx].mlp
    for m in mlp.modules():
        if isinstance(m, QuickGELU) or isinstance(m, nn.GELU):
            return m
    raise RuntimeError(f"No GELU in block {layer_idx}")

class _ViTTokenTap:
    def __init__(self, premodel, layers):
        self.model = premodel
        self.layers = list(sorted(set(layers)))
        self.buff = {}
        self.handles = []
        for l in self.layers:
            gelu = _get_gelu_module(self.model, l)
            self.handles.append(
                gelu.register_forward_hook(self._make_hook(l))
            )
    def _make_hook(self, l):
        def hook(_m, _inp, out):
            self.buff[l] = out  # [B, seq, 4*width], no detach
        return hook
    def encode(self, x):
        self.buff.clear()
        _ = self.model.encode_image(x)
        return self.buff
    def close(self):
        for h in self.handles: h.remove()
        self.handles = []


def _get_grid_hw(premodel):
    patch = premodel.visual.conv1.kernel_size[0]
    inp = premodel.visual.input_resolution
    g = inp // patch
    return g, g

def _load_patchmap_for_block(base_dir, class_name, block_idx):
    p = os.path.join(base_dir, f"uniform_{class_name}_block{block_idx}_patchmap.npy")
    return np.load(p) if os.path.exists(p) else None

# Datasets
# ------------------------

def get_loaders(batch_size=256, n_workers=4, dataset_name='cifar10', return_dataset=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.CIFAR100
    train_dataset = dataset(f'data/datasets/{dataset_name}', download=True,
                            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=n_workers)
    test_dataset = dataset(f'data/datasets/{dataset_name}', download=True, train=False,
                           transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=n_workers)
    if return_dataset:
        return train_loader, test_loader, train_dataset, test_dataset
    return train_loader, test_loader

def get_imagenet(batch_size=256, n_workers=4, path='data/datasets/ILSVRC2012/{}', shuffle=True, modeldims=224):
    train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(modeldims),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), ])

    eval_transforms = transforms.Compose(
        [transforms.Resize(modeldims + 32),
         transforms.CenterCrop(modeldims),
         transforms.ToTensor(), ])

    train_dataset = datasets.ImageFolder(root=path.format('train'),
                                         transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=path.format('val'),
                                        transform=eval_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=n_workers, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              num_workers=n_workers, shuffle=True, pin_memory=True)
    return train_loader, test_loader

# ---------------------------------
#        Adversarial / PGD
# ---------------------------------

def train_step(loader, model_md, loss_fn, opt, epoch_n, scheduler=None, normal_fn=None,
               modify_fn=None, file=None):
    model_md.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for i, (image, label) in enumerate(loader):
        image = image.cuda()
        label = label.cuda()
        opt.zero_grad()
        image = modify_fn(image, label) if modify_fn else image
        image = normal_fn(image) if normal_fn else image
        output = model_md(image)
        preds = torch.argmax(output, -1)
        loss = loss_fn(output, label)
        loss.backward()
        opt.step()
        running_loss += loss.item() * image.shape[0]
        running_corrects += torch.sum(preds == label)
        total += image.shape[0]
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        end = '\n' if i == (len(loader) - 1) else '\r'
        print(f'epoch: {epoch_n:04d}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, {i + 1:04d}/{len(loader)}',
              end=end)
    scheduler.step() if scheduler else None

def test_step(test_loader, model, loss_fn, normal_fn=None, modify_fn=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for i, (image, label) in enumerate(test_loader):
        image = image.cuda()
        label = label.cuda()
        image = modify_fn(image, label) if modify_fn else image
        image = normal_fn(image) if normal_fn else image
        with torch.no_grad():
            output = model(image)
            loss = loss_fn(output, label)
        preds = torch.argmax(output, 1)
        running_loss += loss.item() * image.shape[0]
        running_corrects += torch.sum(preds == label.data)
        total += image.shape[0]
        end = '\n' if i == (len(test_loader) - 1) else '\r'

        loss = running_loss / total
        accuracy = running_corrects.double() / total
        print((
            f'Test Loss: {loss:.4f} Test Acc: {accuracy:.4f}, {i + 1:02d}/{len(test_loader)}'),
            end=end)
    accuracy = running_corrects.double() / total
    return accuracy

def adv_test_step(test_loader, model, loss_fn, revertor=None, normal_fn=None,
                  modify_fn=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for i, (image, label) in enumerate(test_loader):
        image = image.cuda()
        label = label.cuda()
        # if i == 0:
        #     plt.figure()
        #     im = image[0].detach().cpu().numpy()
        #     plt.imshow(np.moveaxis(im, 0, -1))
        #     plt.savefig('images/asli.png')
        image = modify_fn(image, label) if modify_fn else image
        if revertor is not None:
            image = image + revertor[label]
        # import matplotlib.pyplot as plt
        # if i == 0:
        #     plt.figure()
        #     im = image[0].detach().cpu().numpy()
        #     plt.imshow(np.moveaxis(im, 0, -1))
        #     plt.savefig('images/adv.png')
        image = normal_fn(image) if normal_fn else image
        with torch.no_grad():
            output = model(image)
            loss = loss_fn(output, label)
        preds = torch.argmax(output, 1)
        running_loss += loss.item() * image.shape[0]
        running_corrects += torch.sum(preds == label.data)
        total += image.shape[0]
        end = '\n' if i == (len(test_loader) - 1) else '\r'

        loss = running_loss / total
        accuracy = running_corrects.double() / total
        print((
            f'Test Loss: {loss:.4f} Test Acc: {accuracy:.4f}, {i + 1:02d}/{len(test_loader)}'),
            end=end)
    accuracy = running_corrects.double() / total
    return accuracy

def make_pgd(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn, label, eps, step_size=2 / 255,
             iters=10):
    # copy_model = copy.deepcopy(model)
    copy_model = model
    copy_model.eval()
    copy_image = image.detach().clone()
    # freeze_module(copy_model)
    copy_image.requires_grad = True
    for step in range(iters):
        output = normal_fn(copy_image)
        output = copy_model(output)
        loss = loss_fn(output, label)
        loss.backward()
        adv_image = copy_image + step_size * copy_image.grad.sign()
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image.data = torch.clamp(image.data + perturb.data, 0, 1)

    # del copy_model
    return copy_image

def make_pgd_v2(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn, label, eps, step_size=2 / 255,
                iters=10):
    model.eval()
    # freeze_module(model)
    copy_image = image.detach().clone()
    copy_image.requires_grad = True
    for step in range(iters):
        output = normal_fn(copy_image)
        output = model(output)
        loss = loss_fn(output, label)
        loss.backward()
        adv_image = copy_image + step_size * copy_image.grad.sign()
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image.data = torch.clamp(image.data + perturb.data, 0, 1)
    return copy_image

def make_target_pgd(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn,
                    target_label, eps, iters=10):
    copy_model = copy.deepcopy(model)
    copy_image = image.clone().detach()
    freeze_module(copy_model)
    for step in range(iters):
        copy_image.requires_grad = True
        output = normal_fn(copy_image)
        output = copy_model(output)
        loss = loss_fn(output, target_label)
        loss.backward()
        adv_image = copy_image - eps * copy_image.grad.sign()
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image = image + perturb
        copy_image.detach_()
        copy_image.clamp_(0, 1)

    del copy_model
    return copy_image

def make_adv(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn, label, eps,
             lr=0.1):
    copy_model = copy.deepcopy(model)
    copy_image = image.clone().detach()
    freeze_module(copy_model)
    for step in range(10):
        copy_image.requires_grad = True
        output = normal_fn(copy_image)
        output = copy_model(output)
        loss = loss_fn(output, label)
        loss.backward()
        adv_image = copy_image + lr * copy_image.grad
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image = image + perturb
        copy_image.detach_()
        copy_image.clamp_(0, 1)

    del copy_model
    return copy_image

# ---------------------------------
#             BASE HOOKS
# ---------------------------------

class ClipViTWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(x)

class ClipOVViTWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom OpenVision CLIP forward pass, otherwise entangled in open_clip
        x = self.clip.visual.conv1(x)                                           # [B, C, H, W] → [B, width, gh, gw]
        x = x.reshape(x.shape[0], x.shape[1], -1)                               # [B, width, N]
        x = x.permute(0, 2, 1)                                                  # [B, N, width]

        # Add class token + positional embedding
        cls_token = self.clip.visual.class_embedding.expand(x.shape[0], 1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)                                    # [B, N+1, D]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)               # [B, N+1, D]

        # Dropout + LN
        #x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)

        # Transformer
        x = self.clip.visual.transformer(x)

        return x

class ClipTxTWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(x)

class ClipNeuronCaptureHook:
    def __init__(self, module: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = None
        self.top_value = None
        self.top_index = None
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.detach()

    def get_top_neuron(self):
        if self.activations is not None:
            # Flatten across all non-batch dimensions
            batch_size, *feature_dims = self.activations.shape  # Example: (batch, 3072)
            flat_activations = self.activations.view(batch_size, -1)  # Shape: (batch, total_features)

            # Get the **true** max activation across all features
            top_value, flat_index = torch.max(flat_activations, dim=-1)  # Max per batch

            # Convert flat index back to original feature space
            top_index = flat_index[0]  # Get index for first batch element

            self.top_value = top_value[0].item()
            self.top_index = top_index.item()  # This is the true feature index

            return self.layer_idx, self.top_value, self.top_index
        return None, None, None

    def remove(self):
        self.hook_handle.remove()

class ItemIterator:
    @property
    def iterator_item(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.iterator_item)

    def __getitem__(self, item):
        print(self.iterator_item)
        return self.iterator_item[item]

    def __len__(self):
        return len(self.iterator_item)

class HookHolder(ItemIterator):
    def __init__(self, classifier: nn.Module, hook_class, layer_class):
        self.hooks = [hook_class(m) for m in classifier.modules() if isinstance(m, layer_class)]

    @property
    def iterator_item(self):
        return self.hooks

    def check_for_attr(self, attr: str, hook_class):
        for h in self:
            if not hasattr(h, attr):
                raise AttributeError('Class {} does not have attribute {}'.format(hook_class.__name__, attr))

    def _broadcast(self, func_name: str, *to_propagate):
        for i in self:
            func = getattr(i, func_name)
            func(*to_propagate)

    def _gather(self, attr: str) -> list:
        return [getattr(l, attr) for l in self]

    def close(self):
        self._broadcast('close')

class TimedHookHolder(HookHolder):
    def __init__(self, classifier: nn.Module, hook_class, layer_class, use_fixed_random_seed: bool = False):
        super().__init__(classifier, hook_class, layer_class)
        if use_fixed_random_seed:
            fix_random_seed()
    def get_activations(self):
        all_values = []
        for h in self.hooks:
            all_values += h.activations
        return all_values

    def get_layer(self, item):
        all_values = sorted(self.get_activations())
        return all_values[item][1]

    def set_seed(self, seed: int):
        self._broadcast('set_seed', seed)

    def set_target(self, target: list):
        self._broadcast('set_target', target)

    def reset(self):
        all_values = self.get_activations()
        all_values = sum([v.sum() for _, v in all_values])
        self._broadcast('reset')
        return all_values

class BasicHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.base_hook_fn)
        self.activations = None

    def close(self):
        self.hook.remove()

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t
        x = x[0][0] if isinstance(x[0], tuple) else x[0]
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

class ViTHook(BasicHook):
    def __init__(self, module: nn.Module, return_output: bool, name: str):
        super().__init__(module)
        self.mode = return_output
        self.name = name

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t if not self.mode else output_t
        x = x[0] if isinstance(x, tuple) else x
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        self.activations = x

class LayerHook:
    def __init__(self, classifier: nn.Module, layer_class, layer_depth: int, hook_cls):
        self.layer = [m for m in classifier.modules() if isinstance(m, layer_class)][layer_depth]
        self.hook = hook_cls(self.layer)

    def __call__(self) -> torch.tensor:
        return self.hook()

class FakeHookWrapper:
    def __init__(self, value):
        self.activations = value

class ViTAbsHookHolder(nn.Module):
    pass

class Scale(nn.Module):
    def __init__(self, size, mode='bicubic'):
        super(Scale, self).__init__()
        self.mode = mode
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=(self.size, self.size), mode=self.mode)
# ---------------------------------
#             BASE LOSS
# ---------------------------------

# =========================
# InvLoss / LossArray patch
# =========================

class InvLossSophiaViz:
    def __init__(self, coefficient: float = 1.0):
        self.c = coefficient
        self.name = _abbreviation(self.__class__.__name__)
        self.last_value = 0

    def __call__(self, x: torch.tensor) -> torch.tensor:
        tensor = self.loss(x)
        # CHANGED: record stats via detach() to avoid any odd autograd edges
        self.last_value = float(tensor.detach().item())  # CHANGED
        return self.c * tensor

    def loss(self, x: torch.tensor):
        raise NotImplementedError

    def __str__(self):
        return f'{_round(self.c * self.last_value)}({_round(self.last_value)})'

    def reset(self) -> torch.tensor:
        return 0

class LossArraySophizViz:
    def __init__(self, track_stats: bool = True):
        self.losses = []
        self.last_value = 0.0
        self.track_stats = bool(track_stats)

    def __add__(self, other: InvLossSophiaViz):
        self.losses.append(other)
        return self

    def __call__(self, x: torch.tensor, *, track_stats_override: bool | None = None):
        # Avoid Python sum’s 0-seed; do explicit accumulation
        total = None
        for l in self.losses:
            v = l(x)
            total = v if total is None else total + v
        tensor = total

        # CHANGED: optionally skip .item() to avoid GPU->CPU sync in HVP closure
        use_stats = self.track_stats if track_stats_override is None else bool(track_stats_override)
        if use_stats:
            self.last_value = float(tensor.detach().item())
        return tensor

    def header(self) -> str:
        rest = '\t'.join(l.name for l in self.losses)
        return f'Loss\t{rest}'

    def __str__(self):
        rest = '\t'.join(str(l) for l in self.losses)
        return f'{_round(self.last_value)}\t{rest}'

    def reset(self):
        return sum(l.reset() for l in self.losses)

    # Context manager to temporarily suspend stat updates (e.g., during HVP)
    @contextlib.contextmanager
    def suspend_stats(self):  # NEW
        old = self.track_stats
        self.track_stats = False
        try:
            yield
        finally:
            self.track_stats = old

class InvLoss:
    def __init__(self, coefficient: float = 1.0):
        self.c = coefficient
        self.name = _abbreviation(self.__class__.__name__)
        self.last_value = 0

    def __call__(self, x: torch.tensor) -> torch.tensor:
        tensor = self.loss(x)
        self.last_value = tensor.item()
        return self.c * tensor

    def loss(self, x: torch.tensor):
        raise NotImplementedError

    def __str__(self):
        return f'{_round(self.c * self.last_value)}({_round(self.last_value)})'

    def reset(self) -> torch.tensor:
        return 0

class LossArray:
    def __init__(self):
        self.losses = []
        self.last_value = 0

    def __add__(self, other: InvLoss):
        self.losses.append(other)
        return self

    def __call__(self, x: torch.tensor):
        tensor = sum(l(x) for l in self.losses)
        self.last_value = tensor.item()
        return tensor

    def header(self) -> str:
        rest = '\t'.join(l.name for l in self.losses)
        return f'Loss\t{rest}'

    def __str__(self):
        rest = '\t'.join(str(l) for l in self.losses)
        return f'{_round(self.last_value)}\t{rest}'

    def reset(self):
        return sum(l.reset() for l in self.losses)

# ---------------------------------
#                NORMS
# ---------------------------------

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Expect mean and std as lists of 3 elements.
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape((1, -1, 1, 1)))
        self.register_buffer('std', torch.Tensor(std).reshape((1, -1, 1, 1)))

    def forward(self, t: torch.tensor) -> torch.tensor:
        return self.get_normal(t)

    def get_normal(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    def get_unit(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.std) + self.mean

class L1Norm(nn.Module):
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.norm(p=1, dim=(1, 2, 3)).mean()

class L2Norm(nn.Module):
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.norm(p=2, dim=(1, 2, 3)).mean()

class BaseTotalVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_wise = x[:, :, :, 1:] - x[:, :, :, :-1]
        y_wise = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(2, 3)).mean() + y_wise.norm(p=self.p, dim=(2, 3)).mean() + \
               diag_1.norm(p=self.p, dim=(2, 3)).mean() + diag_2.norm(p=self.p, dim=(2, 3)).mean()

class ChromaAreaTarget(InvLoss):
    """
    Encourage a minimum fraction of pixels with nontrivial chroma.
    area_target in (0,1). tau is the chroma threshold in Lab space.
    """
    def __init__(self, area_target: float = 0.15, tau: float = 4.0, k: float = 4.0, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.area_target = float(area_target)
        self.tau = float(tau)  # chroma magnitude where a pixel counts as "colored"
        self.k = float(k)      # steepness of the soft mask

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        a, b = Lab[:,1:2], Lab[:,2:3]
        c = torch.sqrt(a*a + b*b + 1e-8)              # chroma magnitude
        p = torch.sigmoid(self.k * (c - self.tau))    # soft indicator ∈ (0,1)
        area = p.mean()
        # soft hinge: penalize only if area < target
        deficit = torch.relu(self.area_target - area)
        return deficit * deficit

class ChromaEnergyTarget(InvLoss):
    """
    Match mean chroma magnitude to a target (prevents total desaturation).
    Use a small weight; this is a guardrail, not the driver.
    """
    def __init__(self, m_target: float = 8.0, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.m_target = float(m_target)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        a, b = Lab[:,1:2], Lab[:,2:3]
        c = torch.sqrt(a*a + b*b + 1e-8)
        return (c.mean() - self.m_target).pow(2)

class ChromaBandStopPenalty(InvLoss):
    """
    Penalize power of chroma (Lab a/b) only in the very low-frequency band around DC,
    using a smooth raised-cosine mask. Cache the mask per (device, dtype, H, W).
    """
    def __init__(self, r0_frac: float = 0.05, width_frac: float = 0.03, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.r0_frac = float(r0_frac)
        self.width_frac = float(width_frac)
        self._mask_cache = {}  # key: (device, dtype, H, W) -> mask [1,1,H,W]

    @torch.no_grad()
    def _get_mask(self, device, dtype, H: int, W: int) -> torch.Tensor:
        """
        Build (or fetch) a raised-cosine mask that is ~1 near DC and drops to ~0 past r0+width.
        Cached by (device, dtype, H, W).
        """
        key = (device, dtype, H, W)
        cached = self._mask_cache.get(key, None)
        if cached is not None:
            return cached

        # Frequency coordinates for fftshifted spectrum
        yy = torch.arange(-H//2, H//2, device=device, dtype=dtype)
        xx = torch.arange(-W//2, W//2, device=device, dtype=dtype)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        r = torch.sqrt(X*X + Y*Y)  # radius grid, DC at center

        # "Nyquist radius" proxy (diag to corner), as a tensor for dtype/device consistency
        rN = torch.sqrt(torch.tensor((W//2)**2 + (H//2)**2, device=device, dtype=dtype)).clamp(min=1.0)

        r0 = self.r0_frac * rN                    # cutoff radius
        w  = (self.width_frac * rN).clamp(min=1e-6)  # smooth transition width

        # Raised cosine centered at r=0: 1 near 0, taper to 0 by r0+width
        # mask = 0.5 * (1 + cos(pi * clamp((r - r0)/w, -1, 1)))
        z = torch.clamp((r - r0) / w, min=-1.0, max=1.0)
        mask = 0.5 * (1.0 + torch.cos(torch.pi * z))
        # zero outside the transition (where z>1) and 1 inside deep low-freq (where z<-1)
        mask = mask * (z <= 1.0).to(dtype)  # already 0 when z>1
        mask = mask.view(1, 1, H, W).contiguous()

        self._mask_cache[key] = mask
        return mask

    def _band_power_low(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B,1,H,W] chroma channel (Lab a or b).
        Returns mean power in the low-frequency band defined by the cached mask.
        """
        B, C, H, W = y.shape  # C==1 here
        # 2D FFT with DC at center
        Y = torch.fft.fft2(y, dim=(-2, -1))
        Y = torch.fft.fftshift(Y, dim=(-2, -1))
        P = (Y.real**2 + Y.imag**2)  # power spectrum, [B,1,H,W]

        mask = self._get_mask(y.device, y.dtype, H, W)  # [1,1,H,W]
        # Mean over batch and spatial dims; channel is 1 so safe
        return (P * mask).mean()

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to Lab; accept x in [-1,1] or [0,1]
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        a = Lab[:, 1:2]
        b = Lab[:, 2:3]
        # Penalize only the very low-frequency chroma energy
        return self._band_power_low(a) + self._band_power_low(b)

    def reset(self):
        # Optional: clear cache if you change image sizes/devices mid-run
        self._mask_cache.clear()
        return 0

class LuminanceCentering(InvLoss):
    """Keep Lab L near mid-gray (~50)."""
    def __init__(self, L_target: float = 50.0, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.L_target = float(L_target)
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        L = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)[:,0:1]
        return (L.mean() - self.L_target).pow(2)

class GameOfLifeBaseColorVariationLowFreq(torch.nn.Module):
    # Makes
    def __init__(self, p: int = 2, sigma: float = 3.0):
        super().__init__()
        self.p = p
        self.sigma = float(sigma)
        self.tv = BaseTotalVariation(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inter-channel difference (same as your BaseColorVariation core)
        d = x - x.roll(shifts=1, dims=-3)         # [B,3,H,W], cyclic channel diff
        # keep only very-low-freq component (the “wash”)
        d_lp = gaussian_blur_2d(d, sigma=self.sigma)
        # run TV *on that low-frequency field only* to damp global chroma wash
        return self.tv(d_lp)

class BaseColorVariationLowFreq(torch.nn.Module):
    """Penalize only the *low-frequency* inter-channel differences (RGB)."""
    def __init__(self, p: int = 2, sigma: float = 3.0):
        super().__init__()
        self.p = p
        self.sigma = float(sigma)
        self.tv = BaseTotalVariation(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inter-channel difference (same as your BaseColorVariation core)
        d = x - x.roll(shifts=1, dims=-3)         # [B,3,H,W], cyclic channel diff
        gate = edge_gate_from_luminance(x, sigma=1.0)
        # keep only very-low-freq component (the “wash”)
        d_lp = gaussian_blur_2d(d, sigma=self.sigma) * gate  # suppress chroma smoothing near strong edges
        # run TV *on that low-frequency field only* to damp global chroma wash
        return self.tv(d_lp)

class ColorVariationLowFreq(InvLoss):
    def loss(self, x: torch.Tensor):
        return self.base(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1., sigma: float = 3.0):
        super().__init__(coefficient)
        self.base = BaseColorVariationLowFreq(p=p, sigma=sigma)
        self.size = size * size

class AbstractColorDistribution(nn.Module):
    def __init__(self, normalizer: Normalizer):
        super().__init__()
        self.normalizer = normalizer

    def forward(self, x: torch.tensor) -> torch.tensor:
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1])
        mean, std = view.mean(-1), view.std(-1, unbiased=False)
        mean_loss = (mean.view(-1) - self.normalizer.mean.view(-1)).norm()
        std_loss = (std.view(-1) - self.normalizer.std.view(-1)).norm()
        return mean_loss + std_loss

class BaseFakeBN(nn.Module):
    def __init__(self, resnet_function, normalizer: Normalizer):
        super().__init__()
        resnet = resnet_function(pretrained=True)
        self.conv, self.bn = resnet.conv1, resnet.bn1
        self.normalizer = normalizer

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(self.normalizer(x))
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1])
        mean, var = view.mean(1), view.var(1, unbiased=False)
        loss = torch.norm(self.bn.running_var.data - var, 2) + torch.norm(self.bn.running_mean.data - mean, 2)
        return loss

class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape((1, -1, 1, 1)))
        self.register_buffer('std', torch.Tensor(std).reshape((1, -1, 1, 1)))

    def forward(self, t: torch.tensor) -> torch.tensor:
        return self.get_normal(t)

    def get_normal(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    def get_unit(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.std) + self.mean

# ---------------------------------
#                LOSSES 
# ---------------------------------

class MatchBatchNorm(InvLoss):
    def __init__(self, bn: BaseFakeBN, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.bn = bn

    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.bn(x)

class NormalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseNormalVariation(p)
        self.size = size * size

class ColorVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseColorVariation(p)
        self.size = size * size

class FakeColorDistribution(nn.Module):
    def __init__(self, normalizer: Normalizer):
        super().__init__()
        self.normalizer = normalizer

    def forward(self, x: torch.tensor) -> torch.tensor:
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1])
        mean, std = view.mean(-1), view.std(-1, unbiased=False)
        mean_loss = (mean.view(-1) - self.normalizer.mean.view(-1)).norm()
        std_loss = (std.view(-1) - self.normalizer.std.view(-1)).norm()
        return mean_loss + std_loss

class FakeBatchNorm(nn.Module):
    def __init__(self, resnet_function, normalizer: Normalizer):
        super().__init__()
        resnet = resnet_function(pretrained=True)
        self.conv, self.bn = resnet.conv1, resnet.bn1
        self.normalizer = normalizer

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(self.normalizer(x))
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1])
        mean, var = view.mean(1), view.var(1, unbiased=False)
        loss = torch.norm(self.bn.running_var.data - var, 2) + torch.norm(self.bn.running_mean.data - mean, 2)
        return loss

class ColorDistribution(InvLoss):
    def loss(self, x: torch.tensor):
        return self.color_loss(x)

    def __init__(self, normalizer: Normalizer, coefficient: float = 1.):
        super().__init__(coefficient)
        self.color_loss = AbstractColorDistribution(normalizer)

class TotalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseTotalVariation(p)
        self.size = size * size

class BaseColorVariation(TotalVariation):
    def forward(self, x: torch.tensor) -> torch.tensor:
        rolled = x.roll(shifts=1, dims=-3)
        return super(ColorVariation, self).forward(x - rolled)

class BaseNormalVariation(TotalVariation):
    def forward(self, x: torch.tensor, per_sample: bool = True) -> torch.tensor:
        std = x.std() if not per_sample else x.view(x.shape[0], -1).std(dim=-1).view(-1, 1, 1, 1)
        x = (x - x.mean()) / (std + 0.0001)
        return super(NormalVariation, self).forward(x)

class BaseChromaDCLab(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        a = Lab[:,1:2]; b = Lab[:,2:3]
        return (a.mean(dim=[0,2,3]).pow(2).mean() + b.mean(dim=[0,2,3]).pow(2).mean())

class ChromaDCPenalty(InvLoss):
    def __init__(self, coefficient: float = 1.):
        super().__init__(coefficient)
        self.base = BaseChromaDCLab()
    def loss(self, x: torch.Tensor):
        return self.base(x)

class BaseChromaLowFreqLab(torch.nn.Module):
    def __init__(self, sigma: float = 3.0):
        super().__init__()
        self.sigma = float(sigma)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        a = Lab[:,1:2]; b = Lab[:,2:3]
        a_lp = gaussian_blur_2d(a, sigma=self.sigma)
        b_lp = gaussian_blur_2d(b, sigma=self.sigma)
        return a_lp.pow(2).mean() + b_lp.pow(2).mean()

class ChromaLowFreqPenalty(InvLoss):
    def __init__(self, sigma: float = 3.0, coefficient: float = 1.):
        super().__init__(coefficient)
        self.base = BaseChromaLowFreqLab(sigma=sigma)
    def loss(self, x: torch.Tensor):
        return self.base(x)

class BaseOpponentDecorrLab(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        L = Lab[:,0:1]; a = Lab[:,1:2]; b = Lab[:,2:3]
        def cov_abs(u, v):
            u0 = u - u.mean(dim=[2,3], keepdim=True)
            v0 = v - v.mean(dim=[2,3], keepdim=True)
            return (u0 * v0).mean(dim=[2,3]).abs().mean()
        return cov_abs(L,a) + cov_abs(L,b)

class OpponentDecorrelation(InvLoss):
    """
    Reduce correlation between luminance and chroma, but only:
      - for bandpassed L (ignore global lighting)
      - where chroma has enough magnitude (avoid hue flips in near-gray)
      - using *correlation* (scale-invariant), with smooth-L1 to avoid overreacting
    """
    def __init__(self, coefficient: float = 1.0, sigma_lp: float = 2.0, sigma_hp: float = 0.5,
                 chroma_tau: float = 3.0, huber_delta: float = 0.1):
        super().__init__(coefficient)
        self.sigma_lp = float(sigma_lp)     # low-pass for L
        self.sigma_hp = float(sigma_hp)     # high-pass width to get bandpassed L
        self.chroma_tau = float(chroma_tau) # gate threshold on |(a,b)|
        self.huber_delta = float(huber_delta)

    def _gauss(self, x, s):
        k = int(max(3, 2*round(3*s)+1))
        ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k-1)/2
        g = torch.exp(-(ax**2)/(2*s*s)); g = g/(g.sum()+1e-12)
        C = x.size(1)
        gh = g.view(1,1,1,k).expand(C,1,1,k)
        gv = g.view(1,1,k,1).expand(C,1,k,1)
        x = torch.nn.functional.pad(x, (k//2,k//2,0,0), mode="reflect")
        x = torch.nn.functional.conv2d(x, gh, padding=0, groups=C)
        x = torch.nn.functional.pad(x, (0,0,k//2,k//2), mode="reflect")
        x = torch.nn.functional.conv2d(x, gv, padding=0, groups=C)
        return x

    def _bandpass(self, x):
        lp = self._gauss(x, self.sigma_lp)
        if self.sigma_hp > 0:
            hp = x - self._gauss(x, self.sigma_hp)
            return hp - self._gauss(hp, self.sigma_lp) + lp  # gentle band limit
        return x - lp

    def _corr(self, u, v, gate):
        # center within each image; compute correlation under a soft spatial gate
        u0 = u - (u * gate).sum(dim=[2,3], keepdim=True) / (gate.sum(dim=[2,3], keepdim=True) + 1e-8)
        v0 = v - (v * gate).sum(dim=[2,3], keepdim=True) / (gate.sum(dim=[2,3], keepdim=True) + 1e-8)
        num = (u0 * v0 * gate).sum(dim=[2,3])
        den = torch.sqrt(((u0*u0*gate).sum(dim=[2,3]) + 1e-8) * ((v0*v0*gate).sum(dim=[2,3]) + 1e-8))
        return num / (den + 1e-8)  # in [-1,1]

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        # to Lab
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        L  = Lab[:,0:1]
        a  = Lab[:,1:2]
        b  = Lab[:,2:3]

        # band-limit luminance to avoid chasing DC/very low freq
        Lb = self._bandpass(L)

        # gate where chroma is meaningful
        c = torch.sqrt(a*a + b*b + 1e-8)
        gate = torch.sigmoid(4.0*(c - self.chroma_tau))  # ~0 below tau, ~1 above

        # correlations (scale-invariant)
        rho_La = self._corr(Lb, a, gate)
        rho_Lb = self._corr(Lb, b, gate)

        # smooth L1 around zero to avoid oscillatory overcorrection
        def huber(z, d):  # z ~ correlation
            az = z.abs()
            mask = (az < d).float()
            return (0.5*(az**2)/d)*mask + (az - 0.5*d)*(1.0 - mask)
        loss = huber(rho_La, self.huber_delta).mean() + huber(rho_Lb, self.huber_delta).mean()
        return loss


class BatchAugment(InvLoss):
    def loss(self, x: torch.tensor):
        if self.aug is not None:
            x = self.aug(x)
        return self.other(x)

    def __init__(self, other: InvLoss, aug: torch.tensor = None):
        super().__init__(coefficient=1.0)
        self.other = other
        self.aug = aug

class NetworkPass(InvLoss):
    def __init__(self, model: torch.nn.Module):
        super().__init__(coefficient=0.0)
        self.model = model

    def loss(self, x: torch.tensor):
        self.model(x)
        return torch.tensor(0)      

class CrossEntropyLoss(InvLoss):
    def loss(self, x: torch.tensor):
        return self.xent(self.model(x), self.label)

    def __init__(self, model: torch.nn.Module, label: torch.tensor, coefficient: float = 1.):
        super().__init__(coefficient)
        self.model = model
        self.label = label
        self.xent = torch.nn.CrossEntropyLoss()


class BatchNorm1stLayer(InvLoss):
    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.hook.get_layer(self.layer)

    def reset(self) -> torch.tensor:
        return self.hook.reset()

    def __init__(self, bn_hook: TimedHookHolder, layer: int = 0, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.hook = bn_hook
        self.layer = layer

class LayerActivationNorm(InvLoss):
    def __init__(self, hook: LayerHook, model: torch.nn.Module, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook, self.model = hook, model

    def loss(self, x: torch.tensor) -> torch.tensor:
        self.model(x)
        return - self.hook()

class ActivationNorm(InvLoss):
    def loss(self, x: torch.tensor):
        return - self.hook.get_layer(self.layer)

    def __init__(self, activation_hook: TimedHookHolder, layer: int, coefficient: float = 1.):
        super().__init__(coefficient)
        self.hook = activation_hook
        self.layer = layer

    def reset(self) -> torch.tensor:
        return self.hook.reset()

# --- StructureTensorCoherence: prefer coherent edges over isotropic speckle ---
# use F.conv2d; build kernels on x.device/x.dtype; no extra coefficient inside .loss()
def _sobel_xy(x: torch.Tensor):
    # Build per-call so dtype/device always match 'x'
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    C = x.size(1)
    kx = kx.repeat(C, 1, 1, 1)  # depthwise
    ky = ky.repeat(C, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=C)  # [B,C,H,W]
    gy = F.conv2d(x, ky, padding=1, groups=C)
    # average gradients over channels for structure tensor (one tensor per image)
    gx = gx.mean(dim=1, keepdim=True)  # [B,1,H,W]
    gy = gy.mean(dim=1, keepdim=True)
    return gx, gy

def _gauss_blur(x: torch.Tensor, sigma: float = 1.0):
    """
    Depthwise separable Gaussian blur with correct kernel shapes:
      - horizontal: [C, 1, 1, k]
      - vertical:   [C, 1, k, 1]
    Works for any C and matches x.device/x.dtype.
    """
    k = int(max(3, 2*round(3*sigma)+1))
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g1d = torch.exp(-(ax**2) / (2 * sigma * sigma))
    g1d = g1d / (g1d.sum() + 1e-12)                 # [k]

    C = x.size(1)
    # Build [C,1,1,k] for horizontal pass and [C,1,k,1] for vertical pass
    g_h = g1d.view(1, 1, 1, k).expand(C, 1, 1, k)   # horizontal kernel
    g_v = g1d.view(1, 1, k, 1).expand(C, 1, k, 1)   # vertical kernel

    x = F.conv2d(x, g_h, padding=(0, k//2), groups=C)  # pad width
    x = F.conv2d(x, g_v, padding=(k//2, 0), groups=C)  # pad height
    return x


class StructureTensorCoherenceLoss(nn.Module):
    """
    Penalize low coherence = encourage (lambda1 >> lambda2) from the structure tensor.
    Coherence = (λ1 - λ2) / (λ1 + λ2), averaged over the image.
    """
    def __init__(self, sigma: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.sigma = float(sigma)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx, gy = _sobel_xy(x)
        Ixx = gx * gx
        Iyy = gy * gy
        Ixy = gx * gy
        Ixx = _gauss_blur(Ixx, self.sigma)
        Iyy = _gauss_blur(Iyy, self.sigma)
        Ixy = _gauss_blur(Ixy, self.sigma)

        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        disc = torch.clamp(trace * trace - 4.0 * det, min=0.0)
        sq = torch.sqrt(disc + self.eps)
        l1 = 0.5 * (trace + sq)
        l2 = 0.5 * (trace - sq)

        coherence = (l1 - l2) / (trace + self.eps)  # in [0,1]
        return (1.0 - coherence).mean()  # penalize isotropy

class CoherencePenalty(InvLoss):
    def __init__(self, sigma: float = 1.0, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.base = StructureTensorCoherenceLoss(sigma=sigma)

    def loss(self, x: torch.Tensor):
        # Return raw scalar tensor; InvLossSophiaViz.__call__ multiplies by self.c
        return self.base(x)

# --- FeatureConsistencyLoss: stabilize across your existing RepeatBatch augs ---
class FeatureConsistencyLoss(nn.Module):
    """
    CLIP feature-invariance under augmentation:
      minimize variance across the batch of normalized image embeddings.
    Expects x to already be 'pre(img)' (RepeatBatch + jitter + noise + etc).
    """
    def __init__(self, clip_model, post_aug: nn.Module | None = None, eps: float = 1e-8):
        super().__init__()
        self.clip = clip_model
        self.post = post_aug if post_aug is not None else nn.Identity()
        self.eps = float(eps)

    def forward(self, x_aug: torch.Tensor) -> torch.Tensor:
        # x_aug: [B,C,H,W] (already augmented). We need gradients w.r.t. x_aug,
        # so do NOT wrap encode_image in torch.no_grad().
        f = self.clip.encode_image(self.post(x_aug)).float()        # [B,D]
        f = f / (f.norm(dim=-1, keepdim=True) + self.eps)           # unit-norm
        if f.size(0) < 2:                                           # no variance with batch=1
            return f.new_zeros(())
        return f.var(dim=0, unbiased=False).mean()                  # scalar

class CLIPAugVariance(InvLossSophiaViz):
    def __init__(self, clip_model, post_aug: nn.Module | None = None, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.base = FeatureConsistencyLoss(clip_model, post_aug)

    def loss(self, x_aug: torch.Tensor) -> torch.Tensor:
        # Return RAW tensor; InvLossSophiaViz.__call__ applies self.c
        return self.base(x_aug)

class FeatureCosineConsistencyLoss(nn.Module):
    """
    Minimize (1 - cosine) between all pairs of normalized CLIP image features
    in the current augmented batch. Stable for small B (>=2).
    """
    def __init__(self, clip_model, post_aug: nn.Module | None = None, eps: float = 1e-8):
        super().__init__()
        self.clip = clip_model
        self.post = post_aug if post_aug is not None else nn.Identity()
        self.eps = float(eps)

    def forward(self, x_aug: torch.Tensor) -> torch.Tensor:
        # x_aug: [B,C,H,W] already augmented (RepeatBatch/etc.)
        f = self.clip.encode_image(self.post(x_aug)).float()           # [B,D]
        f = f / (f.norm(dim=-1, keepdim=True) + self.eps)              # unit-norm
        B = f.size(0)
        if B < 2:
            return f.new_zeros(())  # no pairs
        G = f @ f.t()                                                   # [B,B], cos-sim
        # take upper triangle (i<j)
        idx = torch.triu_indices(B, B, offset=1, device=f.device)
        sims = G[idx[0], idx[1]]
        return (1.0 - sims).mean()                                     # minimize dissimilarity


class CLIPAugCosine(InvLossSophiaViz):
    def __init__(self, clip_model, post_aug: nn.Module | None = None, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.base = FeatureCosineConsistencyLoss(clip_model, post_aug)

    def loss(self, x_aug: torch.Tensor) -> torch.Tensor:
        return self.base(x_aug)

# --- Cosine-consistency to a small queue of past views (works with B=1) ---
class FeatureCosineQueueLoss(nn.Module):
    """
    Maintain a FIFO queue of past normalized features; minimize (1 - cosine)
    between current features and the queue. Good for repeat_batch=1.
    """
    def __init__(self, clip_model, post_aug: nn.Module | None = None,
                 queue_size: int = 8, eps: float = 1e-8):
        super().__init__()
        self.clip = clip_model
        self.post = post_aug if post_aug is not None else nn.Identity()
        self.eps = float(eps)
        self.queue_size = int(queue_size)
        self.register_buffer("_queue", None, persistent=False)  # [K,D] or None

    def _enqueue(self, f_unit: torch.Tensor):
        # f_unit: [B,D] unit-norm
        with torch.no_grad():
            if self._queue is None:
                self._queue = f_unit.detach()
            else:
                self._queue = torch.cat([self._queue, f_unit.detach()], dim=0)
                if self._queue.size(0) > self.queue_size:
                    self._queue = self._queue[-self.queue_size:]

    def forward(self, x_aug: torch.Tensor) -> torch.Tensor:
        f = self.clip.encode_image(self.post(x_aug)).float()           # [B,D]
        f = f / (f.norm(dim=-1, keepdim=True) + self.eps)
        # loss vs queue (if any)
        if (self._queue is None) or (self._queue.numel() == 0):
            loss = f.new_zeros(())
        else:
            sims = f @ self._queue.t()                                 # [B,K]
            loss = (1.0 - sims).mean()
        # update queue after computing loss (stop-grad)
        self._enqueue(f)
        return loss

class CLIPAugCosineQueue(InvLossSophiaViz):
    def __init__(self, clip_model, post_aug: nn.Module | None = None,
                 queue_size: int = 8, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.base = FeatureCosineQueueLoss(clip_model, post_aug, queue_size=queue_size)

    def loss(self, x_aug: torch.Tensor) -> torch.Tensor:
        return self.base(x_aug)

    def reset(self):
        # optional: clear queue between octaves/runs
        if hasattr(self.base, "_queue"):
            self.base._queue = None
        return 0

# --- SpectrumSlopeLoss: encourage 1/f^alpha spectrum without blurring ---

"""
class SpectrumSlopeLoss(nn.Module):

    #Fit log power vs log freq with a line; penalize deviation from target slope alpha.
    #Optional small L2 on residuals to damp narrowband spikes.

    def __init__(self, alpha: float = 1.2, spike_l2: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.spike_l2 = float(spike_l2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))
        P = (X.real**2 + X.imag**2).mean(dim=1)  # [B,H,W] average over channels

        H, W = P.shape[-2:]
        yy = torch.arange(-H//2, H//2, device=x.device, dtype=x.dtype)
        xx = torch.arange(-W//2, W//2, device=x.device, dtype=x.dtype)
        Y, Xg = torch.meshgrid(yy, xx, indexing="ij")
        r = torch.sqrt(Xg**2 + Y**2)  # [H,W]

        # flatten and mask out DC + zeros
        r_flat = r.flatten()
        P_flat = P.flatten(1)  # [B, H*W]
        mask = r_flat > 1.0
        r_flat = r_flat[mask]
        P_flat = P_flat[:, mask]

        # bin by integer radius
        r_int = r_flat.long().clamp_min(1)
        rmax = int(r_int.max().item())
        if rmax < 2:  # safety
            return torch.zeros((), device=x.device, dtype=x.dtype)

        bins = []
        for k in range(1, rmax + 1):
            m = (r_int == k)
            if m.any():
                bins.append(P_flat[:, m].mean(dim=1))
            else:
                bins.append(torch.zeros(P_flat.size(0), device=x.device, dtype=x.dtype))
        ps = torch.stack(bins, dim=1)                       # [B, R]
        rs = torch.arange(1, rmax + 1, device=x.device, dtype=x.dtype)

        # linear fit: log P ≈ a - alpha * log r
        log_r = torch.log(rs + 1e-12)[None, :]              # [1,R]
        log_p = torch.log(ps + 1e-12)                       # [B,R]
        lr = log_r - log_r.mean(dim=1, keepdim=True)
        lp = log_p - log_p.mean(dim=1, keepdim=True)
        denom = (lr**2).sum(dim=1, keepdim=True) + 1e-12
        slope = (lr * lp).sum(dim=1, keepdim=True) / denom  # [B,1]

        slope_loss = (slope.squeeze(1) + self.alpha).abs().mean()

        if self.spike_l2 > 0:
            # residuals around fitted line (orthogonal projection)
            # proj(lp onto lr): (lp·lr)/(||lr||) * (lr/||lr||)
            lr_norm = torch.sqrt((lr**2).sum(dim=1, keepdim=True) + 1e-12)
            proj = ((lp * lr).sum(dim=1, keepdim=True) / (lr_norm**2)) * lr
            res = lp - proj
            spike_loss = (res**2).mean()
            return slope_loss + self.spike_l2 * spike_loss
        return slope_loss
"""

# === CHANGED: replace your SpectrumSlopeLoss with this jittered version ===
class SpectrumSlopeLoss(torch.nn.Module):
    """
    Fit log power vs log freq with a line; penalize deviation from target slope alpha.
    Adds cheap stochasticity to avoid fixed spatial artifacts:
      - spatial roll jitter (phase dithering)
      - optional Hann window to reduce boundary/grid ringing
      - radial-bin jitter for the log–log fit
    """
    def __init__(self,
                 alpha: float = 1.2,
                 spike_l2: float = 0.0,
                 shift_frac: float = 0.125,   # CHANGED: max |shift| ≈ frac * size (per step random)
                 use_hann: bool = True,       # CHANGED
                 bin_jitter: bool = True):    # CHANGED
        super().__init__()
        self.alpha = float(alpha)
        self.spike_l2 = float(spike_l2)
        self.shift_frac = float(shift_frac)
        self.use_hann = bool(use_hann)
        self.bin_jitter = bool(bin_jitter)
        self._hann_cache = {}  # (device,dtype,H,W) -> [1,1,H,W]

    @torch.no_grad()
    def _hann(self, device, dtype, H, W):
        key = (device, dtype, H, W)
        if key in self._hann_cache:
            return self._hann_cache[key]
        # separable 2D Hann
        hy = torch.hann_window(H, dtype=dtype, device=device)
        hx = torch.hann_window(W, dtype=dtype, device=device)
        win = (hy[:, None] * hx[None, :]).view(1, 1, H, W).contiguous()
        self._hann_cache[key] = win
        return win

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape

        # ----- CHANGED: spatial roll jitter (phase dithering) -----
        if self.shift_frac > 0:
            max_dx = max(1, int(self.shift_frac * W))
            max_dy = max(1, int(self.shift_frac * H))
            dx = torch.randint(-max_dx, max_dx + 1, (1,), device=x.device).item()
            dy = torch.randint(-max_dy, max_dy + 1, (1,), device=x.device).item()
            x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))

        # ----- CHANGED: optional Hann window to reduce edge/grid ringing -----
        if self.use_hann:
            win = self._hann(x.device, x.dtype, H, W)
            x = x * win

        # FFT power (channel-averaged)
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))
        P = (X.real**2 + X.imag**2).mean(dim=1)  # [B,H,W]

        # frequency radii
        yy = torch.arange(-H//2, H//2, device=x.device, dtype=x.dtype)
        xx = torch.arange(-W//2, W//2, device=x.device, dtype=x.dtype)
        Y, Xg = torch.meshgrid(yy, xx, indexing="ij")
        r = torch.sqrt(Xg**2 + Y**2)  # [H,W]

        # flatten and mask out DC + zeros
        r_flat = r.flatten()                    # [H*W]
        P_flat = P.flatten(1)                   # [B, H*W]
        mask = r_flat > 1.0
        r_flat = r_flat[mask]
        P_flat = P_flat[:, mask]

        # ----- CHANGED: radial bin jitter to avoid fixed shell edges -----
        if self.bin_jitter:
            # add U(0,1) jitter before integer binning
            jitter = torch.rand((), device=x.device, dtype=x.dtype)
        else:
            jitter = torch.zeros((), device=x.device, dtype=x.dtype)

        r_int = torch.floor(r_flat + jitter).long().clamp_min(1)
        rmax = int(r_int.max().item())
        if rmax < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        # bin by integer radius
        bins = []
        for k in range(1, rmax + 1):
            m = (r_int == k)
            if m.any():
                bins.append(P_flat[:, m].mean(dim=1))
            else:
                bins.append(torch.zeros(P_flat.size(0), device=x.device, dtype=x.dtype))
        ps = torch.stack(bins, dim=1)                       # [B, R]
        rs = torch.arange(1, rmax + 1, device=x.device, dtype=x.dtype)

        # linear fit: log P ≈ a - alpha * log r
        log_r = torch.log(rs + 1e-12)[None, :]              # [1,R]
        log_p = torch.log(ps + 1e-12)                       # [B,R]
        lr = log_r - log_r.mean(dim=1, keepdim=True)
        lp = log_p - log_p.mean(dim=1, keepdim=True)
        denom = (lr**2).sum(dim=1, keepdim=True) + 1e-12
        slope = (lr * lp).sum(dim=1, keepdim=True) / denom  # [B,1]

        slope_loss = (slope.squeeze(1) + self.alpha).abs().mean()

        if self.spike_l2 > 0:
            # residuals orthogonal to fitted line
            lr_norm2 = (lr**2).sum(dim=1, keepdim=True) + 1e-12
            proj = ((lp * lr).sum(dim=1, keepdim=True) / lr_norm2) * lr
            res = lp - proj
            spike_loss = (res**2).mean()
            return slope_loss + self.spike_l2 * spike_loss
        return slope_loss

class FrequencySlopePenalty(InvLoss):
    def __init__(self, alpha: float = 1.2, spike_l2: float = 0.0, coefficient: float = 1.0,
                 shift_frac: float = 0.125, use_hann: bool = True, bin_jitter: bool = True):
        super().__init__(coefficient)
        self.base = SpectrumSlopeLoss(alpha=alpha, spike_l2=spike_l2,
                                      shift_frac=shift_frac, use_hann=use_hann, bin_jitter=bin_jitter)

    def loss(self, x: torch.Tensor):
        return self.base(x)

def _sobel_xy(x: torch.Tensor):
    # x: [B,1,H,W] or [B,C,H,W]
    B,C,H,W = x.shape
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    kx = kx.repeat(C,1,1,1); ky = ky.repeat(C,1,1,1)
    px = F.pad(x, (1,1,1,1), mode="reflect")
    gx = F.conv2d(px, kx, groups=C)
    gy = F.conv2d(px, ky, groups=C)
    return gx, gy

class EdgeAlignedChroma(InvLoss):
    """
    Penalize the *orthogonal* component of chroma gradients relative to luminance gradients.
    Reduces color halos that sit beside L edges (misregistration).
    """
    def __init__(self, coefficient: float = 1.0, eps: float = 1e-8, power: float = 1.0, edge_gate: float = 0.0):
        super().__init__(coefficient)
        self.eps = float(eps)
        self.power = float(power)      # 1.0 = L1, 2.0 = L2 on orthogonal component
        self.edge_gate = float(edge_gate)  # >0 to weight more on strong L edges

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        L = Lab[:,0:1]; a = Lab[:,1:2]; b = Lab[:,2:3]

        gLx, gLy = _sobel_xy(L)                 # [B,1,H,W]
        ga_x, ga_y = _sobel_xy(a)
        gb_x, gb_y = _sobel_xy(b)

        # Stack gradients as 2-vectors
        gL = torch.stack([gLx, gLy], dim=1)     # [B,2,1,H,W]
        ga = torch.stack([ga_x, ga_y], dim=1)   # [B,2,1,H,W]
        gb = torch.stack([gb_x, gb_y], dim=1)

        # Unit luminance gradient (avoid div by zero)
        gL2 = (gL**2).sum(dim=1, keepdim=True)        # [B,1,1,H,W]
        gL_hat = gL / (gL2.sqrt() + self.eps)

        # Project chroma gradients onto L-direction, subtract → orthogonal component
        def ortho(gc):
            # component along L: (gc·gL_hat) * gL_hat
            dot = (gc * gL_hat).sum(dim=1, keepdim=True)  # [B,1,1,H,W]
            gc_par = dot * gL_hat
            gc_orth = gc - gc_par
            return gc_orth

        oa = ortho(ga); ob = ortho(gb)
        # magnitude of orthogonal components
        oa_mag = (oa**2).sum(dim=1, keepdim=True).sqrt()   # [B,1,1,H,W]
        ob_mag = (ob**2).sum(dim=1, keepdim=True).sqrt()

        # Optional: emphasize at true edges
        if self.edge_gate > 0:
            w = (gL2.sqrt()).pow(self.edge_gate)          # stronger weight at strong L edges
            oa_mag = oa_mag * w
            ob_mag = ob_mag * w

        if self.power == 2.0:
            val = (oa_mag**2 + ob_mag**2).mean()
        else:
            val = (oa_mag + ob_mag).mean()
        return val

class EdgeAwareChromaTV(InvLoss):
    """
    Bilateral-style TV on chroma: strong smoothing where L is flat, weak near L edges.
    """
    def __init__(self, coefficient: float = 1.0, k: float = 4.0, tau: float = 0.02, p: int = 1):
        super().__init__(coefficient)
        self.k = float(k)         # sharpness of the gate
        self.tau = float(tau)     # edge threshold on |∇L|
        self.p = int(p)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        Lab = rgb_to_lab((x + 1)/2 if x.min() < 0 else x)
        L = Lab[:,0:1]; a = Lab[:,1:2]; b = Lab[:,2:3]
        gLx, gLy = _sobel_xy(L)
        gax, gay = _sobel_xy(a)
        gbx, gby = _sobel_xy(b)

        gL = torch.sqrt(gLx**2 + gLy**2 + 1e-8)
        # gate: 1 in flat regions (smooth chroma), ~0 near strong L edges
        gate = torch.sigmoid(self.k * (self.tau - gL))

        def tv_weighted(gx, gy):
            if self.p == 2:
                return (gate * (gx**2 + gy**2)).mean()
            return (gate * (gx.abs() + gy.abs())).mean()

        return tv_weighted(gax, gay) + tv_weighted(gbx, gby)

class PostWrapLoss(InvLoss):
    def __init__(self, base_loss, post_mod):
        super().__init__(base_loss.c)
        self.base_loss, self.post = base_loss, post_mod
        self.name = f"{base_loss.name}|post"
    def loss(self, x): return self.base_loss.loss(self.post(x))


# --- TruePatchCorrelationLoss: decorrelate spatial patches, not channels ---
class TruePatchCorrelationLoss(InvLoss):
    def __init__(self, patch: int = 7, stride: int = 3, coefficient: float = 1.0, max_patches: int = 1024):
        super().__init__(coefficient)
        self.patch = int(patch)
        self.stride = int(stride)
        self.max_patches = int(max_patches)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> patches: [B, C*P, N]
        patches = torch.nn.functional.unfold(x, kernel_size=self.patch, stride=self.stride)  # [B, C*P, N]
        patches = patches - patches.mean(dim=2, keepdim=True)
        # ℓ2 normalize each patch vector
        patches = patches / (patches.norm(dim=1, keepdim=True) + 1e-8)
        B, D, N = patches.shape

        # optional subsampling for speed
        if N > self.max_patches:
            idx = torch.randperm(N, device=x.device)[:self.max_patches]
            patches = patches[:, :, idx]
            N = patches.size(2)

        # Gram matrix across patches per batch: G = P^T P  (shape [B,N,N])
        G = patches.transpose(1, 2) @ patches
        # remove diagonal (self-correlation)
        G = G - torch.eye(N, device=x.device)[None]
        # penalize off-diagonal energy
        return (G.pow(2).mean())

class TruePatchCorrelationLossGPT5(InvLoss):
    def __init__(self,
                 patch: int = 7,
                 stride: int = 3,
                 coefficient: float = 1.0,
                 max_patches: int = 1024,
                 # NEW: seam-safe knobs
                 luma_only: bool = True,           # CHANGED default behavior is safer for color
                 local_sigma_px: float = 0.0,      # 0 ⇒ global; try ~2.5*patch to ~5*patch
                 local_radius_px: float = 0.0,     # 0 ⇒ no hard cutoff; set to ~6*stride to cap range
                 eps: float = 1e-8):
        super().__init__(coefficient)
        self.patch = int(patch)
        self.stride = int(stride)
        self.max_patches = int(max_patches)
        self.luma_only = bool(luma_only)          # NEW
        self.local_sigma_px = float(local_sigma_px)  # NEW
        self.local_radius_px = float(local_radius_px)  # NEW
        self.eps = float(eps)                     # NEW

    def _to_luma(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> [B,1,H,W]
        r, g, b = x[:, :1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # NEW: decorrelate on luma only (prevents chroma crosshair)
        x_used = self._to_luma(x) if (self.luma_only and C == 3) else x   # CHANGED

        # [B, C*P, N] with P = patch*patch
        patches = torch.nn.functional.unfold(x_used, kernel_size=self.patch, stride=self.stride)  # [B, C*P, N]

        # CHANGED: per-patch centering (zero DC of each vector, not per-feature across space)
        patches = patches - patches.mean(dim=1, keepdim=True)             # CHANGED (was dim=2)

        # ℓ2 normalize each patch vector (cosine features)
        patches = patches / (patches.norm(dim=1, keepdim=True) + self.eps)
        B, D, N = patches.shape

        # optional subsampling for speed (preserve index mapping)
        idx = None                                                        # NEW
        if N > self.max_patches:
            idx = torch.randperm(N, device=x.device)[:self.max_patches]
            patches = patches[:, :, idx]
            N = patches.size(2)

        # Gram across patches per batch: G = P^T P in [B,N,N]
        P = patches.transpose(1, 2)                                       # [B,N,D]
        G = P @ P.transpose(1, 2)                                         # [B,N,N]

        # remove diagonal (self-correlation) robustly
        eye = torch.eye(N, device=x.device, dtype=G.dtype)[None]          # NEW
        G = G * (1.0 - eye)                                               # CHANGED (no subtraction drift)

        # NEW: locality weighting → stop global partition walls
        if (self.local_sigma_px > 0.0) or (self.local_radius_px > 0.0):
            # infer patch grid and coords (centers) in pixel units
            nH = (H - self.patch) // self.stride + 1
            nW = (W - self.patch) // self.stride + 1
            yy, xx = torch.meshgrid(
                torch.arange(nH, device=x.device), torch.arange(nW, device=x.device), indexing="ij"
            )
            # centers (only relative distances matter)
            cy = yy * self.stride + (self.patch // 2)
            cx = xx * self.stride + (self.patch // 2)
            coords = torch.stack([cy.reshape(-1), cx.reshape(-1)], dim=1).float()  # [N_full,2]
            if idx is not None:
                coords = coords[idx]  # keep same subset as patches

            d = coords[:, None, :] - coords[None, :, :]                  # [N,N,2]
            d2 = (d * d).sum(dim=-1)                                     # [N,N]

            Wloc = torch.ones((N, N), device=x.device, dtype=G.dtype)
            if self.local_sigma_px > 0.0:
                Wloc = torch.exp(-0.5 * d2 / (self.local_sigma_px ** 2))
            if self.local_radius_px > 0.0:
                Wloc = Wloc * (d2 <= (self.local_radius_px ** 2))

            Wloc = Wloc * (1.0 - torch.eye(N, device=x.device, dtype=G.dtype))    # zero diagonal
            G = G * Wloc[None]  # broadcast over batch

        # penalize off-diagonal energy
        return G.pow(2).mean()                                           # unchanged scaling


class ChannelCorrelationLoss(InvLoss):
    def loss(self, x: torch.Tensor, factor=1) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        flat = x.view(b, c, -1)  # [B, C, H*W]
        mean = flat.mean(-1, keepdim=True)
        flat = flat - mean  # Zero mean

        cov = (flat @ flat.transpose(1, 2)) / (h*w)  # [B, C, C]
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        loss = cov.norm(dim=(1, 2)).mean() - diag.norm(dim=1).mean()
        return factor*loss

class FrequencyPenalty(InvLoss):
    def loss(self, x: torch.Tensor, factor=1) -> torch.Tensor:
        # x: [B, C, H, W]
        # Apply FFT along spatial dims
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        power = fft.abs()
        h, w = power.shape[-2:]
        y = torch.arange(-h//2, h//2, device=x.device).float()
        x_ = torch.arange(-w//2, w//2, device=x.device).float()
        yy, xx = torch.meshgrid(y, x_, indexing="ij")
        freq = torch.sqrt(xx**2 + yy**2)
        # Emphasize high freq
        high_freq = (freq > freq.max() * 0.5).float()
        penalty = (power * high_freq).mean()
        return factor*penalty

class MACOLoss(InvLoss):
    """
    MACO (Magnitude Constrained Optimization) Loss.
    Penalizes deviation of the magnitude spectrum of the optimized image
    from a precomputed average magnitude spectrum (ImageNet or other corpus).
    """
    def __init__(self, avg_magnitude: torch.Tensor, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.avg_magnitude = avg_magnitude  # (H, W) float32, on correct device

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Convert to grayscale: mean over channel
        gray = x.mean(dim=1)  # (B, H, W)
        # Compute 2D FFT for each image
        fft = torch.fft.fft2(gray)  # (B, H, W), complex64
        magnitude = torch.abs(fft)  # (B, H, W)
        # Difference from avg magnitude (ensure shape/device match)
        avg_mag = self.avg_magnitude.to(x.device, x.dtype)  # (H, W)
        # L2 loss per image, then mean over batch
        diff = magnitude - avg_mag.unsqueeze(0)  # (B, H, W)
        loss = (diff ** 2).mean()
        return loss

    # Optional: for display
    def __str__(self):
        return f"MACOLoss({self.c}) {super().__str__()}"

class MeanStdLoss(InvLoss):
    """
    Penalize deviation from target mean and std for each image.
    - mean_target: desired mean pixel value (e.g., 0.5 for [0,1])
    - std_target: desired std (e.g., 0.25 for typical normalized images)
    - reduction: 'batch' (default) = mean loss over batch; or 'all' = loss over all pixels
    """
    def __init__(self, mean_target=0.5, std_target=0.25, mean_coeff=1.0, std_coeff=1.0, reduction='batch', coefficient=1.0):
        super().__init__(coefficient)
        self.mean_target = mean_target
        self.std_target = std_target
        self.mean_coeff = mean_coeff
        self.std_coeff = std_coeff
        self.reduction = reduction

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] in [0,1] range
        B = x.shape[0]
        means = x.mean(dim=(1,2,3))  # [B]
        stds = x.std(dim=(1,2,3))    # [B]
        mean_loss = ((means - self.mean_target) ** 2) * self.mean_coeff
        std_loss = ((stds - self.std_target) ** 2) * self.std_coeff
        if self.reduction == 'batch':
            return (mean_loss + std_loss).mean()
        else:
            return (mean_loss + std_loss).sum()

    def __str__(self):
        return f'MeanStd({self.mean_target:.2f},{self.std_target:.2f}) {super().__str__()}'

class PromptAlignmentLoss(InvLoss):
    def __init__(self, clip_model, image: torch.Tensor, target_text_emb: torch.Tensor, text_coeff=0.5, neg=False) -> torch.Tensor:
        super().__init__()
        self.clip_model = clip_model
        self.target_text_emb = target_text_emb  # [1, D]
        self.text_coeff = text_coeff
        self.neg=neg

    def loss(self, image):
        # If image is batch [B,3,H,W], process all, else add batch dim
        if image.dim() == 3:
            image = image.unsqueeze(0)
        img_feat = self.clip_model.encode_image(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # Cosine similarity: higher = more aligned, but we *minimize* loss
        sim = torch.nn.functional.cosine_similarity(img_feat, self.target_text_emb)
        # Loss: negative cosine sim, so optimizer *maximizes* similarity
        if self.neg:
            return self.text_coeff * sim.mean()
        else:
            return -self.text_coeff * sim.mean()

class AugCLIPEmbeddingConsistency(InvLoss):
    def __init__(self, clip_model, aug_module: nn.Module, pairs: int = 2, coefficient: float = 0.2):
        super().__init__(coefficient)
        self.clip_model = clip_model
        self.aug = aug_module
        self.pairs = pairs
        self.name = "aug_cons"

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        e = self.clip_model.encode_image(x)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-8)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self._embed(x)                          # grads flow to x
        total = 0.0
        for _ in range(self.pairs):
            x2 = self.aug(x) if self.aug is not None else x
            e2 = self._embed(x2)
            total = total + (1.0 - torch.cosine_similarity(anchor, e2, dim=-1)).mean()
        return total / float(self.pairs)

class PatchmapAlphaLoss(InvLoss):
    def __init__(self, premodel, patchmaps: dict, layers, top_p=0.10,
                 coefficient: float = 0.2, cls_weight: float = 0.0, off_lambda: float = 0.25,
                 token_energy: str = "l2"):
        super().__init__(coefficient)
        self.model = premodel
        self.layers = list(sorted(set(layers)))
        self.pm = {l: np.asarray(pm, dtype=np.float32) for l, pm in patchmaps.items()}
        self.top_p = float(top_p)
        self.cls_weight = float(cls_weight)
        self.off_lambda = float(off_lambda)
        self.token_energy = token_energy
        self.tap = _ViTTokenTap(self.model, self.layers)
        Gh, Gw = _get_grid_hw(self.model)
        self.seq_len = 1 + Gh * Gw
        # precompute per-layer token weights [seq]
        self.w = {}
        for l in self.layers:
            pm = self.pm.get(l, None)
            w = np.zeros((self.seq_len,), dtype=np.float32)
            w[0] = self.cls_weight
            if pm is not None and pm.size == Gh * Gw:
                flat = pm.reshape(-1).clip(min=0)
                if flat.max() > 0:
                    flat /= flat.max()
                    k = max(1, int(self.top_p * flat.size))
                    thr = np.partition(flat, -k)[-k]
                    sel = (flat >= thr).astype(np.float32)
                    w[1:] = sel  # binary top-P; swap for `w[1:] = flat * sel` if you want proportional
            self.w[l] = w
        self.name = "pm_alpha"

    def _energy(self, a: torch.Tensor) -> torch.Tensor:
        if self.token_energy == "l1":
            return a.abs().mean(dim=-1)
        # default l2
        return (a.pow(2)).mean(dim=-1)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        bufs = self.tap.encode(x)  # fills self.tap.buff
        loss_terms = []
        for l in self.layers:
            a = bufs.get(l, None)
            if a is None:
                continue
            B, S, _ = a.shape
            if S != self.seq_len:
                # fallback: CLS only weighting
                w = torch.zeros((S,), device=a.device, dtype=a.dtype); w[0] = max(self.cls_weight, 1.0)
            else:
                w = torch.as_tensor(self.w[l], device=a.device, dtype=a.dtype)
            e = self._energy(a)                        # [B, S]
            on = (w * e).sum(dim=1) / (w.sum() + 1e-8) # maximize
            off = ((1.0 - w) * e).mean(dim=1)          # suppress elsewhere
            loss_terms.append(self.off_lambda * off - on)
        if not loss_terms:
            return x.sum() * 0.0  # neutral
        return torch.stack(loss_terms, dim=0).mean()

# ---------------------------------
#             MAIN HOOKS 
# ---------------------------------

class ViTREGFeatHook(InvLoss):# seems that doesn't matter
    def __init__(self, hook: ViTAbsHookHolder, key: str, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook = hook
        self.key = key

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 5:, :].mean(dim=1)  # Exclude CLS and REG
        mn = min(all_feats.shape)
        return - all_feats[:mn, :mn].diag().mean()

class ViTFeatHook(InvLoss):
    def __init__(self, hook: ViTAbsHookHolder, key: str, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook = hook
        self.key = key
    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, :mn].diag().mean()

class ReconstructionLoss(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, x: torch.tensor, key: str, feat: int = 0,
                 coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.ref = self.hook(x).clone().detach()
        self.f = feat

    def loss(self, x: torch.tensor):
        return (self.hook(x) - self.ref).norm()

class ViTFusionEnsFeatHook(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0].mean(dim=-1)
        mn = min(all_feats.shape)
        return - all_feats[self.f].mean()

class minViTREGEnsFeatHook(ViTREGFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 2:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, self.f].diag().mean()

class ViTREGEnsFeatHook(ViTREGFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 5:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, self.f].diag().mean()

class ViTEnsFeatHookGPT(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)  # d[self.key][0]: [batch, seq_len, feature_dim]
        # Exclude CLS token, average over all patches and batch
        patch_feats = d[self.key][0][:, 1:, self.f]  # [batch, seq-1]
        # Maximize mean over all patches and samples
        return -patch_feats.mean()  # Standard sign: maximize by gradient descent

class ViTEnsFeatHookGPT5(InvLoss):
    def __init__(
        self,
        hook: ViTAbsHookHolder,
        key: str,
        feat: int = 0,
        coefficient: float = 1.0,
        *,
        agg: str = "mean",           # NEW: "mean" | "topk" | "center"
        topk_patches: int = 0,       # NEW: used when agg="topk" (0 => auto ~ P/16)
        comp_alpha: float = 0.0,     # NEW: competitor suppression strength
        comp_k: int = 0              # NEW: # of strongest competing features to penalize (0 disables)
    ):
        super().__init__(coefficient)
        self.hook = hook
        self.key = key
        self.f = feat
        self.agg = agg               # NEW
        self.topk_patches = topk_patches
        self.comp_alpha = comp_alpha # NEW
        self.comp_k = comp_k         # NEW

    def _center_mask(self, P: int, device) -> torch.Tensor:  # NEW
        """Gaussian mask over patch grid (exclude CLS), normalized to sum=1."""
        g = int(round(P ** 0.5))
        assert g * g == P, f"Token grid not square: {P}"
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, g, device=device),
            torch.linspace(-1, 1, g, device=device),
            indexing="ij"
        )
        sigma = 0.6
        m = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        m = (m / m.sum()).reshape(1, P, 1)  # [1, P, 1]
        return m

    def loss(self, x: torch.tensor):
        d, _ = self.hook(x)                                # d[self.key][0]: [B, P+1, D] or [B, P, D] depending on hook
        t = d[self.key][0]                                 # [B, Seq, D]
        if t.size(1) > 0 and t.size(1) == int(round((t.size(1)-1)**0.5))**2 + 1:
            t = t[:, 1:, :]                                # CHANGED: exclude CLS if present
        B, P, D = t.shape
        target_vec = t[:, :, self.f]                       # [B, P]

        # ---- target pooling (agg) ----
        if self.agg == "mean":                             # default
            target_score = target_vec.mean()               # scalar
        elif self.agg == "topk":
            k = self.topk_patches if self.topk_patches > 0 else max(1, P // 16)
            vals, _ = torch.topk(target_vec, k=k, dim=1)   # [B, k]
            target_score = vals.mean()
        elif self.agg == "center":
            w = self._center_mask(P, t.device)             # [1, P, 1]
            target_score = (target_vec.unsqueeze(-1) * w).sum()  # scalar
        else:
            raise ValueError(f"Unknown agg='{self.agg}'")

        # ---- competitor suppression (optional) ----
        comp_term = 0.0
        if self.comp_alpha > 0.0 and self.comp_k > 0:
            feat_mean = t.mean(dim=1)                      # [B, D], avg over patches
            # remove self.f and take top-k competitors
            if self.f < D:
                mask = torch.ones(D, dtype=torch.bool, device=t.device)
                mask[self.f] = False
                others = feat_mean[:, mask]                # [B, D-1]
            else:
                others = feat_mean
            kk = min(self.comp_k, others.size(1))
            topk_vals, _ = torch.topk(others, k=kk, dim=1) # [B, kk]
            comp_term = topk_vals.mean()

        # We minimize loss => negative target (maximize) + alpha * competitors
        return -(target_score) + self.comp_alpha * comp_term  # CHANGED

class ViTMultiFeatHook(InvLoss):
    def __init__(self, hook: ViTAbsHookHolder, key: str, features_with_weights, coefficient: float = 1.0):
        """
        features_with_weights: list of (feature_idx, weight)
        """
        super().__init__(coefficient)
        self.hook = hook
        self.key = key
        self.features_with_weights = features_with_weights

    def loss(self, x: torch.tensor):
        d, _ = self.hook(x)  # dict of activations
        # shape: [batch, seq_len, feature_dim]
        feats = d[self.key][0][:, 1:, :]  # Exclude CLS
        # Average across patches in the sequence
        patch_mean = feats.mean(dim=1)  # [batch, feature_dim]
        loss_terms = []
        for f_idx, weight in self.features_with_weights:
            loss_terms.append(weight * patch_mean[:, f_idx].mean())
        # Negative: maximize activations
        return -torch.stack(loss_terms).sum()

class ViTEnsFeatHookGPTnot(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        feats_list = d[self.key]
        if not feats_list:  # Handle empty list edge case
            raise ValueError(f"Hook output for key '{self.key}' is empty.")
        feats = feats_list[0]  # [batch, seq, feature_dim]
        patch_feats = feats[:, 1:, self.f]  # Exclude CLS, [batch, seq-1]
        return -patch_feats.mean()

class ViTEnsFeatHook(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, self.f].diag().mean()

class ViTEnsFeatHookNOEDGE(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0, grid_size: int = 24, border: int = 1):
        super().__init__(hook, key, coefficient)
        self.f = feat
        self.grid_size = grid_size
        self.border = border

        # Precompute central (non-edge) patch indices
        self.central_patch_idx = self.get_non_edge_patch_indices(grid_size, border)

    def get_non_edge_patch_indices(self, grid_size, border=1):
        idx = []
        for i in range(border, grid_size - border):
            for j in range(border, grid_size - border):
                idx.append(i * grid_size + j)
        return idx

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        # d[self.key][0]: [batch, num_patches, feature_dim] (CLS excluded by next line)
        patches = d[self.key][0][:, 1:, :]  # Exclude CLS
        # Only central patches, exclude edge
        central_patches = patches[:, self.central_patch_idx, :]
        mn = min(central_patches.shape)
        return -central_patches[:mn, self.f].diag().mean()

class ViTHeadHook(ViTEnsFeatHook):
    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS and average over words, Result is BSx768
        return -all_feats.view(all_feats.shape[0], 12, -1).mean(dim=-1)[:, self.f].mean()

class ViTScoreHook(ViTEnsFeatHook):
    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        score_head = d[self.key][0][:, self.f, 1:, 1:]
        pw = int(np.sqrt(score_head.shape[-1]))
        patched = score_head.view(-1, pw, pw, pw, pw)
        ret_val = -patched[:, :, :pw // 2, :, pw // 2:].mean()
        return ret_val * 10000        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     CLIP HOOKS ~ ATTN, GELU, ACT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ViTAttHookHolder(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, in_feat: bool = False, keys: bool = False, queries: bool = False,
                 values: bool = False, scores: bool = False, out_feat: bool = False, sl: slice = None):
        super().__init__()
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, MultiHeadedSelfAttention)]
        self.attentions = self.just_save[sl]
        self.in_features = [ViTHook(m, False, 'in') for m in self.attentions] if in_feat else None
        self.keys = [ViTHook(a.proj_k, True, 'k') for a in self.attentions] if keys else None
        self.queries = [ViTHook(a.proj_q, True, 'q') for a in self.attentions] if queries else None
        self.value = [ViTHook(a.proj_v, True, 'v') for a in self.attentions] if values else None
        self.score_behaviour = scores
        self.out_features = [ViTHook(m, True, 'out') for m in self.attentions] if out_feat else None
        # print(in_feat, keys, queries, values, out_feat)

        self.model = classifier

    @property
    def scores(self):
        # for a in self.attentions:
        #     a.scores = None
        # return None
        return [FakeHookWrapper(a.scores) for a in self.attentions] if self.score_behaviour else None

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        # for a in self.just_save:
        #     a.scores = None
        out = None
        if x is not None:
            out = self.model(x)
        options = [self.in_features, self.keys, self.queries, self.value, self.scores, self.out_features]
        options = [[o.activations for o in l] if l is not None else None for l in options]
        names = ['in_feat', 'keys', 'queries', 'values', 'scores', 'out_feat']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class ClipAggReLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        #self.just_save = [m for m in classifier.modules() if isinstance(m, nn.ReLU) and hasattr(m, 'gate_mlp')]
        #self.just_save = [m for name, m in classifier.named_modules() if isinstance(m, nn.ReLU) and "gate_mlp" in name]
        #self.just_save = [m for name, m in classifier.named_modules() if isinstance(m, nn.ReLU) and "intermediate_fusion_mlps" in name]     
        self.just_save = [m for name, m in classifier.named_modules() if isinstance(m, nn.ReLU) and "fusion_mlp." in name]          
        #self.just_save = [m for name, m in classifier.named_modules() if isinstance(m, nn.ReLU) and "fusion_gate" in name]         
        #self.just_save = [m for name, m in classifier.named_modules() if isinstance(m, nn.ReLU) and "freq_proj" in name]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class ClipReLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, nn.ReLU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class LongClipGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, LongQuickGELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class minREGClipGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, nn.GELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class REGClipGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, REGQuickGELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class ClipGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, QuickGELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class ClipSineActHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, SineAct)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class MetaClipGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, QuickGELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class ClipOVGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, GELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations for o in l if o.activations is not None] if l is not None else None # no transpose!
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class ViTGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, PositionWiseFeedForward)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m.fc1, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[F.gelu(o.activations) for o in l] if l is not None else None for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out

class ReconstructionClipGeLUHook(ClipGeLUHook):
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        acts = self.high[0].activations.transpose(0, 1)
        return acts

class SaliencyClipGeLUHook(ClipGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor, l: int, f: int) -> torch.tensor:
        _ = self.cl(x)
        acts = self.high[l].activations.transpose(0, 1)[:, 1:, f]
        return acts

class SpecialSaliencyClipGeLUHook(ClipGeLUHook):
    def __init__(self, classifier: nn.Module, sl: slice = None, layer=None, feature=None):
        super().__init__(classifier, sl)
        # Now, `layer` and `feature` are stored as attributes of the instance
        self.layer = layer
        self.feature = feature

    @torch.no_grad()
    def forward(self, x: torch.tensor, l: int, f: int) -> torch.tensor:
        _ = self.cl(x)
        # Use self.layer and self.feature if they are supposed to override l and f
        acts = self.high[l].activations.transpose(0, 1)[:, 1:, f]
        return acts

class SimpleClipGeLUHook(ClipGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        # :-1 excludes CLS token
        acts = torch.cat([((l.activations.transpose(0, 1))[:, 1:, :]).mean(dim=1).float() for l in self.high
                          if l.activations is not None], dim=-1).clone().detach()
        return acts         

class AbsActivationHook(BasicHook):
    def __init__(self, module: nn.Module, feature: int = 0, targets: list = None):
        super().__init__(module)
        self.activations = []
        self.feature = feature
        self.targets = targets

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

    def reset(self):
        if self.activations is not None:
            for _, v in self.activations:
                del v
            del self.activations
        self.activations = []

    def set_feature(self, feature: int):
        self.feature = feature

    def set_target(self, target: list):
        self.targets = target

    def __call__(self) -> torch.tensor:
        if isinstance(self.activations, list):
            return torch.tensor(0)
        return self.activations

class ActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        diagonal = torch.arange(min(input_t.patch_size()[:2]))
        feats = input_t[diagonal, diagonal]
        self.activations = feats.norm(p=2, dim=(1, 2)).mean()

class ActivationReluHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        input_t = torch.relu(input_t)
        diagonal = torch.arange(min(input_t.size()[:2]))
        feats = input_t[diagonal, diagonal]
        self.activations = feats.norm(p=2, dim=(1, 2)).mean()

class TargetActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        diagonal = torch.arange(min(input_t.patch_size()[:2]))
        feats = input_t[diagonal, self.targets]
        self.activations.append((datetime.now(), feats.norm(p=2, dim=(1, 2)).mean()))

class ContrastiveActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        value = size * feats.norm(p=2, dim=(1, 2)).mean() - input_t[diagonal].norm(p=2, dim=(2, 3)).mean()
        self.activations.append((datetime.now(), value))

class ViTCLSActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats[:, 0].mean() * feats.patch_size(-1)
        self.activations.append((datetime.now(), feats))

class ViTMeanActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats.norm(p=2, dim=-1).mean() * 10 * 10
        self.activations.append((datetime.now(), feats))

class BatchNormHookHookAbs(AbsActivationHook):
    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

    @staticmethod
    def get_mean_var(x: torch.tensor) -> (torch.tensor, torch.tensor):
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1]).to('cuda:0')
        return view.mean(1), view.var(1, unbiased=False)

    @staticmethod
    def normalize_eval(model: nn.Module, x: torch.tensor) -> torch.tensor:
        extra_dim = [1] * (x.dim() - 2)
        mean = model.running_mean.data.view(1, -1, *extra_dim)
        var = model.running_var.data.view(1, -1, *extra_dim)
        return (x - mean) / var

class MatchModelBNStatsHook(BatchNormHookHookAbs):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        mean, var = self.get_mean_var(input_t)
        cur_value = torch.norm(model.running_var.data - var, 2) + torch.norm(model.running_mean.data - mean, 2)
        self.activations.append((datetime.now(), cur_value))

# ---------------------------------
#         PRE-PROCESSING
# ---------------------------------

class Tile(nn.Module):
    def __init__(self, rep: int = 384 // 16):
        super().__init__()
        self.rep = rep

    def forward(self, x: torch.tensor) -> torch.tensor:
        dim = x.dim()
        if dim < 3:
            raise NotImplementedError
        elif dim == 3:
            x.unsqueeze(0)
        final_shape = x.shape[:2] + (x.shape[2] * self.rep, x.shape[3] * self.rep)
        return x.unsqueeze(2).unsqueeze(4).repeat(1, 1, self.rep, 1, self.rep, 1).view(final_shape)

class JitterDims(nn.Module):
    def __init__(self, lim: int = 32, modeldims: int = 224):
        super().__init__()
        self.lim = lim
        self.modeldims = modeldims

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))

class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))

class ColorJitter(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1., use_fixed_random_seed: bool = False):
        super(ColorJitter, self).__init__()
        if use_fixed_random_seed:
            fix_random_seed(seed=random_seed)
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.shuffle_every = shuffle_every
        self.shuffle()

    def shuffle(self):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img - self.mean) / self.std

class ColorJitterGPT5(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1., use_fixed_random_seed: bool = False):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.shuffle_every = shuffle_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if use_fixed_random_seed:
            fix_random_seed(seed=random_seed)

    def shuffle(self):
        # mean in [-mean_p, +mean_p]
        mean = (torch.rand((self.batch_size, 3, 1, 1), device=self.device) - 0.5) * 2 * self.mean_p
        # log-std in [-std_p, +std_p]  -> clamp exp to [0.5, 2.0]
        logstd = (torch.rand((self.batch_size, 3, 1, 1), device=self.device) - 0.5) * 2 * self.std_p   # CHANGED
        std = logstd.exp().clamp_(0.5, 2.0)                                                             # CHANGED
        return mean, std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.shuffle_every or not hasattr(self, 'mean') or not hasattr(self, 'std'):
            self.mean, self.std = self.shuffle()
        return (img - self.mean) / self.std

class GuassianNoiseBase(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(GuassianNoise, self).__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, img):
        out = img + torch.randn(img.size()) * self.std + self.mean
        out = torch.clamp(out, 0., 1.)
        return out

class GaussianNoise(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, std: float = 1., max_iter: int = 400, use_fixed_random_seed: bool = False):
        super(GaussianNoise, self).__init__()
        if use_fixed_random_seed:
            fix_random_seed(seed=random_seed)
        self.batch_size, self.std_p, self.max_iter = batch_size, std, max_iter
        self.shuffle_every = shuffle_every
        self.std = None
        self.rem = max_iter - 1
        self.shuffle()

    def shuffle(self):
        self.std = torch.randn(self.batch_size, 3, 1, 1).cuda() * self.rem * self.std_p / self.max_iter
        self.rem = (self.rem - 1 + self.max_iter) % self.max_iter

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return img + self.std

class GaussianNoiseGPT(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, std: float = 1., use_fixed_random_seed: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.std_p = std
        self.shuffle_every = shuffle_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if use_fixed_random_seed:
            fix_random_seed(seed=random_seed)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(img.shape, device=img.device) * self.std_p
        return img + noise

class GaussianNoiseClassicAuto(torch.nn.Module):
    """
    Exact-classic behavior when max_iter is provided:
      std = N(B,3,1,1) * (rem * std_p / max_iter), rem -> rem-1 (wrap optional)
      - channel-wise noise (constant over HxW)
      - cached unless shuffle_every=True

    When max_iter is None:
      - uses approx_period as the decay window (linear to 0)
      - auto-resets when (H,W) changes (octave change), unless disabled
    """
    def __init__(self,
                 batch_size: int,
                 shuffle_every: bool = False,
                 std: float = 1.0,
                 max_iter: Optional[int] = None,
                 approx_period: int = 400,
                 wrap: bool = False,
                 reset_on_shape_change: bool = True,
                 use_fixed_random_seed: bool = False):
        super().__init__()
        # optional deterministic seeding (mirrors your original hook)
        if use_fixed_random_seed:
            try:
                fix_random_seed(seed=random_seed)  # if you have this helper
            except NameError:
                torch.manual_seed(0)

        self.batch_size = int(batch_size)
        self.shuffle_every = bool(shuffle_every)
        self.std_p = float(std)

        self.max_iter = int(max_iter) if max_iter is not None else None
        self.period = int(max_iter) if max_iter is not None else int(approx_period)
        self.wrap = bool(wrap)
        self.reset_on_shape_change = bool(reset_on_shape_change)

        self.std: Optional[torch.Tensor] = None
        self.rem = self.period - 1
        self._last_hw = None  # (H, W)

    def reset(self, max_iter: Optional[int] = None):
        """Call this if you later learn the exact step budget (e.g., steps_per_octave[oi])."""
        if max_iter is not None:
            self.max_iter = int(max_iter)
            self.period = int(max_iter)
        self.rem = self.period - 1
        self.std = None

    def _advance(self):
        if self.wrap:
            self.rem = (self.rem - 1 + self.period) % self.period
        else:
            self.rem = max(self.rem - 1, 0)

    def _make_noise(self, device, dtype, B: int) -> torch.Tensor:
        # channel-wise, constant over HxW, independent per sample
        return torch.randn(B, 3, 1, 1, device=device, dtype=dtype)

    def _shuffle(self, img: torch.Tensor):
        B = img.shape[0]
        if B != self.batch_size:
            # be robust if repeat_batch changes
            self.batch_size = B

        denom = float(self.max_iter if self.max_iter is not None else self.period)
        scale = (self.rem / max(1.0, denom)) * self.std_p
        self.std = self._make_noise(img.device, img.dtype, self.batch_size) * scale
        self._advance()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, _, H, W = img.shape

        # auto-reset when the resolution changes (octave switch)
        if self.reset_on_shape_change:
            if self._last_hw is None:
                self._last_hw = (H, W)
            elif self._last_hw != (H, W):
                self.reset()              # keep current period, restart decay
                self._last_hw = (H, W)

        # refresh the cached field on demand
        if self.std is None or self.shuffle_every:
            self._shuffle(img)

        return img + self.std

class OctaveAwareGaussianNoise(torch.nn.Module):
    """
    Octave 0  : classic linear decay (ease-out), like the original.
    Octave ≥1 : same plus a linear ease-in over the first ease_in_frac of steps.

    Resampling control:
      - resample_every=1  → new noise every step (shuffle_every=True)
      - resample_every=0  → sample once per octave, then hold (shuffle_every=False)
      - resample_every=N  → new noise every N steps
    """
    def __init__(self,
                 batch_size: int,
                 steps_this_octave: int,
                 octave_index: int,
                 std: float = 0.5,
                 ease_in_frac: float = 0.20,
                 resample_every: int = 1):
        super().__init__()
        self.batch_size = int(batch_size)
        self.T = max(1, int(steps_this_octave))
        self.oi = int(octave_index)
        self.std_p = float(std)
        self.ease_in_frac = float(ease_in_frac)
        self.resample_every = int(resample_every)  # 1=every step, 0=never (hold), N=periodic
        self._i = 0
        self._cached = None  # holds last sampled noise (B,3,1,1)

    @staticmethod
    def _lin_decay(t: int, T: int) -> float:
        # classic: amp(t) = std * ((T-1 - t) / T)
        return max(0.0, (T - 1 - float(t)) / float(T))

    def _amp(self, t: int) -> float:
        base = self.std_p * self._lin_decay(t, self.T)
        if self.oi >= 1 and self.ease_in_frac > 0.0:
            k = max(1, int(round(self.ease_in_frac * self.T)))
            if t < k:
                ramp = float(t + 1) / float(k)  # 0→1 over first k steps
                base *= ramp
        return base

    def _need_resample(self) -> bool:
        if self._cached is None:
            return True
        if self.resample_every <= 0:
            return False  # hold forever
        return (self._i % self.resample_every) == 0

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        t = min(self._i, self.T - 1)
        amp = self._amp(t)

        if amp > 0.0:
            if self._need_resample():
                # channel-wise noise, constant over HxW
                B = img.shape[0] if img.dim() == 4 else self.batch_size
                self._cached = torch.randn(B, 3, 1, 1, device=img.device, dtype=img.dtype) * amp
            out = img + self._cached
        else:
            out = img

        self._i += 1
        return out

class ScheduledGaussianNoise(torch.nn.Module):
    """
    Per-octave scheduled noise:
      - First K steps: cosine-decay from first_std -> first_std_min
      - Micro-bursts: every `burst_every` steps for `burst_len` steps at first_std
      - Tiny dust: always-on very small std each step (optional)
    Resets automatically when the module is reinstantiated (you recreate `pre` each octave).
    """
    def __init__(self,
                 first_k: int = 0,
                 first_std: float = 0.06,
                 first_std_min: float = 0.02,
                 burst_every: int = 0,
                 burst_len: int = 0,
                 tiny_std: float = 0.0):
        super().__init__()
        self.first_k = int(first_k)
        self.first_std = float(first_std)
        self.first_std_min = float(first_std_min)
        self.burst_every = int(burst_every)
        self.burst_len = int(burst_len)
        self.tiny_std = float(tiny_std)
        self._i = -1  # call counter (per octave)

    def _cosine_decay(self, t: float) -> float:
        # t in [0,1] -> 1..0
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        self._i += 1
        std_use = 0.0

        # First-K window (strong → mild)
        if self.first_k > 0 and self._i < self.first_k:
            # cosine decay from first_std → first_std_min
            frac = self._i / max(1, self.first_k - 1)
            w = self._cosine_decay(frac)  # 1..0
            std_use = self.first_std_min + (self.first_std - self.first_std_min) * w

        # Micro-bursts (periodic)
        elif self.burst_every > 0 and self.burst_len > 0:
            if (self._i % self.burst_every) < self.burst_len:
                std_use = max(std_use, self.first_std)

        # Tiny dust (always-on, optional)
        if std_use == 0.0 and self.tiny_std > 0.0:
            std_use = self.tiny_std

        if std_use > 0.0:
            return img + torch.randn_like(img) * std_use
        return img


class GaussianNoiseGPTRnd(nn.Module):
    def __init__(
        self,
        batch_size: int,
        shuffle_every: bool = False,
        std: float = 1.0,
        use_fixed_random_seed: bool = False,
        std_schedule=None  # Can be a function: step -> std
    ):
        super().__init__()
        self.batch_size = batch_size
        self.std_p = std
        self.shuffle_every = shuffle_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.std_schedule = std_schedule
        if use_fixed_random_seed:
            torch.manual_seed(6247423)
            np.random.seed(6247423)

    def forward(self, img: torch.Tensor, step: int = None) -> torch.Tensor:
        # If a schedule is provided, use it
        std = self.std_p
        if self.std_schedule is not None and step is not None:
            std = self.std_schedule(step)
        noise = torch.randn_like(img) * std
        return img + noise


# ---------------------------------
#         POST-PROCESSING
# ---------------------------------

class ClipSTD(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.tensor, inflate: float = 1., per_sample: bool = True) -> torch.tensor:
        std = x.std() if not per_sample else x.view(x.shape[0], -1).std(dim=-1).view(-1, 1, 1, 1)
        mean = x.mean() if not per_sample else x.view(x.shape[0], -1).mean(dim=-1).view(-1, 1, 1, 1)
        x = inflate * (x - mean) / (std * 2)
        return x.clamp(min=-0.5, max=0.5) + 0.5

class Clip(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.clamp(min=0, max=1)

class LInfClip(nn.Module):
    def __init__(self, original: torch.tensor, eps: float = 16 / 255):
        super().__init__()
        self.base = original.detach().clone().cuda()
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + torch.clip(self.base - x, min=-self.eps, max=self.eps)

class L2Clip(nn.Module):
    def __init__(self, original: torch.tensor, eps: float = 16 / 255):
        super().__init__()
        self.base = original.detach().clone().cuda()
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        delta = self.base - x
        norm = delta.norm(p=2)
        delta = self.eps * delta / norm if norm > self.eps else delta
        return x + delta

class Gray4D(nn.Module):
    def __init__(self, n_channels: int = 3):
        super().__init__()
        self.n = n_channels

    def forward(self, x: torch.tensor) -> torch.tensor:
        shape = tuple([1] * (4 - x.dim())) + x.shape
        return x.view(shape).repeat(1)

class Layered(nn.Module):
    def __init__(self, x: torch.tensor):
        super().__init__()
        self.x = x

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + self.x

class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))

class ColorJitter(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1.):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img - self.mean) / self.std

class ColorJitterGPT(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1., use_fixed_random_seed: bool = False):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.shuffle_every = shuffle_every
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if use_fixed_random_seed:
            fix_random_seed(seed=random_seed)

    def shuffle(self):
        mean = (torch.rand((self.batch_size, 3, 1, 1), device=self.device) - 0.5) * 2 * self.mean_p
        std = ((torch.rand((self.batch_size, 3, 1, 1), device=self.device) - 0.5) * 2 * self.std_p).exp()
        return mean, std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.shuffle_every or not hasattr(self, 'mean') or not hasattr(self, 'std'):
            self.mean, self.std = self.shuffle()
        return (img - self.mean) / self.std

class ColorJitterR(ColorJitter):
    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img * self.std) + self.mean

class GaussianNoise(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, std: float = 1., max_iter: int = 400):
        super().__init__()
        self.batch_size, self.std_p, self.max_iter = batch_size, std, max_iter
        self.std = None
        self.rem = max_iter - 1
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.std = torch.randn(self.batch_size, 3, 1, 1).cuda() * self.rem * self.std_p / self.max_iter
        self.rem = (self.rem - 1 + self.max_iter) % self.max_iter

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return img + self.std

class Centering(nn.Module):
    def __init__(self, size: int, std: float):
        super().__init__()
        self.size = size
        self.std = std

    def forward(self, img: torch.tensor) -> torch.tensor:
        pert = (torch.rand(2) * 2 - 1) * self.std
        w, h = img.shape[-2:]
        x = (pert[0] + w // 2 - self.size // 2).long().clamp(min=0, max=w - self.size)
        y = (pert[1] + h // 2 - self.size // 2).long().clamp(min=0, max=h - self.size)
        return img[:, :, x:x + self.size, y:y + self.size]

class Zoom(nn.Module):
    def __init__(self, out_size: int = 384):
        super().__init__()
        self.up = torch.nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False).cuda()

    def forward(self, img: torch.tensor) -> torch.tensor:
        return self.up(img)

class Tile(nn.Module):
    def __init__(self, rep: int = 384 // 16):
        super().__init__()
        self.rep = rep

    def forward(self, x: torch.tensor) -> torch.tensor:
        dim = x.dim()
        if dim < 3:
            raise NotImplementedError
        elif dim == 3:
            x.unsqueeze(0)
        final_shape = x.shape[:2] + (x.shape[2] * self.rep, x.shape[3] * self.rep)
        return x.unsqueeze(2).unsqueeze(4).repeat(1, 1, self.rep, 1, self.rep, 1).view(final_shape)

class TileGPT(nn.Module):
    def __init__(self, rep: int = 1):
        super().__init__()
        self.rep = rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        assert x.dim() == 4, "Expected 4D tensor (B, C, H, W)"
        if self.rep == 1:
            return x
        x = x.repeat_interleave(self.rep, dim=2)
        x = x.repeat_interleave(self.rep, dim=3)
        return x

class RepeatBatch(nn.Module):
    def __init__(self, repeat: int = 32):
        super().__init__()
        self.size = repeat

    def forward(self, img: torch.tensor):
        return img.repeat(self.size, 1, 1, 1)

class MaskBatch(nn.Module):
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.other(x[:self.count] if self.count > 0 else x)

    def __init__(self, count: int = -1):
        super().__init__()
        self.count = count

class Flip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        return torch.flip(x, dims=(3,)) if random.random() < self.p else x

# Very experimental fake repeat batch
# ===== Core helpers =====
def _unit(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def _resize_to(x: torch.Tensor, size: int) -> torch.Tensor:
    # size: int or (H,W)
    if isinstance(size, int):
        size = (size, size)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

# ===== Option A: EMA anchor (cheapest; robust with B=1) =====
class _Anchor224EMA(nn.Module):
    """
    Maintains an EMA anchor of CLIP features at 224 for the current image.
    - 224 path is computed under no_grad() -> tiny VRAM.
    - EMA stabilizes the target across steps.
    """
    def __init__(self, clip_model, post_aug: nn.Module | None, target_size: int = 224,
                 ema: float = 0.9, refresh_every: int = 1, eps: float = 1e-8):
        super().__init__()
        self.clip = clip_model
        self.post = post_aug if post_aug is not None else nn.Identity()
        self.target_size = int(target_size)
        self.ema = float(ema)
        self.refresh_every = int(refresh_every)
        self.eps = float(eps)
        self.register_buffer("_anchor", None, persistent=False)  # [1,D] or None
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _update_anchor(self, x: torch.Tensor):
        # resize to 224, get normalized feature, average over batch if B>1
        x224 = _resize_to(x, self.target_size)
        f224 = self.clip.encode_image(self.post(x224)).float()
        f224 = _unit(f224, self.eps).mean(dim=0, keepdim=True)  # [1,D]
        if self._anchor is None:
            self._anchor = f224.detach()
        else:
            self._anchor = self.ema * self._anchor + (1.0 - self.ema) * f224.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Periodically refresh anchor
        if (int(self._step.item()) % max(1, self.refresh_every)) == 0 or (self._anchor is None):
            self._update_anchor(x)
        self._step += 1

        # Current view at native res (with grad)
        f = self.clip.encode_image(self.post(x)).float()
        f = _unit(f, self.eps)  # [B,D]

        # Cosine nudge to EMA anchor
        # Broadcast [1,D] over batch -> [B]
        sims = (f * self._anchor).sum(dim=-1)
        return (1.0 - sims).mean()  # scalar

class CLIPAnchorNudge224(InvLoss):
    """
    Cheap invariance: nudge current features toward a stop-grad 224 EMA anchor.
    Works with repeat_batch=1; very low VRAM overhead.
    """
    def __init__(self, clip_model, post_aug: nn.Module | None = None,
                 coefficient: float = 1.0, target_size: int = 224,
                 ema: float = 0.9, refresh_every: int = 1):
        super().__init__(coefficient)
        self.base = _Anchor224EMA(clip_model, post_aug, target_size, ema, refresh_every)

    def loss(self, x_aug: torch.Tensor) -> torch.Tensor:
        return self.base(x_aug)

    def reset(self):
        # Clear EMA between octaves/runs if desired
        if hasattr(self.base, "_anchor"):
            self.base._anchor = None
            self.base._step.zero_()
        return 0

# ===== Option B: One-shot pair (no EMA) =====
class _Pair224(nn.Module):
    """
    Compute native-res feature (grad) and 224 feature (no-grad) in one call,
    penalize 1 - cosine. No state, no EMA.
    """
    def __init__(self, clip_model, post_aug: nn.Module | None, target_size: int = 224, eps: float = 1e-8):
        super().__init__()
        self.clip = clip_model
        self.post = post_aug if post_aug is not None else nn.Identity()
        self.target_size = int(target_size)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_hi = self.clip.encode_image(self.post(x)).float()
        f_hi = _unit(f_hi, self.eps)  # [B,D]

        with torch.no_grad():
            x224 = _resize_to(x, self.target_size)
            f_lo = self.clip.encode_image(self.post(x224)).float()
            f_lo = _unit(f_lo, self.eps)  # [B,D]

        return (1.0 - (f_hi * f_lo).sum(dim=-1)).mean()

class CLIPAnchorPair224(InvLoss):
    """
    Stateless version: 1 - cosine(current, stop-grad 224) per step.
    Slightly noisier than EMA, still cheap.
    """
    def __init__(self, clip_model, post_aug: nn.Module | None = None,
                 coefficient: float = 1.0, target_size: int = 224):
        super().__init__(coefficient)
        self.base = _Pair224(clip_model, post_aug, target_size)

    def loss(self, x_aug: torch.Tensor) -> torch.Tensor:
        return self.base(x_aug)
