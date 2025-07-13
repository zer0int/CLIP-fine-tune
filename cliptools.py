"""
Credits to the original author for most of these: 
https://github.com/hamidkazemi22/vit-visualization

A few additions taken from 
https://github.com/stanislavfort/Direct_Ascent_Synthesis

Small changes and further additions by zer0int:
https://github.com/zer0int

Iteration V3 - 04/June/2025
"""
import os
import copy
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.utils.data
from clip.model import QuickGELU # use for original CLIP model
from torch.nn import ReLU, GELU # for Fusion MLP models
import numpy as np
import torch.nn as nn
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.utils
import torchvision
import datetime


_nums = '0123456789'

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

def fix_random_seed(seed: int = 6247423):
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
        print(d)
        print(o)
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
        #self.just_save = [m for name, m in classifier.named_modules() if isinstance(m, nn.ReLU) and "gate_mlp" in name] # fusion_mlp # freq_proj # fusion_gate. fusion_mlp.
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
            fix_random_seed(seed=6247423)
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
            fix_random_seed(seed=6247423)
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
            fix_random_seed(seed=6247423)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(img.shape, device=img.device) * self.std_p
        return img + noise


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
            fix_random_seed(seed=6247423)

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