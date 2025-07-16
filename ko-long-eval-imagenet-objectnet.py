import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
from safetensors.torch import load_file
import argparse
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longclip as clip
from longclip.model import CLIP

from cliptools import fix_random_seed # deterministic
fix_random_seed() 

# ----------- DATASET -------------------------------------
# Download here:
# https://objectnet.dev/mvt/
#
# Then use command-line arguments:
# --data_dir path/to/data_release_2023/all/
# --data_csv path/to/data_release_2023/human_responses.csv
# ---------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate Zero-Shot Accuracy on ImageNet/ObjectNet')
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-KO-LITE-FULL-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--data_dir', default='path/to/dataset-difficulty-CLIP/data_release_2023/all/', help="Path to images /all/")
    parser.add_argument('--data_csv', default='path/to/dataset-difficulty-CLIP/data_release_2023/human_responses.csv', help="Path to labels .csv")
    parser.add_argument('--head', default=10, type=int, help="Attention Head to ablate (0-15). Default: 10. Only active when also passing --ablate_head")
    parser.add_argument("--ablate_head", action='store_true', help="Ablate Attention Head (to benchmark influence thereof)")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

csv_file = args.data_csv
image_folder = args.data_dir

local_path = "results_oai/imagenet-eval"
os.makedirs(local_path, exist_ok=True)

# ---------------- ADVANCED SETTINGS --------------------------
# Set an attention head to ablate & compare results here
target_head = args.head # 0-15 for ViT-L/14
ablate_head = False # True to ablate
if args.ablate_head:
    ablate_head = True
# -------------------------------------------------------------


model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float()

# ---- Zeroing Target Head Attention Output FOR ALL LAYERS ----
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
        # Use a closure to capture head_idx for each layer/hook
        handles.append(block.attn.register_forward_hook(make_hook(head_idx)))
    return handles

# ---- Zeroing Q and V weights for Head 10 IN ALL LAYERS ----
# Other option for ablating head. Not implemented by default. Change 'ablate_head' below to use.
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
    hook_handles = ablate_head_output_all_layers(model, target_head)

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=BICUBIC, max_size=None, antialias=True),
        transforms.Lambda(_convert_image_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    return transform(image)


class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
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

def evaluate_model(model, dataloader):
    correct = 0
    total = 0

    for batch_images, batch_labels in tqdm(dataloader):
        batch_images = batch_images.to(device)
        batch_texts = clip.tokenize(batch_labels).to(device)

        with torch.no_grad():
            image_embeddings = model.encode_image(batch_images)
            text_embeddings = model.encode_text(batch_texts)
            logits_per_image = (image_embeddings @ text_embeddings.T).softmax(dim=-1)

            _, top_indices = logits_per_image.topk(1, dim=-1)
            
            for i, label in enumerate(batch_labels):
                if label == batch_labels[top_indices[i, 0].item()]:
                    correct += 1
                total += 1
    
    accuracy = correct / total
    return accuracy

dataset = CroppedImageCSVFileDataset(csv_file, image_folder, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=48, shuffle=True)

model_accuracy = evaluate_model(model, dataloader)
print(f"Accuracy on MVT ImageNet/ObjectNet: {model_accuracy:.4f}")

with open(f"{local_path}/imagenet-objectnet.txt", "w", encoding='utf-8') as f:
    f.write(f"Accuracy on MVT ImageNet/ObjectNet: {model_accuracy:.4f}")

print(f"\nResults saved to {local_path}.")