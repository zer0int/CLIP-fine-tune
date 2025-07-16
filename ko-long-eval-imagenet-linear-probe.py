import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import json
import numpy as np
import glob
from safetensors.torch import load_file
import argparse

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longclip as clip

# ---------------------------------------------------------
# Linear Probe Eval - requires ILSVRC2012 dataset.
# https://www.image-net.org/download.php
# ---------------------------------------------------------

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate linear probe on ImageNet 1k val')
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-KO-LITE-FULL-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--data_path', default="path/to/your/ILSVRC2012/val/", help="Path to ILSVRC2012 /val/")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

VAL_DIR = args.data_path
JSON_FILE = "image_sets/imagenet_wnid_to_class_filtered.json" # already included with this repo!
local_path ="results_oai/imagenet"
os.makedirs(local_path, exist_ok=True)
CSV_OUTPUT = f"{local_path}/imagenet_clip_linear_probe.csv"

BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-3
SEED = 42

model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float() # full precision

print("Running Linear Probe on ImageNet for original CLIP pre-trained model...")

# Load ImageNet label mapping, the easy and lazy way. No messing with a .mat file / devkit! :)
with open(JSON_FILE, "r", encoding="utf-8") as f:
    wnid_to_label = json.load(f)

# Ensure we are loading the real thing, though, else the numbers won't compare:
assert len(wnid_to_label) == 1000, "Error: Filtered JSON does not contain exactly 1000 classes!"

# Label indexing
unique_labels = list(wnid_to_label.values())
label_to_index = {label: i for i, label in enumerate(unique_labels)}

# Prepare dataset
image_files, image_labels = [], []
for wnid, label in wnid_to_label.items():
    folder_path = os.path.join(VAL_DIR, wnid)
    images = sorted(os.listdir(folder_path))  # Get all images

    for image in images:
        image_files.append(os.path.join(folder_path, image))
        image_labels.append(label_to_index[label])  # Numeric class index

# Ensure we are loading the real thing, though, else the numbers won't compare:
assert len(image_files) == 50000, f"Expected 50,000 images, found {len(image_files)}"

# Convert labels to tensor
image_labels = torch.tensor(image_labels, dtype=torch.long)

# Extract CLIP features
def extract_image_features():
    features = []
    dataloader = DataLoader(image_files, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting CLIP Features"):
            batch_images = [preprocess(Image.open(path).convert("RGB")) for path in batch]
            batch_images = torch.stack(batch_images).to(device)

            image_features = model.encode_image(batch_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            features.append(image_features.cpu())

    return torch.cat(features)

print("Extracting CLIP image embeddings...")
image_embeddings = extract_image_features()

# Save extracted features
torch.save({"features": image_embeddings, "labels": image_labels}, f"{local_path}/imagenet_clip_features.pth")

# Split into train (80%) and val (20%) to prevent overfitting
num_train = int(0.8 * len(image_embeddings))
num_val = len(image_embeddings) - num_train
train_dataset, val_dataset = random_split(TensorDataset(image_embeddings, image_labels), [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define linear probe
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

torch.manual_seed(SEED)
linear_probe = LinearProbe(input_dim=768, num_classes=1000).to(device)
optimizer = optim.AdamW(linear_probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# Train linear probe
print("Training Linear Probe...")
best_val_acc = 0  # Early stopping
for epoch in range(EPOCHS):
    linear_probe.train()
    train_loss, correct, total = 0, 0, 0

    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = linear_probe(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
        train_loss += loss.item()

    train_acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}: Loss = {train_loss / len(train_loader):.4f}, Train Accuracy = {train_acc:.2f}%")

    # Validation
    linear_probe.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = linear_probe(batch_x)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(batch_y).sum().item()
            val_total += batch_y.size(0)

    val_acc = 100.0 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(linear_probe.state_dict(), f"{local_path}/linear_probe_best.pth")
    else:
        print("Early stopping triggered, no improvement.")

# Evaluate final model
linear_probe.load_state_dict(torch.load(f"{local_path}/linear_probe_best.pth"))
linear_probe.eval()
top1_correct, top5_correct, total = 0, 0, 0

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = linear_probe(batch_x)
        _, top1_pred = outputs.topk(1, dim=-1)
        _, top5_pred = outputs.topk(5, dim=-1)

        top1_correct += top1_pred.eq(batch_y.view(-1, 1)).sum().item()
        top5_correct += sum(batch_y[i] in top5_pred[i] for i in range(len(batch_y)))
        total += batch_y.size(0)

top1_acc = top1_correct / total * 100
top5_acc = top5_correct / total * 100

df = pd.DataFrame([{"Top-1 Accuracy": top1_acc, "Top-5 Accuracy": top5_acc}])
df.to_csv(CSV_OUTPUT, index=False)

print(f"\nOpenAI CLIP, Linear Probe Results: Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")

with open(f"{local_path}/linear-probe.txt", "a", encoding='utf-8') as f:
    f.write(f"OpenAI CLIP, Linear Probe Results: Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")
