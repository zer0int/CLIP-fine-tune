import os
import json
import torch
import random
from colorama import Fore, Style
from tqdm import tqdm
from PIL import Image
import kornia
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

import attnclipindiv_train as clip # Geometric Parametrization + Attn Head Dropout

# Suppress warnings spam from torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

training_losses = []
validation_losses = []
k_proj_losses = []
k_proj_losses_running = []

# >>>   Use this script to FINE-TUNE a CLIP model.  <<<
# Requires dataset: https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco
#
# Requires: https://huggingface.co/datasets/zer0int/CLIP-KO-Adversarial-Train-Typo-Attack
# Put the "typoattack" folder with images into the 'ko_adversarial_dataset' folder.

# IMPORTANT >>>  Additional config: CTRL+F for: CFGME

# --------------------- OUTPUT FOLDERS ---------------------

# Save training plots with matplotlib to:
plots_folder = 'ft-plots'
os.makedirs(plots_folder, exist_ok=True)

# Save model .pt files to: 
ft_checkpoints_folder = 'ft-checkpoints'
os.makedirs(ft_checkpoints_folder, exist_ok=True)

# Save verbose text / training logs to:
text_logs_folder = 'ft-logs'
os.makedirs(text_logs_folder, exist_ok=True)

# ------------------------------------------------------------


# -------------------------- Metrics & Housekeeping --------------------------
def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    if epoch < adjust_after:
        return 1  # Initial slower unfreeze rate
    else:
        return increase_rate  # Increased rate after initial pass

def unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=False):
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze_every_n_epochs = adjust_unfreeze_rate(epoch)
        layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
        layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= total_layers - layers_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False

def monitor_gradient_norms(gradient_norms, threshold=1e-5):
    alert_messages = []
    for name, norms in gradient_norms.items():
        mean_norm = sum(norms) / len(norms)
        if mean_norm < threshold:  # Vanishing gradient
            alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        elif mean_norm > 1000:  # Exploding gradient
            alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
    if alert_messages:
        for message in alert_messages:
            print(message)
        # Optionally, you could also implement some automatic adjustment strategies here

def plot_gradient_norms(gradient_norms, epoch, use_log_scale=True):
    plt.figure(figsize=(20, 10))
    
    # Choose a colormap
    cmap = plt.get_cmap('Spectral')
    
    # Sort the layers by the maximum gradient norm value, descending
    sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
    
    # Generate distinct colors from the colormap
    colors = cmap(range(len(sorted_layers)))
    
    for (layer_name, norms), color in zip(sorted_layers, colors):
        plt.plot(norms, label=layer_name, color=color)

    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    # Adjust legend: position at top right with smaller font size
    plt.legend(loc='upper right', fontsize='small')
    
    if use_log_scale:
        plt.yscale('log')
        plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}_log.png")
    else:
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}.png")
    
    plt.close()

def plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts):
    epochs_x = range(1, epoch + 2)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if len(training_losses) == len(epochs_x):
        plt.plot(epochs_x, training_losses, label='Training Loss')
    if len(validation_losses) == len(epochs_x):
        plt.plot(epochs_x, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    if len(logits_images) == len(epochs_x):
        plt.plot(epochs_x, logits_images, label='Average Logits')
    if len(logits_texts) == len(epochs_x):
        plt.plot(epochs_x, logits_texts, label='Average Logits')
    plt.title('Average Logits Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()

def plot_k_proj(k_proj_losses, epoch, plots_folder):
    """
    Plot k_proj orthogonality loss over epochs.
    Args:
        k_proj_losses (list of float): Collected after each epoch.
        epoch (int): Current epoch (zero-based).
        plots_folder (str): Folder to save plots.
    """
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    epochs_x = range(1, len(k_proj_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_x, k_proj_losses, marker='o', color='red', label='k_proj_orthogonality_loss')
    plt.title('k_proj Orthogonality Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('k_proj Loss')
    plt.yscale('log')  # This loss is typically very small
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/k_proj_loss_epoch_{epoch + 1}.png")
    plt.close()


def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1



# ------------------------------- Datasets -------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]  # Fallback to the first label if less than 2 are available
        else:
            label = ''  # Fallback if no labels are available

        text = clip.tokenize([label])  # Tokenize the label

        return image, text.squeeze(0)  # Remove the extra dimension

class AdversarialImageTextDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.transform = transform
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Now image should be [3, H, W] float tensor

        # Defensive: ensure 3 channels
        assert image.shape[0] == 3, f"Image shape before aug: {image.shape}"
        # Batch augment
        image = adversarial_augs(image.unsqueeze(0)).squeeze(0)
        assert image.shape[0] == 3, f"Image shape after aug: {image.shape}"

        labels = self.annotations[image_path]
        while len(labels) < 3:
            labels.append("")
        texts = clip.tokenize(labels)  # shape: (3, seq_len)
        return image, texts  # image: [3,224,224]; texts: [3, seq_len]




# ---------------------------------- Losses ----------------------------------
def k_proj_orthogonality_loss(model, selected_layers=None, lam=1.0):
    """
    Penalizes cosine similarity between heads' key projections for each expanded MLP feature.
    """
    device = next(model.parameters()).device
    loss_total = 0.0
    count = 0

    for idx in selected_layers:
        block = model.visual.transformer.resblocks[idx]
        k_proj = block.attn.k_proj
        W = k_proj.weight  # [embed_dim, expanded_dim]
        embed_dim, expanded_dim = W.shape
        num_heads = block.attn.num_heads
        head_dim = embed_dim // num_heads

        # [num_heads, head_dim, expanded_dim]
        W_heads = W.view(num_heads, head_dim, expanded_dim)
        W_heads_norm = F.normalize(W_heads, p=2, dim=1)  # [num_heads, head_dim, expanded_dim]

        # Cosine similarity for all head pairs for each feature
        sim = torch.einsum('ihf,jhf->ijf', W_heads_norm, W_heads_norm)  # [num_heads, num_heads, expanded_dim]

        # Zero out the diagonal (head self-sim) per feature
        # sim[i, i, f] = 0 for all i, f
        num_heads = sim.shape[0]
        eye = torch.eye(num_heads, device=sim.device).unsqueeze(-1)  # [num_heads, num_heads, 1]
        sim_no_diag = sim * (1 - eye)  # sets diagonal to zero

        # Now sum all off-diagonal squares
        loss = (sim_no_diag ** 2).sum() / (num_heads * (num_heads - 1) * expanded_dim)
        loss_total += loss
        count += 1

    if count > 0:
        loss_total = loss_total / count
    return lam * loss_total


def decatt_loss(model, selected_layers=None, lam=1.0):
    """
    Compute DeCAtt loss across selected layers of ViT visual transformer.
    Returns a scalar tensor.
    """
    device = next(model.parameters()).device
    if selected_layers is None:
        selected_layers = list(range(6))
    decatt_total = 0.0
    count = 0
    for idx in selected_layers:
        block = model.visual.transformer.resblocks[idx]
        x = getattr(block.attn, 'last_attn_output_per_head', None)
        if x is None:
            continue
        # x: [batch, heads, seq, head_dim]
        batch, heads, seq, head_dim = x.shape
        x_flat = x.permute(1, 0, 2, 3).contiguous().view(heads, batch * seq * head_dim)  # [heads, -]
        x_flat = F.normalize(x_flat, p=2, dim=1)
        c = torch.matmul(x_flat, x_flat.t()) / x_flat.shape[1]
        off_diag = c - torch.diag(torch.diag(c))
        loss = (off_diag ** 2).sum() / (heads * (heads - 1))
        decatt_total += loss
        count += 1
    if count > 0:
        decatt_total = decatt_total / count
    # Convert to tensor, keep on device for autograd and printing
    decatt_total = torch.as_tensor(decatt_total, device=device, dtype=torch.float32)
    return lam * decatt_total


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate loss as the mean of the two cross-entropy losses
        loss_img = self.criterion(logits, labels)
        loss_txt = self.criterion(logits.t(), labels)

        return (loss_img + loss_txt) / 2

adversarial_augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=12, translate=0.12, scale=(0.94,1.06), p=0.8),
    kornia.augmentation.RandomResizedCrop(size=(224, 224), scale=(0.92,1.0), ratio=(0.9,1.1), p=0.7),
    kornia.augmentation.ColorJitter(brightness=0.15, contrast=0.15, p=0.5)
).cuda()

contrastive_loss = ContrastiveLoss(temperature=0.07)
# ---------------


# ----------------------------------------------------------------------------
#                      C O N F I G U R A T I O N     CFGME
# ----------------------------------------------------------------------------

clipmodel = 'ViT-L/14'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clipmodel, device=device, jit=False)
model = model.float()

dont_save_every_epoch = False   # Save checkpoints less often; not recommended. Delicate sweetspot may be missed!
save_every_n_epochs = 2         # If not saving every Epoch, save every n Epochs  

unfreeze_all = True
EPOCHS = 20
max_learning_rate = 5e-7
learning_rate = 3e-7
batch_size = 18
adv_batch_size = 1

# ---------------------------------------------------------------------------
# I've provided the labels, but you'll still need to download the images!
# https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco
# See the readme in COCO/ for more details.
# ---------------------------------------------------------------------------

# Define your training dataset and dataloader
dataset1 = ImageTextDataset("path/to/COCO/data-square", "path/to/COCO/data-square/short-coco-sprite-train-0_9.json", transform=preprocess)
concatenated_dataset = ConcatDataset([dataset1])  # Add more datasets to this list as needed ([dataset1, dataset2]) 
train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset and dataloader
val_dataset = ImageTextDataset("path/to/COCO/data-square", "path/to/COCO/data-square/short-coco-sprite-val-10_11.json", transform=preprocess)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# IMPORTANT!
# "adversarial_labels_min_cos_sim.json" - 1st label is CLIP's min cos sim 'opinion'
# "adversarial_labels_max_cos_sim.json" - 1st label is CLIP's max cos sim 'opinion'
#
# The MAX version seems to be more disruptive - extreme reduction of typographic attack vulnerability,
# but as a trade-off, eliminates entire concepts from the model. Potential use-case: SFW training.
# Default: Using MIN as a more balanced, less disruptive option.
#
# To fine-tune WITHOUT adversarial training, simply remove it from trainloop, CTRL+F: Adversarial Injection
# Not recommended. ko-eval-zeroshot-attack.py results:
# Pre-Trained:                2/58 correct
# NO adversarial training:   27/60 correct
# WITH min_cos_sim ADV:      53/60 correct
# WITH max_cos_sim ADV:      59/60 correct

# Adversarial dataset -- First, download: https://huggingface.co/datasets/zer0int/CLIP-KO-Adversarial-Train-Typo-Attack
# -> Put the "typoattack" folder with images into the 'ko_adversarial_dataset' folder.
adversarial_dataset = AdversarialImageTextDataset("ko_adversarial_dataset/adversarial_labels_min_cos_sim.json", transform=preprocess) # MIN
#adversarial_dataset = AdversarialImageTextDataset("ko_adversarial_dataset/adversarial_labels_max_cos_sim.json", transform=preprocess) # MAX

adversarial_loader = DataLoader(adversarial_dataset, batch_size=adv_batch_size, shuffle=True)
total_steps = len(train_dataloader) * EPOCHS


# Define parameter groups for different learning rates
visual_parameters = [p for p in model.visual.transformer.parameters() if p.requires_grad]
transformer_parameters = [p for p in model.transformer.parameters() if p.requires_grad]

param_groups = [
    {'params': visual_parameters, 'lr': 3e-7},
    {'params': transformer_parameters, 'lr': 1e-8},
    {'params': model.token_embedding.parameters(), 'lr': 3e-7},
    {'params': [model.positional_embedding, model.visual.positional_embedding, model.visual.class_embedding], 'lr': 1e-7},
    {'params': [model.visual.proj, model.text_projection], 'lr': 1e-7},
    {'params': [model.visual.ln_pre.weight, model.visual.ln_pre.bias, model.visual.ln_post.weight, model.visual.ln_post.bias], 'lr': 1e-7},
    {'params': [model.ln_final.weight, model.ln_final.bias, model.visual.conv1.weight], 'lr': 1e-7}
]

accumulation_steps = 2  # Effective batch size will be batch_size * accumulation_steps

optimizer = AdaBelief(param_groups, lr=learning_rate, eps=1e-14, betas=(0.9, 0.999), weight_decay=1e-3, weight_decouple=True, rectify=True, print_change_log=False)
scheduler = OneCycleLR(optimizer, max_lr=max_learning_rate, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')


# ----------------------------------------------------------------------------
#                                      END
# ----------------------------------------------------------------------------


print(f"Precision: {model.dtype}")
print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
print("== START == \n")

def trainloop():
    contrastive_loss = ContrastiveLoss(temperature=0.07).to(device)
    logits_images = []
    logits_texts = []
    k_proj_losses = []  

    accumulation_steps = 2  # Adjust 'fake' batch size (accumulation)
    scaler = GradScaler()
    for epoch in range(EPOCHS):
        gradient_norms = {}
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss = 0.0
        total_decatt_loss = 0.0
        total_krpoj_loss = 0.0
        k_proj_losses_running = []
        total_kproj_loss = 0.0
        train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
        adversarial_iter = iter(adversarial_loader)
        ADV_INJECT_EVERY = 5  # Inject adversarial example every __ batches (default: 5)
        adv_lambda = 0.1
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)

        optimizer.zero_grad()

        for batch_idx, (images, texts) in progress_bar:
            images, texts = images.to(device), texts.to(device)
            batch_logits_images = []
            batch_logits_texts = []

            with autocast():
                logits_per_image, logits_per_text = model(images, texts)
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                

                # ----------------------------- Contrastive Loss -----------------------------
                total_loss = contrastive_loss(logits_per_image, logits_per_text)
                # --------------------------- DeCAtt / k_proj Loss ---------------------------
                decatt_lambda = 1.0
                decatt = decatt_loss(model, selected_layers=[0,1,2,9,10,11,12,13,14], lam=decatt_lambda)
                k_proj_lambda = 10.0
                kproj_orth = k_proj_orthogonality_loss(model, selected_layers=[8,9,10,11,12], lam=k_proj_lambda)
                total_loss = total_loss + decatt + kproj_orth
                # -----------------------------------------------------------------------------

                # =========================== Adversarial Injection ===========================
                if (batch_idx + 1) % ADV_INJECT_EVERY == 0:
                    try:
                        adv_images, adv_texts = next(adversarial_iter)
                    except StopIteration:
                        adversarial_iter = iter(adversarial_loader)
                        adv_images, adv_texts = next(adversarial_iter)

                    adv_images = adv_images.to(device)      # [B, C, H, W]
                    adv_texts = adv_texts.to(device)        # [B, 3, seq_len]
                    B, C, H, W = adv_images.shape
                    seq_len = adv_texts.shape[-1]

                    # Repeat images for each label (so [B, C, H, W] -> [B, 3, C, H, W] -> [B*3, C, H, W])
                    adv_images_expanded = adv_images.unsqueeze(1).repeat(1, 3, 1, 1, 1)     # [B, 3, C, H, W]
                    adv_images_flat = adv_images_expanded.view(B * 3, C, H, W)              # [B*3, C, H, W]
                    adv_texts_flat  = adv_texts.view(B * 3, seq_len)                        # [B*3, seq_len]

                    # Get CLIP features
                    adv_image_features = model.encode_image(adv_images_flat)   # [B*3, D]
                    adv_text_features  = model.encode_text(adv_texts_flat)     # [B*3, D]

                    # Cosine similarities for all pairs, reshape to [B, 3]
                    sims = torch.cosine_similarity(adv_image_features, adv_text_features, dim=-1)  # [B*3]
                    sims = sims.view(B, 3)

                    # adv_loss = (sim0 + sim1 + (1 - sim2)).mean()
                    adv_loss = (sims[:, 0] + sims[:, 1] + (1 - sims[:, 2])).mean()
                    total_loss = total_loss + adv_lambda * adv_loss

                # -------------------------------------------------------------


                k_proj_losses_running.append(kproj_orth.item())
                total_kproj_loss += kproj_orth.item()

                acc, f1 = calculate_metrics(logits_per_image, ground_truth)
                train_accs.append(acc)
                train_f1s.append(f1)

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            batch_logits_images.append(logits_per_image.mean().item())
            batch_logits_texts.append(logits_per_text.mean().item())

            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad_norm = parameter.grad.norm().item()
                    gradient_norms.setdefault(name, []).append(grad_norm)

            monitor_gradient_norms(gradient_norms)

            total_train_loss += total_loss.item()
            total_decatt_loss += decatt.item()
            total_krpoj_loss += kproj_orth.item()

            progress_bar.set_postfix({
                'loss': f'{total_train_loss / (batch_idx + 1):.4f}',
                'decatt': f'{total_decatt_loss / (batch_idx + 1):4e}',
                'kproj': f'{total_krpoj_loss / (batch_idx + 1):.4g}',
                '-- Logits': f'{batch_logits_images[-1]:.3f}',
            })
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)
        avg_kproj_loss = total_kproj_loss / len(train_dataloader)
        k_proj_losses.append(avg_kproj_loss)
        plot_k_proj(k_proj_losses, epoch, plots_folder)

        epoch_avg_logits_image = sum(batch_logits_images) / len(batch_logits_images)
        epoch_avg_logits_text = sum(batch_logits_texts) / len(batch_logits_texts)
        logits_images.append(epoch_avg_logits_image)
        logits_texts.append(epoch_avg_logits_text)

        plot_gradient_norms(gradient_norms, epoch)
        plot_k_proj(k_proj_losses, epoch, plots_folder)

        epoch_train_acc = sum(train_accs) / len(train_accs)
        epoch_train_f1 = sum(train_f1s) / len(train_f1s)
        with open(f"{text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
            f.write(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_train_loss:.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS}, K_proj Loss: {avg_kproj_loss}\n")

        model.eval()
        total_val_loss = 0.0
        print("Running Validation...")
        with torch.no_grad():
            for images, texts in val_dataloader:
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                images, texts = images.to(device), texts.to(device)
                logits_per_image, logits_per_text = model(images, texts)
                val_loss = contrastive_loss(logits_per_image, logits_per_text)
                total_val_loss += val_loss.item()
                val_acc, val_f1 = calculate_metrics(logits_per_image, ground_truth)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)

        avg_val_loss = total_val_loss / len(val_dataloader)
        validation_losses.append(avg_val_loss)
        if epoch >= 1:
            plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts)

        epoch_val_acc = sum(val_accs) / len(val_accs)
        epoch_val_f1 = sum(val_f1s) / len(val_f1s)

        if epoch >= 1:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
            plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.savefig(f"{plots_folder}/loss_plot_epoch_{epoch + 1}.png")
            plt.close()

        print(Fore.YELLOW + "======================== STATS =============================")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)

        with open(f"{text_logs_folder}/log_training.txt", "a", encoding='utf-8') as f:
            f.write("======================== STATS =============================\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
            f.write("============================================================\n")

        if dont_save_every_epoch:
            if (epoch + 1) % save_every_n_epochs == 0 or epoch == EPOCHS - 1:
                model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}.pt"
                torch.save(model, model_path)
                print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)
        else:
            model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}.pt"
            torch.save(model, model_path)
            print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)        

trainloop()