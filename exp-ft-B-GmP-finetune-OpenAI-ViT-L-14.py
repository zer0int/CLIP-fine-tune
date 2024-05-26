import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import random
from colorama import Fore, Style
from tqdm import tqdm
from adabelief_pytorch import AdaBelief
from torch.nn.utils import clip_grad_norm_
# If you change the following line to "import clip", you can use this script as-is without GmP / for a "normal" fine-tune!
import gmpclip as clip
training_losses = []
validation_losses = []
print("\n")


# Save training plots with matplotlib to:
plots_folder = 'ft-plots'
os.makedirs(plots_folder, exist_ok=True)

# Save model .pt files to: 
ft_checkpoints_folder = 'ft-checkpoints'
os.makedirs(ft_checkpoints_folder, exist_ok=True)

# Save verbose text / training logs to:
text_logs_folder = 'ft-logs'
os.makedirs(text_logs_folder, exist_ok=True)



def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    """
    Adjusts the rate of unfreezing after a certain number of epochs.
    :param epoch: Current epoch number.
    :param adjust_after: Epoch after which to increase unfreezing rate.
    :param increase_rate: How many layers to unfreeze per epoch after adjust_after.
    :return: Number of layers to unfreeze per epoch.
    """
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
        if mean_norm < threshold:  # Vanishing gradient approaching, warning threshold.
            alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        elif mean_norm > 8e+5:  # Nearing exploding gradient threshold.
            alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
    if alert_messages:
        for message in alert_messages:
            print(message)


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
    #plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
    
    # Adjust legend: position at top right with smaller font size
    plt.legend(loc='upper right', fontsize='small')
    
    # If log scale is requested, change the y-axis to logarithmic
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
        plt.plot(epochs_x, logits_texts, label='')
    plt.title('Average Logits Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()
    

def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1

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
        
        """
        Uses a random choice of multiple labels, if available.
        Example:
        todo: insert example here
        """
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]  # Fallback to the first label if less than 2 are available
        else:
            label = ''  # Fallback if no labels are available

        text = clip.tokenize([label])  # Tokenize the label

        return image, text.squeeze(0)  # Remove the extra dimension
        
        
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

contrastive_loss = ContrastiveLoss(temperature=0.07)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


clipmodel = 'ViT-L/14'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clipmodel, device=device)

model = model.float()

unfreeze_all = True

# Quote from https://arxiv.org/abs/2305.15912v4 : 
# GmP+IMN (the green curve) converges significantly faster than other compared methods: 
# Its top-5 validation accuracy converges within 25 epochs, which is 10 epochs earlier than the second best method BN
# Endofquote. Alas, 20 may be enough!

EPOCHS = 20
learning_rate = 3e-7 # only used if custom param_groups NOT used in optimizer (NOT recommended for GmP fine-tune).
max_learning_rate = 5e-7 # for scheduler.
batch_size = 40 # if you used to fine-tune normal ViT-L/14 with batch size 48, then 40 may be a good start. GmP needs slightly more VRAM.

# Define your training dataset(s) and dataloader
dataset1 = ImageTextDataset("F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square", "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square/short-coco-sprite-train-0_9.json", transform=preprocess)
#dataset2 = ImageTextDataset("path/to/images", "path/to/labels.json", transform=preprocess)

concatenated_dataset = ConcatDataset([dataset1])  # Add more datasets to this list as needed ([dataset1, dataset2]) 
train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset and dataloader
val_dataset = ImageTextDataset("F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square", "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square/short-coco-sprite-val-10_11.json", transform=preprocess)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

total_steps = len(train_dataloader) * EPOCHS


# Define parameter groups for different learning rates so we don't have to explicitly define this for all resblocks
visual_parameters = [p for p in model.visual.transformer.parameters() if p.requires_grad]
transformer_parameters = [p for p in model.transformer.parameters() if p.requires_grad]


# Differential learning rate config. Use caution & check friendly red debug gradient norm value warnings in command line for "inf".
# If "inf" happens in epoch 0, this seems to be acceptable - but "inf" should not happen in later epochs.
# Excellent quality, very diverse dataset, generous shared-mem batch_size: You may try: for-all 1e-7 -> 8e-7 and for 8e-8 > 1e-7
param_groups = [
    {'params': visual_parameters, 'lr': 1e-7},
    {'params': transformer_parameters, 'lr': 1e-7},
    {'params': model.text_projection, 'lr': 1e-7},
    {'params': model.token_embedding.parameters(), 'lr': 1e-7},
    {'params': model.positional_embedding, 'lr': 1e-7},
    {'params': [model.visual.class_embedding, model.visual.positional_embedding, model.visual.proj], 'lr': 8e-8}, # Mess with these muchly ...
    {'params': [model.visual.ln_pre.weight, model.visual.ln_pre.bias, model.visual.ln_post.weight, model.visual.ln_post.bias], 'lr': 8e-8}, #...
    {'params': [model.ln_final.weight, model.ln_final.bias, model.visual.conv1.weight], 'lr': 8e-8}  # and gradients *will* explode.
]

# Just gonna leave this here for a "normal optimizer" template. Not recommended for use.             
# optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.995), eps=1e-6, weight_decay=1e-2)


# Use AdaBelief optimizer WITH THE ABOVE param_groups custom learning rates 
optimizer = AdaBelief(param_groups, lr=learning_rate, eps=1e-16, betas=(0.9, 0.998), weight_decay=1e-3, weight_decouple=True, rectify=True, print_change_log=False)

# Combining warm-up with a slower decay might help overfitting -> 0.3 instead of 0.1
# Also in cosine annealing, try 'linear' + adjust pct_start if "first epochs it's fine but then for no reason it just does [something bad]!"
scheduler = OneCycleLR(optimizer, max_lr=max_learning_rate, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')


print(Fore.RED + 'Find "# OPTIONAL DEBUG" in code to #comment out these RED GRADIENT WARNINGS' + Style.RESET_ALL)
print(Fore.YELLOW + "Large gradients are expected with GmP reconfiguration.\n" + Style.RESET_ALL)
print(Fore.CYAN + "It seems 'inf' gradient norms in Epoch 0 are ok (but should NOT be 'inf' in later epochs)!\n" + Style.RESET_ALL)

model = model.float()
print(f"Precision: {model.dtype}")
print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
print("== START == \n")


def trainloop():
    contrastive_loss = ContrastiveLoss(temperature=0.07).to(device)
    logits_images = []
    logits_texts = []
    
    accumulation_steps = 2  # Adjust as needed, but I found 4 to already be bad.
    scaler = GradScaler()    
    
    for epoch in range(EPOCHS):
        gradient_norms = {}
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss = 0.0
        train_dataloader_prog = train_dataloader
        train_dataloader_all = train_dataloader
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)
        optimizer.zero_grad() # Reset gradients at the beginning of the epoch                                                      
        for batch_idx, (images, texts) in progress_bar:
            images, texts = images.to(device), texts.to(device)
            train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
            batch_logits_images = []
            batch_logits_texts = []
               
            with autocast():
                logits_per_image, logits_per_text = model(images, texts)
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                total_loss = contrastive_loss(logits_per_image, logits_per_text)
                acc, f1 = calculate_metrics(logits_per_image, ground_truth)
                train_accs.append(acc)
                train_f1s.append(f1)

            scaler.scale(total_loss).backward()       
                   
            # Perform optimizer step if accumulation steps are met
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scheduler.get_last_lr()
                scheduler.print_lr(is_verbose=False, group=param_groups, lr=learning_rate)
                scaler.step(optimizer)
                scaler.update()
                scheduler.get_last_lr()
                scheduler.print_lr(is_verbose=False, group=param_groups, lr=learning_rate)
                optimizer.zero_grad()  # Reset gradients after optimizer step
                scheduler.step()
                
            batch_logits_images.append(logits_per_image.mean().item())
            batch_logits_texts.append(logits_per_text.mean().item())
                       
            # Store gradient norms for plot
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad_norm = parameter.grad.norm().item()
                    gradient_norms.setdefault(name, []).append(grad_norm)
            
            # OPTIONAL DEBUG
            monitor_gradient_norms(gradient_norms)
            
            total_train_loss += total_loss.item()

            progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}  --  Logits Image: {batch_logits_images[-1]:.3f}, Text: {batch_logits_texts[-1]:.3f}'})

            epoch_train_acc = sum(train_accs) / len(train_accs)
            epoch_train_f1 = sum(train_f1s) / len(train_f1s)
            with open(f"{text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
                f.write(f"Epoch {epoch + 1}/{EPOCHS}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}\n")
   
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)
        
        epoch_avg_logits_image = sum(batch_logits_images) / len(batch_logits_images)
        epoch_avg_logits_text = sum(batch_logits_texts) / len(batch_logits_texts)
        logits_images.append(epoch_avg_logits_image)
        logits_texts.append(epoch_avg_logits_text)
        
        plot_gradient_norms(gradient_norms, epoch)      

        # Validation
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

        # Save model every 5 epochs + save final model
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}.pt"
            torch.save(model, model_path)      
            print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)
            
            
trainloop()