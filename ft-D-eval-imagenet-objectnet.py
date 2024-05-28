import torch
from torch.utils.data import Dataset, DataLoader
# Import original CLIP code with modification to bypass SHA256 checksum verification
# Don't use this to load arbitrary third-party models, google "pickle vulnerability" for details
# However, this allows you to use clip.load on your own (safe) fine-tuned model:
from orgclipnosha import clip
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os

# Download from https://objectnet.dev/mvt/ 

# With "normal" fine-tuning, your model is expected to overfit on your dataset and alas,
# become worse at generalizing with the above dataset. This is NORMAL.
# Accuracy of 0.5 - 0.7 generally means "good preservation of pre-training",
# likely a great model when using this CLIP as TE for SDXL exclusively.
# Accuracy of 0.3 can be great if you trained some "glitch art" / weird dataset model.
# Accuracy <0.1: No. You ruined the model, that won't be a good guide / TE.

# I am mainly adding this code for replication of my GmP-CLIP results.
# Make sure you use the GmP fine-tuned model after "convert-GmP-back-to-weight" below:

# Load csv labels file from dataset:
csv_file = 'path/to/mvt/dataset/data_release_2023/human_responses.csv'

clipmodel = 'ViT-L/14'
# Your fine-tuned model below:
finetunedclip = "ft-checkpoints/clip_ft_20.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
original_model, preprocess = clip.load(clipmodel, device=device, jit=False)
finetuned_model, preprocess = clip.load(finetunedclip, device=device)

# Dataset class to load images and their corresponding labels from CSV
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
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx]['label']

        return image, label

# Path to the image folder that contains ALL images from the MVT dataset:
image_folder = 'path/to/mvt/dataset/data_release_2023/all/'

# Create dataset and dataloader
dataset = CroppedImageCSVFileDataset(csv_file, image_folder, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=48, shuffle=True)

# Function to evaluate model on custom dataset
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

            # Get the top predictions
            _, top_indices = logits_per_image.topk(1, dim=-1)

            for i, label in enumerate(batch_labels):
                if label == batch_labels[top_indices[i]]:
                    correct += 1
                total += 1

    accuracy = correct / total
    return accuracy

# Evaluate original and fine-tuned models
original_accuracy = evaluate_model(original_model, dataloader)
finetuned_accuracy = evaluate_model(finetuned_model, dataloader)

print(f"Original Model Accuracy on MVT ImageNet/ObjectNet: {original_accuracy}")
print(f"Fine-tuned Model Accuracy on MVT ImageNet/ObjectNet: {finetuned_accuracy}")
