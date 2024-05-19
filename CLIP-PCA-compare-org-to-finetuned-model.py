import torch
import json
import os
import clip
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torch import nn, optim
import numpy as np

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
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        # Flatten the list of labels and tokenize
        labels = ' '.join(labels)
        text = clip.tokenize([labels])

        return image, text.squeeze(0)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

# Insert your validation dataset (from training validation) here (adjust batch_size as you see fit):
val_dataset = ImageTextDataset("path/to/validation/imagefolder", "path/to/my-validation-labels.json", transform=preprocess)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Insert the path to your fine-tuned models .pt here ("original" should match the model you fine-tuned):
models = {
    'original': clip.load('ViT-L/14', device)[0],
    'epoch_05': torch.load('path/to/clip_ft_5.pt').to(device),
    'epoch_10': torch.load('path/to/clip_ft_10.pt').to(device),
    'epoch_15': torch.load('path/to/clip_ft_15.pt').to(device),
    'epoch_20': torch.load('path/to/clip_ft_20.pt').to(device),
    'epoch_25': torch.load('path/to/clip_ft_20.pt').to(device),
}

def pca_on_activations(model, dataloader):
    model.eval()
    activations = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            features = model.encode_image(images)
            activations.append(features.cpu().numpy())
    activations = np.vstack(activations)
    pca = PCA(n_components=2)
    reduced_activations = pca.fit_transform(activations)
    return reduced_activations

def plot_pca(reduced_activations, title, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_activations[:, 0], reduced_activations[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(f"{model_name}_PCA_Plot.png")
    plt.close()

for name, model in models.items():
    reduced_activations = pca_on_activations(model, val_dataloader)
    plot_pca(reduced_activations, f"{name} PCA Plot", name)
