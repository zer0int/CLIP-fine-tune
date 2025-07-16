import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cosine_hilarity
from safetensors.torch import load_file
import argparse
from tqdm import tqdm

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longclip as clip
from longclip.model import CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------------------
# Measures Modality Gap, JSD, Wasserstein Distance, Text-Image Cosine Similarity 
#
# Requires Flickr8k dataset images, get them from e.g.:
# https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
#
# Make sure to unzip the images into an 'Images' folder, e.g.:
# C:/my_datasets/Flickr8k/Images
# then use the command-line argument to point to the base directory:
# --data_dir C:/my_datasets/Flickr8k
# --------------------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate Modality Gap on flickr8k')
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-KO-LITE-FULL-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--data_dir', default="path/to/base/Flickr8k", help="Path to base dataset base dir, should have 'Images' subdirectory")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

prefix = "oai-clip" # Will save embeds to subfolder of base_dir
base_dir = args.data_dir # Flickr8k dataset dir
images_dir = os.path.join(base_dir, "Images")
captions_file = "image_sets/captions_flickr8k.txt"
embeddings_path = os.path.join(base_dir, f"embeddings-{prefix}")
output_path = "results_oai/modality-gap"
os.makedirs(output_path, exist_ok=True)
os.makedirs(embeddings_path, exist_ok=True)


model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float() # full precision

print("\nComputing embeddings...")

# Load captions
captions_dict = {}
with open(captions_file, "r") as f:
    for line in f:
        image_name, caption = line.strip().split(',', 1)
        if image_name not in captions_dict:
            captions_dict[image_name] = []
        captions_dict[image_name].append(caption)

# Initialize lists to store embeddings
image_embeddings = []
text_embeddings = []
all_text_embeddings = []

# Wrap outer loop with tqdm for image progress
for image_name, captions in tqdm(captions_dict.items(), desc="Images", unit="img"):
    # Load image
    image_path = os.path.join(images_dir, image_name)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Compute image embedding once
    with torch.no_grad():
        image_embed = model.encode_image(image)
        image_embed /= image_embed.norm(dim=-1, keepdim=True)

    # Save image embedding
    image_embed_filename = os.path.join(embeddings_path, f"{image_name}.pt")
    torch.save(image_embed.cpu(), image_embed_filename)
    
    # Add image embedding to list for modality gap measurement
    image_embeddings.append(image_embed.cpu().numpy())

    # Wrap inner loop with tqdm for caption progress (nested, appears as a bar per image)
    for i, caption in enumerate(tqdm(captions, desc=f"{image_name} captions", unit="cap", leave=False), 1):
        with torch.no_grad():
            text_embed = model.encode_text(clip.tokenize(caption).to(device))
            text_embed /= text_embed.norm(dim=-1, keepdim=True)

        # Save text embedding
        text_embed_filename = os.path.join(embeddings_path, f"{image_name}_caption_{i}.pt")
        torch.save(text_embed.cpu(), text_embed_filename)
        
        # Add text embedding to list for modality gap measurement
        text_embeddings.append(text_embed.cpu().numpy())

# Convert lists to numpy arrays
image_embeddings = np.vstack(image_embeddings)
text_embeddings = np.vstack(text_embeddings)

print("Calculating Modality Gap...")

# Calculate modality gap
image_center = np.mean(image_embeddings, axis=0)
text_center = np.mean(text_embeddings, axis=0)
modality_gap = np.linalg.norm(image_center - text_center)

print(f"\n-------> Modality Gap is: {modality_gap:.4f}\n")

print("Calculating JSD, plots and more...")

# Function to load embeddings
def load_embedding(filepath):
    return torch.load(filepath).numpy()

# Initialize lists to store cosine similarities
image_text_cosine_values = []
text_text_cosine_values = []
image_image_cosine_values = []

# Get a list of all embedding files
embedding_files = os.listdir(embeddings_path)

# Extract unique image base names
image_files = [f for f in embedding_files if f.endswith('.pt') and '_caption_' not in f]
image_base_names = [os.path.splitext(f)[0] for f in image_files]

# Process each image's embeddings
for image_file, image_base_name in zip(image_files, image_base_names):
    # Load image embedding
    image_embedding_path = os.path.join(embeddings_path, image_file)
    image_embedding = load_embedding(image_embedding_path)
    
    # L2 Normalize embeddings
    image_embedding = image_embedding / np.linalg.norm(image_embedding)

    # Initialize list to store text embeddings for current image
    text_embeddings = []
    
    # Load corresponding text embeddings
    for i in range(1, 6):  # There are 5 captions per image
        caption_filename = f"{image_base_name}_caption_{i}.pt"
        caption_filepath = os.path.join(embeddings_path, caption_filename)
        
        # Check if caption file exists
        if not os.path.isfile(caption_filepath):
            print(f"Caption file not found: {caption_filepath}")
            continue
        
        caption_embedding = load_embedding(caption_filepath)
        caption_embedding = caption_embedding / np.linalg.norm(caption_embedding)  # L2 Normalize

        text_embeddings.append(caption_embedding)
        all_text_embeddings.append(caption_embedding)

        
        # Calculate cosine similarity properly
        #cosine_similarity = np.dot(image_embedding, caption_embedding)
        cosine_similarity = np.dot(image_embedding.reshape(-1), caption_embedding.reshape(-1))

        image_text_cosine_values.append(cosine_similarity)
    
    # Calculate text-text cosine similarities (pairwise for 5 captions)
    for i in range(len(text_embeddings)):
        for j in range(i + 1, len(text_embeddings)):
            cosine_similarity = np.dot(text_embeddings[i].reshape(-1), text_embeddings[j].reshape(-1))
            text_text_cosine_values.append(cosine_similarity)

# Convert lists to numpy arrays
image_text_cosine_values = np.array(image_text_cosine_values)
text_text_cosine_values = np.array(text_text_cosine_values)

# Calculate mean and standard deviation
image_text_mean = np.mean(image_text_cosine_values)
image_text_std = np.std(image_text_cosine_values)
text_text_mean = np.mean(text_text_cosine_values)
text_text_std = np.std(text_text_cosine_values)

print(f"----------------------------------- RESULTS -----------------------------------")
print(f"Model: {prefix} - Modality Gap (Euclidean): {modality_gap:.4f}")
print(f"Model: {prefix} - Image-Text Cosine Similarity - Mean: {image_text_mean:.4f}, Std Dev: {image_text_std:.4f}")
print(f"Model: {prefix} - Text-Text Cosine Similarity - Mean: {text_text_mean:.4f}, Std Dev: {text_text_std:.4f}")

# Determine optimal bin count
num_bins = np.histogram_bin_edges(image_text_cosine_values, bins='fd').size

# Compute histograms with optimal bins
hist_image_text, bin_edges = np.histogram(image_text_cosine_values, bins=num_bins, density=True)
hist_text_text, _ = np.histogram(text_text_cosine_values, bins=num_bins, density=True)

# Normalize histograms properly
epsilon = 1e-10  # To avoid division errors
p_image_text = np.clip(hist_image_text / (np.sum(hist_image_text) + epsilon), epsilon, 1)
p_text_text = np.clip(hist_text_text / (np.sum(hist_text_text) + epsilon), epsilon, 1)

# Compute JSD
jsd = jensenshannon(p_image_text, p_text_text)
print(f"Model: {prefix} - Jensen-Shannon Divergence (JSD): {jsd:.4f}")

# Compute Wasserstein Distance (Earth Mover’s Distance)
wasserstein_dist = wasserstein_distance(image_text_cosine_values, text_text_cosine_values)
print(f"Model: {prefix} - Wasserstein Distance: {wasserstein_dist:.4f}")

print(f"-------------------------------------------------------------------------------")

with open(f"{output_path}/modality-gap.txt", "a", encoding='utf-8') as f:
    f.write(f"Model: {prefix} - Modality Gap (Euclidean): {modality_gap:.4f}\n")
    f.write(f"Model: {prefix} - Image-Text Cosine Similarity - Mean: {image_text_mean:.4f}, Std Dev: {image_text_std:.4f}\n")
    f.write(f"Model: {prefix} - Text-Text Cosine Similarity - Mean: {text_text_mean:.4f}, Std Dev: {text_text_std:.4f}\n")
    f.write(f"Model: {prefix} - Jensen-Shannon Divergence (JSD): {jsd:.4f}\n")
    f.write(f"Model: {prefix} - Wasserstein Distance: {wasserstein_dist:.4f}\n")

# Plot overlay histogram (density)
plt.figure(figsize=(10, 6))
plt.hist(text_text_cosine_values, bins=num_bins, color='orange', alpha=0.5, label='Text-Text', density=True)
plt.hist(image_text_cosine_values, bins=num_bins, color='blue', alpha=0.5, label='Image-Text', density=True)
plt.title(f'Overlay of Image-Text vs. Text-Text Cosine Similarities ({prefix})')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.savefig(f'{output_path}/{prefix}-cos-sim_density.png')
plt.close()

# Convert lists to numpy arrays explicitly
text_embeddings = np.array(all_text_embeddings).squeeze()
image_embeddings = np.array(image_embeddings).squeeze()

# Confirm dimensions
print("Text embeddings shape:", text_embeddings.shape)
print("Image embeddings shape:", image_embeddings.shape)

# Fix embeddings shape mismatch
text_embeddings_flat = text_embeddings.reshape(text_embeddings.shape[0], -1)
image_embeddings_flat = image_embeddings.reshape(image_embeddings.shape[0], -1)

# Combine embeddings for t-SNE
all_embeddings = np.vstack([text_embeddings_flat, image_embeddings_flat])

# Combine embeddings for t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Split back into text and image embeddings
text_emb_2d = embeddings_2d[:len(text_embeddings)]
image_emb_2d = embeddings_2d[len(text_embeddings):]

# Plot embeddings
plt.figure(figsize=(12, 8))
plt.scatter(text_emb_2d[:, 0], text_emb_2d[:, 1], c='blue', alpha=0.5, label='Text')
plt.scatter(image_emb_2d[:, 0], image_emb_2d[:, 1], c='red', alpha=0.5, label='Images')

# Calculate pairwise cosine similarity
cos_sims = cosine_hilarity(text_embeddings, image_embeddings)

# Find highest, lowest, and median cosine similarities
max_idx = np.unravel_index(np.argmax(cos_sims), cos_sims.shape)
min_idx = np.unravel_index(np.argmin(cos_sims), cos_sims.shape)
median_val = np.median(cos_sims)
median_idx = np.unravel_index(np.argmin(np.abs(cos_sims - median_val)), cos_sims.shape)

pairs = [(max_idx, 'solid'), (min_idx, 'dashed'), (median_idx, 'dotted')]

# Connect highlighted pairs
for (text_i, img_i), linestyle in pairs:
    text_point = text_emb_2d[text_i]
    image_point = image_emb_2d[img_i]
    plt.plot([text_point[0], image_point[0]], [text_point[1], image_point[1]], linestyle=linestyle, color='black')
    plt.text((text_point[0]+image_point[0])/2, (text_point[1]+image_point[1])/2, 
             f"{cos_sims[text_i, img_i]:.3f}", fontsize=10, color='black', backgroundcolor='white')

plt.title('2D t-SNE of Text and Image Embeddings with Highlighted Pairs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_path}/{prefix}-tsne_modality_gap.png')
plt.close()

print(f"Done! Results saved to {output_path}.")