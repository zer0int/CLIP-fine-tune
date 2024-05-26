import torch
import gmpclip as clip
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
from itertools import chain
import gc
import numpy as np

def load_model_checkpoint(checkpoint_path):
    model = torch.load(checkpoint_path)
    return model

def extract_geometric_params(model):
    geometric_params = {'theta': [], 'r': []}
    for name, param in model.named_parameters():
        if name.endswith('.theta'):
            geometric_params['theta'].append(param.detach().cpu().numpy())
        elif name.endswith('.r'):
            geometric_params['r'].append(param.detach().cpu().numpy())
    return geometric_params

# Downsampling factor - without, this takes >>100 GB RAM + an hour or so. Factor 10 => 5 minutes (Ryzen 9).
# Adjust factor as needed.
def downsample(data, factor=10):
    return data[::factor]

def visualize_params(params, title, iteration):
    theta_flat = list(chain.from_iterable([item.flatten() for sublist in params['theta'] for item in sublist]))
    r_flat = list(chain.from_iterable([item.flatten() for sublist in params['r'] for item in sublist]))
    
    # Downsample data to reduce memory usage
    theta_flat = downsample(theta_flat)
    r_flat = downsample(r_flat)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(theta_flat, bins=50, kde=True)
    plt.title('Distribution of Theta Components')
    
    plt.subplot(1, 2, 2)
    sns.histplot(r_flat, bins=50, kde=True)
    plt.title('Distribution of R Components')
    
    plt.suptitle(title)
    plt.savefig(f'geometric_params_visualization_{iteration}.png')
    plt.close()

def process_checkpoint(checkpoint, iteration):
    model = load_model_checkpoint(checkpoint)
    params = extract_geometric_params(model)
    visualize_params(params, f'Checkpoint: {checkpoint}', iteration)
    # Explicitly call garbage collection
    gc.collect()

if __name__ == '__main__':
    # List of checkpoints - fine-tuned model saves:
    checkpoints = ["ft-checkpoints/clip_ft_5.pt", "ft-checkpoints/clip_ft_10.pt", 
                   "ft-checkpoints/clip_ft_15.pt", "ft-checkpoints/clip_ft_20.pt"]

    # Split the list of checkpoints into smaller batches if 4 at once consumes too much RAM (RAM, not VRAM!)
    batch_size = 4
    batches = [checkpoints[i:i + batch_size] for i in range(0, len(checkpoints), batch_size)]

    for batch in batches:
        # Parallel processing using concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_checkpoint, checkpoint, i+1) for i, checkpoint in enumerate(batch)]
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
        # Explicitly call garbage collection
        gc.collect()

    print("All visualizations have been generated and saved.")
