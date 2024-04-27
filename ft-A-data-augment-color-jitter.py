import os
import torch
from torchvision import transforms
from PIL import Image

input_dir = 'my-images'
output_dir = 'my-augmented-images'

os.makedirs(output_dir, exist_ok=True)
moderate_jitter_transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

def apply_and_save_jitter(image_path, transform, suffix):
    image = Image.open(image_path)
    jittered_image = transform(image)
    save_path = os.path.join(output_dir, (os.path.basename(image_path)))
    jittered_image.save(save_path)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)        
        apply_and_save_jitter(image_path, moderate_jitter_transform, 'moderate')

print(f"Processing complete. Jittered images saved in {output_dir} folder.")
