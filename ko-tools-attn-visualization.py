import os
import glob
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from captum.attr import visualization
from safetensors.torch import load_file
import argparse

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import attnclip as clip
from attnclip.model import convert_state_dict_inproj_to_qkv
from attnclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------
# This code visualizes attention heatmaps
# for the final (output) layer of a model.
# For all layer attentions, see:
# ko-tools-attn-visualization-all-layers.py
# -----------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize CLIP Attention Heatmaps, Text and Image')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--token_folder', default="image_sets/texts_for_attn", help="Folder with gradient ascent .txt files of CLIP's opinions (or yours)")
    parser.add_argument('--image_folder', default="image_sets/images_for_attn", help="Folder with images, matching for .txt files: 'image.png' -> 'tokens_image.txt'")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

if model_name_or_path.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    state_dict = load_file(model_name_or_path)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        msg = str(e)
        if (
            "Missing key(s) in state_dict" in msg or
            "Unexpected key(s) in state_dict" in msg
        ):
            print("State dict format mismatch, attempting QKV conversion...")
            state_dict = convert_state_dict_inproj_to_qkv(state_dict)
            try:
                model.load_state_dict(state_dict)
                print("OK!")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load state_dict after QKV conversion: {e2}\n"
                    f"Original error: {e}"
                )
        else:
            raise
else:
    print("Detected non-.safetensors file or name. Attempting to load model...")
    model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float()

start_layer = -1 # ViT
start_layer_text = -1

image_folder = args.image_folder
token_folder = args.token_folder
heatmap_folder = "results_oai/attn-heatmap"
os.makedirs(heatmap_folder, exist_ok=True)
overlay_token_in_image=True # set to false to get the heatmaps without the text token info overlay
font_size = 20

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):

    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
      # calculate index of last layer
      start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)

    image_relevance = R[:, 0, 1:]
    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
      # calculate index of last layer
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

def show_heatmap_on_text(text, text_encoding, R_text):
  CLS_idx = text_encoding.argmax(dim=-1)
  R_text = R_text[CLS_idx, 1:CLS_idx]
  text_scores = R_text / R_text.sum()
  text_scores = text_scores.flatten()
  #print(text_scores)
  text_tokens=_tokenizer.encode(text)
  text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
  vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]

def show_image_relevance(image_relevance, image, orig_image, img_path):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        alpha_mask = np.array(mask.copy()) 
        alpha_mask=0.5

        cam = np.zeros_like(img)
        for c in range(3):  # Loop through RGB channels
            cam[:, :, c] = alpha_mask * heatmap[:, :, c] + (1 - alpha_mask) * img[:, :, c]

        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    return vis


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# Get all files in the folder
all_files = os.listdir(image_folder)

# Loop through each file
for file in all_files:
    file_path = os.path.join(image_folder, file)

    try:
        img = Image.open(file_path)
        img = img.convert('RGBA')

        new_file_path = os.path.join(image_folder, os.path.splitext(file)[0] + '.png')

        img.save(new_file_path, "PNG")

        if not file_path.endswith('.png'):
            os.remove(file_path)

    except IOError:
        print(f"{file} is not a valid image.")


image_files = glob.glob(f"{image_folder}/*.png")

for img_file in image_files:
    img_name = os.path.basename(os.path.splitext(img_file)[0])

    token_file = f"{token_folder}/tokens_{img_name}.txt"
    with open(token_file, 'r') as f:
        tokens = f.read().split()

    img = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
    print(f"Processing {img_file} tokens...")

    for token in tokens:
        texts = [token]
        text = clip.tokenize(texts).to(device)

        # run the model
        R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
        batch_size = text.shape[0]
        for i in range(batch_size):
            show_heatmap_on_text(texts[i], text[i], R_text[i])
            show_image_relevance(R_image[i], img, orig_image=Image.open(img_file), img_path=img_file)

        heatmap_filename = f"{heatmap_folder}/{img_name}_{token}_ViT{start_layer}_TxT{start_layer_text}.png"
        vis = show_image_relevance(R_image[i], img, orig_image=Image.open(img_file), img_path=img_file)
        cv2.imwrite(heatmap_filename, vis)


if overlay_token_in_image:
    directory = heatmap_folder

    image_files = glob.glob(os.path.join(directory, '*.png'))

    def get_font_in_order(font_names, font_size):
        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, font_size)
            except IOError:
                continue
        raise ValueError(f"None of the fonts {font_names} are available.")

    def extract_text(filename):
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        parts = name_without_ext.split('_', 1)
        if len(parts) > 1:
            return parts[1]
        return name_without_ext


    primary_font_names = [
        "C:/Windows/Fonts/seguiemj.ttf",  # Windows
        "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS/iOS
        "/System/Library/Fonts/Core/AppleColorEmoji@2x.ttc",  # macOS/iOS
        "/System/Library/Fonts/Core/AppleColorEmoji-160px.ttc",  # macOS/iOS
        "/usr/share/fonts/NotoColorEmoji.ttf",  # Linux/Android
        "arialn.ttf",
        "DejaVuSansCondensed.ttf",
        "segoeui.ttf",
        "NotoSans-Regular.ttf",
        "symbola.ttf",
        "arial.ttf"
    ]
    fallback_font_names = [
        "NotoColorEmoji.ttf",  # Common fallback for emojis
        "Apple Color Emoji.ttc",  # Another fallback for macOS/iOS
        "symbola.ttf"  # Fallback for symbols
    ]


    primary_font = get_font_in_order(primary_font_names, font_size)


    def draw_text_with_fallback(draw, text, position, font, fallback_fonts, fill='white'):
        try:
            draw.text(position, text, font=font, fill=fill)
        except UnicodeEncodeError:
            fallback_font = get_font_in_order(fallback_fonts, font.size)
            draw.text(position, text, font=fallback_font, fill=fill)

    for image_file in image_files:
        img = Image.open(image_file).convert('RGBA')
        draw = ImageDraw.Draw(img)

        text_to_write = extract_text(image_file)

        # Write the extracted text into the image with fallback support for emojis
        draw_text_with_fallback(draw, text_to_write, (10, 10), primary_font, fallback_font_names, 'white')

        img.save(image_file)

    print('Done writing filename overlay into images.')