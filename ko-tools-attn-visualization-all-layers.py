import os
import glob
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from captum.attr import visualization
from tqdm import tqdm
import warnings
import urllib.parse
from safetensors.torch import load_file
import argparse

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
# for ** ALL LAYERS ** of the model.
# Compares original ViT-L/14 to fine-tune.
# -----------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize CLIP Attention Heatmaps, Text and Image')
    parser.add_argument('--use_model', default="models/ViT-L-14-KO-FULL-model-OpenAI-format.safetensors", help="Name or path to CLIP Model")
    parser.add_argument('--token_folder', default="image_sets/special_attn_txt", help="Folder with gradient ascent .txt files of CLIP's opinions (or yours)")
    parser.add_argument('--image_folder', default="image_sets/special_attn_img", help="Folder with images, matching for .txt files: 'image.png' -> 'tokens_image.txt'")
    return parser.parse_args()

args = parse_arguments()
finetune_model_name = args.use_model

# ==== MODEL CHOICES ====
MODEL_CONFIGS = [
    {"name": finetune_model_name, "alias": "KO"},
    {"name": "ViT-L/14", "alias": "ORG"},
]

# ==== INPUT PATHS ====
image_folder = args.image_folder
token_folder = args.token_folder
heatmap_folder = "results_oai/attn-heatmap-all-layers"
os.makedirs(heatmap_folder, exist_ok=True)
font_size = 20

def sanitize_for_filename(text, maxlen=40):
    # URL-encode for safety and readability, and truncate
    enc = urllib.parse.quote(text, safe="")
    return enc[:maxlen]

# ==== HEATMAP UTILS ====
def interpret(image, texts, model, device, start_layer, start_layer_text):
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

def show_image_relevance(image_relevance, image, orig_image, img_path):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        alpha_mask = 0.5
        cam = np.zeros_like(img)
        for c in range(3):
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

# ==== FONT LOGIC FOR OVERLAY ====
primary_font_names = [
    "C:/Windows/Fonts/seguiemj.ttf",  # Windows
    "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS/iOS
    "/System/Library/Fonts/Core/AppleColorEmoji@2x.ttc",
    "/System/Library/Fonts/Core/AppleColorEmoji-160px.ttc",
    "/usr/share/fonts/NotoColorEmoji.ttf",  # Linux/Android
    "arialn.ttf",
    "DejaVuSansCondensed.ttf",
    "segoeui.ttf",
    "NotoSans-Regular.ttf",
    "symbola.ttf",
    "arial.ttf"
]
fallback_font_names = [
    "NotoColorEmoji.ttf",
    "Apple Color Emoji.ttc",
    "symbola.ttf"
]

def get_font_in_order(font_names, font_size):
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except IOError:
            continue
    raise ValueError(f"None of the fonts {font_names} are available.")

primary_font = get_font_in_order(primary_font_names, font_size)

def draw_text_with_fallback(draw, text, position, font, fallback_fonts, fill='white'):
    try:
        draw.text(position, text, font=font, fill=fill)
    except UnicodeEncodeError:
        fallback_font = get_font_in_order(fallback_fonts, font.size)
        draw.text(position, text, font=fallback_font, fill=fill)

def extract_token_from_filename(filename):
    # expects ..._ViTxx_<token_enc>_<imgname>.png
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    parts = name_without_ext.split('_')
    if len(parts) < 4:
        return name_without_ext
    token_enc = parts[2]
    # Decode the URL-encoded string
    try:
        token_dec = urllib.parse.unquote(token_enc)
    except Exception:
        token_dec = token_enc
    return token_dec

for model_cfg in MODEL_CONFIGS:
    print(f"\n== Processing model: {model_cfg['alias']} ==")

    if model_cfg['name'].endswith(".safetensors"):
        print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
        model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
        state_dict = load_file(model_cfg['name'])
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
        model, preprocess = clip.load(model_cfg['name'], device=device, jit=False)

    model = model.eval().float()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_vit_layers = len(image_attn_blocks)
    image_files = glob.glob(f"{image_folder}/*.png")
    for img_file in tqdm(image_files, desc=f"{model_cfg['alias']}"):
        img_name = os.path.basename(os.path.splitext(img_file)[0])
        token_file = f"{token_folder}/tokens_{img_name}.txt"
        if not os.path.exists(token_file):
            continue
        with open(token_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        img = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
        if len(lines) == 1:
            tokens = lines[0].split()
            for token in tokens:
                texts = [token]
                text = clip.tokenize(texts).to(device)
                for vit_layer in range(num_vit_layers):
                    # Encode the token for safe filename (e.g., emoji, unicode)
                    token_enc = sanitize_for_filename(token)
                    heatmap_filename = os.path.join(
                        heatmap_folder,
                        f"{model_cfg['alias']}_ViT{vit_layer:02d}_{token_enc}_{img_name}.png"
                    )
                    R_text, R_image = interpret(model=model, image=img, texts=text, device=device, start_layer=vit_layer, start_layer_text=-1)
                    vis = show_image_relevance(R_image[0], img, orig_image=Image.open(img_file), img_path=img_file)
                    cv2.imwrite(heatmap_filename, vis)
        else:
            for i, line in enumerate(lines):
                texts = [line]
                text = clip.tokenize(texts).to(device)
                for vit_layer in range(num_vit_layers):
                    line_enc = sanitize_for_filename(line)
                    heatmap_filename = os.path.join(
                        heatmap_folder,
                        f"{model_cfg['alias']}_ViT{vit_layer:02d}_{line_enc}_{img_name}.png"
                    )
                    R_text, R_image = interpret(model=model, image=img, texts=text, device=device, start_layer=vit_layer, start_layer_text=-1)
                    vis = show_image_relevance(R_image[0], img, orig_image=Image.open(img_file), img_path=img_file)
                    cv2.imwrite(heatmap_filename, vis)


# ==== TOKEN/LAYER/MODEL OVERLAY + UPSCALE ====
if True:
    directory = heatmap_folder
    image_files = glob.glob(os.path.join(directory, '*.png'))
    for image_file in tqdm(image_files, desc="Overlay Text + Upscale"):
        img = Image.open(image_file).convert('RGBA')
        img = img.resize((img.width * 2, img.height * 2), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        text_to_write = extract_token_from_filename(image_file)
        draw_text_with_fallback(draw, text_to_write, (20, 20), primary_font, fallback_font_names, 'white')
        img.save(image_file)
    print('Done writing overlay and upscaling images.')
