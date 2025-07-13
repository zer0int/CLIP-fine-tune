import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from cliptools import fix_random_seed
from colorama import Fore, Style, init as colorama_init
import wcwidth
import argparse
import numpy as np

fix_random_seed()
colorama_init(autoreset=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Example use for HuggingFace Transformers
# This is the exact same as:  ko-eval-tools-zeroshot-attack-multilingual.py
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate Typographic Attack, Pre-Trained vs. Fine-Tuned, Multilingual')
    parser.add_argument('--use_model', default="zer0int/CLIP-KO-TypoAttack-Attn-Dropout-ViT-L-14", help="Local HF CLIP model path or hub name")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model


ft_model = CLIPModel.from_pretrained(model_name_or_path).to(device)
ft_processor = CLIPProcessor.from_pretrained(model_name_or_path)

org_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
org_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

ft_model = ft_model.float()
org_model = org_model.float()

models = {
    "FT CLIP": (ft_model, ft_processor),
    "ORG CLIP": (org_model, org_processor),
}

image_paths = [
    "image_sets/three_cat/bwcat_cat.png",
    "image_sets/three_cat/bwcat_dog.png",
    "image_sets/three_cat/bwcat_notext.png",
]

language_prompts = [
    ("English", ["a photo of a cat"], ["a photo of a dog"]),
    ("German", ["ein foto von einer katze", "foto einer katze", "foto katze"],
               ["ein foto von einem hund", "foto eines hundes", "foto hund"]),
    ("French", ["une photo d'un chat", "photo d'un chat", "une photo de chat"],
               ["une photo d'un chien", "photo d'un chien", "une photo de chien"]),
    ("Spanish", ["una foto de un gato", "foto de un gato"],
               ["una foto de un perro", "foto de un perro"]),
    ("Italian", ["una foto di un gatto", "foto di un gatto"],
               ["una foto di un cane", "foto di un cane"]),
    ("Portuguese", ["uma foto de um gato", "foto de um gato"],
               ["uma foto de um cachorro", "foto de um cachorro"]),
    ("Dutch", ["een foto van een kat", "foto van een kat"],
               ["een foto van een hond", "foto van een hond"]),
    ("Russian", ["фото кошки", "фотография кошки", "foto koshki"],
               ["фото собаки", "фотография собаки", "foto sobaki"]),
    ("Polish", ["zdjęcie kota", "foto kota", "zdjecie kota"],
               ["zdjęcie psa", "foto psa", "zdjecie psa"]),
    ("Turkish", ["bir kedi fotoğrafı", "kedi fotoğrafı"],
               ["bir köpek fotoğrafı", "köpek fotoğrafı"]),
    ("Czech", ["fotografie kočky", "foto kočky"],
               ["fotografie psa", "foto psa"]),
    ("Swedish", ["ett foto av en katt", "foto på en katt"],
               ["ett foto av en hund", "foto på en hund"]),
    ("Finnish", ["kuva kissasta", "kuva kissa"],
               ["kuva koirasta", "kuva koira"]),
    ("Danish", ["et foto af en kat", "foto af en kat"],
               ["et foto af en hund", "foto af en hund"]),
    ("Japanese", ["猫の写真", "neko no shashin"],
               ["犬の写真", "inu no shashin"]),
    ("Korean", ["고양이 사진", "goyangi sajin"],
               ["개 사진", "gae sajin"]),
    ("Chinese", ["一张猫的照片", "猫的照片", "yī zhāng māo de zhàopiàn"],
               ["一张狗的照片", "狗的照片", "yī zhāng gǒu de zhàopiàn"]),
    ("Arabic", ["صورة لقطة", "sura liqitta"],
               ["صورة لكلب", "sura likalb"]),
    ("Greek", ["μια φωτογραφία γάτας", "φωτογραφία γάτας"],
               ["μια φωτογραφία σκύλου", "φωτογραφία σκύλου"]),
    ("Hindi", ["बिल्ली की फोटो", "billi ki photo"],
               ["कुत्ते की फोटो", "kutte ki photo"]),
]

fallback_prompt = ["a photo of a text"]

all_prompts_flat = []
for _, cats, dogs in language_prompts:
    all_prompts_flat.extend(cats)
    all_prompts_flat.extend(dogs)
all_prompts_flat.extend(fallback_prompt)

def get_display_width(s):
    return sum(wcwidth.wcwidth(ch) for ch in s)

def pad_display(s, width, fillchar='·'):
    display_width = get_display_width(s)
    return s + (fillchar * (width - display_width))

max_prompt_len = max(get_display_width(p) for p in all_prompts_flat)

def get_top_indices(arr, n=3):
    arr = torch.tensor(arr)
    return arr.argsort(descending=True)[:n].tolist()

def color_val(val, idx, top_indices, width=10):
    s = f"{val:.4f}".rjust(width)
    if idx == top_indices[0]:
        return Fore.GREEN + Style.BRIGHT + s + Style.RESET_ALL
    elif idx == top_indices[1]:
        return Fore.CYAN + Style.BRIGHT + s + Style.RESET_ALL
    elif idx == top_indices[2]:
        return Fore.YELLOW + Style.BRIGHT + s + Style.RESET_ALL
    else:
        return s

def compute_results(model, processor, image, text_prompts):
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape [1, num_prompts]
        probs = logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()
        return probs

# --- MAIN EVAL LOOP ---
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    print(f"\n========== IMAGE: {image_path} ==========")
    for language, cat_prompts, dog_prompts in language_prompts:
        choices = cat_prompts + dog_prompts + fallback_prompt
        results_per_model = {}
        top_indices_per_model = {}
        for model_name, (model, processor) in models.items():
            probs = compute_results(model, processor, image, choices)
            results_per_model[model_name] = probs
            top_indices_per_model[model_name] = get_top_indices(probs)

        print(f"\nLANGUAGE: {language}")
        header = f"{'Prompt':<{max_prompt_len}}   Label".ljust(max_prompt_len+10)
        for model_name in models:
            header += f"\t    {model_name:10}"
        print(header)
        print("-" * (max_prompt_len+23*len(models)))

        for i, prompt in enumerate(choices):
            label = "(cat)" if i < len(cat_prompts) else "(dog)" if i < len(cat_prompts) + len(dog_prompts) else "(fallback)"
            row = f"{pad_display(prompt, width=max_prompt_len)}   {label:>8}"
            for model_name in models:
                val = results_per_model[model_name][i]
                idxs = top_indices_per_model[model_name]
                row += f"\t{color_val(val, i, idxs, width=10)}"
            print(row)

        for model_name, probs in results_per_model.items():
            top_idx = np.argmax(probs)
            top_prob = probs[top_idx]
            top_prompt = choices[top_idx]
            print(f"Top for {model_name}: '{top_prompt}' ({top_prob:.4f})")
