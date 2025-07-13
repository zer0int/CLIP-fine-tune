import os
import random
from PIL import Image, ImageDraw, ImageFont
import math

# --------------------------------------------------
# Simple image editing tool:
# Generate a DIY 'typographic attack' by writing
# a word on an image (ImageNet class) to fool CLIP.
# --------------------------------------------------

def overlay_text_on_folder(
    input_dir, 
    output_dir=None, 
    word="word", 
    font_size=32,
    n_positions=3
):
    os.makedirs(output_dir, exist_ok=True)
    font_candidates = ["arial.ttf", "DejaVuSans-Bold.ttf"]
    font = None
    for fnt in font_candidates:
        try:
            font = ImageFont.truetype(fnt, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        shortest_side = min(width, height)
        # Define top-left corner of central square
        x0 = (width  - shortest_side) // 2
        y0 = (height - shortest_side) // 2


        # Get text size for spacing
        bbox = font.getbbox(word)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        spacing = text_w + 10

        # Sample positions
        positions = []
        attempts = 0
        while len(positions) < n_positions and attempts < 1000:
            x = random.randint(x0, x0 + shortest_side - spacing)
            y = random.randint(y0, y0 + shortest_side - spacing)
            if all(math.hypot(x - px, y - py) > spacing for px, py in positions):
                positions.append((x, y))
            attempts += 1

        # Draw text (with shadow)
        directions = [(-1, -1), (0, -1), (1, -1),
                      (-1,  0),         (1,  0),
                      (-1,  1), (0,  1), (1,  1)]
        draw = ImageDraw.Draw(img)
        for x, y in positions:
            for dx, dy in directions:
                draw.text((x + dx, y + dy), word, font=font, fill=(0, 0, 0))
            draw.text((x, y), word, font=font, fill=(255, 255, 255))

        # Save
        out_path = os.path.join(output_dir, fname)
        img.save(out_path, quality=100)
        print(f"[✓] Saved {out_path} with {n_positions}× '{word}' overlays")

if __name__ == "__main__":
    # Usage: path/to/imagenet/wnid + output_dir for adversarial variant with 'word' written on it
    # E.g. cats -> write word='dog' or (for color confusion!) use goldfinch wird word 'bumblebee'
    overlay_text_on_folder("image_sets/n01531178", output_dir="image_sets/n01531178_adv", word="bumblebee", font_size=32, n_positions=3)
