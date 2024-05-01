### Changes 01/May/24:
- Added misc conversion scripts for dataset labels as examples.
- Added YOLO dataset AI auto pre-processing example scripts.
- Download YOLOv7 weights: [https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.weights](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.weights)
- Download YOLOv4 weights: [https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
------

# Fine-tuning code for CLIP! 🤩

- Originally made for Long-CLIP, see [zer0int/Long-CLIP](https://github.com/zer0int/Long-CLIP) for fine-tuning Long-CLIP
- This repo is for fine-tuning the original OpenAI/CLIP models!

Optimized for: ViT-L/14 (Text Encoder of SD / SDXL) + *I have 1 NVIDIA GPU with 24 GB VRAM available...* 😅
But you can train any OpenAI/CLIP model with this (just remember to tweak batch_size etc. for smaller model, if applicable!).

### You won't win benchmarks with throwing small batch_sizes at a big model such as ViT-L/14; but using a finetune as the text encoder for e.g. Stable Diffusion SDXL, this CLIP will win some hearts! 💙🤖

+ Uses AMP (automatic mixed precision) + AdaBelief optimizer (optional: fall back to AdamW) + OneCycleLR scheduler with warmup
+ Gradually unfreeze CLIP (optional) or train whole model (default) + set Learning Rate for individual parameters (optional)
+ Debug print when exploding or vanishing gradients occur + Many fancy logs and plots with live training updates

# How to use:

### 0. Install the dependencies from requirements-finetune.txt.

### 1. ft-A-clip-interrogator-csv-to-json-labels.py
- Converts a "desc.csv" from [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator) to dataset labels .json.
- Example: ft-X-example-my-dataset-labels.json is the expected format for my fine-tuning script; if you have a different format - e.g. single text files next to images - explain that to GPT-4, Claude 3, or any other AI assistant + "and I need to convert them to be labels in a single .json file that should look like so:" *copy-paste the content of ft-X-example-my-dataset-labels.json into prompt as a one-shot example*
- If you load your dataset: dataset1 = ImageTextDataset("path/to/image/folder", "path/to/my-text-labels.json", transform=preprocess), and inside the .json images are: "subpath/to/0001.jpg" -> then the script dataloader will look for the image in "path/to/image/folder/subpath/to/0001.jpg".

### 2. ft-A-augment-data-color-jitter.py
- Data augmentation: If your dataset is ~1000 images, consider augmenting the images by flipping them horizontally etc.
- The script example will create a copy of your images with color jitter, which prevents CLIP from overfitting on specific colors.
- Use augmented images with .json labels and randomly select from multiple labels for a given image. See code in (3) for details.

### 3. ft-B-train-OpenAI-CLIP-ViT-L-14.py
- Fine-tune CLIP. Insert dataset .json and path to images as per previous step. See code # comments for details.
- 10,000 text-image pairs can archive good fine-tuning results within 1-2 hours (RTX 4090).

### 4. ft-C-convert-for-SDXL-comfyUI-OpenAI-CLIP.py
- Convert the torch.save model .pt into a state_dict you can then just plug into SDXL as the text encoder.
- Easy as Pi with ComfyUI, see [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) for details!

![instructions-comfyui](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/ed80001f-00b0-4915-b2de-19907d350d06)

### 5. Example benefit of fine-tuning CLIP: Crazy "DeepDream of CLIP's own Neurons" dataset. Don't ask. ;-)
- Same random seed etc., just swapping out the original CLIP text encoder for my fine-tune. CFG scale 14 = high CLIP influence / guidance.
- Please note: The U-Net of SDXL was also trained on the same dataset, with a frozen CLIP (independent of CLIP).
- For fine-tuning the SDXL U-Net Diffusion Model to complement CLIP, please refer to [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)

![why-you-should](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/6f099007-127a-4fa4-88bc-7622df5383d7)
