## ⭐ Summary: 
This repo is for fine-tuning CLIP in the command line. It does not add custom nodes to ComfyUI; however, you can easily use your fine-tune with ComfyUI:
- First, fine-tune with ft-B-train-OpenAI-CLIP-ViT-L-14.py
- Or, try the experimental and potentially superior exp-ft-B-GmP-finetune-OpenAI-ViT-L-14.py
- If you used "exp-ft-B-GmP", use this to convert the model: exp-ft-C-convert-GmP-back-to-weight.py
- Then, for both fine-tune scripts, use ft-C-convert-for-SDXL-comfyUI-OpenAI-CLIP.py
- Now you have a state_dict you can plug into ComfyUI for use with SD / SDXL!
### 👇 Scroll all the way down for step-by-step instructions with ComfyUI! 👇
----
### Update 07/June/24:
Preliminary results of GmP-CLIP for SDXL-TE repair fine-tune: 
1. Seemingly "bad" results; model not able to predict correct words / opinion for an image (see previous update below)
2. However, it seems to "re-align coherence" -> very much improved results when used as SDXL TE encoder!
3. Separate CLIP fine-tune still superior, but:
4. This is potentially useful for ⚠️ fixing a ruined TE finetuned with U-Net (e.g. kohya) ⚠️ in <1h / 5 Epochs.

Results:
![Untitled-2](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/d851358b-8596-4d8e-8fb4-6f4aeaa3acc1)
The above model, used as SDXL TE again (center samples):
![Untitled-1](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/c37ed7ce-276f-49b9-86f7-a9297f77c0eb)

- 1st row: Ruined TE, once fine-tuned with U-Net (kohya) -> I normally don’t use this TE, but use a separately fine-tuned TE for this U-Net (3rd row).
- Take TE -> reassemble to full CLIP model with original Vision Transformer (see folder for scripts!) -> GmP (!) fine-tune on CoCo-SPRIGHT-40k short-labels for 5 Epochs (see COCO folder for json!) -> "Really bad fine-tune", judging by the numbers -> convert back to weight -> convert to state_dict -> Try it with SDXL *ANYWAY* -> results in 2nd row.
- ✅ Worth a try for a quick fix -> Ruined TE in SDXL by previous fine-tune.
- Note: All generated with same seed / settings / U-Net, of course. All of these use PAG, see [here](https://stable-diffusion-art.com/perturbed-attention-guidance/).
----
### Changes 07/June/24:
Added scripts to puzzle together a full CLIP text-vision transformer from the SDXL text encoder .safetensors file as per [this issue](https://github.com/zer0int/CLIP-fine-tune/issues/3).
See the readme in "merge-SDXL-TE-into-full-CLIP-model-object" for details. You can use this (full model object .pt) with all of my scripts as usual, but beware that if you fine-tuned the TE in SDXL (e.g. kohya), it will be unaligned / misaligned with the vision transformer and alas, latent space.

In other words, the model will be completely bonkers (see below), but you can try fine-tuning it "back into alignment" (freeze TE, fine-tune with careful LR). Good luck!

![model-crazy](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/f5a66c6e-5cee-4033-ad98-3f63185dbfbc)
----
### Changes 28/May/24:

- Added ft-D-eval*.py / validate accuracy of fine-tune against original CLIP model
- Check code for details; use ImageNet/ObjectNet to replicate GmP-CLIP results
----
### Changes 26/May/24:
- Added exp-ft-**.py

![eval-imagenet-objectnet](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/c733e331-5b0a-4166-b009-76e1fa5311a2)


### ⚠️ Extremely experimental Geometric Parameterization (GmP) inspired by [this paper](https://arxiv.org/abs/2305.15912v4).

- Introduces sophisticated per-parameter learning needed for GmP fine-tune.
- Introduces gradient accumulation. Check "finetune" code for details.
- Otherwise, this mainly changes the model architecture (!) via custom CLIP model code.
---
- Change "import gmpclip as clip" to "import clip" in "exp-ft-B-GmP-finetune-OpenAI-ViT-L-14.py" to use for "normal" fine-tune.
- 👇 If you do, you can skip this section & scroll down for the original fine-tuning instructions!
---
- Otherwise, you can also mostly fine-tune as usual; check code comments for pecularities.
- ⚠️ It is normal / inevitable to get large or even 'inf' gradients in Epoch 0. But 'inf' should NOT happen in later epochs!
- Optional: Use "exp-ft-X-visualize-theta-r-barplots.py" to visualize distribution of 'theta' and 'r' components (only works with GmP fine-tuned model).
- Use "exp-ft-C-convert-GmP-back-to-weight.py" to convert fine-tune to normal model object. Otherwise, the model won't be compatible with any third party code at all!
- Once converted back to ".weight", you can use the full model object "as normal" and e.g. convert to state_dict with "ft-C-convert-for-SDXL-comfyUI-OpenAI-CLIP.py".

## What's Geometric Parameterization / GmP, theta, r? 🤔

- GmP replaces linear layer ".weight" with GeometricLinear() for c_fc and c_proj in the MLP (multi-layer perceptron):

```
"Normal" CLIP MLP (multi-layer perceptron):

(mlp): Sequential(
  |-(c_fc): Linear(in_features=1024, out_features=4096, bias=True)
  | (gelu): QuickGELU()
|-}-(c_proj): Linear(in_features=4096, out_features=1024, bias=True)
| | 
| |-- visual.transformer.resblocks.0.mlp.c_fc.weight
| |-- visual.transformer.resblocks.0.mlp.c_fc.bias
|
|---- visual.transformer.resblocks.0.mlp.c_proj.weight
|---- visual.transformer.resblocks.0.mlp.c_proj.bias


GmP CLIP MLP:

Weight decomposition into:
- radial component 'r' as norm of pre-trained weights
- angular component 'theta' as normalized direction
-> preserves weight vectors' directionality and magnitude

(mlp): Sequential(
  |-(c_fc): GeometricLinear()
  | (gelu): QuickGELU()
|-}-(c_proj): GeometricLinear()
| | 
| |-- visual.transformer.resblocks.0.mlp.c_fc.r
| |-- visual.transformer.resblocks.0.mlp.c_fc.theta
| |-- visual.transformer.resblocks.0.mlp.c_fc.bias
|
|---- visual.transformer.resblocks.0.mlp.c_proj.r
|---- visual.transformer.resblocks.0.mlp.c_proj.theta
|---- visual.transformer.resblocks.0.mlp.c_proj.bias

(Same thing for [text] transformer.resblocks)
```

## Huh?!

- In [the paper](https://arxiv.org/abs/2305.15912v4), the authors describe, quote:
- *"In addition to achieving the best performance, GmP+IMN also converges significantly faster than other compared methods: its top-5 validation accuracy converges within 25 epochs, which is 10 epochs earlier than the second best method BN."*
- *"All [related] techniques operate in the Cartesian coordinate and thus suffers from the instability issue, whereas GmP operates in the hyperspherical coordinate to overcome this [mentioned in paper] instability issue."*
---
- The authors mention machine vision models; however, only with regard to ReLU activation functions. 
- However, CLIP's GELU is, in essence, a "kind of a smoother version of ReLU that allows negative values".
- Why not just try it with the multimodal contrastive learner, CLIP? 💡🤓
- I just let GPT-4o summarize the paper, then let the AI help me implement it for CLIP (thanks, GPT-4o!).
- Alas: No solid hypothesis, reasoned in LaTeX, behind my AI & I's doings. But tinkering with CLIP and the results are what grounds this in reality:

![clip-gmp-losses](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/349f994d-1416-4288-b6c4-a6cdeb848772)

---
- Most remarkably, it seems to have curbed CLIP's "text obsession" / [typographic attack vulnerability](https://openai.com/index/multimodal-neurons/), at least to some part:


![GmP-CLIP-fine-tune-fixes-typographic-attack-vulnerability](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/c51ba459-2749-4f82-a9f5-22393c41f058)
---
![poodleadv-MOSAIC-forgit](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/b03fb6bc-6950-404a-aa9e-b6786be12929)
---
![More-Examples](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/df4ed8be-7894-4ceb-a4d7-d49b8e239fc5)
---
- ⚠️ Disclaimer: I cannot "benchmark" this around and perform ablation studies and train on gigantic datasets and whatnot to really "scientifically prove" these results.
- I don't know why this seems to lead to such surprisingly positive results, or if it always applies to any CLIP and any dataset.
- I have 1 GPU, and I am doing this in my free time, alas provided "as-is", including the speculative "freak accident" remediation of CLIP's typographic attack vulnerability. Enjoy! 🙃

![clip-wins](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/9337e23c-96d5-4c22-b2c0-aaa156f8abae)

------
### Changes 19/May/24:
- Added CLIP-PCA-compare-org-to-finetuned-model.py - compare PCA plots of original model vs. fine-tuned model:

![pca-plot-example](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/0936eed9-c62e-46ea-b72e-14936bb89558)

Example for catastrophic overfitting: embeddings collapse and "everything is similar to everything" (cosine similarity). Decrease learning rate, increase batch size, make a better dataset with multiple text labels to choose from, when you see something like this:

![embeddings-collapse](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/7d31fe23-a1cc-471f-92e2-0f7d03281cc0)

------
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
