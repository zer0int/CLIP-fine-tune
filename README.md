## ⭐ Summary: 
This repo is for fine-tuning CLIP in the command line. It does not add custom nodes to ComfyUI; however, you can easily use your fine-tune with ComfyUI:
- First, fine-tune with ft-B-train-OpenAI-CLIP-ViT-L-14.py
- Or, try the experimental and potentially superior exp-ft-B-GmP-finetune-OpenAI-ViT-L-14.py
- If you used "exp-ft-B-GmP", use this to convert the model: exp-ft-C-convert-GmP-back-to-weight.py
- Then, for both fine-tune scripts, use ft-C-convert-for-SDXL-comfyUI-OpenAI-CLIP.py
- Now you have a state_dict you can plug into ComfyUI for use with SD / SDXL!
### 👇 Scroll all the way down for step-by-step instructions with ComfyUI! 👇
### ‼️ Don't want to fine-tune? You can download the model here: [https://huggingface.co/zer0int](https://huggingface.co/zer0int)
-------
## Changes 11/NOV/2024:
- Added a new model saver: Saves either as GmP + full model object (default, legacy behavior)
- Optional conversion to .weight (converting back with extra script no longer needed)
- Option to save as full model, state_dict, or torch.jit.trace model (or all of these)
- Check the top of the code, set True / False as desired to enable / disable!
----
## Changes 23/OKT/2024:
Added folder `Convert-for-HuggingFace-Spaces-etc`

- Includes the [convert_clip_original_pytorch_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/convert_clip_original_pytorch_to_hf.py) script from HuggingFace + configuration .json files.
- Includes optional code to subsequently extract the Text Encoder only model (e.g. for Flux.1 guidance)
- Includes optional code to add metadata `{"format": "pt"}` - use it in case you get an error about 'pt'!
- Please check the included `how-to-use.txt` & code comments for details
----
## Changes 22/OKT/2024:
Added `a-loss-to-penalize-overfit-via-entropy.py`

- A custom loss with an `entropy penalty` term that penalizes over-confidence (overfit)
- For a diverse and general (!) dataset, the results of fine-tuning are good, but slightly worse than without entropy penalty (fine-tune on COCO-SPRIGHT):

1. ImageNet/ObjectNet accuracy without entropy penalty: 0.845 -> 0.914
2. ImageNet/ObjectNet accuracy with entropy penalty: 0.845 -> 0.908

- Alas, I don't want to integrate this loss into the 'best' working code; whether or not it is useful depends entirely on your dataset. If you have a very narrow dataset (e.g. just sneakers), however, and you find CLIP to begin overfitting (val loss increasing) before it has converged to a good (low) loss, then this entropy penalty could be very useful. Simply replace the loss in the actual training script if you observe overfitting, and tinker with the `lambda_entropy` factor. Actual example from `log_train.txt` of `1.`:
```
Epoch n:
Validation Acc: 0.9012, Validation F1: 0.8749
Training Loss: 0.6043, Validation Loss: 1.1853
Epoch n+1:
Validation Acc: 0.8942, Validation F1: 0.8652 <- decrease
Training Loss: 0.6018, Validation Loss: 1.1894 <- increase
```
Now, for the diverse dataset, this was *overtraining*, not *overfitting*; the model had already converged (good Acc/F1, low loss). In this case, early stopping (or saving checkpoints every epoch, then hand-selecting the best one - an earlier one, in this case) is recommended. However, I did not observe such an uptick with entropy penalty for a few epochs of overtraining (albeit the model converged at less ideal `Validation Acc: 0.8902, Validation F1: 0.8600`). So, give it a try if you see CLIP do this with your dataset (very extreme example; better to check `log_train.txt` to catch it early!):

![extreme-example-sm](https://github.com/user-attachments/assets/bd466dd8-f40d-4ac8-bdf1-cb5ac8fa9c80)

----
## Changes 11/AUG/2024:

- Added `ft-C-convert-with-org-dtype-fp16.py` -> Save with mixed precision as per OpenAI, model size ~900 MB
- Added `ft-C-convert-to-safetensors.py` -> Should be obvious, but check code comments for details. :-)
- ### ✨ Added `exp-acts-ft-SMOOTH-finetune-OpenAI-CLIP-ViT-L-14-GmP.py` 🥳
- This is the same as `exp-acts-ft-finetune-OpenAI-CLIP-ViT-L-14-GmP-manipulate-neurons.py`
- BUT it introduces label smoothing and a custom loss function for CLIP (CLIP doesn't have discrete 'classes').
- In general: Label smoothing is a regularization technique that softens the target labels by assigning a small probability to incorrect classes, rather than using hard one-hot labels. This approach prevents the model from becoming overly confident in its predictions, encouraging better generalization and reducing the risk of overfitting, especially in cases where the data might be noisy or limited in diversity.
- Read more in the paper [When Does Label Smoothing Help? / Google Brain](https://arxiv.org/abs/1906.02629).
----
- Is it for me? 🤔
- If you have a small dataset or suboptimal labels or are GPU-poor (24/48 GB VRAM), or all of that:
- YES! This *may* improve your results quite dramatically! 🤓💡
----
It even further improved the results for the high-quality COCO-40K-SPRIGHT dataset to >91% accuracy, trained on 1x RTX4090 🤯:

![gmp-models-extreme-plot-all-evals](https://github.com/user-attachments/assets/d3838637-46c4-469e-98fc-832cfc065f90)

- You can download this model on [my HuggingFace](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) if you don't want to reproduce the fine-tune with the provided code. :-)

- Technical / code summary of changes:

## Normal Contrastive Loss.

```
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate loss as the mean of the two cross-entropy losses
        loss_img = self.criterion(logits, labels)
        loss_txt = self.criterion(logits.t(), labels)

        return (loss_img + loss_txt) / 2
```

## New Custom Loss.
- Shout out to GPT-4o, my team-mate, for implementation help! 🤓🤝🤖

```
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, smoothing=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.smoothing = smoothing

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Apply label smoothing
        N = logits.size(0)
        smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
        smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate loss manually using log-softmax and smoothed labels
        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs).sum(dim=1).mean()

        return (loss_img + loss_txt) / 2
```

----
⬇️ Download my best-performing fine-tune (see Update 12/June/24) here: 
- ⬇️ [https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14)
- It's a state-dict; use with ComfyUI as-is, or load it as the state_dict of the original ViT-L/14 for inference, to fine-tune, etc.

![eval-clip-gpt4-compare](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/358ef8b0-4a92-4405-b271-a80c5890539d)

----
### Update 12/June/24:
- Added exp-acts-ft-finetune-OpenAI-CLIP-ViT-L-14-GmP-manipulate-neurons.py
- Allows to manipulate / scale activation values at c_fc individual neurons
- See [zer0int/Golden-Gate-CLIP](https://github.com/zer0int/Golden-Gate-CLIP) for details

Background: I identified an "adverb neuron" in the vision transformer of ViT-L/14. When the activation value is scaled by a factor of 1000, CLIP's "opinion" about any image will be mainly consisting of adverbs (see link above for code & details). I scaled the activaton value of predominantly this penultimate layer neuron by x1000 during fine-tuning on the usual general dataset (CoCo-40k-SPRIGHT), expecting either overfit / "adverb CLIP" or destruction of the model. Initially, training seemed to converge toward the latter, with Validation Accuracy and Validation F1 being in the 0.0X range while gradients truly exploded (reached inf) even after Epoch 0, and given a LR=1e-7. As the scheduler kicked in to increase the learning rate up to 5e-7, a dramatic drop in loss and val loss was observed, with an immediate jump to Validation Acc 0.8, Val F1 0.75, further improving with every additional Epoch. The final model has an unprecedented ImageNet / ObjectNet accuracy of ~0.90 (original pre-trained model / OpenAI's CLIP: ~0.85). Apparently, the model compensated for those erratic, over-activated neurons, and in turn found a better solution / minimum for generalizing text-image contrastive learning. It unexpectedly turned out to be my best-performing fine-tune thus far. Alas I am sharing the code to reproduce the results (or modify other neuron activations experimentally) as-is.

![results-act-github](https://github.com/zer0int/CLIP-fine-tune/assets/132047210/1a65c639-60c1-4d42-bd2f-98a38ec19ea5)

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
