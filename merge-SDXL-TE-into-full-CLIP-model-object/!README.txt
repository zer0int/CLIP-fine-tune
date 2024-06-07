Step 0: Point to the original CLIP .cache ViT-L-14.pt model file -> state_dict. Then, copy that to the folder where all these scripts AND the SDXL text encoder .safetensors is located.

Step 1: Insert the filename_of_TE.safetensors in the code.

Run all steps one after another. Finally: combined-ViT-L-14-full-model-object.pt

Delete all the other intermediate model files. 

You can use "combined-ViT-L-14-full-model-object.pt" in the same way you could use a normal fine-tuned CLIP model, e.g. with my scripts for gradient ascent, or using it in ComfyUI after converting to a state_dict again.

The model will be unaligned in multimodality, i.e. the text transformer won't be aligned to the vision transformer and their projection space, if the TE was previously fine-tuned in SDXL.

Freeze the Text encoder, and try fine-tuning with the visual.transformer requiring grad (see code for details).

This is highly experimental, and I wouldn't bet on the outcome being positive, i.e. that this works to "re-align" the vision transformer and projection space to whatever the TE has been fine-tuned to in SDXL with U-Net, e.g. using kohya.

However: Happy experimenting!