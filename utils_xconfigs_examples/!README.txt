----> 	General rule of thumb: The larger the BATCH SIZE, the
	higher the LEARNING RATE (LR) - and vice versa!
----> 	Narrow / small dataset: Overfit risk -> reduce LR.

--------------------------------------------------

Have a tiny and very narrow dataset?
For example, just 5000 text-image pairs with sneakers?

1. First, train the full model without hacks:

- train_openai_vit-l-14_on_coco.json
or, example for loading LongCLIP from HF:
- train_hf_longclip_on_coco.json

- Or insert your own (general) dataset.

--------------------------------------------------

2. Use path/to/your/model.pt subsequently.

Alternatively, you can skip step 1 and just replace "ViT-L/14"
with my GmP fine-tune from HF, it's essentially the same (COCO):

zer0int/CLIP-GmP-ViT-L-14

Then, in all-in-one-clip-fine-tune.py:

A) If you predominantly want a text encoder, you could try:

in class TrainConfig:
    ...
    grad_vit_from: Optional[int] = 16 # train blocks 16-23 only
    grad_text_from: Optional[int] = 0 # train from 0 = full TE

B) You only need image embeddings, but the text encoder tries
to predominantly (and degenerately) 'solve' the problem 
(very likely unless huge batch size):

Train only the ViT:

in class TrainConfig:
    ...
    grad_full_text_vit: bool = False
    grad_vit_full: bool = True		# full ViT
    grad_text_full: bool = False

C) After B) but you need the Text-Image projection space
aligned after all:

in class TrainConfig:
    ...
    grad_vit_from: Optional[int] = None
    grad_text_from: Optional[int] = 10 	# or 9 

----> 	You'll have to tinker with this to get it right,
	the above are just some examples! ;)

--------------------------------------------------

Got a large, more general dataset?
Just train the model in full:

in class TrainConfig:
    ...
    grad_full_text_vit: bool = True
    grad_vit_full: bool = False
    grad_text_full: bool = False

-> You trained the full model, but the text encoder did a 
spiderrollercoaster and came up with a degenerate solution?

You can try rescuing it with my 'encoder transplantation' script:
- calibrate-rescue-fail-encoder-and-merge.py

... Then, re-train transplanted text encoder with low LR (e.g. 1e-7), 
and / or train only the the last blocks of the TE (see above).

Consider using 'grad_set_manual=True' next time when training
the full model, and give the ViT params e.g. 5e-6, Text: 5e-7.


--------------------------------------------------