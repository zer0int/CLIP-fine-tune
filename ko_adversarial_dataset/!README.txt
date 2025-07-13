Get the dataset here: https://huggingface.co/datasets/zer0int/CLIP-KO-Adversarial-Train-Typo-Attack

And place it like so:

adversarial_dataset/
 |
 |--- adversarial_labels_max_cos_sim.json
 |--- adversarial_labels_min_cos_sim.json 
 |--- typoattack/
	|
	| --- 000_attack.png
	| --- 001_attack.png
	| --- ...





IMPORTANT NOTE.

The provided text labels (.json) may be biased and / or offensive. They contain three labels:

1. A CLIP gradient ascent 'opinion': Optimize text embeddings for cosine similarity with image embeddings -> tokens (words) describing the image, according to pre-trained CLIP.
2. A stereotypical, biased and / or misleading human label, including slurs and profanity, as informed by empirical evidence about the CLIP model.

3. A normal human label, representing an accurate 'ground truth' about the image.

The images all depict (mostly naturally occuring) text that is highly salient and misleading to CLIP (typographic attack vulnerability), e.g. street signs.



Labels 1. and 2. are used for *ADVERSARIAL* training; they are used to *minimize* cosine similarity, to guide CLIP *away* from such bias, while 3. is used for *maximizing* cosine similarity (positive pair).

The labels in 2. represent the 'worst of bias' as is present in the pre-trained CLIP model due to dataset bias / being trained 'on the (unfiltered) internet' PLUS the typographic attack vulnerability inherent to CLIP. Reproducing such exact bias and naming it explicitly for use as a NEGATIVE example for adversarial training is a necessity.

It does NOT in any way represent the author's (my) own opinion or beliefs. My intent with providing them is solely to *reduce* skewed / biased representations in CLIP - WITHOUT censoring anything or endorsing any specific world-view, but merely archiving a more balanced distribution and preventing misclassification. Alas, all the images are 100% SFW and depict images with text in them; e.g. images of street signs, stop signs from various countries, and so on.

As a concrete example from the dataset, a plain street sign that says 'DATE STREET' is still just a street sign - and NOT a reference to sexuality and 'dating' (which is what pre-trained CLIP associates with this particular image due to being extremely biased towards 'seeing' text as the ground truth about the image, irrespective of what else is in the image).
My (idealized) goal for CLIP is to classify such image as 'street sign', while classifying as 'sexuality-related' ONLY IF that is the *ground truth* about a given image.

Thank you for reading this!