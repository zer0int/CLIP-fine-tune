COCO 40k dataset for testing / reproducing the CLIP GmP implementation: https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco

The 'data' folder with subfolders "0", "1", "2", etc. should be where the json labels are.
Look inside the .json for where the images are expected.

I also just cropped the images to be square (for CLIP's expected input), alas subfolder "data-square" mentioned in default code. 
CLIP doesn't appreciate a squished-to-dimensions image, so: Don't forget it! ;-)

The labels in the .json files are capped to fit CLIP's 77 input tokens. You'll just have to download the images above.