import torch
import copy
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Embeddings Merger')
    parser.add_argument('--checkpoint_path', default="clip_ft_20_backtoweight.pt", help="Name or path to CLIP Model")
    return parser.parse_args()

args = parse_arguments()
finetune_model_name = args.checkpoint_path

model = torch.load(finetune_model_name, map_location='cpu')

if isinstance(model, dict) and 'model' in model:
    model = model['model']

with torch.no_grad():
    assert hasattr(model, "positional_embedding") and hasattr(model, "positional_embedding_res")
    merged = model.positional_embedding.clone()
    merged[20:] = model.positional_embedding_res[20:]

    model.positional_embedding_joined = copy.deepcopy(torch.nn.Parameter(merged))
    del model.positional_embedding
    model.positional_embedding = copy.deepcopy(torch.nn.Parameter(model.positional_embedding_joined))
    del model.positional_embedding_joined
    

torch.save(model, "clip_ready_for_hf.pt")
print("Saved to clip_ready_for_hf.pt")
