import torch
import torch.nn.functional as F
from clip.model import CLIP

# ------------------------------------------
# Converts geometric parametrization:
# linear .theta, .r -> .weight (standard)
# Converts attention: QKV -> concat attn
# ------------------------------------------

which = "20" # Checkpoint number to convert

# Adjust these, if needed:
fine_tuned_model_path = f"ft-checkpoints/clip_ft_{which}.pt"
converted_output_path = f"ft-checkpoints/clip_ft_{which}_backtoweight.pt"

also_save_state_dict = False # Set True to also save as state_dict
state_dict_output_path = f"ft-checkpoints/clip_ft_{which}_state_dict.pt"



def convert_back_to_original(state_dict):
    """Convert Geometric Parameterization back to standard weights."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".theta"):
            base_key = key.replace(".theta", "")
            r_key = base_key + ".r"
            new_weight = state_dict[r_key] * F.normalize(value, p=2, dim=1)
            new_state_dict[base_key + ".weight"] = new_weight
        elif key.endswith(".r") or key.endswith(".theta"):
            continue  # Skip the .r and .theta keys
        else:
            new_state_dict[key] = value
    return new_state_dict

def convert_state_dict_qkv_to_inproj(state_dict):
    """
    Convert Attention QKV back to concatenated form.
    Converts explicit q_proj/k_proj/v_proj weights/biases into in_proj_weight and in_proj_bias.
    Handles missing bias robustly.
    """
    out = {}
    keys_to_remove = set()
    for key in list(state_dict.keys()):
        if key.endswith('.attn.q_proj.weight'):
            base = key[:-len('.q_proj.weight')]
            q = state_dict[base + '.q_proj.weight']
            k = state_dict[base + '.k_proj.weight']
            v = state_dict[base + '.v_proj.weight']
            out[base + '.in_proj_weight'] = torch.cat([q, k, v], dim=0)

            # Only add bias if all exist
            qkbias = [base + '.q_proj.bias', base + '.k_proj.bias', base + '.v_proj.bias']
            if all(bias_key in state_dict for bias_key in qkbias):
                qb = state_dict[base + '.q_proj.bias']
                kb = state_dict[base + '.k_proj.bias']
                vb = state_dict[base + '.v_proj.bias']
                out[base + '.in_proj_bias'] = torch.cat([qb, kb, vb], dim=0)
                keys_to_remove.update(qkbias)

            # Always remove weights; remove biases only if present
            keys_to_remove.update([
                base + '.q_proj.weight',
                base + '.k_proj.weight',
                base + '.v_proj.weight'
            ])
        else:
            if key not in out:
                out[key] = state_dict[key]

    # Remove all keys that are now merged/obsolete
    for k in keys_to_remove:
        if k in out: del out[k]
        if k in state_dict: del state_dict[k]
    return out


# 1. Load the fine-tuned model object (contains extra params, geometric param, etc.)
modelft = torch.load(fine_tuned_model_path)

# 2. Get the state_dict, apply conversion to "back to original" linear weights and concat attention
ft_state_dict = modelft.state_dict()
conversion_state_dict = convert_back_to_original(ft_state_dict)
converted_state_dict = convert_state_dict_qkv_to_inproj(conversion_state_dict)

# 3. Save the **converted state_dict**
if also_save_state_dict:
    torch.save(converted_state_dict, state_dict_output_path)
    print("Converted state_dict saved.")

# 4. Instantiate a new model to save as full model object (pickle / danger-pickle)
# Extract dimensions needed for constructor (from the state_dict)
embed_dim = converted_state_dict["text_projection"].shape[1]
vision_width = converted_state_dict["visual.conv1.weight"].shape[0]
vision_patch_size = converted_state_dict["visual.conv1.weight"].shape[-1]
context_length = converted_state_dict["positional_embedding"].shape[0]
vocab_size = converted_state_dict["token_embedding.weight"].shape[0]
transformer_width = converted_state_dict["ln_final.weight"].shape[0]
transformer_heads = transformer_width // 64
vision_layers = len([k for k in converted_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
transformer_layers = len(set(k.split(".")[2] for k in converted_state_dict if k.startswith("transformer.resblocks")))

grid_size = round((converted_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # Ignore CLS
input_resolution = grid_size * vision_patch_size

# Instantiate the model
model = CLIP(
    embed_dim=embed_dim,
    image_resolution=input_resolution,
    vision_layers=vision_layers,
    vision_width=vision_width,
    vision_patch_size=vision_patch_size,
    context_length=context_length,
    vocab_size=vocab_size,
    transformer_width=transformer_width,
    transformer_heads=transformer_heads,
    transformer_layers=transformer_layers,
)

# 5. Load the **converted** state_dict into this new model
model.load_state_dict(converted_state_dict, strict=True)
print("Model loaded with converted weights.")

# 6. Save the full model object (with weights restored)
torch.save(model, converted_output_path)
print("Full model object (with converted weights) saved.")
