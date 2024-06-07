import torch
from safetensors.torch import load_file

# Load the fine-tuned state dictionary
fine_tuned_text_state_dict = load_file("clip_l_00001_.safetensors")

# Initialize a new state dictionary
converted_state_dict = {}

# Mapping the keys from the fine-tuned state dict to the original OpenAI CLIP model structure
for key, value in fine_tuned_text_state_dict.items():
    if key == "logit_scale":
        converted_state_dict["logit_scale"] = value
    elif key.startswith("text_model.embeddings.position_embedding"):
        converted_state_dict["positional_embedding"] = value
    elif key.startswith("text_model.embeddings.token_embedding"):
        converted_state_dict["token_embedding.weight"] = value
    elif key.startswith("text_model.encoder.layers"):
        layer_num = key.split(".")[3]
        sub_key = key.split(".", 4)[-1]
        if sub_key.startswith("layer_norm1"):
            new_key = f"transformer.resblocks.{layer_num}.ln_1.{sub_key.split('.')[-1]}"
        elif sub_key.startswith("layer_norm2"):
            new_key = f"transformer.resblocks.{layer_num}.ln_2.{sub_key.split('.')[-1]}"
        elif sub_key.startswith("mlp.fc1"):
            new_key = f"transformer.resblocks.{layer_num}.mlp.c_fc.{sub_key.split('.')[-1]}"
        elif sub_key.startswith("mlp.fc2"):
            new_key = f"transformer.resblocks.{layer_num}.mlp.c_proj.{sub_key.split('.')[-1]}"
        elif sub_key.startswith("self_attn.k_proj"):
            if f"transformer.resblocks.{layer_num}.attn.in_proj_weight" not in converted_state_dict:
                converted_state_dict[f"transformer.resblocks.{layer_num}.attn.in_proj_weight"] = torch.cat([
                    fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.weight"],
                    fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.weight"],
                    fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.weight"]
                ], dim=0)
            if f"transformer.resblocks.{layer_num}.attn.in_proj_bias" not in converted_state_dict:
                converted_state_dict[f"transformer.resblocks.{layer_num}.attn.in_proj_bias"] = torch.cat([
                    fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.bias"],
                    fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.bias"],
                    fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.bias"]
                ], dim=0)
        elif sub_key.startswith("self_attn.out_proj"):
            new_key = f"transformer.resblocks.{layer_num}.attn.out_proj.{sub_key.split('.')[-1]}"
        converted_state_dict[new_key] = value
    elif key.startswith("text_model.final_layer_norm"):
        converted_state_dict["ln_final." + key.split('.')[-1]] = value
    elif key.startswith("text_projection"):
        converted_state_dict["text_projection"] = value

# Correct the shape of c_proj weights and biases
for layer_num in range(12):
    converted_state_dict[f"transformer.resblocks.{layer_num}.mlp.c_proj.weight"] = fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.mlp.fc2.weight"].view(768, 3072)
    converted_state_dict[f"transformer.resblocks.{layer_num}.mlp.c_proj.bias"] = fine_tuned_text_state_dict[f"text_model.encoder.layers.{layer_num}.mlp.fc2.bias"]

# Save the converted state dictionary to a .pt file
torch.save(converted_state_dict, "step1-state-dict-clip-like.pt")
