
import argparse
from transformers import AutoConfig
import torch
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
import safetensors
from safetensors.torch import save_file
import os 
from pathlib import Path

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_name", default='meta-llama/Llama-2-7b-hf', type=str, help="original model path")
    parser.add_argument("--output_dir", default=None, type=str, help="deepened model ckpt save path")
    parser.add_argument("--expansion_rate", default=0.1, type=float, help="add new trainable % of layers")

    # Parse the arguments
    args = parser.parse_args()
        
    idx = hub.cached_file(args.model_name, SAFE_WEIGHTS_INDEX_NAME)
    files, _ = hub.get_checkpoint_shard_files(args.model_name, idx)
    
    cfg = AutoConfig.from_pretrained(args.model_name)
    num_original_layers = cfg.num_hidden_layers
    new_layers = num_original_layers + int(num_original_layers * args.expansion_rate)
    
    split = int(num_original_layers / (new_layers - num_original_layers))
    
    if args.output_dir is None:
        output_dir = Path(os.environ['HOME'])/'models'/(args.model_name + f'_blk_exp-{num_original_layers}-{new_layers}')
    else:
        output_dir = Path(args.output_dir)/(args.model_name + f'_blk_exp-{num_original_layers}-{new_layers}')
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in files:
        weights = safetensors.torch.load_file(filename)
        expanded_weights = {}
        for k,v in iter(weights.items()):
            if 'layers' in k:
                layer_no = int(k.split('layers.')[1].split('.')[0])
                # shift existing layers
                new_layer_no = layer_no + layer_no // split
                new_k = k.replace(f'layers.{layer_no}', f'layers.{new_layer_no}')
                expanded_weights[new_k] = v
                # add new layers
                if (layer_no+1) % split == 0:
                    new_layer_no += 1
                    new_k = k.replace(f'layers.{layer_no}', f'layers.{new_layer_no}')
                    if 'down_proj' in k or 'o_proj' in k:
                        expanded_weights[new_k] = torch.zeros_like(v)     
                    else:
                        expanded_weights[new_k] = v.clone()
            else:
                expanded_weights[k] = v
        save_file(expanded_weights, output_dir/Path(filename).name)
    

if __name__ == "__main__":
    main()