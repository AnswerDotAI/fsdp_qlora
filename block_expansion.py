
import argparse
from transformers import AutoModelForCausalLM
import torch

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", default='meta-llama/Llama-2-7b-hf', type=str, help="original model path")
    parser.add_argument("--output_path", default='pytorch_model.bin', type=str, help="deepened model ckpt save path")
    parser.add_argument("--original_layers", default=32, type=int, help="original model num layers")
    parser.add_argument("--layers", default=40, type=int, help="deepen model num layers")

    # Parse the arguments
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    ckpt = model.state_dict()
    
    split = int(args.original_layers / (args.layers - args.original_layers))
    layer_cnt = 0

    output = {}
    for i in range(args.original_layers):
        for k in ckpt:
            if ('layers.' + str(i) + '.') in k:
                output[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = ckpt[k]
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('layers.' + str(i) + '.') in k:
                    if 'down_proj' in k or 'o_proj' in k:
                        output[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = torch.zeros_like(ckpt[k])
                    else:
                        output[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = ckpt[k]


            layer_cnt += 1
        
    assert layer_cnt==args.layers
    for k in ckpt:
        if not 'layers' in k:
            output[k] = ckpt[k]

    torch.save(output, args.output_path)

if __name__ == "__main__":
    main()