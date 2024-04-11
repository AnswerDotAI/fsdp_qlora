"""
Preparing weights for VLLM inference:  https://github.com/AnswerDotAI/vllm/tree/bnb_quant.
"""

from pathlib import Path
import os, json
from safetensors.torch import save_file
import copy
from tqdm import tqdm
import safetensors
import safetensors.torch
from glob import glob
from transformers import AutoConfig
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from fastcore.script import *

import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
import torch

@call_parse()
def main(
    infer_type: Param("", choices=["full_post_quant", "bnb_dora", "merged_bnb_dora"]) = "full_post_quant",
    model_weights_dir: str = None, 
    dora_filename: str = None, 
    model_name: str = "meta-llama/Llama-2-7b-hf", 
    save_dir: str = "/workspace/models/llama-7b-orca-math-100k-full-quantized",
):
    args = dict(locals())
    
    MODEL_NAME = args["model_name"]
    
    save_dir = Path(args["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # Quantized and lora layers.
    quantized_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_layers      = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    
    if args["infer_type"] == "full_post_quant":
        pretrained_files = glob(os.path.join(args["model_weights_dir"], "*.safetensors"))
    
    elif args["infer_type"] in ["bnb_dora", "merged_bnb_dora"]:
        idx = hub.cached_file(MODEL_NAME, SAFE_WEIGHTS_INDEX_NAME)
        pretrained_files, _ = hub.get_checkpoint_shard_files(MODEL_NAME, idx)
        weights = safetensors.torch.load_file(dora_filename)
    
    # Prepare new weights for inference.
    dtype = torch.bfloat16
    lora_rank = 64
    blocksize = 64
    pack_factor = 2

    new_state_dict = {}
    for filename in pretrained_files:
        pretrained_weights = safetensors.torch.load_file(filename)
        for n,p in tqdm(iter(pretrained_weights.items())):
            p = p.to(dtype)
            
            if any(l in n for l in quantized_layers) and "weight" in n:
                # output_size x input_size
                input_size, output_size = p.shape
                param = Params4bit(p, quant_type="nf4", blocksize=blocksize, compress_statistics=False, quant_storage=torch.uint8)
                param.cuda()
                
                if args["infer_type"] in ["full_post_quant", "bnb_dora"]:
                    # reshape for tensor parallelism
                    qweight, absmax = param.data.cpu(), param.quant_state.absmax.cpu()        
                    qweight = qweight.reshape(input_size, output_size // pack_factor)
                    absmax = absmax.reshape(input_size, output_size // blocksize)
                    new_state_dict[n] = qweight
                    new_state_dict[n.replace(".weight", ".absmax")] = absmax
                    
                    if args["infer_type"] == "bnb_dora":
                        if any(l in n for l in lora_layers):
                            lora_a = weights[n.replace(".weight",".dora_layer.lora_A.weight")]
                            lora_b = weights[n.replace(".weight",".dora_layer.lora_B.weight")]
                            m = weights[n.replace(".weight",".magnitude_layer.magnitude")]
                            w = bnb.functional.dequantize_4bit(param.data, param.quant_state).cpu()
                            rescale = m / (w + lora_b @ lora_a).norm(p=2, dim=1).detach()
                            new_state_dict[n.replace(".weight", ".lora_A")] = lora_a
                            new_state_dict[n.replace(".weight", ".lora_B")] = lora_b
                            new_state_dict[n.replace(".weight", ".rescale")] = rescale
                        else:
                            new_state_dict[n.replace(".weight", ".lora_A")] = torch.zeros((lora_rank, input_size), dtype=dtype)
                            new_state_dict[n.replace(".weight", ".lora_B")] = torch.zeros((output_size, lora_rank), dtype=dtype)
                            new_state_dict[n.replace(".weight", ".rescale")] = torch.ones(output_size, dtype=dtype)
                        
                elif args["infer_type"] == "merged_bnb_dora":
                    w = bnb.functional.dequantize_4bit(param.data, param.quant_state).cpu()
                    if any(l in n for l in lora_layers):
                        lora_a = weights[n.replace(".weight",".dora_layer.lora_A.weight")]
                        lora_b = weights[n.replace(".weight",".dora_layer.lora_B.weight")]
                        m = weights[n.replace(".weight",".magnitude_layer.magnitude")]
                        merged_w = w + lora_b @ lora_a
                        rescale = m / (merged_w).norm(p=2, dim=1).detach()
                        merged_w = rescale.view(-1,1) * merged_w
                        new_state_dict[n] = merged_w
                    else:
                        new_state_dict[n] = w.to(dtype)
                    
            else:
                new_state_dict[n] = p.to(dtype)
            
            param = None
            torch.cuda.empty_cache()
            
    # Save quantized weights.
    save_file(new_state_dict, save_dir/"model_state_dict.safetensors", metadata={'format': 'pt'})
            
    # Prepare quantization config.
    if args["infer_type"] in ["full_post_quant", "bnb_dora"]:
        quant_config_dict = {
            "weight_bits" : 4,
            "blocksize" : blocksize,
            "quant_type" : "nf4",
            "quant_storage" : "uint8",
            "compress_statistics" : False
        }
 
        if args["infer_type"] == "bnb_dora":
            quant_config_dict["lora_rank"] = lora_rank
            quant_config_dict["compute_dtype"] = "bfloat16"
                        
        quant_config_filename = save_dir/"quantize_config.json"
        with open(quant_config_filename, "w+") as f: json.dump(quant_config_dict, f)
        
    # Save model config.
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config_filename = save_dir/"config.json"
    with open(model_config_filename, "w+") as f: json.dump(model_config.to_dict(), f)