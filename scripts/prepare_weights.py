"""
Preparing weights for VLLM inference:  https://github.com/AnswerDotAI/vllm/tree/bnb_quant.

NOTE: Highly experimental as vLLM integration might change. For now it is safer to use merged weight methods.
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
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig, Quantizer
from fastcore.script import *

import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
import torch

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger()


@call_parse()
def main(
    infer_type: Param("", choices=["full_post_quant", "bnb_dora", "merged_bnb_lora", "merged_hqq_lora", "merged_bnb_dora", "merged_hqq_dora"]) = "full_post_quant", # Which merge strategy to use for inference.
    model_weights_dir: str = None, # Used for full post quantization.
    lora_or_dora_filename: str = None, # Used for lora/dora inference.
    config_filename: str = None, # Used to get config, might have been saved after training.
    model_name: str = "meta-llama/Llama-2-7b-hf", 
    save_dir: str = "/workspace/models/llama-7b-orca-math-100k-full-quantized",
    dtype: str = "bfloat16"
):
    args = dict(locals())
    
    MODEL_NAME = args["model_name"]
    logger.info(f"Preparing weights for model: {MODEL_NAME}")
    
    save_dir = Path(args["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    config_dict = json.load(open(args["config_filename"])) if args["config_filename"] else {}

    # NOTE: This  probably won't change, but would be better to check for new added models. Quantized and lora layers. 
    # FSDP QLoRA/QDoRA replace all linear layers with quantized linear layers.
    if "phi-3" in model_name.lower():
        quantized_layers = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    else:
        quantized_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # TODO: Get this info from saved model config if exists.
    lora_layers = config_dict.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    if "lora_target_modules" not in config_dict:
        logger.info(f"Using default lora layers: {lora_layers}")
    
    if args["infer_type"] == "full_post_quant":
        pretrained_files = glob(os.path.join(args["model_weights_dir"], "*.safetensors"))
    
    elif args["infer_type"] in ["bnb_dora",  "merged_bnb_lora", "merged_hqq_lora", "merged_bnb_dora", "merged_hqq_dora"]:
        idx = hub.cached_file(MODEL_NAME, SAFE_WEIGHTS_INDEX_NAME)
        pretrained_files, _ = hub.get_checkpoint_shard_files(MODEL_NAME, idx)
        weights = safetensors.torch.load_file(lora_or_dora_filename)
    
    # Prepare new weights for inference.
    # TODO: Read these to saved config file.
    dtype = getattr(torch, config_dict.get("compute_dtype", args["dtype"]))
    lora_rank = config_dict.get("lora_rank", 64)
    lora_alpha = config_dict.get("lora_alpha", 16)
    lora_scale = lora_alpha / lora_rank
    blocksize = config_dict.get("blocksize", 64)
    pack_factor = 2
    nbits = config_dict.get("n_bits", 4)
    quant_type = "hqq" if "hqq" in args["infer_type"] else "bnb"
    
    new_state_dict = {}
    for filename in pretrained_files:
        pretrained_weights = safetensors.torch.load_file(filename)
        for n,p in tqdm(iter(pretrained_weights.items())):
            if "inv_freq" in n: continue
            p = p.to(dtype)
            if any(l in n for l in quantized_layers) and "weight" in n:
                # output_size x input_size
                input_size, output_size = p.shape
                if quant_type == "bnb":
                    param = Params4bit(p, quant_type="nf4", blocksize=blocksize,
                                    compress_statistics=False, quant_storage=torch.uint8)
                    param.cuda()
                    
                elif quant_type == "hqq":
                    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=blocksize, quant_zero=True,
                                    quant_scale=True, offload_meta=True, view_as_float=True)
                    m = torch.nn.Linear(output_size, input_size)
                    m.weight.data.copy_(p)
                    hqq_linear = HQQLinear(linear_layer=m, quant_config=quant_config, compute_dtype=dtype, device="cuda")
                
                if args["infer_type"] in ["full_post_quant", "bnb_dora", "hqq_dora"]:
                    # reshape for tensor parallelism
                    qweight, absmax = param.data.cpu(), param.quant_state.absmax.cpu()        
                    qweight = qweight.reshape(input_size, output_size // pack_factor)
                    absmax = absmax.reshape(input_size, output_size // blocksize)
                    new_state_dict[n] = qweight
                    new_state_dict[n.replace(".weight", ".absmax")] = absmax
                    
                    if args["infer_type"] in ["hqq_dora", "bnb_dora"]:
                        if any(l in n for l in lora_layers):
                            lora_a = weights[n.replace(".weight",".dora_layer.lora_A.weight")]
                            lora_b = weights[n.replace(".weight",".dora_layer.lora_B.weight")]
                            m = weights[n.replace(".weight",".magnitude_layer.magnitude")]
                            if quant_type == "bnb":
                                w = bnb.functional.dequantize_4bit(param.data, param.quant_state).cpu()
                            elif quant_type == "hqq":
                                w = hqq_linear.dequantize_aten().cpu()
                            rescale = m / (w + lora_b @ lora_a).norm(p=2, dim=1).detach()
                            new_state_dict[n.replace(".weight", ".lora_A")] = lora_a
                            new_state_dict[n.replace(".weight", ".lora_B")] = lora_b
                            new_state_dict[n.replace(".weight", ".rescale")] = rescale
                        else:
                            new_state_dict[n.replace(".weight", ".lora_A")] = torch.zeros((lora_rank, input_size), dtype=dtype)
                            new_state_dict[n.replace(".weight", ".lora_B")] = torch.zeros((output_size, lora_rank), dtype=dtype)
                            new_state_dict[n.replace(".weight", ".rescale")] = torch.ones(output_size, dtype=dtype)
                        
                elif args["infer_type"] in ["merged_bnb_lora", "merged_hqq_lora", "merged_hqq_dora", "merged_bnb_dora"]:
                    if quant_type == "bnb":
                        w = bnb.functional.dequantize_4bit(param.data, param.quant_state).cpu()
                    elif quant_type == "hqq":
                            w = hqq_linear.dequantize_aten().cpu()
                            
                    if any(l in n for l in lora_layers):
                        if args["infer_type"] in ["merged_bnb_lora", "merged_hqq_lora"]:
                            try: #HF lora
                                n = "base_model.model." + n
                                lora_a = weights[n.replace(".weight",".lora_A.default.weight")]
                                lora_b = weights[n.replace(".weight",".lora_B.default.weight")]
                                n = n.removeprefix("base_model.model.")
                            except: #custom lora
                                lora_a = weights[n.replace(".weight",".lora_AB.0.weight")]
                                lora_b = weights[n.replace(".weight",".lora_AB.1.weight")]
                            merged_w = w + (lora_b @ lora_a) * lora_scale
                            new_state_dict[n] = merged_w
                        elif args["infer_type"] in ["merged_hqq_dora", "merged_bnb_dora"]:
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
                new_state_dict[n] = p
            
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
    if config_dict == {}:
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config_filename = save_dir/"config.json"
        with open(model_config_filename, "w+") as f: json.dump(model_config.to_dict(), f)