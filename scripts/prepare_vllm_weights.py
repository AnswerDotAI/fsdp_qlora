"""
Preparing HQQ DoRA weights for VLLM inference:  https://github.com/AnswerDotAI/vllm/tree/torchao.

Supporting tinygemm (4bit) and bitblas (4,2,4/2 mixed bit) inference.
"""

from pathlib import Path
import os, json, shutil
from safetensors.torch import save_file
import copy
from tqdm import tqdm
import safetensors
import safetensors.torch
from glob import glob
from transformers import AutoConfig
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from fastcore.script import *
from fastcore.parallel import parallel
import functools
import torch

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig, Quantizer
from hqq.backends.torchao import patch_hqq_to_aoint4

import bitblas
from bitblas.cache import global_operator_cache, get_database_path
from bitblas.module import auto_detect_nvidia_target, BITBLAS_DATABASE_PATH

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger()

BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = "/workspace/.cache/bitblas"
BITBLAS_OPT_M = [1, 16, 32, 64, 128, 256, 512]
def _get_or_create_bitblas_operator(config):
    if global_operator_cache.size() == 0:
        global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)

    bitblas_matmul = global_operator_cache.get(config)
    if bitblas_matmul is None:
        # should disable tuning for the first time because we may require loading bitblas operator from database.
        bitblas_matmul = bitblas.Matmul(config) # default tuning is topk=20
        # bitblas_matmul.hardware_aware_finetune(topk=20)
        global_operator_cache.add(config, bitblas_matmul)
        global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
        logger.info("BitBLAS Tuning done, appended operator to global_operator_cache.")
    else:
        logger.info("BitBLAS Operator found in global_operator_cache.")
    return bitblas_matmul


def quantize_and_save(filename, quantized_layers, layer_nbits, layer_groupsizes, dtype, bitblas_dtype, args, 
                      dora_weights, layernorm_layers, config_dict):
    
    file_shard_name = Path(filename).name
    pretrained_weights = safetensors.torch.load_file(filename)
    existing_quantized_dir = Path(args['use_existing_from']) if args['use_existing_from'] is not None else None
    
    if existing_quantized_dir is not None:
        quantized_state_dict = safetensors.torch.load_file(existing_quantized_dir/file_shard_name)
        print(f"Replacing DoRA weights for {file_shard_name} from {existing_quantized_dir}.")
    else:
        quantized_state_dict = {}
        print(f"Preparing all weights for {file_shard_name} from scratch.")
        
    block_influence_layers = config_dict.get("block_influence_layers", [])    
    for n,p in tqdm(iter(pretrained_weights.items())):
        if "inv_freq" in n: continue
        p = p.to(dtype)
        if any(l in n for l in quantized_layers) and "weight" in n:
            if existing_quantized_dir is None:
                NBITS = layer_nbits[n.split(".")[-2]]
                GROUPSIZE = layer_groupsizes[n.split(".")[-2]]
                
                # Block Influence.
                if any(l + "." in n for l in block_influence_layers):
                    NBITS = 4
                    GROUPSIZE = config_dict['groupsize_4bit']
                    print(f"Block Influence: Setting {n} to 4-bit.")
                    
                print(f"Quantizing {n} with {NBITS}-bit and groupsize: {GROUPSIZE}.")
                
                # Get layer-wise quant config.
                quant_config = BaseQuantizeConfig(nbits=NBITS,
                                                    group_size=GROUPSIZE, 
                                                    quant_zero=False,
                                                    quant_scale=False,
                                                    offload_meta=False,
                                                    view_as_float=False, 
                                                    axis=1)
        
                # Prepare HQQ weights and quantize.
                m = torch.nn.Linear(*p.T.shape, bias=False)
                m.weight.data.copy_(p)
                hqq_linear = HQQLinear(m, quant_config, compute_dtype=dtype)
                W_est = hqq_linear.dequantize()

                # Tinygemm weights.
                if args["infer_type"] == "tinygemm":
                    patched_hqq_linear = patch_hqq_to_aoint4(hqq_linear, None) # patching deletes `hqq_linear.W_q`.
                    quantized_state_dict[n.replace(".weight", ".qweight")] = patched_hqq_linear.weight_int4pack
                    quantized_state_dict[n.replace(".weight", ".scales_and_zeros")] = patched_hqq_linear.scales_and_zeros         
                # Bitblas weights.
                elif args["infer_type"] == "bitblas":
                    W_q_unpacked = Quantizer.unpack[hqq_linear.meta['packing']](hqq_linear.W_q)
                    scale, zero, shape = hqq_linear.meta['scale'], hqq_linear.meta['zero'], hqq_linear.meta['shape']
                    scale = scale.to(bitblas_dtype)
                    zero = zero.to(bitblas_dtype)
        
                    # BitBLAS engine.
                    print(f"Tuning BitBLAS for {hqq_linear.in_features}x{hqq_linear.out_features}")
                    matmul_config = bitblas.MatmulConfig(M=BITBLAS_OPT_M,
                                                            N=hqq_linear.out_features,
                                                            K=hqq_linear.in_features,
                                                            A_dtype="float16",  
                                                            W_dtype={4:"uint4",2:"uint2"}[NBITS],
                                                            accum_dtype="float16",  
                                                            out_dtype="float16",  
                                                            layout="nt",  
                                                            with_bias=False, 
                                                            group_size=GROUPSIZE,
                                                            with_scaling=True,  
                                                            with_zeros=True,  
                                                            zeros_mode="original",  
                                                            #fast_decoding=True,
                                                        )
                    matmul_eng = _get_or_create_bitblas_operator(matmul_config)		
        
                    Wq_bitblas = matmul_eng.transform_weight(W_q_unpacked.reshape(shape))
                    meta_shape_bitblas = (hqq_linear.out_features, hqq_linear.in_features // GROUPSIZE)
                    scales_bitblas = scale.view(meta_shape_bitblas)
                    zeros_bitblas = zero.view(meta_shape_bitblas)

                    quantized_state_dict[n.replace(".weight", ".qweight")] = Wq_bitblas
                    quantized_state_dict[n.replace(".weight", ".scales")]  = scales_bitblas
                    quantized_state_dict[n.replace(".weight", ".zeros")]   = zeros_bitblas
                elif args["infer_type"] == "merged":
                    lora_a = dora_weights[n.replace(".weight",".dora_layer.lora_A.weight")].cuda()
                    lora_b = dora_weights[n.replace(".weight",".dora_layer.lora_B.weight")].cuda()
                    m = dora_weights[n.replace(".weight",".magnitude_layer.magnitude")].cuda()
                    rescale = m / (W_est + lora_b @ lora_a).norm(p=2, dim=1)
                    merged_weight = ((W_est + lora_b @ lora_a) * rescale.view(-1,1)).detach().cpu() 
                    quantized_state_dict[n] = merged_weight
                else:
                    raise ValueError("Invalid inference type.")
            else:
                assert "qweight" in quantized_state_dict
                if args["infer_type"] == "tinygemm":
                    assert "scales_and_zeros" in quantized_state_dict
                elif args["infer_type"] == "bitblas":
                    assert "scales" in quantized_state_dict and "zeros" in quantized_state_dict
                else:
                    raise ValueError("Invalid inference type.")

            # DoRA weights.
            # import pdb; pdb.set_trace()
            SKIP_DORA = (args["infer_type"] == "merged" or config_dict['skip_dora_all'] or (config_dict['skip_dora_4bit'] and NBITS == 4))
            if not SKIP_DORA:
                lora_a = dora_weights[n.replace(".weight",".dora_layer.lora_A.weight")].cuda()
                lora_b = dora_weights[n.replace(".weight",".dora_layer.lora_B.weight")].cuda()
                m = dora_weights[n.replace(".weight",".magnitude_layer.magnitude")]
                rescale = m / (W_est + lora_b @ lora_a).norm(p=2, dim=1).detach().cpu()
                lora_a, lora_b = lora_a.cpu(), lora_b.cpu()
                del W_est; torch.cuda.empty_cache()
                if args["infer_type"] == "bitblas":
                    lora_a = lora_a.to(bitblas_dtype)
                    lora_b = lora_b.to(bitblas_dtype)
                    rescale = rescale.to(bitblas_dtype)
                quantized_state_dict[n.replace(".weight", ".lora_A")] = lora_a
                quantized_state_dict[n.replace(".weight", ".lora_B")] = lora_b
                quantized_state_dict[n.replace(".weight", ".rescale")] = rescale
            
        elif n in layernorm_layers:
            quantized_state_dict[n] = dora_weights[n]
            print(f"Replacing {n}.")
            
        else:
            quantized_state_dict[n] = p
            print(f"Copying {n}.")

    # Save quantized state_dict.
    quantized_state_dict = {k:v.contiguous() for k,v in quantized_state_dict.items()}
    save_dir = Path(args["save_dir"])
    safetensors.torch.save_file(quantized_state_dict, save_dir/f"{file_shard_name}")
    
    # Delete the replaced weights.
    if existing_quantized_dir is not None:
        os.remove(existing_quantized_dir/file_shard_name)
        print(f"Deleted {existing_quantized_dir/file_shard_name}")
            


@call_parse()
def main(
    train_type: Param("", choices=["hqq_dora"]) = "hqq_dora", # Which quantization strategy to use for inference.
    infer_type: Param("", choices=["tinygemm", "bitblas", "merged"]) = "tinygemm", # Which kernel strategy to use for inference.
    dora_safetensors_filename: str = None, # Used for lora/dora inference.
    use_existing_from: str = None, # Used to get quantized base weights from a different directory and replace only dora, and finally rename it.
    config_filename: str = None, # Used to get quantization config, might have been saved after training.
    model_name: str = "meta-llama/Llama-2-7b-hf", 
    save_dir: str = "/workspace/models/llama-7b-orca-math-100k-full-quantized",
    dtype: str = None, # Only used when config file doesn't have this info.
    nbits: int = None, # Only used when config file doesn't have this info.
    groupsize: int = None, # Only used when config file doesn't have this info.
):
    args = dict(locals())
    
    DEFAULT_LORA_TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    MODEL_NAME = args["model_name"]
    logger.info(f"Preparing weights for model: {MODEL_NAME}")
    
    save_dir = Path(args["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    config_dict = json.load(open(args["config_filename"]))["qlora_config"] if args["config_filename"] else {}

    if "lora_target_modules" not in config_dict:
        logger.info(f"Using default lora layers for quantization: {DEFAULT_LORA_TARGET_LAYERS}")
    lora_layers = config_dict.get("lora_target_modules", DEFAULT_LORA_TARGET_LAYERS)
        
    # EXAMPLE: "layer_nbits": {"q_proj": 4,"k_proj": 4,"v_proj": 4,"o_proj": 4,"gate_proj": 4,"up_proj": 4,"down_proj": 4}
    if "layer_nbits" not in config_dict:
        if args["nbits"] is None:
            raise ValueError("nbits must be provided if not in config file.")
        else:
            logger.info(f"Setting all layers to nbits: {args['nbits']}")
    layer_nbits = config_dict.get("layer_nbits", {k:args['nbits'] for k in DEFAULT_LORA_TARGET_LAYERS})
    
    if args["infer_type"] == "tinygemm" and any(nbit != 4 for nbit in layer_nbits.values()):
        raise ValueError("Tinygemm inference only supports 4-bit quantization.")
    
    if args["infer_type"] == "bitblas" and any(nbit not in [2,4] for nbit in layer_nbits.values()):
        raise ValueError("Bitblas inference only supports 2-bit and 4-bit quantization.")
        
    for l in lora_layers:
        if l not in layer_nbits:
            raise ValueError(f"Missing nbits in config for layer: {l}")
    
    # EXAMPLE: "layer_groupsizes": {"q_proj": 64,"k_proj": 64,"v_proj": 64,"o_proj": 64,"gate_proj": 64,"up_proj": 64,"down_proj": 64}
    if "layer_groupsizes" not in config_dict:
        if args["groupsize"] is None:
            raise ValueError("groupsize must be provided if not in config file.")
        else:
            logger.info(f"Setting all layers to groupsize: {args['groupsize']}")
    layer_groupsizes = config_dict.get("layer_groupsizes", {k:args['groupsize'] for k in DEFAULT_LORA_TARGET_LAYERS})
    
    if args["infer_type"] == "tinygemm" and len(set(layer_groupsizes.values())) != 1:
        raise ValueError("Tinygemm inference requires same group size for each layer. This can change in future.")
    
    if not (layer_nbits["q_proj"] == layer_nbits["k_proj"] == layer_nbits["v_proj"]):
        raise ValueError("QKV layers must have same nbits.")
    
    if not (layer_groupsizes["q_proj"] == layer_groupsizes["k_proj"] == layer_groupsizes["v_proj"]):
        raise ValueError("QKV layers must have same group size.")
    
    if not (layer_nbits["gate_proj"] == layer_nbits["up_proj"]):
        raise ValueError("Gate and up layers must have same nbits.")
    
    if not (layer_groupsizes["gate_proj"] == layer_groupsizes["up_proj"]):
        raise ValueError("Gate and up layers must have same group size.")
    
    dtype = getattr(torch, config_dict.get("compute_dtype", "bfloat16"))
    lora_rank = config_dict.get("lora_rank", 64)
    # lora_alpha = config_dict.get("lora_alpha", 16)
    
    MODEL_NAME = args["model_name"]
    idx = hub.cached_file(MODEL_NAME, SAFE_WEIGHTS_INDEX_NAME)
    pretrained_files, _ = hub.get_checkpoint_shard_files(MODEL_NAME, idx)
    
    dora_weights = safetensors.torch.load_file(args["dora_safetensors_filename"])
    
    if config_dict['train_layernorms']:
        layernorm_layers = set([k for k in dora_weights.keys() if "layernorm" in k])
    else:
        layernorm_layers = set([])    
    
    # Here assume quantized layers are same as lora layers.
    quantized_layers = lora_layers
    
    # Need to cast to half for bitblas.
    bitblas_dtype = torch.half
   
    # TODO: Separate quantized weights (one time prep) and lora weights.
    # Save quantized weights for a given config only once.
    # VLLM requires a single directory to have all the files. How to share quantized weights across different directories with different lora weights?
    quantize_and_save_parallel = functools.partial(quantize_and_save, quantized_layers=quantized_layers, layer_nbits=layer_nbits, layer_groupsizes=layer_groupsizes, 
                                                    dtype=dtype, bitblas_dtype=bitblas_dtype, args=args, dora_weights=dora_weights, layernorm_layers=layernorm_layers, 
                                                    config_dict=config_dict)
    if args["infer_type"] in ["tinygemm"]:
        n_workers = 8
        parallel(quantize_and_save_parallel, pretrained_files, n_workers=n_workers, threadpool=True)
    elif args["infer_type"] in ["bitblas", "merged"]:    
        # With bitblas when generating cache files there can be race conditions.
        for pretrained_file in pretrained_files:
            quantize_and_save_parallel(pretrained_file)
    else:
        raise ValueError("Invalid inference type.")

    # save vLLM quant config.
    vllm_nbits 		 = {}
    vllm_group_sizes = {}
    if "gate_proj" in lora_layers:
        vllm_nbits["gate_up_proj"] = layer_nbits["gate_proj"]
        vllm_group_sizes["gate_up_proj"] = layer_groupsizes["gate_proj"]
    if "q_proj" in lora_layers:
        vllm_nbits["qkv_proj"] = layer_nbits["q_proj"]
        vllm_group_sizes["qkv_proj"] = layer_groupsizes["q_proj"]
    if "o_proj" in lora_layers:
        vllm_nbits["o_proj"] = layer_nbits["o_proj"]
        vllm_group_sizes["o_proj"] = layer_groupsizes["o_proj"]
    if "down_proj" in lora_layers:
        vllm_nbits["down_proj"] = layer_nbits["down_proj"]
        vllm_group_sizes["down_proj"] = layer_groupsizes["down_proj"]          
        
    if args["infer_type"] == "tinygemm":
        GROUPSIZE = layer_groupsizes["q_proj"]
        quant_config_dict = {"group_size" : GROUPSIZE, "inner_k_tiles" : 8, "lora_rank": lora_rank}
        
    elif args["infer_type"] == "bitblas":
        quant_config_dict = {"group_size" : vllm_group_sizes, "nbits" : vllm_nbits, "lora_rank" : lora_rank}
            
    # skipped_dora_layers
    block_influence_layers = config_dict.get("block_influence_layers", [])    
    if args["infer_type"] != "merged":
        quant_config_dict["skipped_dora_layers"] = []
        if config_dict['skip_dora_all']:
            quant_config_dict["skipped_dora_layers"] = list(vllm_nbits.keys())
        if config_dict['skip_dora_4bit']:
            quant_config_dict["skipped_dora_layers"] += [k for k,v in vllm_nbits.items() if v == 4]
        
        quant_config_dict["block_influence_layers"] = block_influence_layers
        quant_config_dict["groupsize_4bit"]         = config_dict['groupsize_4bit']
        
        quant_config_filename = save_dir/"quantize_config.json"
        with open(quant_config_filename, "w+") as f: json.dump(quant_config_dict, f)
    
    # save model config.
    model_config = AutoConfig.from_pretrained(MODEL_NAME).to_dict()
    # save model config
    # model_config['rope_scaling'] = {"type" :"dynamic", "factor": 2.0}
    model_config_filename = save_dir/"config.json"
    with open(model_config_filename, "w+") as f: json.dump(model_config, f)