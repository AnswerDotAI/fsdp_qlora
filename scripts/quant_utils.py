import torch
from torch import nn, Tensor
from typing import List
from hqq.core.quantize import HQQLinear


# Utilities related to model loading
def replace_linear(model:nn.Module, 
                   linear_replacement, 
                   quant_config_4bit:dict|None=None, 
                   quant_config_2bit:dict|None=None,
                   layers_4bit:List[str]=[], 
                   layers_2bit:List[str]=[],
                   skip_modules:List[str]=["lm_head"], 
                   block_influence_patterns:List[int]=[],
                   prefix:str="model",
                   **kwargs):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """

    for name, module in model.named_children():
        if name in skip_modules:
            continue

        if len(list(module.children())) > 0:
            print(prefix + "." + name)
            print(block_influence_patterns)
            replace_linear(module, 
                           linear_replacement=linear_replacement, 
                           quant_config_4bit=quant_config_4bit, 
                           quant_config_2bit=quant_config_2bit, 
                           layers_4bit=layers_4bit, 
                           layers_2bit=layers_2bit, 
                           skip_modules=skip_modules, 
                           block_influence_patterns=block_influence_patterns,
                           prefix=prefix + "." + name,
                           **kwargs)

        if isinstance(module, torch.nn.Linear):
            if issubclass(linear_replacement, HQQLinear):
                if name in layers_4bit:
                    quant_config = quant_config_4bit
                    print(f"Replacing {prefix}.{name} with {linear_replacement} with 4-bit groupsize {quant_config['weight_quant_params']['group_size']}")
                elif name in layers_2bit:
                    if any([block_influence_pattern + "." in prefix for block_influence_pattern in block_influence_patterns]):
                        quant_config = quant_config_4bit
                        print(f"Block Influence: Replacing {prefix}.{name} with {linear_replacement} with 4-bit {quant_config['weight_quant_params']['group_size']}")
                    else:
                        quant_config = quant_config_2bit
                        print(f"Replacing {prefix}.{name} with {linear_replacement} with 2-bit {quant_config['weight_quant_params']['group_size']}")
                model._modules[name] = linear_replacement(module, quant_config, **kwargs)
            else:
                raise ValueError(f"Unsupported linear replacement: {type(linear_replacement)}")
    return model


def load_and_quantize(module:nn.Module, name:str, value:Tensor, device:torch.device=None, dtype:torch.dtype=None,
                      skip_names:list[str]=[], to_cpu:bool=False, to_meta:bool=False, verbose:bool=False, 
                      loftq_init:bool=False, lora_rank:int=64):
    """
    Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

    Quantizes `Params4bit` on `device` then places on "cpu" if to_cpu=True or "meta" if to_meta=True.
    """
    def place_on_device(value):
        if to_meta:
            device = 'meta'
        elif to_cpu:
            device = 'cpu'
        return value.to(device=device, dtype=dtype)

    if any([skip_name in name for skip_name in skip_names]):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return

    module_key, _, value_key = name.rpartition('.')
    try:
        submodule = module.get_submodule(module_key)
    except AttributeError as e:
        print(f"Module {module_key} not found:\n{e}")
        return

    try:
        if isinstance(submodule, HQQLinear):
            if value_key == "weight":
                # Like `Params4bit`, this workaround quantizes `HQQLinear`` per device so the quantization
                # meta dictionary is created on all ranks, before converting to meta on non-rank 0.
                value = value.to(device=device, dtype=dtype)
                submodule.linear_layer.to_empty(device=device)   
                if loftq_init:
                    # FIXME: RuntimeError: tensor does not have a device
                    submodule.del_orig = False
                    lora_A = torch.zeros(lora_rank, submodule.linear_layer.in_features, device=device, dtype=dtype) # rank x input
                    lora_B = torch.zeros(submodule.linear_layer.out_features, lora_rank, device=device, dtype=dtype) # output x rank

                    submodule.linear_layer.weight.data.copy_(value)
                    submodule.quantize(submodule.linear_layer.weight.data, **submodule.quant_config)
                    
                    W_dq = submodule.dequantize()
                    lora_B, lora_A = get_lowrank_tuple_torch_gpu(value - W_dq, lora_rank, eps=None)
                    if verbose and to_cpu: 
                        error = (value - (W_dq + lora_B @ lora_A)).norm().item()
                        print(f"LoftQ init error ({name}): {error}") 
                    del submodule.linear_layer
                    setattr(submodule, "lora_A", lora_A.to("cpu")); print(f"Lora A: {lora_A.shape}")
                    setattr(submodule, "lora_B", lora_B.to("cpu")); print(f"Lora B: {lora_B.shape}")
                
                    column_norm_init = (W_dq + lora_B @ lora_A).norm(p=2, dim=1)
                    setattr(submodule, "dora_scale", column_norm_init.to("cpu"))
                    del W_dq; torch.cuda.empty_cache()                        
                else:
                    submodule.linear_layer.weight.data.copy_(value)
                    setattr(submodule, "dora_scale", value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"))
                    submodule.initialize()

                if to_meta:
                    setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("meta")))
                elif to_cpu:
                    setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("cpu")))
                
                submodule.in_gpu = False
        else:
            param = submodule.get_parameter(value_key)
            value = type(param)(place_on_device(value).data)     
                       
    except AttributeError:
        # it's a buffer
        value = place_on_device(value)
        pass
    
    if HQQLinear is None or not isinstance(submodule, HQQLinear):
        setattr(submodule, value_key, value)
        
    
def get_lowrank_tuple_torch_gpu(tensor, max_rank, eps=None):
	t       = tensor.t().float()
	u, s, v = torch.linalg.svd(t)
	u, s, v = u[:,:max_rank], s[:max_rank], v[:max_rank, :]
	us      = torch.matmul(u, torch.diag(s))
	A, B    = (v.t(), us.t()) #t ~ AB
	if(eps is not None):
		A  = A.clamp(min=-eps, max=eps)
		B  = B.clamp(min=-eps, max=eps)
	return A.to(tensor.dtype), B.to(tensor.dtype)