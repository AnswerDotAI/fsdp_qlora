import os
import functools
from tqdm import tqdm
from typing import Dict
import math

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MISTRAL_ATTENTION_CLASSES, MistralMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, QWEN2_ATTENTION_CLASSES, Qwen2MLP
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, PHI3_ATTENTION_CLASSES, Phi3MLP


from .dora import DORALayer, MagnitudeLayer

class Logger:
	def __init__(self, args, log_to="stdout", project_name="fsdp_qlora", entity=None, group=None, name=None, rank=0):
		self.log_to = log_to
		if self.log_to == "wandb" and rank==0:
			import wandb
			wandb.init(project=project_name, entity=entity, group=group, name=name, config=args)

	def log(self, d:Dict, rank:int):
		if rank != 0: return
		if self.log_to == "tqdm":
			for k,v in d.items():
				tqdm.write(f'{k}: {v}')
		elif self.log_to == "wandb":
			wandb.log(d)
		elif self.log_to == "stdout":
			for k,v in d.items():
				print(f'{k}: {v}')

	def finish(self, rank=0):
		if self.log_to == "wandb" and rank==0: wandb.finish()
  

def update_progress_bar(progress_bar:tqdm, epoch:int, log_loss:float, log_lr:float, rank:int):
	"""Updates the progress bar with the current epoch, loss, and learning rate"""
	if rank == 0:
		if log_lr >=0:
			progress_bar.set_description(f"Epoch {epoch}, Loss {log_loss:.3f}, LR {log_lr:.2e}", refresh=True)
		else:
			progress_bar.set_description(f"Epoch {epoch}, Loss {log_loss:.3f}", refresh=True)

def n_loading_workers(quant_method:str, param_count:float):
	devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
	left = int(os.cpu_count()/torch.cuda.device_count())
	right = int((4 if quant_method == "hqq" else 8) * (devprops.total_memory/1e9/40) * (70/(param_count/1e9)))
	return min(left, right)


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(custom_policy:bool=False, vanilla_policy:bool=False):

    if custom_policy:
        def lambda_policy_fn(module):
            # LoRA and DoRA trainable layers.
            return ((isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module)) or 
                    (isinstance(module, (DORALayer, MagnitudeLayer))))
    else:
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
    
    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, tuple((*LLAMA_ATTENTION_CLASSES.values(), 
                                         *MISTRAL_ATTENTION_CLASSES.values(),
                                         *QWEN2_ATTENTION_CLASSES.values(), 
                                         *PHI3_ATTENTION_CLASSES.values())))

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaMLP, MistralMLP, Qwen2MLP, Phi3MLP))
    
    def layernorm_policy_fn(module):
         return isinstance(module, LlamaRMSNorm) and module.weight.requires_grad

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    layernorm_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=layernorm_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer, Phi3DecoderLayer),
    )
    if vanilla_policy:
        return transformer_wrap_policy
    
    policies=[lambda_policy, transformer_wrap_policy]
    if not vanilla_policy:
        policies.extend([self_attn_policy, mlp_policy, layernorm_policy])
    return functools.partial(_or_policy, policies=policies)


def _get_cosine_one_cycle_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_fraction = 0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    scale_term = (1 - min_lr_fraction)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return (math.cos(math.pi * progress)+1) * 0.5 * scale_term + min_lr_fraction

def get_cosine_one_cycle_scheduler(optimizer:optim.Optimizer, num_warmup_steps:int, num_training_steps:int, min_lr_fraction:float=0.1):
    "A more general cosine scheduler with to control the minimum learning rate"
    lr_lambda = functools.partial(
        _get_cosine_one_cycle_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_fraction=min_lr_fraction
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_lr_scheduler(optimizer:optim.Optimizer, dataloader:DataLoader, gradient_accumulation_steps:int, args:Dict):
    """Returns linear, cosine, or constant learning rate scheduler"""
    num_training_steps = args['num_epochs'] * len(dataloader) // gradient_accumulation_steps
    # num_warmup_steps = int(num_training_steps * 0.1)
    num_warmup_steps = 0
    if args['lr_scheduler'] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args['lr_scheduler'] == "cosine":
        lr_scheduler = get_cosine_one_cycle_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_fraction=0.1)
    elif args['lr_scheduler'] == "constant":
        lr_scheduler = None
    else:
        raise NotImplementedError(f"{args['lr_scheduler']} LR scheduler not implemented yet")
    return lr_scheduler, num_training_steps


# Optimizer
def get_optimizer(model:nn.Module, args:Dict, rank:int):
    """Returns an optimizer. We can add more options here if needed."""
    
    params = model.parameters()
    grouped_params = args['train_layernorms'] or (args["nbits"] == "mixed" and args["disc_lr"])
        
    # Iterate through the named modules of the model.
    if grouped_params:
        param_dict = {param_name: param for param_name, param in model.named_parameters()}
        no_decay = []
        if args["train_layernorms"]:
            for module_name, module in model.named_modules():
                # Check if the current module is an instance of any of the desired types (LayerNorm or torch.nn.Embedding).
                if isinstance(module, (LlamaRMSNorm)) and any(layer in module_name for layer in ['input_layernorm', 'post_attention_layernorm']):
                    no_decay.append(f"{module_name}.weight")
                
        # Create an empty list to store the names of the Linear layer weights with weight decay.
        decay_base_lr = [] # e.g. lr
        decay_lower_lr = [] # e.g. lr / 10
        if (args["nbits"] == "mixed") and args["disc_lr"]:
            layers_base_lr  = ["gate_proj", "up_proj", "down_proj"] # mlp 2bit
            layers_lower_lr = ["q_proj", "k_proj", "v_proj", "o_proj"] # attn 4bit
        else:
            layers_base_lr = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            layers_lower_lr  = []
            
        # Iterate through the named modules of the model.
        for module_name, module in model.named_modules():
            # Check if the current module is an instance of the desired type (torch.nn.Linear).
            if isinstance(module, (torch.nn.Linear)):
                suffix = "weight"
            elif isinstance(module, (MagnitudeLayer)):
                suffix = "magnitude"
            else:
                continue
            # If the module is an instance of torch.nn.Linear, append its name with a ".weight" suffix to the decay list.
            if any(layer in module_name for layer in layers_base_lr):
                decay_base_lr.append(f"{module_name}.{suffix}")
            elif any(layer in module_name for layer in layers_lower_lr):
                decay_lower_lr.append(f"{module_name}.{suffix}")
            else:
                continue
        
        base_lr = args['lr']
        lower_lr = args['lr'] / args['lr_div_factor']
        
        if args["verbose"] and rank == 0:
            print("No decay params:")
            for l in no_decay: print(l)
            print(f"Decay base lr={base_lr} params:")
            for l in decay_base_lr: print(l)
            print(f"Decay lower lr={lower_lr} params:")
            for l in decay_lower_lr: print(l)
            
            print("Param dict:")
            for param_name, param in param_dict.items():
                print(param_name, param.requires_grad)
            
        no_decay_param = []
        for param_name in no_decay:
            no_decay_param.append(param_dict[param_name])
        decay_base_lr_param = []
        for param_name in decay_base_lr:
            decay_base_lr_param.append(param_dict[param_name])
        decay_lower_lr_param = []
        for param_name in decay_lower_lr:
            decay_lower_lr_param.append(param_dict[param_name])
            
        grouped_params = []
        if len(no_decay_param) > 0:
            grouped_params.append({"params": no_decay_param, "lr": base_lr, "weight_decay": 0.0})
        if len(decay_base_lr_param) > 0:
            grouped_params.append({"params": decay_base_lr_param, "lr": base_lr, "weight_decay": args['wd']})
        if len(decay_lower_lr_param) > 0:
            grouped_params.append({"params": decay_lower_lr_param, "lr": lower_lr, "weight_decay": args['wd']})
    
    if args["optimizer"] == "adam":
        return optim.Adam(params, lr=args['lr'])
    elif args["optimizer"] == "sgd":
        return optim.SGD(params, lr=args['lr'])
    elif args["optimizer"] == "adadelta":
        return optim.Adadelta(params, lr=args['lr'])
    elif args["optimizer"] == "adamw":
        if grouped_params:
            return optim.AdamW(grouped_params, betas=(0.9,0.95), eps=1e-5)
        else:
            return optim.AdamW(params, lr=args['lr'], betas=(0.9,0.95), eps=1e-5, weight_decay=args['wd'])
    else:
        raise ValueError("Invalid optimizer")
