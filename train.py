"""
This script trains a model using FSDP. It pulls inspiration from
- llama-recipes (https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py)
- PyTorch FSDP docs (https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- bitsandbytes (https://github.com/TimDettmers/bitsandbytes)

For information on the different arguments, run `python train.py --help`

This is still a WIP and has currently only been tested with Llama 7B, Mistal 7B, & TinyLlama on a single node w/ 2 GPUs.
Not all combinations of arguments will work. See the accompanying blog post for more details.
"""

# Imports

# General
import torch, os, gc, time, safetensors, copy, math, types
import functools
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule
import bitsandbytes as bnb
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from safetensors.torch import save_file
from tqdm.auto import tqdm

# Argument parsing
from fastcore.script import call_parse, bool_arg, Param

# Torch + distributed training
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# FSDP
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Model loading
from safetensors import safe_open
from bitsandbytes.nn import Linear4bit, Params4bit
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from peft import get_peft_model, LoraConfig, TaskType
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


# For different model types, we'll want to import the right class for the
# check_fn in activation checkpointing (LlamaDecoderLayer for llama models for example)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
# Set the target class for activation checkpointing here:
GC_LAYER_CLASS = LlamaDecoderLayer

# To get rid of tokenizers warnings for now
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For logging things during training
try:
    import wandb
except ImportError:
    pass

class Logger:
    def __init__(self, args, log_to="stdout", project_name="fsdp-benchmarking", rank=0):
        # self.log_every_n_steps = log_every_n_steps TODO: add this back as an option
        self.log_to = log_to
        if self.log_to == "wandb" and rank==0:
            import wandb
            wandb.init(project=project_name)
            wandb.config.update(args)
        elif self.log_to == "stdout":
            print(args)

    def log(self, d, rank=0):
        if rank != 0: return
        if self.log_to == "wandb": wandb.log(d)
        elif self.log_to == "stdout": print(d)

    def finish(self, rank=0):
        if self.log_to == "wandb" and rank==0: wandb.finish()

# Utilities related to model loading
def replace_linear(model, linear_replacement, skip_modules=["lm_head"], **kwargs):
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
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, **kwargs)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
                **kwargs
            )
    return model


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print("Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def setup_quantized_meta_for_peft(model:nn.Module):
    def temp_to_method(self, *args, **kwargs):
        return self
    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)

def setup_quantized_peft_meta_for_training(model:nn.Module):
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, '_orig_to'):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None


def load_param(module:nn.Module, name:str, value:Tensor, device=None, dtype=None,
               skip_names=[], is_meta_rank:bool=False, verbose:bool=False):
    value = value.to(device=device, dtype=dtype)
    module_key, _, value_key = name.rpartition('.')
    
    try:
        submodule = module.get_submodule(module_key)
    except Exception as e:
        if verbose:
            print(f"Failed to load {name} into {module_key}")
            print(e)
            return    
    
    if any([skip_name in name for skip_name in skip_names]):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return
    
    if verbose:
        print(f"Loading {name} into {module_key}")
    
    try:
        param = submodule.get_parameter(value_key)
        if isinstance(param, Params4bit):
            value = type(param)(value.data, **param.__dict__)
            value = value.cuda(device)

            # With `sync_module_states=True`, a meta device Params4bit needs to be the same
            # shape as the quantized Params4bit with an initialized quant_state. However,
            # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
            # workaround quantizes Params4bit to initialize quant_state on all ranks, then
            # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
            if is_meta_rank:
                value = type(value)(value.data.to("meta"), **value.__dict__)
                # param.quant_state = value.quant_state
            else:
                value = type(value)(value.data.to("cpu"), **value.__dict__)
                # param.quant_state = value.quant_state
        else:
            if is_meta_rank:
                value = type(param)(value.data.to("meta"))
            else:
                value = type(param)(value.data.to("cpu"))
        
    except AttributeError:
        pass  # it's a buffer
    setattr(submodule, value_key, value)


### DATASET (modified from llama recipes)
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, partition="train"):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.dataset[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }

# LR scheduler.
def _get_cosine_one_cycle_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_fraction = 0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    scale_term = (1 - min_lr_fraction)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return (math.cos(math.pi * progress)+1) * 0.5 * scale_term + min_lr_fraction

def get_cosine_one_cycle_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_fraction=0.1):
    "A more general cosine scheduler with to control the minimum learning rate"
    lr_lambda = functools.partial(
        _get_cosine_one_cycle_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_fraction=min_lr_fraction
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

# Main function, run on each process
def fsdp_main(rank, world_size, args):

    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Start logging
    args["logger"] = Logger(args, log_to=args["log_to"], rank=rank)

    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype. The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses `torch.amp.autocast` to control precision.
    # limited qlora testing shows that fp16 only works with autocast while bf16 trains with both pure and autocast modes.
    # TODO: test how often this holds for mp_fp16
    mp_policy = None
    load_param_skip_names = []
    if args["precision"] == "bf16":
        torch_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16
    elif args["precision"] == "fp32":
        torch_dtype = torch.float32
        compute_dtype = torch.float16
    elif args["precision"] == "fp16_autocast":
        compute_dtype = torch.float16
        torch_dtype = torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_autocast":
        compute_dtype = torch.bfloat16
        torch_dtype = torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_buffers_autocast":
        compute_dtype = torch.bfloat16
        torch_dtype = torch.bfloat16
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
        load_param_skip_names = ['inv_freq']
    else:
        raise ValueError("Invalid precision")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id # TODO check if it exists first

    # # Set up dataset
    from datasets import Dataset, load_dataset
    if args["dataset"] == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")['train']
    elif args["dataset"] == "alpaca_sample":
        # dataset = load_dataset("yahma/alpaca-cleaned", split="train[:20]")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train[:128]")
    elif args["dataset"] == "dummy":
        dataset = Dataset.from_dict({
            'instruction': ["instruction"]*16,
            'input': ["input"]*16,
            'output': ["output"*10000]*16} # A long output to test memory usage (gets truncated)
        )

    dataset = InstructionDataset(dataset, tokenizer)
    def collate_fn(batch):
        # To list of tensors
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        # Pad + truncate
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :args["context_length"]]
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:, :args["context_length"]]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)[:, :args["context_length"]]
        # Return dict
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(dataset)

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_fn, sampler=sampler)


    # Create model
    use_flash_attn = args['use_flash_attention']
    print("Creating model", rank)
    if args["train_type"] == "full" or args["train_type"] == "lora": # Full version
        if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
            model = AutoModelForCausalLM.from_pretrained(
                args["model_name"],
                use_cache=False,
                torch_dtype=torch_dtype,
                _attn_implementation="flash_attention_2" if use_flash_attn else "sdpa"
            )
            # model.to(rank).to(torch_dtype) # Don't need, causes OOM. First load to cpu and then shard to GPU.
        else:
            cfg = AutoConfig.from_pretrained(args["model_name"])
            cfg.use_cache = False
            cfg._attn_implementation = "flash_attention_2" if use_flash_attn else "sdpa"
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg)
            model.to(torch_dtype)

    elif args["train_type"] == "hf_qlora": # Quantized version (bnb edited to use bfloat16 for storage)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=compute_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            args["model_name"],
            use_cache=False,
            quantization_config=bnb_config,
            _attn_implementation="flash_attention_2" if use_flash_attn else "sdpa"
        )

    elif args["train_type"] in ["qlora", "custom_qlora"]: # Our custom loading
        cfg = AutoConfig.from_pretrained(args["model_name"])
        cfg.use_cache = False
        cfg._attn_implementation = "flash_attention_2" if use_flash_attn else "sdpa"
        # cfg.update(dict(num_hidden_layers=60)) # debug mode.
        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
            model.model = replace_linear(model.model, Linear4bit, compute_dtype=compute_dtype,
                                         quant_type='nf4', quant_storage=torch_dtype)
        model.is_loaded_in_4bit = True

        # Grab the safetensors files that hold the weights
        try:
            idx = hub.cached_file(args["model_name"], SAFE_WEIGHTS_INDEX_NAME)
            files, _ = hub.get_checkpoint_shard_files(args["model_name"], idx)
        except OSError:
            try:
                # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                files = []
                files.append(hub.cached_file(args["model_name"], SAFE_WEIGHTS_NAME))
            except OSError as e:
                # This means the model probably doesn't have a safetensors file
                raise e

        # Load in the weights, using our custom load_param function which quantizes Params4bit on the fly
        for filename in files:
            weights = safetensors.torch.load_file(filename)
            for name, param in weights.items():
                load_param(model, name, param, dtype=torch_dtype, device=rank, skip_names=load_param_skip_names,
                            is_meta_rank=(args["low_memory"] and rank!=0), verbose=args["verbose"])

    print("Model created", rank, torch.cuda.memory_allocated(rank))

    # PEFT setup (LoRA and QLoRA)
    if args["train_type"] == "lora" or args["train_type"] == "qlora" or args["train_type"] == "hf_qlora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args["lora_rank"],
            lora_alpha=args["lora_alpha"],
            lora_dropout=args["lora_dropout"],
            target_modules=args["lora_target_modules"],
        )
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        if rank!=0 and args["low_memory"]:
            setup_quantized_meta_for_peft(model)

        model = get_peft_model(model, peft_config)

        # And then setup_quantized_peft_meta_for_training returns quant_state.to back to normal
        if rank==0:
            model.print_trainable_parameters()
        elif args['low_memory']:
            setup_quantized_peft_meta_for_training(model)
    elif args["train_type"] == "custom_qlora":
        # Create custom lora module
        class QLORA(nn.Module):
            def __init__(self, base_layer, device="cpu"):
                super().__init__()
                self.base_layer = base_layer
                dtype = base_layer.compute_dtype
                self.lora_A = nn.Linear(base_layer.in_features, args["lora_rank"], bias=False, device=device, dtype=dtype)
                self.lora_B = nn.Linear(args["lora_rank"], base_layer.out_features, bias=False, device=device, dtype=dtype)
                self.lora_alpha = args["lora_alpha"]
                self.lora_dropout = nn.Dropout(args["lora_dropout"])
                self.scaling = self.lora_alpha / args['lora_rank']

            def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

                result = self.base_layer(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(self.lora_A.weight.dtype)

                output = self.lora_B(self.lora_A(self.lora_dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * self.scaling
                
                # print(f"rank {rank} output shape {output.shape}, result shape {result.shape}")
                result += output

                return result
            
        for name, _ in model.named_modules():
            module_key, _, value_key = name.rpartition('.')
            if value_key in args['lora_target_modules']:
                m = model.get_submodule(name)
                qlora_layer = QLORA(m)
                parent_module = model.get_submodule(module_key)
                setattr(parent_module, value_key, qlora_layer)
        
        for n,p in model.named_parameters():
            if any([lora_name in n for lora_name in ['lora_A', 'lora_B']]):
                p.requires_grad = True
            else:
                p.requires_grad = False

        print("LoRA layers added", rank, torch.cuda.memory_allocated(rank))
        
    args["logger"].log({"memory_after_model_creation": torch.cuda.memory_allocated(rank)}, rank)

    # Wrap the model
    if args["wrapping_policy"] == "size":
        # Wrapping policy: wrap anything with more than 8 parameters individually:
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=8
        )
    else:
        # Alternative: policy from llama-recipes:
        from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
        from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
        # I think this checks for lora layers (has weight and requires_grad)
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        # And then this matches the rest?
        transformer_layer_name = LlamaDecoderLayer
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=(
                PrefixEncoder,
                PromptEncoder,
                PromptEmbedding,
                transformer_layer_name,
            ),
        )
        my_auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])

    print("Wrapping model w/ FSDP", rank)
    sharding_strategy = ShardingStrategy.FULL_SHARD if not args['use_ddp'] else ShardingStrategy.NO_SHARD
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True) if args["use_cpu_offload"] else None,
        limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=args["low_memory"], # TODO low memory works with LoRA but not QLoRA
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if (rank!=0 and args["low_memory"]) else None, # TODO note about meta device and why we need this
        mixed_precision=mp_policy,
    )
    print("Wrapped model", rank, torch.cuda.memory_allocated(rank))
    args["logger"].log({"memory_after_model_wrap": torch.cuda.memory_allocated(rank)}, rank)

    if rank == 0: print(model)
    # raise ValueError("Stop here")
    # print("Embed Model dtype (WRAPPED MODEL)", model._fsdp_wrapped_module.base_model.model.model.embed_tokens.weight.dtype)
    # print("Buffers dtype (WRAPPED MODEL)", next(model.buffers()).dtype)
    # print("Params dtype (WRAPPED MODEL)", next(model.parameters()).dtype)
    # print("Model Mixed precision", model.mixed_precision.param_dtype)
    # print("LORA Mixed precision", model.mixed_precision.param_dtype)    
    # # import pdb; pdb.set_trace()
    # decoder_layer = model._fsdp_wrapped_module.base_model.model.model.layers[0]
    # print("Decoder Mixed precision", decoder_layer.mixed_precision.param_dtype)
    # print("Decoder FWD pre-hook:", decoder_layer._forward_pre_hooks)
    # lora_layer = decoder_layer._fsdp_wrapped_module.self_attn.q_proj.lora_A
    # lora_base_layer = decoder_layer._fsdp_wrapped_module.self_attn.q_proj.base_layer
    
    # print(f"rank: {rank}, lora layer dtypes and devices")
    # print([(p.shape, p.dtype, p.device) for p in list(lora_layer.parameters())])
   
    # print(f"rank: {rank}, lora base layer dtypes and devices")
    # print([lora_base_layer.weight.shape, lora_base_layer.weight.dtype, lora_base_layer.weight.device])
    
    
    # save lora base layer weights to verify correct loading in all ranks.
    # torch.save(lora_base_layer.state_dict(), f"data/lora_layer0_q_proj_base_layer_rank{rank}.pt")
    # lora_layer = decoder_layer._fsdp_wrapped_module.self_attn.q_proj.lora_A
    # print("Lora_A FWD pre-hook:", lora_layer._forward_pre_hooks)
    # from torch.distributed.fsdp._common_utils import _is_fsdp_flattened
    # # print([(p.shape, p.dtype, _is_fsdp_flattened(p)) for p in list(decoder_layer.parameters())])
    # torch.save(lora_base_layer.quant_state, f"data/lora_layer0_q_proj_quant_state_rank{rank}.pt")
    # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    #     cpu_state_dict = lora_base_layer.state_dict()
    #     torch.save(cpu_state_dict, f"data/lora_layer0_q_proj_base_layer_params_rank{rank}.pt")
    
        
    # For mem-eff loading testing.
    # Summon module at each rank, and then save for comparsion.
    # Compare quant_state, params, and also compare it with original loaded model weights.
    # decoder_layer = model._fsdp_wrapped_module.base_model.model.model.layers[0]
    # lora_base_layer = decoder_layer._fsdp_wrapped_module.self_attn.q_proj.base_layer
    # with FSDP.summon_full_params(decoder_layer, recurse=True, offload_to_cpu=True, rank0_only=False):
    #     torch.save(lora_base_layer.quant_state, f"data/summoned_lora_layer0_q_proj_quant_state_rank{rank}.pt")
    #     torch.save(list(lora_base_layer.parameters()), f"data/summoned_lora_layer0_q_proj_base_layer_params_rank{rank}.pt")

    
    # Synchronize at the start
    dist.barrier()
    # raise ValueError("Stop here")

    # Apply activation checkpointing
    if args["use_gradient_checkpointing"]:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, GC_LAYER_CLASS)
        print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if rank == 0 and args['verbose']:
        print("Model:")
        print(model)
        print("Starting training")

    # Create optimizer TODO more options here
    if args["optimizer"] == "adam": optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "sgd": optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "adadelta": optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "adamw": optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], 
                                                                     betas=(0.9,0.95), eps=1e-5, weight_decay=args['wd'])
    else: raise ValueError("Invalid optimizer")

    gradient_accumulation_steps = max(1, args['gradient_accumulation_steps'])

    # LR scheduler.
    num_training_steps = args['num_epochs'] * len(dataloader) // gradient_accumulation_steps
    num_scheduler_steps = num_training_steps * dist.get_world_size()
    num_warmup_steps = int(num_scheduler_steps * 0.1)
    if args['lr_scheduler'] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_scheduler_steps)
    elif args['lr_scheduler'] == "cosine":
        lr_scheduler = get_cosine_one_cycle_scheduler(optimizer, num_warmup_steps, num_scheduler_steps, min_lr_fraction=0.1)
    elif args['lr_scheduler'] == "constant":
        lr_scheduler = get_constant_schedule(optimizer)
    else:
        raise NotImplementedError(f"{args['lr_scheduler']} LR scheduler not implemented yet")

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and args['verbose']:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                print(f"Shape: {param.shape}, Requires Grad: {param.requires_grad}")

    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if args["precision"] in ["fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if args["precision"] == "fp16_autocast" else None
    scale_loss = scaler is not None

    # Train loop
    # TODO: no_sync() is needed to accumulate gradients with cpu offloading.
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    if rank == 0: print("Total Training Steps:", num_training_steps)
    init_start_event.record()
    for epoch in range(args['num_epochs']):
        model.train()
        ddp_loss = torch.zeros(2).to(rank)
        for batch_idx, batch in enumerate(dataloader):

            if rank == 0 and args['verbose']: print(f"Epoch {epoch}, Batch {batch_idx}")

            # Start logging memory (first iter) if requested
            if batch_idx==0 and rank == 0 and epoch == 0 and args['profile_memory']:
                torch.cuda.memory._record_memory_history()

            # Reset peak memory to track that
            torch.cuda.reset_peak_memory_stats(rank)

            # Log memory usage
            if batch_idx==0: args["logger"].log({"memory_before_forward": torch.cuda.memory_allocated(rank)}, rank)

            # Forward pass
            with autocast:
                output = model(
                    batch['input_ids'].to(rank),
                    labels=batch['labels'].to(rank),
                    attention_mask=batch['attention_mask'].to(rank)
                )
                loss = output.loss

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Log memory usage
            if batch_idx==0: args["logger"].log({"memory_after_forward": torch.cuda.memory_allocated(rank)}, rank)

            # Backward pass
            if scale_loss:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Record loss
            bs = batch['input_ids'].shape[0]
            ddp_loss[0] += loss.item() * bs
            ddp_loss[1] += bs

            # Step the optimizer (w/ gradient accumulation)
            if batch_idx % gradient_accumulation_steps == 0:
                if args['grad_norm'] is not None:
                    model.clip_grad_norm_(args['grad_norm'], norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                progress_bar.update(1)
            # Log memory usage after backwards
            if batch_idx==0: args["logger"].log({"memory_after_backward": torch.cuda.memory_allocated(rank)}, rank)

            # Print + log peak memory usage for the whole first step of training
            if batch_idx==0:
                peak_memory = torch.cuda.max_memory_allocated(rank)
                print(f"Peak memory usage (training): {peak_memory/1e9:.2f}GB", rank)
                args["logger"].log({"memory_peak": peak_memory}, rank)

            # Delete the output so more memory frees up before the next forward pass
            output = None
            loss = None

            # Stop logging memory (first iter)
            if batch_idx==0 and rank == 0 and epoch == 0 and args['profile_memory']:
                torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

            # Log loss every gradient update steps
            if batch_idx % gradient_accumulation_steps == 0:
                dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                args["logger"].log({"loss": ddp_loss[0] / ddp_loss[1]}, rank)
                ddp_loss = torch.zeros(2).to(rank)
                args["logger"].log({"lr": lr_scheduler.get_last_lr()[0]}, rank)

    # Synchronize at the end and record time
    dist.barrier()
    torch.cuda.synchronize()
    init_end_event.record()

    print("Finished training", rank)

    # Print time and model
    if rank == 0:
        time_taken = init_start_event.elapsed_time(init_end_event) / 1000
        print(f"CUDA event elapsed time: {time_taken} sec")
        args["logger"].log({"time_taken": time_taken})

    # End logging
    args["logger"].finish(rank=rank)

    # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
    if args["save_model"]:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = model.state_dict()
            os.makedirs(args["output_dir"], exist_ok=True)
            if rank==0:
                print("Saving model")
                save_file(cpu_state_dict, os.path.join(args["output_dir"], "model_state_dict.safetensors"))
                print("Done", rank)

    dist.barrier() # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()

# Entry point, using fastcore's call_parse to parse args from command line and then calling fsdp_main
@call_parse()
def main(
    world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
    train_type: str = "lora", # "full", "lora", or "qlora"
    batch_size: int = 1, # Batch size per GPU for training
    context_length: int = 512, # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1, # How many epochs of training to do
    dataset: str = "alpaca_sample", # alpaca, alpaca_sample (for a 20-sample test) or "dummy" for 16 long dummy samples
    use_ddp: bool_arg = False, # Whether to use DDP instead of FSDP with full sharding
    use_flash_attention: bool_arg = True, # Whether to use flash attention
    use_gradient_checkpointing: bool_arg = True, # Whether to use fsdp's activation checkpointing
    use_cpu_offload: bool_arg = False, # Whether to use fsdp's cpu offload
    low_memory: bool_arg = True, # Load model weights only on Rank 0 to reduce CPU memory usage. Currently works for LoRA but not for QLoRA.
    precision: Param("", choices=["fp32", "bf16", "fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]) = "bf16_buffers_autocast", # mixed precision training. "fp32", "bf16", "mp_fp16_autocast", "mp_bf16_autocast", "mp_bf16_buffers_autocast".
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    save_model: bool_arg = False, # Whether to save the resulting model TODO
    output_dir: str = "output", # Output directory to save results to TODO
    lora_rank: int = 64, # LoRA rank for lora/qlora
    lora_alpha: int = 16, # LoRA alpha for lora/qlora
    lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
    lora_target_modules = "all", # If 'none', uses peft defaults. Use 'all' for our best guess for mistral+llama
    verbose: bool_arg = True, # Whether to print extra info for debugging
    lr: float = 1e-5, # Learning rate
    grad_norm: float = 0.3, # Gradient norm clipping
    wd: float = 0.1, # Weight decay
    profile_memory: bool_arg = False, # Whether to profile memory usage for the first batch
    optimizer: str = "adamw", # adam, sgd or adadelta
    lr_scheduler: Param("", choices=["constant", "linear", "cosine"]) = "constant", # lr scheduler to use
    log_to: str = "wandb", # wandb or stdout
    wrapping_policy: str = "llamarecipes", # "size" or "llamarecipes" to test different things TODO size doesn't work for QLoRA
    master_addr: str = "localhost", # For distributed training
    master_port: str = "12355", # For distributed training, must be the same for all processes
    seed: int = 42, # Random seed
):

    # Set world size
    if world_size == -1:
        world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    # Get all args which will be passed to fsdp_main
    args = dict(locals())
    set_seed(args['seed'])
    if args['verbose']: print(args)

    # If lora_target_modules is 'all', set sensible defaults for llama + mistral type modules
    # See peft.utils.constants -> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING for the current defaults
    if lora_target_modules == "all":
        args["lora_target_modules"] = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    elif lora_target_modules.lower() == "none":
        args["lora_target_modules"] = None

    if args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"] and not torch.cuda.is_bf16_supported():
        raise ValueError('Current device does not support bfloat16')

    # Run
    mp.spawn(fsdp_main,
        args=(world_size, args),
        nprocs=world_size,
        join=True)