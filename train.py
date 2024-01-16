"""
This script trains a model using FSDP. It pulls inspiration from
- llama-recipes TODO link
- PyTorch FSDP docs TODO link

For information on the different arguments, run `python train.py --help`

This is still a WIP and has currently only been tested with llama 7B on a single node w/ 2 GPUs

TODO: accompanying blog post
"""

# Imports

# General
import torch, os, gc, time, safetensors, copy
import functools
import torch.optim as optim
import bitsandbytes as bnb
import torch.distributed as dist
import torch.multiprocessing as mp

# Argument parsing
from fastcore.script import call_parse, bool_arg

# Torch + distributed training
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# FSDP
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Model loading
from safetensors import safe_open
from bitsandbytes.nn import Linear4bit, Params4bit
from accelerate import init_empty_weights
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
import wandb
class Logger:
    def __init__(self, args, log_every_n_steps=10, log_to="stdout", project_name="fsdp", rank=0):
        self.log_every_n_steps = log_every_n_steps
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


def load_param(module:nn.Module, name:str, value:Tensor, device=None, dtype=None):
    value = value.to(device=device, dtype=dtype)

    module_key, _, value_key = name.rpartition('.')
    submodule = module.get_submodule(module_key)
    try:
        param = submodule.get_parameter(value_key)
        if isinstance(param, Params4bit):
            value = type(param)(value.data, **param.__dict__)
            value.cuda(device) # Terrible and wrong passing in rank for now hack
        else:
            value = type(param)(value.data)
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

# Main function, run on each process
def fsdp_main(rank, world_size, args):

    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Start logging
    args["logger"] = Logger(args, log_every_n_steps=args["log_every_n_steps"], log_to=args["log_to"], rank=rank)

    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # dtype
    if args["model_dtype"] == "fp16": torch_dtype = torch.float16
    elif args["model_dtype"] == "bf16": torch_dtype = torch.bfloat16
    elif args["model_dtype"] == "fp32": torch_dtype = torch.float32
    else: raise ValueError("Invalid model_dtype")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id # TODO check if it exists first

    # # Set up dataset
    from datasets import Dataset, load_dataset
    if args["dataset"] == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")['train']
    elif args["dataset"] == "alpaca_sample":
        dataset = load_dataset("yahma/alpaca-cleaned", split="train[:20]")
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
    print("Creating model", rank)
    if args["train_type"] == "full" or args["train_type"] == "lora": # Full version
        if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
            model = AutoModelForCausalLM.from_pretrained(
                args["model_name"],
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            model.to(rank).to(torch_dtype)
        else:
            cfg = AutoConfig.from_pretrained(args["model_name"])
            cfg.use_cache = False
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg)
            model.to(torch_dtype)

    elif args["train_type"] == "hf_qlora": # Quantized version (bnb edited to use bfloat16 for storage)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            args["model_name"],
            use_cache=False,
            quantization_config=bnb_config
        )

    elif args["train_type"] == "qlora": # Our custom loading
        cfg = AutoConfig.from_pretrained(args["model_name"])
        cfg.use_cache = False
        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
            model.model = replace_linear(model.model, Linear4bit, compute_dtype=torch_dtype,
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
        # TODO: low_memory doesn't work for QLoRA. Hangs on sharding. Something special to do for this to work, following llama-recipes doesn't work
        if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
            for filename in files:
                weights = safetensors.torch.load_file(filename)
                for name, param in weights.items():
                    load_param(model, name, param, dtype=torch.bfloat16, device=rank)
        model.to(torch_dtype)

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
        model = get_peft_model(model, peft_config)
        if rank==0: model.print_trainable_parameters()
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
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True) if args["use_cpu_offload"] else None,
        limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=args["low_memory"], # TODO low memory works with LoRA but not QLoRA
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if (rank!=0 and args["low_memory"]) else None, # TODO note about meta device and why we need this
        mixed_precision=None,
    )
    print("Wrapped model", rank, torch.cuda.memory_allocated(rank))
    args["logger"].log({"memory_after_model_wrap": torch.cuda.memory_allocated(rank)}, rank)

    # Synchronize at the start
    dist.barrier()

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
    else: raise ValueError("Invalid optimizer")

    # LR scheduler TODO

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and args['verbose']:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                print(f"Shape: {param.shape}, Requires Grad: {param.requires_grad}")


    # Train loop
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

            # Print memory usage once early in training TODO better
            if batch_idx==0:
                print('Training before forwards', torch.cuda.memory_allocated(rank), rank)
                args["logger"].log({"memory_before_forward": torch.cuda.memory_allocated(rank)}, rank)

            # Forward pass
            output = model(
                batch['input_ids'].to(rank), 
                labels=batch['labels'].to(rank), 
                attention_mask=batch['attention_mask'].to(rank)
            )
            loss = output.loss

            # Print memory usage once early in training TODO better
            if batch_idx==0:
                print('Training after forwards', torch.cuda.memory_allocated(rank), rank)
                args["logger"].log({"memory_after_forward": torch.cuda.memory_allocated(rank)}, rank)
                print("Batch shape", batch['input_ids'].shape, rank)

            # Backward pass
            loss.backward()

            # Record loss
            bs = batch['input_ids'].shape[0]
            ddp_loss[0] += loss.item() * bs
            ddp_loss[1] += bs

            # Step the optimizer (w/ gradient accumulation)
            if batch_idx%args['gradient_accumulation_steps']==0:
                optimizer.step()
                optimizer.zero_grad()

            # Print memory usage after backwards
            if batch_idx==0:
                print('Training after backwards', torch.cuda.memory_allocated(rank), rank)
                args["logger"].log({"memory_after_backward": torch.cuda.memory_allocated(rank)}, rank)

            # Print peak memory usage for the whole step
            if batch_idx==0:
                peak_memory = torch.cuda.max_memory_allocated(rank)
                print(f"Peak memory usage (training): {peak_memory/1e9:.2f}GB", rank)
                args["logger"].log({"memory_peak": peak_memory}, rank)

            # Delete the output so more memory frees up (!!)
            output = None
            loss = None

            # Stop logging memory (first iter)
            if batch_idx==0 and rank == 0 and epoch == 0 and args['profile_memory']:
                torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

            # Log loss every n steps (may slow down due to all_reduce?)
            if batch_idx%args['log_every_n_steps']==0:
                dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                args["logger"].log({"loss": ddp_loss[0] / ddp_loss[1]}, rank)
                ddp_loss = torch.zeros(2).to(rank)

        # Print loss NB only last n_steps average not epoch average
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            if args["verbose"]: print(f"Epoch {epoch} loss: {ddp_loss[0] / ddp_loss[1]}")
            args["logger"].log({"loss": ddp_loss[0] / ddp_loss[1]})

    init_end_event.record()

    # Print time and model
    if rank == 0:
        time_taken = init_start_event.elapsed_time(init_end_event) / 1000
        print(f"CUDA event elapsed time: {time_taken} sec")
        args["logger"].log({"time_taken": time_taken})

    # Save model (TODO)
    print("Finished training", rank, torch.cuda.memory_allocated(rank))

    # Clean up
    if rank==0: args["logger"].finish(rank=rank)
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
    dataset: str = "alpaca_sample", # alpaca, alpaca_sample (for a 20-sample test) or "dummy" for 20 long dummy samples
    use_gradient_checkpointing: bool_arg = True, # Whether to use fsdp's activation checkpointing
    use_cpu_offload: bool_arg = False, # Whether to use fsdp's cpu offload
    low_memory: bool_arg = False, # Load model weights only on Rank 0 to reduce CPU memory usage. Currently works for LoRA but not for QLoRA.
    model_dtype: str = "bf16", # fp16, bf16 or fp32
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "output", # Output directory to save results to TODO
    lora_rank: int = 8, # LoRA rank for lora/qlora
    lora_alpha: int = 32, # LoRA alpha for lora/qlora
    lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
    lora_target_modules = "all", # If 'none', uses peft defaults. Use 'all' for our best guess for mistral+llama
    verbose: bool_arg = True, # Whether to print extra info for debugging
    lr: float = 1e-4, # Learning rate
    profile_memory: bool_arg = False, # Whether to profile memory usage for the first batch
    optimizer: str = "adadelta", # adam, sgd or adadelta
    log_to: str = "stdout", # wandb or stdout
    log_every_n_steps: int = 10, # How frequently to log loss
    wrapping_policy: str = "llamarecipes", # "size" or "llamarecipes" to test different things TODO size doesn't work for QLoRA
):
    # Set world size
    if world_size == -1:
        world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    # Get all args which will be passed to fsdp_main
    args = dict(locals())
    if args['verbose']: print(args)

    # If lora_target_modules is 'all', set sensible defaults for llama + mistral type modules
    # See peft.utils.constants -> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING for the current defaults
    if lora_target_modules == "all":
        args["lora_target_modules"] = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    elif lora_target_modules.lower() == "none":
        args["lora_target_modules"] = None

    # Run
    mp.spawn(fsdp_main,
        args=(world_size, args),
        nprocs=world_size,
        join=True)