"""
Read our announcement blog post: https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html.

This script trains a model using FSDP with LoRA & QLoRA. It pulls inspiration from
- llama-recipes (https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py)
- PyTorch FSDP docs (https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- bitsandbytes (https://github.com/TimDettmers/bitsandbytes)

For information on the different arguments, run `python train.py --help`

You should treat this script as an alpha/preview release. If you're not comfortable with testing and debugging
models, we'd suggest holding off for a few months while the community more fully tests the approach.
"""

# Imports

# General
import torch, os, time, safetensors, sys, json
import functools
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from safetensors.torch import save_file
from tqdm.auto import tqdm
from typing import Dict

# Argument parsing
from fastcore.script import call_parse, bool_arg, Param

# FSDP
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Model loading
# from bitsandbytes.nn import Linear4bit, Params4bit
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from fastcore.parallel import parallel

from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig

# To add a new model, import the transformer, attention, & MLP layers
# for the wrapping policy and `check_fn` in activation checkpointing
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaFlashAttention2
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralFlashAttention2
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2FlashAttention2
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3FlashAttention2, Phi3SdpaAttention

# To get rid of tokenizers warnings for now
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For logging things during training
try:
    import wandb
except ImportError:
    pass

# LoRA and DORA modules
sys.path.append(".")
from scripts.dora import HQQDORA
from scripts.quant_utils import replace_linear, load_and_quantize
from scripts.train_utils import Logger, update_progress_bar, get_wrapping_policy, get_optimizer, get_lr_scheduler
from scripts.dataset_utils import get_dataloader


def save_model(rank, model, args, cfg, compute_dtype, layer_nbits, layer_groupsizes, step=None):
    
    if step is None:
        output_dir = args["output_dir"]
    else:
        output_dir = os.path.join(args["output_dir"], f"step_{step}")
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    dist.barrier()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if args["train_type"] in ["hqq_dora"]:
        cpu_state_dict = {}
        trainable_fsdp_modules = [(n,m) for n,m in model.named_modules() if n.endswith(('dora_layer', 'magnitude_layer'))]
        for prefix, module in trainable_fsdp_modules:
            prefix = (prefix.replace("_fsdp_wrapped_module.", "")
                            .replace("_checkpoint_wrapped_module.", "")
                            .replace("_offload_wrapped_module.", "")
                            .replace("_orig_mod.", ""))
            if args['verbose']: print(f"Saving {prefix}")
            with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = {**cpu_state_dict, **{f"{prefix}.{k}":v for k,v in module.state_dict().items()}}
            dist.barrier()
            torch.cuda.synchronize()
            
        if rank==0:
            print("Saving trained LoRA weights.")
            save_file(cpu_state_dict, os.path.join(output_dir, "model_state_dict.safetensors"))
            print("Done", rank)
            print("Saving model config as json.")

            # Save QLoRA config.
            qlora_config_dict = {}
            qlora_config_dict["lora_target_modules"] = args["lora_target_modules"]
            qlora_config_dict["compute_dtype"]       = str(compute_dtype).split(".")[-1]
            qlora_config_dict["lora_rank"]           = args["lora_rank"]
            qlora_config_dict["layer_nbits"]         = layer_nbits
            qlora_config_dict["layer_groupsizes"]    = layer_groupsizes

            model_config_filename = os.path.join(output_dir, "config.json")
            config_dict = cfg.to_dict()
            config_dict["qlora_config"] = qlora_config_dict
            with open(model_config_filename, "w+") as f: 
                json.dump(config_dict, f)    

    
# Main function, run on each process
def fsdp_main(local_rank:int, world_size:int, args:Dict):
    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    if 'SLURM_PROCID' in os.environ:
        # assumes same number of GPUs per node.
        rank = int(os.environ['SLURM_PROCID']) * torch.cuda.device_count() + local_rank
    else:
        rank = local_rank

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # Start logging
    logger = Logger(args, log_to=args["log_to"], project_name=args["project_name"],
                    entity=args["entity"], group=args["group"], name=args["name"], rank=rank)

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
        torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
    elif args["precision"] == "fp32":
        torch_dtype, compute_dtype = torch.float32, torch.float16
    elif args["precision"] == "fp16_autocast":
        compute_dtype, torch_dtype = torch.float16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_buffers_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.bfloat16
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
        load_param_skip_names = ['inv_freq']
    else:
        raise ValueError("Invalid precision")


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id # TODO check if it exists first

    # Set up dataloader
    dataloader = get_dataloader(tokenizer, args)

    # attn_impl = "sdpa" # torch 2.2 sdpa uses flash attn 2
    attn_impl = "flash_attention_2"
    if rank == 0 or args['verbose']:
        print("Creating model", rank)
        
    if args["model_files_dir"] is not None:
        print("Using custom model config")
        cfg = AutoConfig.from_pretrained(os.path.join(args["model_files_dir"], "config.json"))
    else:
        cfg = AutoConfig.from_pretrained(args["model_name"])
    cfg.use_cache = False
    cfg._attn_implementation = attn_impl
    
    ### DEBUG ###
    # cfg.num_hidden_layers = 1
    ### DEBUG END ###
    
    # RoPE scaling.
    if args["scale_rope"] and (args["context_length"] > cfg.max_position_embeddings):
        if args["precision"] != "bf16_buffers_autocast":
            raise Exception(f"Rope scaling will give high loss when casted with long context, we recommend using 'bf16_buffers_autocast'.")
        rope_scaling_factor= args["context_length"] / cfg.max_position_embeddings
        cfg.rope_scaling = {}
        cfg.rope_scaling["type"] = args["rope_type"]
        cfg.rope_scaling["factor"] = rope_scaling_factor
        cfg._rope_scaling_validation()
    if args["rope_type"] != "dynamic":
        cfg.max_position_embeddings = max(cfg.max_position_embeddings, args["context_length"])
    

    # load model on meta device without calling init and replace nn.Linear with Linear4bit
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
        model_type = cfg.model_type
        
        # Attention Implementation.
        attn_cls = None
        if attn_impl == "flash_attention_2" and model_type != "phi3":
            if model_type == "qwen2":
                attn_cls = Qwen2FlashAttention2
            elif model_type == "llama":
                attn_cls = LlamaFlashAttention2
            elif model_type == "mistral":
                attn_cls = MistralFlashAttention2
            elif model_type == "phi3":
                attn_cls = Phi3FlashAttention2 # FIXME: Problem with cu_seq_lens
            model.config._attn_implementation = "flash_attention_2"
            model.config._attn_implementation_internal = "flash_attention_2"
        if model_type == "phi3":
            model.config._attn_implementation = "sdpa"
            model.config._attn_implementation_internal = "sdpa"
            attn_cls = Phi3SdpaAttention
        
        if attn_cls is not None: 	
            for layer in model.model.layers: 
                m = getattr(layer, 'self_attn')
                setattr(layer, 'self_attn', attn_cls(m.config, m.layer_idx))
        
        # Quantization config.
        groupsize_4bit = 128
        groupsize_2bit = 64
        quant_config_4bit = BaseQuantizeConfig(nbits=4, group_size=groupsize_4bit, quant_zero=False,
                                                quant_scale=False, offload_meta=False, view_as_float=True, axis=1)
        quant_config_2bit = BaseQuantizeConfig(nbits=2, group_size=groupsize_2bit, quant_zero=False,
                                                quant_scale=False, offload_meta=False, view_as_float=True, axis=1)
        
        attn_layers = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_layers  = ["gate_proj", "up_proj", "down_proj"]
        if args["nbits"] == "4":
            layers_4bit = attn_layers + mlp_layers
            layers_2bit = []
        elif args["nbits"] == "2":
            layers_4bit = []
            layers_2bit = attn_layers + mlp_layers
        elif args["nbits"] == "mixed":
            layers_4bit = attn_layers
            layers_2bit = mlp_layers
        
        layer_nbits      = {**{layer:4 for layer in layers_4bit}, 
                            **{layer:2 for layer in layers_2bit}}
        layer_groupsizes = {**{layer:groupsize_4bit for layer in layers_4bit},
                            **{layer:groupsize_2bit for layer in layers_2bit}}
            
        skip_modules = ["lm_head"]
        model.model = replace_linear(model=model.model, 
                                        linear_replacement=HQQLinear, 
                                        quant_config_4bit=quant_config_4bit, 
                                        quant_config_2bit=quant_config_2bit,
                                        layers_4bit=layers_4bit, 
                                        layers_2bit=layers_2bit,
                                        device=rank,
                                        compute_dtype=compute_dtype, 
                                        del_orig=True, 
                                        initialize=False, 
                                        skip_modules=skip_modules)
        HQQLinear.set_backend(HQQBackend.PYTORCH_BACKPROP) # needed for axis=1.         
        
    if rank == 0 or args['verbose']:
        print("Replaced model:", model)

    # Pretrained files.
    from pathlib import Path
    if args["model_files_dir"] is not None: # for testing custom model configs.
        files = list(Path(args["model_files_dir"]).glob("*.safetensors"))
    else:        
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

    # Loading and quantization.
    # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
    # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
    def load_and_quantize_parallel(name_param, model, **kwargs):
        name, param = name_param
        load_and_quantize(model, name, param, **kwargs)

    param_count = sum((p.numel() for n,p in model.named_parameters()))
    if rank == 0 or args['verbose']:
        print("Loading model", rank)
        
    if rank == 0 or args['verbose']:
        print(f"Total model params: {param_count}")

    n_workers = 8
    if rank == 0 or args['verbose']:
        print(f"Using n_workers: {n_workers} for loading")

    start = time.time()
    for filename in tqdm(files, desc="Loading & Quantizing Model Shards", disable=rank!=0, position=0):
        weights = safetensors.torch.load_file(filename)
        
        # ### DEBUG ###
        # # remove all other layers but first.
        # weights = {k:v for k,v in weights.items() if ("layers." not in k) or ("layers.0" in k)}
        # if len(weights) == 0:
        #     continue
        # ### DEBUG END ###
        
        parallel(load_and_quantize_parallel, iter(weights.items()), n_workers=n_workers, threadpool=True,
                    model=model, 
                    dtype=torch_dtype, 
                    device=local_rank, 
                    skip_names=load_param_skip_names,
                    to_cpu=(args["low_memory"] and rank==0), 
                    to_meta=(args["low_memory"] and rank!=0),
                    verbose=args["verbose"])
    if rank == 0 and args["verbose"]:
        print(f"Loaded model weights in {time.time()-start:.3f} seconds")
    torch.cuda.empty_cache() # cleanup any extra memory usage from parallel loading.
    
    if rank == 0 or args['verbose']:
        print(f"Rank {rank}: Model created: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")
    
    if rank == 0 or args['verbose']:
        # Create DoRA layers and set trainable params.
        print("Using HQQDORA", rank)
    
    lora_cls = HQQDORA
    for name, _ in model.named_modules():
        module_key, _, value_key = name.rpartition('.')
        if value_key in args['lora_target_modules']:
            m = model.get_submodule(name)
            qlora_layer = lora_cls(m, args["lora_rank"])
            parent_module = model.get_submodule(module_key)
            setattr(parent_module, value_key, qlora_layer)

    for n,p in model.named_parameters():
        if any([lora_name in n for lora_name in ['lora_A', 'lora_B', 'magnitude']]):
            p.requires_grad = True
            if args['verbose'] and rank == 0:
                print("Trainable DoRA layer", n)
        else:
            p.requires_grad = False
    
    if rank == 0 or args['verbose']:
        print(f"Rank {rank}: DoRA layers added: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")

                
    if args["log_to"] == 'wandb':
        logger.log({"memory/allocated_after_model_created": torch.cuda.memory_allocated(local_rank)}, rank)
        logger.log({"memory/reserved_after_model_creation": torch.cuda.memory_reserved(local_rank)}, rank)


    # Wrap model with llama-recipies or custom LoRA policy
    my_auto_wrap_policy = get_wrapping_policy(custom_policy=True, vanilla_policy=False)

    if rank == 0 or args['verbose']:
        print("Wrapping model w/ FSDP", rank)

    if args["sharding_strategy"] == "full_shard":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args["sharding_strategy"] == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args["sharding_strategy"] == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif args["sharding_strategy"] == "hybrid_full_shard":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif args["sharding_strategy"] == "hybrid_shard_grad_op":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError("Invalid FSDP sharding strategy")
    
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True) if args["use_cpu_offload"] else None,
        limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=args["low_memory"],
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if (rank!=0 and args["low_memory"]) else None, # TODO note about meta device and why we need this
        mixed_precision=mp_policy,
    )
    if rank == 0 or args['verbose']:
        print(f"Rank {rank}: Wrapped model: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")
    if args["log_to"] == 'wandb':
        logger.log({"memory/allocated_after_model_wrap": torch.cuda.memory_allocated(local_rank)}, rank)
        logger.log({"memory/reserved_after_model_wrap": torch.cuda.memory_reserved(local_rank)}, rank)

    torch_compiled = False
    # model = torch.compile(model, dynamic=True)
    # torch_compiled = True

    # Synchronize at the start
    dist.barrier()

    # Apply activation checkpointing
    if args["use_gradient_checkpointing"]:
        if args['reentrant_checkpointing']:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT if args['reentrant_checkpointing'] else CheckpointImpl.NO_REENTRANT,

        )

        check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer, Phi3DecoderLayer))
        if rank == 0 or args['verbose']:
            print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if args["use_activation_cpu_offload"]:
        if rank == 0 or args['verbose']:
            print("Applying activation offloading", rank)
        model = offload_wrapper(model)

    if rank == 0 or args['verbose']:
        print("Config:")
        print(cfg)
        print("Model:")
        print(model)
        print("Starting training")


    # Create the optimizer
    optimizer = get_optimizer(model, args)

    # LR scheduler.
    gradient_accumulation_steps = max(1, args['gradient_accumulation_steps'])
    lr_scheduler, num_training_steps = get_lr_scheduler(optimizer, dataloader, gradient_accumulation_steps, args)

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 or args['verbose']:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                print(f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}")


    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if args["precision"] in ["fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if args["precision"] == "fp16_autocast" else None
    scale_grads = scaler is not None


    if rank == 0:
        print("Total Training Steps:", num_training_steps)
        
    # Warm up to compile different sizes.
    model.train()
    if torch_compiled:
        nearest_pad_dims = torch.tensor([128*i for i in range(1,args["context_length"]//128+1)])
        for size in nearest_pad_dims:
            model(torch.zeros((args['batch_size'], size)).to(device=local_rank, dtype=torch.long), 
                    labels=torch.zeros((args['batch_size'], size)).to(device=local_rank, dtype=torch.long), 
                    attention_mask=None)
            
    memory_stats = []
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    init_start_event.record()
    log_loss, log_lr = 0.0, -1
    current_training_step = 0
    # Reset peak memory to track that
    torch.cuda.reset_peak_memory_stats(local_rank)
    for epoch in range(args['num_epochs']):
        update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
        ddp_loss = torch.zeros(2).to(local_rank)
        
        for batch_idx, batch in enumerate(dataloader):
            accumulate_grads = (batch_idx+1) % gradient_accumulation_steps == 0

            print(f"[rank {local_rank}] Batch Size:", batch['input_ids'].size())
            # Prevent gradient syncing until update step if using no_sync option.
            # Documentation states this should only be used on the root FSDP instance
            # We assume this is a one-node setup
            if args['no_sync'] and not accumulate_grads:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            # Start logging memory (first iter) if requested
            if args['profile_memory'] and batch_idx==0 and rank == 0 and epoch == 0:
                torch.cuda.memory._record_memory_history()

            # Log memory usage
            if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                reserved_before_forward = torch.cuda.memory_reserved(local_rank)
                memory_stats.append(f"Rank {rank}: Before forward: {reserved_before_forward/2**30:.2f} GiB")
                if args["log_to"] == 'wandb':
                    logger.log({"memory/allocated_before_forward": torch.cuda.memory_allocated(local_rank)}, rank)
                    logger.log({"memory/reserved_before_forward": reserved_before_forward}, rank)

            # Forward pass
            with sync_context:
                with autocast:
                    output = model(
                        batch['input_ids'].to(local_rank),
                        labels=batch['labels'].to(local_rank),
                        attention_mask=None,
                    )
                    loss = output.loss

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Log memory usage
                if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                    reserved_after_forward = torch.cuda.memory_reserved(local_rank)
                    memory_stats.append(f"Rank {rank}: After forward: {reserved_after_forward/2**30:.2f} GiB")
                    if args["log_to"] == 'wandb':
                        logger.log({"memory/allocated_after_forward": torch.cuda.memory_allocated(local_rank)}, rank)
                        logger.log({"memory/reserved_after_forward": reserved_after_forward}, rank)

                # Backward pass
                if scale_grads:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Record loss
            bs = batch['input_ids'].shape[0]
            ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
            ddp_loss[1] += bs

            # Step the optimizer (w/ gradient accumulation)
            if accumulate_grads:
                if args['apply_gradient_clipping'] and (args['grad_norm'] is not None):
                    model.clip_grad_norm_(args['grad_norm'], norm_type=2.0)
                if scale_grads:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                # avoid overhead when lr is constant.
                if lr_scheduler is not None:
                    lr_scheduler.step()
                progress_bar.update(1)
                current_training_step += 1

            # Log memory usage after backward
            if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                reserved_after_backward = torch.cuda.memory_reserved(local_rank)
                memory_stats.append(f"Rank {rank}: After backward: {reserved_after_backward/2**30:.2f} GiB")
                if args["log_to"] == 'wandb':
                    logger.log({"memory/allocated_after_backward": torch.cuda.memory_allocated(local_rank)}, rank)
                    logger.log({"memory/reserved_after_backward": reserved_after_backward}, rank)

            # Delete the output so more memory frees up before the next forward pass
            output = None
            loss = None

            # Stop logging memory (first iter)
            if args['profile_memory'] and batch_idx==0 and rank == 0 and epoch == 0:
                torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                torch.cuda.memory._record_memory_history(enabled=None) # Stop recording

            # Log loss every gradient update steps
            if accumulate_grads:
                dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                if rank == 0:
                    log_loss = ddp_loss[0] / ddp_loss[1]
                    if lr_scheduler is not None:
                        log_lr = lr_scheduler.get_last_lr()[0]
                    else:
                        log_lr = args["lr"]
                    update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
                    if args["log_to"] == 'wandb':
                        logger.log({"loss": log_loss, "lr": log_lr}, rank)
                ddp_loss = torch.zeros(2).to(local_rank)
                                
            # Save model every_n steps.
            if accumulate_grads and args["save_model"] and (current_training_step % args["save_model_every_n_step"] == 0):
                save_model(rank, model, args, cfg, compute_dtype, layer_nbits, layer_groupsizes, step=current_training_step)

        # Print + log peak memory usage for the whole fourth step of training
        if epoch == 0 and (rank == 0 or args['verbose']):
            peak_allocated_memory = torch.cuda.max_memory_allocated(local_rank)
            peak_reserved_memory  = torch.cuda.max_memory_reserved(local_rank)
            memory_stats.append(f"Rank {rank}: Peak allocated memory: {peak_allocated_memory/2**30:.2f} GiB")
            memory_stats.append(f"Rank {rank}: Peak reserved memory:  {peak_reserved_memory/2**30:.2f} GiB")
            if args["log_to"] == 'wandb':
                logger.log({"memory/allocated_peak": peak_allocated_memory}, rank)
                logger.log({"memory/reserved_peak": peak_reserved_memory}, rank)

    # Synchronize at the end and record time
    init_end_event.record()
    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        print("Finished training", rank)

    # Print time, model, & memory stats
    time_taken = init_start_event.elapsed_time(init_end_event) / 1000
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"CUDA event elapsed time: {time_taken} sec")
        logger.log({"time_taken": time_taken}, rank)
    for line in memory_stats:
        print(line)

    # End logging
    logger.finish(rank=rank)

    # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
    # HQQLinear custom state_dict() method causes issues when saving.
    # Model is saved fine when `state_dict()` method is removed.
    # Non param/buffer types are not saved with FSDP.
    # It might be better to just save the trained lora layers.
    # summon_full_params on lora layers and save.
    if args["save_model"]:
        save_model(rank, model, args, cfg, compute_dtype, layer_nbits, layer_groupsizes, step=None)

    dist.barrier() # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()


# Entry point, using fastcore's call_parse to parse args from command line and then calling fsdp_main
@call_parse()
def main(
    world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
    train_type: Param("", choices=["hqq_dora"]) = "hqq_dora", # "full", "lora", "qlora", or "custom_qlora"
    batch_size: int = 1, # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512, # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1, # How many epochs of training to do
    dataset: Param("") = "alpaca_sample", # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    dataset_samples: int = 512, # Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
    sharding_strategy: Param("", choices=["full_shard", "shard_grad_op", "ddp", "hybrid_full_shard", "hybrid_shard_grad_op"]) = "full_shard", # Sharding strategy for FSDP
    use_gradient_checkpointing: bool_arg = True, # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool_arg = False, # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool_arg = True, # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool_arg = False, # Use FSDP's activation CPU offloading
    low_memory: bool_arg = True, # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool_arg = False, # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: Param("", choices=["fp32", "bf16", "fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]) = "bf16", # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_files_dir: str = None, # Directory containing model files for testing custom model configs
    save_model: bool_arg = False, # Save the resulting model
    save_model_every_n_step: int = 1000, # Save the model every n steps
    nbits: Param("", choices=["2", "4", "mixed"]) = "4", # Number of bits to quantize to
    output_dir: str = "output", # Output directory to save the final model to
    lora_rank: int = 64, # LoRA rank for lora/qlora
    lora_target_modules: Param("", choices=["all", "default"]) = "all", # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool_arg = False, # Whether to print extra info for debugging
    lr: float = 1e-5, # Learning rate
    apply_gradient_clipping: bool_arg = False, # Apply gradient norm clipping
    grad_norm: float = 0.3, # Gradient norm clipping
    wd: float = 0.1, # Weight decay
    scale_rope: bool_arg = False, # Scale the rope if context_length > max_position_embeddings
    rope_type: Param("", choices=["linear", "dynamic"]) = "linear", # Rope scaling type
    profile_memory: bool_arg = False, # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: Param("", choices=["adamw", "adam", "sgd", "adadelta"]) = "adamw", # Optimizer
    lr_scheduler: Param("", choices=["constant", "linear", "cosine"]) = "constant", # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
    loading_workers: int = -1, # Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
    log_to: Param("", choices=["tqdm", "wandb", "stdout"]) = "stdout", # Where to log output
    master_addr: str = "localhost", # For distributed training
    master_port: str = "12355", # For distributed training, must be the same for all processes
    seed: int = 42, # Random seed
    project_name: str = "fsdp_qlora", # For wandb logging
    name: str = None, # For wandb logging
    group: str = None, # For wandb logging
    entity: str = None, # For wandb logging
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
        if "phi-3" in model_name.lower():
            args["lora_target_modules"] = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        else:
            args["lora_target_modules"] = ["q_proj", "k_proj","v_proj", "o_proj", "gate_proj",  "up_proj", "down_proj"]
    elif lora_target_modules.lower() == "default":
        args["lora_target_modules"] = None

    if args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"] and not torch.cuda.is_bf16_supported():
        raise ValueError('Current device does not support bfloat16')

    # Set no_sync if using cpu_offload and gradient accumulation. Turn off if not using gradient accumulation
    if args["use_cpu_offload"] and args["gradient_accumulation_steps"] > 1:
        args["no_sync"] = True
    elif args["no_sync"] and args["gradient_accumulation_steps"] == 1:
        args["no_sync"] = False

    # Run
    mp.spawn(fsdp_main,
        args=(world_size, args),
        nprocs=torch.cuda.device_count(),
        join=True)