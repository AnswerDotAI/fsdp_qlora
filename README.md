# fsdp_qlora
Training LLMs with QLoRA + FSDP

This is still a work-in-progress, but to start experimenting:

- Install the llama-recipes requirements (we can test what's actually needed later)
- Install our bnb version
- Run with defaults: `python train.py`!

## Testing from a fresh jarvislabs/runpod instance:
These instructions have also been tested with Cuda 11.7 & 12.1.

- Clone https://github.com/AnswerDotAI/fsdp_qlora
- `pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes`
- `pip uninstall bitsandbytes`
- Clone AnswerDotAI/bitsandbytes & switch to `cuda_fix_quant_storage_dtype` branch `git clone -b cuda_fix_quant_storage_dtype https://github.com/AnswerDotAI/bitsandbytes`
- in bitsandbytes folder, `make CUDA_VERSION=118` then `python setup.py install` (may need export BNB_CUDA_VERSION=118 and to set cuda path)
- pip install fastcore wandb
- huggingface-cli login (to access Llama 2 7B)
- back in fsdp_qlora folder, run `python train.py` to test lora train

Check out different combos of settings. For example,
`python train.py --train_type qlora` to do qlora instead of the default lora

NBs:

- The low-memory option (loading on only one shard and using sync_module_states) works with LoRA but not QLoRA yet. 
- size-based wrapping policy gave an error but I think it should work, will update if I get it going.
- the undocumented train_type hf_qlora loads with the transformers load_in_4bit option, which by default gives a ValueError: Cannot flatten integer dtype tensors (since quant_storage default is uint8). If you manually edit the def of Linear4Bit to set the default to bf16 this should work (with size wrapping policy, the other one fails because something is fp16) and is a useful comparison for exploring what options we have if we want to avoid all the custom model loading stuff. Hmm after a fresh install this isn't working with the size-based wrapping either, "AssertionError: Expects storage to be allocated", I need to debug this.


## Mixed Precision Training

### `--precision bf16` (pure bfloat16)

This will cast all the model parameters to `torch.bfloat16` before training and won't use FSDP mixed precision. As a result, sharded and unsharded params will be stored in bf16, forward and backward passes will be done in bf16, and gradient reduction and updates will be done in bf16.

### `--precision fp32` (pure float32)

This will cast all the model parameters to `torch.float32` before training and won't use FSDP mixed precision. As a result, sharded and unsharded params will be stored in fp32, forward and backward passes will be done in fp32, and gradient reduction and updates will be done in fp32.


### `--precision mp_fp16_autocast` (mixed float16 with autocast)

This will cast all the model parameters to `torch.float32` before training and will use FSDP mixed precision with 

```
mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
```

As a results, sharded and unsharded params will be stored in fp32. It will use `autocast(torch.float16)` for forward and backward passes, and `autocast(torch.float16)` for gradient reduction and updates.


### `--precision mp_bf16_autocast` (mixed bfloat16 with autocast)

This will cast all the model parameters to `torch.float32` before training and will use FSDP mixed precision with 

```
mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
```

As a results, sharded and unsharded params will be stored in fp32. It will use `autocast(torch.bfloat16)` for forward and backward passes, and `autocast(torch.bfloat16)` for gradient reduction and updates.


### `--precision mp_bf16_buffers_autocast` (bfloat16 params and float32 buffers with autocast)

This will cast all the model parameters to `torch.bfloat16` before training but will keep the buffers in `torch.float32` and will use FSDP mixed precision with 

```
mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
```

As a results, sharded and unsharded params will be stored in bf16. It will use `autocast(torch.bfloat16)` for forward and backward passes, and `autocast(torch.bfloat16)` for gradient reduction and updates. Buffers and only [eligible operations](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16) in autocast will be performed in bf16.
