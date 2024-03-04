# fsdp_qlora

Training LLMs with QLoRA + FSDP. This is a demo script, and still a work in progress. See the blog post (coming soon) for more information.

## Installation:

The following steps should work (tested on cuda 11.7, 11.8 and 12.1):
- Clone https://github.com/AnswerDotAI/fsdp_qlora
- `pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes` as an easy way to get most dependencies
- `pip uninstall bitsandbytes` since for now it must be installed from source to have the necessary changes
- Clone bitsandbytes (recent version has our changes) or clone AnswerDotAI/bitsandbytes & switch to `cuda_fix_quant_storage_dtype` branch `git clone -b cuda_fix_quant_storage_dtype https://github.com/AnswerDotAI/bitsandbytes`
- in bitsandbytes folder, `make CUDA_VERSION=118` then `python setup.py install` (may need export BNB_CUDA_VERSION=118 and to set cuda path, adjust to your preferred version)
- `pip install fastcore wandb` as extra requirements above those in llama-recipes
- huggingface-cli login (to access Llama 2 7B)
- back in fsdp_qlora folder, run `python train.py` to test qlora training
- HQQ installation [instructions](https://github.com/mobiusml/hqq?tab=readme-ov-file#installation). Our training script uses `HQQBackend.ATEN_BACKPROP`, so also make sure to build the custom kernels `cd hqq/kernels && python setup_cuda.py install`.
- [Pytorch >= 2.2](https://pytorch.org/blog/pytorch2-2/) is recommended to make use of the native flash-attention 2 kernel.

## Training Options

### `--train_type full`

Full params fine-tuning.

```bash
export CUDA_VISIBLE_DEVICES=4,5 # optionally set devices
python train.py \
--world_size 2 \ # optional, on a single machine will be set automatically
--master_port 12356 \ # optional, defaults to 12355
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 4 \
--batch_size 8 \
--context_length 512 \
--precision bf16 \
--train_type full \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to wandb \
--dataset alpaca \
--verbose true
```

### `--train_type lora`

LoRA fine-tuning using HF Peft library.

```diff
- --train_type full \
+ --train_type lora \
```

### `--train_type custom_lora`

LoRA fine-tuning using a custom LoRA module.

```diff
- --train_type full \
+ --train_type custom_lora \
```

### `--train_type qlora`

4-bit quantized LoRA fine-tuning using bitsanbytes Linear4bit layer with NF4 quantization and HF Peft library.

```diff
- --train_type full \
+ --train_type qlora \
```

### `--train_type custom_qlora`

4-bit quantized LoRA fine-tuning using bitsanbytes Linear4bit layer with NF4 quantization and a custom LoRA module.

```diff
- --train_type full \
+ --train_type custom_qlora \
```

### `--train_type hqq_lora`

4-bit quantized LoRA fine-tuning using HQQ library and a custom LoRA module.

```diff
- --train_type full \
+ --train_type hqq_lora \
```

## Low Memory Loading

During quantized LoRA training we use a custom quantization and loading code to avoid loading the entire model into GPU memory before sharding it across GPUs. This is the default behavior of our training script when any of the following training options `"qlora", "custom_qlora", "hqq_lora", "hqq_dora"` is used. Other training options are already optimized for low memory loading to their best extent.

We load the weights iteratively, quantize them on the GPU and place them back to CPU or meta device (based on their rank) concurrently N parameters at a time. We do this across all GPUs to initialize the quantization parameters, such as zero and scale, while using `sync_module_states=True` to sync the model weights across all GPUs during FSDP initialization.

**Note:** With HQQ LoRA model loading unexpectedly uses much more memory on rank-0, so we force `n_workers=2`, for example Llama 70B model takes ~22 mins to load and quantize with HQQ.

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

This option is important for RoPE layer which gives incorrect results when cast to lower precision especially with longer context lengths.

## Comparinson to an existing trainer
![Screenshot 2024-02-01 083222](https://github.com/AnswerDotAI/fsdp_qlora/assets/6575163/97bb03fb-c2bb-4679-83ff-63a2e202826f)
`hf_train.py` uses TRL's SFTTrainer for a comparison run. To match with our script, modify the dataloading code to train on everything (not just completions) and then run `train.py --train_type qlora --dataset guanaco --batch_size 8 --lr_scheduler cosine --log_to wandb --save_model True --output_dir guanaco_7B --gradient_accumulation_steps 2 --lr 2e-4`. The SFTTrainer version has to run with a lower batch size (4 vs 8) so we only do 2 gradient accumulation steps vs 4 in the QLoRA+FSDP version. 

## Converting Saved Models

If you specify `--save_model True` the adapter layers will be saved as a state dict. To convert to the regular huggingface format and upload to the hub, see: **Converting the State Dict.ipynb**

If `"custom_qlora", "hqq_lora"` training options are used, then only the trainable LoRA parameters will be saved. Before inference, you need to load and quantize the base model again, and seperately load the saved LoRA parameters. 

You can alternatively test to see if merging base model weights and trained LoRA weights and then quantizing them performs similar to keeping the parameters seperately as done during training. To make use of `torch.compile` with HQQ, see https://github.com/mobiusml/hqq/issues/18.

## Limitations

While QLoRA finetuning works with FSDP, there are some rough edges to be aware of with this alpha release and our example script.

First, the current release of Transformer `AutoModel.from_pretrained` cannot be used to load models into quantized weights, as it does not support the new quant_storage or quantization flag. Loading pretrained models requires writing or using custom model loading code. We provide an example of how to load and quantize a QLoRA model for finetuning in our demo script.
We are actively working with Hugging Face to resolve this incompatibility in future Transformers and PEFT releases.

Second, quantized models must be entirely loaded into all GPUs before FSDP can shard them across GPUs. We mitigate this issue in our example script by loading the model weights layer by layer and quantizing them and placing weights to cpu on rank-0 before loading the next layer. This means the maximum model size which QLoRA and FSDP can finetune on a single 24GB card is currently 34 billion parameters as a 70B model would require ~35GB memory after 4-bit quantization.

~~Third, QLoRA doesn’t support FSDP’s and llama-recipes’ low cpu memory usage option via `sync_module_states`. This option allows FSDP to only load one instance of the model, potentially split evenly across GPUs, and then shard the model weights across GPUs for training. We are actively researching fixes for this issue and hope to resolve it in a future update.~~

Third, while FSDP’s Mixed Precision works with QLoRA, practitioners need to be careful to set the `MixedPrecision.param_type` to match the `Linear4Bit.quant_storage` dtype. Otherwise, FSDP’s Mixed Precision could cast the quantized weights to a different precision, essentially turning them into random weights. Our example script shows how to avoid this potential pitfall, and we will be happy to assist model training libraries in correctly exposing FSDP’s Mixed Precision options to users when training with QLoRA


## Example: Llama 70B 4-A100 40GB Training

```bash
# BnB QLoRA
export CUDA_VISIBLE_DEVICES=4,5,6,7
python train.py \
--world_size 4 \
--master_port 12356 \
--model_name meta-llama/Llama-2-70b-hf \
--gradient_accumulation_steps 4 \
--batch_size 2 \
--context_length 512 \
--precision bf16_buffers_autocast \
--train_type custom_qlora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset alpaca \
--verbose true

# HQQ QLoRA
export CUDA_VISIBLE_DEVICES=4,5,6,7
python train.py \
--world_size 4 \
--master_port 12356 \
--model_name meta-llama/Llama-2-70b-hf \
--gradient_accumulation_steps 4 \
--batch_size 2 \
--context_length 512 \
--precision bf16_buffers_autocast \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset alpaca \
--verbose true
```

**Note:** For large batch size or long context training HQQ LoRA is more memory efficient compared to BnB LoRA. So if you are running into OOM issues, try using HQQ LoRA.


## SLURM Training

See `fsdp_multi_node.sh` for an example training script using multi-node training with SLURM.