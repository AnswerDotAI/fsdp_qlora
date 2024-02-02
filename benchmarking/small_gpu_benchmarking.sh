# Run for 1 fwd-bwd step to find the max bs using a 2xA5000 (24GB each) and 16 CPUs with 88GB RAM machine.
# https://github.com/AnswerDotAI/fsdp_qlora/blob/299f51a98246d77f5e556fe1a27ab29e107530f0/train.py
# Uses different default params for train.py script to reduce clutter in the commands below. 

# Notes:
# 1) LORA CPU offloading with model sizes larger than 7B fails, most probably due to limited CPU memory. 
# QLORA CPU offloading works fine.
# 2) CPU offloading appears exteremely slow. Getting the actual run times will be useful. 
# Check PCIe stuff. 

# Fine.
python train.py --batch_size 48 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing True --train_type lora
# Check why bs is very low in qlora vs lora? Activation overhead?
python train.py --batch_size 24 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# How is qlora full shard same as ddp?
python train.py --batch_size 24 --use_ddp True --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# Extremely slow.
python train.py --batch_size 96 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type lora
# Same as before -> Check why bs is very low in qlora vs lora? Activation overhead with large bs?
python train.py --batch_size 30 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type qlora
# Fine.
python train.py --batch_size 4 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing False --train_type lora
# Now that bs drops (or activations) we can use larger bs than lora.
python train.py --batch_size 6 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# Again -> How is qlora full shard bs same as ddp?
python train.py --batch_size 6 --use_ddp True --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# Again -> Extremely slow also much lower bs than grad ckpt. Probably smart to prefer grad ckpt over cpu offload.
python train.py --batch_size 8 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type lora
# Interesting now lora and qlora have same bs when grad ckpt disabled with cpu offloading.
python train.py --batch_size 8 --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type qlora

# Fine.
python train.py --batch_size 22 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing True --train_type lora
# Fine.
python train.py --batch_size 16 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# 13B -> ~13GB for model. DDP works fine.
python train.py --batch_size 15 --use_ddp True --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# FIXME: torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGKILl. Needs more CPU memory than 88GB?
# Reducing batch size to 1 doesn't fix it, how come storing 13b params (~26GB) need more than 88GB?
python train.py --batch_size 1 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type lora
# Qlora cpu offloading works as opposed to lora, likely due to smaller model size after quantization?
python train.py --batch_size 18 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type qlora
# Fine.
python train.py --batch_size 2 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing False --train_type lora
# Fine.
python train.py --batch_size 4 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# Fine.
python train.py --batch_size 3 --use_ddp True --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# FIXME: torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGKILl. Needs more CPU memory than 88GB?
# Reducing batch size to 1 doesn't fix it, how come storing 13b params (~26GB) need more than 88GB?
python train.py --batch_size 1 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type lora
# Qlora cpu offloading works as opposed to lora, likely due to smaller model size after quantization?
python train.py --batch_size 4 --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type qlora


# # Test low memory
# python train.py --batch_size 1 --model_name meta-llama/Llama-2-7b-hf --context_length 16 --use_gradient_checkpointing True --train_type qlora --low_memory True
# # This works now. Custom QLORA nn.module, no changes needed in bnb.
# python train.py --batch_size 1 --model_name meta-llama/Llama-2-70b-hf --context_length 1 --use_gradient_checkpointing True --train_type qlora --low_memory True

# This is theoretically not possible:
python train.py --batch_size 128 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --train_type lora
# Fine.
python train.py --batch_size 6 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# OOM during training.
python train.py --batch_size 1 --use_ddp True --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# FIXME: torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGKILl. Needs more CPU memory than 88GB?
python train.py --batch_size 1 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type lora
# Qlora cpu offloading works as opposed to lora, likely due to smaller model size after quantization?
python train.py --batch_size 10 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type qlora
# Not possible theoretically, requires at least 68gb/2=34gb per gpu.
python train.py --batch_size 1 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type lora
# OOM. There is some extra memory in gpu:0.
python train.py --batch_size 1 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# Fine with custom qlora.
python train.py --batch_size 1 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type custom_qlora
# OOM.
python train.py --batch_size 1 --use_ddp True --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora

python train.py --batch_size 1 --use_ddp True --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type custom_qlora
# Better. Symmetrically distributed memory across gpus.
# python train.py --batch_size 1 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type custom_qlora
# OOM on cpu. low memory needs to be fixed for lora model loading?
python train.py --batch_size 1 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type lora
python train.py --batch_size 2 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type qlora


# Try with 4xA5000 GPUs with 96 CPUs and 500GB RAM. Model fits in mem. but can't train with seqlen=256.
python train.py --batch_size 2 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# OOM on cpu. low memory needs to be fixed for lora model loading?
python train.py --batch_size 1 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type lora
python train.py --batch_size 4 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type qlora
python train.py --batch_size 1 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# OOM on cpu.
# python train.py --batch_size 128 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type lora
python train.py --batch_size 1 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type qlora