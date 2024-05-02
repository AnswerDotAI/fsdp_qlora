# Full vs QLORA vs HQQ, batch size = 64

# Full
# max batch size / gpu = 8 (38/40 GB)
# 8 * 2 gpus * 4 grad accum  = 64
export CUDA_VISIBLE_DEVICES=4,5
python train.py \
--world_size 2 \
--master_port 12356 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 4 \
--batch_size 8 \
--context_length 512 \
--precision bf16 \
--train_type full \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to wandb \
--dataset alpaca \
--verbose true

# BnB (QLORA)
# max batch size / gpu = 16 (28/40 GB)
# 16 * 2 gpus * 2 grad accum  = 64
export CUDA_VISIBLE_DEVICES=4,5
python train.py \
--world_size 2 \
--master_port 12356 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 16 \
--context_length 512 \
--precision bf16 \
--train_type custom_qlora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to wandb \
--dataset alpaca \
--verbose true

# HQQ (QLORA)
# max batch size / gpu = 32 (28/40 GB)
# 32 * 2 gpus = 64
export CUDA_VISIBLE_DEVICES=4,5
python train.py \
--world_size 2 \
--master_port 12356 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 1 \
--batch_size 32 \
--context_length 512 \
--precision bf16 \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to wandb \
--dataset alpaca \
--verbose true

# DORA: max batch size / gpu = 32 (28/40 GB)
# 32 * 2 gpus = 64
export CUDA_VISIBLE_DEVICES=6,7
python train.py \
--world_size 2 \
--master_port 12357 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 1 \
--batch_size 32 \
--context_length 512 \
--precision bf16 \
--train_type hqq_dora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset alpaca \
--verbose true


# 32 * 2 gpus = 64
export CUDA_VISIBLE_DEVICES=2,6
python train.py \
--world_size 2 \
--master_port 12356 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 1 \
--batch_size 32 \
--context_length 512 \
--precision bf16 \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset dummy \
--verbose true \
--save_model true \
--output_dir /weka/home-keremturgutlu/models/hqq_lora_dummy

export CUDA_VISIBLE_DEVICES=2,6
python train.py \
--lr 1e-3 \
--world_size 2 \
--master_port 12356 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 1 \
--batch_size 32 \
--context_length 512 \
--precision bf16 \
--train_type custom_qlora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset dummy \
--verbose true \
--save_model true \
--output_dir /weka/home-keremturgutlu/models/qlora_dummy


# BNB 70B
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

# HQQ 70B
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