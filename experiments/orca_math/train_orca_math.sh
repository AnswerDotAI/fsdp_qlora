# # 10k full
# python train.py \
# --train_type full \
# --model_name meta-llama/Meta-Llama-3-8B \
# --dataset orca_math \
# --dataset_samples 10000 \
# --batch_size 1 \
# --context_length 2048 \
# --gradient_accumulation_steps 4 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload true \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-orca-math-10k-full

# # 10k qlora
# python train.py \
# --train_type qlora \
# --model_name meta-llama/Meta-Llama-3-8B \
# --dataset orca_math \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-orca-math-10k-bnb-qlora

# # 10k qdora
# python train.py \
# --train_type bnb_dora \
# --model_name meta-llama/Meta-Llama-3-8B \
# --dataset orca_math \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-orca-math-10k-bnb-qdora

# # 100k full
# python train.py \
# --train_type full \
# --model_name meta-llama/Meta-Llama-3-8B \
# --dataset orca_math \
# --dataset_samples 100000 \
# --batch_size 2 \
# --context_length 2048 \
# --gradient_accumulation_steps 4 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing false \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-orca-math-100k-full

# # 100k qlora
# python train.py \
# --train_type qlora \
# --model_name meta-llama/Meta-Llama-3-8B \
# --dataset orca_math \
# --dataset_samples 100000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-orca-math-100k-bnb-qlora

# # 100k qdora
# python train.py \
# --train_type bnb_dora \
# --model_name meta-llama/Meta-Llama-3-8B \
# --dataset orca_math \
# --dataset_samples 100000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-orca-math-100k-bnb-qdora

# # 10k qlora instruct
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type qlora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qlora

# # 10k qdora instruct
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type bnb_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qdora

# # 10k qdora+ instruct, lora_plus_lambda=16 (too high)
# cd /workspace/git/fsdp_qlora && python train.py \
# --lora_plus_lambda 8 \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus

# # 10k qdora (loftq init) instruct
# cd /workspace/git/fsdp_qlora && python train.py \
# --loftq_init true \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-loftq-init

# # 10k qdora+ (loftq init) instruct
# cd /workspace/git/fsdp_qlora && python train.py \
# --lora_plus_lambda 8 \
# --loftq_init true \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus-loftq-init


# # 10k qlora+ instruct, lora_plus_lambda=16 (too high)
# cd /workspace/git/fsdp_qlora && python train.py \
# --lora_plus_lambda 8 \
# --train_type hqq_lora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus

# # 10k qlora (loftq init) instruct
# cd /workspace/git/fsdp_qlora && python train.py \
# --loftq_init true \
# --train_type hqq_lora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-loftq-init

# # 10k qlora+ (loftq init) instruct
# cd /workspace/git/fsdp_qlora && python train.py \
# --lora_plus_lambda 8 \
# --loftq_init true \
# --train_type hqq_lora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus-loftq-init

# # 10k qdora instruct (hqq axis=1 for torchao kernel compat)
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset orca_math_instruct \
# --dataset_samples 10000 \
# --batch_size 4 \
# --context_length 2048 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing true \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to wandb \
# --verbose true \
# --project_name "fsdp-quantized-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-axis-1

# 10k qlora instruct
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_lora \
--nbits 4 \
--groupsize_4bit 64 \
--train_layernorms true \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--dataset orca_math_instruct \
--dataset_samples 10000 \
--batch_size 4 \
--context_length 2048 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--verbose true \
--project_name "fsdp-quantized-ft-exps" \
--save_model true \
--save_model_every_n_step 10 \
--stop_training_at_step 11 \
--output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-hqq-lora-ln