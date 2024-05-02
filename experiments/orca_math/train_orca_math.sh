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

# 10k qlora instruct
cd /workspace/git/fsdp_qlora && python train.py \
--train_type qlora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--dataset orca_math_instruct \
--dataset_samples 10000 \
--batch_size 4 \
--context_length 2048 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing true \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to wandb \
--verbose true \
--project_name "fsdp-quantized-ft-exps" \
--save_model true \
--output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qlora

# 10k qdora instruct
cd /workspace/git/fsdp_qlora && python train.py \
--train_type bnb_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--dataset orca_math_instruct \
--dataset_samples 10000 \
--batch_size 4 \
--context_length 2048 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing true \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to wandb \
--verbose true \
--project_name "fsdp-quantized-ft-exps" \
--save_model true \
--output_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qdora