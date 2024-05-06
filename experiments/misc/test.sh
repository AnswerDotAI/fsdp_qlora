# 10k qdora instruct
python train.py \
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