# 10k qdora instruct
cd ../../ && python train.py \
--train_type hqq_dora \
--loftq_init true \
--model_name meta-llama/Llama-2-7b-hf \
--dataset orca_math \
--dataset_samples 100 \
--batch_size 4 \
--context_length 1024 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing true \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--verbose true \
--project_name "fsdp-quantized-ft-exps" \
--save_model false 