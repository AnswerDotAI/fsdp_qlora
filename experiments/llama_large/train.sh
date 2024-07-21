# HQQ DoRA large LLama3 models.

cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-70B-Instruct \
--dataset /workspace/data/hqq_plus_dataset \
--batch_size 4 \
--context_length 1024 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--verbose true \
--project_name "fsdp-quantized-large-llama-ft-exps" \
--save_model true \
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus