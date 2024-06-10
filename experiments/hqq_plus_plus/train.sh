# HQQ++ dataset Dora (hqq axis=1 for torchao kernel compat)
# quant_zero=False, quant_scale=False, offload_meta=False
# 2 x 48 GB GPUs
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--dataset /workspace/data/hqq_plus_dataset \
--batch_size 8 \
--context_length 2048 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to wandb \
--verbose true \
--project_name "fsdp-quantized-ft-exps" \
--save_model true \
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus