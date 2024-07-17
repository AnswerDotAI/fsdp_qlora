# HQQ++ dataset Dora (hqq axis=1 for torchao kernel compat)
# quant_zero=False, quant_scale=False, offload_meta=False
# 4 x 24 GB GPUs

cd /workspace/git/fsdp_qlora && python train.py \
--train_type full \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
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
--project_name "fsdp-quantized-ft-exps" \
--save_model true \
--use_cla true \
--cla_factor 2 \
--output_dir /workspace/models/llama-3-8b-instruct-hqq-plus-dataset-CLA-2


cd /workspace/git/fsdp_qlora && python train.py \
--train_type full \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
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
--project_name "fsdp-quantized-ft-exps" \
--save_model true \
--use_cla true \
--cla_factor 3 \
--output_dir /workspace/models/llama-3-8b-instruct-hqq-plus-dataset-CLA-3
