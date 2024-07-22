# 4 X A100

cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits 4 \
--model_name meta-llama/Meta-Llama-3-70B-Instruct \
--dataset /workspace/data/llama_large_mix_dataset_v0 \
--batch_size 1 \
--dataset_samples 8 \
--context_length 2048 \
--gradient_accumulation_steps 4 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--verbose true \
--project_name "fsdp-quantized-large-llama-ft-exps" \
--save_model true \
--output_dir /workspace/models/llama-3-70b-instruct-hqq-4bit 2>&1 | tee large_llama_70b_hqq_4bit.log

cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--model_name meta-llama/Meta-Llama-3-70B-Instruct \
--dataset /workspace/data/llama_large_mix_dataset_v0 \
--batch_size 1 \
--dataset_samples 8 \
--context_length 2048 \
--gradient_accumulation_steps 4 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--verbose true \
--project_name "fsdp-quantized-large-llama-ft-exps" \
--save_model true \
--output_dir /workspace/models/llama-3-70b-instruct-hqq-mixed_bit 2>&1 | tee large_llama_70b_hqq_mixed_bit.log

cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits 2 \
--model_name meta-llama/Meta-Llama-3-70B-Instruct \
--dataset /workspace/data/llama_large_mix_dataset_v0 \
--batch_size 1 \
--dataset_samples 8 \
--context_length 2048 \
--gradient_accumulation_steps 4 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--verbose true \
--project_name "fsdp-quantized-large-llama-ft-exps" \
--save_model true \
--output_dir /workspace/models/llama-3-70b-instruct-hqq-2bit 2>&1 | tee large_llama_70b_hqq_2bit.log
