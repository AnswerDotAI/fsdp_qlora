# GPU: 4 X A100
# Effective BS: 32

# find max bs with:
# --dataset dummy \
# --dataset_samples 64 \
# --context_length 2048 \

# actual training with:
# --dataset /workspace/data/llama_large_mix_dataset_v0 \
# --dataset /workspace/data/llama_large_mix_dataset_v0_1536 \

# HQQ-DoRA 4bit training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits 4 \
--model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
--dataset /workspace/data/llama_large_mix_dataset_v0_1536 \
--context_length 1536 \
--batch_size 4 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--save_model true \
--save_model_every_n_step 250 \
--output_dir /workspace/models/llama-3-1-70b-instruct-hqq-4bit 2>&1 | tee /workspace/git/fsdp_qlora/experiments/llama_large/logs/llama_3_1_70b_hqq_4bit.log

# HQQ-DoRA 4/2bit training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
--dataset /workspace/data/llama_large_mix_dataset_v0_1536 \
--context_length 1536 \
--batch_size 4 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--save_model true \
--save_model_every_n_step 250 \
--output_dir /workspace/models/llama-3-1-70b-instruct-hqq-mixed-bit 2>&1 | tee /workspace/git/fsdp_qlora/experiments/llama_large/logs/llama_3_1_70b_hqq_mixed_bit.log

# HQQ-DoRA 2bit training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits 2 \
--model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
--dataset /workspace/data/llama_large_mix_dataset_v0_1536 \
--context_length 1536 \
--batch_size 4 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing false \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to stdout \
--save_model true \
--save_model_every_n_step 250 \
--output_dir /workspace/models/llama-3-1-70b-instruct-hqq-2bit 2>&1 | tee /workspace/git/fsdp_qlora/experiments/llama_large/logs/llama_3_1_70b_hqq_2bit.log

