# HQQ DoRA large LLama3 models.

# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --dataset /workspace/data/hqq_plus_dataset \
# --batch_size 4 \
# --context_length 1024 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing false \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --project_name "fsdp-quantized-large-llama-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee train.log


# # test config A
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-A \
# --dataset dummy \
# --batch_size 1 \
# --context_length 1024 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing false \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --project_name "fsdp-quantized-large-llama-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configA.log

# # # test config B
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-B \
# --dataset dummy \
# --batch_size 1 \
# --context_length 1024 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing false \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --project_name "fsdp-quantized-large-llama-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configB.log

# # # test config C
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-C \
# --dataset dummy \
# --batch_size 1 \
# --context_length 1024 \
# --gradient_accumulation_steps 2 \
# --sharding_strategy full_shard \
# --use_gradient_checkpointing true \
# --reentrant_checkpointing false \
# --use_cpu_offload false \
# --use_activation_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --project_name "fsdp-quantized-large-llama-ft-exps" \
# --save_model true \
# --output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configC.log


# test config A
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-A \
--dataset dummy \
--batch_size 1 \
--dataset_samples 8 \
--context_length 2048 \
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
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configA-2048.log

# # test config B
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-B \
--dataset dummy \
--batch_size 1 \
--dataset_samples 8 \
--context_length 2048 \
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
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configB-2048.log

# # test config C
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-C \
--dataset dummy \
--batch_size 1 \
--dataset_samples 8 \
--context_length 2048 \
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
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configC-2048.log



# test config A
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-A \
--dataset dummy \
--batch_size 1 \
--dataset_samples 8 \
--context_length 4096 \
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
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configA-4096.log

# # test config B
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-B \
--dataset dummy \
--batch_size 1 \
--dataset_samples 8 \
--context_length 4096 \
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
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configB-4096.log

# # test config C
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--model_files_dir /workspace/models/meta-llama/Meta-Llama-3-400B-Instruct-C \
--dataset dummy \
--batch_size 1 \
--dataset_samples 8 \
--context_length 4096 \
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
--output_dir /workspace/models/llama-3-8b-instruct-hqq-dora-plus-plus 2>&1 | tee large_llama_configC-4096.log