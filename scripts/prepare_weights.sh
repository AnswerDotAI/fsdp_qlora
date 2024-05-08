# # llama-3 8b qlora (10k)
# python prepare_weights.py \
# --infer_type merged_bnb_lora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-10k-bnb-qlora/model_state_dict.safetensors \
# --model_name meta-llama/Meta-Llama-3-8B \
# --save_dir /workspace/models/llama-3-8b-orca-math-10k-bnb-qlora-merged

# # llama-3 8b qdora (10k)
# python prepare_weights.py \
# --infer_type merged_bnb_dora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-10k-bnb-qdora/model_state_dict.safetensors \
# --model_name meta-llama/Meta-Llama-3-8B \
# --save_dir /workspace/models/llama-3-8b-orca-math-10k-bnb-qdora-merged

# # llama-3 8b qlora (100k)
# python prepare_weights.py \
# --infer_type merged_bnb_lora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-100k-bnb-qlora/model_state_dict.safetensors \
# --model_name meta-llama/Meta-Llama-3-8B \
# --save_dir /workspace/models/llama-3-8b-orca-math-100k-bnb-qlora-merged

# # llama-3 8b qdora (100k)
# python prepare_weights.py \
# --infer_type merged_bnb_dora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-100k-bnb-qdora/model_state_dict.safetensors \
# --model_name meta-llama/Meta-Llama-3-8B \
# --save_dir /workspace/models/llama-3-8b-orca-math-100k-bnb-qdora-merged

# # llama-3 8b qlora (10k)
# python prepare_weights.py \
# --infer_type merged_bnb_lora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qlora/model_state_dict.safetensors \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qlora-merged

# # llama-3 8b qdora (10k)
# python prepare_weights.py \
# --infer_type merged_bnb_dora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qdora/model_state_dict.safetensors \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-bnb-qdora-merged

# # llama-3 8b qdora plus (10k)
# python prepare_weights.py \
# --infer_type merged_hqq_dora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus/model_state_dict.safetensors \
# --config_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus/config.json \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus-merged

# # llama-3 8b qdora loftq init (10k)
# python prepare_weights.py \
# --infer_type merged_hqq_dora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-loftq-init/model_state_dict.safetensors \
# --loftq_init_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-loftq-init/hqq_loftq_init_weights \
# --config_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-loftq-init/config.json \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-loftq-init-merged

# # llama-3 8b qdora plus loftq init (10k)
# python prepare_weights.py \
# --infer_type merged_hqq_dora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus-loftq-init/model_state_dict.safetensors \
# --loftq_init_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus-loftq-init/hqq_loftq_init_weights \
# --config_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus-loftq-init/config.json \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qdora-plus-loftq-init-merged



# # llama-3 8b qlora plus (10k)
# python prepare_weights.py \
# --infer_type merged_hqq_lora \
# --lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus/model_state_dict.safetensors \
# --config_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus/config.json \
# --model_name meta-llama/Meta-Llama-3-8B-Instruct \
# --save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus-merged

# llama-3 8b qlora loftq init (10k)
python prepare_weights.py \
--infer_type merged_hqq_lora \
--lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-loftq-init/model_state_dict.safetensors \
--loftq_init_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-loftq-init/hqq_loftq_init_weights \
--config_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-loftq-init/config.json \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-loftq-init-merged

# llama-3 8b qlora plus loftq init (10k)
python prepare_weights.py \
--infer_type merged_hqq_lora \
--lora_or_dora_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus-loftq-init/model_state_dict.safetensors \
--loftq_init_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus-loftq-init/hqq_loftq_init_weights \
--config_filename /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus-loftq-init/config.json \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--save_dir /workspace/models/llama-3-8b-instruct-orca-math-10k-hqq-qlora-plus-loftq-init-merged

