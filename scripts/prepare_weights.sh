# llama-3 8b qlora (10k)
python prepare_weights.py \
--infer_type merged_bnb_lora \
--lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-10k-bnb-qlora/model_state_dict.safetensors \
--model_name meta-llama/Meta-Llama-3-8B \
--save_dir /workspace/models/llama-3-8b-orca-math-10k-bnb-qlora-merged

# llama-3 8b qdora (10k)
python prepare_weights.py \
--infer_type merged_bnb_dora \
--lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-10k-bnb-qdora/model_state_dict.safetensors \
--model_name meta-llama/Meta-Llama-3-8B \
--save_dir /workspace/models/llama-3-8b-orca-math-10k-bnb-qdora-merged

# llama-3 8b qlora (100k)
python prepare_weights.py \
--infer_type merged_bnb_lora \
--lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-100k-bnb-qlora/model_state_dict.safetensors \
--model_name meta-llama/Meta-Llama-3-8B \
--save_dir /workspace/models/llama-3-8b-orca-math-100k-bnb-qlora-merged

# llama-3 8b qdora (100k)
python prepare_weights.py \
--infer_type merged_bnb_dora \
--lora_or_dora_filename /workspace/models/llama-3-8b-orca-math-100k-bnb-qdora/model_state_dict.safetensors \
--model_name meta-llama/Meta-Llama-3-8B \
--save_dir /workspace/models/llama-3-8b-orca-math-100k-bnb-qdora-merged

