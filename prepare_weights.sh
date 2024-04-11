# Full Post Quant VLLM Inference
python prepare_weights.py \
--infer_type full_post_quant \
--model_weights_dir /workspace/models/llama-7b-orca-math-100k-full \
--model_name meta-llama/Llama-2-7b-hf \
--save_dir /workspace/models/llama-7b-orca-math-100k-full-quantized-test

# BnB DoRA VLLM Inference
python prepare_weights.py \
--infer_type bnb_dora \
--dora_filename /workspace/models/llama-7b-orca-math-100k-bnb-qdora/model_state_dict.safetensors \
--model_name meta-llama/Llama-2-7b-hf \
--save_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-vllm-test

# Merged DoRA VLLM Inference
python prepare_weights.py \
--infer_type merged_bnb_dora \
--dora_filename /workspace/models/llama-7b-orca-math-100k-bnb-qdora/model_state_dict.safetensors \
--model_name meta-llama/Llama-2-7b-hf \
--save_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-merged-test