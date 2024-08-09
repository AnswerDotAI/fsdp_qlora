python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
--train_type hqq_dora \
--infer_type tinygemm \
--model_name meta-llama/Meta-Llama-3.1-405B-Instruct \
--dora_safetensors_filename /workspace/models/Meta-Llama-3-1-405B-Instruct-4bit-DoRA/step_250/model_state_dict.safetensors \
--config_filename /workspace/models/Meta-Llama-3-1-405B-Instruct-4bit-DoRA/step_250/config.json \
--save_dir /workspace/models/Meta-Llama-3-1-405B-Instruct-4bit-DoRA/vllm_tinygemm
