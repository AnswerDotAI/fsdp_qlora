python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
--train_type hqq_dora \
--infer_type tinygemm \
--dora_safetensors_filename /workspace/models/llama-3-1-405b-instruct-hqq-4bit/step_250/model_state_dict.safetensors \
--config_filename /workspace/models/llama-3-1-405b-instruct-hqq-4bit/step_250/config.json \
--model_name meta-llama/Meta-Llama-3.1-405B-Instruct \
--save_dir /workspace/models/llama-3-1-405b-instruct-hqq-4bit/step_250/vllm_tinygemm
