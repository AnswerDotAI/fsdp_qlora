python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
--train_type hqq_dora \
--infer_type tinygemm \
--dora_safetensors_filename /workspace/models/Meta-Llama-3-70B-Instruct-4bit-DoRA/step_500/model_state_dict.safetensors \
--config_filename /workspace/models/Meta-Llama-3-70B-Instruct-4bit-DoRA/step_500/config.json \
--model_name meta-llama/Meta-Llama-3-70B-Instruct \
--save_dir /workspace/models/Meta-Llama-3-70B-Instruct-4bit-DoRA/step_500/vllm_tinygemm
