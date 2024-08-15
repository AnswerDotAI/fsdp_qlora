export CUDA_DEVICE_ORDER=PCI_BUS_ID

# loop over steps
for step in 1000 1750 2750
do
    MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-hqq-4bit/step_$step
    python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
    --train_type hqq_dora \
    --infer_type tinygemm \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
    --config_filename $MODEL_DIR/config.json \
    --save_dir $MODEL_DIR/vllm_tinygemm
done