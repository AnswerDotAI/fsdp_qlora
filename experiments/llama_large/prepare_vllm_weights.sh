export CUDA_DEVICE_ORDER=PCI_BUS_ID
MODEL_PREFIX=/workspace/models/Meta-Llama-3-1-70B-Instruct
MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct

for step in 250 1000 1750 2750
do
    MODEL_DIR=$MODEL_PREFIX-4bit-DoRA/step_$step
    python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
    --train_type hqq_dora \
    --infer_type tinygemm \
    --model_name $MODEL_NAME \
    --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
    --config_filename $MODEL_DIR/config.json \
    --save_dir $MODEL_DIR/vllm_tinygemm
done

for step in 250 1000 1750 2750
do
    MODEL_DIR=$MODEL_PREFIX-4-2bit-DoRA/step_$step
    python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
    --train_type hqq_dora \
    --infer_type bitblas \
    --model_name $MODEL_NAME \
    --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
    --config_filename $MODEL_DIR/config.json \
    --save_dir $MODEL_DIR/vllm_bitblas
done

for step in 250 1000 1750 2750
do
    MODEL_DIR=$MODEL_PREFIX-2bit-DoRA/step_$step
    python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
    --train_type hqq_dora \
    --infer_type bitblas \
    --model_name $MODEL_NAME \
    --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
    --config_filename $MODEL_DIR/config.json \
    --save_dir $MODEL_DIR/vllm_bitblas
done