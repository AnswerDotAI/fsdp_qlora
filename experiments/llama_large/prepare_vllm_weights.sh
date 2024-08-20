export CUDA_DEVICE_ORDER=PCI_BUS_ID
MODEL_PREFIX=/workspace/models
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct

# # 4bit HQQ+DORA
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-dora-4bit/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type tinygemm \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_tinygemm
# done

# # 4/2bit HQQ+DORA
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-dora-4-2bit/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done

# 4/2bit HQQ+DORA (merged)
for step in 250 1000
do
    MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-dora-4-2bit/step_$step
    python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
    --train_type hqq_dora \
    --infer_type merged \
    --model_name $MODEL_NAME \
    --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
    --config_filename $MODEL_DIR/config.json \
    --save_dir $MODEL_DIR/merged
done

# # 4bit HQQ+DORA+LN
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-dora-4bit-ln/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type tinygemm \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_tinygemm
# done

# # 4bit HQQ+LN
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-4bit-ln/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type tinygemm \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_tinygemm
# done

# # 4/2bit HQQ+DORA+LN
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-dora-4-2bit-ln/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done

# 4/2bit HQQ+DORA+LN (merged)
for step in 250 1000
do
    MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-dora-4-2bit-ln/step_$step
    python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
    --train_type hqq_dora \
    --infer_type merged \
    --model_name $MODEL_NAME \
    --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
    --config_filename $MODEL_DIR/config.json \
    --save_dir $MODEL_DIR/merged
done

# # 4(HQQ)bit 2(HQQ+DORA)bit
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-4-dora-2bit/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done

# # 4(HQQ)bit 2(HQQ+DORA)bit+LN
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-4-dora-2bit-ln/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done

# # 4(HQQ)bit 2(HQQ)bit+LN
# for step in 250 1000
# do
#     MODEL_DIR=$MODEL_PREFIX/llama-3-1-8b-instruct-hqq-4-hqq-2bit-ln/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done