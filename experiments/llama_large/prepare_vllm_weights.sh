export CUDA_DEVICE_ORDER=PCI_BUS_ID
# SAVE_DIR=/workspace/models
# MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct

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
# TRAIN_LAYERNORMS=true
# DISC_LR=true
# for LORA_RANK in 64 256 512; do
#     for BASE_LR in 1e-4 5e-5 2e-5; do
#         for LR_DIV_FACTOR in 10 3 1; do
#             for step in 125 250; do
#                 MODEL_DIR=$SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS/step_$step
#                 python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#                 --train_type hqq_dora \
#                 --infer_type bitblas \
#                 --model_name $MODEL_NAME \
#                 --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#                 --config_filename $MODEL_DIR/config.json \
#                 --save_dir $MODEL_DIR/vllm_bitblas
#             done
#         done
#     done
# done

# for step in 125 250 375 500
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done

# for step in 125 250 375 500
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-loftq/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done

# for step in 125 250 375 500
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-loftq-block-influence/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done


# for step in 125 #125 250 375 500 625 750
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas 2>&1 | tee logs/prepare_vllm_weights.log
# done

# for step in 125 250 375 500 625 750
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged 2>&1 | tee logs/prepare_vllm_weights.log
# done

# for step in 125 250 375 500
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type bitblas \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/vllm_bitblas
# done

# for step in 125 250 375 500
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done


# MODEL_DIR=/workspace/models/llama-3-1-8b-dora-ablations/no_adj_20_pct_step_750
# python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
# --train_type hqq_dora \
# --infer_type merged \
# --model_name $MODEL_NAME \
# --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
# --config_filename $MODEL_DIR/config.json \
# --save_dir $MODEL_DIR/merged

# python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
# --train_type hqq_dora \
# --infer_type bitblas \
# --model_name $MODEL_NAME \
# --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
# --config_filename $MODEL_DIR/config.json \
# --save_dir $MODEL_DIR/vllm_bitblas

# MODEL_DIR=/workspace/models/llama-3-1-8b-dora-ablations/adj_20_pct_step500
# python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
# --train_type hqq_dora \
# --infer_type merged \
# --model_name $MODEL_NAME \
# --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
# --config_filename $MODEL_DIR/config.json \
# --save_dir $MODEL_DIR/merged
# python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
# --train_type hqq_dora \
# --infer_type bitblas \
# --model_name $MODEL_NAME \
# --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
# --config_filename $MODEL_DIR/config.json \
# --save_dir $MODEL_DIR/vllm_bitblas



# for step in 1000
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-32-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done


# for step in 125 250 375 500
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done

# for step in 125 250 375 500
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-loftq-block-influence/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done


# for step in 125 250 375 500 625 750 875 1000
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-32-lora_rank-128-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done

# for step in 125 250 375 500 625 750 875 1000
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-32-lora_rank-64-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done

# for step in 875 1000 #125 250 375 500 625 750 875 1000
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-64-lora_rank-128-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done


# for step in 125 250 375 500 625 750 875 1000
# do
#     MODEL_DIR=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-64-lora_rank-64-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-40pct/step_$step
#     python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
#     --train_type hqq_dora \
#     --infer_type merged \
#     --model_name $MODEL_NAME \
#     --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
#     --config_filename $MODEL_DIR/config.json \
#     --save_dir $MODEL_DIR/merged
# done

# MODEL_DIR=/workspace/data/models/Llama-3.1-70B-Instruct-4-2-Bit-BI-20pct
# MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
# python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
# --train_type hqq_dora \
# --infer_type bitblas \
# --bitblas_dtype bfloat16 \
# --model_name $MODEL_NAME \
# --dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
# --config_filename $MODEL_DIR/config.json \
# --save_dir $MODEL_DIR/vllm_bitblas_bf16

MODEL_DIR=/workspace/data/models/Llama-3.1-70B-Instruct-4-2-Bit-BI-20pct
MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
python /workspace/git/fsdp_qlora/scripts/prepare_vllm_weights.py \
--train_type hqq_dora \
--infer_type gemlite \
--model_name $MODEL_NAME \
--dora_safetensors_filename $MODEL_DIR/model_state_dict.safetensors \
--config_filename $MODEL_DIR/config.json \
--save_dir $MODEL_DIR/vllm_gemlite_fp16