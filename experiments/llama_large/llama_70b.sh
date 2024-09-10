# GPU: 4 X A100
# Effective BS: 32

# find max bs with:
# --dataset dummy \
# --dataset_samples 64 \
# --context_length 2048 \

# actual training with:
# --dataset /workspace/data/llama_large_mix_dataset_v0 \
# --dataset /workspace/data/llama_large_mix_dataset_v0_1536

MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
DATASET_NAME=/workspace/data/llama_large_mix_dataset_v1_1536
# DATASET_NAME=orca_math_instruct
# DATASET_NAME=dummy
TARGET_GLOBAL_BS=64
BS=8

# Get NUM_GPUS form system
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
GRAD_ACCUM_STEPS=$((TARGET_GLOBAL_BS / (BS * NUM_GPUS)))

CONTEXT_LENGTH=1536
SAVE_STEPS=125
STOP_STEP=625
SAVE_DIR=/workspace/models
LOG_DIR=/workspace/git/fsdp_qlora/experiments/llama_large/logs

# (compression 22.6%)
LORA_RANK=64
GROUPSIZE_2BIT=32
TRAIN_LAYERNORMS=true
DISC_LR=true
BASE_LR=5e-5
LR_DIV_FACTOR=10


cd /workspace/git/fsdp_qlora && python train.py \
    --train_type hqq_dora \
    --nbits mixed \
    --groupsize_2bit $GROUPSIZE_2BIT \
    --block_influence_layers 0,13,15,17,19,21,23,26,29,31,33,56,59,68,71,79 \
    --lr $BASE_LR \
    --lr_div_factor $LR_DIV_FACTOR \
    --disc_lr $DISC_LR \
    --train_layernorms $TRAIN_LAYERNORMS \
    --lora_rank $LORA_RANK \
    --sharding_strategy full_shard \
    --model_name $MODEL_NAME \
    --dataset $DATASET_NAME \
    --context_length $CONTEXT_LENGTH \
    --batch_size $BS \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --use_cpu_offload false \
    --log_to stdout \
    --verbose true \
    --save_model true \
    --save_model_every_n_step $SAVE_STEPS \
    --stop_training_at_step $STOP_STEP \
    --output_dir $SAVE_DIR/llama-3-1-70b-instruct-dora-4-2bit-gs-$GROUPSIZE_2BIT-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-no-adj-20pct 2>&1 | tee $LOG_DIR/llama_3_1_70b_dora_4_2bit_gs_$GROUPSIZE_2BIT-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-no-adj-20pct.log