# GPU: 4 X A100
# Effective BS: 32

# find max bs with:
# --dataset dummy \
# --dataset_samples 64 \
# --context_length 2048 \

# actual training with:
# --dataset /workspace/data/llama_large_mix_dataset_v0 \
# --dataset /workspace/data/llama_large_mix_dataset_v0_1536

export CUDA_VISIBLE_DEVICES=1,2,3,4,5

HOME_DIR=$HOME
SAVE_DIR=$HOME_DIR/models
LOG_DIR=$HOME_DIR/git/fsdp_qlora/experiments/qwen/logs

# Create log dir if not exists
mkdir -p $LOG_DIR

MODEL_NAME=Qwen/Qwen2.5-32B-Instruct
DATASET_NAME=$HOME_DIR/data/qwen_mix_dataset_v0_1536
# DATASET_NAME=orca_math_instruct
# DATASET_NAME=dummy
TARGET_GLOBAL_BS=64
BS=8

# Get NUM_GPUS form system
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
GRAD_ACCUM_STEPS=$((TARGET_GLOBAL_BS / (BS * NUM_GPUS)))

CONTEXT_LENGTH=1536
SAVE_STEPS=150
STOP_STEP=150


LORA_RANK=64
GROUPSIZE_2BIT=32
TRAIN_LAYERNORMS=true
DISC_LR=true
BASE_LR=5e-5
LR_DIV_FACTOR=10

# Ablation config. (22.8% compression, 17.3 GB model size)
BI_20_PCT=0,2,3,4,5,7,8,32,49,61,62,63

# Jeremy config. (25.2% compression, 18.85 GB model size)
BI_55_PCT=0,1,2,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,62,63


cd $HOME_DIR/git/fsdp_qlora && python train.py \
    --train_type hqq_dora \
    --nbits mixed \
    --groupsize_2bit $GROUPSIZE_2BIT \
    --block_influence_layers $BI_20_PCT \
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
    --log_to wandb \
    --project_name qwen_32b_qdora \
    --name qwen-32b-instruct-dora-4-2bit-block-influence-no-adj-20pct \
    --verbose true \
    --save_model true \
    --save_model_every_n_step $SAVE_STEPS \
    --stop_training_at_step $STOP_STEP \
    --output_dir $SAVE_DIR/qwen-32b-instruct-dora-4-2bit-block-influence-no-adj-20pct 2>&1 | tee $LOG_DIR/qwen-32b-instruct-dora-4-2bit-block-influence-no-adj-20pct.log
