# GPU: 4 X A100
# Effective BS: 32

# find max bs with:
# --dataset dummy \
# --dataset_samples 64 \
# --context_length 1536 \

# actual training with:
# --dataset /workspace/data/llama_large_mix_dataset_v0 \

MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
DATASET_NAME=/workspace/data/llama_large_mix_dataset_v1_1536
# DATASET_NAME=orca_math_instruct
# DATASET_NAME=dummy
BS=16
CONTEXT_LENGTH=1536
SAVE_STEPS=125
STOP_STEP=750
SAVE_DIR=/workspace/models
LOG_DIR=/workspace/git/fsdp_qlora/experiments/llama_large/logs

# LORA_RANK=64 # 64,256,512
# BASE_LR=5e-4 1e-4 5e-5
# LR_DIV_FACTOR=10 3 1
# TRAIN_LAYERNORMS

# HQQ-DoRA 4/2bit training.

# ABLATIONS 1: Tune LORA_RANK, BASE_LR, LR_DIV_FACTOR
TRAIN_LAYERNORMS=true
DISC_LR=true

# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --lr 1e-4 \
# --lr_div_factor 10 \
# --disc_lr $DISC_LR \
# --train_layernorms $TRAIN_LAYERNORMS \
# --lora_rank 512 \
# --sharding_strategy full_shard \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size $BS \
# --gradient_accumulation_steps 1 \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS.log

# for LORA_RANK in 64 256 512; do
#     for BASE_LR in 1e-4 5e-5 2e-5; do
#         for LR_DIV_FACTOR in 10 3 1; do
#             echo "LORA_RANK: $LORA_RANK, BASE_LR: $BASE_LR, LR_DIV_FACTOR: $LR_DIV_FACTOR"
#             cd /workspace/git/fsdp_qlora && python train.py \
#             --train_type hqq_dora \
#             --nbits mixed \
#             --lr $BASE_LR \
#             --lr_div_factor $LR_DIV_FACTOR \
#             --disc_lr $DISC_LR \
#             --train_layernorms $TRAIN_LAYERNORMS \
#             --lora_rank $LORA_RANK \
#             --sharding_strategy full_shard \
#             --model_name $MODEL_NAME \
#             --dataset $DATASET_NAME \
#             --context_length $CONTEXT_LENGTH \
#             --batch_size $BS \
#             --gradient_accumulation_steps 1 \
#             --use_cpu_offload false \
#             --log_to stdout \
#             --verbose true \
#             --save_model true \
#             --save_model_every_n_step $SAVE_STEPS \
#             --stop_training_at_step $STOP_STEP \
#             --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS.log
#         done
#     done
# done

LORA_RANK=256
BASE_LR=5e-5
LR_DIV_FACTOR=10

# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --block_influence_layers 0,2,7,9,11,31 \
# --lr $BASE_LR \
# --lr_div_factor $LR_DIV_FACTOR \
# --disc_lr $DISC_LR \
# --train_layernorms $TRAIN_LAYERNORMS \
# --lora_rank $LORA_RANK \
# --sharding_strategy full_shard \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size $BS \
# --gradient_accumulation_steps 1 \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence.log


# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --loftq_init true \
# --lr $BASE_LR \
# --lr_div_factor $LR_DIV_FACTOR \
# --disc_lr $DISC_LR \
# --train_layernorms $TRAIN_LAYERNORMS \
# --lora_rank $LORA_RANK \
# --sharding_strategy full_shard \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size $BS \
# --gradient_accumulation_steps 1 \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-loftq 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-loftq.log


# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --loftq_init true \
# --block_influence_layers 0,2,7,9,11,31 \
# --lr $BASE_LR \
# --lr_div_factor $LR_DIV_FACTOR \
# --disc_lr $DISC_LR \
# --train_layernorms $TRAIN_LAYERNORMS \
# --lora_rank $LORA_RANK \
# --sharding_strategy full_shard \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size $BS \
# --gradient_accumulation_steps 1 \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-loftq-block-influence 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-loftq-block-influence.log


# Block Influence (No Adj. top 20%) + 4bit (128) 2bit (32)
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--groupsize_2bit 32 \
--block_influence_layers 0,2,7,9,11,31 \
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
--gradient_accumulation_steps 1 \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-no-adj-20pct 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-no-adj-20pct.log

# # Block Influence (Adj. top 20%) + 4bit (128) 2bit (32)
# # NOTE: NaNs
# cd /workspace/git/fsdp_qlora && python train.py \
# --resume_from_dora_weights /workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-adj-20pct/step_500/model_state_dict.safetensors \
# --resume_from_optimizer /workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-adj-20pct/step_500/optimizer.bin \
# --train_type hqq_dora \
# --nbits mixed \
# --groupsize_2bit 32 \
# --block_influence_layers 0,1,2,7,11,31 \
# --lr $BASE_LR \
# --lr_div_factor $LR_DIV_FACTOR \
# --disc_lr $DISC_LR \
# --train_layernorms $TRAIN_LAYERNORMS \
# --lora_rank $LORA_RANK \
# --sharding_strategy full_shard \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size $BS \
# --gradient_accumulation_steps 1 \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-adj-20pct 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-adj-20pct.log


# # Block Influence (No Adj. top 30%) + 4bit (128) 2bit (32)
# # NOTE: CUDA illegal memory access
# cd /workspace/git/fsdp_qlora && python train.py \
# --resume_from_dora_weights /workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-30pct/step_250/model_state_dict.safetensors \
# --resume_from_optimizer /workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-30pct/step_250/optimizer.bin \
# --train_type hqq_dora \
# --nbits mixed \
# --groupsize_2bit 32 \
# --block_influence_layers 0,2,4,7,9,11,14,22,31 \
# --lr $BASE_LR \
# --lr_div_factor $LR_DIV_FACTOR \
# --disc_lr $DISC_LR \
# --train_layernorms $TRAIN_LAYERNORMS \
# --lora_rank $LORA_RANK \
# --sharding_strategy full_shard \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size $BS \
# --gradient_accumulation_steps 1 \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-no-adj-30pct 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-no-adj-30pct.log

# # Block Influence (Adj. top 30%) + 4bit (128) 2bit (32)
# # NOTE: CUDA illegal memory access
# cd /workspace/git/fsdp_qlora && python train.py \
# --resume_from_dora_weights /workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-30pct/step_125/model_state_dict.safetensors \
# --resume_from_optimizer /workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-30pct/step_125/optimizer.bin \
# --train_type hqq_dora \
# --nbits mixed \
# --groupsize_2bit 32 \
# --block_influence_layers 0,1,2,7,8,9,10,11,31 \
# --lr $BASE_LR \
# --lr_div_factor $LR_DIV_FACTOR \
# --disc_lr $DISC_LR \
# --train_layernorms $TRAIN_LAYERNORMS \
# --lora_rank $LORA_RANK \
# --sharding_strategy full_shard \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size $BS \
# --gradient_accumulation_steps 1 \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-adj-30pct 2>&1 | tee $LOG_DIR/llama_3_1_8b_dora_4_2bit_lora_rank-$LORA_RANK-base_lr-$BASE_LR-lr_div_factor-$LR_DIV_FACTOR-train_layernorms-$TRAIN_LAYERNORMS-block-influence-adj-30pct.log



# # stop azure vm named llama-inference
# az vm deallocate --resource-group resource-group-us-central --name llama-training


# kill all processes such 
# ps aux | grep workers | awk '{print $2}' | xargs kill -9

# ABLATIONS 2: Tune LAYERNORM



# ABLATIONS 3: BLOCK INFLUENCE






















# # HQQ-DoRA 4bit + LN training.
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits 4 \
# --train_layernorms true \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size 8 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-dora-4bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_dora_4bit_ln.log


# # HQQ 4bit (HQQ) + LN training.
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits 4 \
# --skip_dora_all true \
# --train_layernorms true \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size 8 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4bit_ln.log


# # HQQ-DoRA 4/2bit + LN training.
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --train_layernorms true \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size 8 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-dora-4-2bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_dora_4_2bit_ln.log


# # HQQ-4 (HQQ) / 2 (DORA) bit training.
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --skip_dora_4bit true \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size 8 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4-dora-2bit 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4_dora_2bit.log


# # HQQ-4 (HQQ) / 2 (DORA) bit + LN training.
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --skip_dora_4bit true \
# --train_layernorms true \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size 8 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4-dora-2bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4_dora_2bit_ln.log

# # HQQ-4 (HQQ) / 2 (HQQ) bit + LN training.
# cd /workspace/git/fsdp_qlora && python train.py \
# --train_type hqq_dora \
# --nbits mixed \
# --skip_dora_all true \
# --train_layernorms true \
# --model_name $MODEL_NAME \
# --dataset $DATASET_NAME \
# --context_length $CONTEXT_LENGTH \
# --batch_size 8 \
# --gradient_accumulation_steps 1 \
# --sharding_strategy full_shard \
# --use_cpu_offload false \
# --log_to stdout \
# --verbose true \
# --save_model true \
# --save_model_every_n_step $SAVE_STEPS \
# --stop_training_at_step $STOP_STEP \
# --output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4-hqq-2bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4_hqq_2bit_ln.log




