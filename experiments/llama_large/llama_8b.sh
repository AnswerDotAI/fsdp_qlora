# GPU: 4 X A100
# Effective BS: 32

# find max bs with:
# --dataset dummy \
# --dataset_samples 64 \
# --context_length 1536 \

# actual training with:
# --dataset /workspace/data/llama_large_mix_dataset_v0 \

MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
DATASET_NAME=/workspace/data/llama_large_mix_dataset_v0_1536
CONTEXT_LENGTH=1536
SAVE_STEPS=250
STOP_STEP=1000
SAVE_DIR=/workspace/models
LOG_DIR=/workspace/git/fsdp_qlora/experiments/llama_large/logs

# HQQ-DoRA 4bit training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits 4 \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-dora-4bit 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_dora_4bit.log

# HQQ-DoRA 4/2bit training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-dora-4-2bit 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_dora_4_2bit.log


# HQQ-DoRA 4bit + LN training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits 4 \
--train_layernorms true \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-dora-4bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_dora_4bit_ln.log


# HQQ 4bit + LN training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits 4 \
--skip_dora_all true \
--train_layernorms true \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4bit_ln.log


# HQQ-DoRA 4/2bit + LN training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--train_layernorms true \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-dora-4-2bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_dora_4_2bit_ln.log


# HQQ-4 (HQQ) / 2 (DORA) bit training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--skip_dora_4bit true \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4-dora-2bit 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4_dora_2bit.log


# HQQ-4 (HQQ) / 2 (DORA) bit + LN training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--skip_dora_4bit true \
--train_layernorms true \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4-dora-2bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4_dora_2bit_ln.log

# HQQ-4 (HQQ) / 2 (HQQ) bit + LN training.
cd /workspace/git/fsdp_qlora && python train.py \
--train_type hqq_dora \
--nbits mixed \
--skip_dora_all true \
--train_layernorms true \
--model_name $MODEL_NAME \
--dataset $DATASET_NAME \
--context_length $CONTEXT_LENGTH \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--sharding_strategy full_shard \
--use_cpu_offload false \
--log_to stdout \
--verbose true \
--save_model true \
--save_model_every_n_step $SAVE_STEPS \
--stop_training_at_step $STOP_STEP \
--output_dir $SAVE_DIR/llama-3-1-8b-instruct-hqq-4-hqq-2bit-ln 2>&1 | tee $LOG_DIR/llama_3_1_8b_hqq_4_hqq_2bit_ln.log




