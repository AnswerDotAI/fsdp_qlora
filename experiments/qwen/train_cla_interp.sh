#! /bin/bash

# CLA(2) adjacent - layer_idx : kv_cache_idx
CLA2_ADJ='{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10, 23: 10, 24: 10, 25: 10, 26: 10, 27: 10, 28: 10, 29: 10, 30: 11, 31: 12, 32: 13, 33: 14, 34: 15, 35: 16, 36: 17, 37: 17, 38: 17, 39: 17, 40: 17, 41: 17, 42: 17, 43: 17, 44: 18, 45: 19, 46: 20, 47: 21, 48: 22, 49: 23, 50: 24, 51: 25, 52: 26, 53: 27, 54: 27, 55: 27, 56: 27, 57: 27, 58: 27, 59: 27, 60: 28, 61: 29, 62: 30, 63: 31}'

# Get the number of available GPUs using nvidia-smi
gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Generate sequence from 0 to (gpu_count-1)
if [ "$gpu_count" -gt 0 ]; then
    # Create comma-separated list of GPU indices
    gpu_list=$(seq -s ',' 0 $((gpu_count-1)))
    
    # Export the environment variable
    export CUDA_VISIBLE_DEVICES=$gpu_list
    
    echo "Found $gpu_count GPUs"
    echo "Set CUDA_VISIBLE_DEVICES=$gpu_list"
else
    echo "No GPUs found"
    exit 1
fi

# Home is /workspace if exists, otherwise ~
if [ -d "/workspace" ]; then
    HOME=/workspace
else
    HOME=~
fi

# Define the stages and their corresponding steps
MODEL_SIZE=0.5
OUTPUT_DIR=qwen_cla_fp8kv_cla2_adj_full_finetune_interp
STEP=300

cd $HOME/fsdp_qlora && python train.py \
--world_size 8 \
--master_port 12356 \
--model_name Qwen/Qwen2.5-${MODEL_SIZE}B-Instruct \
--cla_kv_cache_map "$(echo $CLA2_ADJ)" \
--cla_full_finetune true \
--fp8_kv_enabled true \
--train_type full \
--sharding_strategy full_shard \
--precision bf16 \
--gradient_accumulation_steps 2 \
--batch_size 2 \
--context_length 1024 \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to wandb \
--project_name qwen_cla_fp8kv \
--group cla2_adj_full_finetune_interp \
--name cla2_adj_full_finetune_interp_step_${STEP} \
--dataset answerdotai/qwen_large_mix_dataset_v0_dedup_1024 \
--verbose true \
--low_memory true \
--save_model true \
--output_dir $HOME/models/$OUTPUT_DIR \
--save_model_every_n_step $STEP \
--stop_training_at_step $STEP