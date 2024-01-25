# Compare LORA and QLORA on Alpaca dataset with same effective batch size ~32, lr sched, and lr.
# Reference for some hyperparams: https://arxiv.org/abs/2305.14314
# LORA (pure bf16)
# https://wandb.ai/answerdotai/fsdp/runs/gb34o6p4?workspace=user-k-answer-ai
# NOTE: Loss curve is flat - 1) use lower lr ? 2) start immediate annealing get_cosine_one_cycle_scheduler(..., min_lr_fraction=0.0)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/lora_alpaca

# QLORA (pure bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/qlora_alpaca

# QLORA (autocast bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--mixed_precision_mode autocast_bf16 \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/qlora_alpaca_autocast_bf16

# stop instance
# requires: az login --use-device-code
az vm deallocate -g resource-group-us-east -n a100-duo















