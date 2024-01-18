# Compare LORA and QLORA on Alpaca dataset with same effective batch size ~32, lr sched, and lr.

# LORA (pure bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--lr 1e-5 \
--lr_scheduler cosine \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca

# QLORA (pure bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--lr 1e-5 \
--lr_scheduler cosine \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca_sample

# QLORA (autocast bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--mixed_precision_mode autocast_bf16 \
--lr 1e-5 \
--lr_scheduler cosine \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca_sample


















