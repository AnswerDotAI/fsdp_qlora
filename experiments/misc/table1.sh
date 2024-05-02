#!/bin/bash

# Continue on error
set +e

# List of commands
commands=(
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 512 --num_epochs 1 --train_type qlora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type qlora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 2048 --num_epochs 1 --train_type qlora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 4096 --num_epochs 1 --train_type qlora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 512 --num_epochs 1 --train_type lora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type lora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 2048 --num_epochs 1 --train_type lora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 4096 --num_epochs 1 --train_type lora --use_gradient_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type lora --use_gradient_checkpointing False --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 4096 --num_epochs 1 --train_type lora --use_gradient_checkpointing False --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type qlora --use_gradient_checkpointing False --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 4096 --num_epochs 1 --train_type qlora --use_gradient_checkpointing False --use_cpu_offload False --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type lora --use_gradient_checkpointing False --use_cpu_offload True --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type lora --use_gradient_checkpointing True --use_cpu_offload True --log_to wandb --dataset dummy"
     "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type qlora --use_gradient_checkpointing False --use_cpu_offload True --log_to wandb --dataset dummy"
    "python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 1024 --num_epochs 1 --train_type qlora --use_gradient_checkpointing True --use_cpu_offload True --log_to wandb --dataset dummy"
)

# Execute each command
for cmd in "${commands[@]}"; do
    echo "Executing: $cmd"
    $cmd
done

# Optional: stop on error for subsequent commands
set -e