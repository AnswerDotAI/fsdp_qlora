# 4 x A6000 (48GB), 128 CPUs 472GB CPU RAM
python train.py --batch_size 32 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --train_type lora
python train.py --batch_size 16 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
python train.py --batch_size 16 --use_ddp True --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# CPU offloading is not needed for this setup.
python train.py --batch_size 32 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type lora
# CPU offloading is not needed for this setup.
python train.py --batch_size 32 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type qlora
python train.py --batch_size 4 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type lora
python train.py --batch_size 6 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
python train.py --batch_size 4 --use_ddp True --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# Ignore now, slow.
python train.py --batch_size 128 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type lora
# Ignore now, slow.
python train.py --batch_size 128 --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type qlora


python train.py --batch_size 1 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --train_type lora
python train.py --batch_size 10 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
python train.py --batch_size 2 --use_ddp True --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --train_type qlora
# python train.py --batch_size 4 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type lora
# python train.py --batch_size 128 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing True --use_cpu_offload True --train_type qlora
# python train.py --batch_size 128 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --train_type lora
# python train.py --batch_size 128 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# python train.py --batch_size 128 --use_ddp True --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --train_type qlora
# python train.py --batch_size 128 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type lora
# python train.py --batch_size 128 --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_gradient_checkpointing False --use_cpu_offload True --train_type qlora