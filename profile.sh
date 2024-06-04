#See PROFILING.md for documentation

# Run profiler contiguously on a 5-step cycle: 4 warmup steps and 1 active (recording) step.
python train.py \
--model_name "hf-internal-testing/tiny-random-LlamaForCausalLM" \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 256 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type qlora \
--use_gradient_checkpointing false \
--use_cpu_offload false \
--log_to stdout \
--dataset dummy \
--profile true \
--export_trace true \
--export_memory_timeline false \
--with_stack true \
--max_steps 20 \
--repeat 0 \
--warmup_steps 4 \
--active_steps 1 \
--profiling_frequency 5 \
--profiling_output llama-test

# Run for 1 cycle then stop profiling
# python train.py \
#   --model_name "hf-internal-testing/tiny-random-LlamaForCausalLM" \
#   --gradient_accumulation_steps 2 \
#   --batch_size 1 \
#   --context_length 256 \
#   --num_epochs 1 \
#   --sharding_strategy full_shard \
#   --precision bf16 \
#   --train_type qlora \
#   --use_gradient_checkpointing false \
#   --use_cpu_offload false \
#   --log_to stdout \
#   --dataset dummy \
#   --profile true \
#   --export_trace true \
#   --export_memory_timeline true \
#   --with_stack true \
#   --num_epochs 1 \
#   --max_steps 20 \
#   --repeat 1 \
#   --warmup_steps 1 \
#   --active_steps 4 \
#   --profiling_output llama-test2