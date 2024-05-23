# This generates uses export_stacks to generate profiling output
# /tmp/profile_0.txt, /tmp/profile_1.txt, etc. (1 file per process.)
#
# Output files are generated using export_stacks(), note there are some
# outstanding issues to be aware of:
# https://github.com/pytorch/pytorch/issues/100253
#
# Profiling output files can be used with speedscope or other tools.
#
# For additional information, see:
# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

# python train.py \
#     --model_name meta-llama/Meta-Llama-3-8B \
#     --train_type hqq_dora \
#     --n_bits 4 \
#     --precision bf16 \
#     --dataset orca_math \
#     --dataset_samples 8 \
#     --batch_size 2 \
#     --context_length 512 \
#     --gradient_accumulation_steps 2 \
#     --use_gradient_checkpointing False \
#     --use_cpu_offload False \
#     --use_activation_cpu_offload False \
#     --save_model False \
#     --profiling_output /tmp/profile
# export CUDA_VISIBLE_DEVICES=3,4
python train.py \
--world_size -1 \
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
--profile false \
--with_stack true \
--verbose true

