export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1

# Eval with HQQ DoRA tinygemm. (groupsize 64)
python /workspace/git/kerem_research/evaluation_benchmarking/eval.py \
--model_path /workspace/models/llama-3-1-405b-instruct-hqq-4bit/step_250/vllm_tinygemm \
--tokenizer_path meta-llama/Meta-Llama-3.1-405B-Instruct \
--num_gpus 4 \
--max_model_len 8192 \
--max_num_seqs 16 \
--quantization torchao \
--dtype bfloat16 2>&1 | tee logs/eval_llama_405b_qdora_tinygemm.log

# # NOTE: Remove "lora_rank" in quantize_config.json
# # Eval with HQQ tinygemm. (groupsize 64)
# python eval.py \
# --model_path /workspace/models/llama-3-1-405b-instruct-hqq-4bit/step_250/vllm_tinygemm \
# --tokenizer_path meta-llama/Meta-Llama-3.1-405B-Instruct \
# --num_gpus 4 \
# --max_model_len 8192 \
# --max_num_seqs 16 \
# --quantization torchao \
# --dtype bfloat16 2>&1 | tee logs/eval_llama_405b_tinygemm.log

# # Eval with BitBlas 4/2 bit HQQ DoRA. (groupsize 128/64)


# # Eval with BitBlas 2 bit HQQ. (groupsize 64)
