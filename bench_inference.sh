python bench_inference.py \
--infer_type full_post_quant \
--model_dir /workspace/models/llama-7b-orca-math-100k-full-quantized-test

python bench_inference.py \
--infer_type bnb_dora \
--model_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-vllm-test

python bench_inference.py \
--infer_type merged_bnb_dora \
--model_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-merged-test

python bench_inference.py \
--infer_type gptq_marlin \
--model_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-marlin \
--tensor_parallel_size 2