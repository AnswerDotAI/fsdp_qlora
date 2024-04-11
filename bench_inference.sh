# loop over variable tp (1,2,4)
for tp in 1 2 4
do
    python bench_inference.py \
    --infer_type full_post_quant \
    --model_dir /workspace/models/llama-7b-orca-math-100k-full-quantized-test \
    --tensor_parallel_size $tp \
    --save_file /workspace/git/fsdp_qlora/infer_results/llama-7b-orca-math-100k-full-quantized-test-tp$tp.json

    python bench_inference.py \
    --infer_type bnb_dora \
    --model_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-vllm-test \
    --tensor_parallel_size $tp \
    --save_file /workspace/git/fsdp_qlora/infer_results/llama-7b-orca-math-100k-bnb-qdora-vllm-test-tp$tp.json

    python bench_inference.py \
    --infer_type merged_bnb_dora \
    --model_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-merged-test \
    --tensor_parallel_size $tp \
    --save_file /workspace/git/fsdp_qlora/infer_results/llama-7b-orca-math-100k-bnb-qdora-merged-test-tp$tp.json

    python bench_inference.py \
    --infer_type gptq_marlin \
    --model_dir /workspace/models/llama-7b-orca-math-100k-bnb-qdora-marlin \
    --tensor_parallel_size $tp \
    --save_file /workspace/git/fsdp_qlora/infer_results/llama-7b-orca-math-100k-bnb-qdora-marlin-tp$tp.json
done