# python evaluate.py \
# --eval_type full_post_quantized \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-10k-full \
# --save_path /workspace/git/fsdp_qlora/eval_results/10k-full-post-quantize.json

# python evaluate.py \
# --eval_type qlora \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-10k-bnb-qlora \
# --save_path /workspace/git/fsdp_qlora/eval_results/10k-qlora.json

# python evaluate.py \
# --eval_type bnb_dora \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-10k-bnb-qdora \
# --save_path /workspace/git/fsdp_qlora/eval_results/10k-bnb-dora.json

# python evaluate.py \
# --eval_type bnb_llama_pro \
# --model_name meta-llama/Llama-2-7b-hf \
# --llama_pro_path /workspace/quantized-ft-models/meta-llama/Llama-2-7b-hf_blk_exp-32-35/ \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-10k-bnb-llama-pro \
# --save_path /workspace/git/fsdp_qlora/eval_results/10k-bnb-llama-pro.json

# python evaluate.py \
# --eval_type full_post_quantized \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-100k-full \
# --save_path /workspace/git/fsdp_qlora/eval_results/100k-full-post-quantize.json

# python evaluate.py \
# --eval_type qlora \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-100k-bnb-qlora \
# --save_path /workspace/git/fsdp_qlora/eval_results/100k-qlora.json

# python evaluate.py \
# --eval_type bnb_dora \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-100k-bnb-qdora \
# --save_path /workspace/git/fsdp_qlora/eval_results/100k-bnb-dora.json

# python evaluate.py \
# --eval_type bnb_llama_pro \
# --model_name meta-llama/Llama-2-7b-hf \
# --llama_pro_path /workspace/quantized-ft-models/meta-llama/Llama-2-7b-hf_blk_exp-32-35/ \
# --models_dir /workspace/quantized-ft-models/ \
# --trained_model_dir llama-7b-orca-math-100k-bnb-llama-pro \
# --save_path /workspace/git/fsdp_qlora/eval_results/100k-bnb-llama-pro.json