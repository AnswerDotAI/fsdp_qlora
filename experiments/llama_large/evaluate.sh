for step in 125 250 375 500
do
    MODEL_NAME=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence/step_$step/merged
    python evaluate.py --model_name $MODEL_NAME
done

for step in 125 250 375 500
do
    MODEL_NAME=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-lora_rank-256-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-loftq-block-influence/step_$step/merged
    python evaluate.py --model_name $MODEL_NAME
done

for step in 125 250 375 500 625 750 875 1000
do
    MODEL_NAME=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-32-lora_rank-128-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step/merged
    python evaluate.py --model_name $MODEL_NAME
done

for step in 125 250 375 500 625 750 875 1000
do
    MODEL_NAME=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-32-lora_rank-64-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step/merged
    python evaluate.py --model_name $MODEL_NAME
done

for step in 125 250 375 500 625 750 875 1000
do
    MODEL_NAME=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-64-lora_rank-128-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-20pct/step_$step/merged
    python evaluate.py --model_name $MODEL_NAME
done

for step in 125 250 375 500 625 750 875 1000
do
    MODEL_NAME=/workspace/models/llama-3-1-8b-instruct-dora-4-2bit-gs-64-lora_rank-64-base_lr-5e-5-lr_div_factor-10-train_layernorms-true-block-influence-no-adj-40pct/step_$step/merged
    python evaluate.py --model_name $MODEL_NAME
done