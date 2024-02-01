python train.py \
--model_name codellama/CodeLlama-34b-hf \
--train_type qlora \
--batch_size 6 \
--gradient_accumulation_steps 3 \
--dataset sql \
--save_model True \
--output_dir sql_model_qlora \
--apply_gradient_clipping True \
--project_name fsdp_qlora_sql \
--precision bf16_buffers_autocast \
--log_to wandb

python train.py \
--model_name codellama/CodeLlama-34b-hf \
--train_type custom_qlora \
--batch_size 6 \
--gradient_accumulation_steps 3 \
--dataset sql \
--save_model True \
--output_dir sql_model_custom_qlora \
--apply_gradient_clipping True \
--project_name fsdp_qlora_sql \
--precision bf16_buffers_autocast \
--log_to wandb

runpodctl stop pod gu42bzl1eo65jp