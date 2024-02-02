# run script also show stdout and save to log file
python train.py \
--context_length 256 \
--model_name codellama/CodeLlama-34b-hf \
--train_type qlora \
--batch_size 4 \
--gradient_accumulation_steps 4 \
--dataset sql \
--save_model True \
--output_dir sql_model_qlora \
--apply_gradient_clipping True \
--project_name fsdp_qlora_sql \
--precision bf16_buffers_autocast \
--log_to wandb 2>&1 | tee ~/qlora_sql.log

python train.py \
--context_length 256 \
--model_name codellama/CodeLlama-34b-hf \
--train_type custom_qlora \
--batch_size 4 \
--gradient_accumulation_steps 4 \
--dataset sql \
--save_model True \
--output_dir sql_model_custom_qlora \
--apply_gradient_clipping True \
--project_name fsdp_qlora_sql \
--precision bf16_buffers_autocast \
--log_to wandb  2>&1 | tee ~/custom_qlora_sql.log