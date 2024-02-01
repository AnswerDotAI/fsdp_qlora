from datasets import load_dataset
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

local_rank = os.getenv("LOCAL_RANK")
device_string = "cuda:" + str(local_rank)

# Load the dataset
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")


# Load the model + tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache = False,
    device_map={'':device_string}
)

# PEFT config
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
)


# Args 
max_seq_length = 512
output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "adamw_hf"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 311 # Approx the size of guanaco at bs 8, ga 2, 2 GPUs. 
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=False, # Otherwise weird loss pattern (see https://github.com/artidoro/qlora/issues/84#issuecomment-1572408347, https://github.com/artidoro/qlora/issues/228, https://wandb.ai/answerdotai/fsdp_qlora/runs/snhj0eyh)
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False}, # Needed for DDP
    report_to="wandb",
)

# Trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Not sure if needed but noticed this in https://colab.research.google.com/drive/1t3exfAVLQo4oKIopQT1SKxK4UcYg7rC1#scrollTo=7OyIvEx7b1GT
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train :)
trainer.train()