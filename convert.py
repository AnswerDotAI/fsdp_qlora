import torch
import warnings
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import LlamaForCausalLM, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

warnings.filterwarnings("ignore")


model_name_or_path = "meta-llama/Meta-Llama-3-70B"
out_folder_path = "L3-70B-custom-v4-merged" 
lora_checkpoint_path = "lora_adapters-custom-v3" 
fsdp_save_path = "l3-70-b-custom-V4/model_state_dict.safetensors"


device = "cpu"


#CONVERT

print("Open FSDP save")

from safetensors import safe_open

tensors = {}
with safe_open(fsdp_save_path, framework="pt" ) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k) # loads the full tensor given a key
        # print(k, tensors[k].dtype, tensors[k].shape) # Uncomment to view

for k in tensors:
    if 'lora' not in k: tensors[k] = None


print("Init BNB q4")


# Make sure the compute type, target modules, rank, alpha etc match!
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("Load model q4")

model = LlamaForCausalLM.from_pretrained(
    model_name_or_path,
    use_cache=False,
    quantization_config=bnb_config,
    device_map=device
)

print("Init peft")
# Freeze
for param in model.parameters():
    param.requires_grad = False

# Add LoRA (make sure your rank (r) and alpha (lora_alpha) values match those used in training!)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=512, lora_alpha=256, lora_dropout=0.1,
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
)
model = get_peft_model(model, peft_config)

print("Check keys")

# Check out the first few keys in the state dict:
print(list(model.state_dict().keys())[:10])

new_sd = model.state_dict()
for k in new_sd:
    if 'lora' in k:
        new_sd[k] = tensors[k]

print("Load state dict")
model.load_state_dict(new_sd)

print("Save lora checkpoint")
model.save_pretrained(lora_checkpoint_path)

print("Start merge")
#MERGE

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    # model_max_length=512,
)

print("Load model")

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, return_dict=True, torch_dtype=torch.float16, device_map=device
)

print("Load peft")
peft_model = PeftModel.from_pretrained(model, lora_checkpoint_path)
peft_model = peft_model.merge_and_unload()

print("Models loaded.")

peft_model.save_pretrained(out_folder_path)
tokenizer.save_pretrained(out_folder_path)

print("Models saved.")

