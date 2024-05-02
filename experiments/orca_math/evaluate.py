"""
Evaluation script for full-post-quantization, qlora, qdora and quantized-llama-pro using
BnB 4bit NF4 quantization.
"""

import safetensors
from safetensors.torch import save_file
from pathlib import Path
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
import copy,os,sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from fastcore.parallel import parallel
from fastcore.script import call_parse, bool_arg, Param
from tqdm import tqdm
import time
from datasets import load_dataset
import json
from peft import get_peft_model, LoraConfig, TaskType

from eval_utils import extract_last_number_or_ratio, replace_linear, load_and_quantize_parallel, Linear4bit, init_empty_weights
from dora import BNBDORA

@call_parse()
def main(
    eval_type: Param("", choices=["full_post_quantized", "qlora", "bnb_dora", "bnb_llama_pro"]) = "full_post_quantized", # "full", "lora", "qlora", or "custom_qlora"
    model_name: str = "meta-llama/Llama-2-7b-hf", 
    llama_pro_path: str = "/weka/home-keremturgutlu/models/meta-llama/Llama-2-7b-hf_blk_exp-32-35/",
    models_dir: str = "/weka/home-keremturgutlu/models/",
    trained_model_dir: str = "llama-7b-orca-math-10k-full",
    save_path: str = "/weka/home-keremturgutlu/git/fsdp_qlora/eval_results/10k-full-post-quantize.json",
    bs: int = 16
):

    args = dict(locals())
    
    save_dir = Path(args['save_path']).parent
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = load_dataset("microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=42)
    # train with 10k for starters. Then 100k.
    # dataset = dataset.select(range(0,100000))
    # select last 5k as validation
    dataset = dataset.select(range(len(dataset)-5000,len(dataset)))
    short_answers_gt = parallel(extract_last_number_or_ratio, dataset['answer'], progress=True)

    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    skip_modules = ["lm_head"]
    load_param_skip_names = ['inv_freq']
    cfg = AutoConfig.from_pretrained(args['model_name'])
    cfg._attn_implementation = "sdpa"
    compute_dtype = torch_dtype = torch.bfloat16
    
    if args['eval_type'] == "full_post_quantized":
        pretrained_files = glob(str(Path(args['models_dir'])/args['trained_model_dir']/"*.safetensors"))[:1]
    elif args['eval_type'] == "bnb_llama_pro":
        pretrained_files = glob(str(Path(args['llama_pro_path'])/"*.safetensors"))
    else:
        idx = hub.cached_file(args['model_name'], SAFE_WEIGHTS_INDEX_NAME)
        pretrained_files, _ = hub.get_checkpoint_shard_files(args['model_name'], idx)
        
    if args['eval_type'] in ["qlora", "bnb_dora", "bnb_llama_pro"]:
        trained_weights = safetensors.torch.load_file(glob(str(Path(args['models_dir'])/args['trained_model_dir']/"*.safetensors"))[0])
        
    if args['eval_type'] == "bnb_llama_pro":
        llama_pro_path = Path(args['llama_pro_path'])
        num_original_layers, num_expanded_layers = llama_pro_path.name.split("blk_exp-")[1].split("-")
        num_original_layers, num_expanded_layers = int(num_original_layers), int(num_expanded_layers)
        total_new_layers = num_expanded_layers - num_original_layers
        split = int(num_original_layers / (num_expanded_layers - num_original_layers))
        new_layer_ids = [split+(split+1)*n for n in range(total_new_layers)]
        new_layer_names = [f"layers.{i}" for i in new_layer_ids]
        skip_modules += [str(lid) for lid in new_layer_ids]
        cfg.num_hidden_layers = num_expanded_layers
        

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
        model.model = replace_linear(model.model, Linear4bit, compute_dtype=compute_dtype,
                                    quant_type='nf4', quant_storage=torch_dtype, skip_modules=skip_modules)
    model.is_loaded_in_4bit = True

    for filename in pretrained_files:
        weights = safetensors.torch.load_file(filename)
        parallel(load_and_quantize_parallel, 
                iter(weights.items()), 
                n_workers=8, 
                threadpool=True,
                model=model, 
                dtype=torch_dtype, 
                device=torch.cuda.current_device(),
                skip_names=load_param_skip_names,
                is_meta_rank=False,
                verbose=True,
                quant_method="bnb",
                is_dora=args['eval_type']=="bnb_dora")
        
    if args['eval_type'] in ["qlora"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, peft_config)
        for n,p in model.named_parameters():
            if "lora" in n: 
                print("Loading trained params:", n)
                p.data.copy_(trained_weights[n])
                
    if args['eval_type'] in ["bnb_dora"]:
        # Create LORA layers.
        lora_target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
        lora_rank=64
        lora_alpha=16
        lora_dropout=0.1
        lora_cls = BNBDORA 

        for name, _ in model.named_modules():
            module_key, _, value_key = name.rpartition('.')
            if value_key in lora_target_modules:
                m = model.get_submodule(name)
                qlora_layer = lora_cls(m, lora_rank, lora_alpha, lora_dropout)
                parent_module = model.get_submodule(module_key)
                setattr(parent_module, value_key, qlora_layer)
        
        for n,p in model.named_parameters():
            if ("dora_layer" in n) or ("magnitude_layer" in n): 
                print("Loading trained params:", n)
                p.data.copy_(trained_weights[n])
        
    if args['eval_type'] in ["bnb_llama_pro"]:
        for n,p in model.named_parameters():
            if n in trained_weights: 
                print("Loading trained params:", n)
                p.data.copy_(trained_weights[n])

    try:           
        model.eval().cuda()
    except:
        import pdb; pdb.set_trace()

    # limit to 500 for now.
    valid_dataset = dataset.select(range(500))
    tokenizer.pad_token_id = tokenizer.eos_token_id

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()
    
    answers_pred = []
    short_answers_pred = []
    bs = args['bs']
    for i in tqdm(range(0,len(valid_dataset),bs)):
        
        inputs = [f"###Question:\n{question}\n###Answer:\n" for question in valid_dataset[i:i+bs]['question']]
        input_ids = tokenizer(inputs)['input_ids']
        
        max_toks = max(len(toks) for toks in input_ids)
        b = torch.stack([torch.tensor(((max_toks-len(toks))*[tokenizer.unk_token_id])+toks) for toks in input_ids])
        input_lens = [len(toks) for toks in input_ids]
        
        output = model.generate(b.cuda(), 
                                do_sample=False, 
                                use_cache=True,
                                pad_token_id=tokenizer.unk_token_id, 
                                eos_token_id=tokenizer.eos_token_id, 
                                max_new_tokens=1024).cpu()
        
        pred = [tokenizer.decode(o[o!=tokenizer.unk_token_id][n:]) for o,n in zip(output,input_lens)]
        short_pred = [extract_last_number_or_ratio(p) for p in pred]
        
        answers_pred.extend(pred)
        short_answers_pred.extend(short_pred)
        
    init_end_event.record()
    time_taken = init_start_event.elapsed_time(init_end_event) / 1000
    
    peak_reserved_memory = torch.cuda.max_memory_reserved()
    peak_allocated_memory = torch.cuda.max_memory_allocated()
    
    # save question, answer, pred of valid dataset as json
    exact_match_score = sum(p==g for p,g in zip(short_answers_pred, short_answers_gt))/len(short_answers_pred)
    print(f"Exact match score: {exact_match_score}")
            
    with open(args['save_path'], 'w+') as f:
        result_dict = {'question':list(valid_dataset['question']), 
                        'answer':list(valid_dataset['answer']),
                        'answer_pred':answers_pred,
                        'short_answer_gt':list(short_answers_gt), 
                        'short_answer_pred':short_answers_pred, 
                        'exact_match_score':exact_match_score, 
                        "time_taken":time_taken, 
                        "peak_reserved_memory":peak_reserved_memory/1e9,
                        "peak_allocated_memory":peak_allocated_memory/1e9}    
        json.dump(result_dict, f)