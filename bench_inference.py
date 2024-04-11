"""
Benchmark VLLM inference:  https://github.com/AnswerDotAI/vllm/tree/bnb_quant.
"""

from pathlib import Path
import os, json
from safetensors.torch import save_file
import copy
from tqdm import tqdm
import safetensors
import safetensors.torch
from glob import glob
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from fastcore.script import *

import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
import torch
import re
from datasets import load_dataset
from fastcore.parallel import parallel
from vllm import LLM, SamplingParams

def extract_last_number_or_ratio(s):
    # Find all sequences of digits, possibly with leading currency symbols, decimal points, and ratios
    patterns = re.findall(r'[\$€£]?\d+(?:\.\d+)?(?:\:\d+(?:\.\d+)?)?', s)
    
    # Return the last pattern found, or None if there are no matches
    if patterns:
        return patterns[-1]
    else:
        return None
    
def timed(fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

@call_parse()
def main(
    infer_type: Param("", choices=["full_post_quant", "bnb_dora", "merged_bnb_dora", "gptq_marlin"]) = "full_post_quant",
    model_dir: str = None, 
    model_name: str = "meta-llama/Llama-2-7b-hf", 
    tensor_parallel_size: int = 4,
):
    args = dict(locals())
    print(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    
    dataset = load_dataset("microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=42)
    # train with 10k for starters. Then 100k.
    # dataset = dataset.select(range(0,100000)
    # select last 5k as validation
    dataset = dataset.select(range(len(dataset)-5000,len(dataset)))
    short_answers_gt = parallel(extract_last_number_or_ratio, dataset['answer'], progress=True)
    inputs = [f"###Question:\n{question}\n###Answer:\n" for question in dataset[:50]['question']]
    
    if args["infer_type"] == "full_post_quant":
        llm = LLM(model=args["model_dir"], 
            tokenizer=args["model_name"], 
            dtype="bfloat16", 
            tensor_parallel_size=args["tensor_parallel_size"], 
            enforce_eager=True, 
            quantization="bnb", 
            gpu_memory_utilization=0.9)
        
    elif args["infer_type"] == "bnb_dora":
        llm = LLM(model=args["model_dir"], 
            tokenizer=args["model_name"], 
            dtype="bfloat16", 
            tensor_parallel_size=args["tensor_parallel_size"], 
            enforce_eager=True, 
            quantization="bnb", 
            gpu_memory_utilization=0.9)
        
    elif args["infer_type"] == "merged_bnb_dora":
        llm = LLM(model=args["model_dir"], 
            tokenizer=args["model_name"], 
            dtype="bfloat16", 
            tensor_parallel_size=args["tensor_parallel_size"], 
            enforce_eager=False, 
            gpu_memory_utilization=0.9)
        
    elif args["infer_type"] == "gptq_marlin":
        llm = LLM(model=args["model_dir"], 
            tokenizer=args["model_name"], 
            dtype="float16", 
            tensor_parallel_size=args["tensor_parallel_size"], 
            enforce_eager=False, 
            quantization="marlin", 
            gpu_memory_utilization=0.9)
        
    # Throughput.
    sampling_params = SamplingParams(temperature=0.0, stop_token_ids=[tokenizer.eos_token_id], max_tokens=1024)
    outputs, time_taken = timed(llm.generate, inputs, sampling_params, use_tqdm=False)
    throughput_req_per_min = len(outputs) / (time_taken / 60)
    throughput_tokens_per_sec = sum([len(o.outputs[0].token_ids) for o in outputs]) / time_taken
    
    # Accuracy.
    short_answers_pred = [extract_last_number_or_ratio(o.outputs[0].text) for o in outputs]
    exact_match_score = sum(p==g for p,g in zip(short_answers_pred, short_answers_gt))/len(short_answers_pred)
    
    # Latency.
    sampling_params = SamplingParams(temperature=0.0, stop_token_ids=[], max_tokens=256)
    outputs, time_taken = timed(llm.generate,["hello world this is"], sampling_params, use_tqdm=False)
    latency_tok_per_sec = len(outputs[0].outputs[0].token_ids) / time_taken
    
    metrics = {
        "throughput_req_per_min": throughput_req_per_min,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "latency_tok_per_sec": latency_tok_per_sec,
        "exact_match_score": exact_match_score
    }
    
    # Pretty print.
    print(json.dumps(metrics, indent=2))