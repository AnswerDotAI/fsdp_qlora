import torch
from pathlib import Path
import os, json
from safetensors.torch import save_file
import copy
from tqdm import tqdm
import safetensors
import safetensors.torch
from glob import glob
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig, Quantizer

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig, Quantizer
from hqq.backends.torchao import patch_hqq_to_aoint4
from fastcore.script import *

from accelerate import init_empty_weights


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
    
def exact_match_score(preds, labels):
    return sum(p==g for p,g in zip(preds, labels))/len(preds)

dataset = load_dataset("microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=42)
dataset = dataset.select(range(len(dataset)-5000,len(dataset)))
short_answers_gt = parallel(extract_last_number_or_ratio, dataset['answer'], progress=True)
valid_dataset = dataset.select(range(500))
inputs = [f"###Question:\n{question}\n###Answer:\n" for question in valid_dataset['question']]
labels = short_answers_gt[:500]


TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

def convert_to_chat_input(question):
    messages = [
        {"role": "system", "content": "You are an AI assistant that excels in solving math problems."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

chat_inputs = [convert_to_chat_input(question) for question in valid_dataset['question']]



@call_parse()
def main(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    num_gpu: int = 1,
    dtype: str = "bfloat16",
    quantization: str = None,
):
    args = dict(locals())
    
    llm = LLM(
    	  model=args["model_name"], 
          tokenizer=TOKENIZER, 
          tensor_parallel_size=num_gpu, 
          max_model_len=8192,
          quantization=args["quantization"],
          dtype=args["dtype"],
		  gpu_memory_utilization=0.8,
          )
    
    outputs = llm.generate(chat_inputs[:128], SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|eot_id|>"]))
    short_answers_pred = [extract_last_number_or_ratio(o.outputs[0].text) for o in outputs]
    score = exact_match_score(short_answers_pred, labels)
    
    print(f"Model name: {model_name}")
    print(f"Exact match score: {score}")
    
    