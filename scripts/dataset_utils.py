import copy
import torch
from pathlib import Path
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# DATASET + DATALOADERS (modified from llama recipes)
# Formatting prompts in alpaca
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# Dataset class
class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, style="alpaca", add_special_tokens=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.style = style
        self.add_special_tokens = add_special_tokens
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        if self.style == "guanaco":
            prompt = self.dataset[index]["text"].split("### Assistant: ")[0]
            example = self.dataset[index]["text"]
        elif self.style == "qna":
            prompt_template = "###Context:\n{context}\n###Question:\n{question}\n###Answer:\n"
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample['answer']
        elif self.style == "qna_no_ctx":
            prompt_template = "###Question:\n{question}\n###Answer:\n"
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample['answer']           
        elif self.style == "local":
            sample = self.dataset[index]
            prompt = sample['input_text']
            example = prompt + sample['output_text']        
        else: # Alpaca
            ann = self.dataset[index]
            if ann.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            example = prompt + ann["output"]

        prompt = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=self.add_special_tokens), dtype=torch.int64
        )
        example = self.tokenizer.encode(example, add_special_tokens=self.add_special_tokens)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }

# And to get the dataloader
def get_dataloader(tokenizer:PreTrainedTokenizerFast, args:Dict, pad_to_nearest=False, debug=False):
    """Creates a dataset and appropriate dataloader with distributed sampler."""
    # Importing here rather than at the start to avoid multiprocessing issues
    from datasets import Dataset, load_dataset, load_from_disk
    
    dataset_path = Path(args['dataset'])
    is_local = dataset_path.exists() and dataset_path.is_dir()

    # Load the source dataset
    if args["dataset"] == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")['train']
    elif args["dataset"] == "alpaca_sample":
        dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{args['dataset_samples']}]")
    elif args["dataset"] == "dummy":
        dataset = Dataset.from_dict({
            'instruction': ["instruction"]*args["dataset_samples"],
            'input': ["input"]*args["dataset_samples"],
            'output': ["output"*args["context_length"]*2]*args["dataset_samples"]} # A long output to test memory usage (gets truncated)
        )
    elif args["dataset"] == "guanaco":
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    elif args["dataset"] == "sql":
        dataset = load_dataset("knowrohit07/know_sql")['validation']
        dataset = dataset.shuffle(seed=args["seed"])
        dataset = dataset.select(range(1000,len(dataset)))
    elif args["dataset"] == "orca_math":
        dataset = load_dataset("microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=42)
        # train with 10k for starters. Then 100k.
        dataset = dataset.select(range(0,args['dataset_samples']))
    elif args["dataset"] == "orca_math":
        dataset = load_dataset("microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=42)
        # train with 10k for starters. Then 100k.
        dataset = dataset.select(range(0,args['dataset_samples']))        
    elif args["dataset"] == "orca_math_instruct":        
        def convert_to_chat(example):
            messages = [
                {"role": "system", "content": "You are an AI assistant that excels in solving math problems."},
                {"role": "user", "content": example['question']},
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output_text = example['answer']
            return {"input_text":input_text, "output_text":output_text}
        dataset = load_dataset("microsoft/orca-math-word-problems-200k")['train'].shuffle(seed=42)
        # train with 10k for starters. Then 100k.
        dataset = dataset.select(range(0,args['dataset_samples']))     
        dataset = dataset.map(convert_to_chat)
    elif is_local:
        dataset = load_from_disk(str(dataset_path)).shuffle(seed=args["seed"])        
    # truncate dataset so it's evenly divisible by grad_accumulation_steps
    dataset = dataset.select(range(0, len(dataset)-len(dataset)%(args["batch_size"]*args["gradient_accumulation_steps"])))

    # # Create the InstructionDataset
    if args["dataset"] == "guanaco":
        dataset = InstructionDataset(dataset, tokenizer, style="guanaco", add_special_tokens=True)
    elif args["dataset"] == "sql":
        dataset = InstructionDataset(dataset, tokenizer, style="qna", add_special_tokens=True)
    elif args["dataset"] == "orca_math":
        dataset = InstructionDataset(dataset, tokenizer, style="qna_no_ctx", add_special_tokens=True)
    elif args["dataset"] == "orca_math_instruct":
        dataset = InstructionDataset(dataset, tokenizer, style="local", add_special_tokens=True)
    elif is_local:
        dataset = InstructionDataset(dataset, tokenizer, style="local", add_special_tokens=False)
    else: # (w/ alpaca prompt formatting)
        dataset = InstructionDataset(dataset, tokenizer, style="alpaca", add_special_tokens=True)
        

    # Collate function
    def collate_fn(batch, with_attention_mask=False, pad_to_nearest=pad_to_nearest, pad_to_context_length=False):
        # To list of tensors
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        # Pad + truncate
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :args["context_length"]]
        if with_attention_mask:
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:, :args["context_length"]]
        else:
            attention_masks = None
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)[:, :args["context_length"]]
        
        if pad_to_nearest:
            # might be useful for torch.compile dynamic kernels.
            nearest_pad_dims = torch.tensor([128*i for i in range(1,args["context_length"]//128+1)])
            nearest_pad_dim_idxs = torch.where(input_ids.shape[1] < nearest_pad_dims)[0]
            if len(nearest_pad_dim_idxs) > 0:
                nearest_pad_dim = nearest_pad_dims[nearest_pad_dim_idxs[0]]
                num_pad = nearest_pad_dim - input_ids.shape[1]
                input_ids = torch.nn.functional.pad(input_ids, pad=(0,num_pad), value=tokenizer.pad_token_id)
                labels    = torch.nn.functional.pad(labels, pad=(0,num_pad), value=-100)
                if with_attention_mask:
                    attention_masks = torch.nn.functional.pad(attention_masks, pad=(0,num_pad), value=0)

        if pad_to_context_length:
            input_ids = torch.nn.functional.pad(input_ids, pad=(0,args["context_length"]-input_ids.shape[1]), value=tokenizer.pad_token_id)
            labels    = torch.nn.functional.pad(labels, pad=(0,args["context_length"]-labels.shape[1]), value=-100)
            if with_attention_mask:
                attention_masks = torch.nn.functional.pad(attention_masks, pad=(0,args["context_length"]-attention_masks.shape[1]), value=0)
                
        # Return dict
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

    # For distributed training, use DistributedSampler
    if debug:
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_fn, shuffle=True)
    # Use the custom collate function in DataLoader
    else:
        sampler = DistributedSampler(dataset, seed=args["seed"])
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_fn, sampler=sampler)

    return dataloader
