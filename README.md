# Llama 405B training

This branch focuses on training quantized Llama3.1-405B with QDoRA. Currently using HQQ.

## System Setup

System requirements (used in runpod which is a docker env with `/workspace` as the shared volume mount):

```bash
sudo apt-get update
sudo apt-get install -y tmux vim htop jq

mkdir -p /workspace/.cache/huggingface
ln -s /workspace/.cache/huggingface ~/.cache/huggingface # place HF token in /workspace/.cache/huggingface/token
```

In azure attach a data disk to `/workspace` and mount it. ([attaching](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/attach-disk-portal))

In google cloud [attaching](https://cloud.google.com/compute/docs/disks/format-mount-disk-linux)

Mount the disk to `/workspace`.

## Training


**NOTE:** Fast inference kernels require axis=1 quantization and axis=1 quantization in HQQ requires the slower `HQQBackend.PYTORCH_BACKPROP` backend. 

- This causes slower training times compared to using `HQQBackend.ATEN_BACKPROP` with axis=0.
- `torch.compile` not faster.
- **TODO:** Data packing, preemtible training (just need to add resume option with path to checkpoint) with Skypilot + Spot VMs.

### Create new virtual environment

```bash
python -m venv /workspace/py_venvs/qdora
source /workspace/py_venvs/qdora/bin/activate
```

### FSDP QDoRA branch

```bash
git clone https://github.com/AnswerDotAI/fsdp_qlora.git
git switch llama400b

pip install torch transformers datasets accelerate fastcore hqq wheel setuptools
pip install flash-attn --no-build-isolation # faster than sdpa
```

1. Donwload model weights in a CPU machine before training. Use `fsdp_qlora/experiments/llama_large/download_weights.py` to download the weights. Approx 7-8 mins to download for llama-70b.

2. Prepare instruction tuning data mixture in a CPU machine. Use `fsdp_qlora/experiments/llama_large/prepare_data_mix.ipynb` to prepare the data.

### Train

Use 

```bash
sh fsdp_qlora/experiments/llama_large/llama_[70b|405b].sh
```

to train the models.

Some ideas to try:

1. Different data mixture and/or sampling.
2. Different hyperparameters: lora rank, lr sched, lr.

Approx 5 mins (llama70b) and 20-30 mins (llama405b) to load and quantize model weights during training.


### Upload DoRA model weights and model configs.

1. Create a HF model repo, for example answerdotai/Meta-Llama-3-70B-Instruct-4bit-DoRA.
2. Upload with 

```bash
huggingface-cli upload answerdotai/Meta-Llama-3-70B-Instruct-4bit-DoRA llama-3-70b-instruct-hqq-4bit/
```


## Inference

### Create new virtual environment

```bash
python -m venv /workspace/py_venvs/vllm
source /workspace/py_venvs/vllm/bin/activate
```

Before any other installation build the vllm fork branch locally (takes a while) - this also installs torch.

```bash
git clone https://github.com/AnswerDotAI/vllm
cd vllm && git switch torchao && pip install -e .
```

Build torchao locally (which has the fused tinygemm dequant kernel)

```bash
git clone https://github.com/pytorch/ao.git
cd ao && python setup.py install
```

VLLM model prep utils.

```bash
git clone https://github.com/AnswerDotAI/fsdp_qlora.git
cd fsdp_qlora && git switch llama400b
```

Quantization and utils (hqq comes with bitblas)

```bash
pip install fastcore hqq openai
```

### Model Download

Download the model weights and configs.

```bash
huggingface-cli download answerdotai/Meta-Llama-3-1-405B-Instruct-4bit-DoRA --local-dir ./Meta-Llama-3-1-405B-Instruct-4bit-DoRA
huggingface-cli download answerdotai/Meta-Llama-3-1-405B-Instruct-4-2bit-DoRA --local-dir ./Meta-Llama-3-1-405B-Instruct-4-2bit-DoRA
huggingface-cli download answerdotai/Meta-Llama-3-1-405B-Instruct-2bit-DoRA --local-dir ./Meta-Llama-3-1-405B-Instruct-2bit-DoRA
```

### vLLM Model Prep

Prepare vLLM weights. (takes a while). TODO: quantize once and share the weights, (but VLLM expects all the weights in the same directory).

```bash
sh /workspace/git/fsdp_qlora/experiments/llama_large/prepare_vllm_weights.sh
```

Inference benchmarking and evaluation.

### Latency and throughput benchmarks

Script: https://github.com/AnswerDotAI/kerem_research/blob/main/inference_benchmarking/benchmark.py

Example: https://github.com/AnswerDotAI/kerem_research/blob/main/inference_benchmarking/benchmark.sh


### Evaluation benchmarks

**TODO:** Add MLMMLU and human eval prompts with lmsys-1m.

Script: https://github.com/AnswerDotAI/kerem_research/blob/main/evaluation_benchmarking/eval.py

Example: https://github.com/AnswerDotAI/kerem_research/blob/main/evaluation_benchmarking/eval.sh
