# fsdp_qlora
Training LLMs with QLoRA + FSDP

This is still a work-in-progress, but to start experimenting:

- Install the llama-recipes requirements (we can test what's actually needed later)
- Install our bnb version
- Run with defaults: `python train.py`!

## Testing from a fresh jarvislabs instance:
- Clone https://github.com/AnswerDotAI/fsdp_qlora
- `pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes`
- `pip uninstall bitsandbytes`
- Clone https://github.com/AnswerDotAI/bitsandbytes + change into cuda_fix_quant_storage_dtype branch
- in bitsandbytes folder, `make CUDA_VERSION=117` then `python setup.py install` (may need export BNB_CUDA_VERSION=117 and to set cuda path)
- pip install fastcore wandb
- huggingface-cli login (to access Llama 2 7B)
- back in fsdp_qlora folder, run `python train.py` to test lora train

Check out different combos of settings. For example,
`python train.py --wrapping_policy size --train_type qlora` to do qlora with every layer wrapped individually. 

The low-memory option (loading on only one shard and using sync_module_states) works with LoRA but not QLoRA yet. 

