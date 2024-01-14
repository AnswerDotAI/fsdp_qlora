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
`python train.py --train_type qlora` to do qlora instead of the default lora

NBs:

- The low-memory option (loading on only one shard and using sync_module_states) works with LoRA but not QLoRA yet. 
- size-based wrapping policy gave an error but I think it should work, will update if I get it going.
- the undocumented train_type hf_qlora loads with the transformers load_in_4bit option, which by default gives a ValueError: Cannot flatten integer dtype tensors (since quant_storage default is uint8). If you manually edit the def of Linear4Bit to set the default to bf16 this should work (with size wrapping policy, the other one fails because something is fp16) and is a useful comparison for exploring what options we have if we want to avoid all the custom model loading stuff. Hmm after a fresh install this isn't working with the size-based wrapping either, "AssertionError: Expects storage to be allocated", I need to debug this.
