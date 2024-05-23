#  Running below will result in a directory `Llama-2-7b_qlora-{local_rank}` with the following artifacts:
# - Llama-2-7b_qlora-chrome-trace.json.gz - interactive trace that can be viewed using `chrome::tracing` or `perfetto`
# - Llama-2-7b_qlora-key_averages.txt - sorted table of events, e.g.:
# | Name                                  | Self CPU % | Self CPU   | CPU total % | CPU total  | CPU time avg | Self CUDA   | Self CUDA % | CUDA total  | CUDA time avg | CPU Mem | Self CPU Mem | CUDA Mem | Self CUDA Mem | # of Calls | Source Location                                                                  |
# |---------------------------------------|------------|------------|-------------|------------|--------------|-------------|-------------|-------------|---------------|---------|--------------|----------|---------------|------------|----------------------------------------------------------------------------------|
# | ProfilerStep*                         | 0.00%      | 0.000us    | 0.00%       | 0.000us    | 0.000us      | 4.816s      | 44.60%      | 4.816s      | 963.233ms     | 0 b     | 0 b          | 0 b      | 0 b           | 5          | <built-in method to of Tensor object at 0x7f20bf709310>                         |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | train.py(962): fsdp_main                                                         |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | torch/multiprocessing/spawn.py(75): _wrap                                        |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | multiprocessing/process.py(108): run                                            |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | multiprocessing/process.py(314): _bootstrap                                      |
# | FullyShardedDataParallel.forward      | 0.00%      | 0.000us    | 0.00%       | 0.000us    | 0.000us      | 2.208s      | 20.45%      | 2.208s      | 441.555ms     | 0 b     | 0 b          | 0 b      | 0 b           | 5          | <built-in method embedding of type object at 0x7f21e21797c0>                   |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | torch/nn/functional.py(2154): embedding                                          |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | torch/nn/modules/sparse.py(162): forward                                         |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | torch/nn/modules/module.py(1534): _call_impl                                     |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | nn.Module: Embedding_0                                                           |
# | aten::mm                              | 0.44%      | 31.314ms   | 0.69%       | 48.739ms   | 43.517us     | 332.421ms   | 3.08%       | 337.208ms   | 301.079us     | 0 b     | 0 b          | 3.26 Gb  | 3.26 Gb       | 1120       | <built-in function linear>                                                       |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | bitsandbytes/autograd/_functions.py(492): forward                               |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | <built-in method apply of FunctionMeta object at 0x827a410>                     |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | torch/autograd/function.py(582): apply                                          |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | bitsandbytes/autograd/_functions.py(559): matmul_4bit                           |
# | MatMul4Bit                            | 2.81%      | 198.511ms  | 4.93%       | 347.437ms  | 310.212us    | 284.169ms   | 2.63%       | 630.417ms   | 562.872us     | 0 b     | 0 b          | 3.26 Gb  | -62.31 Gb     | 1120       | <built-in method apply of FunctionMeta object at 0x827a410>                     |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | torch/autograd/function.py(582): apply                                          |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | bitsandbytes/autograd/_functions.py(559): matmul_4bit                           |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | bitsandbytes/nn/modules.py(442): forward                                        |
# |                                       |            |            |             |            |              |             |             |             |               |         |              |          |               |            | torch/nn/modules/module.py(1534): _call_impl                                    |

# - Llama-2-7b_qlora-memory-timeline.html - Stacked time series plot of memory use broken down by `Parameter`, `Gradients`, `Activations`, etc. 
# - Llama-2-7b_qlora-stacks.txt - Stack trace.  See [docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

# Detailed `CLI` options:
# - `profile` - whether to profile
# - `profiling_outputs` - output directory for `torch.profiler` artifacts
# - `export_trace` - enables exporting of interactive trace that can be viewed and analyzed using `chrome::tracing`
# - `export_memory_timeline` - exports an HTML memory timeline which shows memory use by category (`parameters`, `activations`, `gradients`, etc.)
# - `with_stack` - exports stack trace
# - `with_shapes` - adds shapes of operators to the trace
# - `{wait, warmup, active}_steps` - controls how many profiling steps are recorded:
#     - `wait_steps` - number of steps for the profiler to wait before starting to profile
#     - `warmup_steps` - number of steps for profiler to profile without recording
#     - `active_steps` - number of steps to record
#     See [docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) for further details.

# The default schedule for the profiler is set such that only 2 steps of the each epoch are recorded (not counting `wait` and `warmup` steps which are not recorded).  

# Note that `with_stack` and `with_shapes` are overridden by `export_memory_timeline` since the memory profile requires these options to be `True`.

#**IMPORTANT** There are issues with recording stack traces and exporting traces simultaneously (see this [issue](https://github.com/pytorch/pytorch/issues/113564)) depending on `python` version.  The only combination I was able to get both to work at the same time was with `python=3.11.9` and `torch=2.3.0`.
#Tested on `python=3.11.9 and torch=2.3.0``

python train.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 256 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type qlora \
--use_gradient_checkpointing false \
--use_cpu_offload false \
--log_to stdout \
--dataset dummy \
--profile true \
--export_trace true \
--export_memory_timeline true \
--max_steps 10
