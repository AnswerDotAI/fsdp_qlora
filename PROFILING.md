## Profiling

Documentation for how to profile your training runs.

**Tips**

- Only record what is necessary as profiling can significantly slow down training process.
- Set a `torch.profile.schedule` when running the profiler (description below), as trace artifacts are exported at the end of each profiling cycle and can be very large (on the order of hundreds of MBs each).

**IMPORTANT**
There are issues with recording stack traces and exporting traces simultaneously (see this [issue](https://github.com/pytorch/pytorch/issues/113564)) depending on `python` version.

Tested with `python=3.11.9` and `torch=2.3.0`.

## Quickstart

Running the following:

```
python train.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--train_type qlora \
--profile true \
--export_trace true \
--export_memory_timeline true \
--max_steps 10
```

will result in a directory `{model_name}_{train_type}-{local_rank}` with the following artifacts:

- `{model_name}-{train_type}-chrome-trace.json.gz` - interactive trace that can be viewed using `chrome::tracing`, `perfetto`, or `tensorboard`
- `{model_name}-{train_type}-key_averages.txt` - sorted table of events, e.g.:

| Name                                                                              | Self CPU % | Self CPU | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls | Source Location                                                          |
| --------------------------------------------------------------------------------- | ---------- | -------- | ----------- | --------- | ------------ | --------- | ----------- | ---------- | ------------- | ---------- | ------------------------------------------------------------------------ |
| ncclDevKernel_AllGather_RING_LL(ncclDevComm*, unsigned int*, unsigned int\*, int) | 0.00%      | 0.000us  | 0.00%       | 0.000us   | 0.000us      | 88.038ms  | 12.14%      | 88.038ms   | 830.547us     | 106        | <built-in method \_allgather_base of PyCapsule object at 0x7f2760c2ea30> |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | torch/distributed/distributed_c10d.py(2864): all_gather_into_tensor      |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | torch/distributed/c10d_logger.py(72): wrapper                            |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | torch/distributed/fsdp/\_flat_param.py(1366): \_all_gather_flat_param    |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | torch/distributed/fsdp/\_flat_param.py(1285): unshard                    |
| FullyShardedDataParallel.forward                                                  | 0.00%      | 0.000us  | 0.00%       | 0.000us   | 0.000us      | 59.050ms  | 8.14%       | 59.050ms   | 59.050ms      | 1          | <built-in method embedding of type object at 0x7f281c5787c0>             |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | torch/nn/functional.py(2154): embedding                                  |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | torch/nn/modules/sparse.py(162): forward                                 |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | torch/nn/modules/module.py(1534): \_call_impl                            |
|                                                                                   |            |          |             |           |              |           |             |            |               |            | nn.Module: Embedding_0                                                   |

- `{model_name}-{train_type}-memory-timeline.html` - Stacked time series plot of memory use broken down by `Parameter`, `Gradients`, `Activations`, etc.
- `{model_name}-{train_type}-stacks.txt` - Stack trace. See [docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

## Detailed Usage

`CLI` options in full:

- `profile` - whether to profile
- `profiling_outputs` - output directory for `torch.profiler` artifacts
- `export_trace` - enables exporting of interactive trace that can be viewed and analyzed using `chrome::tracing`
- `export_memory_timeline` - exports an HTML memory timeline which shows memory use by category (`parameters`, `activations`, `gradients`, etc.)
- `with_stack` - exports stack trace
- `with_shapes` - adds shapes of operators to the trace
- `{wait, warmup, active}_steps, repeat, profiling_frequency` - controls the profiling schedule:

  - `wait_steps` - number of steps for the profiler to wait before starting to profile. Overridden if `repeat=0` (see note below).
  - `warmup_steps` - number of steps for profiler to profile without recording
  - `active_steps` - number of steps to record
  - `repeat` - number of times to repeat the above cycle of `wait, warmup, active` if `repeat > 0` else cycles forever
  - `profiling_frequency` - profiling frequency in steps. Only used if `repeat = 0`, in which case `wait_steps = profiling_frequency - (warmup_steps + active_steps)` such that the effective cycle length = `profiling_frequency`. E.g., if `profiling_frequency=10`, `warmup_steps=2`, `active_steps=1`, then the profiler will wait 8 steps, warmup for 2, record for 1, then repeat.

    **Note**: Simplest to think of 2 ways of scheduling the profiler:

    1. Set `repeat` to the number of total number of desired profiling cycles. For example if `wait=1`, `warmup=1`, `active=1`, and `repeat=1`, then the profiler will wait for 1 step, warmup for 1, and record for 1 then stop.
    2. Set `repeat` to `0` and `profiling_frequency` to the cycle length. E.g., with `repeat=0`, `profiling_frequency=10`, `warmup=2`, `active=1`, then `wait` will be automatically set to `profiling_frequency - (warmup + active) = 7`. The profiler will then continuously execute the following cycle: wait for 7 steps, warmup for 2, record for 1 for the entire training run.

    See [docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) for further details.

- `max_steps` - maximum number of batches per epoch. E.g., with `num_epochs=1`, stops training after `max_steps` of batches. Note that this is automatically adjusted to accommodate the profiler schedule; for example, if `max_steps < wait_steps + warmup_steps + active_steps`, it will automatically be set to `wait_steps + warmup_steps + active_steps` such that the profiler can run for at least 1 cycle.

## Additional Notes

The default schedule for the profiler is set to continuously execute a 10-step cycle: wait for 7, warmup for 2, record for 1.

`with_stack` and `with_shapes` are overridden by `export_memory_timeline` since the memory profile requires these options to be `True`.

## Examples

- Record every 5th step, exporting a `chrome` / `tensorboard` trace for each cycle:

  ```
  python train.py \
  --model_name "hf-internal-testing/tiny-random-LlamaForCausalLM" \
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
  --export_memory_timeline false \
  --with_stack true \
  --num_epochs 1 \
  --max_steps 20 \
  --repeat 0 \
  --warmup_steps 4 \
  --active_steps 1 \
  --profiling_frequency 5 \
  --profiling_output llama-test
  ```

  The output will be a 4 trace output folders, at iteration 5, 10, ..., each containing a trace with a single training step at that iteration.

  Also in the folder will be exported stacks (which can be visualized using flamegraphs or other stack viewers) and `key_averages`, which is a summary table of operations ordered by `cuda` time.

  Note that we set `max_steps=20` so that the training loop will exit after 20 batches. If `max_steps=-1` (the default setting), the profiler will repeat the cycle during the entire training run.

- Record 5 steps (after 1 warmup step) then stop profiling:
  ```
  python train.py \
  --model_name "hf-internal-testing/tiny-random-LlamaForCausalLM" \
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
  --with_stack true \
  --num_epochs 1 \
  --max_steps 20 \
  --warmup_steps 1 \
  --active_steps 5 \
  --repeat 1 \
  --profiling_output llama-test2
  ```
  The output will be a single trace at `iteration_6` which contains 5 training steps.
  In addition to the `stacks` and `key_averages` artifacts, there will be a `memory_timeline` `html`, which shows a breakdown of memory usage by `parameter`, `gradients`, `activations`, etc.
