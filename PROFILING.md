## Profiling

## Usage

**IMPORTANT**
There are issues with recording stack traces and exporting traces simultaneously (see this [issue](https://github.com/pytorch/pytorch/issues/113564)) depending on `python` version. The only combination I was able to get both to work at the same time was with `python=3.11.9` and `torch=2.3.0`.

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

- `{model_name}-{train_type}-chrome-trace.json.gz` - interactive trace that can be viewed using `chrome::tracing` or `perfetto`
- `{model_name}-{train_type}-key_averages.txt` - sorted table of events, e.g.:

```

```

- `{model_name}-{train_type}-memory-timeline.html` - Stacked time series plot of memory use broken down by `Parameter`, `Gradients`, `Activations`, etc.
- `{model_name}-{train_type}-stacks.txt` - Stack trace. See [docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

Detailed `CLI` options:

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
    2. Set `repeat` to `0` and `profiling_frequency` to the cycle length. E.g., with `repeat=0`, `profiling_frequency=10`, `warmup=2`, `active=1`, then `wait` will be automatically set to `profiling_frequency - (warmup + active) = 7`. The profiler will then continuously execute the following cycle: wait for 7 steps, warmup for 2, record for 1.

    See [docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) for further details.

- `max_steps` - maximum number of batches per epoch. E.g., with `num_epochs=1`, stops training after `max_steps` of batches. Note that this is automatically adjusted to accommodate the profiler schedule; for example, if `max_steps < wait_steps + warmup_steps + active_steps`, it will automatically be set to `wait_steps + warmup_steps + active_steps` such that the profiler can run for at least 1 cycle.

#### Additional Notes

The default schedule for the profiler is set such that to continuously execute a 10-step cycle: wait for 7, warmup for 2, record for 1.

`with_stack` and `with_shapes` are overridden by `export_memory_timeline` since the memory profile requires these options to be `True`.
